"""
Coding performance benchmark using direct LLM generation.
Respects all generation flags (temp, top_p, top_k, etc.) via LlamaClient.

Active benchmarks (val_score = 0.35*LCB + 0.25*HE_strict + 0.25*MBPP + 0.15*BigCode):
  - HumanEval+ (strict)  : 164 algorithmic problems, evalplus `test` field
  - MBPP+                : 974 entry-level problems, evalplus `test` field
  - LiveCodeBench v6     : contamination-free competitive programming, sampled
  - BigCodeBench Hard    : 148 library-call tasks, sampled

Loaders fetch data on first use and cache to data/ subdir. No new pip deps:
  - HumanEval+/MBPP+  : evalplus (already required)
  - LiveCodeBench v6  : base64+zlib+pickle JSONL fetched via huggingface_hub
  - BigCodeBench Hard : HF datasets cache (datasets + pyarrow already installed)
"""

from __future__ import annotations

import base64
import json
import os
import pickle
import re
import signal
import subprocess
import sys
import tempfile
import time
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional

from autoresearch.core import config
from autoresearch.core.llama_client import LlamaClient
from autoresearch.benchmarks.benchmark_harness import BenchmarkResult

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR.parent / "data" / "benchmark_cache"


# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ctx-size", type=int, default=config.CTX_SIZE)
    parser.add_argument("--model", type=str, default=config.MODEL)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--coding-task-limit", type=int, default=config.CODING_TASK_LIMIT)
    parser.add_argument("--lcb-task-limit", type=int, default=getattr(config, "LCB_TASK_LIMIT", 10))
    parser.add_argument("--bigcode-task-limit", type=int, default=getattr(config, "BIGCODE_TASK_LIMIT", 10))
    parser.add_argument("--port", type=int, default=18080)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_THINK_CLOSED_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_THINK_OPEN_RE = re.compile(r"<think>.*$", re.DOTALL)
_FENCE_RE = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL)
# Lines that look like Python code (start of statement / decorator / continuation)
_CODE_LINE_PREFIXES = (
    "def ", "class ", "import ", "from ", "if __name__",
    "@", "async def ", "with ", "try:", "for ", "while ",
    "print(", "return ", "raise ", "yield ",
    "n =", "x =", "a, b =", "a,b=", "result ", "out ",
)


def _strip_code(text: str) -> str:
    """
    Extract Python code from a model response, handling thinking-model output.

    Handles (in order):
      1. <think>...</think> blocks (closed) - stripped.
      2. Unclosed <think> (truncated thinking) - stripped from <think> to end of text.
      3. Markdown fences ```python ... ``` or ``` ... ``` - extracted.
      4. Plain code with no fence - returned as-is, or extracted from the
         first code-looking line if a prose prefix is present.
      5. Empty / think-only responses - returns "" (caller treats as "no code").
    """
    if not text:
        return ""
    text = text.strip()

    # 1) Strip closed <think>...</think> blocks (handles multiple, non-greedy).
    text = _THINK_CLOSED_RE.sub("", text)
    # 2) Strip unclosed <think> block (truncated thinking, no code emitted).
    text = _THINK_OPEN_RE.sub("", text)
    text = text.strip()

    if not text:
        return ""

    # 3) Fenced code block.
    m = _FENCE_RE.search(text)
    if m:
        return m.group(1).strip()

    # 4) No fence. If first non-empty line looks like Python, return whole text.
    first = next((ln for ln in text.splitlines() if ln.strip()), "")
    if any(first.lstrip().startswith(p) for p in _CODE_LINE_PREFIXES):
        return text

    # 5) Prose prefix: scan for the first line that looks like Python.
    lines = text.splitlines()
    code_start = None
    for i, line in enumerate(lines):
        if any(line.lstrip().startswith(p) for p in _CODE_LINE_PREFIXES):
            code_start = i
            break
    if code_start is not None:
        return "\n".join(lines[code_start:]).strip()

    # Last resort: return as-is. Let the test runner surface the syntax error.
    return text


def _run_subprocess(script: str, stdin_input: str | None = None, timeout: float = 10.0) -> tuple[int, str, str]:
    """Run a Python script in a sandboxed subprocess. Returns (returncode, stdout, stderr)."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            input=stdin_input,
            capture_output=True, text=True, timeout=timeout,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "TIMEOUT"
    except Exception as e:
        return -1, "", str(e)


def _run_tests(code: str, test_code: str, timeout: float = 10.0) -> bool:
    """Run generated code + test code in a sandboxed subprocess. Returns True if all tests pass."""
    rc, _, _ = _run_subprocess(f"{code}\n\n{test_code}\n", timeout=timeout)
    return rc == 0


# ---------------------------------------------------------------------------
# Evalplus loaders (HumanEval+ / MBPP+)
# ---------------------------------------------------------------------------

def _load_problems(dataset: str) -> dict:
    """Load problem definitions from evalplus data."""
    try:
        if dataset == "humaneval":
            from evalplus.data import get_human_eval_plus
            return get_human_eval_plus()
        elif dataset == "mbpp":
            from evalplus.data import get_mbpp_plus
            return get_mbpp_plus()
    except ImportError:
        pass
    try:
        if dataset == "humaneval":
            from evalplus.data import get_human_eval
            return get_human_eval()
        elif dataset == "mbpp":
            from evalplus.data import get_mbpp
            return get_mbpp()
    except ImportError:
        print(f"  [CODING] Cannot load {dataset} problems — evalplus not installed.")
    return {}


def _build_prompt(entry: dict, dataset: str) -> str:
    """Build a code generation prompt from a problem entry."""
    if dataset == "humaneval":
        return (
            "Complete the following Python function. Return ONLY the function body, no explanations.\n\n"
            f"{entry.get('prompt', '')}"
        )
    if dataset == "mbpp":
        text = entry.get("text", entry.get("prompt", ""))
        prompt = entry.get("prompt", "")
        return (
            "Write a Python function that satisfies the following description.\n"
            f"Description: {text}\n\n{prompt}"
        )
    return ""


def _get_test_code(entry: dict, dataset: str) -> str:
    """
    Extract test assertions from a problem entry.

    Strict mode (default): prefer the evalplus `test` field which contains the
    rigorous plus-tests. If absent, fall back to building asserts from
    base+plus input/output pairs.
    """
    strict = bool(getattr(config, "EVALPLUS_STRICT", True))
    if dataset == "humaneval":
        test = entry.get("test", "")
        if not test:
            entry_point = entry.get("entry_point", "")
            base_tests = entry.get("base_input_output_tests", []) or []
            plus_tests = entry.get("plus_input_output_tests", []) or []
            pairs = (plus_tests if strict else []) + base_tests
            if pairs and entry_point:
                lines = [f"assert {entry_point}(*{repr(inp)}) == {repr(out)}" for inp, out in pairs]
                test = "\n".join(lines)
        return test
    if dataset == "mbpp":
        test = entry.get("test", "")
        if not test:
            assertion = entry.get("assertion", "")
            if assertion:
                test = assertion
            else:
                test_list = entry.get("test_list", [])
                test = "\n".join(test_list)
        return test
    return ""


# ---------------------------------------------------------------------------
# LiveCodeBench v6 loader
# ---------------------------------------------------------------------------

LCB_CACHE_DIR = DATA_DIR / "livecodebench_v6"
LCB_FILENAME = "test6.jsonl"  # 175 problems, ~134 MB, all v6 (newest)


def _download_lcb_file(force: bool = False) -> Path:
    """Download LiveCodeBench v6 test6.jsonl to local cache. Returns path."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LCB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    target = LCB_CACHE_DIR / LCB_FILENAME
    if target.exists() and target.stat().st_size > 0 and not force:
        return target
    try:
        from huggingface_hub import hf_hub_download
        # download to a temp name then move (atomic-ish)
        tmp = hf_hub_download(
            "livecodebench/code_generation_lite",
            filename=LCB_FILENAME,
            repo_type="dataset",
            cache_dir=str(LCB_CACHE_DIR / "_hf_cache"),
        )
        # symlink to canonical name for stability
        if target.exists() or target.is_symlink():
            target.unlink()
        target.symlink_to(tmp)
        return target
    except Exception as e:
        raise RuntimeError(f"LiveCodeBench download failed: {e}")


def _decode_lcb_private_cases(raw: str) -> list:
    """Decode LiveCodeBench's base64+zlib+pickle-wrapped JSON test cases."""
    try:
        b = base64.b64decode(raw)
        decoded = zlib.decompress(b)
        s = pickle.loads(decoded)
        if isinstance(s, str):
            return json.loads(s)
        return s
    except Exception:
        return []


def _load_livecodebench(task_limit: int = 10, platform_filter: str | None = None) -> list[dict]:
    """
    Load LiveCodeBench v6 problems from local cache (downloads on first use).

    platform_filter: if set (e.g. "atcoder"), keep only that platform.
                     Default None keeps atcoder+codeforces (stdin/stdout style),
                     drops leetcode (class-based; harder to eval in subprocess).
    """
    try:
        path = _download_lcb_file()
    except Exception as e:
        print(f"  [LCB] {e}")
        return []

    problems: list[dict] = []
    with open(path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if platform_filter and rec.get("platform") != platform_filter:
                continue
            # Decode private tests so we don't re-decode every trial
            private_tests = _decode_lcb_private_cases(rec.get("private_test_cases", ""))
            if not private_tests:
                continue
            rec["_private_tests_decoded"] = private_tests
            problems.append(rec)
            if len(problems) >= task_limit:
                break
    return problems


def _build_lcb_prompt(entry: dict) -> str:
    """
    Build a LiveCodeBench prompt. Instructs the model to write a complete
    stdin/stdout program (matches atcoder/codeforces test pattern).
    """
    title = entry.get("question_title", "")
    content = entry.get("question_content", "").strip()
    starter = entry.get("starter_code", "").strip()
    platform = entry.get("platform", "")

    parts = [f"# {title}" if title else "", content]
    if starter:
        parts.append(f"\nStarter code:\n```python\n{starter}\n```")
    parts.append(
        "\nSolve this problem in Python. Read from standard input, write the answer to standard output. "
        "If a function signature is provided, wrap the call in `if __name__ == \"__main__\": solve()` "
        "or invoke the class appropriately. Return ONLY the code, no explanations."
    )
    return "\n".join(p for p in parts if p).strip()


def _run_lcb_tests(code: str, entry: dict, timeout: float = 10.0) -> bool:
    """
    Run a LiveCodeBench problem: feed each test case's `input` to stdin,
    compare stdout to expected `output` (whitespace-stripped tolerance).
    """
    tests = entry.get("_private_tests_decoded", []) or []
    if not tests:
        return False
    for tc in tests:
        inp = tc.get("input", "")
        expected = tc.get("output", "").rstrip("\n")
        if "input" not in tc:
            continue
        rc, stdout, _ = _run_subprocess(code, stdin_input=inp, timeout=timeout)
        if rc != 0:
            return False
        if stdout.rstrip("\n") != expected:
            return False
    return True


# ---------------------------------------------------------------------------
# BigCodeBench Hard loader
# ---------------------------------------------------------------------------

BCB_CACHE_DIR = DATA_DIR / "bigcodebench_hard"
BCB_DATASET = "bigcode/bigcodebench-hard"
BCB_SPLIT = "v0.1.0_hf"  # 148 problems, ~2.76 MB; newest stable revision


def _load_bigcodebench_hard(task_limit: int = 10) -> list[dict]:
    """Load BigCodeBench Hard problems via HF datasets (auto-caches)."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("  [BigCode] datasets library not available; skipping BigCodeBench Hard.")
        return []
    try:
        BCB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        ds = load_dataset(BCB_DATASET, split=BCB_SPLIT, cache_dir=str(BCB_CACHE_DIR / "_hf_cache"))
    except Exception as e:
        print(f"  [BigCode] load_dataset failed: {e}")
        return []
    problems: list[dict] = []
    for row in ds:
        problems.append(dict(row))
        if len(problems) >= task_limit:
            break
    return problems


def _build_bigcode_prompt(entry: dict) -> str:
    """Build a BigCodeBench prompt using the instruct split (chat-model friendly)."""
    instr = (entry.get("instruct_prompt") or "").strip()
    return (
        f"{instr}\n\n"
        "Return ONLY a single Python code block containing the complete function definition. "
        "No explanations, no usage examples."
    )


def _run_bigcode_tests(code: str, entry: dict, timeout: float = 15.0) -> bool:
    """
    Run BigCodeBench unittest suite. The dataset ships a full unittest.TestCase
    class as the `test` field. We prepend the model's code (defines `task_func`)
    and run the suite. A unittest returning exit 0 means all assertions passed.
    """
    test_code = entry.get("test", "")
    entry_point = entry.get("entry_point", "task_func")
    if not test_code:
        return False
    # BigCodeBench test uses unittest.TestCase — invoke it via the unittest runner
    # rather than pytest (not always installed). We use unittest.main with
    # exit=False but redirect to a synthetic runner that propagates the result.
    script = (
        f"import os, sys, unittest\n"
        f"{code}\n\n"
        f"{test_code}\n\n"
        f"loader = unittest.TestLoader()\n"
        f"suite = loader.loadTestsFromTestCase(TestCases)\n"
        f"runner = unittest.TextTestRunner(verbosity=0, stream=open(os.devnull, 'w'))\n"
        f"result = runner.run(suite)\n"
        f"sys.exit(0 if result.wasSuccessful() else 1)\n"
    )
    rc, _, _ = _run_subprocess(script, timeout=timeout)
    return rc == 0


# ---------------------------------------------------------------------------
# Eval runner (single dataset, prompt -> code -> tests)
# ---------------------------------------------------------------------------

def run_coding_eval(
    client: LlamaClient,
    dataset: str,
    task_limit: int = 0,
    timeout_at: float | None = None,
    problems: list[dict] | None = None,
    max_tokens: int | None = None,
    **gen_kwargs,
) -> tuple[float, int, float]:
    """
    Run coding evaluation on a dataset. Returns (pass_at_1, total_tokens, total_seconds).

    If `problems` is provided, uses that preloaded list instead of re-loading
    (used for non-evalplus datasets like LCB and BigCodeBench).
    """
    if problems is None:
        problems_dict = _load_problems(dataset)
        if not problems_dict:
            return 0.0, 0, 0.0
        task_ids = list(problems_dict.keys())
        entries = [(tid, problems_dict[tid]) for tid in task_ids]
    else:
        entries = [(i, p) for i, p in enumerate(problems)]

    if task_limit > 0:
        entries = entries[:task_limit]

    # Per-dataset max_tokens override (thinking models need more room for LCB/BigCode).
    # Copy gen_kwargs so we do not mutate the caller's dict, and pop max_tokens
    # to avoid "got multiple values for keyword argument 'max_tokens'" when the
    # caller passed it via **gen_kwargs.
    gen_kwargs = dict(gen_kwargs)
    gen_kwargs.pop("max_tokens", None)
    if max_tokens is not None:
        gen_kwargs["max_tokens"] = max_tokens

    total = len(entries)
    passed = 0
    total_tokens = 0
    t_start = time.time()
    dataset_label = dataset

    print(f"  [CODING] {dataset_label}: {total} tasks", flush=True)

    for i, (tid, entry) in enumerate(entries):
        if timeout_at and time.time() > timeout_at:
            print(f"  [CODING] {dataset_label}: timeout at {i}/{total}", flush=True)
            break

        # Build prompt + tests per dataset kind
        if dataset in ("humaneval", "mbpp"):
            prompt = _build_prompt(entry, dataset)
            test_code = _get_test_code(entry, dataset)
        elif dataset == "lcb":
            prompt = _build_lcb_prompt(entry)
            test_code = None  # LCB uses stdin/stdout, not appended
        elif dataset == "bigcode":
            prompt = _build_bigcode_prompt(entry)
            test_code = None  # BigCode tests are pre-defined in entry["test"]
        else:
            continue

        if not prompt:
            continue

        try:
            res = client.complete(prompt, **gen_kwargs)
            usage = res.get("usage", {})
            total_tokens += int(usage.get("total_tokens", 0) or 0)
            # Combine content + reasoning_content. For thinking models, llama-server
            # emits think tokens into a separate reasoning_content field; if the
            # model runs out of tokens mid-think, content may be empty while
            # reasoning_content holds the think block. _strip_code handles both.
            raw_response = res.get("content", "") or ""
            reasoning = res.get("reasoning_content", "") or ""
            combined = (raw_response + "\n" + reasoning).strip() if reasoning else raw_response
        except Exception as e:
            print(f"    {tid} FAIL: {e}", flush=True)
            continue

        code = _strip_code(combined)
        if not code:
            print(f"    {tid} FAIL (no code extracted) ({i+1}/{total})", flush=True)
            continue

        # Run the dataset-appropriate test
        ok = False
        if dataset in ("humaneval", "mbpp"):
            if not test_code:
                continue
            ok = _run_tests(code, test_code)
        elif dataset == "lcb":
            ok = _run_lcb_tests(code, entry)
        elif dataset == "bigcode":
            ok = _run_bigcode_tests(code, entry)

        if ok:
            passed += 1
            print(f"    {tid} PASS ({i+1}/{total})", flush=True)
        else:
            print(f"    {tid} FAIL ({i+1}/{total})", flush=True)

    elapsed = time.time() - t_start
    evaluated = total  # we ran through the full list unless timed out
    pass_at_1 = passed / total if total > 0 else 0.0
    print(
        f"  [CODING] {dataset_label}: {passed}/{evaluated} passed "
        f"(pass@1={pass_at_1:.4f}) TPS={total_tokens/elapsed if elapsed > 0 else 0:.1f}",
        flush=True,
    )
    return pass_at_1, total_tokens, elapsed


# ---------------------------------------------------------------------------
# Unified benchmark
# ---------------------------------------------------------------------------

# Val Score weights (sum to 1.0)
WEIGHT_LCB = 0.35
WEIGHT_HE = 0.25
WEIGHT_MBPP = 0.25
WEIGHT_BIGCODE = 0.15

# Per-dataset max_tokens override. Default 1024 is too tight for thinking models
# on competitive-programming / library-call tasks where the prompt already eats
# context and the think block eats output budget before any code is emitted.
LCB_MAX_TOKENS = 2048
BIGCODE_MAX_TOKENS = 2048


def run_benchmark(client: LlamaClient, **kwargs) -> BenchmarkResult:
    """
    Unified entry point. Runs LCB, HE+, MBPP+, BigCodeBench Hard.

    val_pass1 = LCB, val_pass2 = HE, val_pass3 = MBPP, val_pass4 = BigCode.
    val_score = 0.35*LCB + 0.25*HE + 0.25*MBPP + 0.15*BigCode
    """
    task_limit = kwargs.get("task_limit", 30)
    lcb_limit = kwargs.get("lcb_task_limit", getattr(config, "LCB_TASK_LIMIT", 10))
    bigcode_limit = kwargs.get("bigcode_task_limit", getattr(config, "BIGCODE_TASK_LIMIT", 10))
    timeout_at = kwargs.get("timeout_at", None)

    # Forward only generation kwargs (drop bookkeeping params).
    # max_tokens is handled per-dataset below, so strip it from gen_kwargs here.
    gen_keys = {"task_limit", "lcb_task_limit", "bigcode_task_limit", "timeout_at", "model_name", "is_test"}
    gen_kwargs = {k: v for k, v in kwargs.items() if k not in gen_keys and v is not None and k != "max_tokens"}

    # Pre-load non-evalplus datasets once (they have no `test_limit` arg path)
    lcb_problems = _load_livecodebench(task_limit=lcb_limit) if lcb_limit > 0 else []
    bigcode_problems = _load_bigcodebench_hard(task_limit=bigcode_limit) if bigcode_limit > 0 else []

    # 1) HumanEval+ (strict)
    he_pass, he_tokens, he_time = run_coding_eval(
        client, "humaneval", task_limit=task_limit, timeout_at=timeout_at, **gen_kwargs,
    )

    # 2) MBPP+
    mbpp_pass, mbpp_tokens, mbpp_time = run_coding_eval(
        client, "mbpp", task_limit=task_limit, timeout_at=timeout_at, **gen_kwargs,
    )

    # 3) LiveCodeBench v6 (sampled). Thinking models need more headroom for think+code.
    lcb_pass, lcb_tokens, lcb_time = run_coding_eval(
        client, "lcb", problems=lcb_problems, timeout_at=timeout_at,
        max_tokens=LCB_MAX_TOKENS, **gen_kwargs,
    )

    # 4) BigCodeBench Hard (sampled). Same headroom for thinking models.
    bigcode_pass, bigcode_tokens, bigcode_time = run_coding_eval(
        client, "bigcode", problems=bigcode_problems, timeout_at=timeout_at,
        max_tokens=BIGCODE_MAX_TOKENS, **gen_kwargs,
    )

    val_score = round(
        WEIGHT_LCB * lcb_pass
        + WEIGHT_HE * he_pass
        + WEIGHT_MBPP * mbpp_pass
        + WEIGHT_BIGCODE * bigcode_pass,
        6,
    )

    total_tokens = he_tokens + mbpp_tokens + lcb_tokens + bigcode_tokens
    total_seconds = he_time + mbpp_time + lcb_time + bigcode_time
    avg_tps = total_tokens / total_seconds if total_seconds > 0 else 0.0

    return BenchmarkResult(
        val_score=val_score,
        val_pass1=lcb_pass,
        val_pass2=he_pass,
        val_pass3=mbpp_pass,
        val_pass4=bigcode_pass,
        avg_tps=round(avg_tps, 2),
        total_seconds=round(total_seconds, 2),
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=config.MODEL)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--coding-task-limit", type=int, default=config.CODING_TASK_LIMIT)
    parser.add_argument("--lcb-task-limit", type=int, default=getattr(config, "LCB_TASK_LIMIT", 10))
    parser.add_argument("--bigcode-task-limit", type=int, default=getattr(config, "BIGCODE_TASK_LIMIT", 10))
    parser.add_argument("--port", type=int, default=18080)
    args = parser.parse_args()

    from autoresearch.core.llama_runner import LlamaServerRunner, ServerIntent

    models_dir = ROOT_DIR.parent.parent / "models"
    model_path = models_dir / args.model

    intent = ServerIntent(
        model_path=model_path, ctx_size=config.CTX_SIZE,
        kv_cache=config.KV_CACHE, flash_attn=config.FLASH_ATTN,
        port=args.port, batch_size=config.BATCH_SIZE, ubatch_size=config.UBATCH_SIZE,
        threads=config.THREADS, ngl=99, parallel=1,
        kv_cache_k=config.KV_CACHE_K, kv_cache_v=config.KV_CACHE_V,
        threads_batch=config.THREADS_BATCH, spec_draft_n_max=config.SPEC_DRAFT_N_MAX,
    )

    with LlamaServerRunner(intent) as runner:
        client = LlamaClient(runner.port)
        result = run_benchmark(
            client,
            task_limit=args.coding_task_limit,
            lcb_task_limit=args.lcb_task_limit,
            bigcode_task_limit=args.bigcode_task_limit,
            max_tokens=args.max_tokens,
        )
        print(f"\nCoding Score: {result.val_score:.4f}")
        print(f"LCB pass@1:     {result.val_pass1:.4f}")
        print(f"HE pass@1:      {result.val_pass2:.4f}")
        print(f"MBPP pass@1:    {result.val_pass3:.4f}")
        print(f"BigCode pass@1: {result.val_pass4:.4f}")


if __name__ == "__main__":
    main()
