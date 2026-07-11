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
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional
from abc import ABC, abstractmethod

from autoresearch.core.llama_client import LlamaClient, GenerationParams
from autoresearch.benchmarks.benchmark_harness import BenchmarkResult
from autoresearch.benchmarks import bench_config

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR.parent / "data" / "benchmark_cache"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_THINK_CLOSED_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_THINK_OPEN_RE = re.compile(r"<think>.*$", re.DOTALL)
_FENCE_RE = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL)


def _strip_empty_lines(s: str) -> str:
    if not s:
        return ""
    lines = s.splitlines()
    start = 0
    while start < len(lines) and not lines[start].strip():
        start += 1
    end = len(lines)
    while end > start and not lines[end - 1].strip():
        end -= 1
    return "\n".join(lines[start:end])


def _strip_code(text: str) -> str:
    """Extract Python code from model response. Strips think blocks, extracts fenced code."""
    if not text:
        return ""
    # Strip <think>...</think> blocks (closed or truncated).
    text = _THINK_CLOSED_RE.sub("", text)
    text = _THINK_OPEN_RE.sub("", text)
    text = _strip_empty_lines(text)
    if not text:
        return ""
    # Fenced code block.
    m = _FENCE_RE.search(text)
    if m:
        return _strip_empty_lines(m.group(1))
    # No fence: return dedented text as-is.
    return text


def _run_subprocess(script: str, stdin_input: str | None = None) -> tuple[int, str, str]:
    """Run a Python script in a sandboxed subprocess. Returns (returncode, stdout, stderr)."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            input=stdin_input,
            capture_output=True, text=True,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        return result.returncode, result.stdout, result.stderr
    except Exception as e:
        return -1, "", str(e)


def _run_tests(code: str, test_code: str) -> bool:
    """Run generated code + test code in a sandboxed subprocess. Returns True if all tests pass."""
    rc, _, _ = _run_subprocess(f"{code}\n\n{test_code}\n")
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
    strict = bool(getattr(bench_config, "EVALPLUS_STRICT", True))
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


def _run_lcb_tests(code: str, entry: dict) -> bool:
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
        rc, stdout, _ = _run_subprocess(code, stdin_input=inp)
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


def _run_bigcode_tests(code: str, entry: dict) -> bool:
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
    rc, _, _ = _run_subprocess(script)
    return rc == 0


# ---------------------------------------------------------------------------
# EvalTask protocol — one class per benchmark dataset
# ---------------------------------------------------------------------------

# Per-dataset max_tokens override (used by EvalTask.max_tokens). Default 1024
# is too tight for thinking models on competitive-programming / library-call
# tasks where the prompt eats context and the think block eats output budget
# before any code is emitted.
_MAX_TOKENS = 2048


class EvalTask(ABC):
    """Protocol for a coding benchmark dataset.

    Each task knows how to load problems, build prompts, and run tests.
    Each task also knows its own *weight* for the composite val_score.
    Adding a fifth benchmark means implementing this protocol once and
    adding one entry to the task list in run_benchmark().
    """

    name: str = ""
    weight: float = 0.0        # contribution to val_score (sum across tasks == 1.0)
    default_task_limit: int = 0  # 0 = load all

    @property
    def max_tokens(self) -> int:
        return _MAX_TOKENS

    @abstractmethod
    def load_problems(self, task_limit: int) -> list[tuple]:
        """Load up to *task_limit* problems. Returns list of (id, entry) tuples."""
        ...

    @abstractmethod
    def build_prompt(self, entry: dict) -> str:
        """Build a generation prompt from a problem entry."""
        ...

    @abstractmethod
    def run_tests(self, code: str, entry: dict) -> bool:
        """Run tests against generated code. Returns True if all pass."""
        ...


class _EvalplusTask(EvalTask):
    """Shared base for HumanEval+ and MBPP+ — same loader/prompt/test pattern.
    Uses ``self.name`` as the dataset key for the evalplus helpers."""

    def load_problems(self, task_limit: int) -> list[tuple]:
        problems_dict = _load_problems(self.name)
        if not problems_dict:
            return []
        task_ids = list(problems_dict.keys())
        entries = [(tid, problems_dict[tid]) for tid in task_ids]
        if task_limit > 0:
            entries = entries[:task_limit]
        return entries

    def build_prompt(self, entry: dict) -> str:
        return _build_prompt(entry, self.name)

    def run_tests(self, code: str, entry: dict) -> bool:
        test_code = _get_test_code(entry, self.name)
        if not test_code:
            return False
        # If the model didn't include the function signature, prepend it
        entry_point = entry.get("entry_point", "")
        if entry_point and f"def {entry_point}" not in code:
            prompt_sig = entry.get("prompt", "")
            code = textwrap.indent(code, "    ")
            code = prompt_sig + "\n" + code
        return _run_tests(code, test_code)


class HumanEvalTask(_EvalplusTask):
    """HumanEval+ (164 algorithmic problems, evalplus strict tests)."""
    name = "humaneval"
    weight = 0.25


class MBPPTask(_EvalplusTask):
    """MBPP+ (974 entry-level problems, evalplus strict tests)."""
    name = "mbpp"
    weight = 0.25


class LiveCodeBenchTask(EvalTask):
    """LiveCodeBench v6 — contamination-free competitive programming, sampled."""
    name = "lcb"
    weight = 0.35
    default_task_limit = 10

    def load_problems(self, task_limit: int) -> list[tuple]:
        problems = _load_livecodebench(task_limit=task_limit)
        return [(i, p) for i, p in enumerate(problems)]

    def build_prompt(self, entry: dict) -> str:
        return _build_lcb_prompt(entry)

    def run_tests(self, code: str, entry: dict) -> bool:
        return _run_lcb_tests(code, entry)


class BigCodeBenchTask(EvalTask):
    """BigCodeBench Hard — 148 library-call tasks, sampled."""
    name = "bigcode"
    weight = 0.15
    default_task_limit = 10

    def load_problems(self, task_limit: int) -> list[tuple]:
        problems = _load_bigcodebench_hard(task_limit=task_limit)
        return [(i, p) for i, p in enumerate(problems)]

    def build_prompt(self, entry: dict) -> str:
        return _build_bigcode_prompt(entry)

    def run_tests(self, code: str, entry: dict) -> bool:
        return _run_bigcode_tests(code, entry)


# ---------------------------------------------------------------------------
# Eval runner (single dataset, prompt -> code -> tests)
# ---------------------------------------------------------------------------

def run_coding_eval(
    client: LlamaClient,
    task: EvalTask,
    gen_params: GenerationParams | None = None,
    task_limit: int | None = None,
) -> tuple[float, int, float]:
    """Run coding evaluation on a task. Returns (pass_at_1, total_tokens, total_seconds).

    The *task* parameter is any EvalTask implementation. Each task owns its
    loading, prompting, and test-execution strategy — no per-dataset dispatch
    inside this function.
    """
    limit = task_limit if task_limit is not None else task.default_task_limit
    entries = task.load_problems(limit)
    if not entries:
        return 0.0, 0, 0.0

    # Per-task max_tokens override via gen_params.with_overrides().
    gen = (gen_params or GenerationParams()).with_overrides(max_tokens=task.max_tokens)

    total = len(entries)
    passed = 0
    total_tokens = 0
    t_start = time.time()

    print(f"  [CODING] {task.name}: {total} tasks", flush=True)

    for i, (tid, entry) in enumerate(entries):
        prompt = task.build_prompt(entry)
        if not prompt:
            continue

        try:
            res = client.complete(prompt, gen=gen)
            usage = res.get("usage", {})
            total_tokens += int(usage.get("total_tokens", 0) or 0)
            # Combine content + reasoning_content. For thinking models, llama-server
            # emits think tokens into a separate reasoning_content field; if the
            # model runs out of tokens mid-think, content may be empty while
            # reasoning_content holds the think block. _strip_code handles both.
            raw_response = res.get("content", "") or ""
            reasoning = res.get("reasoning_content", "") or ""
            if raw_response.strip():
                combined = raw_response
            else:
                combined = reasoning
        except Exception as e:
            print(f"    {tid} FAIL: {e}", flush=True)
            continue

        code = _strip_code(combined)
        if not code:
            print(f"    {tid} FAIL (no code extracted) ({i+1}/{total})", flush=True)
            continue

        code = textwrap.dedent(code)

        ok = task.run_tests(code, entry)

        if ok:
            passed += 1
            print(f"    {tid} PASS ({i+1}/{total})", flush=True)
        else:
            print(f"    {tid} FAIL ({i+1}/{total})", flush=True)

    elapsed = time.time() - t_start
    pass_at_1 = passed / total if total > 0 else 0.0
    print(
        f"  [CODING] {task.name}: {passed}/{total} passed "
        f"(pass@1={pass_at_1:.4f}) TPS={total_tokens/elapsed if elapsed > 0 else 0:.1f}",
        flush=True,
    )
    return pass_at_1, total_tokens, elapsed


# ---------------------------------------------------------------------------
# Unified benchmark
# ---------------------------------------------------------------------------

def run_benchmark(client: LlamaClient, gen_params: GenerationParams | None = None, **kwargs) -> BenchmarkResult:
    """
    Unified entry point. Runs LCB, HE+, MBPP+, BigCodeBench Hard.

    val_pass1 = LCB, val_pass2 = HE, val_pass3 = MBPP, val_pass4 = BigCode.
    val_score is a weighted sum — each task carries its own weight
    (sum of weights == 1.0). Adding a fifth benchmark means implementing
    EvalTask once and adding one entry to the specs list below.
    """
    task_limit = kwargs.get("task_limit", 30)
    lcb_limit = kwargs.get("lcb_task_limit", getattr(bench_config, "LCB_TASK_LIMIT", 10))
    bigcode_limit = kwargs.get("bigcode_task_limit", getattr(bench_config, "BIGCODE_TASK_LIMIT", 10))

    # ── Run each task ──────────────────────────────────────────────────
    specs: list[tuple[EvalTask, int]] = [
        (HumanEvalTask(), task_limit),
        (MBPPTask(), task_limit),
        (LiveCodeBenchTask(), lcb_limit),
        (BigCodeBenchTask(), bigcode_limit),
    ]

    passes: dict[str, float] = {}
    total_tokens = 0
    total_seconds = 0.0

    for task, limit in specs:
        pass1, tokens, elapsed = run_coding_eval(
            client, task, gen_params=gen_params, task_limit=limit,
        )
        passes[task.name] = pass1
        total_tokens += tokens
        total_seconds += elapsed

    # ── Weighted composite score ───────────────────────────────────────
    val_score = round(sum(task.weight * passes.get(task.name, 0.0) for task, _ in specs), 6)

    lcb_pass = passes.get("lcb", 0.0)
    he_pass = passes.get("humaneval", 0.0)
    mbpp_pass = passes.get("mbpp", 0.0)
    bigcode_pass = passes.get("bigcode", 0.0)

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

