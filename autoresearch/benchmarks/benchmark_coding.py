"""
Coding performance benchmark using direct LLM generation.
Respects all generation flags (temp, top_p, top_k, etc.) via LlamaClient.

Active benchmarks (val_score = 0.35*LCB + 0.25*HE_strict + 0.25*MBPP + 0.15*BigCode):
  - HumanEval+ (strict)  : 164 algorithmic problems, evalplus `test` field
  - MBPP+                : 974 entry-level problems, evalplus `test` field
  - LiveCodeBench v6     : contamination-free competitive programming, sampled
  - BigCodeBench Hard    : 148 library-call tasks, sampled

Loaders fetch data on first use and cache to data/ subdir. Install
requirements-legacy.txt before running these optional quality evaluations:
  - HumanEval+/MBPP+  : evalplus
  - LiveCodeBench v6  : base64+zlib+pickle JSONL fetched via huggingface_hub
  - BigCodeBench Hard : HF datasets cache (datasets + pyarrow)

Coding test subprocesses always run with an isolated temp cwd so relative
artifacts (notably BigCodeBench ``files.zip``) cannot leak into the repo root.
"""

from __future__ import annotations

import base64
import json
import os
import pickle
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
import zlib
from datetime import UTC, datetime
from pathlib import Path

from autoresearch.benchmarks import bench_config
from autoresearch.benchmarks.benchmark_harness import BenchmarkResult
from autoresearch.core.evaluation_protocol import CODING_MAX_TOKENS, CODING_PROTOCOL
from autoresearch.core.llama_client import GenerationParams, LlamaClient
from autoresearch.core.process_guard import spawn_kwargs

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR.parent / "data" / "benchmark_cache"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_THINK_CLOSED_RE = re.compile(
    r"<(?:ifm\|)?think(?:_fast|_faster)?(?:\s+[^>\r\n]*?)?>.*?</(?:ifm\|)?think(?:_fast|_faster)?\s*>",
    re.DOTALL | re.IGNORECASE,
)
_THINK_OPEN_RE = re.compile(
    r"(?:^|(?<=[\r\n\s]))<(?:ifm\|)?think(?:_fast|_faster)?(?:\s+[^>\r\n]*?)?>.*$",
    re.DOTALL | re.IGNORECASE,
)
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
    """Run a Python script in a sandboxed subprocess. Returns (returncode, stdout, stderr).

    Always uses a private temp cwd. BigCodeBench (and bad model solutions) can
    write relative paths such as ``files.zip``; without isolation those leak into
    the repo root during coding-10 Trials.
    """
    try:
        with tempfile.TemporaryDirectory(prefix="coding_sandbox_") as tmp:
            result = subprocess.run(
                **spawn_kwargs(
                    args=[sys.executable, "-c", script],
                    input=stdin_input,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    cwd=tmp,
                    env={
                        **os.environ,
                        "PYTHONDONTWRITEBYTECODE": "1",
                        # Headless: a solution can call plt.show()/WordCloud
                        # (BigCodeBench Hard task 4) — Agg renders to a file
                        # instead of popping a Figure window on the desktop rig.
                        "MPLBACKEND": "Agg",
                    },
                )
            )
            return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "TIMEOUT"
    except Exception as e:
        return -1, "", str(e)


def _run_tests(code: str, test_code: str, errors: list[str] | None = None) -> bool:
    """Run generated code + test code in a sandboxed subprocess. Returns True if all tests pass."""
    rc, stdout, stderr = _run_subprocess(f"{code}\n\n{test_code}\n")
    if rc != 0 and errors is not None:
        errors.append(stderr or stdout or f"test_exit={rc}")
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
    if target.exists() and not target.is_symlink() and target.stat().st_size > 0 and not force:
        return target
    try:
        from huggingface_hub import hf_hub_download

        # download into HF cache, then copy to canonical name (no symlink —
        # Windows without Developer Mode raises WinError 1314 on symlink_to).
        src = hf_hub_download(
            "livecodebench/code_generation_lite",
            filename=LCB_FILENAME,
            repo_type="dataset",
            cache_dir=str(LCB_CACHE_DIR / "_hf_cache"),
        )
        if target.exists() or target.is_symlink():
            target.unlink()
        shutil.copy2(src, target)
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
        'If a function signature is provided, wrap the call in `if __name__ == "__main__": solve()` '
        "or invoke the class appropriately. Return ONLY the code, no explanations."
    )
    return "\n".join(p for p in parts if p).strip()


def _run_lcb_tests(code: str, entry: dict, errors: list[str] | None = None) -> bool:
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
        rc, stdout, stderr = _run_subprocess(code, stdin_input=inp)
        if rc != 0:
            if errors is not None:
                errors.append(stderr or stdout or f"test_exit={rc}")
            return False
        if stdout.rstrip("\n") != expected:
            if errors is not None:
                errors.append(f"stdout mismatch: expected={expected!r}, actual={stdout!r}")
            return False
    return True


# ---------------------------------------------------------------------------
# BigCodeBench Hard loader
# ---------------------------------------------------------------------------

BCB_CACHE_DIR = DATA_DIR / "bigcodebench_hard"
BCB_DATASET = "bigcode/bigcodebench-hard"
BCB_SPLIT = "v0.1.4"  # 148 problems, ~2.76 MB; newest upstream revision


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


def _run_bigcode_tests(code: str, entry: dict, errors: list[str] | None = None) -> bool:
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
    # rather than pytest (not always installed). We route failures to stderr so
    # assertion details are observable during debug / diagnosis.
    # On Windows, normalize POSIX-specific shell "not found" assertions.
    win_compat = ""
    if sys.platform == "win32":
        win_compat = (
            "_orig_assertIn = unittest.TestCase.assertIn\n"
            "def _compat_assertIn(self, member, container, msg=None):\n"
            "    if member == 'not found' and isinstance(container, str) and ('not recognized' in container or 'not found' in container):\n"
            "        return\n"
            "    return _orig_assertIn(self, member, container, msg=msg)\n"
            "unittest.TestCase.assertIn = _compat_assertIn\n\n"
        )
    script = (
        f"import os, sys, unittest\n"
        f"{win_compat}"
        f"{code}\n\n"
        f"{test_code}\n\n"
        f"loader = unittest.TestLoader()\n"
        f"suite = loader.loadTestsFromTestCase(TestCases)\n"
        f"runner = unittest.TextTestRunner(verbosity=0, stream=sys.stderr)\n"
        f"result = runner.run(suite)\n"
        f"sys.exit(0 if result.wasSuccessful() else 1)\n"
    )
    rc, stdout, stderr = _run_subprocess(script)
    if rc != 0 and errors is not None:
        errors.append(stderr or stdout or f"test_exit={rc}")
    if rc != 0 and stderr and os.environ.get("AUTORESEARCH_DEBUG"):
        print(f"    [BigCode fail] {stderr.strip()[:300]}")
    return rc == 0


# ---------------------------------------------------------------------------
# EvalTask — one instance per dataset (weight lives on the instance)
# ---------------------------------------------------------------------------

# Fixed capability protocol: thinking and final code share this output pool.
_MAX_TOKENS = CODING_MAX_TOKENS


class EvalTask:
    """One coding dataset: load problems, build prompts, run tests, hold weight."""

    name: str = ""
    weight: float = 0.0  # contribution to val_score (sum across tasks == 1.0)
    default_task_limit: int = 0  # 0 = load all

    @property
    def max_tokens(self) -> int:
        return _MAX_TOKENS

    def load_problems(self, task_limit: int) -> list[tuple]:
        raise NotImplementedError

    def build_prompt(self, entry: dict) -> str:
        raise NotImplementedError

    def run_tests(self, code: str, entry: dict) -> bool:
        raise NotImplementedError


class EvalplusTask(EvalTask):
    """HumanEval+ / MBPP+ — same loader/prompt/test pattern via evalplus helpers."""

    def __init__(self, name: str, weight: float = 0.25):
        self.name = name
        self.weight = weight

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
        return _run_tests(code, test_code, errors=self.test_errors)


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
        return _run_lcb_tests(code, entry, errors=self.test_errors)


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
        return _run_bigcode_tests(code, entry, errors=self.test_errors)


# ---------------------------------------------------------------------------
# Eval runner (single dataset, prompt -> code -> tests)
# ---------------------------------------------------------------------------


def run_coding_eval(
    client: LlamaClient,
    task: EvalTask,
    gen_params: GenerationParams | None = None,
    task_limit: int | None = None,
    artifact: dict | None = None,
    artifact_path: Path | None = None,
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
        record = {
            "dataset": task.name,
            "task_id": str(tid),
            "max_tokens": gen.max_tokens,
            "content": "",
            "reasoning_content": "",
            "code": "",
            "usage": {},
            "finish_reason": None,
            "completion_tokens": None,
            "budget_exhausted": False,
            "passed": False,
            "error": "",
        }
        task.test_errors = []
        started = time.time()
        try:
            prompt = task.build_prompt(entry)
            record["prompt"] = prompt
            if not prompt:
                record["error"] = "empty_prompt"
                continue
            res = client.complete(prompt, gen=gen)
            usage = res.get("usage", {})
            total_tokens += int(usage.get("total_tokens", 0) or 0)
            content = res.get("content") or ""
            reasoning = res.get("reasoning_content") or ""
            completion_tokens = usage.get("completion_tokens")
            exhausted = completion_tokens is not None and completion_tokens >= gen.max_tokens
            record.update(
                content=content,
                reasoning_content=reasoning,
                usage=usage,
                finish_reason=res.get("finish_reason"),
                completion_tokens=completion_tokens,
                budget_exhausted=exhausted,
            )
            if exhausted:
                task.test_errors.append("budget_exhausted")
            # Reasoning is diagnostic material, never an executable final solution.
            code = textwrap.dedent(_strip_code(content))
            record["code"] = code
            if not code:
                task.test_errors.append("reasoning_only" if reasoning else "no_code")
                continue
            record["passed"] = bool(task.run_tests(code, entry))
            if record["passed"]:
                passed += 1
            elif not task.test_errors:
                task.test_errors.append("tests_failed")
        except Exception as e:
            task.test_errors.append(str(e))
        finally:
            record["error"] = " | ".join(filter(None, [record["error"], *task.test_errors]))
            record["elapsed_sec"] = round(time.time() - started, 3)
            if artifact is not None and artifact_path is not None:
                artifact["tasks"].append(record)
                _write_artifact(artifact_path, artifact)
            label = "PASS" if record["passed"] else "FAIL"
            print(f"    {tid} {label} ({i + 1}/{total}) {record['error'][:300]}", flush=True)

    elapsed = time.time() - t_start
    pass_at_1 = passed / total if total > 0 else 0.0
    print(
        f"  [CODING] {task.name}: {passed}/{total} passed "
        f"(pass@1={pass_at_1:.4f}) TPS={total_tokens / elapsed if elapsed > 0 else 0:.1f}",
        flush=True,
    )
    return pass_at_1, total_tokens, elapsed


# ---------------------------------------------------------------------------
# Unified benchmark
# ---------------------------------------------------------------------------


def _write_artifact(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def run_benchmark(
    client: LlamaClient, gen_params: GenerationParams | None = None, **kwargs
) -> BenchmarkResult:
    """
    Unified entry point. Runs LCB, HE+, MBPP+, BigCodeBench Hard.

    val_pass1 = LCB, val_pass2 = HE, val_pass3 = MBPP, val_pass4 = BigCode.
    val_score is a weighted sum — each task carries its own weight
    (sum of weights == 1.0). Adding a fifth benchmark means implementing
    EvalTask once and adding one entry to the specs list below.
    """
    task_limit = kwargs.get("task_limit", 30)
    lcb_limit = kwargs.get("lcb_task_limit", getattr(bench_config, "LCB_TASK_LIMIT", 10))
    bigcode_limit = kwargs.get(
        "bigcode_task_limit", getattr(bench_config, "BIGCODE_TASK_LIMIT", 10)
    )

    model = Path(kwargs.get("model_name") or "model").name
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")
    filename = re.sub(r"[^\w.-]", "_", model)
    artifact_path = Path(
        kwargs.get("artifact_path")
        or (ROOT_DIR.parent / "runners" / "logs" / f"coding-{stamp}-{filename}.json")
    )
    artifact = {
        "model": model,
        "coding_protocol": CODING_PROTOCOL,
        "max_tokens": CODING_MAX_TOKENS,
        "generation_params": (gen_params or GenerationParams())
        .with_overrides(max_tokens=CODING_MAX_TOKENS)
        .to_payload(),
        "tasks": [],
    }
    _write_artifact(artifact_path, artifact)
    print(f"  [coding-artifact] {artifact_path}", flush=True)

    # ── Run each task ──────────────────────────────────────────────────
    specs: list[tuple[EvalTask, int]] = [
        (EvalplusTask("humaneval", 0.25), task_limit),
        (EvalplusTask("mbpp", 0.25), task_limit),
        (LiveCodeBenchTask(), lcb_limit),
        (BigCodeBenchTask(), bigcode_limit),
    ]

    passes: dict[str, float] = {}
    total_tokens = 0
    total_seconds = 0.0

    for task, limit in specs:
        pass1, tokens, elapsed = run_coding_eval(
            client,
            task,
            gen_params=gen_params,
            task_limit=limit,
            artifact=artifact,
            artifact_path=artifact_path,
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
