"""
Coding performance benchmark using direct LLM generation.
Respects all generation flags (temp, top_p, top_k, etc.) via LlamaClient.
Uses evalplus data for problem definitions and test cases.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from autoresearch.core import config
from autoresearch.core.llama_client import LlamaClient
from autoresearch.benchmarks.benchmark_harness import BenchmarkResult

ROOT_DIR = Path(__file__).resolve().parent


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ctx-size", type=int, default=config.CTX_SIZE)
    parser.add_argument("--model", type=str, default=config.MODEL)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--coding-task-limit", type=int, default=config.CODING_TASK_LIMIT)
    parser.add_argument("--port", type=int, default=18080)
    args, _ = parser.parse_known_args()
    return args


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
    # Fallback: try older API
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
        prompt = entry.get("prompt", "")
        # Add instruction prefix
        return f"Complete the following Python function. Return ONLY the function body, no explanations.\n\n{prompt}"
    elif dataset == "mbpp":
        # MBPP has 'text' (description) and 'prompt' (function signature)
        text = entry.get("text", entry.get("prompt", ""))
        prompt = entry.get("prompt", "")
        return f"Write a Python function that satisfies the following description.\nDescription: {text}\n\n{prompt}"
    return ""


def _extract_code(raw_response: str) -> str:
    """Extract Python code from model response, stripping markdown and think tags."""
    text = raw_response.strip()

    # Remove thinking tags if present
    if "<think>" in text and "</think>" in text:
        start = text.find("</think>")
        if start != -1:
            text = text[start + len("</think>"):].strip()

    # Extract code from markdown blocks
    if "```python" in text:
        start = text.find("```python") + len("```python")
        end = text.find("```", start)
        if end != -1:
            text = text[start:end].strip()
    elif "```" in text:
        start = text.find("```") + 3
        end = text.find("```", start)
        if end != -1:
            text = text[start:end].strip()

    return text


def _run_tests(code: str, test_code: str, timeout: float = 10.0) -> bool:
    """Run generated code + test code in a sandboxed subprocess. Returns True if all tests pass."""
    # Build test script: code + test assertions
    test_script = f"{code}\n\n{test_code}\n"

    try:
        result = subprocess.run(
            [sys.executable, "-c", test_script],
            capture_output=True, text=True, timeout=timeout,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, Exception):
        return False


def _get_test_code(entry: dict, dataset: str) -> str:
    """Extract test assertions from a problem entry."""
    if dataset == "humaneval":
        # HumanEval+ has 'plus_input_output_tests' and 'base_input_output_tests'
        # Fall back to 'test' field
        test = entry.get("test", "")
        if not test:
            # Build from entry_point + check
            entry_point = entry.get("entry_point", "")
            base_tests = entry.get("base_input_output_tests", [])
            plus_tests = entry.get("plus_input_output_tests", [])
            if base_tests or plus_tests:
                # Generate assert statements from I/O pairs
                lines = []
                for inp, out in (base_tests + plus_tests):
                    lines.append(f"assert {entry_point}(*{repr(inp)}) == {repr(out)}")
                test = "\n".join(lines)
        return test
    elif dataset == "mbpp":
        # MBPP has 'test' field with assert statements
        test = entry.get("test", "")
        if not test:
            test_list = entry.get("test_list", [])
            test = "\n".join(test_list)
        return test
    return ""


def run_coding_eval(
    client: LlamaClient,
    dataset: str,
    task_limit: int = 0,
    timeout_at: float | None = None,
    **gen_kwargs
) -> tuple[float, int, float]:
    """
    Run coding evaluation on a dataset using LlamaClient directly.
    Returns (pass_at_1, total_tokens, total_seconds).
    """
    problems = _load_problems(dataset)
    if not problems:
        return 0.0, 0, 0.0

    task_ids = list(problems.keys())
    if task_limit > 0:
        task_ids = task_ids[:task_limit]

    total = len(task_ids)
    passed = 0
    total_tokens = 0
    t_start = time.time()

    print(f"  [CODING] {dataset}: {total} tasks", flush=True)

    for i, tid in enumerate(task_ids):
        if timeout_at and time.time() > timeout_at:
            print(f"  [CODING] {dataset}: timeout at {i}/{total}", flush=True)
            break

        entry = problems[tid]
        prompt = _build_prompt(entry, dataset)
        test_code = _get_test_code(entry, dataset)

        if not prompt or not test_code:
            continue

        # Generate code via LlamaClient — respects all gen_kwargs
        try:
            res = client.complete(prompt, **gen_kwargs)
            usage = res.get("usage", {})
            total_tokens += int(usage.get("total_tokens", 0) or 0)
            raw_response = res.get("content", "")
        except Exception as e:
            print(f"    {tid} FAIL: {e}", flush=True)
            continue

        # Extract code from response
        code = _extract_code(raw_response)

        # Run tests
        if _run_tests(code, test_code):
            passed += 1
            print(f"    {tid} PASS ({i+1}/{total})", flush=True)
        else:
            print(f"    {tid} FAIL ({i+1}/{total})", flush=True)

    elapsed = time.time() - t_start
    evaluated = len(task_ids) if not (timeout_at and time.time() > timeout_at) else passed + (total - passed)
    # Count actual evaluations from the loop
    pass_at_1 = passed / total if total > 0 else 0.0

    print(f"  [CODING] {dataset}: {passed}/{evaluated} passed (pass@1={pass_at_1:.4f}) TPS={total_tokens/elapsed:.1f}", flush=True)

    return pass_at_1, total_tokens, elapsed


def run_benchmark(client: LlamaClient, **kwargs) -> BenchmarkResult:
    """Unified entry point for coding benchmark. Respects all gen_kwargs."""
    task_limit = kwargs.get("task_limit", 30)
    timeout_at = kwargs.get("timeout_at", None)

    # Extract gen_kwargs (everything except our known params)
    gen_keys = {"task_limit", "timeout_at", "model_name", "is_test"}
    gen_kwargs = {k: v for k, v in kwargs.items() if k not in gen_keys and v is not None}

    # Run HumanEval
    he_pass, he_tokens, he_time = run_coding_eval(
        client, "humaneval", task_limit=task_limit, timeout_at=timeout_at, **gen_kwargs
    )

    # Run MBPP
    mbpp_pass, mbpp_tokens, mbpp_time = run_coding_eval(
        client, "mbpp", task_limit=task_limit, timeout_at=timeout_at, **gen_kwargs
    )

    avg_score = (he_pass + mbpp_pass) / 2
    total_tokens = he_tokens + mbpp_tokens
    total_seconds = he_time + mbpp_time
    avg_tps = total_tokens / total_seconds if total_seconds > 0 else 0.0

    return BenchmarkResult(
        val_score=round(avg_score, 4),
        val_pass1=he_pass,
        val_pass2=mbpp_pass,
        avg_tps=round(avg_tps, 2),
        total_seconds=round(total_seconds, 2)
    )


def main():
    """Standalone CLI entry point."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=config.MODEL)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--coding-task-limit", type=int, default=config.CODING_TASK_LIMIT)
    parser.add_argument("--port", type=int, default=18080)
    args, _ = parser.parse_known_args()

    from autoresearch.core.llama_runner import LlamaServerRunner, ServerIntent
    from pathlib import Path as P

    models_dir = P(os.environ.get("AUTORESEARCH_MODELS_DIR", ROOT_DIR.parent.parent / "models"))
    model_path = models_dir / args.model

    intent = ServerIntent(
        model_path=model_path, ctx_size=config.CTX_SIZE,
        kv_cache=config.KV_CACHE, flash_attn=config.FLASH_ATTN,
        port=args.port, batch_size=config.BATCH_SIZE, ubatch_size=config.UBATCH_SIZE,
        threads=config.THREADS, ngl=99, parallel=1,
        kv_cache_k=config.KV_CACHE_K, kv_cache_v=config.KV_CACHE_V,
        threads_batch=config.THREADS_BATCH, spec_draft_n_max=config.SPEC_DRAFT_N_MAX
    )

    with LlamaServerRunner(intent) as runner:
        client = LlamaClient(runner.port)
        result = run_benchmark(client, task_limit=args.coding_task_limit, max_tokens=args.max_tokens)
        print(f"\nCoding Score: {result.val_score:.4f}")
        print(f"HE pass@1: {result.val_pass1:.4f}")
        print(f"MBPP pass@1: {result.val_pass2:.4f}")


if __name__ == "__main__":
    main()
