"""
Coding performance benchmark using EvalPlus (HumanEval+ / MBPP+).
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path
import sys
from typing import Any

from autoresearch.core import config
import argparse

# Default module-level constants (not mutated)
MODELS_TO_BENCHMARK = [config.MODEL]
INCLUDE_CODING = config.INCLUDE_CODING
CODING_TASK_LIMIT = config.CODING_TASK_LIMIT
CTX_SIZE = config.CTX_SIZE
PORT = 18080

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ctx-size", type=int, default=config.CTX_SIZE)
    parser.add_argument("--kv-cache", type=str, default=config.KV_CACHE)
    parser.add_argument("--kv-cache-k", type=str, default=config.KV_CACHE_K)
    parser.add_argument("--kv-cache-v", type=str, default=config.KV_CACHE_V)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--model", type=str, default=config.MODEL)
    parser.add_argument("--include-coding", action="store_true", default=config.INCLUDE_CODING, dest="include_coding")
    parser.add_argument("--coding-task-limit", type=int, default=config.CODING_TASK_LIMIT)
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--ubatch-size", type=int, default=config.UBATCH_SIZE)
    parser.add_argument("--threads", type=int, default=config.THREADS)
    parser.add_argument("--threads-batch", type=int, default=config.THREADS_BATCH)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--ngl", type=int, default=99)
    parser.add_argument("--flash-attn", type=str, choices=["on", "off"], default=config.FLASH_ATTN)
    parser.add_argument("--spec-draft-n-max", type=int, default=config.SPEC_DRAFT_N_MAX)
    
    args, unknown = parser.parse_known_args()
    return args

# ---------------------------------------------------------------------------
# Plumbing & CLI Overrides
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent
MODELS_DIR = Path(os.environ.get("AUTORESEARCH_MODELS_DIR", ROOT_DIR / "models"))
import urllib.request

if __name__ == "__main__":
    if len(sys.argv) > 1:
        parse_args()

def run_evalplus(dataset: str, port: int, output_dir: Path, model_name: str, task_limit: int = 30, timeout_at: float | None = None, max_tokens: int = 1024) -> dict:
    """Run EvalPlus codegen and evaluate."""
    print(f"  [EvalPlus] Running {dataset} (limit: {task_limit if task_limit > 0 else 'full'})...")
    
    # We use a subfolder for each model to avoid overwriting
    model_safe_name = model_name.replace("/", "_").replace(".gguf", "")
    res_root = output_dir / model_safe_name
    res_root.mkdir(parents=True, exist_ok=True)

    # Step 1: Codegen
    env = os.environ.copy()
    env["EVALPLUS_MAX_TOKENS"] = str(max_tokens)

    try:
        import evalplus
        has_evalplus = True
    except ImportError:
        has_evalplus = False

    if has_evalplus:
        base_cmd = [sys.executable]
    else:
        base_cmd = ["uv", "run", "--with", "evalplus", "python3"]

    codegen_cmd = base_cmd + [
        str(ROOT_DIR / "evalplus_wrapper.py"),
        "--model", "local-model",
        "--dataset", dataset,
        "--base_url", f"http://localhost:{port}/v1",
        "--greedy",
        "--backend", "openai",
        "--root", str(res_root),
        "--resume", "False"
    ]
    
    # Use global ID_RANGE if provided via CLI argument --id_range (legacy)
    if task_limit > 0:
        codegen_cmd += ["--id_range", f"[0, {task_limit}]"]
    
    print(f"    Executing: {' '.join(codegen_cmd)}")
    
    remaining = None
    if timeout_at is not None:
        remaining = max(1.0, timeout_at - time.time())
        if remaining <= 1.0:
            raise TimeoutError("Trial time budget exceeded")
            
    try:
        subprocess.run(codegen_cmd, check=True, env=env, timeout=remaining)
    except subprocess.TimeoutExpired as e:
        raise TimeoutError("Trial time budget exceeded") from e
    
    # Step 2: Evaluate
    # Find the specific jsonl for this dataset
    search_dir = res_root / dataset
    samples_files = list(search_dir.glob("*.jsonl"))
    samples_files = [f for f in samples_files if not f.name.endswith(".raw.jsonl")]
    
    if not samples_files:
        print(f"FAIL: No samples found in {search_dir}")
        return {}
    
    samples_file = samples_files[0]
    print(f"    Evaluating samples: {samples_file}")

    if has_evalplus:
        eval_cmd = [sys.executable, "-m", "evalplus.evaluate"]
    else:
        eval_cmd = ["uv", "run", "--with", "evalplus", "python3", "-m", "evalplus.evaluate"]
    eval_cmd += [
        "--dataset", dataset,
        "--samples", str(samples_file)
    ]
    
    if task_limit > 0 and task_limit <= 30:
        eval_cmd += ["--i_just_wanna_run"]
    
    if timeout_at is not None:
        remaining = max(1.0, timeout_at - time.time())
        if remaining <= 1.0:
            raise TimeoutError("Trial time budget exceeded")
            
    try:
        result = subprocess.run(eval_cmd, capture_output=True, text=True, timeout=remaining)
    except subprocess.TimeoutExpired as e:
        raise TimeoutError("Trial time budget exceeded") from e
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    
    # Parse scores
    scores = {}
    for line in result.stdout.splitlines():
        # Look for pass@1 scores
        if "pass@1" in line.lower():
            try:
                # Format could be: "humaneval (base) pass@1: 0.XXXX"
                # or "humaneval (plus) pass@1: 0.XXXX"
                val = float(line.split(":")[-1].strip())
                if "plus" in line.lower():
                    scores["pass1_plus"] = val
                else:
                    scores["pass1_base"] = val
            except:
                pass
    
    # Load stats if available
    import json
    stats_file = res_root / f"stats_{dataset}.json"
    if stats_file.exists():
        try:
            with open(stats_file) as f:
                stats = json.load(f)
                scores["total_tokens"] = stats.get("total_tokens", 0)
                scores["total_seconds"] = stats.get("total_seconds", 0.0)
        except Exception:
            pass

    return scores

from autoresearch.core.llama_client import LlamaClient
from autoresearch.benchmarks.benchmark_harness import BenchmarkResult

def run_benchmark(client: LlamaClient, **kwargs) -> BenchmarkResult:
    """Unified entry point for in-process orchestration (Coding focus)."""
    task_limit = kwargs.get("task_limit", 30)
    timeout_at = kwargs.get("timeout_at", None)
    max_tokens = kwargs.get("max_tokens", 1024)
    output_base = ROOT_DIR / "coding_results"
    if output_base.exists():
        import shutil
        shutil.rmtree(output_base, ignore_errors=True)
    output_base.mkdir(parents=True, exist_ok=True)
    model_name = kwargs.get("model_name", "local-model")


    # Run HumanEval and MBPP
    he_scores = run_evalplus("humaneval", client.port, output_base, model_name, task_limit=task_limit, timeout_at=timeout_at, max_tokens=max_tokens)
    mbpp_scores = run_evalplus("mbpp", client.port, output_base, model_name, task_limit=task_limit, timeout_at=timeout_at, max_tokens=max_tokens)
    
    # Map to BenchmarkResult
    # val_pass1 = HumanEval+, val_pass2 = MBPP+
    he_val = he_scores.get("pass1_plus", he_scores.get("pass1_base", 0))
    mbpp_val = mbpp_scores.get("pass1_plus", mbpp_scores.get("pass1_base", 0))
    
    avg_score = (he_val + mbpp_val) / 2
    
    total_tokens = he_scores.get("total_tokens", 0) + mbpp_scores.get("total_tokens", 0)
    total_seconds = he_scores.get("total_seconds", 0.0) + mbpp_scores.get("total_seconds", 0.0)
    avg_tps = total_tokens / total_seconds if total_seconds > 0 else 0.0

    return BenchmarkResult(
        val_score=round(avg_score, 4),
        val_pass1=he_val,
        val_pass2=mbpp_val,
        avg_tps=round(avg_tps, 2),
        total_seconds=round(total_seconds, 2)
    )

def main():
    print("Starting Coding Benchmark script...")
    args = parse_args()
    task_limit = 30 if "--test" in sys.argv else args.coding_task_limit
    
    from autoresearch.core.llama_runner import LlamaServerRunner, ServerIntent
    
    model_name = args.model
    model_path = MODELS_DIR / model_name
    
    intent = ServerIntent(
        model_path=model_path,
        ctx_size=args.ctx_size,
        kv_cache=args.kv_cache,
        flash_attn=args.flash_attn,
        port=args.port,
        batch_size=args.batch_size,
        ubatch_size=args.ubatch_size,
        threads=args.threads,
        ngl=args.ngl,
        parallel=args.parallel,
        kv_cache_k=args.kv_cache_k,
        kv_cache_v=args.kv_cache_v,
        threads_batch=args.threads_batch,
        spec_draft_n_max=args.spec_draft_n_max
    )

    try:
        with LlamaServerRunner(intent) as runner:
            print(f"Server is ready on port {runner.port}. Starting EvalPlus...")
            client = LlamaClient(runner.port)
            
            result = run_benchmark(client, model_name=model_name, task_limit=task_limit, max_tokens=args.max_tokens)
            
            print("\n" + "="*80)
            print("CODING BENCHMARK RESULTS")
            print("="*80)
            print(f"Model: {model_name}")
            print(f"val_score: {result.val_score:.4f}")
            print(f"HE (pass1): {result.val_pass1:.4f}")
            print(f"MBPP (pass2): {result.val_pass2:.4f}")
            print("="*80)
    except RuntimeError as e:
        print(f"FAIL: {e}")

if __name__ == "__main__":
    main()