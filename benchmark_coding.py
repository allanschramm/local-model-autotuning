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

# Import search surface defaults from benchmark_search.py (single source of truth)
import benchmark_search
import config

MODELS_TO_BENCHMARK = [benchmark_search.MODEL]
INCLUDE_CODING = config.INCLUDE_CODING
CODING_TASK_LIMIT = config.CODING_TASK_LIMIT

CTX_SIZE = benchmark_search.CTX_SIZE
KV_CACHE_TYPE = benchmark_search.KV_CACHE
KV_CACHE_K = benchmark_search.KV_CACHE_K
KV_CACHE_V = benchmark_search.KV_CACHE_V
BATCH_SIZE = benchmark_search.BATCH_SIZE
UBATCH_SIZE = benchmark_search.UBATCH_SIZE
THREADS = benchmark_search.THREADS
THREADS_BATCH = benchmark_search.THREADS_BATCH
PARALLEL = 1
NGL = 99
FLASH_ATTN = benchmark_search.FLASH_ATTN
SPEC_DRAFT_N_MAX = benchmark_search.SPEC_DRAFT_N_MAX

TEMP = 0.0 # Greedy for coding
MAXTOK = 1024
PORT = 18080
# Throughput target or other globals...

import argparse

def parse_args():
    global CTX_SIZE, KV_CACHE_TYPE, MAXTOK, MODELS_TO_BENCHMARK, PORT
    global KV_CACHE_K, KV_CACHE_V, BATCH_SIZE, UBATCH_SIZE, THREADS, THREADS_BATCH, PARALLEL, NGL, FLASH_ATTN, SPEC_DRAFT_N_MAX
    global INCLUDE_CODING, CODING_TASK_LIMIT
    parser = argparse.ArgumentParser()
    parser.add_argument("--ctx-size", type=int, default=CTX_SIZE)
    parser.add_argument("--kv-cache", type=str, default=KV_CACHE_TYPE)
    parser.add_argument("--kv-cache-k", type=str, default=None)
    parser.add_argument("--kv-cache-v", type=str, default=None)
    parser.add_argument("--max-tokens", type=int, default=MAXTOK)
    parser.add_argument("--model", type=str)
    parser.add_argument("--include-coding", action="store_true", default=INCLUDE_CODING, dest="include_coding")
    parser.add_argument("--coding-task-limit", type=int, default=CODING_TASK_LIMIT)
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--ubatch-size", type=int, default=UBATCH_SIZE)
    parser.add_argument("--threads", type=int, default=THREADS)
    parser.add_argument("--threads-batch", type=int, default=None)
    parser.add_argument("--parallel", type=int, default=PARALLEL)
    parser.add_argument("--ngl", type=int, default=NGL)
    parser.add_argument("--flash-attn", type=str, choices=["on", "off"], default=FLASH_ATTN)
    parser.add_argument("--spec-draft-n-max", type=int, default=SPEC_DRAFT_N_MAX)
    
    args, unknown = parser.parse_known_args()
    
    CTX_SIZE = args.ctx_size
    KV_CACHE_TYPE = args.kv_cache
    KV_CACHE_K = args.kv_cache_k
    KV_CACHE_V = args.kv_cache_v
    MAXTOK = args.max_tokens
    CODING_TASK_LIMIT = args.coding_task_limit
    INCLUDE_CODING = args.include_coding
    PORT = args.port
    BATCH_SIZE = args.batch_size
    UBATCH_SIZE = args.ubatch_size
    THREADS = args.threads
    THREADS_BATCH = args.threads_batch
    PARALLEL = args.parallel
    NGL = args.ngl
    FLASH_ATTN = args.flash_attn
    SPEC_DRAFT_N_MAX = args.spec_draft_n_max
    if args.model:
        MODELS_TO_BENCHMARK = [args.model]

# Benchmarks to run
INCLUDE_CODING = getattr(benchmark_search, "INCLUDE_CODING", True)
CODING_TASK_LIMIT = getattr(benchmark_search, "CODING_TASK_LIMIT", 30)

# ---------------------------------------------------------------------------
# Plumbing & CLI Overrides
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent
MODELS_DIR = Path(os.environ.get("AUTORESEARCH_MODELS_DIR", ROOT_DIR / "models"))
import urllib.request

if __name__ == "__main__":
    if len(sys.argv) > 1:
        parse_args()

def run_evalplus(dataset: str, port: int, output_dir: Path, model_name: str, task_limit: int = 30) -> dict:
    """Run EvalPlus codegen and evaluate."""
    print(f"  [EvalPlus] Running {dataset} (limit: {task_limit if task_limit > 0 else 'full'})...")
    
    # We use a subfolder for each model to avoid overwriting
    model_safe_name = model_name.replace("/", "_").replace(".gguf", "")
    res_root = output_dir / model_safe_name
    res_root.mkdir(parents=True, exist_ok=True)

    # Step 1: Codegen
    env = os.environ.copy()
    env["EVALPLUS_MAX_TOKENS"] = str(MAXTOK)

    codegen_cmd = [
        "uv", "run", "--with", "evalplus", "python3", str(ROOT_DIR / "evalplus_wrapper.py"),
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
    subprocess.run(codegen_cmd, check=True, env=env)
    
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

    eval_cmd = [
        "uv", "run", "--with", "evalplus", "python3", "-m", "evalplus.evaluate",
        "--dataset", dataset,
        "--samples", str(samples_file)
    ]
    
    if task_limit > 0 and task_limit <= 30:
        eval_cmd += ["--i_just_wanna_run"]
    
    result = subprocess.run(eval_cmd, capture_output=True, text=True)
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
    
    return scores

from llama_client import LlamaClient
from benchmark_harness import BenchmarkResult

def run_benchmark(client: LlamaClient, **kwargs) -> BenchmarkResult:
    """Unified entry point for in-process orchestration (Coding focus)."""
    task_limit = kwargs.get("task_limit", 30)
    output_base = ROOT_DIR / "coding_results"
    output_base.mkdir(parents=True, exist_ok=True)
    model_name = kwargs.get("model_name", "local-model")

    # Run HumanEval and MBPP
    he_scores = run_evalplus("humaneval", client.port, output_base, model_name, task_limit=task_limit)
    mbpp_scores = run_evalplus("mbpp", client.port, output_base, model_name, task_limit=task_limit)
    
    # Map to BenchmarkResult
    # val_pass1 = HumanEval+, val_pass2 = MBPP+
    he_val = he_scores.get("pass1_plus", he_scores.get("pass1_base", 0))
    mbpp_val = mbpp_scores.get("pass1_plus", mbpp_scores.get("pass1_base", 0))
    
    avg_score = (he_val + mbpp_val) / 2
    
    return BenchmarkResult(
        val_score=round(avg_score, 4),
        val_pass1=he_val,
        val_pass2=mbpp_val,
        avg_tps=0.0, # EvalPlus doesn't easily provide raw TPS here
        total_seconds=0.0
    )

def main():
    print("Starting Coding Benchmark script...")
    task_limit = 30 if "--test" in sys.argv else CODING_TASK_LIMIT
    
    from llama_runner import LlamaServerRunner, ServerIntent
    
    model_name = MODELS_TO_BENCHMARK[0]
    model_path = MODELS_DIR / model_name
    
    intent = ServerIntent(
        model_path=model_path,
        ctx_size=CTX_SIZE,
        kv_cache=KV_CACHE_TYPE,
        flash_attn=FLASH_ATTN,
        port=PORT,
        batch_size=BATCH_SIZE,
        ubatch_size=UBATCH_SIZE,
        threads=THREADS,
        ngl=NGL,
        parallel=PARALLEL,
        kv_cache_k=KV_CACHE_K,
        kv_cache_v=KV_CACHE_V,
        threads_batch=THREADS_BATCH,
        spec_draft_n_max=SPEC_DRAFT_N_MAX
    )

    try:
        with LlamaServerRunner(intent) as runner:
            print(f"Server is ready on port {runner.port}. Starting EvalPlus...")
            client = LlamaClient(runner.port)
            
            result = run_benchmark(client, model_name=model_name, task_limit=task_limit)
            
            print("\\n" + "="*80)
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