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

# ---------------------------------------------------------------------------
# Search surface
# ---------------------------------------------------------------------------

MODELS_TO_BENCHMARK = [
    "gemma-4-E4B-it-Q4_K_M.gguf",
]

CTX_SIZE = 131072
KV_CACHE_TYPE = "q4_0"
BATCH_SIZE = 1024
UBATCH_SIZE = 256
THREADS = 8
PARALLEL = 1
NGL = 999
FLASH_ATTN = "on"

TEMP = 0.0 # Greedy for coding
MAXTOK = 1024
PORT = 18081 # Different port to avoid conflicts
# Throughput target or other globals...

import argparse

def parse_args():
    global CTX_SIZE, KV_CACHE_TYPE, MAXTOK, MODELS_TO_BENCHMARK, ID_RANGE
    parser = argparse.ArgumentParser()
    parser.add_argument("--ctx-size", type=int, default=CTX_SIZE)
    parser.add_argument("--kv-cache", type=str, default=KV_CACHE_TYPE)
    parser.add_argument("--max-tokens", type=int, default=MAXTOK)
    parser.add_argument("--model", type=str)
    parser.add_argument("--id_range", type=str, default=None)
    args = parser.parse_args()
    
    CTX_SIZE = args.ctx_size
    KV_CACHE_TYPE = args.kv_cache
    MAXTOK = args.max_tokens
    ID_RANGE = args.id_range
    if args.model:
        MODELS_TO_BENCHMARK = [args.model]

ID_RANGE = None
if __name__ == "__main__":
    if len(sys.argv) > 1:
        parse_args()

# ---------------------------------------------------------------------------
# Execution plumbing
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent
MODELS_DIR = Path(os.environ.get("AUTORESEARCH_MODELS_DIR", ROOT_DIR / "models"))
import urllib.request

def run_evalplus(dataset: str, port: int, output_dir: Path, model_name: str, is_test: bool = False) -> dict:
    """Run EvalPlus codegen and evaluate."""
    print(f"  [EvalPlus] Running {dataset}...")
    
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
    
    # Use global ID_RANGE if provided
    if ID_RANGE:
        codegen_cmd += ["--id_range", ID_RANGE]
    elif is_test:
        # Run only 2 tasks for quick validation
        codegen_cmd += ["--id_range", "[0, 2]"]
    
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
    
    if is_test:
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
    is_test = kwargs.get("is_test", False)
    output_base = ROOT_DIR / "coding_results"
    output_base.mkdir(parents=True, exist_ok=True)
    model_name = kwargs.get("model_name", "local-model")

    # Run HumanEval and MBPP
    he_scores = run_evalplus("humaneval", client.port, output_base, model_name, is_test=is_test)
    mbpp_scores = run_evalplus("mbpp", client.port, output_base, model_name, is_test=is_test)
    
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
    is_test = "--test" in sys.argv
    
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
        ngl=NGL
    )

    try:
        with LlamaServerRunner(intent) as runner:
            print(f"Server is ready on port {runner.port}. Starting EvalPlus...")
            client = LlamaClient(runner.port)
            
            result = run_benchmark(client, is_test=is_test, model_name=model_name)
            
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