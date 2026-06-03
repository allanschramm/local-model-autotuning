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
LLAMA_CPP_ROOT = Path(
    os.environ.get("AUTORESEARCH_LLAMA_CPP_ROOT", ROOT_DIR / "llama.cpp")
)
LLAMA_SERVER_CANDIDATES = (
    LLAMA_CPP_ROOT / "build-cuda" / "bin" / "llama-server",
    LLAMA_CPP_ROOT / "build" / "bin" / "llama-server",
)

def resolve_llama_server() -> Path:
    for candidate in LLAMA_SERVER_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("llama-server not found.")

import urllib.request

def build_llama_cmd(
    llama_server: Path,
    model_path: Path,
) -> list[str]:
    return [
        str(llama_server),
        "--model", str(model_path),
        "--host", "127.0.0.1",
        "--ctx-size", str(CTX_SIZE),
        "--batch-size", str(BATCH_SIZE),
        "--ubatch-size", str(UBATCH_SIZE),
        "--threads", str(THREADS),
        "--parallel", "1",
        "--n-gpu-layers", str(NGL),
        "--cache-type-k", KV_CACHE_TYPE,
        "--cache-type-v", KV_CACHE_TYPE,
        "--flash-attn", FLASH_ATTN,
        "--verbose"
    ]

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

def main():
    print("Starting Coding Benchmark script...")
    is_test = "--test" in sys.argv
    try:
        llama_server = resolve_llama_server()
        print(f"Using llama-server: {llama_server}")
    except Exception as e:
        print(f"FATAL: {e}")
        return

    server_env = os.environ.copy()
    
    results = []
    
    output_base = ROOT_DIR / "coding_results"
    output_base.mkdir(parents=True, exist_ok=True)

    print(f"Checking models in: {MODELS_DIR}")
    for model_name in MODELS_TO_BENCHMARK:
        model_path = MODELS_DIR / model_name
        if not model_path.exists():
            print(f"SKIP: {model_name} not found at {model_path}.")
            continue

        print(f"\nBenchmarking Coding: {model_name}")
        
        from llama_runner import LlamaServerRunner
        
        cmd = build_llama_cmd(llama_server, model_path)
        log_file = ROOT_DIR / "llama_server_coding.log"
        
        try:
            with LlamaServerRunner(cmd, server_env, PORT, timeout=600, log_path=log_file) as runner:
                print(f"Server is ready on port {runner.port}. Starting EvalPlus...")
                
                # Run HumanEval and MBPP
                he_scores = run_evalplus("humaneval", runner.port, output_base, model_name, is_test=is_test)
                mbpp_scores = run_evalplus("mbpp", runner.port, output_base, model_name, is_test=is_test)
                
                results.append({
                    "model": model_name,
                    "he_base": he_scores.get("pass1_base", 0),
                    "he_plus": he_scores.get("pass1_plus", 0),
                    "mbpp_base": mbpp_scores.get("pass1_base", 0),
                    "mbpp_plus": mbpp_scores.get("pass1_plus", 0)
                })
        except RuntimeError as e:
            print(f"FAIL: {e}")
            continue

    print("\n" + "="*80)
    print("CODING BENCHMARK RESULTS")
    print("="*80)
    print(f"{'Model':<30} {'HE (Base)':>10} {'HE (Plus)':>10} {'MBPP (B)':>10} {'MBPP (P)':>10}")
    for r in results:
        print(f"{r['model']:<30} {r['he_base']:>10.1%} {r['he_plus']:>10.1%} {r['mbpp_base']:>10.1%} {r['mbpp_plus']:>10.1%}")
    print("="*80)

if __name__ == "__main__":
    main()