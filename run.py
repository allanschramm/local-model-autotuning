#!/usr/bin/env python3
import os
# Set default llama.cpp root directory
os.environ.setdefault("AUTORESEARCH_LLAMA_CPP_ROOT", "/home/shark/workspace/Nexus-System/llama.cpp")

import sys
import csv
import time
import argparse
import subprocess
from pathlib import Path
from typing import Dict, Any

from llama_runner import LlamaServerRunner, ServerIntent
from llama_client import LlamaClient
from benchmark_harness import BenchmarkResult

# Benchmarks
from prepare import run_benchmark as run_nexus
from prepare_claw import run_benchmark as run_claw
from benchmark_coding import run_benchmark as run_coding

BASE_DIR = Path(__file__).resolve().parent
RESULTS_FILE = BASE_DIR / "results.tsv"
MODELS_DIR = BASE_DIR / "models"

def parse_args():
    parser = argparse.ArgumentParser(description="Unified AutoResearch Benchmark Runner")
    parser.add_argument("--desc", type=str, help="Description of the experiment (required for logging single runs)")
    parser.add_argument("--model", type=str, default="g4-opt-it-Q4_K_M.gguf", help="Model filename in models/ directory")
    parser.add_argument("--kv", type=str, default="q4_0", help="KV cache type (e.g. q4_0, q4_1, f16)")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Max generation tokens")
    parser.add_argument("--ctx-size", type=int, default=16384, help="Context size")
    parser.add_argument("--port", type=int, default=18080, help="Port for llama-server")
    parser.add_argument("--threads", type=int, default=12, help="Number of threads for llama-server")
    parser.add_argument("--ngl", type=int, default=99, help="Number of GPU layers to offload")
    parser.add_argument("--context-tokens", type=int, default=8192, help="Context tokens padding length")
    parser.add_argument("--include-coding", action="store_true", help="Include Coding benchmark (Humaneval+ & MBPP+)")
    
    # Grid sweep options
    parser.add_argument("--grid", action="store_true", help="Run in grid search sweep mode (ignores --desc)")
    parser.add_argument("--grid-kvs", type=str, default="q4_0,q4_1", help="Comma-separated KV cache options for grid search")
    parser.add_argument("--grid-max-tokens", type=str, default="1024", help="Comma-separated Max Token options for grid search")
    
    return parser.parse_args()

def get_git_commit() -> str:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode("utf-8").strip()
        status = subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.DEVNULL).decode("utf-8").strip()
        if status:
            commit += "-dirty"
        return commit
    except Exception:
        return "unknown"

def get_previous_best(results_file: Path) -> float:
    if not results_file.exists():
        return 0.0
    best_score = 0.0
    try:
        with open(results_file, "r") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                if row.get("status") == "keep":
                    try:
                        score = float(row.get("val_score", 0.0))
                        if score > best_score:
                            best_score = score
                    except ValueError:
                        pass
    except Exception as e:
        print(f"Error reading results.tsv: {e}")
    return best_score

def write_row(results_file: Path, commit: str, val_score: float, memory_gb: float, status: str, description: str):
    file_exists = results_file.exists() and results_file.stat().st_size > 0
    with open(results_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["commit", "val_score", "memory_gb", "status", "description"], delimiter="\t")
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "commit": commit,
            "val_score": f"{val_score:.6f}",
            "memory_gb": f"{memory_gb:.1f}",
            "status": status,
            "description": description
        })

def run_evaluation(args, model_filename: str, kv_cache: str, max_tokens: int, include_coding: bool) -> Dict[str, Any]:
    intent = ServerIntent(
        model_path=MODELS_DIR / model_filename,
        ctx_size=args.ctx_size,
        kv_cache=kv_cache,
        flash_attn="on",
        port=args.port,
        ngl=args.ngl,
        batch_size=512,
        threads=args.threads
    )
    
    server_log = BASE_DIR / "llama_server.log"
    if server_log.exists():
        try:
            server_log.unlink()
        except OSError:
            pass
            
    res = {
        "status": "OK",
        "nexus_val": 0.0, "nexus_tps": 0.0, "nexus_vram": 0.0,
        "claw_val": 0.0, "claw_tps": 0.0, "claw_vram": 0.0,
        "coding_val": 0.0, "coding_vram": 0.0,
        "val_score": 0.0, "avg_tps": 0.0, "peak_vram_gb": 0.0
    }
    
    try:
        with LlamaServerRunner(intent, log_path=server_log) as runner:
            client = LlamaClient(runner.port)
            system_prefix = "<|think|>\n"
            
            # 1. Nexus (Retrieval)
            print("  [nexus] Running...")
            nexus_res = run_nexus(client, max_tokens=max_tokens, system_prefix=system_prefix, context_tokens=args.context_tokens)
            res["nexus_val"] = nexus_res.val_score
            res["nexus_tps"] = nexus_res.avg_tps
            res["nexus_vram"] = runner.peak_vram_mb
            
            # 2. Claw (Agency)
            print("  [claw] Running...")
            claw_res = run_claw(client, max_tokens=max_tokens, system_prefix=system_prefix, context_tokens=args.context_tokens)
            res["claw_val"] = claw_res.val_score
            res["claw_tps"] = claw_res.avg_tps
            res["claw_vram"] = runner.peak_vram_mb
            
            # 3. Coding (optional)
            if include_coding:
                print("  [coding] Running (lite range)...")
                coding_res = run_coding(client, is_test=True, model_name=model_filename)
                res["coding_val"] = coding_res.val_score
                res["coding_vram"] = runner.peak_vram_mb
            
            # Compute combined metrics
            avg_tps = (nexus_res.avg_tps + claw_res.avg_tps) / 2.0
            res["avg_tps"] = avg_tps
            res["peak_vram_gb"] = max(runner.peak_vram_mb, 0.0) / 1024.0
            
            if avg_tps < 30.0:
                print(f"  [WARNING] Combined TPS {avg_tps:.2f} is below 30.0! Score set to 0.0.")
                res["val_score"] = 0.0
            else:
                res["val_score"] = (res["claw_val"] * 0.6) + (res["nexus_val"] * 0.4)
                
    except Exception as e:
        print(f"  [FAIL] Evaluation failed: {e}")
        res["status"] = f"FAIL: {str(e)[:50]}"
        res["val_score"] = 0.0
        
    return res

def handle_single_run(args):
    if not args.desc:
        print("Error: --desc is required for logging single runs. Example: --desc 'Tweak system prompt'")
        sys.exit(1)
        
    print(f"Starting single run for model: {args.model}")
    commit = get_git_commit()
    
    # Read previous best score
    prev_best = get_previous_best(RESULTS_FILE)
    print(f"Previous best 'keep' score: {prev_best:.6f}")
    
    # Run evaluation
    res = run_evaluation(args, args.model, args.kv, args.max_tokens, args.include_coding)
    
    if res["status"] != "OK":
        print(f"Evaluation failed: {res['status']}")
        write_row(RESULTS_FILE, commit, 0.0, res["peak_vram_gb"], "discard", f"FAIL: {res['status']} | {args.desc}")
        sys.exit(1)
        
    val_score = res["val_score"]
    improved = val_score > prev_best
    status = "keep" if improved else "discard"
    
    # Format details in description
    details = f"{args.model} kv={args.kv} ctx={args.ctx_size} TPS={res['avg_tps']:.1f} VRAM={res['peak_vram_gb']:.1f}GB retrieval={res['nexus_val']:.4f} agency={res['claw_val']:.4f}"
    if args.include_coding:
        details += f" coding={res['coding_val']:.4f}"
    details += f" | {args.desc}"
    
    # Log to results.tsv
    write_row(RESULTS_FILE, commit, val_score, res["peak_vram_gb"], status, details)
    
    print("\n" + "="*40)
    print("EVALUATION COMPLETE")
    print("="*40)
    print(f"Model:          {args.model}")
    print(f"KV Cache:       {args.kv}")
    print(f"Context Size:   {args.ctx_size}")
    print("-"*40)
    print(f"Nexus (Retrieval): {res['nexus_val']:.4f} (TPS: {res['nexus_tps']:.1f})")
    print(f"Claw (Agency):    {res['claw_val']:.4f} (TPS: {res['claw_tps']:.1f})")
    if args.include_coding:
        print(f"Coding Score:     {res['coding_val']:.4f}")
    print("-"*40)
    print(f"Combined TPS:     {res['avg_tps']:.1f} (Threshold: >= 30.0)")
    print(f"Peak VRAM:        {res['peak_vram_gb']:.1f} GB")
    print(f"Current Score:    {val_score:.6f}")
    print(f"Previous Best:    {prev_best:.6f}")
    print("-"*40)
    
    if improved:
        print(f"\n>>> STATUS: KEEP (Improved by +{val_score - prev_best:.6f})")
        print(">>> Run this to commit your tweak:")
        print(f"    git commit -am \"keep: {args.desc} (score: {val_score:.6f})\"")
    else:
        print(f"\n>>> STATUS: DISCARD (Regressed or no improvement by {val_score - prev_best:.6f})")
        print(">>> Run this to discard your tweak:")
        print("    git checkout . && git clean -fd")

def handle_grid_run(args):
    print("Starting grid sweep...")
    kvs = [k.strip() for k in args.grid_kvs.split(",") if k.strip()]
    max_tokens_list = [int(m.strip()) for m in args.grid_max_tokens.split(",") if m.strip()]
    
    commit = get_git_commit()
    
    for kv in kvs:
        for mt in max_tokens_list:
            print(f"\n{'='*80}")
            print(f"GRID sweep: KV={kv}, Max Tokens={mt}")
            print(f"{'='*80}")
            
            res = run_evaluation(args, args.model, kv, mt, args.include_coding)
            
            status = "keep"
            details = f"GRID Sweep: model={args.model} kv={kv} max_tokens={mt} ctx={args.ctx_size} TPS={res['avg_tps']:.1f} VRAM={res['peak_vram_gb']:.1f}GB retrieval={res['nexus_val']:.4f} agency={res['claw_val']:.4f}"
            if args.include_coding:
                details += f" coding={res['coding_val']:.4f}"
            
            write_row(RESULTS_FILE, commit, res["val_score"], res["peak_vram_gb"], status, details)
            print(f"Grid sweep entry logged: score={res['val_score']:.6f}")

def main():
    args = parse_args()
    if args.grid:
        handle_grid_run(args)
    else:
        handle_single_run(args)

if __name__ == "__main__":
    main()
