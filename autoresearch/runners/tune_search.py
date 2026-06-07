#!/usr/bin/env python3
"""
Hill-Climbing Heuristic Auto-Tuner for local-model-autoresearch.
Optimizes KV cache, threads, batch sizes, and speculative draft tokens.
Uses pre-flight VRAM estimation to skip configurations expected to OOM.
"""

import os
# Set default llama.cpp root directory
os.environ.setdefault("AUTORESEARCH_LLAMA_CPP_ROOT", "/home/shark/workspace/Nexus-System/llama.cpp")

import sys
import argparse
import random
from pathlib import Path
from typing import Dict, Any

from autoresearch.core.llama_runner import ServerIntent, estimate_vram_mb
from autoresearch.runners.run import run_evaluation, get_git_commit, write_row, RESULTS_FILE

# Search Space definition
PARAMETER_SEARCH_SPACE = {
    "kv_cache_k": ["q4_0", "q8_0", "turbo2", "turbo3", "turbo4", "f16"],
    "kv_cache_v": ["q4_0", "q8_0", "turbo2", "turbo3", "turbo4", "f16"],
    "threads": [6, 8, 12, 16],
    "threads_batch": [None, 8, 12, 16, 24],
    "batch_size": [256, 512, 1024],
    "ubatch_size": [64, 128, 256, 512],
    "spec_draft_n_max": [0, 1, 2, 3, 4]
}

def get_neighbors(current_config: Dict[str, Any]) -> list[Dict[str, Any]]:
    neighbors = []
    for param, values in PARAMETER_SEARCH_SPACE.items():
        current_value = current_config[param]
        try:
            idx = values.index(current_value)
        except ValueError:
            idx = -1
        
        # Add neighbor with next value
        if idx < len(values) - 1:
            neighbor = current_config.copy()
            neighbor[param] = values[idx + 1]
            neighbors.append(neighbor)
            
        # Add neighbor with previous value
        if idx > 0:
            neighbor = current_config.copy()
            neighbor[param] = values[idx - 1]
            neighbors.append(neighbor)
            
    # Shuffle to avoid parameter bias
    random.shuffle(neighbors)
    return neighbors

def parse_args():
    parser = argparse.ArgumentParser(description="Heuristic Auto-Tuner")
    parser.add_argument("--model", type=str, default="g4-opt-it-Q4_K_M.gguf", help="Model filename in models/ directory")
    parser.add_argument("--ctx-size", "-c", type=int, default=16384, help="Context size")
    parser.add_argument("--port", type=int, default=18080, help="Port for llama-server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host for llama-server")
    parser.add_argument("--ngl", "--n-gpu-layers", "-ngl", type=int, default=99, help="Number of GPU layers to offload")
    parser.add_argument("--context-tokens", type=int, default=8192, help="Context tokens padding length")
    parser.add_argument("--max-steps", type=int, default=15, help="Maximum search steps")
    parser.add_argument("--vram-limit-mb", type=float, default=7900.0, help="Max safe VRAM limit in MB")
    parser.add_argument("--include-coding", action="store_true", help="Include coding benchmark")
    
    # Advanced parameters to pass down to evaluation
    parser.add_argument("--flash-attn", "-fa", nargs="?", const="on", default="on", choices=["on", "off", "auto"], help="Enable/disable/auto Flash Attention")
    parser.add_argument("--spec-type", type=str, default=None, help="Speculative decoding type (e.g. draft-mtp, mtp)")
    parser.add_argument("--no-mmap", action="store_true", help="Disable mmap")
    parser.add_argument("--jinja", action="store_true", help="Enable Jinja chat template engine")
    parser.add_argument("--reasoning-budget", type=int, default=None, help="Thinking budget tokens limit")
    parser.add_argument("--reasoning-budget-message", type=str, default=None, help="Message on thinking budget exhaust")
    parser.add_argument("--reasoning", type=str, choices=["on", "off", "auto"], default=None, help="Reasoning mode (on/off/auto)")
    parser.add_argument("--cont-batching", action="store_true", help="Enable continuous batching")
    parser.add_argument("--temp", type=float, default=0.2, help="Generation temperature")
    parser.add_argument("--top-p", type=float, default=None, help="Top-p sampling")
    parser.add_argument("--min-p", type=float, default=None, help="Min-p sampling")
    parser.add_argument("--top-k", type=int, default=None, help="Top-k sampling")
    parser.add_argument("--repeat-penalty", type=float, default=None, help="Repeat penalty")
    parser.add_argument("--presence-penalty", type=float, default=None, help="Presence penalty")
    parser.add_argument("--frequency-penalty", type=float, default=None, help="Frequency penalty")
    
    return parser.parse_args()

def main():
    args = parse_args()
    print(f"Starting Heuristic Auto-Tuner for model: {args.model}")
    print(f"VRAM Budget limit: {args.vram_limit_mb} MB")

    # Initial default baseline configuration
    current_config = {
        "kv_cache_k": "q4_0",
        "kv_cache_v": "q4_0",
        "threads": 12,
        "threads_batch": None,
        "batch_size": 512,
        "ubatch_size": 128,
        "spec_draft_n_max": 1
    }

    # Evaluate baseline
    print("\nEvaluating baseline configuration...")
    print(f"Baseline config: {current_config}")
    
    # Check pre-flight VRAM estimate
    est_vram = estimate_vram_mb(
        Path("models") / args.model, args.ctx_size,
        current_config["kv_cache_k"], current_config["kv_cache_v"], "q4_0"
    )
    print(f"Pre-flight baseline VRAM estimate: {est_vram:.1f} MB")
    
    if est_vram >= args.vram_limit_mb:
        print("[WARNING] Baseline VRAM estimate exceeds budget! Proceeding with caution.")

    baseline_res = run_evaluation(
        args, args.model, "q4_0", 1024, args.include_coding,
        kv_k=current_config["kv_cache_k"], kv_v=current_config["kv_cache_v"],
        threads=current_config["threads"], threads_batch=current_config["threads_batch"],
        batch_size=current_config["batch_size"], ubatch_size=current_config["ubatch_size"],
        spec_draft_n_max=current_config["spec_draft_n_max"]
    )
    
    current_score = baseline_res.get("val_score", 0.0)
    current_tps = baseline_res.get("avg_tps", 0.0)
    current_vram = baseline_res.get("peak_vram_gb", 0.0) * 1024.0
    
    print(f"Baseline Score: {current_score:.6f} | TPS: {current_tps:.2f} | VRAM: {current_vram:.1f} MB")
    
    commit = get_git_commit()
    write_row(
        RESULTS_FILE, commit, current_score, current_vram / 1024.0, "keep",
        f"Tuner Baseline: model={args.model} score={current_score:.6f} vram={current_vram/1024.0:.1f}GB"
    )

    visited_configs = {str(sorted(current_config.items()))}
    step = 0
    improved_runs = 0

    while step < args.max_steps:
        step += 1
        print(f"\n=== TUNING STEP {step}/{args.max_steps} ===")
        neighbors = get_neighbors(current_config)
        
        found_improvement = False
        for neighbor in neighbors:
            config_str = str(sorted(neighbor.items()))
            if config_str in visited_configs:
                continue
            visited_configs.add(config_str)
            
            # 1. Pre-flight VRAM estimation check
            est_vram = estimate_vram_mb(
                Path("models") / args.model, args.ctx_size,
                neighbor["kv_cache_k"], neighbor["kv_cache_v"], "q4_0"
            )
            if est_vram >= args.vram_limit_mb:
                print(f"Skipping neighbor (estimated VRAM {est_vram:.1f} MB exceeds budget {args.vram_limit_mb:.1f} MB): {neighbor}")
                continue
                
            print(f"Evaluating candidate: {neighbor} (est VRAM: {est_vram:.1f} MB)...")
            res = run_evaluation(
                args, args.model, "q4_0", 1024, args.include_coding,
                kv_k=neighbor["kv_cache_k"], kv_v=neighbor["kv_cache_v"],
                threads=neighbor["threads"], threads_batch=neighbor["threads_batch"],
                batch_size=neighbor["batch_size"], ubatch_size=neighbor["ubatch_size"],
                spec_draft_n_max=neighbor["spec_draft_n_max"]
            )
            
            score = res.get("val_score", 0.0)
            tps = res.get("avg_tps", 0.0)
            vram = res.get("peak_vram_gb", 0.0) * 1024.0
            
            print(f"Result - Score: {score:.6f} (baseline was {current_score:.6f}) | TPS: {tps:.2f} | VRAM: {vram:.1f} MB")
            
            # Log to results.tsv
            write_row(
                RESULTS_FILE, commit, score, vram / 1024.0, 
                "keep" if score > current_score else "discard",
                f"Tuner Heuristic Sweep: model={args.model} score={score:.6f} config={neighbor}"
            )
            
            # Check if strictly improved
            if score > current_score and vram < args.vram_limit_mb:
                print(f">>> IMPROVEMENT DETECTED! Score +{score - current_score:.6f}")
                current_score = score
                current_config = neighbor
                current_tps = tps
                current_vram = vram
                found_improvement = True
                improved_runs += 1
                break # Move to next search step from the new local optimum
                
        if not found_improvement:
            print("No improvements found in neighborhood. Search converged or budget constraints met.")
            break

    print(f"\n==========================================")
    print(f"AUTO-TUNING COMPLETE")
    print(f"==========================================")
    print(f"Best Config: {current_config}")
    print(f"Best Score:  {current_score:.6f}")
    print(f"Best TPS:    {current_tps:.2f}")
    print(f"Peak VRAM:   {current_vram / 1024.0:.1f} GB")
    print(f"Steps:       {step}")
    print(f"Improvements: {improved_runs}")

if __name__ == "__main__":
    main()
