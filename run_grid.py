#!/usr/bin/env python3
import csv
import os
import time
from pathlib import Path
from typing import Dict, Any

from llama_runner import LlamaServerRunner, ServerIntent
from llama_client import LlamaClient
from benchmark_harness import BenchmarkResult

# Benchmarks
from benchmark_search import run_benchmark as run_nexus
from benchmark_search_claw import run_benchmark as run_claw
from benchmark_coding import run_benchmark as run_coding

# Configuration
KV_CACHES = ["q4_0", "q4_1", "q5_0", "q5_1", "q8_0"]
MAX_TOKENS_LIST = [512, 1024]
MODEL = "gemma-4-E4B-it-Q4_K_M.gguf"
CTX_SIZE = 131072
PORT = 18080

BASE_DIR = Path(__file__).resolve().parent
RESULTS_FILE = BASE_DIR / "grid_results.csv"
MODELS_DIR = BASE_DIR / "models"

def main():
    headers = [
        "kv_cache", "max_tokens", "status",
        "nexus_val_score", "nexus_pass1", "nexus_pass2", "nexus_tps", "nexus_vram",
        "claw_val_score", "claw_pass1", "claw_pass2", "claw_tps", "claw_vram",
        "humaneval_plus", "mbpp_plus", "coding_vram"
    ]
    
    if not RESULTS_FILE.exists():
        with open(RESULTS_FILE, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()

    for kv in KV_CACHES:
        for mt in MAX_TOKENS_LIST:
            print(f"\n{'='*80}")
            print(f"CONFIG: KV={kv}, MAX_TOKENS={mt}")
            print(f"{'='*80}")
            
            intent = ServerIntent(
                model_path=MODELS_DIR / MODEL,
                ctx_size=CTX_SIZE,
                kv_cache=kv,
                port=PORT,
                ngl=999
            )
            
            row = {"kv_cache": kv, "max_tokens": mt, "status": "OK"}
            
            try:
                # Orchestrator manages the server lifecycle once per config
                with LlamaServerRunner(intent) as runner:
                    client = LlamaClient(runner.port)
                    
                    system_prefix = ""
                    if "GEMMA-4" in MODEL.upper() or "THINKING" in MODEL.upper():
                        system_prefix = "<|think|>\\n"
                        
                    retrieval_entries = prepare_eval_data()
                    claw_tasks_data = discover_claw_tasks()
                    padding = build_retrieval_padding(50000)
                    
                    # 1. Nexus
                    print("  [nexus] Running...")
                    nexus_harness = BenchmarkHarness(client, target_tps=30.0)
                    nexus_res = nexus_harness.evaluate(
                        [NexusEvalTask(retrieval_entries)], 
                        context_padding=padding, 
                        system_prefix=system_prefix,
                        temp=0.2,
                        maxtok=mt
                    )
                    row.update({
                        "nexus_val_score": nexus_res.val_score,
                        "nexus_pass1": nexus_res.val_pass1,
                        "nexus_pass2": nexus_res.val_pass2,
                        "nexus_tps": nexus_res.avg_tps,
                        "nexus_vram": runner.peak_vram_mb
                    })
                    
                    # 2. Claw
                    print("  [claw] Running...")
                    claw_harness = BenchmarkHarness(client, target_tps=30.0)
                    claw_res = claw_harness.evaluate(
                        [ClawEvalTask(d) for d in claw_tasks_data], 
                        context_padding=padding, 
                        system_prefix=system_prefix,
                        temp=0.2,
                        maxtok=mt
                    )
                    row.update({
                        "claw_val_score": claw_res.val_score,
                        "claw_pass1": claw_res.val_pass1,
                        "claw_pass2": claw_res.val_pass2,
                        "claw_tps": claw_res.avg_tps,
                        "claw_vram": runner.peak_vram_mb
                    })
                    
                    # 3. Coding (In-process, but still uses subprocess for evalplus)
                    print("  [coding] Running (lite range)...")
                    coding_res = run_coding(client, is_test=True, model_name=MODEL)
                    row.update({
                        "humaneval_plus": coding_res.val_pass1,
                        "mbpp_plus": coding_res.val_pass2,
                        "coding_vram": runner.peak_vram_mb
                    })
                    
            except Exception as e:
                print(f"  [FAIL] Config {kv}/{mt} failed: {e}")
                row["status"] = f"FAIL: {str(e)[:50]}"
            
            # Save row
            with open(RESULTS_FILE, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writerow(row)
            
            print(f"\nCompleted configuration: KV={kv}, MT={mt}")

if __name__ == "__main__":
    main()