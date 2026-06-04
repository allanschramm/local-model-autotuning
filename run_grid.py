#!/usr/bin/env python3
import os
os.environ["AUTORESEARCH_LLAMA_CPP_ROOT"] = "/home/shark/workspace/Nexus-System/llama.cpp"
import csv
import time
from pathlib import Path
from typing import Dict, Any

from llama_runner import LlamaServerRunner, ServerIntent
from llama_client import LlamaClient
from benchmark_harness import BenchmarkResult

# Benchmarks
from benchmark_coding import run_benchmark as run_coding
from prepare import run_benchmark as run_nexus
from prepare_claw import run_benchmark as run_claw

# Configuration
KV_CACHES = ["q4_0", "q4_1"]
MAX_TOKENS_LIST = [1024]
MODEL = "g4-opt-it-Q4_K_M.gguf"
CTX_SIZE = 16384
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
                flash_attn="on",
                port=PORT,
                ngl=99,
                batch_size=512,
                threads=12
            )
            
            # Use a log file for the server to debug
            server_log = BASE_DIR / "llama_server.log"
            if server_log.exists(): server_log.unlink()
            
            row = {"kv_cache": kv, "max_tokens": mt, "status": "OK"}
            
            try:
                # Orchestrator manages the server lifecycle once per config
                with LlamaServerRunner(intent, log_path=server_log) as runner:
                    client = LlamaClient(runner.port)
                    
                    system_prefix = "<|think|>\n"
                    context_tokens = 8192
                        
                    # 1. Nexus
                    print("  [nexus] Running...")
                    nexus_res = run_nexus(client, max_tokens=mt, system_prefix=system_prefix, context_tokens=context_tokens)
                    row.update({
                        "nexus_val_score": nexus_res.val_score,
                        "nexus_pass1": nexus_res.val_pass1,
                        "nexus_pass2": nexus_res.val_pass2,
                        "nexus_tps": nexus_res.avg_tps,
                        "nexus_vram": runner.peak_vram_mb
                    })
                    
                    # 2. Claw
                    print("  [claw] Running...")
                    claw_res = run_claw(client, max_tokens=mt, system_prefix=system_prefix, context_tokens=context_tokens)
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