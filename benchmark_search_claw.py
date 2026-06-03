"""
Autoresearch tuning surface (ClawBench focus).

This is the only file the agent should edit during the search loop.
prepare.py and program.md define the fixed harness and rules.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

from llama_client import LlamaClient
from benchmark_harness import BenchmarkHarness, BenchmarkResult
from prepare import NexusEvalTask, prepare_eval_data, build_context_padding as build_retrieval_padding
from prepare_claw import ClawEvalTask, discover_tasks as discover_claw_tasks


# ---------------------------------------------------------------------------
# Search surface: everything below is fair game for the autoresearch agent.
# ---------------------------------------------------------------------------

MODEL_NAME = "Qwen3.5-4B-Q4_K_M.gguf"  # default for single-run

MODELS_TO_BENCHMARK = [
    "Qwen3.5-9B-MTP-Q4_K_M.gguf",
    "Qwopus3.5-9B-Coder-MTP-Q4_K_M.gguf",
]

CTX_SIZE = 131072
KV_CACHE_TYPE = "q4_0"
BATCH_SIZE = 512
UBATCH_SIZE = 128
THREADS = 8
PARALLEL = 1
NGL = 999
FLASH_ATTN = "on"  # on|off|auto

TEMP = 0.2
TOP_P = 0.9
TOP_K = 40
MIN_P = 0.05
FREQ_PENALTY = 0.0
PRES_PENALTY = 0.0
REP_PENALTY = 1.1
MAXTOK = 512
MEMORY_LIMIT = 1
PORT = 18080

# Throughput target for Pass 2 scoring (tokens/sec at short context).
TARGET_TPS = 30.0

import argparse

def parse_args():
    global CTX_SIZE, KV_CACHE_TYPE, MAXTOK, MODELS_TO_BENCHMARK
    parser = argparse.ArgumentParser()
    parser.add_argument("--ctx-size", type=int, default=CTX_SIZE)
    parser.add_argument("--kv-cache", type=str, default=KV_CACHE_TYPE)
    parser.add_argument("--max-tokens", type=int, default=MAXTOK)
    parser.add_argument("--model", type=str)
    args = parser.parse_args()
    
    CTX_SIZE = args.ctx_size
    KV_CACHE_TYPE = args.kv_cache
    MAXTOK = args.max_tokens
    if args.model:
        MODELS_TO_BENCHMARK = [args.model]

if __name__ == "__main__" or "run_grid" in sys.argv[0]:
    if len(sys.argv) > 1:
        parse_args()

# ---------------------------------------------------------------------------
# Fixed execution plumbing. Do not turn this into a second harness.
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent
# Claw script often uses a different models subfolder
MODELS_DIR = ROOT_DIR / "llm" / "gguf"

def run_benchmark(client: LlamaClient, **kwargs) -> BenchmarkResult:
    """Unified entry point for in-process orchestration (Claw emphasis)."""
    system_prefix = kwargs.get("system_prefix", "")
    
    # Pass 1: Retrieval Score
    retrieval_entries = prepare_eval_data()
    claw_tasks_data = discover_claw_tasks()
    tasks = [NexusEvalTask(retrieval_entries)] + [ClawEvalTask(d) for d in claw_tasks_data]
    padding = build_retrieval_padding(50000)

    # Standardized Evaluation Harness
    harness = BenchmarkHarness(client, target_tps=kwargs.get("target_tps", TARGET_TPS))
    
    return harness.evaluate(
        tasks,
        context_padding=padding,
        system_prefix=system_prefix,
        temp=kwargs.get("temp", TEMP),
        maxtok=kwargs.get("maxtok", MAXTOK)
    )

def main(model_name: str = MODEL_NAME, ctx_size: int | None = None, flash_attn: str | None = None, kv_cache: str | None = None) -> int:
    global MODEL_FILE
    MODEL_FILE = MODELS_DIR / model_name
    if not MODEL_FILE.exists():
        print(f"FAIL: Model {model_name} not found at {MODEL_FILE}")
        return 1

    t_start = time.time()
    
    from llama_runner import LlamaServerRunner, ServerIntent
    
    intent = ServerIntent(
        model_path=MODEL_FILE,
        ctx_size=ctx_size if ctx_size is not None else CTX_SIZE,
        kv_cache=kv_cache if kv_cache is not None else KV_CACHE_TYPE,
        flash_attn=flash_attn if flash_attn is not None else FLASH_ATTN,
        port=PORT,
        batch_size=BATCH_SIZE,
        ubatch_size=UBATCH_SIZE,
        threads=THREADS,
        parallel=PARALLEL,
        ngl=NGL
    )
    
    try:
        with LlamaServerRunner(intent) as runner:
            print("Server started. Running evaluation...")
            client = LlamaClient(runner.port)

            # System Prefix Logic (Thinking Mode)
            system_prefix = ""
            if "GEMMA-4" in model_name.upper() or "THINKING" in model_name.upper():
                print(f"  [Thinking] Enabling Thinking Mode prefix (< | t h i n k | >) for {model_name}.")
                system_prefix = "<|think|>\\n"

            result = run_benchmark(
                client, 
                system_prefix=system_prefix,
                maxtok=MAXTOK
            )
            
            val_score = result.val_score
            tokens_per_sec = result.avg_tps
            eval_seconds = result.total_seconds
            peak_vram_val = runner.peak_vram_mb
            
    except RuntimeError as e:
        print(f"FAIL: {e}")
        return 1
        
    total_seconds = time.time() - t_start

    print("---")
    print(f"val_score:        {val_score:.6f}")
    print(f"val_pass1:        {result.val_pass1:.6f}")
    print(f"val_pass2:        {result.val_pass2:.6f}")
    print(f"tokens_per_sec:   {tokens_per_sec:.2f}")
    print(f"total_seconds:    {total_seconds:.1f}")
    print(f"peak_vram_mb:     {peak_vram_val:.1f}")
    print(f"ctx_size:         {CTX_SIZE}")
    print(f"kv_cache:         {KV_CACHE_TYPE}")
    print(f"model:            {model_name}")
    print(f"eval_seconds:     {eval_seconds:.3f}")
    return 0

    print("---")
    print(f"val_score:        {val_score:.6f}")
    print(f"val_pass1:        {result.val_pass1:.6f}")
    print(f"val_pass2:        {result.val_pass2:.6f}")
    print(f"tokens_per_sec:   {tokens_per_sec:.2f}")
    print(f"total_seconds:    {total_seconds:.1f}")
    print(f"peak_vram_mb:     {peak_vram['value']:.1f}")
    print(f"ctx_size:         {CTX_SIZE}")
    print(f"kv_cache:         {KV_CACHE_TYPE}")
    print(f"model:            {model_name}")
    print(f"eval_seconds:     {eval_seconds:.3f}")
    return 0


def run_all_models():
    """Run benchmark on all models and print a summary table."""
    global MAXTOK, CTX_SIZE, KV_CACHE_TYPE
    results = []
    for model in MODELS_TO_BENCHMARK:
        levels = ["High", "XHigh"]
        REASONING_LEVELS = {"High": 1024, "XHigh": 2048}

        for level in levels:
            current_maxtok = REASONING_LEVELS.get(level)
            if current_maxtok is None: continue
            if len(sys.argv) > 1 and MAXTOK != 512 and current_maxtok != MAXTOK: continue

            print(f"\n{'='*60}")


            print(f"Benchmarking: {model}")
            print(f"Effort Level: {level} (maxtok={current_maxtok})")
            print(f"\n{'='*60}")


            current_ctx = CTX_SIZE
            import io
            from contextlib import redirect_stdout
            buf = io.StringIO()
            try:
                old_maxtok = MAXTOK
                MAXTOK = current_maxtok
                with redirect_stdout(buf):
                    rc = main(model, ctx_size=current_ctx, kv_cache=KV_CACHE_TYPE)
                MAXTOK = old_maxtok
                
                output = buf.getvalue()
                print(output, end="")
                if rc == 0:
                    metrics = {}
                    for line in output.splitlines():
                        for key in ("val_score", "tokens_per_sec", "peak_vram_mb"):
                            if line.startswith(key + ":"):
                                try: metrics[key] = float(line.split(":", 1)[1].strip())
                                except: pass
                    results.append({"model": model, "level": level, "status": "OK", "ctx": current_ctx, **metrics})
                else:
                    results.append({"model": model, "level": level, "status": "FAIL", "ctx": current_ctx})
            except Exception as exc:
                print(buf.getvalue(), end="")
                print(f"ERROR: {exc}")
                results.append({"model": model, "level": level, "status": f"ERROR: {exc}"})

    print(f"\n{'='*110}")

    print("BENCHMARK RESULTS (Fixed 128k Context)")
    print(f"{'='*110}")
    header = f"{'Model':<45} {'Level':<10} {'Status':<8} {'Ctx':>8} {'Score':>8} {'TPS':>8} {'VRAM(MB)':>10}"
    print(header)
    print("-" * 110)
    for r in results:
        if r["status"] == "OK":
            print(f"{r['model']:<45} {r['level']:<10} {'OK':<8} {r.get('ctx', 0):>8} {r.get('val_score', 0):>8.4f} {r.get('tokens_per_sec', 0):>8.1f} {r.get('peak_vram_mb', 0):>10.0f}")
        else:
            print(f"{r['model']:<45} {r['level']:<10} {r['status']:<8} {r.get('ctx', 0):>8}")
    print(f"{'='*110}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
        main(sys.argv[1])
    else:
        run_all_models()