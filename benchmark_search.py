"""
Autoresearch tuning surface.

This is the only file the agent should edit during the search loop.
prepare.py and program.md define the fixed harness and rules.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import sys

sys.path.append(str(Path(__file__).parent))
from prepare import evaluate_agentic_workflow as evaluate_retrieval  # noqa: E402
from prepare_claw import evaluate_claw_workflow as evaluate_agency  # noqa: E402


# ---------------------------------------------------------------------------
# Search surface: everything below is fair game for the autoresearch agent.
# ---------------------------------------------------------------------------

MODEL_NAME = "Qwen3.5-4B-Q4_K_M.gguf"  # default for single-run; overridden by MODELS_TO_BENCHMARK loop

MODELS_TO_BENCHMARK = [
    "gemma-4-E4B-it-Q4_K_M.gguf",
]

# Reasoning Effort Configs (Sacrificing speed for score)
REASONING_LEVELS = {
    "High": 1024,
    "XHigh": 2048
}

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
# Configs at or above this speed receive full throughput credit in val_score.
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
    # Only parse if not imported or specifically requested
    if len(sys.argv) > 1:
        parse_args()

# ---------------------------------------------------------------------------
# Fixed execution plumbing. Do not turn this into a second harness.
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
    raise FileNotFoundError(
        "llama-server not found. Expected one of: "
        + ", ".join(str(path) for path in LLAMA_SERVER_CANDIDATES)
    )


def build_llama_cmd(
    llama_server,
    port: int,
    ctx_size: int | None = None,
    flash_attn: str | None = None,
    kv_cache: str | None = None,
) -> list[str]:
    _ctx = ctx_size if ctx_size is not None else CTX_SIZE
    _fa = flash_attn if flash_attn is not None else FLASH_ATTN
    _kv = kv_cache if kv_cache is not None else KV_CACHE_TYPE
    
    cmd = [
        str(llama_server),
        "--model",
        str(MODEL_FILE),
        "--host",
        "127.0.0.1",
        "--ctx-size",
        str(_ctx),
        "--batch-size",
        str(BATCH_SIZE),
        "--ubatch-size",
        str(UBATCH_SIZE),
        "--threads",
        str(THREADS),
        "--parallel",
        str(PARALLEL),
        "--n-gpu-layers",
        str(NGL),
        "--cache-type-k",
        _kv,
        "--cache-type-v",
        _kv,
        "--flash-attn",
        _fa,
    ]

    # MTP Optimization: Detect MTP models and enable built-in speculative decoding.
    if "MTP" in str(MODEL_FILE).upper():
        print(f"  [MTP] Multi-Token Prediction detected for {MODEL_FILE.name}. Enabling draft-mtp.")
        cmd += [
            "--spec-type", "draft-mtp",
            "--spec-draft-n-max", "1",
            "--spec-draft-type-k", _kv,
            "--spec-draft-type-v", _kv,
        ]

    # VITRIOL Optimization: Hardware Necromancy for large MoE models.
    moe_indicators = ["MOE", "A3B", "A4B", "A1B", "A2B", "8X3B", "8X4B", "10B", "11B", "12B", "13B", "14B", "15B", "16B", "17B", "18B", "19B", "20B", "21B", "22B", "23B", "24B", "25B", "26B", "35B"]
    model_name_up = str(MODEL_FILE).upper()
    is_moe = any(ind in model_name_up for ind in moe_indicators)
    is_small_dense = any(f"-{x}B" in model_name_up for x in ["2", "4", "7", "8", "9"]) and not ("MOE" in model_name_up or "A1B" in model_name_up)

    if is_moe and not is_small_dense:
        print(f"  [VITRIOL] MoE Expert Streaming enabled for {MODEL_FILE.name}. Offloading experts to CPU.")
        cmd += ["--override-tensor", ".*exps.*=CPU"]

    return cmd

def main(model_name: str = MODEL_NAME, ctx_size: int | None = None, flash_attn: str | None = None, kv_cache: str | None = None) -> int:
    global MODEL_FILE
    MODEL_FILE = MODELS_DIR / model_name
    llama_server = resolve_llama_server()
    if not MODEL_FILE.exists():
        print(f"FAIL: Model {model_name} not found at {MODEL_FILE}")
        return 1

    server_env = os.environ.copy()
    llama_lib_dir = str(llama_server.parent)
    existing = server_env.get("LD_LIBRARY_PATH", "")
    server_env["LD_LIBRARY_PATH"] = f"{llama_lib_dir}:{existing}" if existing else llama_lib_dir

    t_start = time.time()
    
    cmd = build_llama_cmd(llama_server, PORT, ctx_size=ctx_size, flash_attn=flash_attn, kv_cache=kv_cache)
    
    from llama_runner import LlamaServerRunner
    
    try:
        with LlamaServerRunner(cmd, server_env, PORT) as runner:
            port = runner.port
            peak_vram = {"value": runner.peak_vram_mb}
            print("Server started. Running evaluation...")

            # System Prefix Logic (Thinking Mode)
            system_prefix = ""
            if "GEMMA-4" in model_name.upper() or "THINKING" in model_name.upper():
                print(f"  [Thinking] Enabling Thinking Mode prefix (< | t h i n k | >) for {model_name}.")
                system_prefix = "<|think|>\n"

            # Pass 1: Retrieval Score (from original prepare.py)
            val_retrieval, p1_ret, p2_ret, tps_ret, time_ret = evaluate_retrieval(
                model_name=model_name,
                port=port,
                temp=TEMP,
                top_p=TOP_P,
                top_k=TOP_K,
                min_p=MIN_P,
                presence_penalty=PRES_PENALTY,
                frequency_penalty=FREQ_PENALTY,
                repeat_penalty=REP_PENALTY,
                maxtok=MAXTOK,
                target_tps=TARGET_TPS,
                context_target_tokens=50000,
                system_prefix=system_prefix,
            )
            
            # Pass 2: Agency Score (from prepare_claw.py)
            val_agency, p1_age, p2_age, tps_age, time_age = evaluate_agency(
                model_name=model_name,
                port=port,
                temp=TEMP,
                target_tps=TARGET_TPS,
                context_target_tokens=0,
                system_prefix=system_prefix,
            )
            
            tokens_per_sec = round((tps_ret + tps_age) / 2, 2)
            if tokens_per_sec < 30.0:
                print(f"  [penalty] Throughput ({tokens_per_sec:.2f}) below 30 TPS floor. Score nulled.")
                val_score = 0.0
            else:
                val_score = round(0.4 * val_retrieval + 0.6 * val_agency, 6)
            
            eval_seconds = time_ret + time_age
            peak_vram["value"] = runner.peak_vram_mb
            
    except RuntimeError as e:
        print(f"FAIL: {e}")
        return 1
        
    total_seconds = time.time() - t_start

    print("---")
    print(f"val_score:        {val_score:.6f}")
    print(f"val_retrieval:    {val_retrieval:.6f}")
    print(f"val_agency:       {val_agency:.6f}")
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
        # Force High and XHigh for this study
        levels = ["High", "XHigh"]
        
        for level in levels:
            # If MAXTOK was set via CLI, only run that specific one?
            # No, usually grid search handles the loop.
            # But here we want benchmark_search.py to be a single-shot tool for run_grid.py
            current_maxtok = REASONING_LEVELS.get(level)
            if current_maxtok is None:
                continue
            
            # If we are running via CLI with specific tokens, skip others
            if len(sys.argv) > 1 and MAXTOK != 512 and current_maxtok != MAXTOK:
                 continue

            print(f"\n{'='*60}")
            print(f"Benchmarking: {model}")
            print(f"Effort Level: {level} (maxtok={current_maxtok})")
            print(f"{'='*60}")
            
            # Use CTX_SIZE from globals (which might be updated by parse_args)
            current_ctx = CTX_SIZE
            
            try:
                # Patch MAXTOK globally for this run
                old_maxtok = MAXTOK
                MAXTOK = current_maxtok
                
                rc = main(model, ctx_size=current_ctx, kv_cache=KV_CACHE_TYPE)
                
                MAXTOK = old_maxtok # Restore
                
                if rc == 0:
                    results.append({"model": model, "level": level, "status": "OK", "ctx": current_ctx})
                else:
                    results.append({"model": model, "level": level, "status": "FAIL", "ctx": current_ctx})
            except Exception as exc:
                print(f"ERROR: {exc}")
                results.append({"model": model, "level": level, "status": f"ERROR: {exc}"})

    # Print summary table
    print(f"\n{'='*110}")
    print("BENCHMARK RESULTS (Reasoning Effort Study)")
    print(f"{'='*110}")
    header = f"{'Model':<45} {'Level':<10} {'Status':<8} {'Ctx':>8}"
    print(header)
    print("-" * 110)
    for r in results:
        print(f"{r['model']:<45} {r['level']:<10} {r['status']:<8} {r.get('ctx', 0):>8}")
    print(f"{'='*110}")


RETEST_MODELS: list[dict] = [
    {"model": "Phi-4-mini-instruct-Q4_K_M.gguf", "flash_attn": "off", "ctx_size": 8192, "kv_cache": "f16", "note": "flash_attn=off + kv=f16 + ctx=8192"},
]


def retest_failed_models() -> None:
    """Retest models that failed with adjusted configs."""
    results = []
    for cfg in RETEST_MODELS:
        model = cfg["model"]
        note = cfg["note"]
        ctx = cfg["ctx_size"]
        fa = cfg["flash_attn"]
        print(f"\n{'='*60}")
        print(f"Retesting: {model}")
        print(f"Config:    {note}")
        print(f"{'='*60}")
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        try:
            with redirect_stdout(buf):
                rc = main(model, ctx_size=ctx, flash_attn=fa, kv_cache=cfg.get("kv_cache"))
            output = buf.getvalue()
            print(output, end="")
            if rc == 0:
                metrics: dict = {}
                for line in output.splitlines():
                    for key in ("val_score", "val_pass1", "val_pass2", "tokens_per_sec", "peak_vram_mb"):
                        if line.startswith(key + ":"):
                            try:
                                metrics[key] = float(line.split(":", 1)[1].strip())
                            except ValueError:
                                pass
                results.append({"model": model, "note": note, "status": "OK", **metrics})
            else:
                results.append({"model": model, "note": note, "status": "FAIL"})
        except Exception as exc:
            print(buf.getvalue(), end="")
            print(f"ERROR: {exc}")
            results.append({"model": model, "note": note, "status": f"ERROR: {exc}"})

    print(f"\n{'='*90}")
    print("RETEST RESULTS")
    print(f"{'='*90}")
    header = f"{'Model':<45} {'Status':<8} {'Score':>8} {'Pass1':>8} {'Pass2':>8} {'TPS':>8} {'VRAM(MB)':>10}"
    print(header)
    print("-" * 90)
    for r in results:
        if r["status"] == "OK":
            print(
                f"{r['model']:<45} {'OK':<8}"
                f" {r.get('val_score', 0):>8.4f}"
                f" {r.get('val_pass1', 0):>8.4f}"
                f" {r.get('val_pass2', 0):>8.4f}"
                f" {r.get('tokens_per_sec', 0):>8.1f}"
                f" {r.get('peak_vram_mb', 0):>10.0f}"
            )
            print(f"  note: {r['note']}")
        else:
            print(f"{r['model']:<45} {r['status']:<8}")
            print(f"  note: {r['note']}")
    print(f"{'='*90}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
        # Run specific model
        main(sys.argv[1])
    elif "--retest" in sys.argv:
        retest_failed_models()
    else:
        run_all_models()
