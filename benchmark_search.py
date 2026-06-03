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


import urllib.request

def wait_for_server(server_proc: subprocess.Popen[str], port: int, timeout: int = 300) -> bool:
    # ⚡ Bolt Optimization: Use urllib instead of spawning curl process.
    # Saves significant overhead (~0.1s to ~0.2s per iteration).
    deadline = time.time() + timeout
    while time.time() < deadline:
        if server_proc.poll() is not None:
            return False
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{port}/health")
            with urllib.request.urlopen(req, timeout=0.5) as response:
                if response.status == 200:
                    return True
        except Exception:
            time.sleep(0.1)
    return False


def candidate_ports(preferred: int) -> list[int]:
    return list(dict.fromkeys((preferred, preferred + 1, preferred + 2, 18080, 28080)))


def start_vram_sampler(stop_event: threading.Event, peak: dict[str, float]) -> threading.Thread:
    def sampler() -> None:
        while not stop_event.is_set():
            try:
                res = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits",
                        "-i",
                        "0",
                    ],
                    text=True,
                )
                current = float(res.strip() or 0.0)
                if current > peak["value"]:
                    peak["value"] = current
            except FileNotFoundError:
                print("Error: nvidia-smi not found. VRAM sampling stopped.")
                break
            except (subprocess.CalledProcessError, ValueError) as e:
                print(f"Warning: VRAM sampling error: {e}")
            stop_event.wait(0.2)

    thread = threading.Thread(target=sampler, daemon=True)
    thread.start()
    return thread


def read_server_log(path: str) -> str:
    return Path(path).read_text(encoding="utf-8", errors="replace")


def start_server(
    llama_server: Path,
    port: int,
    server_env: dict[str, str],
    ctx_size: int | None = None,
    flash_attn: str | None = None,
    kv_cache: str | None = None,
) -> tuple[subprocess.Popen[str], tempfile._TemporaryFileWrapper[str], list[str]]:
    _ctx = ctx_size if ctx_size is not None else CTX_SIZE
    _fa = flash_attn if flash_attn is not None else FLASH_ATTN
    _kv = kv_cache if kv_cache is not None else KV_CACHE_TYPE
    
    cmd = [
        str(llama_server),
        "--model",
        str(MODEL_FILE),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
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

    # 🚀 MTP Optimization: Detect MTP models and enable built-in speculative decoding.
    # Qwen 3.5 MTP models have dedicated prediction heads that can be used as a draft.
    if "MTP" in str(MODEL_FILE).upper():
        print(f"  [MTP] Multi-Token Prediction detected for {MODEL_FILE.name}. Enabling draft-mtp.")
        cmd += [
            "--spec-type", "draft-mtp",
            "--spec-draft-n-max", "1",
            "--spec-draft-type-k", _kv,
            "--spec-draft-type-v", _kv,
        ]

    # ⚗️ VITRIOL Optimization: Hardware Necromancy for large MoE models.
    # If the model is a large MoE (>= 10B total params) and we are on 8GB VRAM,
    # we offload expert weights to System RAM while keeping Attention/KV on GPU.
    moe_indicators = ["MOE", "A3B", "A4B", "A1B", "A2B", "8X3B", "8X4B", "10B", "11B", "12B", "13B", "14B", "15B", "16B", "17B", "18B", "19B", "20B", "21B", "22B", "23B", "24B", "25B", "26B", "35B"]
    model_name_up = str(MODEL_FILE).upper()
    is_moe = any(ind in model_name_up for ind in moe_indicators)
    
    # Precise exclusion: only skip if it's truly a small model (2B/4B/7B/8B/9B) AND not a multi-expert one.
    # We check for the presence of small numbers surrounded by dashes or boundaries.
    is_small_dense = any(f"-{x}B" in model_name_up for x in ["2", "4", "7", "8", "9"]) and not ("MOE" in model_name_up or "A1B" in model_name_up)

    if is_moe and not is_small_dense:
        print(f"  [VITRIOL] MoE Expert Streaming enabled for {MODEL_FILE.name}. Offloading experts to CPU.")
        # Standard VITRIOL pattern for llama.cpp GGUF
        cmd += ["--override-tensor", ".*exps.*=CPU"]

    server_log = tempfile.NamedTemporaryFile(
        mode="w+",
        encoding="utf-8",
        prefix="autoresearch-llama-server-",
        suffix=".log",
        delete=True,
    )
    server_proc = subprocess.Popen(
        cmd,
        stdout=server_log,
        stderr=subprocess.STDOUT,
        env=server_env,
        text=True,
    )
    return server_proc, server_log, cmd


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
    server_proc: subprocess.Popen[str] | None = None
    server_log: tempfile._TemporaryFileWrapper[str] | None = None
    port = PORT
    peak_vram = {"value": 0.0}
    stop_event = threading.Event()
    sampler = start_vram_sampler(stop_event, peak_vram)

    try:
        startup_tail: list[str] = []
        for candidate_port in candidate_ports(PORT):
            server_proc, server_log, cmd = start_server(llama_server, candidate_port, server_env, ctx_size=ctx_size, flash_attn=flash_attn, kv_cache=kv_cache)
            port = candidate_port
            print(f"Starting server: {' '.join(cmd)}")
            if wait_for_server(server_proc, port):
                break

            server_log.flush()
            startup_tail = read_server_log(server_log.name).splitlines()[-20:]
            bind_failed = any("couldn't bind http server socket" in line.lower() for line in startup_tail)

            if server_proc.poll() is None:
                server_proc.terminate()
                try:
                    server_proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    server_proc.kill()
                    server_proc.wait(timeout=5)
            server_log.close()
            server_proc = None
            server_log = None

            if not bind_failed:
                print("FAIL: Server crashed or failed to start.")
                if startup_tail:
                    print("SERVER LOG:")
                    for line in startup_tail:
                        print(line)
                return 1
        else:
            print("FAIL: Server could not bind any candidate port.")
            if startup_tail:
                print("SERVER LOG:")
                for line in startup_tail:
                    print(line)
            return 1

        assert server_log is not None
        server_log.flush()
        log_content = read_server_log(server_log.name)
        server_output = log_content.lower()
        if "no usable gpu found" in server_output or "gpu-layers option will be ignored" in server_output:
            print("FAIL: Server started without GPU support. GPU-only mode required.")
            return 1

        print("Server started. Running dual evaluation (Nexus Retrieval + ClawBench Agency)...")
        
        # 🧪 System Prefix Logic (Thinking Mode)
        system_prefix = ""
        if "GEMMA-4" in model_name.upper() or "THINKING" in model_name.upper():
            print(f"  [Thinking] Enabling Thinking Mode prefix (<|think|>) for {model_name}.")
            system_prefix = "<|think|>\n"

        # Pass 1: Retrieval Score (from original prepare.py)
        # Note: target_tokens=50000 matches user stress requirement
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
        
        # Composite Nexus-Claw Score (Equal weights for Memory and Agency)
        tokens_per_sec = round((tps_ret + tps_age) / 2, 2)

        # Apply strict minimum TPS constraint (30 TPS floor)
        # If the combined TPS is below 30, aggressively penalize the score to force a discard.
        if tokens_per_sec < 30.0:
            print(f"  [penalty] Throughput ({tokens_per_sec:.2f}) below 30 TPS floor. Score nulled.")
            val_score = 0.0
        else:
            # Composite Nexus-Claw Score
            # Priorities: Agency (60%) > Retrieval (40%) > Throughput (Gateway at 30 TPS)
            val_score = round(0.4 * val_retrieval + 0.6 * val_agency, 6)
        
        eval_seconds = time_ret + time_age
    finally:
        stop_event.set()
        sampler.join(timeout=1)
        if server_proc is not None and server_proc.poll() is None:
            server_proc.terminate()
            try:
                server_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server_proc.kill()
                server_proc.wait(timeout=5)
        if server_log is not None:
            server_log.close()

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
