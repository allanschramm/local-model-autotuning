import sys
import csv
import time
import argparse
import subprocess
import itertools
from pathlib import Path
from typing import Dict, Any

from autoresearch.core.llama_runner import LlamaServerRunner, ServerIntent
from autoresearch.core.llama_client import LlamaClient
from autoresearch.benchmarks.benchmark_harness import BenchmarkResult
from autoresearch.core import config

from autoresearch.benchmarks.benchmark_coding import run_benchmark as run_coding

BASE_DIR = Path(__file__).resolve().parent
RESULTS_FILE = BASE_DIR.parent.parent / "results.tsv"
MODELS_DIR = BASE_DIR.parent.parent / "models"

def parse_args():
    parser = argparse.ArgumentParser(description="Unified AutoResearch Benchmark Runner")
    parser.add_argument("--desc", type=str, help="Description of the experiment (required for logging single runs)")
    parser.add_argument("--model", type=str, default=config.MODEL, help="Model filename in models/ directory")
    parser.add_argument("--kv", type=str, default=config.KV_CACHE, help="KV cache type (e.g. q4_0, q4_1, f16)")
    parser.add_argument("--kv-k", "--cache-type-k", "-ctk", dest="kv_k", type=str, default=None, help="Key cache type (overrides --kv if set)")
    parser.add_argument("--kv-v", "--cache-type-v", "-ctv", dest="kv_v", type=str, default=None, help="Value cache type (overrides --kv if set)")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Max generation tokens")
    parser.add_argument("--ctx-size", "-c", type=int, default=config.CTX_SIZE, help="Context size")
    parser.add_argument("--port", type=int, default=18080, help="Port for llama-server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host for llama-server")
    parser.add_argument("--threads", "-t", type=int, default=config.THREADS, help="Number of threads for llama-server")
    parser.add_argument("--threads-batch", type=int, default=config.THREADS_BATCH, help="Number of batch/prefill threads for llama-server")
    parser.add_argument("--ngl", "--n-gpu-layers", "-ngl", type=int, default=99, help="Number of GPU layers to offload")
    parser.add_argument("--n-cpu-moe", "-ncmoe", type=int, default=getattr(config, 'N_CPU_MOE', None), help="Keep MoE expert weights of first N layers on CPU (VITRIOL)")
    parser.add_argument("--batch-size", "-b", type=int, default=config.BATCH_SIZE, help="Batch size for llama-server")
    parser.add_argument("--ubatch-size", "-ub", type=int, default=config.UBATCH_SIZE, help="Micro-batch size for llama-server")
    parser.add_argument("--parallel", type=int, default=1, help="Parallel slots count")
    parser.add_argument("--flash-attn", "-fa", nargs="?", const="on", default=config.FLASH_ATTN, choices=["on", "off", "auto"], help="Enable/disable/auto Flash Attention")
    parser.add_argument("--spec-type", type=str, default=config.SPEC_TYPE, help="Speculative decoding type (e.g. draft-mtp, mtp)")
    parser.add_argument("--spec-draft-n-max", type=int, default=config.SPEC_DRAFT_N_MAX, help="Speculative draft max tokens count for MTP")
    parser.add_argument("--context-tokens", type=int, default=8192, help="Context tokens padding length")
    parser.add_argument("--include-coding", action="store_true", default=True, help="Include Coding benchmark (Humaneval+ & MBPP+)")
    parser.add_argument("--no-coding", dest="include_coding", action="store_false", help="Disable Coding benchmark")
    parser.add_argument("--include-nexus", action="store_true", default=getattr(config, "INCLUDE_NEXUS", False), help="Include Nexus benchmark")
    parser.add_argument("--include-claw", action="store_true", default=getattr(config, "INCLUDE_CLAW", False), help="Include Claw benchmark")
    parser.add_argument("--coding-task-limit", type=int, default=getattr(config, "CODING_TASK_LIMIT", 30), help="Tasks per dataset (0=full dataset)")
    parser.add_argument("--no-mmap", action="store_true", default=config.NO_MMAP, help="Disable mmap")
    parser.add_argument("--jinja", action="store_true", default=config.JINJA, help="Enable Jinja chat template engine")
    parser.add_argument("--reasoning-budget", type=int, default=config.REASONING_BUDGET, help="Thinking budget tokens limit")
    parser.add_argument("--reasoning-budget-message", type=str, default=config.REASONING_BUDGET_MESSAGE, help="Message on thinking budget exhaust")
    parser.add_argument("--reasoning", type=str, choices=["on", "off", "auto"], default=config.REASONING, help="Reasoning mode (on/off/auto)")
    parser.add_argument("--cont-batching", action="store_true", default=config.CONT_BATCHING, help="Enable continuous batching")
    parser.add_argument("--temp", type=float, default=config.TEMP, help="Generation temperature")
    parser.add_argument("--top-p", type=float, default=config.TOP_P, help="Top-p sampling")
    parser.add_argument("--min-p", type=float, default=config.MIN_P, help="Min-p sampling")
    parser.add_argument("--top-k", type=int, default=config.TOP_K, help="Top-k sampling")
    parser.add_argument("--repeat-penalty", type=float, default=config.REPEAT_PENALTY, help="Repeat penalty")
    parser.add_argument("--presence-penalty", type=float, default=config.PRESENCE_PENALTY, help="Presence penalty")
    parser.add_argument("--frequency-penalty", type=float, default=config.FREQUENCY_PENALTY, help="Frequency penalty")
    
    # Grid sweep options
    parser.add_argument("--grid", action="store_true", help="Run in grid search sweep mode (ignores --desc)")
    parser.add_argument("--grid-kvs", type=str, default=None, help="Comma-separated KV cache options (sweeps both K & V)")
    parser.add_argument("--grid-kvs-k", type=str, default=None, help="Comma-separated Key cache options (overrides K)")
    parser.add_argument("--grid-kvs-v", type=str, default=None, help="Comma-separated Value cache options (overrides V)")
    parser.add_argument("--grid-max-tokens", type=str, default="1024", help="Comma-separated Max Token options for grid search")
    parser.add_argument("--grid-threads", type=str, default=None, help="Comma-separated Thread count options")
    parser.add_argument("--grid-threads-batch", type=str, default=None, help="Comma-separated batch thread count options")
    parser.add_argument("--grid-batch-sizes", type=str, default=None, help="Comma-separated batch size options")
    parser.add_argument("--grid-ubatch-sizes", type=str, default=None, help="Comma-separated ubatch size options")
    parser.add_argument("--grid-spec-draft-n-max", type=str, default=None, help="Comma-separated speculative draft max tokens options")
    
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

def write_row(results_file: Path, commit: str, val_score: float, swe_score: float, he_score: float, mbpp_score: float, memory_gb: float, status: str, description: str, lcb_score: float = 0.0, bigcode_score: float = 0.0):
    fieldnames = ["commit", "val_score", "swe_score", "lcb_score", "he_score", "mbpp_score", "bigcode_score", "memory_gb", "status", "description"]
    file_exists = results_file.exists() and results_file.stat().st_size > 0
    needs_header = True
    if file_exists:
        # Check if header already has the new columns; if not, rewrite the file
        with open(results_file, "r", newline="") as f:
            existing_header = f.readline().strip().split("\t")
        if existing_header == fieldnames:
            needs_header = False
    with open(results_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        if needs_header:
            writer.writeheader()
        writer.writerow({
            "commit": commit,
            "val_score": f"{val_score:.6f}",
            "swe_score": f"{swe_score:.6f}",
            "lcb_score": f"{lcb_score:.6f}",
            "he_score": f"{he_score:.6f}",
            "mbpp_score": f"{mbpp_score:.6f}",
            "bigcode_score": f"{bigcode_score:.6f}",
            "memory_gb": f"{memory_gb:.1f}",
            "status": status,
            "description": description,
        })

def run_evaluation(cfg: dict | Any, **overrides) -> Dict[str, Any]:
    norm_cfg = {}
    is_dict = False
    try:
        is_dict = isinstance(cfg, dict)
    except Exception:
        pass

    if is_dict:
        try:
            for k, v in cfg.items():
                if v is not None:
                    try:
                        norm_cfg[str(k).lower()] = v
                    except Exception:
                        pass
        except Exception:
            pass
    else:
        has_dict = False
        try:
            has_dict = hasattr(cfg, "__dict__")
        except Exception:
            pass
        if has_dict:
            try:
                for k, v in cfg.__dict__.items():
                    if not k.startswith('_') and k not in ('method_calls', 'mock_calls'):
                        if v is not None:
                            norm_cfg[k.lower()] = v
            except Exception:
                pass
        else:
            is_none = True
            try:
                is_none = (cfg is None)
            except Exception:
                pass
            if not is_none:
                attrs = []
                try:
                    attrs = dir(cfg)
                except Exception:
                    pass
                for attr in attrs:
                    if not attr.startswith('_'):
                        try:
                            val = getattr(cfg, attr)
                            if val is not None:
                                norm_cfg[attr.lower()] = val
                        except Exception:
                            pass

    norm_overrides = {k.lower(): v for k, v in overrides.items() if v is not None}

    def get_val(name: str, default: Any = None) -> Any:
        name_lower = name.lower()
        if name_lower in norm_overrides:
            return norm_overrides[name_lower]
        return norm_cfg.get(name_lower, default)

    model_filename = get_val("model", "g4-opt-it-Q4_K_M.gguf")
    kv_cache = get_val("kv", "q4_0")
    k_val = get_val("kv_k")
    if k_val is None:
        k_val = kv_cache
    v_val = get_val("kv_v")
    if v_val is None:
        v_val = kv_cache
    max_tokens = get_val("max_tokens", 1024)
    include_coding = get_val("include_coding", True)
    t_val = get_val("threads", 12)
    tb_val = get_val("threads_batch")
    b_val = get_val("batch_size", 512)
    ub_val = get_val("ubatch_size", 128)
    spec_val = get_val("spec_draft_n_max", 1)
    spec_type_val = get_val("spec_type")
    task_limit_val = get_val("coding_task_limit", 30)
    flash_attn_val = get_val("flash_attn", "on")
    parallel_val = get_val("parallel", 1)
    ctx_size_val = get_val("ctx_size", 16384)
    port_val = get_val("port", 18080)
    ngl_val = get_val("ngl", 99)
    host_val = get_val("host", "127.0.0.1")
    no_mmap_val = get_val("no_mmap", False)
    jinja_val = get_val("jinja", False)
    budget_val = get_val("reasoning_budget")
    msg_val = get_val("reasoning_budget_message")
    reasoning_val = get_val("reasoning")
    cont_batch_val = get_val("cont_batching", False)
    n_cpu_moe_val = get_val("n_cpu_moe")
    include_nexus_val = get_val("include_nexus", False)
    include_claw_val = get_val("include_claw", False)
    context_tokens_val = get_val("context_tokens", 8192)
    trial_budget = get_val("trial_budget")

    intent = ServerIntent(
        model_path=MODELS_DIR / model_filename,
        ctx_size=ctx_size_val,
        kv_cache=kv_cache,
        flash_attn=flash_attn_val,
        port=port_val,
        host=host_val,
        ngl=ngl_val,
        batch_size=b_val,
        ubatch_size=ub_val,
        threads=t_val,
        parallel=parallel_val,
        kv_cache_k=k_val,
        kv_cache_v=v_val,
        threads_batch=tb_val,
        spec_draft_n_max=spec_val,
        no_mmap=no_mmap_val,
        jinja=jinja_val,
        reasoning_budget=budget_val,
        reasoning_budget_message=msg_val,
        reasoning=reasoning_val,
        cont_batching=cont_batch_val,
        spec_type=spec_type_val,
        n_cpu_moe=n_cpu_moe_val
    )
    
    server_log = BASE_DIR / "llama_server.log"
    if server_log.exists():
        try:
            server_log.unlink()
        except OSError:
            pass
            
    res = {
        "status": "OK",
        "coding_val": 0.0, "coding_vram": 0.0,
        "he_val": 0.0, "mbpp_val": 0.0, "swe_val": 0.0,
        "val_score": 0.0, "avg_tps": 0.0, "peak_vram_gb": 0.0
    }
    
    gen_kwargs = {
        "temp": get_val("temp", 0.2),
        "top_p": get_val("top_p"),
        "min_p": get_val("min_p"),
        "top_k": get_val("top_k"),
        "repeat_penalty": get_val("repeat_penalty"),
        "presence_penalty": get_val("presence_penalty"),
        "frequency_penalty": get_val("frequency_penalty"),
    }
    gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}
    
    trial_start = time.time()
    timeout_at = trial_start + trial_budget if trial_budget else None

    try:
        with LlamaServerRunner(intent, log_path=server_log) as runner:
            client = LlamaClient(runner.port)
            system_prefix = "<|think|>\n"
            
            # 1. Coding (HumanEval + MBPP + SWE stub)
            
            # 3. Coding (If Enabled)
            if include_coding:
                print(f"  [coding] Running (limit={task_limit_val})...")
                lcb_limit_val = getattr(config, "LCB_TASK_LIMIT", 10)
                bigcode_limit_val = getattr(config, "BIGCODE_TASK_LIMIT", 10)
                coding_res = run_coding(
                    client,
                    is_test=False,
                    model_name=model_filename,
                    task_limit=task_limit_val,
                    lcb_task_limit=lcb_limit_val,
                    bigcode_task_limit=bigcode_limit_val,
                    timeout_at=timeout_at,
                    max_tokens=max_tokens,
                    **gen_kwargs,
                )
                res["coding_val"] = coding_res.val_score
                res["coding_vram"] = runner.peak_vram_mb
                res["coding_tps"] = coding_res.avg_tps
                # Pass field layout (see benchmark_coding.run_benchmark):
                #   val_pass1 = LCB, val_pass2 = HE, val_pass3 = MBPP, val_pass4 = BigCode
                res["lcb_val"] = getattr(coding_res, "val_pass1", 0.0)
                res["he_val"] = getattr(coding_res, "val_pass2", 0.0)
                res["mbpp_val"] = getattr(coding_res, "val_pass3", 0.0)
                res["bigcode_val"] = getattr(coding_res, "val_pass4", 0.0)
                # Keep legacy swe_val slot (unused) populated with 0 for results.tsv compat
                res["swe_val"] = 0.0
            
            # Compute combined metrics
            tps_list = []
            if include_coding and res.get("coding_tps", 0) > 0:
                tps_list.append(res["coding_tps"])
                
            if tps_list:
                avg_tps = sum(tps_list) / len(tps_list)
                res["avg_tps"] = avg_tps
                if avg_tps < 20.0:
                    print(f"  [WARNING] Combined TPS {avg_tps:.2f} is below 20.0! Score set to 0.0.")
                    res["val_score"] = 0.0
                    return res
            else:
                res["avg_tps"] = 0.0
                
            res["peak_vram_gb"] = max(runner.peak_vram_mb, 0.0) / 1024.0
            
            # Normalize weights: Coding = 100% (SWE 40%, HumanEval 30%, MBPP 30%)
            # Currently benchmark_coding returns the average of HumanEval and MBPP as val_score.
            # SWE will be added in benchmark_coding later.
            res["val_score"] = res.get("coding_val", 0.0)
                
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
    
    include_nexus_val = getattr(args, "include_nexus", False)
    include_claw_val = getattr(args, "include_claw", False)

    # Run evaluation
    res = run_evaluation(
        args, include_nexus=include_nexus_val, include_claw=include_claw_val
    )
    
    if res["status"] != "OK":
        print(f"Evaluation failed: {res['status']}")
        write_row(RESULTS_FILE, commit, 0.0, 0.0, 0.0, 0.0, res["peak_vram_gb"], "discard", f"FAIL: {res['status']} | {args.desc}")
        sys.exit(1)
        
    val_score = res["val_score"]
    improved = val_score > prev_best
    status = "keep" if improved else "discard"
    
    # Format details in description
    details = f"{args.model} kv={args.kv} ctx={args.ctx_size} TPS={res['avg_tps']:.1f} VRAM={res['peak_vram_gb']:.1f}GB coding={res['coding_val']:.4f}"
    details += f" lcb={res.get('lcb_val', 0.0):.4f} he={res.get('he_val', 0.0):.4f} mbpp={res.get('mbpp_val', 0.0):.4f} bigcode={res.get('bigcode_val', 0.0):.4f}"
    details += f" | {args.desc}"
    
    # Log to results.tsv
    write_row(
        RESULTS_FILE, commit, val_score,
        res.get("swe_val", 0.0), res.get("he_val", 0.0), res.get("mbpp_val", 0.0),
        res["peak_vram_gb"], status, details,
        lcb_score=res.get("lcb_val", 0.0),
        bigcode_score=res.get("bigcode_val", 0.0),
    )
    
    print("\n" + "="*40)
    print("EVALUATION COMPLETE")
    print("="*40)
    print(f"Model:          {args.model}")
    print(f"KV Cache:       {args.kv}")
    print(f"Context Size:   {args.ctx_size}")
    print("-"*40)
    print(f"Coding Score:     {res['coding_val']:.4f}")
    print(f"  LCB:            {res.get('lcb_val', 0.0):.4f}")
    print(f"  HumanEval+:     {res.get('he_val', 0.0):.4f}")
    print(f"  MBPP+:          {res.get('mbpp_val', 0.0):.4f}")
    print(f"  BigCode Hard:   {res.get('bigcode_val', 0.0):.4f}")
    print(f"Combined TPS:     {res['avg_tps']:.1f} (Threshold: >= 20.0)")
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
    print("Starting multidimensional grid sweep...")
    
    grid_kvs = args.grid_kvs if isinstance(getattr(args, "grid_kvs", None), str) else None
    kvs = [k.strip() for k in grid_kvs.split(",") if k.strip()] if grid_kvs else [args.kv]
    
    grid_max_tokens = args.grid_max_tokens if isinstance(getattr(args, "grid_max_tokens", None), str) else "1024"
    max_tokens_list = [int(m.strip()) for m in grid_max_tokens.split(",") if m.strip()]
    
    grid_kvs_k = args.grid_kvs_k if isinstance(getattr(args, "grid_kvs_k", None), str) else None
    kvs_k = [k.strip() for k in grid_kvs_k.split(",") if k.strip()] if grid_kvs_k else [None]
    
    grid_kvs_v = args.grid_kvs_v if isinstance(getattr(args, "grid_kvs_v", None), str) else None
    kvs_v = [v.strip() for v in grid_kvs_v.split(",") if v.strip()] if grid_kvs_v else [None]
    
    grid_threads = args.grid_threads if isinstance(getattr(args, "grid_threads", None), str) else None
    threads_list = [int(t.strip()) for t in grid_threads.split(",") if t.strip()] if grid_threads else [args.threads if isinstance(getattr(args, "threads", None), int) else 12]
    
    grid_threads_batch = args.grid_threads_batch if isinstance(getattr(args, "grid_threads_batch", None), str) else None
    threads_batch_list = [int(tb.strip()) for tb in grid_threads_batch.split(",") if tb.strip()] if grid_threads_batch else [args.threads_batch if isinstance(getattr(args, "threads_batch", None), int) else None]
    
    grid_batch_sizes = args.grid_batch_sizes if isinstance(getattr(args, "grid_batch_sizes", None), str) else None
    batch_sizes = [int(b.strip()) for b in grid_batch_sizes.split(",") if b.strip()] if grid_batch_sizes else [args.batch_size if isinstance(getattr(args, "batch_size", None), int) else 512]
    
    grid_ubatch_sizes = args.grid_ubatch_sizes if isinstance(getattr(args, "grid_ubatch_sizes", None), str) else None
    ubatch_sizes = [int(u.strip()) for u in grid_ubatch_sizes.split(",") if u.strip()] if grid_ubatch_sizes else [args.ubatch_size if isinstance(getattr(args, "ubatch_size", None), int) else 128]
    
    grid_spec_draft_n_max = args.grid_spec_draft_n_max if isinstance(getattr(args, "grid_spec_draft_n_max", None), str) else None
    spec_draft_list = [int(s.strip()) for s in grid_spec_draft_n_max.split(",") if s.strip()] if grid_spec_draft_n_max else [args.spec_draft_n_max if isinstance(getattr(args, "spec_draft_n_max", None), int) else 1]

    commit = get_git_commit()
    
    combinations = list(itertools.product(
        kvs, max_tokens_list, kvs_k, kvs_v, threads_list, threads_batch_list, batch_sizes, ubatch_sizes, spec_draft_list
    ))
    
    print(f"Total configurations to evaluate: {len(combinations)}")
    
    for kv, mt, kv_k, kv_v, threads, threads_batch, batch_size, ubatch_size, spec_draft in combinations:
        k_lbl = kv_k if kv_k is not None else kv
        v_lbl = kv_v if kv_v is not None else kv
        
        print(f"\n{'='*80}")
        print(f"GRID Sweep: KV_K={k_lbl}, KV_V={v_lbl}, Max Tokens={mt}, Threads={threads}, Threads_Batch={threads_batch}, Batch={batch_size}, Ubatch={ubatch_size}, SpecDraft={spec_draft}")
        print(f"{'='*80}")
        
        res = run_evaluation(
            args, kv=kv, max_tokens=mt,
            kv_k=kv_k, kv_v=kv_v, threads=threads, threads_batch=threads_batch,
            batch_size=batch_size, ubatch_size=ubatch_size, spec_draft_n_max=spec_draft
        )
        
        status = "keep"
        details = (f"GRID Sweep: model={args.model} kv_k={k_lbl} kv_v={v_lbl} max_tokens={mt} "
                   f"ctx={args.ctx_size} threads={threads} threads_batch={threads_batch} "
                   f"batch={batch_size} ubatch={ubatch_size} spec_draft={spec_draft} "
                   f"TPS={res['avg_tps']:.1f} VRAM={res['peak_vram_gb']:.1f}GB "
                   f"coding={res.get('coding_val', 0.0):.4f} lcb={res.get('lcb_val', 0.0):.4f} "
                   f"he={res.get('he_val', 0.0):.4f} mbpp={res.get('mbpp_val', 0.0):.4f} "
                   f"bigcode={res.get('bigcode_val', 0.0):.4f}")
            
        write_row(
            RESULTS_FILE, commit, res["val_score"],
            res.get("swe_val", 0.0), res.get("he_val", 0.0), res.get("mbpp_val", 0.0),
            res["peak_vram_gb"], status, details,
            lcb_score=res.get("lcb_val", 0.0),
            bigcode_score=res.get("bigcode_val", 0.0),
        )
        print(f"Grid sweep entry logged: score={res['val_score']:.6f}")

def main():
    args = parse_args()
    if args.grid:
        handle_grid_run(args)
    else:
        handle_single_run(args)

if __name__ == "__main__":
    main()
