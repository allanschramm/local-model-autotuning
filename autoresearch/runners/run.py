#!/usr/bin/env python3
import os
# Set default llama.cpp root directory
os.environ.setdefault("AUTORESEARCH_LLAMA_CPP_ROOT", "/home/shark/workspace/Nexus-System/llama.cpp")

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

# Benchmarks
from autoresearch.benchmarks.prepare import run_benchmark as run_nexus
from autoresearch.benchmarks.prepare_claw import run_benchmark as run_claw
from autoresearch.benchmarks.benchmark_coding import run_benchmark as run_coding

BASE_DIR = Path(__file__).resolve().parent
RESULTS_FILE = BASE_DIR.parent.parent / "results.tsv"
MODELS_DIR = BASE_DIR.parent.parent / "models"

def parse_args():
    parser = argparse.ArgumentParser(description="Unified AutoResearch Benchmark Runner")
    parser.add_argument("--desc", type=str, help="Description of the experiment (required for logging single runs)")
    parser.add_argument("--model", type=str, default=config.MODEL, help="Model filename in models/ directory")
    parser.add_argument("--kv", type=str, default=config.KV_CACHE, help="KV cache type (e.g. q4_0, q4_1, f16)")
    parser.add_argument("--kv-k", "--cache-type-k", "-ctk", dest="kv_k", type=str, default=config.KV_CACHE_K, help="Key cache type (overrides --kv if set)")
    parser.add_argument("--kv-v", "--cache-type-v", "-ctv", dest="kv_v", type=str, default=config.KV_CACHE_V, help="Value cache type (overrides --kv if set)")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Max generation tokens")
    parser.add_argument("--ctx-size", "-c", type=int, default=config.CTX_SIZE, help="Context size")
    parser.add_argument("--port", type=int, default=18080, help="Port for llama-server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host for llama-server")
    parser.add_argument("--threads", "-t", type=int, default=config.THREADS, help="Number of threads for llama-server")
    parser.add_argument("--threads-batch", type=int, default=config.THREADS_BATCH, help="Number of batch/prefill threads for llama-server")
    parser.add_argument("--ngl", "--n-gpu-layers", "-ngl", type=int, default=99, help="Number of GPU layers to offload")
    parser.add_argument("--batch-size", "-b", type=int, default=config.BATCH_SIZE, help="Batch size for llama-server")
    parser.add_argument("--ubatch-size", "-ub", type=int, default=config.UBATCH_SIZE, help="Micro-batch size for llama-server")
    parser.add_argument("--parallel", type=int, default=1, help="Parallel slots count")
    parser.add_argument("--flash-attn", "-fa", nargs="?", const="on", default=config.FLASH_ATTN, choices=["on", "off", "auto"], help="Enable/disable/auto Flash Attention")
    parser.add_argument("--spec-type", type=str, default=config.SPEC_TYPE, help="Speculative decoding type (e.g. draft-mtp, mtp)")
    parser.add_argument("--spec-draft-n-max", type=int, default=config.SPEC_DRAFT_N_MAX, help="Speculative draft max tokens count for MTP")
    parser.add_argument("--context-tokens", type=int, default=8192, help="Context tokens padding length")
    parser.add_argument("--include-coding", action="store_true", default=True, help="Include Coding benchmark (Humaneval+ & MBPP+), always enabled")
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

def run_evaluation(args, model_filename: str, kv_cache: str, max_tokens: int, include_coding: bool = True,
                   kv_k: str | None = None, kv_v: str | None = None,
                   threads: int | None = None, threads_batch: int | None = None,
                   batch_size: int | None = None, ubatch_size: int | None = None,
                   spec_draft_n_max: int | None = None, spec_type: str | None = None,
                   coding_task_limit: int | None = None,
                   trial_budget: float | None = None,
                   include_nexus: bool = False,
                   include_claw: bool = False) -> Dict[str, Any]:
    k_val = kv_k if kv_k is not None else (args.kv_k if isinstance(getattr(args, "kv_k", None), str) else kv_cache)
    v_val = kv_v if kv_v is not None else (args.kv_v if isinstance(getattr(args, "kv_v", None), str) else kv_cache)
    t_val = threads if threads is not None else (args.threads if isinstance(getattr(args, "threads", None), int) else 12)
    tb_val = threads_batch if threads_batch is not None else (args.threads_batch if isinstance(getattr(args, "threads_batch", None), int) else None)
    b_val = batch_size if batch_size is not None else (args.batch_size if isinstance(getattr(args, "batch_size", None), int) else 512)
    ub_val = ubatch_size if ubatch_size is not None else (args.ubatch_size if isinstance(getattr(args, "ubatch_size", None), int) else 128)
    spec_val = spec_draft_n_max if spec_draft_n_max is not None else (args.spec_draft_n_max if isinstance(getattr(args, "spec_draft_n_max", None), int) else 1)
    spec_type_val = spec_type if spec_type is not None else (args.spec_type if isinstance(getattr(args, "spec_type", None), str) else None)
    task_limit_val = coding_task_limit if coding_task_limit is not None else (args.coding_task_limit if isinstance(getattr(args, "coding_task_limit", None), int) else 30)
    
    flash_attn_val = args.flash_attn if isinstance(getattr(args, "flash_attn", None), str) else "on"
    parallel_val = args.parallel if isinstance(getattr(args, "parallel", None), int) else 1
    ctx_size_val = args.ctx_size if isinstance(getattr(args, "ctx_size", None), int) else 16384
    port_val = args.port if isinstance(getattr(args, "port", None), int) else 18080
    ngl_val = args.ngl if isinstance(getattr(args, "ngl", None), int) else 99
    host_val = args.host if isinstance(getattr(args, "host", None), str) else "127.0.0.1"

    no_mmap_val = args.no_mmap if isinstance(getattr(args, "no_mmap", None), bool) else False
    jinja_val = args.jinja if isinstance(getattr(args, "jinja", None), bool) else False
    budget_val = args.reasoning_budget if isinstance(getattr(args, "reasoning_budget", None), int) else None
    msg_val = args.reasoning_budget_message if isinstance(getattr(args, "reasoning_budget_message", None), str) else None
    reasoning_val = args.reasoning if isinstance(getattr(args, "reasoning", None), str) else None
    cont_batch_val = args.cont_batching if isinstance(getattr(args, "cont_batching", None), bool) else False

    include_nexus_val = include_nexus or getattr(args, "include_nexus", False)
    include_claw_val = include_claw or getattr(args, "include_claw", False)

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
        spec_type=spec_type_val
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
    
    gen_kwargs = {
        "temp": args.temp if isinstance(getattr(args, "temp", None), float) else 0.2,
        "top_p": args.top_p if isinstance(getattr(args, "top_p", None), float) else None,
        "min_p": args.min_p if isinstance(getattr(args, "min_p", None), float) else None,
        "top_k": args.top_k if isinstance(getattr(args, "top_k", None), int) else None,
        "repeat_penalty": args.repeat_penalty if isinstance(getattr(args, "repeat_penalty", None), float) else None,
        "presence_penalty": args.presence_penalty if isinstance(getattr(args, "presence_penalty", None), float) else None,
        "frequency_penalty": args.frequency_penalty if isinstance(getattr(args, "frequency_penalty", None), float) else None,
    }
    gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}
    
    trial_start = time.time()
    timeout_at = trial_start + trial_budget if trial_budget else None

    try:
        with LlamaServerRunner(intent, log_path=server_log) as runner:
            client = LlamaClient(runner.port)
            system_prefix = "<|think|>\n"
            
            # 1. Nexus (Retrieval)
            if include_nexus_val:
                print("  [nexus] Running...")
                nexus_res = run_nexus(client, max_tokens=max_tokens, system_prefix=system_prefix, context_tokens=args.context_tokens, timeout_at=timeout_at, **gen_kwargs)
                res["nexus_val"] = nexus_res.val_score
                res["nexus_tps"] = nexus_res.avg_tps
                res["nexus_vram"] = runner.peak_vram_mb
            
            # 2. Claw (Agency)
            if include_claw_val:
                print("  [claw] Running...")
                claw_res = run_claw(client, max_tokens=max_tokens, system_prefix=system_prefix, context_tokens=args.context_tokens, timeout_at=timeout_at, **gen_kwargs)
                res["claw_val"] = claw_res.val_score
                res["claw_tps"] = claw_res.avg_tps
                res["claw_vram"] = runner.peak_vram_mb
            
            # 3. Coding (Always Enabled)
            print(f"  [coding] Running (limit={task_limit_val})...")
            coding_res = run_coding(client, is_test=False, model_name=model_filename, task_limit=task_limit_val, timeout_at=timeout_at)
            res["coding_val"] = coding_res.val_score
            res["coding_vram"] = runner.peak_vram_mb
            
            # Compute combined metrics
            tps_list = []
            if include_nexus_val:
                tps_list.append(res["nexus_tps"])
            if include_claw_val:
                tps_list.append(res["claw_tps"])
                
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
            
            # Normalize weights: Coding = 80, Nexus = 10 (if enabled), Claw = 10 (if enabled)
            total_weight = 80.0
            weighted_score = res["coding_val"] * 80.0
            
            if include_nexus_val:
                total_weight += 10.0
                weighted_score += res["nexus_val"] * 10.0
            if include_claw_val:
                total_weight += 10.0
                weighted_score += res["claw_val"] * 10.0
                
            res["val_score"] = round(weighted_score / total_weight, 6)
                
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
        args, args.model, args.kv, args.max_tokens, args.include_coding,
        include_nexus=include_nexus_val, include_claw=include_claw_val
    )
    
    if res["status"] != "OK":
        print(f"Evaluation failed: {res['status']}")
        write_row(RESULTS_FILE, commit, 0.0, res["peak_vram_gb"], "discard", f"FAIL: {res['status']} | {args.desc}")
        sys.exit(1)
        
    val_score = res["val_score"]
    improved = val_score > prev_best
    status = "keep" if improved else "discard"
    
    # Format details in description
    details = f"{args.model} kv={args.kv} ctx={args.ctx_size} TPS={res['avg_tps']:.1f} VRAM={res['peak_vram_gb']:.1f}GB coding={res['coding_val']:.4f}"
    if include_nexus_val:
        details += f" retrieval={res['nexus_val']:.4f}"
    if include_claw_val:
        details += f" agency={res['claw_val']:.4f}"
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
    print(f"Coding Score:     {res['coding_val']:.4f}")
    if include_nexus_val:
        print(f"Nexus (Retrieval): {res['nexus_val']:.4f} (TPS: {res['nexus_tps']:.1f})")
    if include_claw_val:
        print(f"Claw (Agency):    {res['claw_val']:.4f} (TPS: {res['claw_tps']:.1f})")
    print("-"*40)
    if include_nexus_val or include_claw_val:
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
            args, args.model, kv, mt, args.include_coding,
            kv_k=kv_k, kv_v=kv_v, threads=threads, threads_batch=threads_batch,
            batch_size=batch_size, ubatch_size=ubatch_size, spec_draft_n_max=spec_draft
        )
        
        status = "keep"
        details = (f"GRID Sweep: model={args.model} kv_k={k_lbl} kv_v={v_lbl} max_tokens={mt} "
                   f"ctx={args.ctx_size} threads={threads} threads_batch={threads_batch} "
                   f"batch={batch_size} ubatch={ubatch_size} spec_draft={spec_draft} "
                   f"TPS={res['avg_tps']:.1f} VRAM={res['peak_vram_gb']:.1f}GB "
                   f"retrieval={res['nexus_val']:.4f} agency={res['claw_val']:.4f}")
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
