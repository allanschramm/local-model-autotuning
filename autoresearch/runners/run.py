import sys
import csv
import argparse
import subprocess
import itertools
import json
import uuid
import shutil
from pathlib import Path
from typing import Dict, Any

from autoresearch.core import config
from autoresearch.benchmarks import bench_config
from autoresearch.benchmarks import format_agentic_benchmarks, format_claw_tiers

from autoresearch.runners.evaluation import ExperimentRunner, BENCH_TPS_THRESHOLD

BASE_DIR = Path(__file__).resolve().parent
RESULTS_FILE = BASE_DIR.parent.parent / "results.tsv"
MODELS_DIR = BASE_DIR.parent.parent / "models"

# ── Run categories for TSV ────────────────────────────────────────────
CATEGORY_VALIDATION = "validation"
CATEGORY_10_TASK = "10-task"
CATEGORY_FULL_SUITE = "full-suite"


def determine_category(args) -> str:
    """Infer run category from CLI args."""
    if getattr(args, "validation", False):
        return CATEGORY_VALIDATION
    if getattr(args, "agentic_full", False):
        return "agentic-full"
    if getattr(args, "agentic_quick", False):
        return "agentic-quick"
    coding = getattr(args, "coding_task_limit", 10)
    lcb = getattr(args, "lcb_task_limit", 10)
    bigcode = getattr(args, "bigcode_task_limit", 10)
    if coding <= 10 and lcb <= 10 and bigcode <= 10:
        return CATEGORY_10_TASK
    return CATEGORY_FULL_SUITE

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
    parser.add_argument("--context-tokens", type=int, default=config.CTX_SIZE, help="Context tokens padding length (100k minimum)")
    parser.add_argument("--include-coding", action="store_true", default=getattr(bench_config, "INCLUDE_CODING", False), help="Run the optional 10-task direct-coding preflight")
    parser.add_argument("--no-coding", dest="include_coding", action="store_false", help="Disable Coding benchmark")
    parser.add_argument("--include-nexus", action="store_true", default=getattr(bench_config, "INCLUDE_NEXUS", False), help="Include Nexus benchmark")
    parser.add_argument("--include-claw", action="store_true", default=getattr(bench_config, "INCLUDE_CLAW", False), help="Include Claw benchmark")
    parser.add_argument(
        "--agentic-quick",
        action=argparse.BooleanOptionalAction,
        default=getattr(bench_config, "INCLUDE_AGENTIC_QUICK", False),
        help="Run Claw-Eval quick tier smoke (5 tasks; use --no-agentic-quick to disable)",
    )
    parser.add_argument(
        "--agentic-full",
        action=argparse.BooleanOptionalAction,
        default=getattr(bench_config, "INCLUDE_AGENTIC_FULL", False),
        help="Run Claw-Eval full tier quality gate (15 tasks; use --no-agentic-full to disable)",
    )
    parser.add_argument("--list-agentic-benchmarks", action="store_true", help="List long-horizon agentic benchmark targets and exit")
    parser.add_argument("--list-claw-tiers", action="store_true", help="List Claw-Eval quick/full task tiers and exit")
    parser.add_argument("--coding-task-limit", type=int, default=getattr(bench_config, "CODING_TASK_LIMIT", 30), help="Tasks per dataset (0=full dataset)")
    parser.add_argument("--lcb-task-limit", type=int, default=getattr(bench_config, "LCB_TASK_LIMIT", 10), help="LiveCodeBench task limit")
    parser.add_argument("--bigcode-task-limit", type=int, default=getattr(bench_config, "BIGCODE_TASK_LIMIT", 10), help="BigCodeBench task limit")
    parser.add_argument("--validation", action="store_true",
        help="Validation mode: run llama-bench + Claw quick smoke evaluation. "
             "Validates model load, throughput, and basic agentic behavior. "
             "No extended eval, no keep/discard. Useful for quick config sanity checks.")
    parser.add_argument("--bench-tts-threshold", type=float, default=BENCH_TPS_THRESHOLD,
        help=f"Minimum text generation t/s from llama-bench validation (default: {BENCH_TPS_THRESHOLD})")
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

def get_previous_best(results_file: Path, model_name: str | None = None) -> float:
    if not results_file.exists():
        return 0.0
    best_score = 0.0
    try:
        with open(results_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                if row.get("status") == "keep":
                    if model_name:
                        row_model = row.get("model", "")
                        if row_model and row_model != model_name:
                            continue
                        if not row_model and model_name not in row.get("description", ""):
                            continue
                    try:
                        score = float(row.get("val_score", 0.0))
                        if score > best_score:
                            best_score = score
                    except ValueError:
                        pass
    except Exception as e:
        print(f"Error reading results.tsv: {e}")
    return best_score

CATEGORY_FIELDNAMES = [
    "schema_version", "trial_id", "commit", "model", "model_id", "backend",
    "category", "evaluation_profile", "scoring_benchmark", "outcome", "diagnostic", "status",
    "val_score", "swe_score", "lcb_score", "he_score",
    "mbpp_score", "bigcode_score", "memory_gb", "elapsed_sec",
    "tps", "bench_tg", "kv", "ctx",
    "threads", "threads_batch", "batch_size", "ubatch_size",
    "n_cpu_moe", "temp", "top_p", "top_k", "min_p",
    "repeat_penalty", "presence_penalty", "cont_batching",
    "flash_attn", "no_mmap", "spec_draft_n_max",
    "task_ids", "random_seed", "config_json", "binary_version", "tps_source", "description",
]


def _ensure_category_column(results_file: Path) -> None:
    """One-time migration: add missing columns."""
    if not results_file.exists() or results_file.stat().st_size == 0:
        return
    with open(results_file, "r", encoding="utf-8") as f:
        header = f.readline().strip()
    cols = header.split("\t")
    if cols == CATEGORY_FIELDNAMES:
        return  # already migrated
    backup = results_file.with_suffix(results_file.suffix + ".bak")
    if results_file.is_file() and not backup.exists():
        shutil.copy2(results_file, backup)
    rows = []
    with open(results_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(row)
    with open(results_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CATEGORY_FIELDNAMES, delimiter="\t",
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_row(results_file: Path, commit: str, val_score: float, swe_score: float, he_score: float, mbpp_score: float, memory_gb: float, status: str, description: str, lcb_score: float = 0.0, bigcode_score: float = 0.0, category: str = "", elapsed_sec: float = 0.0, model: str = "", tps: float = 0.0, bench_tg: float = 0.0, kv: str = "", ctx: int = 0, threads: int = 0, threads_batch: int = 0, batch_size: int = 0, ubatch_size: int = 0, n_cpu_moe: int = 0, temp: float = 0.0, top_p: float = 0.0, top_k: int = 0, min_p: float = 0.0, repeat_penalty: float = 0.0, presence_penalty: float = 0.0, cont_batching: str = "", flash_attn: str = "", no_mmap: str = "", spec_draft_n_max: int = 0, outcome: str = "", diagnostic: str = "", evaluation_profile: str = "", scoring_benchmark: str = "", task_ids: str = "", config_json: str = "", tps_source: str = ""):
    _ensure_category_column(results_file)
    new_file = not results_file.exists() or results_file.stat().st_size == 0
    with open(results_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CATEGORY_FIELDNAMES, delimiter="\t")
        if new_file:
            writer.writeheader()
        row = {
            "schema_version": "2",
            "trial_id": str(uuid.uuid4()),
            "commit": commit,
            "model": model,
            "model_id": model,
            "backend": "sglang" if model and not model.lower().endswith(".gguf") else "llama.cpp",
            "category": category,
            "evaluation_profile": evaluation_profile or category,
            "scoring_benchmark": scoring_benchmark or ("claw-eval" if category.startswith("agentic") else "coding"),
            "outcome": outcome or ("OK" if not status.lower().startswith("fail") else "MODEL_REJECTED"),
            "diagnostic": diagnostic,
            "status": status,
            "val_score": f"{val_score:.6f}",
            "swe_score": f"{swe_score:.6f}",
            "lcb_score": f"{lcb_score:.6f}",
            "he_score": f"{he_score:.6f}",
            "mbpp_score": f"{mbpp_score:.6f}",
            "bigcode_score": f"{bigcode_score:.6f}",
            "memory_gb": f"{memory_gb:.1f}",
            "elapsed_sec": f"{elapsed_sec:.0f}",
            "tps": f"{tps:.1f}" if tps else "",
            "bench_tg": f"{bench_tg:.1f}" if bench_tg else "",
            "kv": kv,
            "ctx": str(ctx) if ctx else "",
            "threads": str(threads) if threads else "",
            "threads_batch": str(threads_batch) if threads_batch else "",
            "batch_size": str(batch_size) if batch_size else "",
            "ubatch_size": str(ubatch_size) if ubatch_size else "",
            "n_cpu_moe": str(n_cpu_moe) if n_cpu_moe else "",
            "temp": f"{temp}" if temp else "",
            "top_p": f"{top_p}" if top_p else "",
            "top_k": str(top_k) if top_k else "",
            "min_p": f"{min_p}" if min_p else "",
            "repeat_penalty": f"{repeat_penalty}" if repeat_penalty else "",
            "presence_penalty": f"{presence_penalty}" if presence_penalty else "",
            "cont_batching": str(cont_batching) if cont_batching else "",
            "flash_attn": flash_attn,
            "no_mmap": str(no_mmap) if no_mmap else "",
            "spec_draft_n_max": str(spec_draft_n_max) if spec_draft_n_max else "",
            "task_ids": task_ids,
            "random_seed": "",
            "config_json": config_json or json.dumps({
                "kv": kv, "ctx": ctx, "threads": threads, "threads_batch": threads_batch,
                "batch_size": batch_size, "ubatch_size": ubatch_size,
                "flash_attn": flash_attn, "spec_draft_n_max": spec_draft_n_max,
            }, separators=(",", ":"), sort_keys=True),
            "binary_version": "",
            "tps_source": tps_source,
            "description": description,
        }
        writer.writerow(row)

def run_evaluation(cfg: dict | Any, skip_bench: bool = False, **overrides) -> Dict[str, Any]:
    """Run one trial and return results as a dict (backward-compat wrapper).

    New code should use ExperimentRunner.run_trial() directly for a typed TrialResult.
    """
    runner = ExperimentRunner(MODELS_DIR)
    tr = runner.run_trial(cfg, skip_bench=skip_bench, **overrides)
    return {
        "status": tr.status,
        "val_score": tr.val_score,
        "coding_val": tr.coding_val,
        "coding_tps": tr.coding_tps,
        "lcb_val": tr.lcb_val,
        "he_val": tr.he_val,
        "mbpp_val": tr.mbpp_val,
        "bigcode_val": tr.bigcode_val,
        "swe_val": tr.swe_val,
        "agentic_val": tr.agentic_val,
        "agentic_tier": tr.agentic_tier,
        "agentic_task_count": tr.agentic_task_count,
        "avg_tps": tr.avg_tps,
        "peak_vram_gb": tr.peak_vram_gb,
        "bench_tg_tps": tr.bench_tg_tps,
        "bench_pp_tps": tr.bench_pp_tps,
        "elapsed_sec": tr.elapsed_sec,
        "outcome": tr.outcome.value,
        "diagnostic": tr.diagnostic,
        "task_ids": list(tr.task_ids),
        "tps_source": tr.tps_source,
    }

def handle_single_run(args):
    if not args.desc:
        print("Error: --desc is required for logging single runs. Example: --desc 'Tweak system prompt'")
        sys.exit(1)
        
    print(f"Starting single run for model: {args.model}")
    commit = get_git_commit()
    
    # Read previous best score
    prev_best = get_previous_best(RESULTS_FILE, args.model)
    print(f"Previous best 'keep' score: {prev_best:.6f}")
    
    include_nexus_val = getattr(args, "include_nexus", False)
    include_claw_val = getattr(args, "include_claw", False)
    agentic_quick = getattr(args, "agentic_quick", False) is True
    agentic_full = getattr(args, "agentic_full", False) is True

    # Run evaluation
    res = run_evaluation(
        args, include_nexus=include_nexus_val, include_claw=include_claw_val,
        agentic_quick=agentic_quick, agentic_full=agentic_full,
    )
    
    if res["status"] != "OK":
        print(f"Evaluation failed: {res['status']}")
        write_row(
            RESULTS_FILE, commit, 0.0, 0.0, 0.0, 0.0, res["peak_vram_gb"],
            "discard", f"FAIL: {res['status']} | {args.desc}",
            category=determine_category(args), elapsed_sec=res.get("elapsed_sec", 0.0),
            model=args.model, outcome=res.get("outcome", ""),
            diagnostic=res.get("diagnostic", ""),
            task_ids=",".join(res.get("task_ids", [])),
            tps_source=res.get("tps_source", ""),
        )
        sys.exit(1)
        
    val_score = res["val_score"]
    is_validation = getattr(args, "validation", False)
    if not isinstance(is_validation, bool):
        is_validation = False

    improved = val_score > prev_best
    status = "keep" if (improved and not is_validation) else "discard"

    details = f"{args.model} kv={args.kv} ctx={args.ctx_size} TPS={res['avg_tps']:.1f} VRAM={res['peak_vram_gb']:.1f}GB coding={res['coding_val']:.4f}"
    details += f" lcb={res.get('lcb_val', 0.0):.4f} he={res.get('he_val', 0.0):.4f} mbpp={res.get('mbpp_val', 0.0):.4f} bigcode={res.get('bigcode_val', 0.0):.4f}"
    if res.get("agentic_tier"):
        details += f" agentic_{res['agentic_tier']}={res.get('agentic_val', 0.0):.4f} (n={res.get('agentic_task_count', 0)})"
    details += f" bench_tg={res.get('bench_tg_tps', 0.0):.1f}"
    details += f" | {args.desc}"
    
    # Log to results.tsv
    write_row(
        RESULTS_FILE, commit, val_score,
        res.get("swe_val", 0.0), res.get("he_val", 0.0), res.get("mbpp_val", 0.0),
        res["peak_vram_gb"], status, details,
        lcb_score=res.get("lcb_val", 0.0),
        bigcode_score=res.get("bigcode_val", 0.0),
        category=determine_category(args),
        elapsed_sec=res.get("elapsed_sec", 0.0),
        model=args.model,
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
    print(f"Bench tg:         {res.get('bench_tg_tps', 0.0):.1f} t/s")
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
    """Run a multidimensional sweep over comma-separated grid parameters."""
    def _g(attr, default):
        """Grid param: comma-separated string → list of strings, or [default]."""
        raw = getattr(args, attr, None)
        return [x.strip() for x in raw.split(",") if x.strip()] if isinstance(raw, str) else [default]

    def _fa(attr, default):
        """Safe getattr: skip MagicMock auto-created attrs, return default instead."""
        v = getattr(args, attr, None)
        return v if isinstance(v, (int, str, type(None), float, bool)) else default

    print("Starting multidimensional grid sweep...")
    
    kvs = _g("grid_kvs", args.kv)
    max_tokens_list = [int(x) for x in _g("grid_max_tokens", "1024")]
    kvs_k = _g("grid_kvs_k", None)
    kvs_v = _g("grid_kvs_v", None)
    threads_list = [int(x) for x in _g("grid_threads", _fa("threads", 12))]
    tb_raw = getattr(args, "grid_threads_batch", None)
    threads_batch_list = (
        [int(x.strip()) for x in tb_raw.split(",") if x.strip()]
        if isinstance(tb_raw, str) else
        [_fa("threads_batch", None)]
    )
    batch_sizes = [int(x) for x in _g("grid_batch_sizes", _fa("batch_size", 512))]
    ubatch_sizes = [int(x) for x in _g("grid_ubatch_sizes", _fa("ubatch_size", 128))]
    spec_draft_list = [int(x) for x in _g("grid_spec_draft_n_max", _fa("spec_draft_n_max", 1))]

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
        
        status = "keep" if res["status"] == "OK" else "discard"
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
            category=determine_category(args),
            elapsed_sec=res.get("elapsed_sec", 0.0),
            model=args.model,
            outcome=res.get("outcome", ""),
            diagnostic=res.get("diagnostic", ""),
            task_ids=",".join(res.get("task_ids", [])),
            tps_source=res.get("tps_source", ""),
        )
        print(f"Grid sweep entry logged: score={res['val_score']:.6f}")

def main():
    args = parse_args()
    if args.list_agentic_benchmarks:
        print(format_agentic_benchmarks())
        return
    if args.list_claw_tiers:
        print(format_claw_tiers())
        return
    if args.grid:
        handle_grid_run(args)
    else:
        handle_single_run(args)

if __name__ == "__main__":
    main()
