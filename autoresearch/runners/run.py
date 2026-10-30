from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

from autoresearch.benchmarks import bench_config, format_agentic_benchmarks, format_claw_tiers
from autoresearch.core import (
    classify,
    config,
    fingerprint,
)
from autoresearch.core.evaluation_protocol import CODING_PROTOCOL
from autoresearch.core.trial_store import (
    CATEGORY_FIELDNAMES as CATEGORY_FIELDNAMES,
)
from autoresearch.core.trial_store import (
    _engine_version as _engine_version,
)
from autoresearch.core.trial_store import (
    _ensure_category_column as _ensure_category_column,
)
from autoresearch.core.trial_store import (
    _replace_tsv as _replace_tsv,
)
from autoresearch.core.trial_store import (
    _results_lock as _results_lock,
)
from autoresearch.core.trial_store import (
    _tsv_cell as _tsv_cell,
)
from autoresearch.core.trial_store import (
    get_git_commit as get_git_commit,
)
from autoresearch.core.trial_store import (
    read_rows as read_rows,
)
from autoresearch.core.trial_store import (
    recompute_statuses as recompute_statuses,
)
from autoresearch.core.trial_store import (
    relabel_watchdog_kills as relabel_watchdog_kills,
)
from autoresearch.core.trial_store import (
    tsv_fields_from_cfg as tsv_fields_from_cfg,
)
from autoresearch.core.trial_store import (
    write_row as write_row,
)
from autoresearch.runners.evaluation import ExperimentRunner, resolve_tps_floor

BASE_DIR = Path(__file__).resolve().parent
from autoresearch.core.trial_store import MODELS_DIR as MODELS_DIR
from autoresearch.core.trial_store import RESULTS_FILE as RESULTS_FILE

# ── Run categories for TSV ────────────────────────────────────────────
CATEGORY_VALIDATION = "validation"
CATEGORY_10_TASK = "10-task"
CATEGORY_FULL_SUITE = "full-suite"

# Trial Status (ADR 0006): on_front | dominated | incomplete | rejected.
# keep/discard are not accepted on write (deleted from the local store).
from autoresearch.core.trial_store import TRIAL_STATUSES as TRIAL_STATUSES

BASELINE_CLI_FLAGS = {
    "--model",
    "--kv",
    "--kv-k",
    "--cache-type-k",
    "-ctk",
    "--kv-v",
    "--cache-type-v",
    "-ctv",
    "--max-tokens",
    "--ctx-size",
    "-c",
    "--threads",
    "-t",
    "--threads-batch",
    "--n-cpu-moe",
    "-ncmoe",
    "--ngl",
    "--n-gpu-layers",
    "-ngl",
    "--parallel",
    "--context-tokens",
    "--batch-size",
    "-b",
    "--ubatch-size",
    "-ub",
    "--flash-attn",
    "-fa",
    "--spec-type",
    "--spec-draft-n-max",
    "--spec-draft-model",
    "--no-mmap",
    "--jinja",
    "--reasoning-budget",
    "--reasoning-budget-message",
    "--reasoning",
    "--reasoning-effort",
    "--no-reasoning-preserve",
    "--cont-batching",
    "--temp",
    "--top-p",
    "--min-p",
    "--top-k",
    "--repeat-penalty",
    "--presence-penalty",
    "--frequency-penalty",
    "--coding-task-limit",
    "--lcb-task-limit",
    "--bigcode-task-limit",
    "--bench-tts-threshold",
}


def determine_category(args) -> str:
    """Infer run category from CLI args."""
    if getattr(args, "validation", False):
        return CATEGORY_VALIDATION
    if getattr(args, "mini_swe_agent", False):
        return "mini-swe-agent"
    if getattr(args, "agentic_full", False):
        return "agentic-full"
    if getattr(args, "agentic_coding", False):
        return "agentic-coding"
    if getattr(args, "agentic_quick", False):
        return "agentic-quick"
    coding = getattr(args, "coding_task_limit", 10)
    lcb = getattr(args, "lcb_task_limit", 10)
    bigcode = getattr(args, "bigcode_task_limit", 10)
    if coding <= 10 and lcb <= 10 and bigcode <= 10:
        return CATEGORY_10_TASK
    return CATEGORY_FULL_SUITE


def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified AutoResearch Benchmark Runner", allow_abbrev=False
    )
    parser.add_argument(
        "--desc", type=str, help="Description of the experiment (required for logging single runs)"
    )
    parser.add_argument(
        "--model", type=str, default=config.MODEL, help="Model filename in models/ directory"
    )
    parser.add_argument(
        "--kv", type=str, default=config.KV_CACHE, help="KV cache type (e.g. q4_0, q4_1, f16)"
    )
    parser.add_argument(
        "--kv-k",
        "--cache-type-k",
        "-ctk",
        dest="kv_k",
        type=str,
        default=config.KV_CACHE_K,
        help="Key cache type (overrides --kv if set)",
    )
    parser.add_argument(
        "--kv-v",
        "--cache-type-v",
        "-ctv",
        dest="kv_v",
        type=str,
        default=config.KV_CACHE_V,
        help="Value cache type (overrides --kv if set)",
    )
    parser.add_argument("--max-tokens", type=int, default=1024, help="Max generation tokens")
    parser.add_argument("--ctx-size", "-c", type=int, default=config.CTX_SIZE, help="Context size")
    parser.add_argument("--port", type=int, default=18080, help="Port for llama-server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host for llama-server")
    parser.add_argument(
        "--threads",
        "-t",
        type=int,
        default=config.THREADS,
        help="Number of threads for llama-server",
    )
    parser.add_argument(
        "--threads-batch",
        type=int,
        default=config.THREADS_BATCH,
        help="Number of batch/prefill threads for llama-server",
    )
    parser.add_argument(
        "--ngl",
        "--n-gpu-layers",
        "-ngl",
        type=int,
        default=99,
        help="Number of GPU layers to offload",
    )
    parser.add_argument(
        "--n-cpu-moe",
        "-ncmoe",
        type=int,
        default=getattr(config, "N_CPU_MOE", None),
        help="Keep MoE expert weights of first N layers on CPU (VITRIOL)",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=config.BATCH_SIZE,
        help="Batch size for llama-server",
    )
    parser.add_argument(
        "--ubatch-size",
        "-ub",
        type=int,
        default=config.UBATCH_SIZE,
        help="Micro-batch size for llama-server",
    )
    parser.add_argument("--parallel", type=int, default=1, help="Parallel slots count")
    parser.add_argument(
        "--flash-attn",
        "-fa",
        nargs="?",
        const="on",
        default=config.FLASH_ATTN,
        choices=["on", "off", "auto"],
        help="Enable/disable/auto Flash Attention",
    )
    parser.add_argument(
        "--spec-type",
        type=str,
        default=config.SPEC_TYPE,
        help="Speculative decoding type (e.g. draft-mtp, mtp)",
    )
    parser.add_argument(
        "--spec-draft-n-max",
        type=int,
        default=config.SPEC_DRAFT_N_MAX,
        help="Speculative draft max tokens count for MTP",
    )
    parser.add_argument(
        "--spec-draft-model",
        type=str,
        default=config.SPEC_DRAFT_MODEL,
        help="Speculative draft model filename/path",
    )
    parser.add_argument(
        "--context-tokens",
        type=int,
        default=config.CTX_SIZE,
        help="Context tokens padding length (100k minimum)",
    )
    parser.add_argument(
        "--include-coding",
        action="store_true",
        default=getattr(bench_config, "INCLUDE_CODING", False),
        help="Run the optional 10-task direct-coding preflight",
    )
    parser.add_argument(
        "--no-coding", dest="include_coding", action="store_false", help="Disable Coding benchmark"
    )
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
    parser.add_argument(
        "--agentic-coding",
        action=argparse.BooleanOptionalAction,
        default=getattr(bench_config, "INCLUDE_AGENTIC_CODING", False),
        help="Run SWE-lite issue loop (ADR 0013 Night selector; use --agentic-coding to enable)",
    )
    parser.add_argument(
        "--mini-swe-agent",
        dest="mini_swe_agent",
        action=argparse.BooleanOptionalAction,
        default=getattr(bench_config, "INCLUDE_MINI_SWE_AGENT", False),
        help="Run DM-Code-Agent 30-task real-SWE eval via mini-swe-agent "
        "(layers next to SWE-lite per ADR 0013; off by default; opt-in)",
    )
    parser.add_argument(
        "--mini-swe-agent-task-limit",
        type=int,
        default=getattr(bench_config, "MINI_SWE_AGENT_TASK_LIMIT", 0),
        help="Limit DM-Code-Agent scoreboard to first N tasks (0 = full suite; "
        "smoke runs use a small N to bound Trial wall-time)",
    )
    parser.add_argument(
        "--mini-swe-agent-suite",
        type=str,
        default=getattr(bench_config, "MINI_SWE_AGENT_SUITE", "all"),
        choices=["coding", "maintenance", "all"],
        help="DM-Code-Agent suite to run (default: bench_config MINI_SWE_AGENT_SUITE)",
    )
    parser.add_argument(
        "--list-agentic-benchmarks",
        action="store_true",
        help="List long-horizon agentic benchmark targets and exit",
    )
    parser.add_argument(
        "--list-claw-tiers",
        action="store_true",
        help="List Claw-Eval quick/full task tiers and exit",
    )
    parser.add_argument(
        "--coding-task-limit",
        type=int,
        default=getattr(bench_config, "CODING_TASK_LIMIT", 30),
        help="Tasks per dataset (0=full dataset)",
    )
    parser.add_argument(
        "--lcb-task-limit",
        type=int,
        default=getattr(bench_config, "LCB_TASK_LIMIT", 10),
        help="LiveCodeBench task limit",
    )
    parser.add_argument(
        "--bigcode-task-limit",
        type=int,
        default=getattr(bench_config, "BIGCODE_TASK_LIMIT", 10),
        help="BigCodeBench task limit",
    )
    parser.add_argument(
        "--validation",
        action="store_true",
        help="Validation mode: run llama-bench + Claw quick smoke evaluation. "
        "Validates model load, throughput, and basic agentic behavior. "
        "No extended eval, no keep/discard. Useful for quick config sanity checks.",
    )
    parser.add_argument(
        "--bench-tts-threshold",
        type=float,
        default=resolve_tps_floor(),
        help="Minimum text generation t/s (default: Baseline TPS_FLOOR; config.py-only)",
    )
    parser.add_argument(
        "--no-mmap", action="store_true", default=config.NO_MMAP, help="Disable mmap"
    )
    parser.add_argument(
        "--jinja",
        action="store_true",
        default=config.JINJA,
        help="Enable Jinja chat template engine",
    )
    parser.add_argument(
        "--reasoning-budget",
        type=int,
        default=config.REASONING_BUDGET,
        help="Thinking budget tokens limit",
    )
    parser.add_argument(
        "--reasoning-budget-message",
        type=str,
        default=config.REASONING_BUDGET_MESSAGE,
        help="Message on thinking budget exhaust",
    )
    parser.add_argument(
        "--reasoning",
        type=str,
        choices=["on", "off", "auto"],
        default=config.REASONING,
        help="Reasoning mode (on/off/auto)",
    )
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        choices=["minimal", "low", "medium", "high", "xhigh", "max"],
        default=getattr(config, "REASONING_EFFORT", None),
        help="Reasoning effort level passed to the chat template (config.py-only)",
    )
    parser.add_argument(
        "--reasoning-preserve",
        action=argparse.BooleanOptionalAction,
        default=getattr(config, "REASONING_PRESERVE", None),
        help="Preserve reasoning traces in full chat history (config.py-only)",
    )
    parser.add_argument(
        "--cont-batching",
        action="store_true",
        default=config.CONT_BATCHING,
        help="Enable continuous batching",
    )
    parser.add_argument("--temp", type=float, default=config.TEMP, help="Generation temperature")
    parser.add_argument("--top-p", type=float, default=config.TOP_P, help="Top-p sampling")
    parser.add_argument("--min-p", type=float, default=config.MIN_P, help="Min-p sampling")
    parser.add_argument("--top-k", type=int, default=config.TOP_K, help="Top-k sampling")
    parser.add_argument(
        "--repeat-penalty", type=float, default=config.REPEAT_PENALTY, help="Repeat penalty"
    )
    parser.add_argument(
        "--presence-penalty", type=float, default=config.PRESENCE_PENALTY, help="Presence penalty"
    )
    parser.add_argument(
        "--frequency-penalty",
        type=float,
        default=config.FREQUENCY_PENALTY,
        help="Frequency penalty",
    )

    for action in parser._actions:
        if BASELINE_CLI_FLAGS.intersection(action.option_strings):
            action.help = argparse.SUPPRESS

    forbidden = [
        arg.split("=", 1)[0] for arg in sys.argv[1:] if arg.split("=", 1)[0] in BASELINE_CLI_FLAGS
    ]
    if forbidden:
        parser.error(f"Baseline flags are config.py-only: {', '.join(forbidden)}")
    return parser.parse_args()


def _result_config() -> dict[str, Any]:
    return tsv_fields_from_cfg(config.load_config())


def get_previous_best(results_file: Path, model_name: str | None = None) -> float:
    if not results_file.exists():
        return 0.0
    best_score = 0.0
    try:
        with open(results_file, encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                if classify.is_on_front(row.get("status")):
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
        print(f"Error reading results store: {e}")
    return best_score


def run_evaluation(cfg: dict | Any, skip_bench: bool = False, **overrides) -> dict[str, Any]:
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
        "agentic_coding_val": tr.agentic_coding_val,
        "agentic_coding_detail": tr.agentic_coding_detail,
        "mini_swe_agent_val": getattr(tr, "mini_swe_agent_val", None),
        "mini_swe_agent_detail": getattr(tr, "mini_swe_agent_detail", ""),
        "avg_tps": tr.avg_tps,
        "peak_vram_gb": tr.peak_vram_gb,
        "bench_tg_tps": tr.bench_tg_tps,
        "bench_pp_tps": tr.bench_pp_tps,
        "elapsed_sec": tr.elapsed_sec,
        "outcome": tr.outcome.value,
        "diagnostic": tr.diagnostic,
        "task_ids": list(tr.task_ids),
        "tps_source": tr.tps_source,
        "gpu_temp_c": tr.gpu_temp_c,
        "tps_reps": list(tr.tps_reps),
        "tps_spread": tr.tps_spread,
    }


def fingerprint_reject_reason(args) -> str | None:
    """Trial gate (issue #53): None = Trial may proceed, else the reject reason.

    Baseline CLI flags are config.py-only (parse_args rejects them), so the
    Trial always serves the live Baseline — compare it against the Fingerprint
    file for the Trial model. No file → None (Baseline-only behavior).
    """
    baseline = config.load_config()
    model = (getattr(args, "model", "") or "").strip() or str(baseline.get("MODEL", "")).strip()
    return fingerprint.mismatch_reason(model, baseline)


def handle_single_run(args):
    if not args.desc:
        print(
            "Error: --desc is required for logging single runs. Example: --desc 'Tweak system prompt'"
        )
        sys.exit(1)

    fp_reason = fingerprint_reject_reason(args)
    if fp_reason is not None:
        # Fingerprint mismatch (issue #53): reject before scoring — the Trial
        # would log numbers for the wrong flags. Rejected row, never a score.
        print(f"Evaluation failed: {fp_reason}")
        write_row(
            RESULTS_FILE,
            get_git_commit(),
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            "rejected",
            f"FAIL: {fp_reason} | {args.desc}",
            category=determine_category(args),
            elapsed_sec=0.0,
            tps=None,
            bench_tg=None,
            outcome="MODEL_REJECTED",
            diagnostic=fp_reason,
            task_ids="",
            tps_source="",
            **_result_config(),
        )
        recompute_statuses(RESULTS_FILE)
        sys.exit(1)

    print(f"Starting single run for model: {args.model}")
    commit = get_git_commit()

    agentic_quick = getattr(args, "agentic_quick", False) is True
    agentic_full = getattr(args, "agentic_full", False) is True
    agentic_coding = getattr(args, "agentic_coding", False) is True
    mini_swe_agent = getattr(args, "mini_swe_agent", False) is True
    include_coding = getattr(args, "include_coding", False) is True and not mini_swe_agent

    # Run evaluation
    res = run_evaluation(
        args,
        include_coding=include_coding,
        agentic_quick=agentic_quick,
        agentic_full=agentic_full,
        agentic_coding=agentic_coding,
        mini_swe_agent=mini_swe_agent,
    )

    failed = res["status"] != "OK" or res.get("outcome", "OK") != "OK"
    if failed:
        # status is "FAIL: <detail>" when set; outcome is the enum. A TPS-floor
        # reject leaves status at its OK default, so fall back to the outcome.
        reason = (
            res["status"] if str(res["status"]).startswith("FAIL") else (res.get("outcome") or "OK")
        )
        print(f"Evaluation failed: {reason}")
        write_row(
            RESULTS_FILE,
            commit,
            0.0,
            0.0,
            0.0,
            0.0,
            res["peak_vram_gb"],
            "rejected",  # classifier: hard failure (VRAM policy / crash / invalid)
            f"FAIL: {reason} | {args.desc}",
            category=determine_category(args),
            elapsed_sec=res.get("elapsed_sec", 0.0),
            tps=res.get("avg_tps"),
            bench_tg=res.get("bench_tg_tps"),
            outcome=res.get("outcome", ""),
            diagnostic=res.get("diagnostic", ""),
            task_ids=",".join(res.get("task_ids", [])),
            tps_source=res.get("tps_source", ""),
            **_result_config(),
        )
        recompute_statuses(RESULTS_FILE)  # literal "after every write", issue #5
        sys.exit(1)

    val_score = res["val_score"]

    # Classify via the Pareto nucleus (issue #4): rejected on hard failure,
    # incomplete while any axis is missing, else on_front/dominated vs the
    # known Set for this hardware+budget bucket.
    # coding-10 is canonical-Trial work; validation = smoke gates only (run_trial
    # forces include_coding off). Mirror that so the vector never claims a
    # 0.0 coding axis on a validation row.
    coding_measured = include_coding and not getattr(args, "validation", False)
    # Claw quick is smoke, not the agentic axis (ADR 0006: agentic = Claw full).
    agentic_full = res.get("agentic_tier") == "full"
    vector = classify.ObjectiveVector(
        ctx=args.ctx_size,
        tps=res["avg_tps"] or None,
        agentic=res["agentic_val"] if agentic_full else None,
        coding=res["coding_val"] if coding_measured else None,
    )
    _ensure_category_column(RESULTS_FILE)
    rows = read_rows(RESULTS_FILE)
    fp = classify.fp_from_baseline(dict(config.load_config(), CODING_PROTOCOL=CODING_PROTOCOL))
    status, _ = classify.plan_write(
        rows,
        fp=fp,
        vector=vector,
        bucket_gb=classify.bucket(res["peak_vram_gb"]),
        model=(args.model or "").strip() or None,
        coding_protocol=CODING_PROTOCOL,
    )

    details = f"{args.model} kv={args.kv} ctx={args.ctx_size} TPS={res['avg_tps']:.1f} VRAM={res['peak_vram_gb']:.1f}GB coding={res['coding_val']:.4f}"
    details += f" lcb={res.get('lcb_val', 0.0):.4f} he={res.get('he_val', 0.0):.4f} mbpp={res.get('mbpp_val', 0.0):.4f} bigcode={res.get('bigcode_val', 0.0):.4f}"
    if res.get("agentic_tier"):
        details += f" agentic_{res['agentic_tier']}={res.get('agentic_val', 0.0):.4f} (n={res.get('agentic_task_count', 0)})"
    if res.get("agentic_coding_val") is not None:
        details += (
            f" agentic_coding={res['agentic_coding_val']:.4f}"
            f" ({res.get('agentic_coding_detail') or ''})"
        )
    if res.get("mini_swe_agent_val") is not None:
        details += (
            f" mini_swe_agent={res['mini_swe_agent_val']:.4f}"
            f" ({res.get('mini_swe_agent_detail') or ''})"
        )
    details += f" bench_tg={res.get('bench_tg_tps', 0.0):.1f}"
    details += f" | {args.desc}"

    # Log to the results store (DB + legacy TSV)
    write_row(
        RESULTS_FILE,
        commit,
        val_score,
        res.get("swe_val", 0.0),
        res.get("he_val", 0.0),
        res.get("mbpp_val", 0.0),
        res["peak_vram_gb"],
        status,
        details,
        lcb_score=res.get("lcb_val", 0.0),
        bigcode_score=res.get("bigcode_val", 0.0),
        agentic=vector.agentic,
        coding=vector.coding,
        agentic_coding=res.get("agentic_coding_val"),
        mini_swe_agent=res.get("mini_swe_agent_val"),
        mini_swe_agent_detail=res.get("mini_swe_agent_detail") or "",
        category=determine_category(args),
        elapsed_sec=res.get("elapsed_sec", 0.0),
        tps=res.get("avg_tps"),
        bench_tg=res.get("bench_tg_tps"),
        outcome=res.get("outcome", ""),
        diagnostic=res.get("diagnostic", ""),
        task_ids=",".join(res.get("task_ids", [])),
        tps_source=res.get("tps_source", ""),
        gpu_temp_c=res.get("gpu_temp_c"),
        tps_reps=",".join(f"{x:.4g}" for x in res.get("tps_reps", ()) or ()),
        tps_spread=res.get("tps_spread"),
        **_result_config(),
    )
    recompute_statuses(RESULTS_FILE)

    print("\n" + "=" * 40)
    print("EVALUATION COMPLETE")
    print("=" * 40)
    print(f"Model:          {args.model}")
    print(f"KV Cache:       {args.kv}")
    print(f"Context Size:   {args.ctx_size}")
    print("-" * 40)
    print(f"Coding Score:     {res['coding_val']:.4f}")
    print(f"  LCB:            {res.get('lcb_val', 0.0):.4f}")
    print(f"  HumanEval+:     {res.get('he_val', 0.0):.4f}")
    print(f"  MBPP+:          {res.get('mbpp_val', 0.0):.4f}")
    print(f"  BigCode Hard:   {res.get('bigcode_val', 0.0):.4f}")
    print(f"Combined TPS:     {res['avg_tps']:.1f} (Threshold: >= {resolve_tps_floor():.1f})")
    print(f"Bench tg:         {res.get('bench_tg_tps', 0.0):.1f} t/s")
    print(f"Peak VRAM:        {res['peak_vram_gb']:.1f} GB")
    print(f"Current Score:    {val_score:.6f}")
    print("-" * 40)
    print(f"\n>>> TRIAL STATUS: {status}")


def main():
    args = parse_args()
    if args.list_agentic_benchmarks:
        print(format_agentic_benchmarks())
        return
    if args.list_claw_tiers:
        print(format_claw_tiers())
        return
    handle_single_run(args)


if __name__ == "__main__":
    main()
