#!/usr/bin/env python3
"""
Autonomous Hill-Climbing Evaluation Loop.

Reads Baseline from autoresearch/core/config.py →
runs active benchmarks → perturbs one flag → if improved, writes Baseline
back to config.py → loops forever.

Stop with Ctrl+C (SIGINT). Visited memory persists in .autoresearch_state.json;
Baseline persists in config.py; results in results.tsv.
"""

import sys
import json
import signal
from pathlib import Path
from typing import Any

from autoresearch.core.llama_runner import resolve_model_path, estimate_vram_mb
from autoresearch.core.config import (
    ENGINE_DEFAULTS,
    SAMPLER_DEFAULTS,
)
from autoresearch.runners.run import get_git_commit, write_row, RESULTS_FILE, MODELS_DIR
from autoresearch.runners.evaluation import ExperimentRunner
from autoresearch.runners.evaluation import TrialOutcome
from autoresearch.core.search import SearchStrategy
from autoresearch.core.state import SearchState

BASE_DIR = Path(__file__).resolve().parent

# ── Search space: param_name → list of candidate values ──────────────────
SEARCH_SPACE = {
    "KV_CACHE_K":        ["q4_0", "q8_0", "turbo2", "turbo3", "turbo4", "f16"],
    "KV_CACHE_V":        ["q4_0", "q8_0", "turbo2", "turbo3", "turbo4", "f16"],
    "THREADS":           [6, 8, 12, 16],
    "THREADS_BATCH":     [None, 8, 12, 16, 24],
    "BATCH_SIZE":        [256, 512, 1024],
    "UBATCH_SIZE":       [64, 128, 256, 512],
    "SPEC_DRAFT_N_MAX":  [0, 1, 2, 3, 4],
    "CONT_BATCHING":     [False, True],
    "FLASH_ATTN":        ["on"],
    "NO_MMAP":           [False, True],
    "TEMP":              [0.0, 0.2, 0.4, 0.6, 0.7, 1.0],
    "TOP_P":             [None, 0.8, 0.9, 0.95],
    "TOP_K":             [None, 20, 40, 64],
    "MIN_P":             [None, 0.0, 0.02, 0.05],
    "PRESENCE_PENALTY":  [None, 0.0, 1.5],
    "REPEAT_PENALTY":    [None, 1.0, 1.1],
}

# Params not in search space but needed for config persistence
# Core params (in autoresearch.core.config)
CORE_PASSTHROUGH = [
    "KV_CACHE", "MODEL", "CTX_SIZE", "JINJA", "REASONING_BUDGET", "REASONING_BUDGET_MESSAGE",
    "REASONING", "SPEC_TYPE", "SPEC_DRAFT_MODEL", "FREQUENCY_PENALTY", "N_CPU_MOE",
]
# Bench params (in autoresearch.benchmarks.bench_config)
BENCH_PASSTHROUGH = [
    "INCLUDE_CODING", "CODING_TASK_LIMIT",
    "INCLUDE_NEXUS", "INCLUDE_CLAW", "INCLUDE_AGENTIC_QUICK", "INCLUDE_AGENTIC_FULL",
]
PASSTHROUGH_PARAMS = CORE_PASSTHROUGH + BENCH_PASSTHROUGH

# ── Graceful shutdown ────────────────────────────────────────────────────
_stop_requested = False

def _signal_handler(_sig, _frame):
    global _stop_requested
    _stop_requested = True
    print("\n[AUTOLOOP] Graceful stop requested. Finishing current evaluation...")

signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)

def load_config(baseline_cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load immutable defaults overlaid by the local Baseline state."""
    from autoresearch.benchmarks import bench_config as _bc
    if baseline_cfg is None:
        from autoresearch.core.config import load_config as _core_load_config
        baseline_cfg = _core_load_config()
    result = dict(baseline_cfg)
    # Merge bench params
    bench_vals = {p: getattr(_bc, p, None) for p in BENCH_PASSTHROUGH}
    result.update({k: v for k, v in bench_vals.items() if v is not None})
    return result


def update_model_alias(model_name: str, new_cfg: dict, tps: float, mode: str) -> None:
    """Resolve model alias in models/aliases/ and update its config.yaml with new flags and TPS."""
    import yaml
    aliases_dir = Path(__file__).resolve().parent / "models" / "aliases"
    if not aliases_dir.exists():
        return

    # Option A: Automatic convention matching (lowercase prefix)
    model_lower = model_name.lower()
    target_dir = None
    for d in aliases_dir.iterdir():
        if d.is_dir() and model_lower.startswith(d.name.lower()):
            target_dir = d
            break

    if not target_dir:
        return

    yaml_path = target_dir / "config.yaml"
    if not yaml_path.exists():
        return

    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        # 1. Compile flags from new_cfg
        flags = []
        if new_cfg.get("JINJA"): flags.append("--jinja")
        if new_cfg.get("CTX_SIZE"): flags.append(f"--ctx-size {new_cfg['CTX_SIZE']}")
        flags.append("--n-gpu-layers 99")
        
        k_val = new_cfg.get("KV_CACHE_K") or new_cfg.get("KV_CACHE")
        if k_val: flags.append(f"--cache-type-k {k_val}")
        
        v_val = new_cfg.get("KV_CACHE_V") or new_cfg.get("KV_CACHE")
        if v_val: flags.append(f"--cache-type-v {v_val}")
        
        if new_cfg.get("FLASH_ATTN"): flags.append(f"--flash-attn {new_cfg['FLASH_ATTN']}")
        if new_cfg.get("THREADS"): flags.append(f"--threads {new_cfg['THREADS']}")
        if new_cfg.get("THREADS_BATCH"): flags.append(f"--threads-batch {new_cfg['THREADS_BATCH']}")
        if new_cfg.get("BATCH_SIZE"): flags.append(f"--batch-size {new_cfg['BATCH_SIZE']}")
        if new_cfg.get("UBATCH_SIZE"): flags.append(f"--ubatch-size {new_cfg['UBATCH_SIZE']}")
        if new_cfg.get("CONT_BATCHING"): flags.append("--cont-batching")
        
        spec_type = new_cfg.get("SPEC_TYPE")
        if spec_type and spec_type != "none":
            flags.append(f"--spec-type {spec_type}")
            if new_cfg.get("SPEC_DRAFT_N_MAX", 0) > 0:
                flags.append(f"--spec-draft-n-max {new_cfg['SPEC_DRAFT_N_MAX']}")
            if new_cfg.get("SPEC_DRAFT_MODEL"):
                flags.append(f"--spec-draft-model models/{new_cfg['SPEC_DRAFT_MODEL']}")

        for p in ["TEMP", "TOP_P", "TOP_K", "MIN_P", "REPEAT_PENALTY", "PRESENCE_PENALTY"]:
            val = new_cfg.get(p)
            if val is not None:
                flags.append(f"--{p.lower().replace('_', '-')} {val}")

        data["flags"] = flags

        # 2. Update metrics
        if "metrics" not in data or not isinstance(data["metrics"], dict):
            data["metrics"] = {}
        data["metrics"]["tps"] = float(tps)
        
        import datetime
        data["metrics"]["measured_at"] = datetime.date.today().strftime("%Y-%m-%d")
        data["metrics"]["measured_by"] = "autoloop"
        
        notes = data["metrics"].get("notes", "")
        if notes and "(Auto-updated by autoloop)" not in notes:
            data["metrics"]["notes"] = f"{notes} (Auto-updated by autoloop)"
        elif not notes:
            data["metrics"]["notes"] = "Auto-updated by autoloop"

        # 3. Write back with formatting preserved
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)

        print(f"  [ALIAS] Automatically updated alias config at {yaml_path}")
    except Exception as e:
        print(f"  [WARNING] Failed to auto-update alias config: {e}")


def trial_config(cfg: dict[str, Any], defaults: dict[str, Any], include_ppl: bool = False) -> dict[str, Any]:
    """Map bench_config INCLUDE_* flags onto evaluation.py agentic_*/include_coding keys."""
    res_cfg = {**defaults, **cfg}
    if include_ppl:
        for k in ["INCLUDE_CODING", "INCLUDE_AGENTIC_QUICK", "INCLUDE_AGENTIC_FULL",
                  "include_coding", "agentic_quick", "agentic_full",
                  "include_agentic_quick", "include_agentic_full"]:
            res_cfg[k] = False
        res_cfg["include_perplexity"] = True
    else:
        res_cfg["include_coding"] = bool(cfg.get("INCLUDE_CODING", False))
        res_cfg["agentic_quick"] = bool(cfg.get("INCLUDE_AGENTIC_QUICK", False))
        res_cfg["agentic_full"] = bool(cfg.get("INCLUDE_AGENTIC_FULL", False))
        res_cfg["include_perplexity"] = False
    return res_cfg


def preflight_vram_ok(cfg: dict[str, Any], vram_limit: float | None) -> bool:
    """Estimate VRAM for a config (incl. draft) and return True if it fits budget."""
    if vram_limit is None:
        return True

    model = cfg.get("MODEL", "g4-opt-it-Q4_K_M.gguf")
    ctx = cfg.get("CTX_SIZE", 131072)
    kv_k = cfg.get("KV_CACHE_K") or cfg.get("KV_CACHE", "q4_0")
    kv_v = cfg.get("KV_CACHE_V") or cfg.get("KV_CACHE", "q4_0")
    if not kv_k:
        kv_k = "q4_0"
    if not kv_v:
        kv_v = "q4_0"
    draft = cfg.get("SPEC_DRAFT_MODEL")
    draft_path = resolve_model_path(MODELS_DIR, draft) if draft else None
    # Prefer module-level estimate_vram_mb so tests can patch autoloop.estimate_vram_mb.
    est = estimate_vram_mb(
        resolve_model_path(MODELS_DIR, model),
        ctx,
        kv_k,
        kv_v,
        draft_path=draft_path,
    )
    return est <= vram_limit


def _available_gguf_names(models_dir: Path) -> list[str]:
    """Basenames of main GGUFs under models/ (nested OK). Skips draft/vision/aliases/cache."""
    skip_roots = {".cache", "aliases", "huggingface", "draft", "vision"}
    names: set[str] = set()
    for path in models_dir.rglob("*.gguf"):
        rel = path.relative_to(models_dir)
        if rel.parts and rel.parts[0] in skip_roots:
            continue
        if any(part in {".cache", "aliases", "huggingface"} for part in rel.parts):
            continue
        names.add(path.name)
    return sorted(names)

def main():
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    if hasattr(sys.stderr, "reconfigure"):
        try:
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass
    import argparse
    parser = argparse.ArgumentParser(description="Autonomous Hill-Climbing Evaluation Loop")
    parser.add_argument("--vram-limit-mb", type=float, default=7900.0, help="Max safe VRAM in MB")
    parser.add_argument("--max-rounds", type=int, default=0, help="Max rounds (0=infinite)")
    parser.add_argument("--reset-visited", action="store_true", help="Clear visited memory only (Baseline stays in config.py)")
    parser.add_argument("--models", nargs="+", help="Space-separated list of model filenames to optimize (1 or more)")
    parser.add_argument("--perplexity-val", action="store_true", help="Enable perplexity validation to act as a quality ceiling constraint while optimizing for TPS")
    parser.add_argument("--mode", choices=["tps", "quality", "both"], default="both", help="Optimization mode: 'tps' (speed), 'quality' (accuracy), 'both' (everything)")
    cli_args = parser.parse_args()

    vram_limit = cli_args.vram_limit_mb
    max_rounds = cli_args.max_rounds

    state_manager = SearchState()

    if cli_args.reset_visited:
        state_manager.reset()
        print("[AUTOLOOP] Cleared visited memory. Baseline unchanged in config.py.")

    # 1. Resolve selected models
    available_models = _available_gguf_names(MODELS_DIR)
    if not available_models:
        print("[AUTOLOOP] Error: No GGUF models found in models/ directory!")
        sys.exit(1)
        
    selected_models = []
    if cli_args.models:
        for m in cli_args.models:
            if m not in available_models:
                matches = [am for am in available_models if m.lower() in am.lower()]
                if matches:
                    selected_models.append(matches[0])
                else:
                    print(f"[AUTOLOOP] Error: Model '{m}' not found in models/.")
                    sys.exit(1)
            else:
                selected_models.append(m)
    elif sys.stdin.isatty():
        print("\nAvailable models in models/:")
        for idx, m in enumerate(available_models, 1):
            print(f"  {idx}) {m}")
        print("\nChoose 1 or more models to run the loop (comma-separated numbers, e.g. 1,3 or 'all'):")
        while True:
            choice = input("Choice: ").strip()
            if not choice:
                continue
            if choice.lower() == "all":
                selected_models = available_models
                break
            try:
                indices = [int(i.strip()) for i in choice.split(",")]
                selected_models = [available_models[i-1] for i in indices if 1 <= i <= len(available_models)]
                if selected_models:
                    break
            except Exception:
                pass
            print("Invalid choice, try again.")
    else:
        baseline_cfg = load_config(state_manager.get_baseline())
        selected_models = [baseline_cfg.get("MODEL", "g4-opt-it-Q4_K_M.gguf")]

    print("=" * 60)
    print("  AUTONOMOUS HILL-CLIMBING LOOP")
    print(f"  Target models: {', '.join(selected_models)}")
    print("  Trial budget: none (runs to completion)")
    print("  Stop with Ctrl+C. State persists in .autoresearch_state.json + results.tsv")
    print("=" * 60)

    print(f"[AUTOLOOP] Loaded {len(state_manager.visited)} previously visited configs.")

    runner = ExperimentRunner(MODELS_DIR)
    _defaults = {
        "port": 18080,
        "host": "127.0.0.1",
        "parallel": 1,
        "ngl": 99,
        "max_tokens": 1024,
        "context_tokens": 131072,
    }
    ENGINE_KEYS = set(ENGINE_DEFAULTS.keys())
    SAMPLER_KEYS = set(SAMPLER_DEFAULTS.keys())

    active_search_space = dict(SEARCH_SPACE)
    if cli_args.mode == "tps":
        active_search_space = {k: v for k, v in SEARCH_SPACE.items() if k in ENGINE_KEYS}
    elif cli_args.mode == "quality":
        active_search_space = {k: v for k, v in SEARCH_SPACE.items() if k in SAMPLER_KEYS}

    search_strategy = SearchStrategy(active_search_space, use_pareto_tiebreaker=True)

    for model_name in selected_models:
        if _stop_requested:
            break
            
        print(f"\n{'#' * 60}")
        print(f"  OPTIMIZING MODEL: {model_name}")
        print(f"{'#' * 60}")
        
        # Load config and update MODEL
        cfg = load_config(state_manager.get_baseline())
        cfg["MODEL"] = model_name
        state_manager.update_baseline(cfg)

        round_num = 0
        while not _stop_requested:
            round_num += 1
            if max_rounds > 0 and round_num > max_rounds:
                print(f"\n[AUTOLOOP] Reached max rounds ({max_rounds}) for {model_name}. Stopping.")
                break

            print(f"\n{'=' * 60}")
            print(f"  ROUND {round_num} ({model_name})")
            print(f"{'=' * 60}")

            # ── Step 1: Load current baseline from config.py ─────────────
            baseline_cfg = load_config(state_manager.get_baseline())
            baseline_key = search_strategy.get_config_key(baseline_cfg)
            if not state_manager.is_visited(baseline_key):
                state_manager.mark_visited(baseline_key)

            print(f"[BASELINE] {search_strategy.format_config_summary(baseline_cfg)}")

            # ── Step 2: Evaluate baseline ────────────────────────────────
            print("\n[EVAL] Running baseline benchmarks...")
            is_tps_mode = (cli_args.mode == "tps")
            baseline_res = runner.run_trial(trial_config(baseline_cfg, _defaults, include_ppl=(is_tps_mode or cli_args.perplexity_val)))
            if getattr(baseline_res, "outcome", TrialOutcome.OK) in (TrialOutcome.INFRA_ERROR, TrialOutcome.CODE_ERROR):
                raise RuntimeError(f"Search stopped: {baseline_res.status}")
            baseline_score = baseline_res.val_score
            baseline_tps = baseline_res.avg_tps
            baseline_vram = baseline_res.peak_vram_gb
            baseline_outcome = getattr(baseline_res, "outcome", TrialOutcome.OK)
            baseline_status = "discard" if baseline_outcome in (TrialOutcome.INVALID_CONFIG, TrialOutcome.MODEL_REJECTED) else "keep"

            if cli_args.mode == "tps":
                tsv_category = "engine-tps"
            elif cli_args.mode == "quality":
                tsv_category = "sampler-quality"
            else:
                tsv_category = "agentic-full"

            commit = get_git_commit()
            write_row(
                RESULTS_FILE, commit, baseline_score,
                baseline_res.swe_val, baseline_res.he_val, baseline_res.mbpp_val,
                baseline_vram,
                baseline_status,
                f"AutoLoop R{round_num} baseline for {model_name}: {search_strategy.format_config_summary(baseline_cfg)} "
                f"TPS={baseline_tps:.1f} PPL={getattr(baseline_res, 'bench_ppl', 0.0):.4f}",
                lcb_score=baseline_res.lcb_val, bigcode_score=baseline_res.bigcode_val,
                category=tsv_category,
                model=model_name,
                outcome=getattr(getattr(baseline_res, "outcome", TrialOutcome.OK), "value", "OK"),
                diagnostic=getattr(baseline_res, "diagnostic", ""),
                evaluation_profile=tsv_category,
                scoring_benchmark="claw-eval",
                task_ids=",".join(getattr(baseline_res, "task_ids", ())),
                config_json=json.dumps(baseline_cfg, sort_keys=True, default=repr),
                tps_source=getattr(baseline_res, "tps_source", ""),
            )

            ppl_str = f" PPL={getattr(baseline_res, 'bench_ppl', 0.0):.4f}" if (is_tps_mode or cli_args.perplexity_val) else ""
            print(f"[BASELINE] Score={baseline_score:.6f} TPS={baseline_tps:.1f}{ppl_str} VRAM={baseline_vram:.1f}GB")

            if baseline_status == "discard":
                print(f"[BASELINE] Rejected ({baseline_res.status}); attempting Random Restart.")
                new_baseline = search_strategy.random_restart(state_manager.visited, baseline_cfg)
                if new_baseline and preflight_vram_ok(new_baseline, vram_limit):
                    state_manager.update_baseline(new_baseline)
                    continue
                print("[AUTOLOOP] No unvisited VRAM-safe restart available. Stopping.")
                break

            if _stop_requested:
                break

            # ── Step 3: Generate and evaluate neighbors ──────────────────
            neighbors = search_strategy.get_neighbors(baseline_cfg)
            improved = False

            for neighbor in neighbors:
                if _stop_requested:
                    break

                n_key = search_strategy.get_config_key(neighbor.config)
                if state_manager.is_visited(n_key):
                    continue
                state_manager.mark_visited(n_key)

                changed = neighbor.changed
                old_val = neighbor.old
                new_val = neighbor.new

                # Pre-flight VRAM check
                if not preflight_vram_ok(neighbor.config, vram_limit):
                    print(f"  [SKIP] {changed}: {old_val} -> {new_val} (VRAM over budget)")
                    continue

                print(f"\n  [EVAL] Trying {changed}: {old_val} -> {new_val}")
                res = runner.run_trial(trial_config(neighbor.config, _defaults, include_ppl=(is_tps_mode or cli_args.perplexity_val)))
                if getattr(res, "outcome", TrialOutcome.OK) in (TrialOutcome.INFRA_ERROR, TrialOutcome.CODE_ERROR):
                    raise RuntimeError(f"Search stopped: {res.status}")
                score = res.val_score
                tps = res.avg_tps
                vram = res.peak_vram_gb

                delta = score - baseline_score
                
                # Pareto tie-breaker logic
                is_improvement, reason = search_strategy.is_improvement(
                    baseline_score, baseline_tps, baseline_vram,
                    score, tps, vram
                )

                # Apply Perplexity Quality Ceiling Constraint
                if is_tps_mode or cli_args.perplexity_val:
                    n_ppl = getattr(res, "bench_ppl", 0.0)
                    b_ppl = getattr(baseline_res, "bench_ppl", 0.0)
                    if n_ppl > b_ppl * 1.01:
                        is_improvement = False
                        reason = f"Perplexity degraded too much (PPL={n_ppl:.4f} vs base={b_ppl:.4f})"

                status = "keep" if is_improvement else "discard"

                write_row(
                    RESULTS_FILE, commit, score,
                    res.swe_val, res.he_val, res.mbpp_val,
                    vram, status,
                    f"AutoLoop R{round_num} {changed}={new_val}: "
                    f"{search_strategy.format_config_summary(neighbor.config)} TPS={tps:.1f} PPL={getattr(res, 'bench_ppl', 0.0):.4f} Δ={delta:+.6f}",
                    lcb_score=res.lcb_val, bigcode_score=res.bigcode_val,
                    category=tsv_category,
                    model=model_name,
                    outcome=getattr(getattr(res, "outcome", TrialOutcome.OK), "value", "OK"),
                    diagnostic=getattr(res, "diagnostic", ""),
                    evaluation_profile=tsv_category,
                    scoring_benchmark="claw-eval",
                    task_ids=",".join(getattr(res, "task_ids", ())),
                    config_json=json.dumps(neighbor.config, sort_keys=True, default=repr),
                    tps_source=getattr(res, "tps_source", ""),
                )

                if is_improvement:
                    print(f"  >>> IMPROVEMENT! {changed}: {old_val} -> {new_val} "
                          f"({reason})")
                    # Persist new baseline to config.py
                    state_manager.update_baseline(neighbor.config)
                    # Automatically update model alias config
                    update_model_alias(model_name, neighbor.config, tps, cli_args.mode)
                    improved = True
                    break
                else:
                    print(f"  [DISCARD] {changed}: {old_val} -> {new_val} "
                          f"(Score={score:.6f}, Δ={delta:+.6f})")

            if not improved and not _stop_requested:
                print(f"\n[AUTOLOOP] Local maxima reached in round {round_num}.")
                print("[AUTOLOOP] Attempting Random Restart...")
                new_baseline = None
                for _ in range(50):
                    candidate = search_strategy.random_restart(state_manager.visited, baseline_cfg)
                    if not candidate:
                        break
                    # Pre-flight VRAM check
                    if preflight_vram_ok(candidate, vram_limit):
                        new_baseline = candidate
                        break
                    else:
                        # Mark as visited in memory so we don't try it again in this round, but do not write to disk
                        state_manager.mark_visited(search_strategy.get_config_key(candidate), persist=False)
                
                if new_baseline:
                    print("[AUTOLOOP] Found unvisited VRAM-safe random configuration. Restarting search.")
                    state_manager.update_baseline(new_baseline)
                else:
                    print("[AUTOLOOP] Exhausted random search space or cannot find VRAM-safe config. Stopping.")
                    break

    # ── Shutdown summary ─────────────────────────────────────────────
    final_cfg = load_config(state_manager.get_baseline())
    print(f"\n{'=' * 60}")
    print("  AUTOLOOP STOPPED")
    print(f"{'=' * 60}")
    print(f"  Final config: {search_strategy.format_config_summary(final_cfg)}")
    print(f"  Results logged to: {RESULTS_FILE}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
