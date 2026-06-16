#!/usr/bin/env python3
"""
Autonomous Hill-Climbing Evaluation Loop.

Reads config.py as baseline → runs ALL benchmarks → perturbs one flag →
runs again → if improved, saves as new baseline in config.py → loops forever.

Stop with Ctrl+C (SIGINT). State persists in config.py and results.tsv.
"""

import os
import sys
import json
import signal
import random
import importlib
import time
from pathlib import Path
from typing import Any

from autoresearch.core.llama_runner import estimate_vram_mb
from autoresearch.runners.run import run_evaluation, get_git_commit, write_row, RESULTS_FILE, MODELS_DIR
from autoresearch.core.search import SearchStrategy

BASE_DIR = Path(__file__).resolve().parent
VISITED_FILE = BASE_DIR / ".autoloop_visited.json"

# ── Search space: param_name → list of candidate values ──────────────────
SEARCH_SPACE = {
    "KV_CACHE_K":        ["q4_0", "q8_0", "turbo2", "turbo3", "turbo4", "f16"],
    "KV_CACHE_V":        ["q4_0", "q8_0", "turbo2", "turbo3", "turbo4", "f16"],
    "THREADS":           [6, 8, 12, 16],
    "THREADS_BATCH":     [None, 8, 12, 16, 24],
    "BATCH_SIZE":        [256, 512, 1024],
    "UBATCH_SIZE":       [64, 128, 256, 512],
    "SPEC_DRAFT_N_MAX":  [0, 1, 2, 3, 4],
    "CTX_SIZE":          [8192, 16384, 32768, 65536],
    "CONT_BATCHING":     [False, True],
    "FLASH_ATTN":        ["on"],
    "NO_MMAP":           [False, True],
    "TEMP":              [0.0, 0.2, 0.6, 0.7, 1.0],
    "TOP_P":             [None, 0.8, 0.9, 0.95],
    "TOP_K":             [None, 20, 40, 64],
    "MIN_P":             [None, 0.0, 0.02, 0.05],
    "PRESENCE_PENALTY":  [None, 0.0, 1.5],
    "REPEAT_PENALTY":    [None, 1.0, 1.1],
}

# Params not in search space but needed for config persistence
PASSTHROUGH_PARAMS = [
    "KV_CACHE", "MODEL", "JINJA", "REASONING_BUDGET", "REASONING_BUDGET_MESSAGE",
    "REASONING", "SPEC_TYPE", "FREQUENCY_PENALTY",
    "INCLUDE_CODING", "CODING_TASK_LIMIT",
    "INCLUDE_NEXUS", "INCLUDE_CLAW", "N_CPU_MOE",
]

# ── Graceful shutdown ────────────────────────────────────────────────────
_stop_requested = False

def _signal_handler(_sig, _frame):
    global _stop_requested
    _stop_requested = True
    print("\n[AUTOLOOP] Graceful stop requested. Finishing current evaluation...")

signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


def load_config() -> dict[str, Any]:
    """Hot-reload config.py and return current values as dict."""
    from autoresearch.core import config
    importlib.reload(config)
    result = {}
    for param in list(SEARCH_SPACE.keys()) + PASSTHROUGH_PARAMS:
        result[param] = getattr(config, param, None)
    return result


def write_config(cfg: dict[str, Any]) -> None:
    """Persist config dict back to config.py."""
    lines = [
        "# config.py",
        "# The ONLY changeable file for agent tweaks",
        "",
    ]

    # Server/runtime params
    server_params = [
        "MODEL", "CTX_SIZE", "KV_CACHE", "KV_CACHE_K", "KV_CACHE_V",
        "BATCH_SIZE", "UBATCH_SIZE", "THREADS", "THREADS_BATCH",
        "FLASH_ATTN", "SPEC_TYPE", "SPEC_DRAFT_N_MAX", "NO_MMAP",
        "JINJA", "REASONING_BUDGET", "REASONING_BUDGET_MESSAGE",
        "REASONING", "CONT_BATCHING", "N_CPU_MOE",
    ]
    for p in server_params:
        lines.append(f"{p} = {repr(cfg.get(p))}")

    lines.append("")
    lines.append("# Generation options")
    gen_params = [
        "TEMP", "TOP_P", "MIN_P", "TOP_K",
        "REPEAT_PENALTY", "PRESENCE_PENALTY", "FREQUENCY_PENALTY",
    ]
    for p in gen_params:
        lines.append(f"{p} = {repr(cfg.get(p))}")

    lines.append("")
    lines.append("# Benchmarks to run")
    lines.append(f"INCLUDE_CODING = {repr(cfg.get('INCLUDE_CODING', True))}")
    lines.append(f"INCLUDE_NEXUS = {repr(cfg.get('INCLUDE_NEXUS', False))}")
    lines.append(f"INCLUDE_CLAW = {repr(cfg.get('INCLUDE_CLAW', False))}")
    limit = cfg.get("CODING_TASK_LIMIT", 30)
    lines.append(f"CODING_TASK_LIMIT = {limit}  # Tasks per dataset (HumanEval/MBPP). 0 = full dataset.")
    lines.append("")

    # Write config to package path
    path = BASE_DIR / "autoresearch" / "core" / "config.py"
    path.write_text("\n".join(lines), encoding="utf-8")


def preflight_vram_ok(cfg: dict[str, Any], vram_limit: float | None) -> bool:
    """Estimate VRAM for a config and return True if it fits budget."""
    if vram_limit is None:
        return True
    
    model = cfg.get("MODEL", "g4-opt-it-Q4_K_M.gguf")
    ctx = cfg.get("CTX_SIZE", 16384)
    kv_k = cfg.get("KV_CACHE_K") or cfg.get("KV_CACHE", "q4_0")
    kv_v = cfg.get("KV_CACHE_V") or cfg.get("KV_CACHE", "q4_0")
    
    # Needs fallback to default cache if kv_k/kv_v are not defined
    if not kv_k: kv_k = "q4_0"
    if not kv_v: kv_v = "q4_0"

    est = estimate_vram_mb(MODELS_DIR / model, ctx, kv_k, kv_v)
    return est <= vram_limit

def evaluate(cfg: dict[str, Any], trial_budget: float | None = None) -> dict[str, Any]:
    """Run full benchmark suite with given config."""
    defaults = {
        "port": 18080,
        "host": "127.0.0.1",
        "parallel": 1,
        "ngl": 99,
        "max_tokens": 1024,
        "context_tokens": 8192,
    }
    merged = {**defaults, **cfg}
    return run_evaluation(merged, trial_budget=trial_budget)

def load_visited() -> set[str]:
    """Load previously visited configs from disk."""
    if VISITED_FILE.exists():
        try:
            data = json.loads(VISITED_FILE.read_text(encoding="utf-8"))
            return set(data)
        except Exception:
            pass
    return set()


def save_visited(visited: set[str]) -> None:
    """Persist visited configs to disk."""
    VISITED_FILE.write_text(json.dumps(list(visited)), encoding="utf-8")


def parse_budget_seconds(budget_str: str) -> float:
    budget_str = budget_str.strip().lower()
    if budget_str.endswith("m"):
        return float(budget_str[:-1]) * 60
    if budget_str.endswith("s"):
        return float(budget_str[:-1])
    try:
        return float(budget_str)
    except ValueError:
        print(f"[AUTOLOOP] Invalid budget format: {budget_str}. Defaulting to 5 minutes.")
        return 300.0


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Autonomous Hill-Climbing Evaluation Loop")
    parser.add_argument("--vram-limit-mb", type=float, default=7900.0, help="Max safe VRAM in MB")
    parser.add_argument("--max-rounds", type=int, default=0, help="Max rounds (0=infinite)")
    parser.add_argument("--reset-visited", action="store_true", help="Clear visited config history")
    parser.add_argument("--models", nargs="+", help="Space-separated list of model filenames to optimize (1 or more)")
    parser.add_argument("--trial-budget", type=str, help="Max time trial budget (e.g. 5m, 300s, 15m)")
    cli_args = parser.parse_args()

    vram_limit = cli_args.vram_limit_mb
    max_rounds = cli_args.max_rounds

    if cli_args.reset_visited and VISITED_FILE.exists():
        VISITED_FILE.unlink()
        print("[AUTOLOOP] Cleared visited config history.")

    # 1. Resolve selected models
    available_models = sorted([f.name for f in MODELS_DIR.glob("*.gguf")])
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
        baseline_cfg = load_config()
        selected_models = [baseline_cfg.get("MODEL", "g4-opt-it-Q4_K_M.gguf")]

    # 2. Resolve trial budget
    trial_budget_sec = None
    if cli_args.trial_budget:
        trial_budget_sec = parse_budget_seconds(cli_args.trial_budget)
    elif sys.stdin.isatty():
        choice = input("\nEnter max trial budget (e.g. 5m, 300s, 15m) [default 5m]: ").strip()
        if choice:
            trial_budget_sec = parse_budget_seconds(choice)
        else:
            trial_budget_sec = 300.0
    else:
        trial_budget_sec = 300.0

    print("=" * 60)
    print("  AUTONOMOUS HILL-CLIMBING LOOP")
    print(f"  Target models: {', '.join(selected_models)}")
    print(f"  Trial budget: {trial_budget_sec / 60:.1f} minutes ({trial_budget_sec:.0f} seconds)")
    print("  Stop with Ctrl+C. State persists in config.py + results.tsv")
    print("=" * 60)

    visited = load_visited()
    print(f"[AUTOLOOP] Loaded {len(visited)} previously visited configs.")

    search_strategy = SearchStrategy(SEARCH_SPACE, use_pareto_tiebreaker=True)

    for model_name in selected_models:
        if _stop_requested:
            break
            
        print(f"\n{'#' * 60}")
        print(f"  OPTIMIZING MODEL: {model_name}")
        print(f"{'#' * 60}")
        
        # Load config and update MODEL
        cfg = load_config()
        cfg["MODEL"] = model_name
        write_config(cfg)

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
            baseline_cfg = load_config()
            baseline_key = search_strategy.get_config_key(baseline_cfg)
            visited.add(baseline_key)

            print(f"[BASELINE] {search_strategy.format_config_summary(baseline_cfg)}")

            # ── Step 2: Evaluate baseline ────────────────────────────────
            print("\n[EVAL] Running baseline benchmarks...")
            baseline_res = evaluate(baseline_cfg, trial_budget=trial_budget_sec)
            baseline_score = baseline_res.get("val_score", 0.0)
            baseline_tps = baseline_res.get("avg_tps", 0.0)
            baseline_vram = baseline_res.get("peak_vram_gb", 0.0)

            commit = get_git_commit()
            write_row(
                RESULTS_FILE, commit, baseline_score, baseline_vram,
                "keep",
                f"AutoLoop R{round_num} baseline for {model_name}: {search_strategy.format_config_summary(baseline_cfg)} "
                f"TPS={baseline_tps:.1f}"
            )

            print(f"[BASELINE] Score={baseline_score:.6f} TPS={baseline_tps:.1f} VRAM={baseline_vram:.1f}GB")

            if _stop_requested:
                break

            # ── Step 3: Generate and evaluate neighbors ──────────────────
            neighbors = search_strategy.get_neighbors(baseline_cfg)
            improved = False

            for neighbor in neighbors:
                if _stop_requested:
                    break

                n_key = search_strategy.get_config_key(neighbor.config)
                if n_key in visited:
                    continue
                visited.add(n_key)
                save_visited(visited)

                changed = neighbor.changed
                old_val = neighbor.old
                new_val = neighbor.new

                # Pre-flight VRAM check
                if not preflight_vram_ok(neighbor.config, vram_limit):
                    print(f"  [SKIP] {changed}: {old_val} → {new_val} (VRAM over budget)")
                    continue

                print(f"\n  [EVAL] Trying {changed}: {old_val} → {new_val}")
                res = evaluate(neighbor.config, trial_budget=trial_budget_sec)
                score = res.get("val_score", 0.0)
                tps = res.get("avg_tps", 0.0)
                vram = res.get("peak_vram_gb", 0.0)

                delta = score - baseline_score
                
                # Pareto tie-breaker logic
                is_improvement, reason = search_strategy.is_improvement(
                    baseline_score, baseline_tps, baseline_vram,
                    score, tps, vram
                )

                status = "keep" if is_improvement else "discard"

                write_row(
                    RESULTS_FILE, commit, score, vram, status,
                    f"AutoLoop R{round_num} {changed}={new_val}: "
                    f"{search_strategy.format_config_summary(neighbor.config)} TPS={tps:.1f} Δ={delta:+.6f}"
                )

                if is_improvement:
                    print(f"  >>> IMPROVEMENT! {changed}: {old_val} → {new_val} "
                          f"({reason})")
                    # Persist new baseline to config.py
                    write_config(neighbor.config)
                    improved = True
                    break
                else:
                    print(f"  [DISCARD] {changed}: {old_val} → {new_val} "
                          f"(Score={score:.6f}, Δ={delta:+.6f})")

            if not improved and not _stop_requested:
                print(f"\n[AUTOLOOP] Local maxima reached in round {round_num}.")
                print("[AUTOLOOP] Attempting Random Restart...")
                new_baseline = None
                for restart_attempt in range(50):
                    candidate = search_strategy.random_restart(visited, baseline_cfg)
                    if not candidate:
                        break
                    # Pre-flight VRAM check
                    if preflight_vram_ok(candidate, vram_limit):
                        new_baseline = candidate
                        break
                    else:
                        # Mark as visited so we don't try it again
                        visited.add(search_strategy.get_config_key(candidate))
                
                if new_baseline:
                    print("[AUTOLOOP] Found unvisited VRAM-safe random configuration. Restarting search.")
                    write_config(new_baseline)
                else:
                    print("[AUTOLOOP] Exhausted random search space or cannot find VRAM-safe config. Stopping.")
                    break

    # ── Shutdown summary ─────────────────────────────────────────────
    final_cfg = load_config()
    print(f"\n{'=' * 60}")
    print("  AUTOLOOP STOPPED")
    print(f"{'=' * 60}")
    print(f"  Final config: {search_strategy.format_config_summary(final_cfg)}")
    print(f"  Results logged to: {RESULTS_FILE}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
