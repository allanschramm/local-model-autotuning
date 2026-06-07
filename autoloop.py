#!/usr/bin/env python3
"""
Autonomous Hill-Climbing Evaluation Loop.

Reads config.py as baseline → runs ALL benchmarks → perturbs one flag →
runs again → if improved, saves as new baseline in config.py → loops forever.

Stop with Ctrl+C (SIGINT). State persists in config.py and results.tsv.
"""

import os
os.environ.setdefault("AUTORESEARCH_LLAMA_CPP_ROOT", "/home/shark/workspace/Nexus-System/llama.cpp")

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
    "FLASH_ATTN":        ["on", "off", "auto"],
    "NO_MMAP":           [False, True],
    "TEMP":              [0.0, 0.1, 0.2, 0.4, 0.6],
}

# Params not in search space but needed for config persistence
PASSTHROUGH_PARAMS = [
    "KV_CACHE", "MODEL", "JINJA", "REASONING_BUDGET", "REASONING_BUDGET_MESSAGE",
    "REASONING", "SPEC_TYPE", "TOP_P", "MIN_P", "TOP_K",
    "REPEAT_PENALTY", "PRESENCE_PENALTY", "FREQUENCY_PENALTY",
    "INCLUDE_CODING", "CODING_TASK_LIMIT",
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
        "REASONING", "CONT_BATCHING",
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
    limit = cfg.get("CODING_TASK_LIMIT", 30)
    lines.append(f"CODING_TASK_LIMIT = {limit}  # Tasks per dataset (HumanEval/MBPP). 0 = full dataset.")
    lines.append("")

    config_path = BASE_DIR / "config.py"
    config_path.write_text("\n".join(lines), encoding="utf-8")


def config_to_args(cfg: dict[str, Any]) -> object:
    """Build a namespace object compatible with run_evaluation()."""
    class Args:
        pass
    a = Args()
    a.model = cfg.get("MODEL", "g4-opt-it-Q4_K_M.gguf")
    a.ctx_size = cfg.get("CTX_SIZE", 16384)
    a.kv = cfg.get("KV_CACHE", "q4_0")
    a.kv_k = cfg.get("KV_CACHE_K")
    a.kv_v = cfg.get("KV_CACHE_V")
    a.batch_size = cfg.get("BATCH_SIZE", 512)
    a.ubatch_size = cfg.get("UBATCH_SIZE", 128)
    a.threads = cfg.get("THREADS", 12)
    a.threads_batch = cfg.get("THREADS_BATCH")
    a.flash_attn = cfg.get("FLASH_ATTN", "on")
    a.spec_type = cfg.get("SPEC_TYPE")
    a.spec_draft_n_max = cfg.get("SPEC_DRAFT_N_MAX", 1)
    a.no_mmap = cfg.get("NO_MMAP", False)
    a.jinja = cfg.get("JINJA", False)
    a.reasoning_budget = cfg.get("REASONING_BUDGET")
    a.reasoning_budget_message = cfg.get("REASONING_BUDGET_MESSAGE")
    a.reasoning = cfg.get("REASONING")
    a.cont_batching = cfg.get("CONT_BATCHING", False)
    a.temp = cfg.get("TEMP", 0.2)
    a.top_p = cfg.get("TOP_P")
    a.min_p = cfg.get("MIN_P")
    a.top_k = cfg.get("TOP_K")
    a.repeat_penalty = cfg.get("REPEAT_PENALTY")
    a.presence_penalty = cfg.get("PRESENCE_PENALTY")
    a.frequency_penalty = cfg.get("FREQUENCY_PENALTY")
    a.include_coding = cfg.get("INCLUDE_CODING", True)
    a.coding_task_limit = cfg.get("CODING_TASK_LIMIT", 30)
    a.port = 18080
    a.host = "127.0.0.1"
    a.ngl = 99
    a.parallel = 1
    a.max_tokens = 1024
    a.context_tokens = 8192
    return a


def get_neighbors(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Generate single-parameter perturbations of current config."""
    neighbors = []
    for param, candidates in SEARCH_SPACE.items():
        current = cfg.get(param)
        try:
            idx = candidates.index(current)
        except ValueError:
            # Current value not in search space; try all candidates
            for val in candidates:
                if val != current:
                    n = cfg.copy()
                    n[param] = val
                    n["_changed"] = param
                    n["_old"] = current
                    n["_new"] = val
                    neighbors.append(n)
            continue

        # Adjacent neighbors in the ordered list
        if idx > 0:
            n = cfg.copy()
            n[param] = candidates[idx - 1]
            n["_changed"] = param
            n["_old"] = current
            n["_new"] = candidates[idx - 1]
            neighbors.append(n)
        if idx < len(candidates) - 1:
            n = cfg.copy()
            n[param] = candidates[idx + 1]
            n["_changed"] = param
            n["_old"] = current
            n["_new"] = candidates[idx + 1]
            neighbors.append(n)

    random.shuffle(neighbors)
    return neighbors


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


def random_restart(visited: set[str], current_cfg: dict[str, Any]) -> dict[str, Any] | None:
    """Generate a random valid configuration not in visited to escape local maxima."""
    for _ in range(100):
        new_cfg = current_cfg.copy()
        for param, values in SEARCH_SPACE.items():
            new_cfg[param] = random.choice(values)
        
        n_key = str(sorted((k, v) for k, v in new_cfg.items() if k in SEARCH_SPACE))
        if n_key not in visited:
            return new_cfg
    return None


def evaluate(cfg: dict[str, Any]) -> dict[str, Any]:
    """Run full benchmark suite with given config."""
    args = config_to_args(cfg)
    return run_evaluation(
        args, args.model, args.kv, args.max_tokens, args.include_coding,
        kv_k=args.kv_k, kv_v=args.kv_v,
        threads=args.threads, threads_batch=args.threads_batch,
        batch_size=args.batch_size, ubatch_size=args.ubatch_size,
        spec_draft_n_max=args.spec_draft_n_max, spec_type=args.spec_type,
        coding_task_limit=args.coding_task_limit
    )


def format_config_summary(cfg: dict[str, Any]) -> str:
    """One-line summary of tunable params for logging."""
    parts = []
    for p in SEARCH_SPACE:
        v = cfg.get(p)
        if v is not None:
            parts.append(f"{p}={v}")
    return " ".join(parts)


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


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Autonomous Hill-Climbing Evaluation Loop")
    parser.add_argument("--vram-limit-mb", type=float, default=7900.0, help="Max safe VRAM in MB")
    parser.add_argument("--max-rounds", type=int, default=0, help="Max rounds (0=infinite)")
    parser.add_argument("--reset-visited", action="store_true", help="Clear visited config history")
    cli_args = parser.parse_args()

    vram_limit = cli_args.vram_limit_mb
    max_rounds = cli_args.max_rounds

    if cli_args.reset_visited and VISITED_FILE.exists():
        VISITED_FILE.unlink()
        print("[AUTOLOOP] Cleared visited config history.")

    print("=" * 60)
    print("  AUTONOMOUS HILL-CLIMBING LOOP")
    print("  Stop with Ctrl+C. State persists in config.py + results.tsv")
    print("=" * 60)

    round_num = 0
    visited = load_visited()
    print(f"[AUTOLOOP] Loaded {len(visited)} previously visited configs.")

    while not _stop_requested:
        round_num += 1
        if max_rounds > 0 and round_num > max_rounds:
            print(f"\n[AUTOLOOP] Reached max rounds ({max_rounds}). Stopping.")
            break

        print(f"\n{'=' * 60}")
        print(f"  ROUND {round_num}")
        print(f"{'=' * 60}")

        # ── Step 1: Load current baseline from config.py ─────────────
        baseline_cfg = load_config()
        baseline_key = str(sorted((k, v) for k, v in baseline_cfg.items() if k in SEARCH_SPACE))
        visited.add(baseline_key)

        print(f"[BASELINE] {format_config_summary(baseline_cfg)}")

        # ── Step 2: Evaluate baseline ────────────────────────────────
        print("\n[EVAL] Running baseline benchmarks...")
        baseline_res = evaluate(baseline_cfg)
        baseline_score = baseline_res.get("val_score", 0.0)
        baseline_tps = baseline_res.get("avg_tps", 0.0)
        baseline_vram = baseline_res.get("peak_vram_gb", 0.0)

        commit = get_git_commit()
        write_row(
            RESULTS_FILE, commit, baseline_score, baseline_vram,
            "keep",
            f"AutoLoop R{round_num} baseline: {format_config_summary(baseline_cfg)} "
            f"TPS={baseline_tps:.1f}"
        )

        print(f"[BASELINE] Score={baseline_score:.6f} TPS={baseline_tps:.1f} VRAM={baseline_vram:.1f}GB")

        if _stop_requested:
            break

        # ── Step 3: Generate and evaluate neighbors ──────────────────
        neighbors = get_neighbors(baseline_cfg)
        improved = False

        for neighbor in neighbors:
            if _stop_requested:
                break

            n_key = str(sorted((k, v) for k, v in neighbor.items() if k in SEARCH_SPACE))
            if n_key in visited:
                continue
            visited.add(n_key)
            save_visited(visited)

            changed = neighbor.pop("_changed", "?")
            old_val = neighbor.pop("_old", "?")
            new_val = neighbor.pop("_new", "?")

            # Pre-flight VRAM check
            if not preflight_vram_ok(neighbor, vram_limit):
                print(f"  [SKIP] {changed}: {old_val} → {new_val} (VRAM over budget)")
                continue

            print(f"\n  [EVAL] Trying {changed}: {old_val} → {new_val}")
            res = evaluate(neighbor)
            score = res.get("val_score", 0.0)
            tps = res.get("avg_tps", 0.0)
            vram = res.get("peak_vram_gb", 0.0)

            delta = score - baseline_score
            
            # Pareto tie-breaker logic
            is_improvement = False
            reason = ""
            if score > baseline_score + 0.0001:
                is_improvement = True
                reason = f"Score improved (Δ={delta:+.6f})"
            elif abs(score - baseline_score) <= 0.0001:
                if tps > baseline_tps * 1.05:
                    is_improvement = True
                    reason = f"Score tied, TPS improved (+{tps - baseline_tps:.1f})"
                elif tps >= baseline_tps * 0.95 and vram < baseline_vram * 0.95:
                    is_improvement = True
                    reason = f"Score/TPS tied, VRAM improved (-{baseline_vram - vram:.1f}GB)"

            status = "keep" if is_improvement else "discard"

            write_row(
                RESULTS_FILE, commit, score, vram, status,
                f"AutoLoop R{round_num} {changed}={new_val}: "
                f"{format_config_summary(neighbor)} TPS={tps:.1f} Δ={delta:+.6f}"
            )

            if is_improvement:
                print(f"  >>> IMPROVEMENT! {changed}: {old_val} → {new_val} "
                      f"({reason})")
                # Persist new baseline to config.py
                write_config(neighbor)
                improved = True
                break
            else:
                print(f"  [DISCARD] {changed}: {old_val} → {new_val} "
                      f"(Score={score:.6f}, Δ={delta:+.6f})")

        if not improved and not _stop_requested:
            print(f"\n[AUTOLOOP] Local maxima reached in round {round_num}.")
            print("[AUTOLOOP] Attempting Random Restart...")
            new_baseline = random_restart(visited, baseline_cfg)
            if new_baseline:
                print("[AUTOLOOP] Found unvisited random configuration. Restarting search.")
                write_config(new_baseline)
            else:
                print("[AUTOLOOP] Exhausted random search space. Stopping.")
                break

    # ── Shutdown summary ─────────────────────────────────────────────
    final_cfg = load_config()
    print(f"\n{'=' * 60}")
    print("  AUTOLOOP STOPPED")
    print(f"{'=' * 60}")
    print(f"  Rounds completed: {round_num}")
    print(f"  Final config: {format_config_summary(final_cfg)}")
    print(f"  Results logged to: {RESULTS_FILE}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
