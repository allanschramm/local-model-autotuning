#!/usr/bin/env python3
"""
Baseline smoke test: 1 run per model, ~5min budget.
MTP for Qwen (has nextn_predict_layers), TurboQuant (turbo4) for both.
Validates codebase works end-to-end.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from autoresearch.runners.run import run_evaluation

MODELS = [
    "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
    "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf",
]

TRIAL_BUDGET = 300  # 5 minutes per model
TASK_LIMIT = 1


def run_baseline(model: str) -> dict:
    # MTP only for Qwen (has nextn_predict_layers=1)
    has_mtp = "Qwen" in model

    print(f"\n{'='*60}")
    print(f"BASELINE SMOKE: {model}")
    print(f"  MTP={'on' if has_mtp else 'N/A'}, KV=turbo4, budget={TRIAL_BUDGET}s, tasks={TASK_LIMIT}")
    print(f"{'='*60}")

    cfg = {
        "model": model,
        "kv": "turbo4",
        "kv_k": "turbo4",
        "kv_v": "turbo4",
        "ctx_size": 4096,
        "batch_size": 128,
        "ubatch_size": 64,
        "threads": 8,
        "threads_batch": 8,
        "flash_attn": "on",
        "coding_task_limit": TASK_LIMIT,
        "include_coding": True,
        "include_nexus": False,
        "include_claw": False,
        "trial_budget": TRIAL_BUDGET,
        "temp": 0.4,
        "top_p": 0.95,
        "top_k": 20,
    }

    if has_mtp:
        cfg["spec_type"] = "mtp"
        cfg["spec_draft_n_max"] = 5

    start = time.time()
    res = run_evaluation(cfg)
    elapsed = time.time() - start

    print(f"\n  Result: {res['status']}")
    print(f"  Score:  {res['val_score']:.6f}")
    print(f"  TPS:    {res['avg_tps']:.1f}")
    print(f"  VRAM:   {res['peak_vram_gb']:.1f} GB")
    print(f"  Time:   {elapsed:.1f}s")
    return res


if __name__ == "__main__":
    results = {}
    for model in MODELS:
        results[model] = run_baseline(model)

    print(f"\n{'='*60}")
    print("SMOKE TEST SUMMARY")
    print(f"{'='*60}")
    all_ok = True
    for model, res in results.items():
        status = "PASS" if res["status"] == "OK" else "FAIL"
        if status == "FAIL":
            all_ok = False
        mtp = "MTP" if "Qwen" in model else "no-MTP"
        print(f"  {status} | {model} | {mtp} | turbo4 | score={res['val_score']:.4f} | tps={res['avg_tps']:.1f}")

    if not all_ok:
        print("\nSome tests FAILED.")
        sys.exit(1)
    else:
        print("\nAll smoke tests PASSED.")
        sys.exit(0)
