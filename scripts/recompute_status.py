#!/usr/bin/env python3
"""Recompute Trial statuses in a results store (issue #5; ADR 0017).

Reads a results.tsv (canonical results.db first), refreshes every row's
status. Domination is same-model only (ADR 0017): store-wide recompute never
demotes one model for another — a complete basename vector is on_front,
partial vectors stay incomplete, and legacy cross-model `dominated` labels
flip to on_front via the normal pass. Default scope is per-model (`--scope
model`); `--scope bucket` is retained as an accepted argument form with the
same same-basename competition. Idempotent: running twice changes nothing.
No GPU required.

Usage (repo root):
    .\\venv\\Scripts\\python.exe scripts\\recompute_status.py
    .\\venv\\Scripts\\python.exe scripts\\recompute_status.py path\\to\\results.tsv
    .\\venv\\Scripts\\python.exe scripts\\recompute_status.py --scope bucket
    .\\venv\\Scripts\\python.exe scripts\\recompute_status.py --relabel-watchdog
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from autoresearch.core import recompute
from autoresearch.runners import run


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Refresh Trial statuses in a results store (same-model domination only, ADR 0017)."
    )
    parser.add_argument(
        "results_file",
        nargs="?",
        default=str(REPO_ROOT / "results.tsv"),
        help="path to results.tsv (canonical results.db read first)",
    )
    parser.add_argument(
        "--scope",
        choices=sorted(recompute.SCOPES),
        default=recompute.DEFAULT_SCOPE,
        help=(
            "model (default): per-basename statuses, persisted (ADR 0017); "
            "bucket: accepted argument form with the same same-basename competition"
        ),
    )
    parser.add_argument(
        "--relabel-watchdog",
        action="store_true",
        help=(
            "one-shot issue #72 migration: relabel legacy NVML device-wide "
            "policy kills (MODEL_REJECTED/VRAM_LIMIT_EXCEEDED) to WATCHDOG_KILL "
            "before the status refresh; idempotent, no GPU"
        ),
    )
    args = parser.parse_args()
    results_file = Path(args.results_file)
    if not results_file.exists():
        print(f"results store not found: {results_file}", file=sys.stderr)
        return 1
    if args.relabel_watchdog:
        relabeled = run.relabel_watchdog_kills(results_file)
        print(f"watchdog kills relabeled: {relabeled} rows in {results_file}")
    rows = run.read_rows(results_file)
    updated = recompute.recompute_rows(rows, scope=args.scope)
    for row in updated:
        print(f"{row.get('trial_id', '')}\t{row.get('model', '')}\t{row['status']}")
    run.recompute_statuses(results_file, scope=args.scope)
    print(f"statuses refreshed: {results_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
