"""Harness-backed Trials reader for the dashboard (#25)."""

from __future__ import annotations

from typing import Any

from autoresearch.core import trial_store as run_mod


def read_last_50_trials() -> list[dict[str, str]]:
    """Last 50 results.tsv rows, newest first ([] when missing/empty)."""
    rows = run_mod.read_rows(run_mod.RESULTS_FILE)
    if not rows:
        return []
    # File order is append-chronological; newest last → reverse for UI.
    return list(reversed(rows))[:50]


# Canonical → pt-BR display mapping for Trial Status pills.
# Canonical labels stay untouched in the data/API.
_STATUS_PT: dict[str, str] = {
    "on_front": "na fronteira",
    "dominated": "dominado",
    "incomplete": "incompleto",
    "rejected": "rejeitado",
}


def status_pt(canonical: str) -> str:
    """Return pt-BR display label for a canonical Trial status."""
    return _STATUS_PT.get(canonical, canonical)


def format_trial_for_ui(row: dict[str, str]) -> dict[str, Any]:
    """Operator columns; pass through ADR 0006 status as stored.

    Adds ``status_pt`` presentation field (pt-BR pill label).
    The canonical ``status`` field stays untouched.
    """
    return {
        "status": row.get("status") or "",
        "status_pt": status_pt(row.get("status") or ""),
        "outcome": row.get("outcome") or "",
        "ctx": row.get("ctx") or "",
        "tps": row.get("tps") or "",
        "agentic": row.get("agentic") or "",
        "coding": row.get("coding") or "",
        "memory": row.get("memory_gb") or "",
        "elapsed": row.get("elapsed_sec") or "",
        "diagnostic": row.get("diagnostic") or "",
        "description": row.get("description") or "",
    }
