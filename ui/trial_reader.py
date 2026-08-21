"""Harness-backed Trials reader for the dashboard (#25).

The canonical ADR 0006 ``status`` (on_front | dominated | incomplete |
rejected) is passed through untouched. A pure pt-BR presentation mapping adds
a ``status_pt`` label (and a pill ``status_color``) per row so the dashboard can
localize the Trial Status column server-side without mutating the payload.
"""

from __future__ import annotations

from typing import Any

from autoresearch.runners import run as run_mod

# pt-BR presentation labels for the canonical ADR 0006 Trial Status values.
# The English ``status`` key is the source of truth; ``status_pt`` only affects
# how the dashboard renders the status pill.
STATUS_PT: dict[str, str] = {
    "on_front": "na fronteira",
    "dominated": "dominado",
    "incomplete": "incompleto",
    "rejected": "rejeitado",
}

# Pill accent color per canonical status (CSS custom-property value).
STATUS_COLOR: dict[str, str] = {
    "on_front": "#0a7a2f",  # green
    "dominated": "#666666",  # gray
    "incomplete": "#b8860b",  # amber
    "rejected": "#d70018",  # red
}


def read_last_50_trials() -> list[dict[str, str]]:
    """Last 50 results.tsv rows, newest first ([] when missing/empty)."""
    rows = run_mod.read_rows(run_mod.RESULTS_FILE)
    if not rows:
        return []
    # File order is append-chronological; newest last → reverse for UI.
    return list(reversed(rows))[:50]


def status_pt(status: Any) -> str:
    """pt-BR presentation label for a canonical ADR 0006 status.

    Unknown/empty values fall back to the raw status so the label is never
    lost. This is the pure server-side localization helper.
    """
    key = (status or "").strip()
    return STATUS_PT.get(key, key or "—")


def status_color(status: Any) -> str:
    """Pill accent color for a canonical ADR 0006 status (default gray)."""
    key = (status or "").strip()
    return STATUS_COLOR.get(key, "#999999")


def format_trial_for_ui(row: dict[str, str]) -> dict[str, Any]:
    """Operator columns; pass through ADR 0006 ``status`` as stored and add the
    pt-BR ``status_pt`` presentation field (+ pill ``status_color``)."""
    status = row.get("status") or ""
    return {
        "status": status,
        "status_pt": status_pt(status),
        "status_color": status_color(status),
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
