"""Read bounded tuning evidence without modifying historical SQLite tables."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

DATABASE = Path(__file__).resolve().parents[1] / "results.db"


def read_last_50_trials() -> list[dict[str, Any]]:
    """Newest tuning attempts; historical quality stores produce an empty view."""
    if not DATABASE.is_file():
        return []
    with sqlite3.connect(DATABASE.as_uri() + "?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if not {"tuning_trials", "tuning_runs"} <= tables:
            return []
        records = db.execute("""
            SELECT t.*, r.frontier_json, r.reference_ppl
            FROM tuning_trials AS t JOIN tuning_runs AS r USING (run_id)
            WHERE r.protocol = 'tuning-runtime-v1'
            ORDER BY t.rowid DESC LIMIT 50
        """).fetchall()
    rows = []
    for record in records:
        row = dict(record)
        frontier = {point["trial_id"] for point in json.loads(row.pop("frontier_json") or "[]")}
        configuration = json.loads(row.pop("config_json"))
        row["model"] = configuration.get("MODEL", "")
        row["status"] = (
            "on_front"
            if row["trial_id"] in frontier
            else "dominated"
            if row["accepted"]
            else "rejected"
        )
        rows.append(row)
    return rows


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


def format_trial_for_ui(row: dict[str, Any]) -> dict[str, Any]:
    """Format tuning columns and computed two-axis frontier membership.

    Adds ``status_pt`` presentation field (pt-BR pill label).
    The canonical ``status`` field stays untouched.
    """
    return {
        "status": row.get("status") or "",
        "status_pt": status_pt(row.get("status") or ""),
        "outcome": row.get("outcome") or "",
        "ctx": row.get("ctx") or "",
        "tps": row.get("tps") or "",
        "ppl": row.get("ppl"),
        "reference_ppl": row.get("reference_ppl"),
        "model": row.get("model", ""),
        "description": row.get("model", ""),
    }
