"""Trial write path serialization tests (issue #3).

Objective Vector axes (ctx, TPS, agentic, coding — partials allowed) and
Trial Status (on_front | dominated | incomplete | rejected) persist through
write_row into results.tsv. Legacy keep/discard are rejected on write.
"""

from __future__ import annotations

import csv
import json

import pytest

from autoresearch.runners import run
from autoresearch.runners.run import (
    TRIAL_STATUSES,
    get_previous_best,
    write_row,
)


def _read(tmp_tsv) -> list[dict]:
    with open(tmp_tsv, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _write_row(tmp_tsv, status: str = "on_front", val_score: float = 0.5, **kw) -> None:
    write_row(
        tmp_tsv,
        "abc123",
        val_score,
        0.4,
        0.3,
        0.2,
        1.0,
        status,
        "trial write test",
        **kw,
    )


def test_write_row_persists_objective_vector_axes(tmp_path):
    tsv = tmp_path / "results.tsv"
    _write_row(
        tsv,
        ctx=131072,
        tps=30.5,
        agentic=0.6000,
        coding=0.650000,
    )
    row = _read(tsv)[0]
    assert row["ctx"] == "131072"
    assert row["tps"] == "30.5"
    assert row["agentic"] == "0.6000"
    assert row["coding"] == "0.650000"


def test_write_row_preserves_engine_version_tag(tmp_path, monkeypatch):
    """Trial evidence carries engine@tag when the server comes from a fork release."""
    from pathlib import Path

    from autoresearch.runners import run

    tsv = tmp_path / "results.tsv"
    fork = Path("llama.cpp-releases/turboquant/tqp-v0.3.0/build-cuda/bin/llama-server.exe")
    monkeypatch.setattr(run, "resolve_llama_server", lambda: fork)
    _write_row(tsv, ctx=100000, tps=158.5)
    row = _read(tsv)[0]
    assert row["binary_version"] == "turboquant@tqp-v0.3.0"


def test_partial_axes_render_blank(tmp_path):
    tsv = tmp_path / "results.tsv"
    _write_row(tsv, ctx=32768, tps=20.0)  # agentic + coding unmeasured
    row = _read(tsv)[0]
    assert row["ctx"] == "32768"
    assert row["tps"] == "20.0"
    assert row["agentic"] == ""
    assert row["coding"] == ""
    assert row["agentic_coding"] == ""


def test_on_front_persists_literally(tmp_path):
    tsv = tmp_path / "results.tsv"
    _write_row(tsv, status="on_front")
    assert _read(tsv)[0]["status"] == "on_front"


def test_new_trial_statuses_persist_literally(tmp_path):
    tsv = tmp_path / "results.tsv"
    for status in ("dominated", "incomplete", "rejected"):
        _write_row(tsv, status=status)
    persisted = [row["status"] for row in _read(tsv)]
    assert persisted == ["dominated", "incomplete", "rejected"]


def test_legacy_keep_discard_rejected_on_write(tmp_path):
    tsv = tmp_path / "results.tsv"
    for status in ("keep", "discard"):
        with pytest.raises(ValueError, match=status):
            _write_row(tsv, status=status)
    assert not tsv.exists()


def test_invalid_status_rejected(tmp_path):
    tsv = tmp_path / "results.tsv"
    try:
        _write_row(tsv, status="bogus")
    except ValueError as e:
        assert "bogus" in str(e)
    else:
        raise AssertionError("expected ValueError for invalid status")
    assert not tsv.exists()  # nothing written


def test_status_enum_matches_issue_vocabulary():
    assert {
        "on_front",
        "dominated",
        "incomplete",
        "rejected",
    } == TRIAL_STATUSES


def test_previous_best_sees_on_front_rows(tmp_path):
    tsv = tmp_path / "results.tsv"
    _write_row(tsv, status="on_front", val_score=0.42)
    _write_row(tsv, status="dominated", val_score=0.10)
    assert get_previous_best(tsv) == 0.42


def test_previous_best_ignores_legacy_keep_cells(tmp_path):
    """Stale keep cells (no longer recognized) do not count as frontier."""
    tsv = tmp_path / "results.tsv"
    rows = [
        {"trial_id": "a", "model": "m.gguf", "status": "keep", "val_score": "0.30"},
        {"trial_id": "b", "model": "m.gguf", "status": "on_front", "val_score": "0.55"},
        {"trial_id": "c", "model": "m.gguf", "status": "discard", "val_score": "0.90"},
    ]
    with open(tsv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    assert get_previous_best(tsv) == 0.55


def test_write_row_gpu_temp_and_tps_reps(tmp_path):
    tsv = tmp_path / "results.tsv"
    _write_row(tsv, gpu_temp_c=71.0, tps_reps="1,2,3", tps_spread=10.0)
    row = _read(tsv)[0]
    assert row["gpu_temp_c"] == "71.0"
    assert row["tps_reps"] == "1,2,3"
    assert row["tps_spread"] == "10.0"


def test_write_row_missing_hygiene_kwargs_blank(tmp_path):
    tsv = tmp_path / "results.tsv"
    _write_row(tsv)
    row = _read(tsv)[0]
    assert row["gpu_temp_c"] == ""
    assert row["tps_reps"] == ""
    assert row["tps_spread"] == ""


def _read_db(tsv):
    import sqlite3

    db = tsv.with_name("results.db")
    if not db.exists():
        return None
    conn = sqlite3.connect(db)
    try:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(trials)")]
        return cols, conn.execute("SELECT COUNT(*) FROM trials").fetchone()[0]
    finally:
        conn.close()


def test_recompute_statuses_refreshes_mirror(tmp_path, monkeypatch):
    """Every recompute call (write or no-op) leaves the mirror in sync."""
    from autoresearch.runners import run

    monkeypatch.setattr(run, "RESULTS_FILE", tmp_path / "results.tsv")
    tsv = tmp_path / "results.tsv"
    _write_row(tsv, status="on_front")
    run.recompute_statuses(tsv)
    assert _read_db(tsv) is not None
    cols, n = _read_db(tsv)
    assert n == 1 and "trial_id" in cols and "tps" in cols

    # No-op path (statuses unchanged) must ALSO refresh: append a dominated
    # row directly, then recompute — mirror must now hold 2 rows.
    _write_row(tsv, status="dominated")
    run.recompute_statuses(tsv)
    assert _read_db(tsv)[1] == 2


def test_recompute_statuses_survives_mirror_failure(tmp_path, monkeypatch):
    """A broken mirror never breaks the TSV write path."""
    from autoresearch.runners import run

    tsv = tmp_path / "results.tsv"
    _write_row(tsv, status="on_front")
    monkeypatch.setattr(run.results_db, "try_sync_from_tsv", lambda *a, **k: 0)
    run.recompute_statuses(tsv)  # must not raise
    rows = _read(tsv)
    assert len(rows) == 1 and rows[0]["status"] == "on_front"


# ── dual-write contract: canonical SQLite primary, legacy TSV fallback ──


def test_write_row_writes_both_stores(tmp_path):
    tsv = tmp_path / "results.tsv"
    _write_row(tsv, status="on_front", tps=74.9, ctx=131072)
    # Legacy TSV append-log captured the row.
    assert len(_read(tsv)) == 1
    # Canonical DB captured the same row (keyed by trial_id).
    db_rows = run.read_rows(tsv)
    assert len(db_rows) == 1
    assert db_rows[0]["tps"] == "74.9"
    assert db_rows[0]["ctx"] == "131072"


def test_read_rows_survives_missing_db_via_tsv_fallback(tmp_path):
    tsv = tmp_path / "results.tsv"
    _write_row(tsv, status="on_front", tps=50.0)
    (tmp_path / "results.db").unlink()  # simulate lost canonical store
    rows = run.read_rows(tsv)
    assert len(rows) == 1
    assert rows[0]["status"] == "on_front"


def _fp_json() -> str:
    """Minimal valid Baseline fingerprint (same shape as test_recompute.py)."""
    baseline = {
        "model": "M.gguf",
        "ctx_size": 131072,
        "kv_cache": "turbo2",
        "threads": 6,
        "temp": 0.4,
        "top_p": 0.95,
    }
    return json.dumps(baseline, sort_keys=True, separators=(",", ":"))


def test_recompute_statuses_updates_both_stores(tmp_path):
    tsv = tmp_path / "results.tsv"
    # ADR 0017: two models in the same budget bucket never demote each other
    # — both stay on_front in BOTH stores (same-model domination only).
    _write_row(
        tsv,
        status="on_front",
        model="B.gguf",
        tps=100.0,
        ctx=131072,
        agentic=0.9,
        coding=0.9,
        config_json=_fp_json(),
    )
    _write_row(
        tsv,
        status="on_front",
        model="M.gguf",
        tps=50.0,
        ctx=131072,
        agentic=0.5,
        coding=0.5,
        config_json=_fp_json(),
    )
    run.recompute_statuses(tsv)
    tsv_statuses = sorted(r["status"] for r in _read(tsv))
    db_statuses = sorted(r["status"] for r in run.read_rows(tsv))
    assert tsv_statuses == db_statuses == ["on_front", "on_front"]
