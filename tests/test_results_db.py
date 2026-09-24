"""Tests for the derived SQLite mirror of results.tsv (results_db)."""

from __future__ import annotations

import csv
import sqlite3

from autoresearch.core import results_db

NUMERIC_COLS = [
    "val_score",
    "tps",
    "ctx",
    "agentic",
    "coding",
    "mini_swe_agent",
    "memory_gb",
]


def _conn(tmp_path):
    conn = sqlite3.connect(tmp_path / "results.db")
    results_db.ensure_schema(conn)
    return conn


def test_schema_creates_trials_table(tmp_path):
    conn = _conn(tmp_path)
    names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "trials" in names


def test_numeric_columns_are_real(tmp_path):
    conn = _conn(tmp_path)
    cols = {r[1]: r[2] for r in conn.execute("PRAGMA table_info(trials)")}
    for c in NUMERIC_COLS:
        assert cols[c].upper() == "REAL", c


def test_trial_id_is_primary_key(tmp_path):
    conn = _conn(tmp_path)
    pk = [r[1] for r in conn.execute("PRAGMA table_info(trials)") if r[5]]
    assert pk == ["trial_id"]


def test_to_cell_coercion():
    assert results_db._to_cell("tps", "27.8") == 27.8
    assert results_db._to_cell("tps", "") is None
    assert results_db._to_cell("tps", None) is None
    assert results_db._to_cell("model", "Ornith-35B") == "Ornith-35B"
    assert results_db._to_cell("status", "on_front") == "on_front"


def _row(**overrides):
    base = {
        c: ""
        for c in (
            "schema_version",
            "trial_id",
            "commit",
            "model",
            "backend",
            "status",
            "val_score",
            "tps",
            "ctx",
            "agentic",
            "coding",
            "description",
        )
    }
    base.update(
        trial_id="t-0001",
        model="Ornith-35B",
        status="on_front",
        val_score="0.57",
        tps="27.8",
        ctx="32768",
    )
    base.update(overrides)
    return base


def _write_tsv(path, rows):
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        w.writeheader()
        w.writerows(rows)


def test_replace_all_inserts_and_is_idempotent(tmp_path):
    conn = _conn(tmp_path)
    rows = [_row(), _row(trial_id="t-0002", model="Mythos", status="dominated")]
    results_db.replace_all(conn, rows)
    results_db.replace_all(conn, rows)  # rerun changes nothing, no dupes
    assert conn.execute("SELECT COUNT(*) FROM trials").fetchone()[0] == 2


def test_upsert_row_updates_existing_trial(tmp_path):
    conn = _conn(tmp_path)
    results_db.replace_all(conn, [_row()])
    results_db.upsert_row(conn, _row(status="dominated"))
    got = conn.execute("SELECT status FROM trials WHERE trial_id='t-0001'").fetchone()[0]
    assert got == "dominated"


def test_numeric_values_stored_typed(tmp_path):
    conn = _conn(tmp_path)
    results_db.replace_all(conn, [_row()])
    tps, ctx, model = conn.execute(
        "SELECT tps, ctx, model FROM trials WHERE trial_id='t-0001'"
    ).fetchone()
    assert tps == 27.8 and isinstance(tps, float)
    assert ctx == 32768.0
    assert model == "Ornith-35B"


def test_mini_swe_agent_values_are_typed_and_detail_is_text(tmp_path):
    conn = _conn(tmp_path)
    results_db.upsert_row(
        conn,
        _row(mini_swe_agent="0.7500", mini_swe_agent_detail="suite=abc; task=pass"),
    )
    value, detail = conn.execute(
        "SELECT mini_swe_agent, mini_swe_agent_detail FROM trials WHERE trial_id='t-0001'"
    ).fetchone()
    assert value == 0.75
    assert detail == "suite=abc; task=pass"

    tsv = tmp_path / "results.tsv"
    _write_tsv(tsv, [_row(), _row(trial_id="t-0002")])
    n = results_db.sync_from_tsv(tsv, tmp_path / "results.db")
    assert n == 2
    conn = sqlite3.connect(tmp_path / "results.db")
    assert conn.execute("SELECT COUNT(*) FROM trials").fetchone()[0] == 2


# ── canonical-first store contract (SQLite primary, legacy TSV fallback) ──


def test_read_rows_round_trips_writer_text(tmp_path):
    tsv = tmp_path / "results.tsv"
    _write_tsv(tsv, [_row(tps="27.8", ctx="32768", val_score="0.570000")])
    db = tmp_path / "results.db"
    results_db.sync_from_tsv(tsv, db)
    rows = results_db.read_rows(db)
    assert rows[0]["tps"] == "27.8"
    assert rows[0]["ctx"] == "32768"
    assert rows[0]["val_score"] == "0.570000"
    assert rows[0]["model"] == "Ornith-35B"


def test_store_rows_falls_back_to_tsv_when_db_unseeded(tmp_path):
    tsv = tmp_path / "results.tsv"
    _write_tsv(tsv, [_row()])
    rows, source = results_db.store_rows(tsv, tmp_path / "results.db")
    assert source == "tsv"
    assert rows[0]["trial_id"] == "t-0001"


def test_store_rows_prefers_db_over_stale_tsv(tmp_path):
    tsv = tmp_path / "results.tsv"
    _write_tsv(tsv, [_row(status="dominated")])
    db = tmp_path / "results.db"
    results_db.sync_from_tsv(tsv, db)
    # TSV now edited to a stale status; DB must win.
    _write_tsv(tsv, [_row(status="on_front")])
    rows, source = results_db.store_rows(tsv, db)
    assert source == "db"
    assert rows[0]["status"] == "dominated"


def _legacy_conn(tmp_path):
    """Connection with the pre-reasoning-column schema (pre-2026-08-29 layout)."""
    conn = sqlite3.connect(tmp_path / "results.db")
    legacy_cols = [
        c
        for c in results_db._COLUMNS
        if c
        not in (
            "reasoning_budget",
            "reasoning_effort",
            "mini_swe_agent",
            "mini_swe_agent_detail",
        )
    ]
    cols_sql = ",\n  ".join(
        f"{results_db._q(c)} {'REAL' if c in results_db._NUMERIC_COLUMNS else 'TEXT'}"
        + (" PRIMARY KEY" if c == "trial_id" else "")
        for c in legacy_cols
    )
    conn.execute(f"CREATE TABLE trials (\n  {cols_sql}\n)")
    return conn


def test_ensure_schema_migrates_legacy_db_without_reasoning_or_mini_columns(tmp_path):
    conn = _legacy_conn(tmp_path)
    conn.execute(
        f"INSERT INTO trials ({results_db._q('trial_id')}, {results_db._q('model')}) VALUES (?, ?)",
        ("t-legacy", "Ornith-35B"),
    )
    conn.commit()
    results_db.ensure_schema(conn)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(trials)")}
    assert "reasoning_budget" in cols
    assert "reasoning_effort" in cols
    assert "mini_swe_agent" in cols
    assert "mini_swe_agent_detail" in cols


def test_backfill_reasoning_columns_from_config_json(tmp_path):
    conn = _legacy_conn(tmp_path)
    rows = [
        ("t-budget", '{"kv":"q4_0","reasoning_budget":4096,"reasoning":null}'),
        ("t-upper", '{"REASONING_BUDGET":8192,"REASONING_EFFORT":"low"}'),
        ("t-plain", '{"kv":"q4_0","reasoning_budget":null,"reasoning":null}'),
    ]
    for trial_id, cfg in rows:
        conn.execute(
            f"INSERT INTO trials ({results_db._q('trial_id')}, {results_db._q('config_json')}) VALUES (?, ?)",
            (trial_id, cfg),
        )
    conn.commit()
    results_db.ensure_schema(conn)
    assert (
        conn.execute("SELECT reasoning_budget FROM trials WHERE trial_id = 't-budget'").fetchone()[
            0
        ]
        == 4096
    )
    assert conn.execute(
        "SELECT reasoning_budget, reasoning_effort FROM trials WHERE trial_id = 't-upper'"
    ).fetchone() == (8192, "low")
    plain = conn.execute(
        "SELECT reasoning_budget, reasoning_effort FROM trials WHERE trial_id = 't-plain'"
    ).fetchone()
    assert plain[0] is None and plain[1] is None
    # Marker: a second ensure_schema is a no-op (NULLs stay NULL, no resurrection).
    conn.execute("UPDATE trials SET reasoning_budget = NULL")
    results_db.ensure_schema(conn)
    assert (
        conn.execute("SELECT reasoning_budget FROM trials WHERE trial_id = 't-budget'").fetchone()[
            0
        ]
        is None
    )


def test_upsert_derives_reasoning_columns_from_config_json(tmp_path):
    conn = _legacy_conn(tmp_path)
    results_db.ensure_schema(conn)
    row = {
        "trial_id": "t-1",
        "config_json": '{"reasoning_budget":2048,"reasoning_effort":"low"}',
    }
    results_db.upsert_rows(tmp_path / "results.db", [row])
    got = conn.execute(
        "SELECT reasoning_budget, reasoning_effort FROM trials WHERE trial_id = 't-1'"
    ).fetchone()
    assert got == (2048, "low")
