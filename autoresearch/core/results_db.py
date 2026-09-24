"""Canonical SQLite results store with legacy TSV fallback.

``results.db`` is the ground-truth Trial store (indexed, typed columns);
``results.tsv`` is the legacy append-log kept in sync as a fallback. Reads
prefer the DB and fall back to the TSV when it is missing or unseeded
(:func:`load_rows` / :func:`store_rows`). Writes go to both (:func:`upsert_rows`
primary, TSV append best-effort in ``run.write_row``). Either store can be
rebuilt from the other: :func:`sync_from_tsv` seeds the DB, :func:`sync_to_tsv`
rewrites the legacy log.
"""

from __future__ import annotations

import csv
import json
import os
import sqlite3
import tempfile
from pathlib import Path

# Columns stored as REAL (blank -> NULL). Everything else TEXT.
_NUMERIC_COLUMNS = frozenset(
    {
        "val_score",
        "swe_score",
        "lcb_score",
        "he_score",
        "mbpp_score",
        "bigcode_score",
        "agentic",
        "coding",
        "agentic_coding",
        "mini_swe_agent",
        "memory_gb",
        "elapsed_sec",
        "tps",
        "bench_tg",
        "ctx",
        "threads",
        "threads_batch",
        "batch_size",
        "ubatch_size",
        "n_cpu_moe",
        "temp",
        "top_p",
        "top_k",
        "min_p",
        "repeat_penalty",
        "presence_penalty",
        "reasoning_budget",
        "gpu_temp_c",
        "tps_spread",
    }
)

_COLUMNS: list[str] = [
    "schema_version",
    "trial_id",
    "commit",
    "model",
    "model_id",
    "backend",
    "category",
    "evaluation_profile",
    "scoring_benchmark",
    "outcome",
    "diagnostic",
    "status",
    "val_score",
    "swe_score",
    "lcb_score",
    "he_score",
    "mbpp_score",
    "bigcode_score",
    "agentic",
    "coding",
    "agentic_coding",
    "mini_swe_agent",
    "mini_swe_agent_detail",
    "memory_gb",
    "elapsed_sec",
    "tps",
    "bench_tg",
    "kv",
    "ctx",
    "threads",
    "threads_batch",
    "batch_size",
    "ubatch_size",
    "n_cpu_moe",
    "temp",
    "top_p",
    "top_k",
    "min_p",
    "repeat_penalty",
    "presence_penalty",
    "cont_batching",
    "flash_attn",
    "no_mmap",
    "spec_draft_n_max",
    "task_ids",
    "random_seed",
    "config_json",
    "binary_version",
    "tps_source",
    "gpu_temp_c",
    "tps_reps",
    "tps_spread",
    "reasoning_budget",
    "reasoning_effort",
    "description",
]


def _to_cell(column: str, raw: str | None) -> float | str | None:
    """Blank -> NULL; numeric columns coerced to float; others verbatim."""
    if raw is None or raw == "":
        return None
    if column in _NUMERIC_COLUMNS:
        try:
            return float(raw)
        except ValueError:
            return None
    return raw


def default_db_path(results_file: Path) -> Path:
    """Mirror lives next to its TSV: results.tsv -> results.db."""
    return Path(results_file).with_name("results.db")


def ensure_schema(conn: sqlite3.Connection) -> None:
    cols = ",\n  ".join(
        f"{_q(c)} {'REAL' if c in _NUMERIC_COLUMNS else 'TEXT'}"
        + (" PRIMARY KEY" if c == "trial_id" else "")
        for c in _COLUMNS
    )
    conn.execute(f"CREATE TABLE IF NOT EXISTS trials (\n  {cols}\n)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_trials_model ON trials(model)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_trials_status ON trials(status)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_trials_model_status ON trials(model, status)")
    _migrate_columns(conn)
    _backfill_reasoning_columns(conn)
    conn.commit()


# Columns added after the initial schema; legacy DBs migrate via ALTER TABLE.
_MIGRATED_COLUMNS: dict[str, str] = {
    "reasoning_budget": "REAL",
    "reasoning_effort": "TEXT",
    "mini_swe_agent": "REAL",
    "mini_swe_agent_detail": "TEXT",
}
_BACKFILL_MARKER_KEY = "reasoning_columns_backfill_v1"


def _migrate_columns(conn: sqlite3.Connection) -> None:
    existing = {r[1] for r in conn.execute("PRAGMA table_info(trials)")}
    for name, decl in _MIGRATED_COLUMNS.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE trials ADD COLUMN {_q(name)} {decl}")


def _config_reasoning(cfg: dict) -> tuple[object, object]:
    """Reasoning knobs from a config_json dict (lowercase, legacy uppercase)."""
    budget = cfg.get("reasoning_budget", cfg.get("REASONING_BUDGET"))
    effort = cfg.get("reasoning_effort", cfg.get("REASONING_EFFORT"))
    return budget, effort


def _backfill_reasoning_columns(conn: sqlite3.Connection) -> None:
    """One-shot: legacy rows kept the reasoning knobs only inside config_json.

    The marker table makes reruns a no-op; rows inserted through ``_cells``
    derive the columns themselves, so only pre-existing rows need this pass.
    """
    conn.execute("CREATE TABLE IF NOT EXISTS schema_meta (key TEXT PRIMARY KEY, value TEXT)")
    done = conn.execute(
        "SELECT 1 FROM schema_meta WHERE key = ?", (_BACKFILL_MARKER_KEY,)
    ).fetchone()
    if done:
        return
    rows = conn.execute("SELECT trial_id, config_json FROM trials").fetchall()
    for trial_id, cfg_json in rows:
        try:
            cfg = json.loads(cfg_json) if cfg_json else {}
        except ValueError:
            continue
        if not isinstance(cfg, dict):
            continue
        budget, effort = _config_reasoning(cfg)
        if budget is not None or effort is not None:
            conn.execute(
                "UPDATE trials SET reasoning_budget = ?, reasoning_effort = ? WHERE trial_id = ?",
                (budget, effort, trial_id),
            )
    conn.execute(
        "INSERT OR REPLACE INTO schema_meta (key, value) VALUES (?, 'done')",
        (_BACKFILL_MARKER_KEY,),
    )


def _q(name: str) -> str:
    """Quote an identifier — `commit` is a SQLite reserved word."""
    return f'"{name}"'


def _insert_sql() -> str:
    placeholders = ", ".join("?" for _ in _COLUMNS)
    cols = ", ".join(_q(c) for c in _COLUMNS)
    return f"INSERT OR REPLACE INTO trials ({cols}) VALUES ({placeholders})"


def _cells(row: dict) -> list:
    row = _derive_reasoning_cells(row)
    return [_to_cell(c, row.get(c)) for c in _COLUMNS]


def _derive_reasoning_cells(row: dict) -> dict:
    """Fill missing reasoning columns from the row's config_json (legacy rows)."""
    budget = row.get("reasoning_budget")
    effort = row.get("reasoning_effort")
    if budget not in (None, "") and effort not in (None, ""):
        return row
    try:
        cfg = json.loads(row.get("config_json") or "{}")
    except ValueError:
        return row
    if not isinstance(cfg, dict):
        return row
    out = dict(row)
    cfg_budget, cfg_effort = _config_reasoning(cfg)
    if budget in (None, "") and cfg_budget is not None:
        out["reasoning_budget"] = cfg_budget
    if effort in (None, "") and cfg_effort is not None:
        out["reasoning_effort"] = cfg_effort
    return out


def replace_all(conn: sqlite3.Connection, rows: list[dict]) -> int:
    """Transactional wipe + bulk insert from TSV-shaped dict rows."""
    with conn:  # implicit BEGIN/COMMIT; rolls back on error
        conn.execute("DELETE FROM trials")
        conn.executemany(_insert_sql(), (_cells(r) for r in rows))
    return len(rows)


def upsert_row(conn: sqlite3.Connection, row: dict) -> None:
    with conn:
        conn.execute(_insert_sql(), _cells(row))


def _read_tsv(results_file: Path) -> list[dict]:
    """Minimal local reader — avoids importing runners (keeps core leaf-pure)."""
    if not results_file.exists() or results_file.stat().st_size == 0:
        return []
    with open(results_file, encoding="utf-8") as f:
        return [dict(r) for r in csv.DictReader(f, delimiter="\t")]


def sync_from_tsv(results_file: Path, db_path: Path | None = None) -> int:
    """Seed / repair the canonical DB from the legacy TSV. Returns row count."""
    db_path = db_path or default_db_path(Path(results_file))
    rows = _read_tsv(Path(results_file))
    conn = sqlite3.connect(db_path)
    try:
        ensure_schema(conn)
        return replace_all(conn, rows)
    finally:
        conn.close()


def try_sync_from_tsv(results_file: Path, db_path: Path | None = None) -> int:
    """Best-effort DB seed/heal from the legacy TSV: never raises, logs.

    Used for one-shot seeding when the canonical DB is missing and for
    healing after a failed DB write (the TSV mirror holds the rows).
    """
    try:
        return sync_from_tsv(results_file, db_path)
    except Exception as exc:  # mirror must never break a Trial write
        print(f"[results-db] mirror sync failed (TSV unaffected): {exc}")
        return 0


def parity_check(results_file: Path, db_path: Path | None = None) -> tuple[bool, str]:
    """Compare the canonical DB vs the legacy TSV: row count + trial_id set.

    A missing or un-migrated DB (no `trials` table) is drift, not a
    crash: reports (False, reason) so callers can rebuild.
    """
    db_path = db_path or default_db_path(Path(results_file))
    rows = _read_tsv(Path(results_file))
    if not db_path.exists():
        return False, f"canonical DB missing: {db_path}"
    conn = sqlite3.connect(db_path)
    try:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "trials" not in tables:
            return False, "mirror has no 'trials' table (un-migrated or corrupt)"
        db_ids = {r[0] for r in conn.execute("SELECT trial_id FROM trials")}
    finally:
        conn.close()
    tsv_ids = {r.get("trial_id", "") for r in rows}
    problems: list[str] = []
    if len(rows) != len(tsv_ids):
        problems.append(f"duplicate trial_id in TSV ({len(rows)} rows, {len(tsv_ids)} ids)")
    missing = sorted(tsv_ids - db_ids)
    extra = sorted(db_ids - tsv_ids)
    if missing:
        problems.append(f"in TSV, not in mirror: {missing[:5]}{'…' if len(missing) > 5 else ''}")
    if extra:
        problems.append(f"in mirror, not in TSV: {extra[:5]}{'…' if len(extra) > 5 else ''}")
    if problems:
        return False, "; ".join(problems)
    return True, f"parity OK: {len(rows)} rows"


# Text formats mirroring the write path (run.write_row cell formatting), so DB
# reads round-trip to the same text the TSV writer produced.
_TEXT_FMT: dict[str, str] = {
    "val_score": "{:.6f}",
    "swe_score": "{:.6f}",
    "lcb_score": "{:.6f}",
    "he_score": "{:.6f}",
    "mbpp_score": "{:.6f}",
    "bigcode_score": "{:.6f}",
    "coding": "{:.6f}",
    "agentic": "{:.4f}",
    "agentic_coding": "{:.4f}",
    "mini_swe_agent": "{:.4f}",
    "memory_gb": "{:.1f}",
    "tps": "{:.1f}",
    "bench_tg": "{:.1f}",
    "elapsed_sec": "{:.0f}",
}
_INT_COLUMNS = frozenset(
    {
        "ctx",
        "threads",
        "threads_batch",
        "batch_size",
        "ubatch_size",
        "n_cpu_moe",
        "top_k",
        "spec_draft_n_max",
        "reasoning_budget",
    }
)


def _from_cell(column: str, value: float | str | None) -> str:
    """DB cell -> TSV text: NULL -> blank; floats via the writer's format."""
    if value is None:
        return ""
    if isinstance(value, float):
        if column in _INT_COLUMNS:
            return str(int(value))
        fmt = _TEXT_FMT.get(column)
        return fmt.format(value) if fmt else str(value)
    return value


def read_rows(db_path: Path) -> list[dict[str, str]] | None:
    """All trials as TSV-shaped dict rows from the canonical store.

    None when the DB is missing or un-migrated (no ``trials`` table) —
    callers fall back to the legacy TSV. Legacy DBs missing derived columns
    (reasoning_budget / reasoning_effort) are migrated in place on first
    read (ALTER + one-shot config_json backfill), so the SELECT below always
    sees every column. Numeric cells read back as their TSV text form
    (integral floats as ints, blanks as "").
    """
    db_path = Path(db_path)
    if not db_path.exists():
        return None
    conn = sqlite3.connect(db_path)
    try:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "trials" not in tables:
            return None
        ensure_schema(conn)
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            f"SELECT {', '.join(_q(c) for c in _COLUMNS)} FROM trials ORDER BY rowid"
        )
        rows = cursor.fetchall()
    finally:
        conn.close()
    return [{c: _from_cell(c, row[c]) for c in _COLUMNS} for row in rows]


def upsert_rows(db_path: Path, rows: list[dict]) -> None:
    """Upsert TSV-shaped dict rows into the canonical store (own connection)."""
    conn = sqlite3.connect(db_path)
    try:
        ensure_schema(conn)
        with conn:
            conn.executemany(_insert_sql(), (_cells(r) for r in rows))
    finally:
        conn.close()


def store_rows(results_file: Path, db_path: Path | None = None) -> tuple[list[dict[str, str]], str]:
    """Canonical-first store read with legacy TSV fallback.

    Returns ``(rows, source)`` where source is ``"db"`` or ``"tsv"`` so
    callers know which store needs backfill after a read-modify-rewrite.
    An existing-but-empty DB over a non-empty TSV counts as unseeded and
    falls back (a fresh DB must not hide legacy rows).
    """
    db_path = db_path or default_db_path(Path(results_file))
    rows = read_rows(db_path)
    if rows is None or not rows:
        tsv_rows = _read_tsv(Path(results_file))
        if rows is None or tsv_rows:
            return tsv_rows, "tsv"
    return rows, "db"


def load_rows(results_file: Path, db_path: Path | None = None) -> list[dict[str, str]]:
    """Store rows via :func:`store_rows`, dropping the source tag."""
    return store_rows(results_file, db_path)[0]


def sync_to_tsv(results_file: Path, db_path: Path | None = None) -> int:
    """Rewrite the legacy TSV from the canonical store (atomic replace)."""
    db_path = db_path or default_db_path(Path(results_file))
    rows = read_rows(db_path)
    if rows is None:
        raise FileNotFoundError(f"canonical store unavailable: {db_path}")
    results_file = Path(results_file)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            newline="",
            encoding="utf-8",
            dir=results_file.parent,
            prefix=f".{results_file.name}.",
            suffix=".tmp",
            delete=False,
        ) as f:
            temp_path = Path(f.name)
            writer = csv.DictWriter(f, fieldnames=_COLUMNS, delimiter="\t", extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temp_path, results_file)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
    return len(rows)


def try_sync_to_tsv(results_file: Path, db_path: Path | None = None) -> int:
    """Best-effort legacy-TSV rewrite: never raises, logs on failure."""
    try:
        return sync_to_tsv(results_file, db_path)
    except Exception as exc:  # legacy mirror must never break the store
        print(f"[results-db] legacy TSV rewrite failed (DB unaffected): {exc}")
        return 0
