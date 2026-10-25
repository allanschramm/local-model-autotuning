"""Trial persistence shared by tuning, legacy evaluation, scripts, and read-only UI."""

from __future__ import annotations

import contextlib
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

from autoresearch.core import recompute, results_db
from autoresearch.core.evaluation_protocol import CODING_PROTOCOL
from autoresearch.core.llama_runner import engine_version_tag, resolve_llama_server
from autoresearch.core.process_guard import spawn_kwargs

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl
RESULTS_FILE = Path(__file__).resolve().parents[2] / "results.tsv"
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
TRIAL_STATUSES = {"on_front", "dominated", "incomplete", "rejected"}


def tsv_fields_from_cfg(baseline: dict[str, Any]) -> dict[str, Any]:
    """Map a Baseline-style config dict to write_row flat-column kwargs."""
    recorded = {key.lower(): value for key, value in baseline.items()}
    recorded["coding_protocol"] = CODING_PROTOCOL
    return {
        "model": baseline.get("MODEL", ""),
        "kv": baseline.get("KV_CACHE", ""),
        "ctx": baseline.get("CTX_SIZE"),
        "threads": baseline.get("THREADS"),
        "threads_batch": baseline.get("THREADS_BATCH"),
        "batch_size": baseline.get("BATCH_SIZE"),
        "ubatch_size": baseline.get("UBATCH_SIZE"),
        "n_cpu_moe": baseline.get("N_CPU_MOE"),
        "temp": baseline.get("TEMP"),
        "top_p": baseline.get("TOP_P"),
        "top_k": baseline.get("TOP_K"),
        "min_p": baseline.get("MIN_P"),
        "repeat_penalty": baseline.get("REPEAT_PENALTY"),
        "presence_penalty": baseline.get("PRESENCE_PENALTY"),
        "cont_batching": baseline.get("CONT_BATCHING"),
        "flash_attn": baseline.get("FLASH_ATTN", ""),
        "no_mmap": baseline.get("NO_MMAP"),
        "spec_draft_n_max": baseline.get("SPEC_DRAFT_N_MAX"),
        "reasoning_budget": baseline.get("REASONING_BUDGET"),
        "reasoning_effort": baseline.get("REASONING_EFFORT"),
        "config_json": json.dumps(recorded, separators=(",", ":"), sort_keys=True),
    }


def get_git_commit() -> str:
    try:
        commit = (
            subprocess.check_output(
                **spawn_kwargs(
                    args=["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
                )
            )
            .decode("utf-8")
            .strip()
        )
        status = (
            subprocess.check_output(
                **spawn_kwargs(args=["git", "status", "--porcelain"], stderr=subprocess.DEVNULL)
            )
            .decode("utf-8")
            .strip()
        )
        if status:
            commit += "-dirty"
        return commit
    except Exception:
        return "unknown"


def read_rows(results_file: Path) -> list[dict[str, str]]:
    """All store rows as dicts ([] when missing or empty).

    Reads the canonical SQLite store (``results.db``) first; falls back to
    the legacy ``results.tsv`` append-log when the DB is missing or unseeded.
    """
    return results_db.load_rows(results_file)


@contextlib.contextmanager
def _results_lock(results_file: Path):
    """Cross-process mutex serializing results-store writes (DB upsert + legacy TSV append/rewrite)."""
    lock_path = results_file.with_name(results_file.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    # os.open so tests that patch builtins.open do not substitute the lock fd.
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    fd = os.open(lock_path, flags)
    try:
        if sys.platform == "win32":
            # Ensure the lockfile has at least one byte so msvcrt.locking has a
            # region to lock.
            size = os.lseek(fd, 0, os.SEEK_END)
            if size == 0:
                os.write(fd, b"\n")
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
        else:
            fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            yield
        finally:
            if sys.platform == "win32":
                try:
                    os.lseek(fd, 0, os.SEEK_SET)
                    msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
                except OSError:
                    pass
            else:
                fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def recompute_statuses(results_file: Path, *, scope: str = recompute.DEFAULT_SCOPE) -> None:
    """Store-wide status refresh after a Trial write (issue #5; ADR 0017).

    Domination is same-model only: recompute groups rows by basename ×
    budget bucket — every complete basename vector is on_front, partial
    vectors stay incomplete, and legacy cross-model ``dominated`` labels
    flip to on_front. rejected rows are left out; fingerprint-less legacy
    rows without a config_json fingerprint are untouched. Idempotent:
    rerunning changes nothing, so a no-change store is not rewritten.

    Reads the canonical SQLite store first (legacy TSV when unseeded).
    Refreshed statuses are written to the DB (primary) and the TSV is
    rewritten as the legacy mirror. A missing DB is seeded from the TSV
    (one-shot migration).
    """
    with _results_lock(results_file):
        rows, source = results_db.store_rows(results_file)
        updated = recompute.recompute_rows(rows, scope=scope)
        if updated != rows:
            if source == "db":
                changed = [u for u, o in zip(updated, rows, strict=False) if u != o]
                try:
                    results_db.upsert_rows(results_db.default_db_path(results_file), changed)
                except Exception as exc:
                    print(f"[results] DB status update failed, healing from TSV: {exc}")
                    results_db.try_sync_from_tsv(results_file)
            try:
                _replace_tsv(results_file, updated)
            except Exception as exc:
                print(f"[results] legacy TSV rewrite failed: {exc}")
        if source == "tsv":
            # One-shot migration: seed the canonical store from legacy TSV.
            results_db.try_sync_from_tsv(results_file)


def relabel_watchdog_kills(results_file: Path) -> int:
    """One-shot issue #72 migration: legacy policy kills → WATCHDOG_KILL.

    Rewrites rows stored as MODEL_REJECTED/VRAM_LIMIT_EXCEEDED (all pre-fix
    watchdog kills were device-wide NVML policy kills) to the honest
    WATCHDOG_KILL outcome. Store status stays ``rejected``. Idempotent:
    returns the relabeled row count (0 = nothing to do).
    """
    with _results_lock(results_file):
        rows, source = results_db.store_rows(results_file)
        updated = recompute.relabel_watchdog_kills(rows)
        changed = [u for u, o in zip(updated, rows, strict=False) if u != o]
        if not changed:
            return 0
        if source == "db":
            try:
                results_db.upsert_rows(results_db.default_db_path(results_file), changed)
            except Exception as exc:
                print(f"[results] DB relabel failed, healing from TSV: {exc}")
                results_db.try_sync_from_tsv(results_file)
        try:
            _replace_tsv(results_file, updated)
        except Exception as exc:
            print(f"[results] legacy TSV rewrite failed: {exc}")
        if source == "tsv":
            results_db.try_sync_from_tsv(results_file)
        return len(changed)


def _replace_tsv(results_file: Path, rows: list[dict[str, str]]) -> None:
    """Atomically rewrite the legacy TSV (a killed Search must not truncate it)."""
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
            writer = csv.DictWriter(
                f, fieldnames=CATEGORY_FIELDNAMES, delimiter="\t", extrasaction="ignore"
            )
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temp_path, results_file)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


CATEGORY_FIELDNAMES = [
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


def _ensure_category_column(results_file: Path) -> None:
    """One-time migration: add missing columns."""
    if not results_file.exists() or results_file.stat().st_size == 0:
        return
    with open(results_file, encoding="utf-8") as f:
        header = f.readline().strip()
    cols = header.split("\t")
    if cols == CATEGORY_FIELDNAMES:
        return  # already migrated
    backup = results_file.with_suffix(results_file.suffix + ".bak")
    if results_file.is_file() and not backup.exists():
        shutil.copy2(results_file, backup)
    rows = []
    with open(results_file, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(row)
    with open(results_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=CATEGORY_FIELDNAMES, delimiter="\t", extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


def _tsv_cell(value: Any, fmt: str | None = None) -> str:
    """Format a TSV cell; None stays blank, zeros stay zeros."""
    if value is None:
        return ""
    return format(value, fmt) if fmt else str(value)


def _engine_version() -> str:
    """Engine identity for the Trial evidence (binary_version cell).

    Resolves the engine actually used; falls back to empty when no
    llama-server is resolvable (e.g. pure row-serialization contexts).
    """
    try:
        return engine_version_tag(resolve_llama_server())
    except FileNotFoundError:
        return ""


def write_row(
    results_file: Path,
    commit: str,
    val_score: float,
    swe_score: float,
    he_score: float,
    mbpp_score: float,
    memory_gb: float,
    status: str,
    description: str,
    lcb_score: float = 0.0,
    bigcode_score: float = 0.0,
    agentic: float | None = None,
    coding: float | None = None,
    agentic_coding: float | None = None,
    mini_swe_agent: float | None = None,
    mini_swe_agent_detail: str = "",
    category: str = "",
    elapsed_sec: float = 0.0,
    model: str = "",
    tps: float | None = None,
    bench_tg: float | None = None,
    kv: str = "",
    ctx: int | None = None,
    threads: int | None = None,
    threads_batch: int | None = None,
    batch_size: int | None = None,
    ubatch_size: int | None = None,
    n_cpu_moe: int | None = None,
    temp: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    min_p: float | None = None,
    repeat_penalty: float | None = None,
    presence_penalty: float | None = None,
    cont_batching: Any = None,
    flash_attn: str = "",
    no_mmap: Any = None,
    spec_draft_n_max: int | None = None,
    outcome: str = "",
    diagnostic: str = "",
    evaluation_profile: str = "",
    scoring_benchmark: str = "",
    task_ids: str = "",
    config_json: str = "",
    tps_source: str = "",
    gpu_temp_c: float | None = None,
    tps_reps: str = "",
    tps_spread: float | None = None,
    reasoning_budget: int | None = None,
    reasoning_effort: str | None = None,
):
    _ensure_category_column(results_file)
    if status not in TRIAL_STATUSES:
        raise ValueError(f"invalid trial status: {status!r}; allowed: {sorted(TRIAL_STATUSES)}")
    row = {
        "schema_version": "2",
        "trial_id": str(uuid.uuid4()),
        "commit": commit,
        "model": model,
        "model_id": model,
        "backend": "sglang" if model and not model.lower().endswith(".gguf") else "llama.cpp",
        "category": category,
        "evaluation_profile": evaluation_profile or category,
        "scoring_benchmark": scoring_benchmark
        or (
            "mini-swe-agent"
            if category == "mini-swe-agent"
            else "agentic-coding"
            if category == "agentic-coding"
            else "claw-eval"
            if category.startswith("agentic")
            else "coding"
        ),
        "outcome": outcome or ("OK" if not status.lower().startswith("fail") else "MODEL_REJECTED"),
        "diagnostic": diagnostic,
        "status": status,
        "val_score": f"{val_score:.6f}",
        "swe_score": f"{swe_score:.6f}",
        "lcb_score": f"{lcb_score:.6f}",
        "he_score": f"{he_score:.6f}",
        "mbpp_score": f"{mbpp_score:.6f}",
        "bigcode_score": f"{bigcode_score:.6f}",
        "agentic": _tsv_cell(agentic, ".4f"),
        "coding": _tsv_cell(coding, ".6f"),
        "agentic_coding": _tsv_cell(agentic_coding, ".4f"),
        "mini_swe_agent": _tsv_cell(mini_swe_agent, ".4f"),
        "mini_swe_agent_detail": mini_swe_agent_detail,
        "memory_gb": f"{memory_gb:.1f}",
        "elapsed_sec": f"{elapsed_sec:.0f}",
        "tps": _tsv_cell(tps, ".1f"),
        "bench_tg": _tsv_cell(bench_tg, ".1f"),
        "kv": kv,
        "ctx": _tsv_cell(ctx),
        "threads": _tsv_cell(threads),
        "threads_batch": _tsv_cell(threads_batch),
        "batch_size": _tsv_cell(batch_size),
        "ubatch_size": _tsv_cell(ubatch_size),
        "n_cpu_moe": _tsv_cell(n_cpu_moe),
        "temp": _tsv_cell(temp),
        "top_p": _tsv_cell(top_p),
        "top_k": _tsv_cell(top_k),
        "min_p": _tsv_cell(min_p),
        "repeat_penalty": _tsv_cell(repeat_penalty),
        "presence_penalty": _tsv_cell(presence_penalty),
        "cont_batching": _tsv_cell(cont_batching),
        "flash_attn": flash_attn,
        "no_mmap": _tsv_cell(no_mmap),
        "spec_draft_n_max": _tsv_cell(spec_draft_n_max),
        "task_ids": task_ids,
        "random_seed": "",
        "config_json": config_json
        or json.dumps(
            {
                "kv": kv,
                "ctx": ctx,
                "threads": threads,
                "threads_batch": threads_batch,
                "batch_size": batch_size,
                "ubatch_size": ubatch_size,
                "flash_attn": flash_attn,
                "spec_draft_n_max": spec_draft_n_max,
            },
            separators=(",", ":"),
            sort_keys=True,
        ),
        "binary_version": _engine_version(),
        "tps_source": tps_source,
        "gpu_temp_c": _tsv_cell(gpu_temp_c),
        "tps_reps": tps_reps,
        "tps_spread": _tsv_cell(tps_spread),
        "reasoning_budget": _tsv_cell(reasoning_budget),
        "reasoning_effort": reasoning_effort or "",
        "description": description,
    }
    recorded = json.loads(row["config_json"])
    recorded.setdefault("coding_protocol", CODING_PROTOCOL)
    row["config_json"] = json.dumps(recorded, separators=(",", ":"), sort_keys=True)
    with _results_lock(results_file):
        # Canonical store first: SQLite upsert. On failure the legacy TSV
        # append below still captures the row (heal via try_sync_from_tsv).
        db_ok = True
        try:
            results_db.upsert_rows(results_db.default_db_path(results_file), [row])
        except Exception as exc:
            db_ok = False
            print(f"[results] canonical DB write failed ({exc}); logging to legacy TSV only")
        # Legacy append-log: kept in sync as the fallback store (best-effort).
        try:
            new_file = not results_file.exists() or results_file.stat().st_size == 0
            with open(results_file, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=CATEGORY_FIELDNAMES, delimiter="\t")
                if new_file:
                    writer.writeheader()
                writer.writerow(row)
        except Exception as exc:
            if not db_ok:
                raise  # both stores failed — the Trial result would be lost
            print(f"[results] legacy TSV append failed (DB unaffected): {exc}")
