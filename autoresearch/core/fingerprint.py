"""Fingerprint file IO (issue #49, ADR 0014 bus).

The implementation now lives in `autoresearch_core.fingerprint` (a
pure-Rust crate exposed via PyO3 + maturin). This module is a thin
compatibility shim so every existing import — including the dual-write
TSV mirror log and the Trial gate (issue #53) — continues to work
unchanged.

History (carried verbatim from the previous pure-Python module):

    Portable hill-climb -> launcher bus: GGUF **basename** + the
    ENGINE_DEFAULTS used for that climb, optional SAMPLER_DEFAULTS.
    Machine-local JSON under a gitignored ``fingerprints/`` directory.
    This is NOT the Pareto Fingerprint hash in
    :mod:`autoresearch.core.pareto` (a sha256 of engine + sampler).

    No GPU, no launcher/eval imports — pure stdlib so roundtrip works in
    tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from autoresearch_core.fingerprint import (  # noqa: F401
    FINGERPRINT_SCHEMA_VERSION,
    SCHEMA_VERSION,
    FingerprintError,
)
from autoresearch_core.fingerprint import (
    dump as _rust_dump,
)
from autoresearch_core.fingerprint import (
    load as _rust_load,
)
from autoresearch_core.fingerprint import (
    mismatch_reason as _rust_mismatch_reason,
)

# Re-export the Rust constant under the canonical Python name.
SCHEMA_VERSION: int = int(SCHEMA_VERSION)

DEFAULT_DIR_NAME = "fingerprints"


def default_dir(root: Path | str | None = None) -> Path:
    """Return the machine-local fingerprints directory (created on dump)."""
    base = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    return base / DEFAULT_DIR_NAME


def path_for(model_basename: str, directory: Path | str | None = None) -> Path:
    """Return the one-file-per-basename path for ``model_basename``.

    Pure-Python implementation kept here so consumer-side patches (e.g.
    ``unittest.mock.patch``) can intercept it; the path layout is part of
    the on-disk contract and does not benefit from the Rust port.
    """
    from pathlib import Path as _P

    base = str(model_basename)
    # Strip POSIX + Windows directory components to a basename.
    cleaned = base.replace("\\", "/").strip()
    stem = _P(cleaned).stem if cleaned else ""
    parent = default_dir() if directory is None else _P(directory)
    return parent / f"{stem}.json"


def dump(
    path: Path | str,
    *,
    model: str,
    engine: dict[str, Any],
    sampler: dict[str, Any] | None = None,
) -> Path:
    """Write a Fingerprint file; return the path written. Mirrors the
    Python contract."""
    written = _rust_dump(path, model=model, engine=engine, sampler=sampler)
    return Path(written)


def load(path: Path | str) -> dict[str, Any]:
    """Load a Fingerprint file; reject missing schema or private leakage."""
    return _rust_load(path)


def apply(
    path: Path | str,
    *,
    baseline_path: Path | str | None = None,
) -> dict[str, Any]:
    """Copy a Fingerprint file into the mutable Baseline (issue #50).

    Engine is always applied; the optional sampler only when the file
    carries one — an omitted sampler leaves the Baseline sampler alone.

    The Rust port handles schema + scrub validation; the merge into the
    live `config.py` stays Python-side because `config.write_baseline`
    owns the operator's mutable Baseline file.
    """
    data = load(path)
    from autoresearch.core import config as config_module

    cfg = dict(data["engine"])
    cfg["MODEL"] = data["model"]
    if data.get("sampler") is not None:
        cfg.update(data["sampler"])
    unknown = sorted(k for k in cfg if k not in config_module.CONFIG_KEYS)
    if unknown:
        raise FingerprintError(f"unknown Baseline keys in {path}: {unknown}")
    if baseline_path is not None:
        return config_module.write_baseline(cfg, path=baseline_path)
    return config_module.write_baseline(cfg)


# ENGINE_DEFAULTS split (single source; scripts/model_up.py imports both):
SERVER_ENGINE_KEYS = frozenset(
    {
        "CTX_SIZE",
        "BATCH_SIZE",
        "UBATCH_SIZE",
        "THREADS",
        "PARALLEL",
        "N_GPU_LAYERS",
        "NUMA",
        "KV_CACHE",
        "KV_CACHE_K",
        "KV_CACHE_V",
        "FLASH_ATTN",
        "THREADS_BATCH",
        "NO_MMAP",
        "MLOCK",
        "JINJA",
        "REASONING_BUDGET",
        "REASONING_BUDGET_MESSAGE",
        "REASONING",
        "REASONING_PRESERVE",
        "REASONING_EFFORT",
        "CONT_BATCHING",
        "CACHE_REUSE",
        "SPEC_TYPE",
        "SPEC_DRAFT_N_MAX",
        "SPEC_DRAFT_MODEL",
        "MOE_CACHE_PROFILE",
        "MOE_CACHE_SLOTS",
        "N_CPU_MOE",
    }
)
HARNESS_ONLY_ENGINE_KEYS = frozenset(
    {
        "VRAM_LIMIT_MB",
        "VRAM_HEADROOM_MB",
        "HOST_MEMORY_HEADROOM_MB",
        "FREE_RAM_FLOOR_MB",
        "RAM_WATCHDOG_POLL_S",
        "RAM_WATCHDOG_RESERVE_MB",
        "RAM_PREFLIGHT_MARGIN_MB",
        "TPS_FLOOR",
        "TPS_REPS",
        "THERMAL_WAIT",
    }
)


def mismatch_reason(
    model_basename: str,
    baseline_engine: dict[str, Any],
    *,
    directory: Path | str | None = None,
    _target_path: str | Path | None = None,
) -> str | None:
    """Trial gate (issue #53): None = Trial may proceed, else the reject reason.

    The optional ``_target_path`` argument is reserved for the Python shim
    to pre-resolve the on-disk path (after running ``path_for``), so that
    tests can patch ``path_for`` and still drive the gating logic through
    Rust. Production callers use either ``directory=...`` (Rust computes the
    path) or pass nothing (defaults to ``./fingerprints/<stem>.json``).
    """
    target: str | None = None
    if _target_path is not None:
        target = str(_target_path)
    elif directory is not None:
        target = str(path_for(model_basename, directory))
    else:
        # Fall back to the Python `path_for` so consumer-side patches
        # (e.g. ``unittest.mock.patch``) still drive the gate. The Rust
        # implementation would otherwise resolve the path itself and the
        # mock would be ignored.
        target = str(path_for(model_basename))
    return _rust_mismatch_reason(
        model_basename,
        baseline_engine,
        directory=str(directory) if directory else None,
        _target_path=target,
    )


__all__ = [
    "SCHEMA_VERSION",
    "DEFAULT_DIR_NAME",
    "FingerprintError",
    "default_dir",
    "path_for",
    "dump",
    "load",
    "apply",
    "SERVER_ENGINE_KEYS",
    "HARNESS_ONLY_ENGINE_KEYS",
    "mismatch_reason",
]
