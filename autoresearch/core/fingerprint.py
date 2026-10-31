"""Fingerprint file IO (issue #49, ADR 0014 bus).

Portable hill-climb -> launcher bus: GGUF **basename** + the ENGINE_DEFAULTS
used for that climb, optional SAMPLER_DEFAULTS. Machine-local JSON under a
gitignored ``fingerprints/`` directory. This is NOT the Pareto Fingerprint
configuration identity used internally during Search.

No GPU, no launcher/eval imports — pure stdlib so roundtrip works in tests.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path, PurePath, PureWindowsPath
from typing import Any

SCHEMA_VERSION = 1

DEFAULT_DIR_NAME = "fingerprints"

# Key names that must never land in a shared file (case-insensitive match).
_PRIVATE_KEYS = frozenset(
    {
        "hostname",
        "host",
        "machine",
        "user",
        "username",
        "email",
        "gpu",
        "gpu_sku",
        "gpu_model",
        "alias",
        "alias_name",
        "model_alias",
    }
)

_ABSOLUTE_PATH_RE = re.compile(
    r"^(?:[A-Za-z]:[\\/]|\\\\|/|~[\\/]?)"  # drive, UNC, POSIX abs, tilde
)
_EMAIL_RE = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")
_URL_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9+.-]*://")
_GPU_SKU_RE = re.compile(r"\b(?:RTX|GTX|GT|RX|ARC|A100|H100|H200|B200)\b", re.IGNORECASE)
_HOSTNAME_RE = re.compile(
    r"(?i)^(localhost|([a-z0-9]([a-z0-9-]*[a-z0-9])?\.)+(local|lan|internal|[a-z]{2,}))$"
)
# Dotted basenames that must not trip the hostname check (e.g. "model.gguf").
_FILENAME_EXT_RE = re.compile(r"(?i)\.(gguf|bin|safetensors|csv|json|md|txt|yaml|yml|log|ggml)$")


class FingerprintError(ValueError):
    """Raised when a Fingerprint payload leaks private data or breaks schema."""


def _basename(name: str) -> str:
    """Strip any directory components (POSIX + Windows) to a pure basename."""
    base = PureWindowsPath(name).name or PurePath(name).name or name
    return PurePath(base).name


def _looks_like_basename(name: str) -> bool:
    return (
        "/" not in name
        and "\\" not in name
        and not _ABSOLUTE_PATH_RE.match(name)
        and PureWindowsPath(name).name == name
        and PurePath(name).name == name
    )


def _check_value(key: str, value: Any) -> None:
    if isinstance(value, str):
        if _ABSOLUTE_PATH_RE.match(value.strip()):
            raise FingerprintError(f"private absolute path in {key!r}: {value!r}")
        if _EMAIL_RE.search(value) or _URL_RE.search(value):
            raise FingerprintError(f"private contact/host value in {key!r}: {value!r}")
        if _GPU_SKU_RE.search(value):
            raise FingerprintError(f"private GPU SKU value in {key!r}: {value!r}")
        stripped = value.strip()
        if (
            ("." in stripped or stripped.lower() == "localhost")
            and "/" not in stripped
            and "\\" not in stripped
            and not _FILENAME_EXT_RE.search(stripped)
            and _HOSTNAME_RE.match(stripped)
        ):
            raise FingerprintError(f"private hostname value in {key!r}: {value!r}")
    elif isinstance(value, Mapping):
        _check_mapping(value)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _check_value(key, item)


def _check_mapping(data: Mapping[str, Any]) -> None:
    for key, value in data.items():
        if str(key).strip().lower() in _PRIVATE_KEYS:
            raise FingerprintError(f"private key {key!r} must not be fingerprinted")
        _check_value(str(key), value)


def default_dir(root: Path | str | None = None) -> Path:
    """Return the machine-local fingerprints directory (created on dump)."""
    base = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    return base / DEFAULT_DIR_NAME


def path_for(model_basename: str, directory: Path | str | None = None) -> Path:
    """Return the one-file-per-basename path for ``model_basename``."""
    stem = PurePath(_basename(model_basename)).stem
    parent = default_dir() if directory is None else Path(directory)
    return parent / f"{stem}.json"


def dump(
    path: Path | str,
    *,
    model: str,
    engine: Mapping[str, Any],
    sampler: Mapping[str, Any] | None = None,
) -> Path:
    """Write a Fingerprint file; return the path written.

    ``model`` may carry directories — only the basename is stored. Absolute
    user paths, hostnames, emails, alias names, and GPU SKUs in ``engine`` /
    ``sampler`` raise :class:`FingerprintError`.
    """
    if not isinstance(engine, Mapping):
        raise FingerprintError("engine must be a mapping of ENGINE_DEFAULTS")
    if sampler is not None and not isinstance(sampler, Mapping):
        raise FingerprintError("sampler must be a mapping of SAMPLER_DEFAULTS or None")

    base = _basename(model.strip())
    if not base:
        raise FingerprintError("model must be a non-empty GGUF basename")

    engine_clean = dict(engine)
    engine_clean["MODEL"] = base
    _check_mapping(engine_clean)
    sampler_clean = dict(sampler) if sampler is not None else None
    if sampler_clean is not None:
        _check_mapping(sampler_clean)

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "model": base,
        "engine": engine_clean,
    }
    if sampler_clean is not None:
        payload["sampler"] = sampler_clean

    target = Path(path)
    if target.parent != Path(".") and str(target.parent):
        target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def _checked_mapping(
    value: Any, what: str, target: Path, *, required: bool
) -> dict[str, Any] | None:
    """Validate a load() engine/sampler section and scrub it for private data."""
    if value is None and not required:
        return None
    if not isinstance(value, dict) or not value:
        raise FingerprintError(f"invalid Fingerprint {what} in {target}: mapping required")
    _check_mapping(value)
    return dict(value)


def load(path: Path | str) -> dict[str, Any]:
    """Load a Fingerprint file; reject missing schema or private leakage."""
    target = Path(path)
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise FingerprintError(f"invalid Fingerprint JSON in {target}: {exc}") from exc
    if not isinstance(payload, dict):
        raise FingerprintError(f"invalid Fingerprint payload in {target}: not an object")

    version = payload.get("schema_version")
    if version != SCHEMA_VERSION:
        raise FingerprintError(
            f"unsupported Fingerprint schema_version {version!r} in {target} "
            f"(want {SCHEMA_VERSION}); schema_version is required"
        )

    model = payload.get("model")
    if not isinstance(model, str) or not model or not _looks_like_basename(model):
        raise FingerprintError(
            f"invalid Fingerprint model {model!r} in {target}: GGUF basename only"
        )

    engine = _checked_mapping(payload.get("engine"), "engine", target, required=True)
    sampler = _checked_mapping(payload.get("sampler"), "sampler", target, required=False)

    return {
        "schema_version": SCHEMA_VERSION,
        "model": model,
        "engine": dict(engine),
        "sampler": dict(sampler) if sampler is not None else None,
    }


def apply(
    path: Path | str,
    *,
    baseline_path: Path | str | None = None,
) -> dict[str, Any]:
    """Copy a Fingerprint file into the mutable Baseline (issue #50).

    Engine is always applied; the optional sampler only when the file
    carries one — an omitted sampler leaves the Baseline sampler alone.
    Existing benches keep reading Baseline (no scorer change); only the
    Baseline moves. Values validate via ``write_baseline``.

    Unknown keys raise: ``write_baseline`` only copies ``CONFIG_KEYS``, so
    a typo'd key would otherwise vanish while the caller reports success.
    The config module imports lazily so this file stays pure-stdlib at
    import time (roundtrip works with no Baseline present).
    """
    data = load(path)
    from autoresearch.core import config as config_module

    cfg = dict(data["engine"])
    # Top-level `model` is the file's validated identity (ADR 0014); the
    # engine mapping may lack MODEL or carry a stale one (hand-edited or
    # third-party file), so pin it — never run new flags on the old GGUF.
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
# server keys map to llama-server flags (same engine = same server Pi sees);
# harness-only keys are budgets/floors/gates carried in the file but never
# emitted as flags. A Trial scores flags, so only server keys (+ MODEL
# identity) can reject it as a Fingerprint mismatch (issue #53).
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
    baseline_engine: Mapping[str, Any],
    *,
    directory: Path | str | None = None,
) -> str | None:
    """Trial gate (issue #53): None = Trial may proceed, else the reject reason.

    No Fingerprint file for ``model_basename`` → None (Baseline-only behavior
    unchanged). Otherwise the file's frozen server flags must equal the live
    Baseline engine; harness-only drift (TPS_FLOOR, VRAM limits, …) never
    rejects. Unreadable/invalid files and unknown keys fail closed with a
    reason — never an exception, never a quality score.
    """
    base = _basename((model_basename or "").strip())
    if not base:
        return "Fingerprint check: Trial has no model basename; refusing to score"
    baseline = {str(k).upper(): v for k, v in dict(baseline_engine).items()}
    target = path_for(base, directory)
    if not target.is_file():
        return None
    try:
        data = load(target)
    except FingerprintError as exc:
        return f"Fingerprint {target.name} for {base} is invalid ({exc}); delete it or re-climb"
    except OSError as exc:
        return f"Fingerprint {target.name} for {base} is unreadable ({exc}); delete it or re-climb"
    if data["model"] != base:
        return (
            f"Fingerprint {target.name} serves {data['model']!r}, not the Trial model "
            f"{base!r}; delete it or re-climb"
        )
    file_engine = data["engine"]
    unknown = sorted(
        k
        for k in file_engine
        if k not in SERVER_ENGINE_KEYS and k not in HARNESS_ONLY_ENGINE_KEYS and k != "MODEL"
    )
    if unknown:
        return f"Fingerprint {target.name} for {base} has unknown engine keys {unknown}; delete it or re-climb"
    diffs = [
        f"{key} Baseline={baseline.get(key)!r} file={file_engine.get(key)!r}"
        for key in sorted(SERVER_ENGINE_KEYS | {"MODEL"})
        if baseline.get(key) != file_engine.get(key)
    ]
    if not diffs:
        return None
    return (
        f"Fingerprint mismatch for {base} ({target.name}): " + "; ".join(diffs) + " "
        "(Baseline engine differs from the TPS-climbed flags; "
        "apply the Fingerprint or re-climb before scoring)"
    )
