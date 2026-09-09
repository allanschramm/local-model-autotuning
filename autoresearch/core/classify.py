"""Trial classifier over the Pareto nucleus (issue #4).

Pure decision logic — no file I/O. The run.py write path feeds results-store
rows as dicts and persists the outcome. Vocabulary follows CONTEXT.md /
ADR 0006: `rejected` | `incomplete` | `on_front` | `dominated`.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from autoresearch.core.config import DEFAULTS
from autoresearch.core.pareto import ObjectiveVector, Trial, dominates, fingerprint, merge

# Fingerprint identity = ENGINE_DEFAULTS + SAMPLER_DEFAULTS only (ADR 0006).
# Extra keys (e.g. bench INCLUDE_* flags in AutoLoop configs) must not split
# two write paths that share the same engine+sampler Baseline.
_IDENTITY_KEYS = frozenset(str(k).lower() for k in DEFAULTS)

# Diagnostic profiles never compete for any front (ADR 0016): Morris screen
# rows carry reps=1 llama-cli TPS probes, not Trial measurements.
MORRIS_SCREEN_PROFILE = "morris-screen"

# Hardware+budget identity of the known Set (ADR 0006: the global front is
# ranked per hardware+budget). Prefer configured VRAM_LIMIT_MB; peak memory_gb
# is legacy fallback only (same Fingerprint can peak differently across Trials).
BUCKET_PROXY = "memory_gb"


def bucket(memory_gb: float) -> int:
    return round(memory_gb)


def _vram_limit_bucket(config_json: Any) -> int | None:
    """Budget bucket from configured VRAM_LIMIT_MB (GiB, rounded).

    Peak ``memory_gb`` varies Trial-to-Trial for the same Fingerprint (coding vs
    claw peaks differ by hundreds of MiB) and was splitting Objective Vector
    merges across buckets. Configured limit is the intended hardware budget.
    """
    if not config_json:
        return None
    try:
        loaded = json.loads(config_json) if isinstance(config_json, str) else config_json
    except (TypeError, ValueError):
        return None
    if not isinstance(loaded, dict):
        return None
    raw = loaded.get("vram_limit_mb", loaded.get("VRAM_LIMIT_MB"))
    limit = _cell_float(raw)
    if limit is None or limit <= 0:
        return None
    return bucket(limit / 1024.0)


def _cell_float(cell: Any) -> float | None:
    if cell is None or cell == "":
        return None
    try:
        return float(cell)
    except (TypeError, ValueError):
        return None


def vector_from_row(row: Mapping[str, Any]) -> ObjectiveVector:
    """Objective Vector of a results-store row; blank axis = not measured."""
    return ObjectiveVector(
        ctx=_cell_float(row.get("ctx")),
        tps=_cell_float(row.get("tps")),
        agentic=_cell_float(row.get("agentic")),
        coding=_cell_float(row.get("coding")),
    )


def fp_from_baseline(baseline: Mapping[str, Any]) -> str:
    """Fingerprint of the full Baseline = ENGINE_DEFAULTS + SAMPLER_DEFAULTS.

    Keys are lowercased so the live config dict and the persisted config_json
    cell hash identically (ADR 0006: engine + sampler, not model-only).
    """
    canonical = {str(k).lower(): v for k, v in baseline.items() if str(k).lower() in _IDENTITY_KEYS}
    return fingerprint({"baseline": canonical}, {})


def fp_from_config_json(config_json: Any) -> str | None:
    """Recompute a row's Fingerprint from its persisted config_json cell."""
    if not config_json:
        return None
    try:
        loaded = json.loads(config_json)
    except (TypeError, ValueError):
        return None
    if not isinstance(loaded, dict):
        return None
    return fp_from_baseline(loaded)


def is_on_front(status: Any) -> bool:
    """True iff status is the canonical ADR 0006 `on_front` label."""
    return status == "on_front"


def classify_trial(
    *, failed: bool, vector: ObjectiveVector, known: Iterable[ObjectiveVector]
) -> str:
    """ADR 0006 status for one Trial: rejected | incomplete | on_front | dominated."""
    if failed:
        return "rejected"
    if not vector.complete:
        return "incomplete"
    if any(dominates(other, vector) for other in known):
        return "dominated"
    return "on_front"


def known_vectors(
    rows: Sequence[Mapping[str, Any]], *, model: str, bucket_gb: int | None
) -> list[ObjectiveVector]:
    """Complete merged points for ONE basename in a hardware+budget bucket.

    ADR 0017 Known-Set shape shared by classify and autoloop's Search seed:
    rejected rows never compete, Morris screen points are diagnostic-only
    (ADR 0016), and repeated measurements of one basename merge to a
    best-of-each-axis vector. ``bucket_gb=None`` means no bucket filter.
    """
    vectors: list[ObjectiveVector] = []
    for row in rows:
        if row.get("status") == "rejected":
            continue  # rejected Trials never compete for the front
        if (row.get("evaluation_profile") or "").strip() == MORRIS_SCREEN_PROFILE:
            continue  # Morris screen points are diagnostic, never front seeds
        if (row.get("model") or "").strip() != model:
            continue
        if bucket_gb is not None and row_bucket(row) != bucket_gb:
            continue
        vectors.append(vector_from_row(row))
    merged = merge([Trial(fp=model, vector=v) for v in vectors])
    return [trial.vector for trial in merged if trial.vector.complete]


def _known_vectors(
    rows: Sequence[Mapping[str, Any]],
    bucket_gb: int,
    model: str | None = None,
    fp: str | None = None,
) -> list[ObjectiveVector]:
    """Complete points already in this hardware+budget bucket, merged per basename.

    ADR 0017: when the Trial has a model key (or fp), the known set is scoped
    to the Trial's own basename — a stronger model never demotes another
    model; only same-basename configs compete.
    """
    if model is not None:
        return known_vectors(rows, model=model, bucket_gb=bucket_gb)
    by_model: dict[str, list[ObjectiveVector]] = {}
    for row in rows:
        if row.get("status") == "rejected":
            continue  # rejected Trials never compete for the front
        if (row.get("evaluation_profile") or "").strip() == MORRIS_SCREEN_PROFILE:
            continue  # Morris screen points are diagnostic, never front seeds
        if row_bucket(row) != bucket_gb:
            continue
        row_model = (row.get("model") or "").strip()
        if not row_model:
            continue
        if model is not None:
            if row_model != model:
                continue
        elif fp is not None and fp_from_config_json(row.get("config_json")) != fp:
            continue
        by_model.setdefault(row_model, []).append(vector_from_row(row))
    known = []
    for row_model, vectors in by_model.items():
        merged = merge([Trial(fp=row_model, vector=v) for v in vectors])[0].vector
        if merged.complete:
            known.append(merged)
    return known


def row_bucket(row: Mapping[str, Any]) -> int | None:
    """Bucket of a results-store row (None when budget cannot be resolved).

    Prefer ``round(VRAM_LIMIT_MB / 1024)`` from ``config_json`` so Trials of
    the same basename merge across peaks that round differently (e.g. 7.4 vs
    7.8). Legacy rows without a limit fall back to peak memory_gb.
    """
    from_limit = _vram_limit_bucket(row.get("config_json"))
    if from_limit is not None:
        return from_limit
    mem = _cell_float(row.get(BUCKET_PROXY))
    return None if mem is None else bucket(mem)


def plan_write(
    rows: Sequence[Mapping[str, Any]],
    *,
    fp: str,
    vector: ObjectiveVector,
    bucket_gb: int,
    failed: bool = False,
    model: str | None = None,
) -> tuple[str, dict[str, str]]:
    """Status for a new Trial plus prior rows to flip.

    Returns (status, {trial_id: status}). Merge identity is the GGUF basename
    when ``model`` is set (ADR 0012). ``fp``-only callers (legacy tests) still
    merge by Fingerprint. Known Set is always basename-scoped in the bucket.
    """
    if failed:
        return "rejected", {}
    model_key = (model or "").strip()
    if model_key:
        prior = [
            vector_from_row(row)
            for row in rows
            if row.get("status") != "rejected"
            and (row.get("model") or "").strip() == model_key
            and row_bucket(row) == bucket_gb
        ]
        merge_id = model_key
    else:
        prior = [
            vector_from_row(row)
            for row in rows
            if row.get("status") != "rejected"
            and fp_from_config_json(row.get("config_json")) == fp
            and row_bucket(row) == bucket_gb
        ]
        merge_id = fp
    merged = merge([Trial(fp=merge_id, vector=v) for v in [*prior, vector]])[0].vector
    status = classify_trial(
        failed=False,
        vector=merged,
        known=_known_vectors(
            rows, bucket_gb, model=model_key or None, fp=None if model_key else fp
        ),
    )
    if status == "incomplete":
        return status, {}
    if model_key:
        flips = {
            row.get("trial_id", ""): status
            for row in rows
            if row.get("status") == "incomplete"
            and (row.get("model") or "").strip() == model_key
            and row_bucket(row) == bucket_gb
        }
    else:
        flips = {
            row.get("trial_id", ""): status
            for row in rows
            if row.get("status") == "incomplete"
            and fp_from_config_json(row.get("config_json")) == fp
            and row_bucket(row) == bucket_gb
        }
    return status, flips
