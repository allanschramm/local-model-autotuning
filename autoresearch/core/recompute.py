"""Store-wide status recompute (issue #5; ADR 0017).

Pure decision logic — no file I/O. Takes results-store rows and returns a new
list with every row's status refreshed. Domination is a same-model config
verdict (ADR 0017): store-wide recompute never demotes one model for another.
Rows group by basename × budget bucket (config scope is retained as an
accepted argument form; both scopes compete only within one basename).
Every row of one group shares the merged vector's status: incomplete stays
incomplete; a complete merged vector is on_front. Cross-model `dominated`
labels cease to exist via the normal recompute pass — same-model config A/B
verdicts (`dominated`) are written only by hill-climb bookkeeping
(classify.plan_write) and are flipped back to on_front here.
Point identity = GGUF basename (ADR 0012).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from autoresearch.core.classify import (
    MORRIS_SCREEN_PROFILE,
    row_bucket,
    vector_from_row,
)
from autoresearch.core.pareto import ObjectiveVector, Trial, merge

# ADR 0017: per-model competition is the canonical scope; `bucket` remains an
# accepted argument form with the same same-basename competition.
SCOPES = ("model", "bucket")
DEFAULT_SCOPE = "model"


def recompute_rows(
    rows: Sequence[Mapping[str, Any]], *, scope: str = DEFAULT_SCOPE
) -> list[dict[str, Any]]:
    """New row list with refreshed statuses (idempotent, pure).

    Point identity is the GGUF basename (ADR 0012). Rows without a model name
    or a budget bucket are left untouched. rejected rows never compete. Every
    row of one group (basename × bucket) shares the merged vector's status:
    incomplete stays incomplete; a complete merged basename vector is
    on_front. No group ever demotes another — domination is same-model only
    (ADR 0017) and lives in hill-climb A/B bookkeeping, not store-wide
    recompute.
    """
    if scope not in SCOPES:
        raise ValueError(f"invalid scope: {scope!r}; allowed: {sorted(SCOPES)}")
    groups: dict[tuple[Any, Any], list[int]] = {}
    for idx, row in enumerate(rows):
        model = (row.get("model") or "").strip()
        bucket_gb = row_bucket(row)
        if (
            not model
            or bucket_gb is None
            or row.get("status") == "rejected"
            or (row.get("evaluation_profile") or "").strip() == MORRIS_SCREEN_PROFILE
        ):
            continue
        # Both scopes compete only within one basename: (model, bucket).
        key = (model, bucket_gb)
        groups.setdefault(key, []).append(idx)
    merged_by_group: dict[tuple[Any, Any], ObjectiveVector] = {}
    for key, idxs in groups.items():
        vectors = [vector_from_row(rows[i]) for i in idxs]
        # merge() keys on Trial.fp — use basename as the merge id.
        merge_id = key[0]
        merged_by_group[key] = merge([Trial(fp=str(merge_id), vector=v) for v in vectors])[0].vector
    statuses: dict[tuple[Any, Any], str] = {
        key: "on_front" if merged.complete else "incomplete"
        for key, merged in merged_by_group.items()
    }
    out = [dict(row) for row in rows]
    for key, idxs in groups.items():
        status = statuses[key]
        for idx in idxs:
            out[idx]["status"] = status
    return out


#: Pre-fix watchdog kill marker (issue #72): every such row was a device-wide
#: NVML policy kill — the CUDA-free guard did not exist yet.
WATCHDOG_KILL_LEGACY_MARKER = "VRAM_LIMIT_EXCEEDED"

WATCHDOG_KILL_RELABELED_DIAGNOSTIC = (
    f"WATCHDOG_KILL (legacy {WATCHDOG_KILL_LEGACY_MARKER}, scope=nvml-device-wide)"
)


def relabel_watchdog_kills(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Relabel legacy policy kills to WATCHDOG_KILL (issue #72). Pure, idempotent.

    Pre-fix watchdog kills were stored as MODEL_REJECTED/VRAM_LIMIT_EXCEEDED —
    indistinguishable from real OOMs. Every such row was a device-wide NVML
    policy kill, so the scope is honest. Genuine model rejects (preflight,
    TPS floor) and already-honest rows pass through untouched. Store status
    stays ``rejected``: still failed, still out of the rank — only the
    outcome/diagnostic become honest.
    """
    out: list[dict[str, Any]] = []
    for row in rows:
        relabeled = dict(row)
        if (relabeled.get("outcome") or "") == "MODEL_REJECTED" and (
            WATCHDOG_KILL_LEGACY_MARKER in (relabeled.get("diagnostic") or "")
            or WATCHDOG_KILL_LEGACY_MARKER in (relabeled.get("status") or "")
        ):
            relabeled["outcome"] = "WATCHDOG_KILL"
            relabeled["diagnostic"] = WATCHDOG_KILL_RELABELED_DIAGNOSTIC
        out.append(relabeled)
    return out
