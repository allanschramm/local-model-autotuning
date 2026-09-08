"""Trial classifier tests (issue #4): classify + fingerprint merge, no GPU needed."""

from __future__ import annotations

import json

from autoresearch.core.classify import (
    bucket,
    classify_trial,
    fp_from_baseline,
    fp_from_config_json,
    is_on_front,
    plan_write,
    row_bucket,
    vector_from_row,
)
from autoresearch.core.pareto import ObjectiveVector

BASELINE = {
    "MODEL": "M.gguf",
    "CTX_SIZE": 131072,
    "KV_CACHE": "turbo2",
    "THREADS": 6,
    "TEMP": 0.4,
    "TOP_P": 0.95,
}


def v(**kw) -> ObjectiveVector:
    return ObjectiveVector(**kw)


def cfg_json(baseline: dict) -> str:
    return json.dumps(
        {k.lower(): val for k, val in baseline.items()}, sort_keys=True, separators=(",", ":")
    )


def row(**kw) -> dict:
    base = {
        "trial_id": "t1",
        "status": "incomplete",
        "memory_gb": "8.0",
        "config_json": cfg_json(BASELINE),
        "ctx": "",
        "tps": "",
        "agentic": "",
        "coding": "",
    }
    base.update(kw)
    return base


def test_fp_from_config_json_rejects_non_object_json():
    assert fp_from_config_json("[]") is None
    assert fp_from_config_json("42") is None
    assert fp_from_config_json("[1, 2]") is None


def test_fp_ignores_non_baseline_keys():
    # Extra keys (AutoLoop bench INCLUDE_* flags) must not split the Fingerprint.
    assert fp_from_baseline({**BASELINE, "INCLUDE_CODING": True}) == fp_from_baseline(BASELINE)


def test_bucket_rounds_memory_gb():
    assert bucket(7.9) == 8
    assert bucket(16.0) == 16


def test_row_bucket_prefers_vram_limit_over_peak_memory():
    """Same Fingerprint must not split when coding peaks 7.8 and claw peaks 7.4."""
    cfg = cfg_json({**BASELINE, "VRAM_LIMIT_MB": 8100.0})
    coding = row(memory_gb="7.8", config_json=cfg)
    claw = row(memory_gb="7.4", config_json=cfg)
    assert row_bucket(coding) == row_bucket(claw) == 8
    legacy = row(memory_gb="7.4", config_json=cfg_json(BASELINE))  # no limit
    assert row_bucket(legacy) == 7


def test_fp_from_baseline_matches_persisted_config_json():
    assert fp_from_baseline(BASELINE) == fp_from_config_json(row()["config_json"])
    assert fp_from_baseline(dict(BASELINE, THREADS=8)) != fp_from_baseline(BASELINE)
    assert fp_from_config_json("") is None
    assert fp_from_config_json("not json") is None


def test_vector_from_row_blank_axes_are_none():
    vec = vector_from_row(row(ctx="131072", tps="30.0", agentic="0.6"))
    assert (vec.ctx, vec.tps, vec.agentic) == (131072, 30.0, 0.6)
    assert vec.coding is None
    assert not vec.complete


def test_is_on_front_only_accepts_on_front():
    assert is_on_front("on_front")
    assert not is_on_front("keep")
    assert not is_on_front("discard")
    assert not is_on_front("dominated")
    assert not is_on_front("incomplete")
    assert not is_on_front("rejected")
    assert not is_on_front(None)


def test_classify_trial_all_statuses():
    full = v(ctx=131072, tps=30.0, agentic=0.6, coding=0.6)
    assert classify_trial(failed=True, vector=full, known=[]) == "rejected"
    assert (
        classify_trial(failed=False, vector=v(ctx=131072, tps=30.0, agentic=0.6), known=[])
        == "incomplete"
    )
    assert classify_trial(failed=False, vector=full, known=[full]) == "on_front"
    better = v(ctx=131072, tps=40.0, agentic=0.7, coding=0.7)
    assert classify_trial(failed=False, vector=full, known=[better]) == "dominated"


def test_plan_write_partial_is_incomplete_no_flips():
    status, flips = plan_write(
        [], fp="fp", vector=v(ctx=131072, tps=30.0, agentic=0.6), bucket_gb=8
    )
    assert status == "incomplete"
    assert flips == {}


def test_plan_write_failed_short_circuits_to_rejected():
    status, flips = plan_write(
        [],
        fp="fp",
        vector=v(ctx=131072, tps=30.0, agentic=0.6, coding=0.6),
        bucket_gb=8,
        failed=True,
    )
    assert status == "rejected"
    assert flips == {}


def test_merge_completes_prior_incomplete_and_flips_its_row():
    fp = fp_from_baseline(BASELINE)
    prior = row(trial_id="old1", status="incomplete", ctx="131072", tps="30.0", agentic="0.6")
    status, flips = plan_write([prior], fp=fp, vector=v(ctx=131072, coding=0.5), bucket_gb=8)
    assert status == "on_front"  # merged with prior axes -> complete, undominated
    assert flips == {"old1": "on_front"}


def test_merge_still_incomplete_leaves_rows_alone():
    fp = fp_from_baseline(BASELINE)
    prior = row(trial_id="old1", status="incomplete", ctx="131072")
    status, flips = plan_write([prior], fp=fp, vector=v(ctx=131072, tps=30.0), bucket_gb=8)
    assert status == "incomplete"
    assert flips == {}


def test_plan_write_known_set_scoped_to_own_basename():
    # ADR 0017: a stronger model in the same bucket NEVER demotes a new
    # Trial; only same-basename configs compete.
    fp = fp_from_baseline(BASELINE)
    stronger_other_model = row(
        trial_id="old1",
        model="Better.gguf",
        status="on_front",
        config_json=cfg_json(dict(BASELINE, MODEL="Better.gguf", THREADS=8)),
        ctx="131072",
        tps="40.0",
        agentic="0.7",
        coding="0.7",
    )
    status, flips = plan_write(
        [stronger_other_model],
        fp=fp,
        vector=v(ctx=131072, tps=30.0, agentic=0.6, coding=0.6),
        bucket_gb=8,
        model="M.gguf",
    )
    assert status == "on_front"
    assert flips == {}


def test_plan_write_same_basename_better_config_merges_to_on_front():
    # ADR 0017: a better config of the SAME basename merges into the vector
    # (on_front everywhere with the merged vector). Same-model domination
    # verdicts live in hill-climb bookkeeping (search.py), not store labels.
    fp = fp_from_baseline(BASELINE)
    better_config = row(
        trial_id="cfg1",
        model="M.gguf",
        status="on_front",
        config_json=cfg_json(dict(BASELINE, THREADS=8)),
        ctx="131072",
        tps="40.0",
        agentic="0.7",
        coding="0.7",
    )
    status, flips = plan_write(
        [better_config],
        fp=fp,
        vector=v(ctx=131072, tps=30.0, agentic=0.6, coding=0.6),
        bucket_gb=8,
        model="M.gguf",
    )
    assert status == "on_front"
    # cfg1 is already on_front: no flip needed (statuses agree with merge).
    assert flips == {}


def test_known_set_respects_memory_bucket():
    fp = fp_from_baseline(BASELINE)
    strong_other_bucket = row(
        trial_id="old1",
        status="on_front",
        memory_gb="16.0",
        config_json=cfg_json(dict(BASELINE, THREADS=8)),
        ctx="131072",
        tps="40.0",
        agentic="0.7",
        coding="0.7",
    )
    status, _ = plan_write(
        [strong_other_bucket],
        fp=fp,
        vector=v(ctx=131072, tps=30.0, agentic=0.6, coding=0.6),
        bucket_gb=8,
    )
    assert status == "on_front"  # the 16GB point does not compete


def test_merge_and_flips_are_bucket_scoped():
    fp = fp_from_baseline(BASELINE)
    other_bucket = row(
        trial_id="big",
        status="incomplete",
        memory_gb="16.0",
        ctx="131072",
        tps="30.0",
        agentic="0.6",
    )
    status, flips = plan_write([other_bucket], fp=fp, vector=v(ctx=131072, coding=0.5), bucket_gb=8)
    assert status == "incomplete"  # the 16GB partial does not merge into the 8GB point
    assert flips == {}


def test_rejected_rows_excluded_from_known_set():
    fp = fp_from_baseline(BASELINE)
    rejected_strong = row(
        trial_id="rej",
        status="rejected",
        config_json=cfg_json(dict(BASELINE, THREADS=8)),
        ctx="131072",
        tps="40.0",
        agentic="0.7",
        coding="0.7",
    )
    status, flips = plan_write(
        [rejected_strong],
        fp=fp,
        vector=v(ctx=131072, tps=30.0, agentic=0.6, coding=0.6),
        bucket_gb=8,
    )
    assert status == "on_front"  # a rejected point never dominates
    assert flips == {}


def test_rejected_rows_excluded_from_merge():
    fp = fp_from_baseline(BASELINE)
    rejected = row(trial_id="rej", status="rejected", ctx="131072", tps="30.0", agentic="0.6")
    status, flips = plan_write([rejected], fp=fp, vector=v(ctx=131072, coding=0.5), bucket_gb=8)
    assert status == "incomplete"  # a rejected partial never fills the new Trial's axes
    assert flips == {}
