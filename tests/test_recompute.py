"""Store-wide Pareto status recompute tests (issue #5), no GPU needed.

Fixture store = tmp_path results.tsv seeded with rows; recompute runs over
the file exactly as the operator CLI would (run.recompute_statuses).
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from autoresearch.core import recompute
from autoresearch.core.pareto import ObjectiveVector
from autoresearch.runners import run

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


def cfg_json(**over) -> str:
    baseline = dict(BASELINE, **over)
    return json.dumps(
        {k.lower(): val for k, val in baseline.items()}, sort_keys=True, separators=(",", ":")
    )


def row(**kw) -> dict:
    base = {
        "trial_id": "t",
        "model": "M.gguf",
        "status": "incomplete",
        "memory_gb": "8.0",
        "config_json": cfg_json(),
        "ctx": "131072",
        "tps": "30.0",
        "agentic": "0.6",
        "coding": "0.6",
    }
    base.update(kw)
    return base


@pytest.fixture
def store(tmp_path) -> Path:
    path = tmp_path / "results.tsv"
    yield path
    path.unlink(missing_ok=True)


def write_store(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=run.CATEGORY_FIELDNAMES, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def read_store(path: Path) -> dict[str, str]:
    return {r["trial_id"]: r["status"] for r in run.read_rows(path)}


def test_cross_model_strength_never_demotes_anything(store):
    # ADR 0017: domination is same-model only — a stronger model in the same
    # bucket never demotes another model; both stay on_front.
    better = row(
        trial_id="better",
        model="B.gguf",
        tps="40.0",
        agentic="0.7",
        coding="0.7",
        config_json=cfg_json(MODEL="B"),
    )
    worse = row(trial_id="worse", model="M.gguf", tps="30.0", agentic="0.6", coding="0.6")
    write_store(store, [better, worse])
    run.recompute_statuses(store)
    assert read_store(store) == {"better": "on_front", "worse": "on_front"}


def test_incomplete_and_rejected_left_out_of_domination(store):
    partial = row(
        trial_id="partial", model="P.gguf", agentic="", coding="", config_json=cfg_json(MODEL="P")
    )
    rejected = row(
        trial_id="rejected",
        model="R.gguf",
        status="rejected",
        tps="99.0",
        agentic="0.9",
        coding="0.9",
        config_json=cfg_json(MODEL="R"),
    )
    better = row(trial_id="better", tps="40.0", agentic="0.7", coding="0.7")
    write_store(store, [partial, rejected, better])
    run.recompute_statuses(store)
    assert read_store(store) == {
        "partial": "incomplete",  # never dominated, never on_front
        "rejected": "rejected",  # untouched, and does not demote better
        "better": "on_front",
    }


def test_domination_never_crosses_buckets(store):
    fp8 = row(trial_id="a8", tps="40.0", agentic="0.7", coding="0.7")
    same_fp_other_bucket = row(
        trial_id="b16", memory_gb="16.0", tps="30.0", agentic="0.6", coding="0.6"
    )
    write_store(store, [fp8, same_fp_other_bucket])
    run.recompute_statuses(store)
    assert read_store(store) == {"a8": "on_front", "b16": "on_front"}


def test_legacy_rows_without_config_json_recomputed_by_basename(store):
    # ADR 0012: basename + budget enough; config_json not required for status.
    legacy_keep = {
        "trial_id": "legacy-keep",
        "model": "L.gguf",
        "status": "keep",
        "memory_gb": "8.0",
        "config_json": "",
        "ctx": "131072",
        "tps": "30.0",
        "agentic": "0.6",
        "coding": "0.6",
    }
    legacy_discard = {**legacy_keep, "trial_id": "legacy-discard", "status": "discard"}
    write_store(store, [legacy_keep, legacy_discard])
    run.recompute_statuses(store)
    assert read_store(store) == {"legacy-keep": "on_front", "legacy-discard": "on_front"}


def test_same_fingerprint_partials_merge_to_one_status(store):
    agentic_only = row(trial_id="ag", coding="")
    coding_only = row(trial_id="cod", agentic="", tps="30.0")
    write_store(store, [agentic_only, coding_only])
    run.recompute_statuses(store)
    # Merged vector is complete -> both rows share the merged status.
    assert read_store(store) == {"ag": "on_front", "cod": "on_front"}


def test_same_fp_different_peak_vram_merges_via_vram_limit(store):
    """Coding peak 7.8 vs claw peak 7.4 must not leave the Objective Vector incomplete."""
    cfg = cfg_json(VRAM_LIMIT_MB=8100.0)
    coding_only = row(
        trial_id="cod",
        memory_gb="7.8",
        agentic="",
        coding="0.54",
        tps="48.6",
        config_json=cfg,
    )
    agentic_only = row(
        trial_id="ag",
        memory_gb="7.4",
        agentic="0.3333",
        coding="",
        tps="42.2",
        config_json=cfg,
    )
    write_store(store, [coding_only, agentic_only])
    run.recompute_statuses(store)
    assert read_store(store) == {"cod": "on_front", "ag": "on_front"}


def test_input_order_does_not_matter_and_all_complete_groups_on_front(store):
    # Input order must not matter: both complete basenames are on_front under
    # same-model-only domination (ADR 0017).
    a = row(
        trial_id="a",
        model="A.gguf",
        tps="30.0",
        agentic="0.6",
        coding="0.6",
        config_json=cfg_json(MODEL="A"),
    )
    b = row(
        trial_id="b",
        model="B.gguf",
        tps="40.0",
        agentic="0.7",
        coding="0.7",
        config_json=cfg_json(MODEL="B"),
    )
    write_store(store, [a, b])
    run.recompute_statuses(store)
    assert read_store(store) == {"a": "on_front", "b": "on_front"}
    # Reversed input order, same verdict.
    write_store(store, [b, a])
    run.recompute_statuses(store)
    assert read_store(store) == {"a": "on_front", "b": "on_front"}


def test_idempotent_run_twice(store):
    a = row(
        trial_id="a",
        model="A.gguf",
        tps="30.0",
        agentic="0.6",
        coding="0.6",
        config_json=cfg_json(MODEL="A"),
    )
    b = row(
        trial_id="b",
        model="B.gguf",
        tps="40.0",
        agentic="0.7",
        coding="0.7",
        config_json=cfg_json(MODEL="B"),
    )
    write_store(store, [a, b])
    run.recompute_statuses(store)
    first = read_store(store)
    run.recompute_statuses(store)
    assert read_store(store) == first == {"a": "on_front", "b": "on_front"}


def test_stored_dominated_labels_flip_to_on_front(store):
    # ADR 0017 realignment: a previously stored cross-model dominated label
    # flips to on_front via the normal recompute pass (same-model verdicts
    # live only in hill-climb A/B bookkeeping, not store-wide recompute).
    weaker = row(trial_id="weak", model="W.gguf", tps="30.0", agentic="0.6", coding="0.6")
    stronger = row(
        trial_id="strong",
        model="S.gguf",
        tps="40.0",
        agentic="0.7",
        coding="0.7",
        config_json=cfg_json(MODEL="S"),
    )
    weaker["status"] = "dominated"  # legacy stored label
    write_store(store, [weaker, stronger])
    run.recompute_statuses(store)
    assert read_store(store) == {"weak": "on_front", "strong": "on_front"}


def test_model_scope_default_and_bucket_scope_agree(store):
    # ADR 0017: recompute default scope is per-model; bucket scope is kept as
    # an accepted argument form with the same same-basename competition.
    # Neither scope demotes across basenames.
    a = row(
        trial_id="a",
        model="A.gguf",
        tps="40.0",
        agentic="0.7",
        coding="0.7",
        config_json=cfg_json(MODEL="A"),
    )
    b = row(
        trial_id="b",
        model="B.gguf",
        tps="30.0",
        agentic="0.6",
        coding="0.6",
        config_json=cfg_json(MODEL="B"),
    )
    for scope in ("model", "bucket"):
        out = {r["trial_id"]: r["status"] for r in recompute.recompute_rows([a, b], scope=scope)}
        assert out == {"a": "on_front", "b": "on_front"}
    # Default scope is per-model.
    assert recompute.SCOPES[0] == "model"


def test_model_scope_respects_bucket_isolation(store):
    # Same model, two buckets: the 8GB point must not be demoted by the 16GB
    # point in the per-model lens either (domination never crosses budgets).
    rows = [
        row(trial_id="a8", tps="30.0", agentic="0.6", coding="0.6"),
        row(trial_id="b16", memory_gb="16.0", tps="40.0", agentic="0.7", coding="0.7"),
    ]
    out = {r["trial_id"]: r["status"] for r in recompute.recompute_rows(rows, scope="model")}
    assert out == {"a8": "on_front", "b16": "on_front"}


def test_invalid_scope_rejected():
    with pytest.raises(ValueError):
        recompute.recompute_rows([], scope="machine")


def test_recompute_rows_pure_and_idempotent():
    rows = [
        row(
            trial_id="a",
            model="A.gguf",
            tps="30.0",
            agentic="0.6",
            coding="0.6",
            config_json=cfg_json(MODEL="A"),
        ),
        row(
            trial_id="b",
            model="B.gguf",
            tps="40.0",
            agentic="0.7",
            coding="0.7",
            config_json=cfg_json(MODEL="B"),
        ),
    ]
    first = recompute.recompute_rows(rows)
    # Input untouched (pure function).
    assert rows[0]["status"] == "incomplete" and rows[1]["status"] == "incomplete"
    assert recompute.recompute_rows(first) == first


def test_cli_runs_from_repo_root(store):
    write_store(store, [row(trial_id="a", tps="30.0", agentic="0.6", coding="0.6")])
    proc = subprocess.run(
        [sys.executable, "scripts/recompute_status.py", str(store)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "statuses refreshed" in proc.stdout
    assert read_store(store) == {"a": "on_front"}


def test_cli_model_scope_rewrites_with_per_model_statuses(store):
    # ADR 0017: the default scope is per-model and persists; --scope model
    # rewrites the store with same-basename-only statuses.
    a = row(
        trial_id="a",
        model="A.gguf",
        tps="40.0",
        agentic="0.7",
        coding="0.7",
        config_json=cfg_json(MODEL="A"),
    )
    b = row(
        trial_id="b",
        model="B.gguf",
        tps="30.0",
        agentic="0.6",
        coding="0.6",
        config_json=cfg_json(MODEL="B"),
    )
    write_store(store, [a, b])
    proc = subprocess.run(
        [sys.executable, "scripts/recompute_status.py", "--scope", "model", str(store)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert read_store(store) == {"a": "on_front", "b": "on_front"}


def test_cli_help_exits_zero():
    proc = subprocess.run(
        [sys.executable, "scripts/recompute_status.py", "--help"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0


def test_morris_screen_rows_never_join_domination(store):
    # ADR 0016: a reps=1 Morris screen point must not compete with, demote,
    # or merge into real Trials — even in the same bucket and model.
    real = row(trial_id="real", tps="30.0", agentic="0.6", coding="0.6")
    screen = row(
        trial_id="scr",
        evaluation_profile="morris-screen",
        category="morris-screen",
        tps="99.0",
        agentic="",
        coding="",
        outcome="OK",
    )
    write_store(store, [screen, real])
    run.recompute_statuses(store)
    assert read_store(store) == {"real": "on_front", "scr": "incomplete"}


def test_relabel_watchdog_kills_legacy_policy_rows():
    """Issue #72: legacy NVML device-wide policy kills relabel to WATCHDOG_KILL."""
    rows = [
        row(
            trial_id="policy",
            status="rejected",
            outcome="MODEL_REJECTED",
            diagnostic="VRAM_LIMIT_EXCEEDED",
        ),
    ]
    out = recompute.relabel_watchdog_kills(rows)
    assert out[0]["outcome"] == "WATCHDOG_KILL"
    assert out[0]["status"] == "rejected"  # still failed: out of the front
    assert out[0]["diagnostic"].startswith("WATCHDOG_KILL")
    assert "nvml-device-wide" in out[0]["diagnostic"]
    assert rows[0]["outcome"] == "MODEL_REJECTED"  # pure: input untouched


def test_relabel_watchdog_kills_leaves_honest_rows():
    """Genuine model rejects (preflight, TPS floor) and honest kills are untouched."""
    honest = row(
        trial_id="honest",
        status="rejected",
        outcome="WATCHDOG_KILL",
        diagnostic="WATCHDOG_KILL scope=cuda_free",
    )
    preflight = row(
        trial_id="preflight",
        status="rejected",
        outcome="MODEL_REJECTED",
        diagnostic="VRAM_PREFLIGHT est=8108MB > limit=7900MB",
    )
    out = recompute.relabel_watchdog_kills([honest, preflight])
    assert out[0]["outcome"] == "WATCHDOG_KILL"
    assert out[0]["diagnostic"] == "WATCHDOG_KILL scope=cuda_free"
    assert out[1]["outcome"] == "MODEL_REJECTED"
    assert out[1]["diagnostic"] == "VRAM_PREFLIGHT est=8108MB > limit=7900MB"


def test_relabel_watchdog_kills_idempotent():
    """Relabeling twice changes nothing the second time."""
    rows = [
        row(
            trial_id="policy",
            status="rejected",
            outcome="MODEL_REJECTED",
            diagnostic="VRAM_LIMIT_EXCEEDED",
        ),
    ]
    once = recompute.relabel_watchdog_kills(rows)
    twice = recompute.relabel_watchdog_kills(once)
    assert twice == once


def test_relabel_watchdog_kills_store_migration(store):
    """Issue #72: store migration rewrites the outcome and holds on rerun."""
    write_store(
        store,
        [
            row(
                trial_id="policy",
                status="rejected",
                outcome="MODEL_REJECTED",
                diagnostic="VRAM_LIMIT_EXCEEDED",
            ),
            row(
                trial_id="preflight",
                status="rejected",
                outcome="MODEL_REJECTED",
                diagnostic="VRAM_PREFLIGHT est=8108MB > limit=7900MB",
            ),
        ],
    )
    assert run.relabel_watchdog_kills(store) == 1
    rows = {r["trial_id"]: r for r in run.read_rows(store)}
    assert rows["policy"]["outcome"] == "WATCHDOG_KILL"
    assert rows["policy"]["status"] == "rejected"
    assert "nvml-device-wide" in rows["policy"]["diagnostic"]
    assert rows["preflight"]["outcome"] == "MODEL_REJECTED"
    assert run.relabel_watchdog_kills(store) == 0
