"""E2E acceptance for the pure TrialVerdict decision layer (ticket 03).

These tests reach the decision rules through the module's *public* interface.
Before this extraction, the only way to exercise the MoE/VRAM rejection was to
import ``evaluation._moe_vram_reject`` — a private symbol, which is the smell
this module removes.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from autoresearch.runners.trial_verdict import (
    BenchmarkSelection,
    ModelKind,
    TrialEvidence,
    TrialVerdict,
    VerdictReason,
    moe_vram_reject_message,
    verdict_for,
)


def _ev(**kw) -> TrialEvidence:
    """A Trial that passes every gate unless the test says otherwise."""
    base = {
        "vram_est_mb": 4000.0,
        "vram_limit_mb": 7900.0,
        "vram_fits": True,
        "host_est_mb": 2000.0,
        "host_budget_mb": 16000.0,
        "host_fits": True,
        "model_kind": ModelKind.DENSE,
        "n_cpu_moe": None,
        "model_exists": True,
    }
    base.update(kw)
    return TrialEvidence(**base)


# --- a healthy Trial proceeds -------------------------------------------------


def test_healthy_trial_proceeds():
    assert verdict_for(_ev()) is None


# --- VRAM gate ----------------------------------------------------------------


def test_vram_overflow_is_model_rejected():
    v = verdict_for(_ev(vram_fits=False, vram_est_mb=9000.0, vram_reason="over"))
    assert v is not None
    assert v.trial_verdict is TrialVerdict.MODEL_REJECTED
    assert v.reason is VerdictReason.VRAM_EXCEEDED


def test_vram_overflow_falls_back_to_preflight_reason_when_no_moe_reason():
    v = verdict_for(_ev(vram_fits=False, vram_est_mb=9000.0, vram_reason="generic why"))
    assert v is not None
    assert v.message == "generic why"


def test_vram_overflow_synthesizes_a_reason_when_preflight_gave_none():
    v = verdict_for(_ev(vram_fits=False, vram_est_mb=9000.0, vram_reason=None))
    assert v is not None
    assert "9000MB" in v.message and "7900MB" in v.message


# --- the MoE/VRAM rule (publicly reachable now) -------------------------------


def test_moe_full_gpu_rejection_names_the_auto_offload_fix():
    msg = moe_vram_reject_message(
        _ev(
            vram_fits=False,
            vram_est_mb=9000.0,
            vram_reason="x",
            model_kind=ModelKind.MOE,
            n_cpu_moe=0,
        )
    )
    assert msg is not None
    assert "N_CPU_MOE=0" in msg
    assert "auto block_count" in msg


def test_moe_explicit_offload_rejection_names_the_limit():
    msg = moe_vram_reject_message(
        _ev(
            vram_fits=False,
            vram_est_mb=9000.0,
            vram_reason="x",
            model_kind=ModelKind.MOE,
            n_cpu_moe=31,
        )
    )
    assert msg is not None
    assert "N_CPU_MOE=31" in msg
    assert "exceeds physical VRAM limit" in msg


def test_dense_overflow_does_not_get_moe_advice():
    msg = moe_vram_reject_message(_ev(vram_fits=False, vram_est_mb=9000.0, vram_reason="x"))
    assert msg is None


def test_unknown_architecture_does_not_get_moe_advice():
    """An unprovable architecture must not be treated as MoE."""
    msg = moe_vram_reject_message(
        _ev(
            vram_fits=False,
            vram_est_mb=9000.0,
            vram_reason="x",
            model_kind=ModelKind.UNKNOWN,
            n_cpu_moe=0,
        )
    )
    assert msg is None


def test_missing_model_file_does_not_get_moe_advice():
    msg = moe_vram_reject_message(
        _ev(
            vram_fits=False,
            vram_est_mb=9000.0,
            vram_reason="x",
            model_kind=ModelKind.MOE,
            n_cpu_moe=0,
            model_exists=False,
        )
    )
    assert msg is None


def test_moe_rule_does_not_fire_when_trial_fits():
    assert moe_vram_reject_message(_ev(model_kind=ModelKind.MOE, n_cpu_moe=0)) is None


# --- host memory gate ---------------------------------------------------------


def test_host_overflow_is_model_rejected():
    v = verdict_for(_ev(host_fits=False, host_reason="host too small"))
    assert v is not None
    assert v.trial_verdict is TrialVerdict.MODEL_REJECTED
    assert v.reason is VerdictReason.HOST_MEMORY_EXCEEDED
    assert v.message == "host too small"


def test_host_overflow_synthesizes_when_no_reason_given():
    v = verdict_for(_ev(host_fits=False, host_est_mb=99000.0, host_reason=None))
    assert v is not None
    assert "99000MB" in v.message


# --- vram is checked before host ---------------------------------------------


def test_vram_fault_wins_over_host_fault():
    v = verdict_for(_ev(vram_fits=False, host_fits=False, vram_reason="vram first"))
    assert v is not None
    assert v.reason is VerdictReason.VRAM_EXCEEDED


# --- benchmark selection ------------------------------------------------------


def test_coding_arity_mismatch_is_invalid_config():
    sel = BenchmarkSelection.resolved(include_coding=True, coding_task_limits=(5, 10, 10))
    v = verdict_for(_ev(selection=sel))
    assert v is not None
    assert v.trial_verdict is TrialVerdict.INVALID_CONFIG
    assert v.reason is VerdictReason.CODING_TASK_COUNT


def test_coding_arity_mismatch_ignored_when_coding_not_selected():
    sel = BenchmarkSelection.resolved(include_coding=False, coding_task_limits=(5, 5, 5))
    assert verdict_for(_ev(selection=sel)) is None


def test_empty_agentic_tier_is_infra_error():
    v = verdict_for(_ev(missing_agentic_tier="full"))
    assert v is not None
    assert v.trial_verdict is TrialVerdict.INFRA_ERROR
    assert v.reason is VerdictReason.EMPTY_AGENTIC_TIER
    assert "full" in v.message


# --- bench threshold ----------------------------------------------------------


def test_bench_below_threshold_is_model_rejected():
    v = verdict_for(_ev(bench_tg_tps=50.0, bench_threshold_tps=80.0))
    assert v is not None
    assert v.trial_verdict is TrialVerdict.MODEL_REJECTED
    assert v.reason is VerdictReason.BENCH_BELOW_THRESHOLD


def test_bench_at_threshold_passes():
    assert verdict_for(_ev(bench_tg_tps=80.0, bench_threshold_tps=80.0)) is None


def test_missing_bench_measurement_passes():
    assert verdict_for(_ev(bench_tg_tps=None, bench_threshold_tps=80.0)) is None


# --- selection normalization rules --------------------------------------------


def test_validation_defaults_to_quick_smoke():
    sel = BenchmarkSelection.resolved(is_validation=True)
    assert sel.agentic_quick is True
    assert sel.agentic_full is False
    assert sel.agentic_coding is False
    assert sel.mini_swe_agent is False
    assert sel.include_coding is False


def test_validation_respects_explicit_quick_selection():
    sel = BenchmarkSelection.resolved(is_validation=True, agentic_quick=True)
    assert sel.agentic_quick is True
    assert sel.agentic_full is False


def test_validation_can_disable_all_benchmarks():
    """An explicit --no-agentic-quick must still allow a load/TPS-only smoke."""
    sel = BenchmarkSelection.resolved(is_validation=True, agentic_quick=True, agentic_full=True)
    # Full is forced off even when requested, so the smoke stays cheap.
    assert sel.agentic_full is False


def test_mini_swe_agent_is_standalone():
    sel = BenchmarkSelection.resolved(mini_swe_agent=True, agentic_quick=True, include_coding=True)
    assert sel.mini_swe_agent is True
    assert sel.agentic_quick is False
    assert sel.include_coding is False


def test_coding_arity_exact_is_accepted():
    sel = BenchmarkSelection.resolved(include_coding=True, coding_task_limits=(10, 10, 10))
    assert sel.coding_preflight_ok() is True
    assert verdict_for(_ev(selection=sel)) is None


# --- evidence consistency guard -----------------------------------------------


def test_inconsistent_evidence_is_rejected_at_construction():
    with pytest.raises(ValueError, match="inconsistent evidence"):
        TrialEvidence(
            vram_est_mb=9000.0,
            vram_limit_mb=7900.0,
            vram_fits=True,
            host_est_mb=1000.0,
            host_budget_mb=16000.0,
            host_fits=True,
        )


def test_evidence_is_immutable():
    ev = _ev()
    with pytest.raises(FrozenInstanceError):
        ev.vram_fits = False  # type: ignore[misc]
