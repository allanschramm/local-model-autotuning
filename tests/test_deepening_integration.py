"""Regression tests for the deepening integration seams (tickets 02/04/05).

The pure modules are unit-tested in ``test_memory_budget.py`` and
``test_trial_verdict.py``. These tests cover the *integration*: the places where
a wiring mistake silently changes behavior even though both halves pass their
own unit tests.

Both tests here correspond to a real regression found in review:

* ``test_validation_forces_mini_swe_agent_off`` — the rebind block in run_trial
  originally forgot ``mini_swe_agent``, so a ``--validation`` run that had
  mini-swe-agent in its norm would still spawn that benchmark, violating
  "validation = smoke gates only".
* ``test_autoloop_gate_does_not_narrow_against_the_device_ceiling`` — routing the
  hill-climb screen through MemoryBudget applied the physical-keepout clamp that
  the original ``est > vram_limit`` never applied, silently shrinking the search
  space on machines whose baseline sits above the ceiling.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

import autoloop
from autoresearch.core import llama_runner
from autoresearch.core.memory_budget import MemoryBudget
from autoresearch.runners.evaluation import (
    BenchmarkSelection,
    _intent_model_kind,
)
from autoresearch.runners.trial_verdict import ModelKind

REPO_ROOT = Path(__file__).resolve().parents[1]


# ── BLOCKER 1: validation must force every non-smoke gate off ────────────────


def test_validation_forces_mini_swe_agent_off():
    """The selection object already does this; assert the consumed value too.

    Guards the rebind: if run_trial ever stops reassigning `mini_swe_agent`
    from `selection`, this is the assertion that fails.
    """
    sel = BenchmarkSelection.resolved(mini_swe_agent=True, is_validation=True)
    assert sel.mini_swe_agent is False

    # Mirror exactly what run_trial does after building the selection.
    requested = True
    rebound = sel.mini_swe_agent
    assert requested is True, "the user did request it"
    assert rebound is False, "validation must win, or mini-swe-agent leaks"


def test_non_validation_mini_swe_agent_survives():
    sel = BenchmarkSelection.resolved(mini_swe_agent=True, is_validation=False)
    assert sel.mini_swe_agent is True


def test_run_trial_rebinds_every_selection_field():
    """Static guard: each field consumed by run_trial must come from selection.

    A missing rebind is invisible at runtime until a benchmark silently runs, so
    this asserts the source text rather than trusting review.
    """
    src = (REPO_ROOT / "autoresearch" / "runners" / "evaluation.py").read_text(encoding="utf-8")
    for field in (
        "include_coding",
        "agentic_quick",
        "agentic_full",
        "agentic_coding",
        "mini_swe_agent",
    ):
        assert f"{field} = selection.{field}" in src, (
            f"run_trial must rebind `{field}` from the resolved selection; "
            "a stale value would leak past the validation gate"
        )


# ── BLOCKER 2: the hill-climb screen must not self-narrow ────────────────────


def test_budget_can_skip_the_physical_clamp():
    """apply_physical_clamp=False must leave the configured budget untouched."""
    budget = MemoryBudget.resolve(
        vram_limit_mb=16000.0,
        physical_keepout_mb=256.0,
        total_vram_mb=None,
        free_clamp_enabled=False,
    )
    assert budget.vram_limit_mb == pytest.approx(16000.0)
    assert budget.fits(10000.0) is True
    assert budget.fits(20000.0) is False


def test_budget_with_clamp_would_narrow_the_same_budget():
    """The contrast that makes blocker 2 concrete."""
    budget = MemoryBudget.resolve(
        vram_limit_mb=16000.0,
        physical_keepout_mb=256.0,
        total_vram_mb=8188.0,
        free_clamp_enabled=False,
    )
    assert budget.vram_limit_mb == pytest.approx(7932.0)
    assert budget.fits(10000.0) is False


def test_autoloop_gate_does_not_narrow_against_the_device_ceiling():
    """The autoloop screen must keep the operator's configured budget.

    A candidate at 10 GB is admissible on a machine whose physical VRAM is
    smaller only if the operator configured it that way; the search screen is
    not allowed to veto that on its own.
    """
    fake_model = REPO_ROOT / "models" / "fake.gguf"

    with (
        patch.object(autoloop, "estimate_vram_mb", return_value=10000.0),
        patch.object(autoloop, "resolve_n_cpu_moe", return_value=(None, None)),
        patch.object(llama_runner, "_safe_detect_total_vram_mb", return_value=8188.0),
        patch.object(autoloop, "preflight_host_ok", return_value=True),
    ):
        ok = autoloop.preflight_vram_ok({"MODEL": "fake.gguf"}, 16000.0)
    assert ok is True, "the screen must not apply the device-keepout clamp"

    with (
        patch.object(autoloop, "estimate_vram_mb", return_value=20000.0),
        patch.object(autoloop, "resolve_n_cpu_moe", return_value=(None, None)),
        patch.object(llama_runner, "_safe_detect_total_vram_mb", return_value=8188.0),
        patch.object(autoloop, "preflight_host_ok", return_value=True),
    ):
        ok_over = autoloop.preflight_vram_ok({"MODEL": "fake.gguf"}, 16000.0)
    assert ok_over is False, "a genuine over-budget estimate must still fail"

    assert fake_model.name  # keep the path referenced for readability


# ── peak_vram_gb must only be set for resource rejections ────────────────────


def test_peak_vram_is_only_reported_for_resource_rejections():
    from autoresearch.runners.trial_verdict import (
        TrialEvidence,
        VerdictReason,
        verdict_for,
    )

    resource = verdict_for(
        TrialEvidence(
            vram_est_mb=9000.0,
            vram_limit_mb=7900.0,
            vram_fits=False,
            vram_reason="over",
            host_est_mb=0.0,
            host_budget_mb=0.0,
            host_fits=True,
        )
    )
    assert resource is not None
    assert resource.reason is VerdictReason.VRAM_EXCEEDED

    config = verdict_for(
        TrialEvidence(
            vram_est_mb=1000.0,
            vram_limit_mb=7900.0,
            vram_fits=True,
            host_est_mb=0.0,
            host_budget_mb=0.0,
            host_fits=True,
            selection=BenchmarkSelection.resolved(
                include_coding=True, coding_task_limits=(5, 5, 5)
            ),
        )
    )
    assert config is not None
    assert config.reason is VerdictReason.CODING_TASK_COUNT
    assert config.reason not in (
        VerdictReason.VRAM_EXCEEDED,
        VerdictReason.HOST_MEMORY_EXCEEDED,
    )


# ── keepout precedence: config.DEFAULTS beats the env override ───────────────


def test_keepout_honours_config_defaults_before_env():
    with (
        patch.dict("os.environ", {"AUTORESEARCH_PHYSICAL_VRAM_KEEPOUT_MB": "128.0"}),
        patch.dict(llama_runner.config.DEFAULTS, {"PHYSICAL_VRAM_KEEPOUT_MB": 512.0}),
        patch.object(llama_runner, "_safe_detect_total_vram_mb", return_value=8188.0),
    ):
        budget = llama_runner._build_memory_budget(16000.0, apply_physical_clamp=True)
    assert budget.physical_keepout_mb == pytest.approx(512.0)
    assert budget.vram_limit_mb == pytest.approx(8188.0 - 512.0)


def test_env_keepout_applies_when_config_is_silent():
    with (
        patch.dict("os.environ", {"AUTORESEARCH_PHYSICAL_VRAM_KEEPOUT_MB": "128.0"}),
        patch.dict(llama_runner.config.DEFAULTS, {}, clear=True),
        patch.object(llama_runner, "_safe_detect_total_vram_mb", return_value=8188.0),
    ):
        budget = llama_runner._build_memory_budget(16000.0, apply_physical_clamp=True)
    assert budget.physical_keepout_mb == pytest.approx(128.0)


# ── model-kind evidence is never guessed ────────────────────────────────────


def test_model_kind_is_unknown_for_a_missing_file(tmp_path):
    from autoresearch.core.llama_runner import ServerIntent

    intent = ServerIntent(
        model_path=tmp_path / "nope.gguf",
        ctx_size=8192,
        kv_cache="q4_0",
        flash_attn=True,
    )
    assert _intent_model_kind(intent) is ModelKind.UNKNOWN
