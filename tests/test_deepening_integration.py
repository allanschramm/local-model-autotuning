"""Integration-level regression tests for the deepening refactor.

The pure modules are unit-tested in ``test_memory_budget.py`` and
``test_trial_verdict.py``. These tests cover the *integration*: the seams where
a wiring mistake silently changes behaviour even though both halves pass their
own unit tests.

Each test here corresponds to a real regression found in review:

* ``test_validation_trial_runs_only_the_quick_smoke`` — the rebind block in
  ``run_trial`` originally forgot ``mini_swe_agent``, so a ``--validation`` run
  with mini-swe-agent in its norm could still spawn that benchmark, violating
  "validation = smoke gates only".
* ``test_autoloop_gate_does_not_narrow_against_the_device_ceiling`` — routing the
  hill-climb screen through ``MemoryBudget`` applied the physical-keepout clamp
  that the original ``est > vram_limit`` never applied, silently shrinking the
  search space on machines whose baseline sits above the ceiling.
* ``test_preflight_ports_delegate_to_the_budget`` — both memory preflights must
  read their figures from the object, so the budget, the effective number and
  the rejection message cannot drift apart.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

import autoloop
from autoresearch.core import llama_runner
from autoresearch.runners.evaluation import (
    BenchmarkSelection,
    _intent_model_kind,
)
from autoresearch.runners.trial_verdict import ModelKind

REPO_ROOT = Path(__file__).resolve().parents[1]


def _stub_intent():
    """A ServerIntent-shaped object the preflight stubs can swallow."""
    from autoresearch.core.llama_runner import ServerIntent

    return ServerIntent(
        model_path=Path("nope.gguf"),
        ctx_size=4096,
        kv_cache="q4_0",
        flash_attn=True,
    )


# ── BLOCKER 1: validation must run the quick smoke and nothing else ───────────


def test_validation_trial_runs_only_the_quick_smoke(capsys):
    """Drive the real run_trial with a fake server; only the quick tier may run.

    This exercises the whole selection → rebind → dispatch chain rather than
    asserting on source text: a grep for ``mini_swe_agent =
    selection.mini_swe_agent`` passes on code that never executes and fails on
    a benign refactor.

    Every benchmark entry point except the quick smoke is wired to a sentinel.
    The sentinels raise, and `run_trial` catches benchmark failures in its own
    try/except — so a raise alone would NOT fail the test. The assertion is on
    the Trial's own log line instead: `run_trial` announces mini-swe-agent right
    before spawning it, and that line only appears if the flag leaked.
    """
    from autoresearch.runners import evaluation as ev

    # `from_config` returns (intent, norm). `norm` is what carries the
    # operator's benchmark selection into the resolution, so the flag has to be
    # there — exactly as a real Trial receives it.
    norm = {
        "mini_swe_agent": True,
        "include_agentic_quick": True,
        "validation": True,
    }
    cfg = {
        "MODEL": "does-not-matter.gguf",
        "CTX_SIZE": 4096,
        "VRAM_LIMIT_MB": 100000,
    }

    quick_runs: list[int] = []

    def _quick(*_a, **_kw):
        quick_runs.append(1)
        return {"score": 0.5, "total": 2, "passed": 1, "elapsed_sec": 1.0, "task_results": []}

    def _forbidden(name):
        def _f(*_a, **_kw):  # pragma: no cover - must never run
            raise AssertionError(f"{name} ran during a validation Trial")

        return _f

    class _FakeServer:
        """Minimal stand-in for LlamaServerRunner's context-manager surface."""

        def __init__(self, intent, **_kw):
            self.intent = intent
            self.port = 1
            self.vram_killed = False
            self.vram_kill_reason = ""
            self.peak_vram_mb = 0.0

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

    runner = ev.ExperimentRunner.__new__(ev.ExperimentRunner)
    runner.models_dir = REPO_ROOT / "models"
    runner._idle_gpu_c = None
    runner.thermal_wait = False

    with (
        patch.object(ev.ServerIntent, "from_config", return_value=(_stub_intent(), norm)),
        patch.object(ev, "get_quick_tier_tasks", return_value=["t1", "t2"]),
        patch.object(ev, "_run_bench_precheck", return_value=True),
        patch.object(ev, "LlamaServerRunner", _FakeServer),
        patch.object(ev, "LlamaClient", lambda *_a, **_kw: object()),
        patch.object(ev, "run_agentic_eval", _quick),
        patch.object(ev, "run_mini_swe_agent_eval", _forbidden("run_mini_swe_agent_eval")),
        patch.object(ev, "run_agentic_coding_eval", _forbidden("run_agentic_coding_eval")),
        patch.object(ev, "run_coding", _forbidden("run_coding")),
    ):
        res = runner.run_trial(cfg, skip_bench=True)

    out = capsys.readouterr().out

    assert "DM-Code-Agent scoreboard" not in out, (
        "validation spawned the mini-swe-agent benchmark; the rebind leaked"
    )
    assert quick_runs, "the quick smoke should have run; otherwise this proves nothing"
    assert res.agentic_tier == "quick", (
        f"only the quick tier may run in validation, got {res.agentic_tier!r}"
    )
    # `agentic_coding_val` defaults to None (never scored), not 0.0 — asserting
    # on the wrong sentinel would pass even if SWE-lite had run.
    assert res.agentic_coding_val is None, "validation must not run SWE-lite"
    assert res.coding_val == 0.0, "validation must not run coding-10"


def test_non_validation_selection_keeps_mini_swe_agent():
    """The counter-case: without validation, the request survives."""
    sel = BenchmarkSelection.resolved(mini_swe_agent=True, is_validation=False)
    assert sel.mini_swe_agent is True
    assert sel.agentic_quick is False and sel.agentic_full is False


def test_validation_selection_forces_every_non_smoke_tier_off():
    sel = BenchmarkSelection.resolved(
        mini_swe_agent=True,
        agentic_quick=True,
        agentic_full=True,
        agentic_coding=True,
        include_coding=True,
        is_validation=True,
    )
    assert sel.mini_swe_agent is False
    assert sel.agentic_full is False
    assert sel.agentic_coding is False
    assert sel.include_coding is False
    assert sel.agentic_quick is True, "the quick smoke is the validation default"


# ── BLOCKER 2: the hill-climb screen must not self-narrow ────────────────────


def test_budget_can_skip_the_physical_clamp():
    """apply_physical_clamp=False must leave the configured budget untouched."""
    budget = llama_runner._build_memory_budget(
        16000.0,
        apply_physical_clamp=False,
    )
    assert budget.vram_limit_mb == pytest.approx(16000.0)
    assert budget.fits(10000.0) is True
    assert budget.fits(20000.0) is False


def test_budget_with_clamp_would_narrow_the_same_budget():
    """The contrast that makes blocker 2 concrete."""
    with patch.object(llama_runner, "_safe_detect_total_vram_mb", return_value=8188.0):
        budget = llama_runner._build_memory_budget(16000.0, apply_physical_clamp=True)
    assert budget.vram_limit_mb == pytest.approx(8188.0 - 256.0)
    assert budget.fits(10000.0) is False


def test_autoloop_gate_does_not_narrow_against_the_device_ceiling():
    """The autoloop screen must keep the operator's configured budget.

    A candidate at 10 GB is admissible on a machine whose physical VRAM is
    smaller only if the operator configured it that way; the search screen is
    not allowed to veto that on its own.
    """
    with (
        patch.object(autoloop, "estimate_vram_mb", return_value=10000.0),
        patch.object(autoloop, "resolve_n_cpu_moe", return_value=(None, None)),
        patch.object(autoloop, "preflight_host_ok", return_value=True),
    ):
        ok = autoloop.preflight_vram_ok({"MODEL": "fake.gguf"}, 16000.0)
    assert ok is True, "the screen must not apply the device-keepout clamp"

    with (
        patch.object(autoloop, "estimate_vram_mb", return_value=20000.0),
        patch.object(autoloop, "resolve_n_cpu_moe", return_value=(None, None)),
        patch.object(autoloop, "preflight_host_ok", return_value=True),
    ):
        ok_over = autoloop.preflight_vram_ok({"MODEL": "fake.gguf"}, 16000.0)
    assert ok_over is False, "a genuine over-budget estimate must still fail"


# ── both preflights must read from the object, not re-derive ─────────────────


def test_preflight_ports_delegate_to_the_budget():
    """The host preflight's numbers must come from MemoryBudget, not from a
    second implementation of the same rule.

    Pins the wire-up that ticket 02 promises: a divergent inline formula would
    pass both this module's own tests and the preflight's.
    """
    ram = 32000.0
    with (
        patch("autoresearch.core.hardware.detect_host_ram_mb", return_value=ram),
        patch("autoresearch.core.hardware.is_unified_memory_host", return_value=False),
        patch.object(llama_runner, "estimate_host_memory_mb", return_value=1000.0),
    ):
        ok, est, budget, _reason = llama_runner.preflight_host_memory(
            model_path=Path("nope.gguf"),
            ctx_size=4096,
            headroom_mb=4096.0,
        )

    assert ok is True
    assert est == pytest.approx(1000.0)
    assert budget == pytest.approx(ram - 4096.0)


def test_host_reject_message_is_byte_equivalent():
    with (
        patch("autoresearch.core.hardware.detect_host_ram_mb", return_value=32000.0),
        patch("autoresearch.core.hardware.is_unified_memory_host", return_value=False),
        patch.object(llama_runner, "estimate_host_memory_mb", return_value=30000.0),
    ):
        ok, _est, budget, reason = llama_runner.preflight_host_memory(
            model_path=Path("nope.gguf"),
            ctx_size=4096,
            headroom_mb=4096.0,
        )

    assert ok is False
    assert budget == pytest.approx(27904.0)
    assert reason == (
        "HOST_MEMORY_PREFLIGHT est=30000MB > budget=27904MB "
        "(ram=32000 headroom=4096 class=discrete_gpu)"
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
        ctx_size=4096,
        kv_cache="q4_0",
        flash_attn=True,
    )
    assert _intent_model_kind(intent) is ModelKind.UNKNOWN
