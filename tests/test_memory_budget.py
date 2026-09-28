"""E2E acceptance for the MemoryBudget policy (ticket 01).

The whole point of the deepening is that the memory policy is now testable as
*data*, not as patched module globals. These tests exercise the object through
its public interface only — no monkeypatching of hardware probes, no env
mutation, no GPU.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from autoresearch.core.memory_budget import (
    DEFAULT_PHYSICAL_VRAM_KEEPOUT_MB,
    DEFAULT_VRAM_HEADROOM_MB,
    DEFAULT_VRAM_LIMIT_MB,
    MemoryBudget,
)


def _budget(**kw) -> MemoryBudget:
    """A budget with everything explicit unless the test overrides it."""
    base = {
        "vram_limit_mb": 7900.0,
        "physical_keepout_mb": 256.0,
        "shared_vram_limit_mb": 2048.0,
        "cuda_free_floor_mb": 256.0,
        "vram_headroom_mb": 512.0,
        "free_clamp_enabled": False,
        "total_vram_mb": 8188.0,
    }
    base.update(kw)
    return MemoryBudget.resolve(**base)


# --- precedence: explicit > defaults mapping > module default -----------------


def test_explicit_arg_wins_over_defaults_mapping():
    b = MemoryBudget.resolve(
        vram_limit_mb=6000.0,
        defaults={"VRAM_LIMIT_MB": 4000.0},
    )
    assert b.vram_limit_mb == pytest.approx(6000.0)


def test_defaults_mapping_wins_over_module_default():
    b = MemoryBudget.resolve(defaults={"VRAM_LIMIT_MB": 4000.0})
    assert b.vram_limit_mb == pytest.approx(4000.0)


def test_module_default_applies_when_nothing_else_specifies():
    b = MemoryBudget.resolve()
    assert b.vram_limit_mb == pytest.approx(DEFAULT_VRAM_LIMIT_MB)
    assert b.vram_headroom_mb == pytest.approx(DEFAULT_VRAM_HEADROOM_MB)
    assert b.physical_keepout_mb == pytest.approx(DEFAULT_PHYSICAL_VRAM_KEEPOUT_MB)


# --- the physical-keepout clamp ----------------------------------------------


def test_limit_above_physical_minus_keepout_is_clamped():
    b = _budget(vram_limit_mb=16000.0, total_vram_mb=8188.0, physical_keepout_mb=256.0)
    assert b.vram_limit_mb == pytest.approx(8188.0 - 256.0)


def test_limit_below_physical_minus_keepout_is_left_alone():
    b = _budget(vram_limit_mb=6000.0, total_vram_mb=8188.0, physical_keepout_mb=256.0)
    assert b.vram_limit_mb == pytest.approx(6000.0)


def test_unknown_physical_total_does_not_clamp():
    b = _budget(vram_limit_mb=16000.0, total_vram_mb=None)
    assert b.vram_limit_mb == pytest.approx(16000.0)
    assert b.kill_ceil_mb == pytest.approx(16000.0)


def test_zero_physical_total_does_not_clamp():
    b = _budget(vram_limit_mb=16000.0, total_vram_mb=0.0)
    assert b.vram_limit_mb == pytest.approx(16000.0)


def test_kill_ceil_is_min_of_limit_and_physical_minus_keepout():
    b = _budget(vram_limit_mb=3000.0, total_vram_mb=8188.0, physical_keepout_mb=256.0)
    assert b.kill_ceil_mb == pytest.approx(3000.0)


# --- the effective budget: free-at-start clamp -------------------------------


def test_free_clamp_off_by_default_uses_configured():
    b = _budget(free_clamp_enabled=False)
    assert b.effective_vram_limit_mb(free_vram_mb=4000.0) == pytest.approx(7900.0)


def test_free_clamp_on_shrinks_to_free_minus_headroom():
    b = _budget(free_clamp_enabled=True, vram_limit_mb=7900.0, vram_headroom_mb=512.0)
    assert b.effective_vram_limit_mb(free_vram_mb=4000.0) == pytest.approx(4000.0 - 512.0)


def test_free_clamp_never_exceeds_configured():
    b = _budget(free_clamp_enabled=True, vram_limit_mb=3000.0, vram_headroom_mb=512.0)
    assert b.effective_vram_limit_mb(free_vram_mb=8000.0) == pytest.approx(3000.0)


def test_unknown_free_vram_leaves_budget_unchanged():
    b = _budget(free_clamp_enabled=True)
    assert b.effective_vram_limit_mb(free_vram_mb=None) == pytest.approx(7900.0)
    assert b.effective_vram_limit_mb(free_vram_mb=0.0) == pytest.approx(7900.0)


def test_explicit_headroom_overrides_the_budget_default():
    b = _budget(free_clamp_enabled=True, vram_headroom_mb=512.0)
    assert b.effective_vram_limit_mb(free_vram_mb=4000.0, headroom_mb=100.0) == pytest.approx(
        3900.0
    )


# --- MoE offload exception ----------------------------------------------------


def test_moe_expert_offload_uses_configured_even_with_clamp_on():
    """n_cpu_moe > 0 ignores free-at-start: OS-reserved VRAM false-rejects it."""
    b = _budget(free_clamp_enabled=True, vram_limit_mb=7900.0)
    assert b.effective_vram_limit_mb(n_cpu_moe=31, free_vram_mb=3000.0) == pytest.approx(7900.0)


def test_moe_offload_zero_is_not_an_exception():
    b = _budget(free_clamp_enabled=True, vram_limit_mb=7900.0, vram_headroom_mb=512.0)
    assert b.effective_vram_limit_mb(n_cpu_moe=0, free_vram_mb=4000.0) == pytest.approx(3488.0)


# --- fits / reject_reason ----------------------------------------------------


def test_fits_when_estimate_is_under_budget():
    b = _budget(vram_limit_mb=7900.0)
    assert b.fits(7000.0) is True


def test_fails_when_estimate_is_over_budget():
    b = _budget(vram_limit_mb=7900.0)
    assert b.fits(9000.0) is False


def test_reject_reason_is_none_when_it_fits():
    b = _budget(vram_limit_mb=7900.0)
    assert b.reject_reason(7000.0) is None


def test_reject_reason_names_budget_when_clamp_did_not_apply():
    b = _budget(vram_limit_mb=7900.0, free_clamp_enabled=False)
    reason = b.reject_reason(9000.0)
    assert reason is not None
    assert "7900MB" in reason


def test_reject_reason_names_configured_and_effective_when_clamp_applied():
    b = _budget(vram_limit_mb=7900.0, free_clamp_enabled=True, vram_headroom_mb=512.0)
    reason = b.reject_reason(9000.0, free_vram_mb=4000.0)
    assert reason is not None
    assert "configured=7900MB" in reason
    assert "effective=3488MB" in reason
    assert "free=4000MB" in reason
    assert "headroom=512MB" in reason


def test_moe_offload_is_measured_against_the_configured_budget():
    b = _budget(vram_limit_mb=7900.0, free_clamp_enabled=True)
    assert b.reject_reason(6000.0, n_cpu_moe=31, free_vram_mb=3000.0) is None


# --- immutability ------------------------------------------------------------


def test_budget_is_immutable():
    b = _budget()
    with pytest.raises(FrozenInstanceError):
        b.vram_limit_mb = 1.0  # type: ignore[misc]


# --- the host side (for_host / host_fits / host_reject_reason) ---------------
#
# The host figures are injected already-resolved, so these tests never touch
# env, config, or a RAM probe — that resolution is the caller's job and the
# purity of this module is exactly what makes it testable here.


def _host(ram: float | None, *, unified: bool, headroom: float) -> MemoryBudget:
    return MemoryBudget.for_host(
        ram,
        unified=unified,
        headroom_mb=headroom,
        memory_class="unified_memory" if unified else "discrete_gpu",
    )


def test_host_budget_is_ram_minus_headroom():
    h = _host(32000.0, unified=False, headroom=4096.0)
    assert h.host_budget_mb == pytest.approx(27904.0)
    assert h.host_headroom_mb == pytest.approx(4096.0)
    assert h.host_ram_mb == pytest.approx(32000.0)
    assert h.host_memory_class == "discrete_gpu"


def test_host_budget_never_goes_negative():
    h = _host(1000.0, unified=False, headroom=99999.0)
    assert h.host_budget_mb == pytest.approx(0.0)


def test_unknown_ram_yields_no_budget():
    h = _host(None, unified=False, headroom=4096.0)
    assert h.host_budget_mb is None
    assert h.host_fits(1.0) is False, "unknown RAM must fail closed"


def test_zero_ram_yields_no_budget():
    assert _host(0.0, unified=False, headroom=4096.0).host_budget_mb is None


def test_host_fits_under_budget():
    assert _host(32000.0, unified=False, headroom=4096.0).host_fits(20000.0) is True


def test_host_fails_over_budget():
    assert _host(32000.0, unified=False, headroom=4096.0).host_fits(30000.0) is False


def test_host_reject_reason_is_none_when_it_fits():
    assert _host(32000.0, unified=False, headroom=4096.0).host_reject_reason(20000.0) is None


def test_host_reject_reason_carries_ram_headroom_and_class():
    """Byte-equivalent to the historical inline message, tokens included."""
    h = _host(32000.0, unified=False, headroom=4096.0)
    reason = h.host_reject_reason(30000.0)
    assert reason == (
        "HOST_MEMORY_PREFLIGHT est=30000MB > budget=27904MB "
        "(ram=32000 headroom=4096 class=discrete_gpu)"
    )


def test_host_reject_reason_for_unknown_ram_uses_the_ram_unknown_form():
    h = _host(None, unified=True, headroom=0.0)
    reason = h.host_reject_reason(1234.0)
    assert reason == "HOST_MEMORY_PREFLIGHT ram_unknown class=unified_memory est=1234MB"


def test_for_host_does_not_read_env(monkeypatch):
    """Purity guard: the object must not consult AUTORESEARCH_HOST_HEADROOM_MB."""
    monkeypatch.setenv("AUTORESEARCH_HOST_HEADROOM_MB", "999999")
    monkeypatch.setenv("AUTORESEARCH_VRAM_LIMIT_MB", "1")
    h = _host(32000.0, unified=False, headroom=4096.0)
    assert h.host_budget_mb == pytest.approx(27904.0)
