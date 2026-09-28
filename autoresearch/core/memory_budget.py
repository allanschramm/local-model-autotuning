"""MemoryBudget — the single source of truth for a machine's memory policy.

Deepening of the budget-resolution cluster that used to live as seven
order-dependent resolvers inside ``llama_runner`` (``dedicated_vram_kill_ceil``,
``resolve_shared_vram_limit_mb``, ``resolve_cuda_free_floor_mb``,
``resolve_vram_limit_mb``, ``resolve_vram_headroom_mb``,
``free_vram_clamp_enabled``, ``effective_vram_limit_mb``). Callers had to know
the *order* in which to call them; this module internalizes that order.

Design constraints (see ``docs/adr/0018-rust-native-core-phase0.md`` and the
architecture review):

* **Pure.** No env reads, no subprocess, no GPU probe, no filesystem. The
  physical VRAM total is *injected* by the caller. That is what makes the whole
  policy table-testable without monkeypatching module globals — the old shape
  forced tests to patch ``llama_runner.detect_total_vram_mb``.
* **Immutable.** A Budget is a snapshot of policy at Trial start, not a live
  view of the machine.
* **Locality.** One implementation, one place. The autoloop and the evaluation
  runner both consume this object instead of re-deriving the rule.

This module deliberately does NOT own the *runtime* VRAM kill guard (the device
sampling sampler in ``LlamaServerRunner``) nor the preflight estimate itself.
It owns only: how much memory a Trial is allowed to use, and why a config that
asks for more is refused.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "DEFAULT_CUDA_FREE_FLOOR_MB",
    "DEFAULT_PHYSICAL_VRAM_KEEPOUT_MB",
    "DEFAULT_SHARED_VRAM_LIMIT_MB",
    "DEFAULT_VRAM_HEADROOM_MB",
    "DEFAULT_VRAM_LIMIT_MB",
    "MemoryBudget",
]

# Defaults mirror the historical llama_runner constants. Kept here (not
# imported) so this module stays free of any llama_runner import cycle; the
# llama_runner constants remain exported for backwards compatibility.
DEFAULT_VRAM_LIMIT_MB = 7900.0
"""Configured VRAM budget when nothing else specifies one."""

DEFAULT_VRAM_HEADROOM_MB = 512.0
"""Safety margin subtracted from free VRAM at Trial start (issue #10)."""

DEFAULT_PHYSICAL_VRAM_KEEPOUT_MB = 256.0
"""Dedicated VRAM left free so WDDM CUDA Sysmem Fallback never arms."""

DEFAULT_SHARED_VRAM_LIMIT_MB = 2048.0
"""Absolute Shared-GPU kill ceiling (WDDM host-map thrash guard)."""

DEFAULT_CUDA_FREE_FLOOR_MB = 256.0
"""CUDA-free kill floor used when the CUDA-free probe is available."""


@dataclass(frozen=True)
class MemoryBudget:
    """Resolved, immutable memory policy for one Trial.

    Build it with :meth:`resolve` (the pure policy application) and hand the
    result to the preflight path. Every field is already clamped — callers never
    re-derive anything.
    """

    vram_limit_mb: float
    """Configured VRAM budget after the physical-keepout clamp."""

    physical_keepout_mb: float
    """Dedicated VRAM kept free on the device."""

    shared_vram_limit_mb: float
    """Shared-GPU kill ceiling."""

    cuda_free_floor_mb: float
    """CUDA-free kill floor."""

    vram_headroom_mb: float
    """Margin subtracted from free-at-start VRAM."""

    free_clamp_enabled: bool
    """Whether the free-at-start clamp applies (operator opt-in)."""

    total_vram_mb: float | None = None
    """Physical dedicated VRAM, when the caller could detect it."""

    # --- host-memory policy (populated by :meth:`for_host`) ---
    host_budget_mb: float | None = None
    """Usable host RAM after headroom, or None when RAM could not be detected."""

    host_headroom_mb: float = 0.0
    """The headroom that was subtracted to produce ``host_budget_mb``."""

    host_ram_mb: float | None = None
    """Total physical RAM, kept for the rejection message."""

    host_memory_class: str = ""
    """``unified`` or ``discrete_gpu``, per the hardware module's MEMORY_CLASS_*.
    Kept so the rejection message is byte-identical to the historical one."""

    unified_memory: bool = False
    """Whether the host is unified-memory (Apple silicon) rather than discrete."""

    @classmethod
    def for_host(
        cls,
        ram_mb: float | None,
        *,
        unified: bool,
        headroom_mb: float,
        memory_class: str = "",
    ) -> MemoryBudget:
        """Build a host-memory view. Pure: RAM and headroom are both injected.

        ``headroom_mb`` is the ALREADY-RESOLVED headroom, not a request for
        one. Resolution is deliberately not done here: `resolve_host_headroom_mb`
        reads `os.environ` and `config.DEFAULTS`, which would make this
        constructor impure and break the "everything impure lives in
        `_build_memory_budget`" contract. Callers pass the resolved figure.
        """
        if ram_mb is None or ram_mb <= 0:
            return cls(
                vram_limit_mb=DEFAULT_VRAM_LIMIT_MB,
                physical_keepout_mb=DEFAULT_PHYSICAL_VRAM_KEEPOUT_MB,
                shared_vram_limit_mb=DEFAULT_SHARED_VRAM_LIMIT_MB,
                cuda_free_floor_mb=DEFAULT_CUDA_FREE_FLOOR_MB,
                vram_headroom_mb=DEFAULT_VRAM_HEADROOM_MB,
                free_clamp_enabled=False,
                host_budget_mb=None,
                host_headroom_mb=0.0,
                host_ram_mb=None,
                host_memory_class=memory_class,
                unified_memory=unified,
            )
        return cls(
            vram_limit_mb=DEFAULT_VRAM_LIMIT_MB,
            physical_keepout_mb=DEFAULT_PHYSICAL_VRAM_KEEPOUT_MB,
            shared_vram_limit_mb=DEFAULT_SHARED_VRAM_LIMIT_MB,
            cuda_free_floor_mb=DEFAULT_CUDA_FREE_FLOOR_MB,
            vram_headroom_mb=DEFAULT_VRAM_HEADROOM_MB,
            free_clamp_enabled=False,
            host_budget_mb=max(0.0, float(ram_mb) - float(headroom_mb)),
            host_headroom_mb=float(headroom_mb),
            host_ram_mb=float(ram_mb),
            host_memory_class=memory_class,
            unified_memory=unified,
        )

    def host_fits(self, est_mb: float) -> bool:
        """Whether a host estimate fits. Unknown RAM fails closed."""
        if self.host_budget_mb is None:
            return False
        return float(est_mb) <= self.host_budget_mb

    def host_reject_reason(self, est_mb: float) -> str | None:
        """Why the host gate refuses, or None when it fits.

        Byte-equivalent to the historical inline string, including the `ram=`
        and `class=` tokens, so any consumer matching on the message keeps
        working.
        """
        if self.host_fits(est_mb):
            return None
        if self.host_budget_mb is None:
            return (
                f"HOST_MEMORY_PREFLIGHT ram_unknown class={self.host_memory_class} "
                f"est={float(est_mb):.0f}MB"
            )
        return (
            f"HOST_MEMORY_PREFLIGHT est={float(est_mb):.0f}MB > "
            f"budget={self.host_budget_mb:.0f}MB "
            f"(ram={self.host_ram_mb:.0f} headroom={self.host_headroom_mb:.0f} "
            f"class={self.host_memory_class})"
        )

    @property
    def kill_ceil_mb(self) -> float:
        """Absolute dedicated-VRAM ceiling: ``min(limit, physical - keepout)``.

        When the physical total is unknown the configured limit stands
        unclamped — an unknown device is not a reason to refuse a Trial.
        """
        if self.total_vram_mb is None or self.total_vram_mb <= 0:
            return self.vram_limit_mb
        return min(
            self.vram_limit_mb,
            max(0.0, self.total_vram_mb - self.physical_keepout_mb),
        )

    def effective_vram_limit_mb(
        self,
        *,
        n_cpu_moe: int | None = None,
        free_vram_mb: float | None = None,
        headroom_mb: float | int | None = None,
    ) -> float:
        """The budget a Trial actually gets, given offload and free-at-start.

        ``min(configured, free - headroom)``, with two documented exceptions:

        * MoE with ``n_cpu_moe > 0`` uses the configured budget only. OS-reserved
          VRAM otherwise false-rejects expert-CPU offload, whose measured peaks
          sit far below free.
        * The free-at-start clamp is off by default (``free_clamp_enabled``);
          WDDM desktop reservations make ``free - headroom`` false-reject loads
          that physically fit. The runtime VRAM monitor is the kill guard in
          both modes.
        """
        configured = self.vram_limit_mb
        moe_offload = n_cpu_moe is not None and int(n_cpu_moe) > 0
        if moe_offload or not self.free_clamp_enabled:
            return float(configured)
        if free_vram_mb is None or free_vram_mb <= 0:
            return float(configured)
        headroom = float(headroom_mb) if headroom_mb is not None else self.vram_headroom_mb
        return min(configured, max(0.0, float(free_vram_mb) - headroom))

    def fits(
        self,
        est_mb: float,
        *,
        n_cpu_moe: int | None = None,
        free_vram_mb: float | None = None,
        headroom_mb: float | int | None = None,
    ) -> bool:
        """Whether a Trial estimating ``est_mb`` fits the effective budget."""
        effective = self.effective_vram_limit_mb(
            n_cpu_moe=n_cpu_moe,
            free_vram_mb=free_vram_mb,
            headroom_mb=headroom_mb,
        )
        return float(est_mb) <= effective

    def reject_reason(
        self,
        est_mb: float,
        *,
        n_cpu_moe: int | None = None,
        free_vram_mb: float | None = None,
        headroom_mb: float | int | None = None,
    ) -> str | None:
        """Why this config is refused, or ``None`` when it fits.

        The message names both the configured and the effective budget whenever
        the clamp actually bit, so an operator reading a rejection can tell a
        "the clamp shrank your budget" case from a "you simply need more room"
        case.
        """
        effective = self.effective_vram_limit_mb(
            n_cpu_moe=n_cpu_moe,
            free_vram_mb=free_vram_mb,
            headroom_mb=headroom_mb,
        )
        est = float(est_mb)
        if est <= effective:
            return None
        if effective == self.vram_limit_mb:
            return f"VRAM_PREFLIGHT est={est:.0f}MB > budget={effective:.0f}MB"
        headroom = float(headroom_mb) if headroom_mb is not None else self.vram_headroom_mb
        free_txt = f"free={float(free_vram_mb):.0f}MB " if free_vram_mb is not None else ""
        return (
            f"VRAM_PREFLIGHT est={est:.0f}MB > effective={effective:.0f}MB "
            f"(configured={self.vram_limit_mb:.0f}MB "
            f"{free_txt}headroom={headroom:.0f}MB)"
        )

    @classmethod
    def resolve(
        cls,
        *,
        vram_limit_mb: float | int | None = None,
        vram_headroom_mb: float | int | None = None,
        shared_vram_limit_mb: float | int | None = None,
        cuda_free_floor_mb: float | int | None = None,
        physical_keepout_mb: float | int | None = None,
        free_clamp_enabled: bool = False,
        total_vram_mb: float | None = None,
        defaults: dict | None = None,
    ) -> MemoryBudget:
        """Apply the policy. Pure: takes every input, reads nothing.

        Precedence for each axis is ``explicit arg > defaults mapping > module
        default``, matching the historical resolver precedence of
        ``arg > env > config > default`` with env and config already folded into
        ``defaults`` by the caller.

        The physical-keepout clamp is applied here (not in a later step), so
        ``vram_limit_mb`` on the result is already safe to compare against.
        """
        cfg = defaults or {}

        def _pick(explicit, key, fallback):
            if explicit is not None:
                return float(explicit)
            val = cfg.get(key)
            if val is not None:
                return float(val)
            return float(fallback)

        configured = _pick(vram_limit_mb, "VRAM_LIMIT_MB", DEFAULT_VRAM_LIMIT_MB)
        keepout = _pick(
            physical_keepout_mb, "PHYSICAL_VRAM_KEEPOUT_MB", DEFAULT_PHYSICAL_VRAM_KEEPOUT_MB
        )
        headroom = _pick(vram_headroom_mb, "VRAM_HEADROOM_MB", DEFAULT_VRAM_HEADROOM_MB)
        shared = _pick(shared_vram_limit_mb, "SHARED_VRAM_LIMIT_MB", DEFAULT_SHARED_VRAM_LIMIT_MB)
        cuda_floor = _pick(cuda_free_floor_mb, "CUDA_FREE_FLOOR_MB", DEFAULT_CUDA_FREE_FLOOR_MB)

        budget = cls(
            vram_limit_mb=configured,
            physical_keepout_mb=keepout,
            shared_vram_limit_mb=shared,
            cuda_free_floor_mb=cuda_floor,
            vram_headroom_mb=headroom,
            free_clamp_enabled=bool(free_clamp_enabled),
            total_vram_mb=None if total_vram_mb is None else float(total_vram_mb),
        )
        if budget.kill_ceil_mb < configured:
            budget = cls(
                vram_limit_mb=budget.kill_ceil_mb,
                physical_keepout_mb=keepout,
                shared_vram_limit_mb=shared,
                cuda_free_floor_mb=cuda_floor,
                vram_headroom_mb=headroom,
                free_clamp_enabled=bool(free_clamp_enabled),
                total_vram_mb=budget.total_vram_mb,
            )
        return budget
