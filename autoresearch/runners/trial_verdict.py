"""TrialVerdict — the pure decision layer of a Trial.

Deepening of the decision logic that used to be inlined in
``evaluation.run_trial`` (a 570-LOC function). The tell that the depth was in
the wrong place: the only way to unit-test the MoE/VRAM rejection rule was to
import a private symbol (``_moe_vram_reject``), because the module had no public
seam for "what do we decide, and why".

This module owns *deciding*. It is pure: given evidence (VRAM estimate, host
estimate, offload intent, model kind, bench measurement, benchmark selection),
it returns a verdict. It does not touch the filesystem, spawn a server, or read
env — the caller collects evidence and passes it in.

Orchestration (spawning, streaming, persisting) stays in ``run_trial``, which
after this extraction reads as: collect evidence → ask the verdict → persist.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

__all__ = [
    "BenchmarkSelection",
    "ModelKind",
    "TrialEvidence",
    "TrialVerdict",
    "VerdictReason",
]


class ModelKind(str, Enum):
    """What the loader could prove about the model architecture.

    ``UNKNOWN`` is a first-class value on purpose: the historical rejection
    rule only refused MoE models it could *prove* were MoE, so an unprovable
    architecture must not silently become "dense fits".
    """

    DENSE = "dense"
    MOE = "moe"
    UNKNOWN = "unknown"


class VerdictReason(str, Enum):
    """Why a Trial was refused, when it was."""

    INVALID_CONFIG = "invalid_config"
    VRAM_EXCEEDED = "vram_exceeded"
    HOST_MEMORY_EXCEEDED = "host_memory_exceeded"
    CODING_TASK_COUNT = "coding_task_count"
    EMPTY_AGENTIC_TIER = "empty_agentic_tier"
    BENCH_BELOW_THRESHOLD = "bench_below_threshold"


class TrialVerdict(Enum):
    """The decision itself: proceed, or refuse with a reason."""

    PROCEED = "PROCEED"
    INVALID_CONFIG = "INVALID_CONFIG"
    MODEL_REJECTED = "MODEL_REJECTED"
    INFRA_ERROR = "INFRA_ERROR"


@dataclass(frozen=True)
class BenchmarkSelection:
    """Which benchmark suites this Trial will run.

    A pure value object so the selection rules (validation defaults, the
    mini-swe-agent standalone rule, the coding-10 arity requirement) can be
    asserted without constructing a runner.
    """

    agentic_quick: bool = False
    agentic_full: bool = False
    agentic_coding: bool = False
    mini_swe_agent: bool = False
    include_coding: bool = False
    coding_task_limits: tuple[int, int, int] = (10, 10, 10)
    """(evalplus, livecodebench, bigcodebench) — canonical is exactly 10 each."""

    @classmethod
    def resolved(
        cls,
        *,
        agentic_quick: bool = False,
        agentic_full: bool = False,
        agentic_coding: bool = False,
        mini_swe_agent: bool = False,
        include_coding: bool = False,
        coding_task_limits: tuple[int, int, int] = (10, 10, 10),
        is_validation: bool = False,
    ) -> BenchmarkSelection:
        """Apply the normalization rules that used to be inline in run_trial.

        Three documented rules, preserved verbatim:

        1. **Validation** is a load/TPS smoke. It defaults Claw quick on (so a
           smoke run still has a real gate) and never runs the full/coding
           tiers — coding-10 is canonical-Trial work, not smoke work.
        2. **mini-swe-agent is standalone.** Selecting it disables every other
           benchmark; mixing them would blend incomparable scores.
        3. **Coding preflight is arity-exact.** The coding-10 axes are only
           comparable when each dataset contributes exactly 10 tasks.
        """
        limits = tuple(int(v) for v in coding_task_limits)  # type: ignore[assignment]

        if is_validation:
            if not (agentic_quick or agentic_full):
                agentic_quick = True
            agentic_full = False
            agentic_coding = False
            mini_swe_agent = False
            include_coding = False

        if mini_swe_agent:
            include_coding = False
            agentic_quick = False
            agentic_full = False
            agentic_coding = False

        return cls(
            agentic_quick=agentic_quick,
            agentic_full=agentic_full,
            agentic_coding=agentic_coding,
            mini_swe_agent=mini_swe_agent,
            include_coding=include_coding,
            coding_task_limits=limits,  # type: ignore[arg-type]
        )

    def coding_preflight_ok(self) -> bool:
        """Whether the requested coding task counts are the canonical 10/10/10."""
        if not self.include_coding:
            return True
        return self.coding_task_limits == (10, 10, 10)


@dataclass(frozen=True)
class TrialEvidence:
    """Everything a verdict needs, gathered by the caller.

    Deliberately flat and primitive: every field is a plain value so a verdict
    is reproducible from a serialized Trial and needs no machine access.
    """

    # --- VRAM ---
    vram_est_mb: float
    vram_limit_mb: float
    vram_fits: bool
    host_est_mb: float
    host_budget_mb: float
    host_fits: bool
    vram_reason: str | None = None
    """Preflight explanation; may be set even when the Trial fits."""

    # --- host memory ---
    host_reason: str | None = None

    # --- architecture / offload ---
    model_kind: ModelKind = ModelKind.DENSE
    n_cpu_moe: int | None = None
    model_exists: bool = True
    """When False, the MoE-specific rejection must not fire."""

    # --- benchmark selection ---
    selection: BenchmarkSelection = BenchmarkSelection()
    missing_agentic_tier: str | None = None
    """Name of a requested tier that resolved to zero tasks."""

    # --- measured bench (optional; absent when skip_bench) ---
    bench_tg_tps: float | None = None
    bench_threshold_tps: float | None = None

    def __post_init__(self) -> None:
        if self.vram_fits and self.vram_reason is None and self.vram_est_mb > self.vram_limit_mb:
            raise ValueError(
                "inconsistent evidence: vram_fits=True but est "
                f"{self.vram_est_mb}MB exceeds limit {self.vram_limit_mb}MB"
            )


@dataclass(frozen=True)
class VerdictReasonDetail:
    """A refusal reason: the class of refusal plus the operator-facing text."""

    reason: VerdictReason
    message: str
    trial_verdict: TrialVerdict


class TrialDecider:
    """The pure rule set. One instance, no state, safe to share.

    Named ``TrialDecider`` rather than ``TrialVerdict`` because the latter is
    the enum result; the decider is the thing that produces it. The import in
    ``evaluation`` is aliased so the public name reads ``TrialVerdict.of(...)``.
    """

    def moe_vram_reject(self, ev: TrialEvidence) -> str | None:
        """MoE-specific VRAM refusal, or ``None``.

        Only fires for a model proven to be MoE. A dense model that overflows
        gets the generic preflight reason instead, because "set N_CPU_MOE=None"
        is meaningless advice for a dense load.
        """
        if ev.vram_fits or ev.vram_est_mb <= ev.vram_limit_mb:
            return None
        if not ev.model_exists:
            return None
        if ev.model_kind is not ModelKind.MOE:
            return None
        if ev.n_cpu_moe == 0:
            return (
                f"MoE full-GPU (N_CPU_MOE=0) est={ev.vram_est_mb:.0f}MB > "
                f"limit={ev.vram_limit_mb:.0f}MB; "
                "set N_CPU_MOE=None for auto block_count offload"
            )
        n = ev.n_cpu_moe
        return (
            f"MoE offload (N_CPU_MOE={n}) est={ev.vram_est_mb:.0f}MB > "
            f"limit={ev.vram_limit_mb:.0f}MB; exceeds physical VRAM limit"
        )

    def of(self, ev: TrialEvidence) -> VerdictReasonDetail | None:
        """The verdict for this evidence, or ``None`` to proceed.

        Order matters and mirrors the historical run_trial short-circuit order,
        so a Trial that violates several rules reports the same one it always
        did: VRAM, then host memory, then benchmark selection, then bench.
        """
        if not ev.vram_fits:
            reason = self.moe_vram_reject(ev) or ev.vram_reason
            if reason is None:
                reason = (
                    f"VRAM_PREFLIGHT est={ev.vram_est_mb:.0f}MB > limit={ev.vram_limit_mb:.0f}MB"
                )
            return VerdictReasonDetail(
                reason=VerdictReason.VRAM_EXCEEDED,
                message=reason,
                trial_verdict=TrialVerdict.MODEL_REJECTED,
            )

        if not ev.host_fits:
            reason = ev.host_reason or (
                f"HOST_PREFLIGHT est={ev.host_est_mb:.0f}MB > budget={ev.host_budget_mb:.0f}MB"
            )
            return VerdictReasonDetail(
                reason=VerdictReason.HOST_MEMORY_EXCEEDED,
                message=reason,
                trial_verdict=TrialVerdict.MODEL_REJECTED,
            )

        if not ev.selection.coding_preflight_ok():
            return VerdictReasonDetail(
                reason=VerdictReason.CODING_TASK_COUNT,
                message="Coding preflight requires exactly 10 tasks per dataset",
                trial_verdict=TrialVerdict.INVALID_CONFIG,
            )

        if ev.missing_agentic_tier:
            tier = ev.missing_agentic_tier
            return VerdictReasonDetail(
                reason=VerdictReason.EMPTY_AGENTIC_TIER,
                message=f"No agentic {tier} tasks found",
                trial_verdict=TrialVerdict.INFRA_ERROR,
            )

        if (
            ev.bench_tg_tps is not None
            and ev.bench_threshold_tps is not None
            and ev.bench_tg_tps < ev.bench_threshold_tps
        ):
            return VerdictReasonDetail(
                reason=VerdictReason.BENCH_BELOW_THRESHOLD,
                message=(
                    f"bench tg {ev.bench_tg_tps:.1f} t/s below threshold "
                    f"{ev.bench_threshold_tps:.1f}"
                ),
                trial_verdict=TrialVerdict.MODEL_REJECTED,
            )

        return None


_DEFAULT = TrialDecider()


def verdict_for(evidence: TrialEvidence) -> VerdictReasonDetail | None:
    """Module-level convenience over the shared decider."""
    return _DEFAULT.of(evidence)


def moe_vram_reject_message(evidence: TrialEvidence) -> str | None:
    """Public seam for the MoE/VRAM rule (previously ``_moe_vram_reject``)."""
    return _DEFAULT.moe_vram_reject(evidence)
