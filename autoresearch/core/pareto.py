"""Pareto nucleus — re-exported from Rust (Phase 0 Sprint 0.4).

The implementation now lives in `autoresearch_core.pareto` (a pure-Rust
crate exposed via PyO3 + maturin). This module is a thin compatibility
shim: every public symbol the project has been importing from
``autoresearch.core.pareto`` continues to be importable unchanged.

History (carried verbatim from the previous pure-Python module):

    Pure Pareto nucleus: Fingerprint, Objective Vector, Domination, merge,
    Pareto Set. Issue #1. No harness I/O, no results store, no Search loop.
    Vocabulary follows CONTEXT.md / ADR 0006: agentic = Claw-Eval full,
    coding = coding-10, four maximize axes = configured ctx, TPS, agentic,
    coding.
"""

from __future__ import annotations

from typing import Protocol

from autoresearch_core.pareto import (  # noqa: F401  (re-exported)
    ObjectiveVector,
    Trial,
    dominates,
    fingerprint,
    merge,
    pareto_set,
)

# Four maximize axes of the Objective Vector (ADR 0006).
AXES = ("ctx", "tps", "agentic", "coding")


class VectorLike(Protocol):
    """Anything exposing the four Objective Vector axes
    (e.g. rank_results.Point)."""

    ctx: int | float | None
    tps: float | None
    agentic: float | None
    coding: float | None
    complete: bool


__all__ = [
    "AXES",
    "ObjectiveVector",
    "Trial",
    "VectorLike",
    "dominates",
    "fingerprint",
    "merge",
    "pareto_set",
]
