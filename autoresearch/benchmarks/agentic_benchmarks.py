"""Catalog of approved agentic coding benchmarks.

This module is intentionally metadata-only. The current benchmark runner is a
single-completion code harness; repo-editing and terminal agents need a separate
adapter layer before they can be executed safely from this project.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AgenticBenchmarkSpec:
    """Stable description of an external agentic benchmark target."""

    key: str
    name: str
    status: str
    scope: str
    harness: str
    source_url: str
    notes: str
    priority: int


AGENTIC_BENCHMARKS: tuple[AgenticBenchmarkSpec, ...] = (
    AgenticBenchmarkSpec(
        key="claw-eval",
        name="Claw-Eval",
        status="adopt-next",
        scope="300 human-verified autonomous-agent tasks across general, multimodal, and multi-turn splits",
        harness="claw-eval",
        source_url="https://github.com/claw-eval/claw-eval",
        notes=(
            "External harness with Pass^3 scoring. Needs a local-model adapter, "
            "task subset policy, runtime budget, and result parser before execution."
        ),
        priority=10,
    ),
)


def list_agentic_benchmarks(status: str | None = None) -> list[AgenticBenchmarkSpec]:
    """Return benchmark specs sorted by migration priority."""

    specs = sorted(AGENTIC_BENCHMARKS, key=lambda spec: spec.priority)
    if status is None:
        return specs
    return [spec for spec in specs if spec.status == status]


def format_agentic_benchmarks(status: str | None = None) -> str:
    """Format benchmark specs for CLI display."""

    rows = [
        "key\tstatus\tharness\tname",
        *(
            f"{spec.key}\t{spec.status}\t{spec.harness}\t{spec.name}"
            for spec in list_agentic_benchmarks(status=status)
        ),
    ]
    return "\n".join(rows)