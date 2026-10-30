"""Legacy benchmark compatibility API with lazy evaluation imports."""

from importlib import import_module

from autoresearch.benchmarks.agentic_benchmarks import (
    AGENTIC_BENCHMARKS,
    AgenticBenchmarkSpec,
    discover_claw_tasks,
    format_agentic_benchmarks,
    format_claw_tiers,
    get_full_tier_tasks,
    get_quick_tier_tasks,
)
from autoresearch.benchmarks.benchmark_harness import BenchmarkResult

_RUNNERS = {
    "run_benchmark": "benchmark_coding",
    "run_coding_eval": "benchmark_coding",
    "run_agentic_eval": "agentic_runner",
    "run_agent_loop": "agentic_runner",
    "run_agentic_coding_eval": "agentic_coding.runner",
    "score_task": "agentic_runner",
}


def __getattr__(name):
    if name in _RUNNERS:
        value = getattr(import_module(f"{__name__}.{_RUNNERS[name]}"), name)
    elif name in {
        "bench_config",
        "agentic_runner",
        "benchmark_coding",
        "agentic_coding",
        "agentic_benchmarks",
        "benchmark_harness",
    }:
        value = import_module(f"{__name__}.{name}")
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))


__all__ = [
    # Coding benchmarks
    "BenchmarkResult",
    *_RUNNERS,
    "discover_claw_tasks",
    "get_quick_tier_tasks",
    "get_full_tier_tasks",
    "format_agentic_benchmarks",
    "format_claw_tiers",
    "AgenticBenchmarkSpec",
    "AGENTIC_BENCHMARKS",
]
