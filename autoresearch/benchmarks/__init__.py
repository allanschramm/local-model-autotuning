"""Bench Project — benchmark catalog, harnesses, scoring, and agentic eval.

All benchmark code lives here. New benchmarks can be added without touching
autoresearch.core (the Running Project). Benchmarks consume the stable API
exported by autoresearch.core (LlamaClient, GenerationParams, etc.).
"""

from autoresearch.benchmarks.benchmark_harness import BenchmarkResult
from autoresearch.benchmarks.benchmark_coding import run_benchmark, run_coding_eval
from autoresearch.benchmarks.agentic_runner import run_agentic_eval, run_agent_loop, score_task
from autoresearch.benchmarks.agentic_benchmarks import (
    discover_claw_tasks,
    get_quick_tier_tasks,
    get_full_tier_tasks,
    format_agentic_benchmarks,
    format_claw_tiers,
    AgenticBenchmarkSpec,
    AGENTIC_BENCHMARKS,
)

__all__ = [
    # Coding benchmarks
    "BenchmarkResult",
    "run_benchmark",
    "run_coding_eval",
    # Agentic benchmarks
    "run_agentic_eval",
    "run_agent_loop",
    "score_task",
    "discover_claw_tasks",
    "get_quick_tier_tasks",
    "get_full_tier_tasks",
    "format_agentic_benchmarks",
    "format_claw_tiers",
    "AgenticBenchmarkSpec",
    "AGENTIC_BENCHMARKS",
]
