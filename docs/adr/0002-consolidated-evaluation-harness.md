# ADR 0002: Consolidated Evaluation Harness

## Status
Accepted

## Context
Evaluation logic (dual-pass loops, agentic multi-turn orchestration, scoring) was duplicated across `prepare.py` and `prepare_claw.py`. This created "dark matter" logic that was hard to maintain and test.

## Decision
Create a deep `BenchmarkHarness` module that owns the **Dual-Pass Policy** (sequencing Pass 1 for Accuracy and Pass 2 for Throughput). Use a polymorphic `EvalTask` protocol to decouple the execution policy from specific task data (retrieval vs tool-use).

## Consequences
- **Positive**: Eliminated logic duplication; adding new benchmarks (e.g. vision) is now a simple adapter implementation.
- **Positive**: Multi-turn agentic support is handled autonomously by the harness.
- **Neutral**: Requires tasks to satisfy a strict protocol (`get_initial_prompt`, `process_step`, `get_final_score`).
