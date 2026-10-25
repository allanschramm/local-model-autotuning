# ADR 0004: Agentic-First Search and Local State

**Date:** 2026-07-10
**Status:** Superseded in part by [0005](0005-config-py-mutable-baseline.md) (Baseline location), [0006](0006-pareto-frontier-search.md) (scalar Val Score / keep-discard as Search truth → Pareto Set), and [0018](0018-independent-pi-benchmark-and-context-tuning.md) (new Benchmark permits cloud model endpoints). Claw-Eval full + coding-10 remain the legacy intelligence axes; quick remains smoke.

## Context
Direct code-generation scores do not measure long-horizon tool use. Rewriting Python configuration also mixed durable defaults with local Search state.

## Decision
Claw-Eval full is the canonical Val Score and Claw-Eval quick is smoke validation. Direct-coding is optional and uses exactly 10 tasks per dataset. Runtime invariants are validated before process launch. Mutable Baseline and visited state live in ignored, atomically written `.autoresearch_state.json`; `config.py` remains immutable. Future agentic adapters must run locally without Docker, remote judges, or external APIs.

## Consequences
- Search decisions measure agentic behavior rather than isolated code generation.
- Local state no longer dirties tracked Python source.
- Invalid model configurations can be rejected while infrastructure and code failures stop the Search.
- Existing TSV columns remain readable while new rows carry reproducibility metadata.
- Claw-Eval quick is observational smoke: it records score but does not reject on low quality. Only the TPS Floor (and invalid/infra/code failures) rejects a Trial.
