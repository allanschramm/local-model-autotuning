# ADR 0004: Agentic-First Search and Local State

**Date:** 2026-07-10
**Status:** Accepted

## Context
Direct code-generation scores do not measure long-horizon tool use. Rewriting Python configuration also mixed durable defaults with local Search state.

## Decision
Claw-Eval full is the canonical Val Score and Claw-Eval quick is smoke validation. Direct-coding is optional and uses exactly 10 tasks per dataset. Runtime invariants are validated before process launch. Mutable Baseline and visited state live in ignored, atomically written `.autoresearch_state.json`; `config.py` remains immutable. Future agentic adapters must run locally without Docker, remote judges, or external APIs.

## Consequences
- Search decisions measure agentic behavior rather than isolated code generation.
- Local state no longer dirties tracked Python source.
- Invalid model configurations can be rejected while infrastructure and code failures stop the Search.
- Existing TSV columns remain readable while new rows carry reproducibility metadata.
