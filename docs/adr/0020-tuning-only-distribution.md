# ADR 0020: Ship auto-tuning without legacy quality evaluation

**Status:** Accepted
**Date:** 2026-10-30
**Supersedes in part:** ADR 0019 for temporary legacy evaluator/leaderboard retention.

## Context & Problem Statement

The repository must serve local runtime tuning only. Quality evaluation belongs to the deferred independent Pi Benchmark, and teaching is outside this distribution.

## Decision

Retire legacy quality evaluators, task fixtures, Trial dispatchers, autonomous four-axis Search, quality rankings and their installation profile. Retain the bounded tuning protocol, engine/runtime safeguards, model provisioning and explicit Fingerprint application.

The dashboard reads only tuning-runtime-v1 attempts and per-run frontier/PPL reference. It uses read-only SQLite and never migrates, drops or rescores historical tables.

Preserve original quality code/documents through the pre-cleanup snapshot and benchmark branch. Keep model architecture/configuration and runtime evidence here; derived performance excerpts link to the immutable original and are not new measurements. The archive index retains historical decision IDs; do not reuse them.

## Consequences

- **Positive:** runtime installation and operation contain no quality evaluator, Pi or teaching dependency.
- **Negative:** legacy benchmark/Trial commands stop being available in this distribution. Use the preserved benchmark work for future quality evaluation.
- **Neutral:** private models, Baseline, aliases and existing results.db files are untouched. Historical four-axis data are not comparable with the new two-axis frontier.
