# ADR 0019: Release local auto-tuning independently

**Status:** Accepted
**Date:** 2026-10-30
**Supersedes in part:** [ADR 0018](0018-independent-pi-benchmark-and-context-tuning.md), for the delivery order and in-tree publication of the independent Benchmark.

## Context & Problem Statement

The original repository serves local model auto-tuning. Releasing its bounded Search should not depend on deciding whether the new Pi Benchmark will be public or on distributing its datasets.

## Decision

Release bounded TPS × context tuning, shared runtime safeguards, SQLite persistence and Fingerprints from this repository. Preserve existing legacy evaluation entrypoints and frozen task paths. Keep the new Pi Benchmark outside this release; its packaging, publication and repository remain deferred. The independent-module interfaces in ADR 0018 remain the intended boundary.

Main requires Python validation, tuning and legacy-consumer E2E evidence, Python CodeQL analysis and its security result. Independent Pi matrix jobs belong to the deferred Benchmark delivery.

## Consequences

- **Positive:** tuning installs and validates without Node, Pi or the new dataset snapshots.
- **Negative:** this release does not deliver an independent quality harness.
- **Neutral:** historical quality results and protocols remain unchanged. Existing public staging history is not removed by this release split.
