# ADR 0021: Independent repositories and documentation ownership

**Status:** Accepted
**Date:** 2026-10-31
**Supersedes in part:** ADR 0018's temporary in-tree packaging and ADR 0019's deferred Benchmark repository. ADR 0020's tuning-only distribution remains unchanged.

## Context & Problem Statement

The Pi Benchmark now has an independent private repository. Keeping quality documentation only in the former combined repository would leave new Benchmark sessions dependent on its worktree and could mix historical multi-harness results with the current protocol. Rewriting mixed originals would lose the provenance of the retained runtime extracts.

## Decision

- This public source-available repository owns local runtime tuning, provisioning/serving, context/TPS/PPL constraints, resource safeguards, Fingerprints and live runtime/performance documentation.
- [ailocal-benchmark](https://github.com/allanschramm/ailocal-benchmark) is the independent private owner of the pinned Pi harness, quality packs, vibe prompts, verification, bounded suites, raw run evidence and staged per-model reports. A connected endpoint is sufficient; a tuning checkout or prior tuning is not required.
- [benchmarks](https://github.com/allanschramm/benchmarks) remains the public results repository. Teaching stays in the independent ailocal-teach project; the original Rust experiment remains separate from this packaging decision.
- Complete changed/retired documentation, old model cards and mixed records are preserved with source hashes in the Benchmark's [legacy archive](https://github.com/allanschramm/ailocal-benchmark/blob/main/docs/legacy/INDEX.md). Its manifest inventories both source snapshots, records runtime-only documents retained here and excludes teaching ownership. Old agent/loop instructions are inert text, not restored commands.
- Historical documents, score protocols and accepted decisions stay unchanged. Existing runtime extracts keep their immutable public provenance links; the private archive is the maintained quality-history destination. Original relative references are mapped explicitly, including targets absent in the old source.
- Existing operator SQLite files and historical tables remain preserved in place. This documentary migration does not rescore them or merge legacy quality results into Pi outcomes. Model assets, aliases and machine Baseline are not moved.

## Consequences

- **Positive:** both projects have independent installation, operation, ownership and documentation entrypoints. Benchmark development no longer needs the former staging worktree to retrieve quality context.
- **Negative:** the Benchmark archive requires private-repository access; public tuning users use the retained immutable source links for historical provenance.
- **Neutral:** old public Git history remains historical evidence. The separation does not make previously public source or prompts retroactively private, change licenses or turn legacy scores into new measurements.
