# ADR 0005: config.py as Mutable Baseline

**Date:** 2026-07-20
**Status:** Accepted
**Supersedes:** ADR 0004 decision that Baseline lives in `.autoresearch_state.json` (agentic-first Val Score decision in 0004 remains)

## Context
Operators and agents expected a single editable Baseline file (`config.py`, with ENGINE vs SAMPLER segregation). The prior design stored Baseline overrides in ignored `.autoresearch_state.json` and treated `config.py` as immutable, which forced CLI flag soup for `run.py --validation` and diverged from the intended Search contract (`program.md` fixed; config mutable).

## Decision
- `autoresearch/core/config.py` is the only mutable Baseline (`ENGINE_DEFAULTS` = performance, `SAMPLER_DEFAULTS` = quality). Keeps persist via `write_baseline`.
- `.autoresearch_state.json` stores visited memory only (schema v2).
- `program.md` and harnesses remain fixed unless the user explicitly requests a change.
- Trials are driven from `config.py`, not CLI flag overrides.

## Consequences
- Manual and autonomous Search share one Baseline file.
- `serve-config.py` / `run.py` argparse defaults follow the live Baseline.
- Search keeps dirty `config.py` in the working tree (expected; commit only when the operator wants).
- Legacy state files with a `baseline` key still load visited entries; baseline payload is ignored.
