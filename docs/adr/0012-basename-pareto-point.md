# ADR 0012: Pareto Point = GGUF Basename

**Date:** 2026-08-10
**Status:** Accepted
**Supersedes in part:** [0006](0006-pareto-frontier-search.md) § Decision (2)/(4) Fingerprint-as-Point identity for global Day/Night ranking and store status merge.
**Does not supersede:** ENGINE+SAMPLER Fingerprint for Neighbor Search / visited memory; four maximize axes (ctx × TPS × agentic × coding); Day/Night selection lenses ([0009](0009-day-profile-tps-floor.md) Day / Night ctx floor — **both demoted to historical lens notes by [0017](0017-rank-membership-quality-first.md)**: floors no longer filter, they only break near-ties).

## Context & Problem Statement

ADR 0006 treated a full ENGINE+SAMPLER Fingerprint as the Pareto point. Hill-climbing TPS (flag tweaks) or remasuring Claw (sampler / harness fixes) often wrote a *new* `config_json` hash — schema drift or intentional Baseline edits — so better TPS/Claw sat as incomplete orphans while Day/Night still showed the old complete pair for that GGUF. Operators care about **one line per quantized file**: improve Ornith UD Q4 on any knob, upgrade that model’s frontier vector.

## Decision

1. **Global Pareto Point identity = GGUF basename** (`results.tsv` `model` column). Different quants / publishers / versions stay distinct files (`…-Q4_K_M` ≠ `…-UD-Q4_K_XL`).
2. **Axis merge per basename:** max claw, max coding, max TPS, max ctx across OK Trials for that file (any Baseline). Complete when both claw and coding exist for the basename.
3. **`rank_results.py`**, **store `recompute`**, and **classify known Set** use basename merge for Day/Night and persisted `on_front` / `dominated` / `incomplete`.
4. **Fingerprint remains** a optional pick *hint* (config of the best-claw row) so autoloop can load a Baseline; Search neighbors may still track Fingerprints for visited configs.
5. Hardware+budget buckets (configured `VRAM_LIMIT_MB`) still isolate domination.

## Consequences

### Positive
- Remasure Claw / hill-climb TPS / sampler wins update the model’s front vector without requiring identical `config_json`.
- Day/Night match operator intuition: MODEL × TPS × Claw × coding.

### Negative
- A complete vector can mix claw from Baseline A with coding from Baseline B — reproducible as max-over-history, not a single-run Fingerprint.
- Replays that must pin one Baseline still use `config_json` on individual Trial rows.

### Neutral
- Quants and alternate GGUF basenames remain separate points.
- ADR 0006 axes and Day/Night lenses unchanged.
