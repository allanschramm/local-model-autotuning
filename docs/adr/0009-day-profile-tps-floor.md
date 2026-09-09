# ADR 0009: Day Usage Profile uses a TPS Floor (>= 50.0 TPS), then maximizes IQ

**Date:** 2026-08-06  
**Status:** Superseded in part — Day pick demoted to a historical lens note by [0017](0017-rank-membership-quality-first.md) (2026-08-26): `DAY_TPS_FLOOR` no longer filters membership anywhere; TPS only breaks ±0.05 near-ties in the Day table.  
**Supersedes:** [0008](0008-day-iq-epsilon-then-tps.md) **Day** pick only (Night + Pareto Set membership unchanged).  
**Background:** [pareto-selection.md](../discovery/pareto-selection.md).

## Context & Problem Statement

ADR 0008 Day used a relative IQ ε-constraint (`min(agentic, coding) >= 0.75 * IQ_best`) then maximized TPS.  
However, when the highest-IQ model on the front had low TPS (e.g. `POCKET-35B` at 35 TPS), the 0.75 ratio allowed models as slow as **30 TPS** (`Kwaipilot-KAT`) or **41 TPS** (`Qwythos-9B-v2`) to win the Day pick.

A model running at ~30–40 TPS is too slow for snappy daytime interactive coding/terminal work. Day's core intent is **high throughput / speed**.

## Decision

1. **Day pick (replace ADR 0008 Day):** On the Pareto Set (complete vectors),
   - Filter points clearing a **Day TPS Floor** (`TPS >= DAY_TPS_FLOOR`, default **`50.0 TPS`**).
   - Among points meeting the TPS floor, maximize `min(agentic, coding)` (highest IQ among fast models); ties → higher TPS, then higher ctx.
   - **Fallback:** If no front point clears `DAY_TPS_FLOOR`, fall back to sorting by TPS descending (fastest available complete model).
2. **Night unchanged:** `CTX_SIZE >= NIGHT_CTX_FLOOR` (65536) then max `min(agentic, coding)`.
3. **Membership unchanged:** Four-axis Pareto Set (`ctx` × `TPS` × `agentic` × `coding`); no TPS floor on `on_front` membership.
4. **CLI override:** `--day-tps-floor` flag in `scripts/rank_results.py` allows customizing the floor (e.g. 50.0, 60.0, 80.0, or 0.0 for pure TPS).

## Consequences

### Positive
- Models slower than 50 TPS (like `POCKET-35B` or `Kwaipilot-KAT`) can never win the Day pick.
- Supervised daytime terminal/IDE users get guaranteed snappy generation.
- Among fast-enough models, the smartest one is selected.

### Negative
- High-IQ models running under 50 TPS are excluded from the Day pick table (they remain candidates for Night).

### Neutral
- Manual Baseline override remains allowed in `autoresearch/core/config.py`.
