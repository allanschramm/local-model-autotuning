# ADR 0008: Day Usage Profile uses an IQ ε-constraint, then max TPS

**Date:** 2026-07-27
**Status:** Accepted; superseded in part by [0009](0009-day-profile-tps-floor.md) **Day** (2026-08-06), then demoted to a historical lens note by [0017](0017-rank-membership-quality-first.md) (2026-08-26) — Day is now IQ-first with a TPS near-tie break; floors no longer filter anything.
**Supersedes:** [0007](0007-day-profile-speed-band.md) **Day** pick only (Night + Pareto Set membership unchanged).
**Background:** [pareto-selection.md](../discovery/pareto-selection.md) (ε-constraint / maximin citations).

## Context & Problem Statement

ADR 0006 Day = pure max TPS → TPS-only survivors (e.g. LFM 8B).  
ADR 0007 Day = speed band then max `min(agentic, coding)` → still picks LFM 1.2B (`min` **0.35**) when Night champ is POCKET (`min` **0.615**) — ~half the intelligence for the same task class.

Day wants snappy supervised use, but “slightly lower IQ” ≠ “collapse IQ.” The axis that must not be sacrificed should be the **constraint**; the axis being traded (TPS) should be the **objective**. ADR 0007 inverted that order.

## Decision

1. **Day pick (replace ADR 0007 Day):** On the Pareto Set (complete vectors),  
   - `IQ_best = max(min(agentic, coding))`  
   - Keep points with `min(agentic, coding) ≥ DAY_IQ_RATIO × IQ_best` (default **`DAY_IQ_RATIO = 0.75`**)  
   - Among that **IQ band**, maximize TPS; ties → higher `min(agentic, coding)`, then higher ctx.  
2. **Empty band fallback:** If none clear the ratio (degenerate front), maximize `min(agentic, coding)`; ties → max TPS.  
3. **Night unchanged:** `CTX_SIZE ≥ NIGHT_CTX_FLOOR` then max `min(agentic, coding)`; else max ctx with a complete vector.  
4. **Membership unchanged:** four-axis Pareto Set; no TPS Floor on `on_front`.  
5. **Tuning:** raise `DAY_IQ_RATIO` to **0.8** when Day work is as quality-sensitive as Night (tighter IQ band, usually slower Day pick).

## Consequences

### Positive
- Day cannot recommend a model with collapsed `min(agentic, coding)` relative to the front champion.
- Still prefers the fastest model *among* those that stay near the IQ champion (portable ratio, no absolute tok/s).
- Matches standard ε-constraint MCDM (constrain sacrosanct objective, optimize the trade).

### Negative
- Day may land on a mid-TPS model (e.g. Ornith-9B-MTP) instead of the absolute TPS leader — intentional.
- `DAY_IQ_RATIO` is still a tunable; 0.75 is the teaching default.

### Neutral
- Day and Night can still coincide when the Night pick also wins TPS inside the IQ band.
- Manual Baseline override remains allowed.

## Considered Options (rejected)

- **Day = max TPS** (ADR 0006) — IQ collapse.  
- **Day = speed band then IQ** (ADR 0007) — constrains the wrong axis; still allows half-IQ picks.  
- **Day = Night without ctx floor** — no speed preference left.  
- **Full TOPSIS / interactive EMO / knee search** — overkill for a small teachable front; see [pareto-selection.md](../discovery/pareto-selection.md).
