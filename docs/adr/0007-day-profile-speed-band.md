# ADR 0007: Day Usage Profile uses a speed band, then intelligence

**Date:** 2026-07-26
**Status:** Accepted. **Day Usage Profile pick superseded by [0008](0008-day-iq-epsilon-then-tps.md)** (2026-07-27). Night + membership unchanged. Day chain note: the Day pick is now IQ-first with a ±0.05 TPS near-tie break and the old floors are historical lens notes ([0017](0017-rank-membership-quality-first.md)).
**Supersedes:** [0006](0006-pareto-frontier-search.md) Decision §3 **Day** pick only (Night + Pareto Set membership unchanged).

## Context & Problem Statement

ADR 0006 defined **Day** as “max TPS on the model front.” That lens treats throughput as the primary Day objective. On a real global front it selects TPS-only survivors (e.g. LFM2.5-8B) that lose on agentic and coding to other fast points (LFM2.5-1.2B, Gemma-4-E4B) while barely winning tok/s.

Supervised daytime chat still wants speed — but not a dumb model that is worse on every intelligence axis than another already-fast front point.

## Decision

1. **Day pick (historical — superseded by [0008](0008-day-iq-epsilon-then-tps.md)):** From the Pareto Set, keep points with  
   `TPS ≥ DAY_TPS_RATIO × max(TPS on that set)`  
   (default `DAY_TPS_RATIO = 0.5`). Among that **speed band**, maximize `min(agentic, coding)`; ties → max TPS.  
   **Current Day:** IQ ε-constraint then max TPS — see ADR 0008.
2. **Empty band fallback:** If no point clears the ratio, maximize `min(agentic, coding)` on the full set; ties → max TPS (same intelligence-first spirit, no ratio gate).
3. **Night unchanged:** `CTX_SIZE ≥ NIGHT_CTX_FLOOR` then max `min(agentic, coding)`; else max ctx with a complete vector.
4. **Membership unchanged:** TPS stays an Objective Vector axis; no TPS Floor on `on_front`.

## Consequences

### Positive
- Day prefers the smartest model that is already in the fast half of the front (portable across rigs without a fixed tok/s number).
- Avoids recommending a pure-TPS Pareto point that is dominated on quality by other Day-plausible points.

### Negative
- `DAY_TPS_RATIO` is a tunable; 0.5 is a starting default and may need revisiting when front TPS spreads are weird (one ultra-fast outlier vs a tight cluster).
- Day and Night can still land on the same Fingerprint when the smartest complete vector is also fast enough for the band.

### Neutral
- Manual Baseline override remains allowed.
- Search/autoloop profile pick (Phase 2) must implement this rule when wired.

## Considered Options (rejected)

- **Day = max TPS** (ADR 0006) — picks TPS-only survivors over smarter fast peers.
- **Day = max `min(agentic, coding)` on full front** — collapses into Night without a ctx floor; ignores “I am sitting here, I want snappy.”
- **Fixed absolute `DAY_TPS_FLOOR` (tok/s)** — not portable across hardware budgets.
- **Quality-only front then max TPS** — collapses Day to the (ctx, agentic, coding) survivors (often Night-ish models), under-weights speed.
