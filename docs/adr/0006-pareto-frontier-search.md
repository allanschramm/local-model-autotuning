# ADR 0006: Multi-Objective Pareto Frontier Search

**Date:** 2026-07-25
**Status:** Accepted. **Day Usage Profile pick superseded by [0007](0007-day-profile-speed-band.md)** then **[0008](0008-day-iq-epsilon-then-tps.md)** (2026-07-27). Night + Pareto Set membership unchanged.
**Superseded in part (2026-08-26, operator decision):** cross-model front membership is retired — the rank lists every complete model quality-first and `dominated` becomes a same-model config label ([0017](0017-rank-membership-quality-first.md)).
**Superseded in part (issue #15, 2026-08-05):** `keep`/`discard` retired as Search truth — `discard` superseded by `dominated`/`incomplete`/`rejected`. Canonical decision = Trial Status via the Pareto nucleus.
**Superseded in part (2026-08-10):** `keep` TSV persistence alias deleted — writers persist `on_front` only; leftover `keep` cells are not treated as frontier.
**Superseded in part (2026-08-10):** global Point identity for Day/Night + store status merge → GGUF **basename** ([0012](0012-basename-pareto-point.md)); Fingerprint remains for Neighbor Search / Baseline pick hint.
**Superseded in part (2026-08-13):** Night pick may use `min(agentic, coding, agentic_coding)` when the SWE-lite column is present ([0013](0013-agentic-coding-night-selector.md)); Pareto Set axes unchanged.
**Superseded in part (2026-08-15):** required product journey and auto picker → Fingerprint bus + TPS-then-Pi ([0014](0014-fingerprint-bus-product-split.md)). Pareto Set membership and TSV axes unchanged; Day/Night maximin is no longer the required elect-what-you-run path.
**Supersedes in part:** [0004](0004-agentic-first-search.md) (canonical scalar Val Score + keep/discard as Search truth). Agentic + coding benchmarks remain the intelligence measurements; Baseline location remains [0005](0005-config-py-mutable-baseline.md).

## Context & Problem Statement

The repo teaches people to find the best local model for *their* rig. Real use is multi-objective: large configured context (long agent loops), throughput (daytime supervised chat), agentic tool-use (Claw), and coding skill (coding-10). A single Val Score and one Baseline champion collapse those tradeoffs. Historical “Pareto Tie-Breaker” only broke exact Val Score ties — it was not a frontier.

Day vs night use differs: supervised daytime wants speed; unsupervised night `/loop` wants balanced intelligence and enough context. One TPS Floor on keep cannot serve both.

## Decision

1. **Pareto Set** is the keep surface: non-dominated Trials under four maximize axes — configured `CTX_SIZE`, TPS, agentic (Claw-Eval full), coding (coding-10).
2. **Search / Neighbors stay per model.** The **global** frontier (union across models) is ranked for a hardware+budget identity so users pick a model for their rig.
   - **Practical bucket (Phase 1, issue #4):** `round(peak_vram_gb)` from the Trial row. The TSV has no hardware column, so peak VRAM rounded is the budget proxy — the known Set and the Fingerprint merge never cross buckets. Known limitation: rounding can split Trials of one machine near a boundary or join two machines with equal VRAM; a hardware column would be the full fix (Phase 2+).
3. **Baseline** is the Neighbor origin (active point), not the sole champion. Profile **Day** / **Night** selects that origin (manual override allowed): Day → max TPS *(superseded: [0007](0007-day-profile-speed-band.md) speed band, then [0008](0008-day-iq-epsilon-then-tps.md) IQ ε-band then max TPS)*; Night → among points with `CTX_SIZE ≥ NIGHT_CTX_FLOOR` (default 65536), max `min(agentic, coding)`; if none qualify, fallback to max ctx with a complete vector. **Both profile floors are historical lens notes since [0017](0017-rank-membership-quality-first.md)** — they no longer filter anything; TPS/ctx break ±0.05 near-ties in the Day/Night tables.
4. **Complete vector required for `on_front`.** Partial Trials (coding-only, claw-only, …) stay `incomplete` and **merge** into the same **Fingerprint** (full `ENGINE_DEFAULTS` + `SAMPLER_DEFAULTS`) when axes arrive later.
5. **Status vocabulary:** `on_front` | `dominated` | `incomplete` | `rejected`. `keep`/`discard` are deleted — not accepted on write.
6. **No TPS Floor on frontier membership.** TPS remains an axis. Day/Night apply throughput (and Night ctx floor) only when *selecting* a point. Legacy `TPS_FLOOR` may remain for smoke/tooling until removed.
7. **Ship cut:** Phase 0 (done 2026-07-25) = domain docs (`CONTEXT.md` + this ADR + AGENTS preferences). Phase 1 = frontier nucleus (domination, fingerprint merge, status, leaderboard/TSV). Phase 2 = Search/autoloop profile pick + honest peak preflight / dynamic headroom. No big-bang eval rewrite.

## Consequences

### Positive
- Teaching and leaderboards show tradeoffs instead of a fake single winner.
- Day/Night map to real workflows without splitting measurement into two frontiers.
- Partial cheap Trials still contribute via fingerprint merge.

### Negative
- Breaking change for `keep`/`discard` consumers and for “beat Baseline Val Score” Search logic.
- Four-axis completeness makes a point slower/more expensive to land on the front than a scalar keep.

### Neutral
- Claw full and coding-10 remain the intelligence proxies; they are axes, not a blended “intelligence” score.
- `Val Score` becomes legacy display/compat, not Search truth (see glossary).

## Considered Options (rejected)

- Single Baseline + scalar Val Score with richer tie-breakers — hides Day/Night and ctx vs smart tradeoffs.
- Blending agentic+coding into one “intelligence” axis — recreates Gemma-vs-SWE style lies.
- Separate day/night Pareto Sets — duplicates measurement; teaching worse.
- Hard TPS Floor on keep — fights night-loop models that are slow but capable.
- Night pick = max ctx first — can prefer huge-context weak models over balanced ones that still clear a ctx floor.
- Day pick = pure max TPS — see [0008](0008-day-iq-epsilon-then-tps.md) (current; [0007](0007-day-profile-speed-band.md) was intermediate).
