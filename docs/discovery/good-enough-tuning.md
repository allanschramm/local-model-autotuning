# Legacy Good-Enough Tuning (Compatibility)

**Goal:** find a runtime config that is *fast enough* on your rig, quickly — not the global optimum.

The current default is the [independent bounded TPS × context Search](../../autoresearch/TUNING.md), followed by the [Pi quality Benchmark](../../benchmark/README.md). It measures generation at the configured allocation and keeps several Pareto alternatives under a fixed PPL ceiling. This page retains the legacy operator-run path below; its short CLI TPS probe is not proof of target-context throughput.

**Operator path ([ADR 0014](../adr/0014-fingerprint-bus-product-split.md)):** pick GGUF → TPS climb (PPL guard) **writes the Fingerprint file** → `model-up` serves that file → **Pi** is the IQ that gates daily use. Numeric benches (Claw full / coding-10) are **optional** on the same file via apply-to-Baseline — never a required stop before Pi. The Day/Night maximin pick off the Pareto front is a numeric report lens, not what you ship to Pi.

Claw-Eval full / coding-10 complete the **Objective Vector** on one Fingerprint — a point only reaches `on_front` when that vector is complete ([ADR 0006](../adr/0006-pareto-frontier-search.md)). They are not a per-neighbor search signal.

## Cost–benefit rule

| Approach | When | Cost |
|---|---|---|
| **Legacy autoloop `--mode tps`** (operator only) | Maintaining an existing legacy workflow | Low — short CLI bench + PPL, no Claw full per Trial |
| Manual A/B via `config.py` | Closing 2–3 rivals after a plateau | Medium — one explicit Baseline at a time |
| Autoloop `--mode both` / quality overnight | After speed is acceptable | High — Claw full + coding-10 (complete the Objective Vector) |

**Do not** run Claw full or coding-10 on every neighbor. Quality axes complete the Objective Vector once on the final Fingerprint; they are not a per-neighbor search step.

## Knobs that usually matter for speed

Touch these first (edit `autoresearch/core/config.py` only — no CLI Baseline soup):

- `KV_CACHE` / `KV_CACHE_K` / `KV_CACHE_V`
- `BATCH_SIZE` / `UBATCH_SIZE`
- `THREADS` / `THREADS_BATCH`
- `SPEC_DRAFT_N_MAX` (and draft model) when MTP/draft exists
- MoE: `N_CPU_MOE` (`None` = auto full expert CPU offload; `0` = full GPU only if it fits **physical** VRAM)

Leave sampler at the card recommendation through the TPS speed path. Mutate sampler only in a later quality pass.

## Default recipe

### 1. Seed Baseline

```bash
cp autoresearch/core/config.py.example autoresearch/core/config.py
# Set MODEL=…gguf basename; adjust TPS_FLOOR / CTX_SIZE / N_CPU_MOE for the rig
# Seed SAMPLER_DEFAULTS from docs/models/<card>.md Recommended settings
# (agentic/general vs coding profile) BEFORE any Trial — not template TEMP=0.4
```

### 2. Smoke that it loads

```bash
.\venv\Scripts\python.exe benchmark_search.py --validation --desc "validate <model>"
```

Gates: load + TPS floor + Claw quick smoke. One model at a time. Read the latest legacy Trial row from canonical SQLite `results.db`.

### 3. Hill-climb for speed

> Operator-run step. The agent must **not** launch `autoloop.py` / the
> `/autoresearch` autonomous loop on a user `hill climb` / `bench` / `validate`
> request — use explicit `benchmark_search.py` / `python -m autoresearch.runners.run`
> commands. The user starts `autoloop.py` themselves.

```bash
.\venv\Scripts\python.exe autoloop.py --mode tps
```

Set `VRAM_LIMIT_MB` in `config.py`; the same Baseline value governs preflight and runtime monitoring.

- Mutates **one** engine knob per Trial.
- Optimizes TPS; perplexity acts as quality ceiling (do not keep big PPL regressions).
- **Fingerprint bus (ADR 0014):** every kept TPS+PPL Neighbor rewrites `fingerprints/<GGUF stem>.json` (engine only — sampler stays user/card choice, never a TPS Neighbor). A failed or rejected Trial never overwrites a good file. This file — not a Pareto Day/Night pick — is what the launcher serves to Pi.
- Stop on local maxima / Ctrl+C. Baseline stays in `config.py`; canonical history is in `results.db`.

Optional: small manual A/B of 2–3 configs via `config.py` + `--validation` if you already know the candidates.

### 4. Ship to Pi (default endpoint)

The climb wrote `fingerprints/<stem>.json`. Serve it and judge quality where it matters — daily use in Pi:

```bash
.\venv\Scripts\python.exe scripts/model_up.py <alias>
```

- The alias gives identity + local bind (host/port); **engine flags come from the Fingerprint file** — the exact engine that won the TPS climb. Missing/invalid file fails closed; re-run the climb instead of hand-editing flags.
- Sampler is user/card choice; it is not part of the climb and not in the file.
- No numeric bench is required first: Pi (plus the optional [`visual-pack.md`](./visual-pack.md) camera run) is the quality gate ([ADR 0014](../adr/0014-fingerprint-bus-product-split.md)).
- Engine change after TPS = new Fingerprint file, new Pi run, new optional numeric row.

### 5. Optional: numeric benches (same file, apply-to-Baseline)

Only if you want comparable numbers for the engine Pi actually sees:

```bash
.\venv\Scripts\python.exe scripts/apply_fingerprint.py --model <model>.gguf
# Edit config.py: enable INCLUDE_AGENTIC_FULL and INCLUDE_CODING=10 tasks
.\venv\Scripts\python.exe benchmark_search.py --agentic-full --desc "<model> objective-vector"
```

- `apply_fingerprint.py` copies the Fingerprint engine (+ optional sampler) into the mutable Baseline, so benches measure the shipped engine — the bus and the benches share one truth.
- Claw full = agentic axis; coding-10 = coding axis; both must land on the same Fingerprint.
- A point becomes `on_front` **only** when the Objective Vector is complete and non-dominated ([ADR 0006](../adr/0006-pareto-frontier-search.md)). Partial vectors stay `incomplete` and merge by Fingerprint — later Trials complete the point, they do not spawn new ones.
- Read status, not a scalar: `scripts/recompute_status.py` → `on_front` | `dominated` | `incomplete` | `rejected`. Val Score / keep is legacy display, never Search truth.
- Day/Night picks off the front (`autoloop.py --profile day|night`) are a **numeric report lens** over that archive ([pareto-selection.md](./pareto-selection.md)) — fine as a Baseline start for another Search; **not** the pick you ship to Pi.

## Anti-patterns

1. **Overnight `autoloop` default/`both` before a TPS pass** — burns hours on slow configs.
2. **CLI sweep flags** — every Trial must edit `config.py` first.
3. **Coding-10 / Claw full inside the TPS search loop** — complete the Objective Vector once, on the final Fingerprint; do not re-run quality axes per neighbor.
4. **Dense models spilling to shared GPU memory** — cut `CTX_SIZE` / KV / draft, or reject; never “spill and hope”.
5. **Parallel validations** — one Trial at a time on the shared GPU/port.
6. **Treating the Day/Night maximin pick as what you ship to Pi** — Pi sees the Fingerprint file the TPS climb wrote ([ADR 0014](../adr/0014-fingerprint-bus-product-split.md)); the front and its picks are a numeric report.

## Related

- [`agent-onboarding.md`](./agent-onboarding.md) — bootstrap for agents
- [`discover-models.md`](./discover-models.md) — pick which GGUF first (Pareto), then this guide
- [`visual-pack.md`](./visual-pack.md) — Pi camera run on the same Fingerprint file
- [`pareto-selection.md`](./pareto-selection.md) — Objective Vector, Fingerprint merge, Day/Night report lens
- [`../adr/0006-pareto-frontier-search.md`](../adr/0006-pareto-frontier-search.md) — Pareto Set membership (four axes, non-domination)
- [`../adr/0014-fingerprint-bus-product-split.md`](../adr/0014-fingerprint-bus-product-split.md) — Fingerprint bus; TPS-then-Pi is the ship path
- `GOLDEN-RULES.md` — validation gates and harness rules
- `program.md` — Search terminology and Trial logging
