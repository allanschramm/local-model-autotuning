# Reasoning Levels & Budgets — Reading the GGUF, Wiring the Harness

How to discover which reasoning knobs a GGUF actually supports, which llama.cpp
levers map to them, and what the results store records. Verified against the
pinned b10549 build and the local GGUF store on 2026-08-29.

Companion docs: [thinking-models-claw-harness.md](https://github.com/allanschramm/local-model-autotuning/blob/ea43a71f48571811ec1ba25df4ab71ed272d34f1/docs/discovery/thinking-models-claw-harness.md)
(measurement policy) and
[speculative-and-multi-gpu-tuning.md](./speculative-and-multi-gpu-tuning.md)
(reasoning-effort controls across families).

## 1. Where the levels live

Reasoning levels are **not a dedicated GGUF metadata key** (unlike
`block_count` / `expert_count`). They live inside the embedded
`tokenizer.chat_template` Jinja source. Reading recipe (~10 s per multi-GB
file, tensors untouched):

```bash
./venv/Scripts/python.exe llama.cpp/gguf-py/gguf/scripts/gguf_dump.py \
  --no-tensors --json models/<pub>/<model>/<file>.gguf > "$TEMP/kv.json"
# extract d["metadata"]["tokenizer.chat_template"]["value"], then count/read:
#   reasoning_effort | enable_thinking | thinking_budget | preserve_reasoning
# the conditional lines give the ladder, the default, and the validation
```

**Tool caveat (verified 2026-08-29):** `llama-template-analysis.exe`
(b10549) reported *"No reasoning/thinking-related variables were queried"*
for a template that reads `reasoning_effort` eight times — the probe misses
reads via `set x = reasoning_effort|default(...)` indirection. The direct
template grep above is the reliable check; treat a clean exe report as
insufficient evidence either way.

## 2. Template surface by family (verified 2026-08-29)

| Template family | Vars read | Working levers |
|---|---|---|
| Qwen3.8-27B open-source (arch `qwen35`) | `reasoning_effort` + `enable_thinking` | `--reasoning-effort` **real here**: ladder `xhigh` (default) / `medium` / `low`; `high` aliases to `xhigh`; anything else raises in Jinja. Default render injects the xhigh instruction into **every** prompt |
| qwen35 family: Qwen3.5/3.6, Qwen3.8-4B-Distill, Ornith (arch `qwen35`/`qwen35moe`) | `enable_thinking` only | `--reasoning on/off` (maps to `enable_thinking`), `--reasoning-budget N`; `--reasoning-effort` is a **silent no-op** |
| Nemotron 3 (`nemotron_h`) | `enable_thinking` (default true) | same as qwen35 family |
| LFM2.5 (`lfm2`) | none | no thinking mode; no reasoning flags |

Re-verify per GGUF before seeding anything — a family label is not a
template guarantee (Qwen3.8-27B vs Qwen3.8-4B-Distill differ exactly this
way).

## 3. Engine levers (b10549) and harness wiring

| llama.cpp flag (arg.cpp) | Mechanism | Harness Baseline key |
|---|---|---|
| `--reasoning on/off/auto` | binds `enable_thinking` template kwarg | `REASONING` |
| `--reasoning-effort LEVEL` | binds `reasoning_effort` kwarg; C++ does not validate the ladder, the template does | `REASONING_EFFORT` (added 2026-08-29) |
| `--reasoning-budget N` | **server-side** think-token cap (forces the end-of-thinking tag); template-independent, works on every model | `REASONING_BUDGET` |
| `--reasoning-budget-message S` | nudge injected before the forced end tag on exhaustion | `REASONING_BUDGET_MESSAGE` |
| `--reasoning-preserve / --no-` | binds `preserve_reasoning`; only when `/props` reports `supports_preserve_reasoning` | `REASONING_PRESERVE` |

All five are **config.py-only** Baseline keys (CLI equivalents exist but are
suppressed in `run.py`); `None` = omit the flag (template default). None of
them are Search neighbors — they are seeded from the card / verified per
GGUF, not hill-climbed. `REASONING_EFFORT` validates against
`minimal/low/medium/high/xhigh/max` in `config.py.example`; seed it only
after the template check in §1 (elsewhere it is a no-op).

Wire surface (per-request, no restart — llama-server OpenAI API):
`chat_template_kwargs` (JSON merged into the template context),
`reasoning_effort: "none"` (disables thinking), `reasoning_budget_tokens` /
`thinking_budget_tokens`, `reasoning_budget_message`. The harness benchmark
clients send none of these today; server-side `REASONING_BUDGET` is the
loop lever in use.

## 4. What the results store records

- Since 2026-08-29, `results.db` has flat `reasoning_budget` (numeric) and
  `reasoning_effort` (text) columns, derived from each row's `config_json`
  and backfilled once for legacy rows. The legacy TSV keeps the knobs inside
  its `config_json` column only (header untouched).
- Before that date the knobs existed **only** inside `config_json`
  (lowercase keys; some very old rows uppercase). Query recipe:

```bash
./venv/Scripts/python.exe -c "
import sqlite3, json, collections
con = sqlite3.connect('results.db')
for model, cfg in con.execute('SELECT model, config_json FROM trials'):
    knobs = {k: v for k, v in json.loads(cfg or '{}').items()
             if 'REASON' in k.upper()}
    print(model, knobs)
"
```

Audit of the store at migration time (3803 rows): every Trial ran at
**template default effort with `reasoning_budget` unset** (capability-first
policy, `max_tokens >= 4096`) except one Ornith-1.5-9B A/B row at budget
4096; `REASONING="on"` only on three non-qwen35 families. Daily-driver
alias budgets (2048/4096 + budget-message) are unmeasured profiles — no
Trial claims. For the Qwen3.8-27B template that default means **xhigh on
every measured row**.

Measurement policy (unchanged): capability evals run at default effort with
`max_tokens >= 4096`; a bounded-thinking efficiency profile is a **separate
Fingerprint** (`REASONING_BUDGET` N [+ message], modest `max_tokens`),
documented in the model card. Never seed a budget to "fix" a low score —
see the remasure checklist in [thinking-models-claw-harness.md](https://github.com/allanschramm/local-model-autotuning/blob/ea43a71f48571811ec1ba25df4ab71ed272d34f1/docs/discovery/thinking-models-claw-harness.md).

## 5. Alias-creation checklist

1. Dump the template (§1); list the vars it reads.
2. `reasoning_effort` present → the ladder is real; note default + allowed
   values in the card, then `REASONING_EFFORT` is seedable.
3. Only `enable_thinking` → use `--reasoning on/off` (or accept default);
   budget goes through `REASONING_BUDGET`, never `--reasoning-effort`.
4. No reasoning vars → no reasoning flags at all.
5. Record what you seeded in the card; the store now flat-columns it, so
   future Trials are queryable without parsing `config_json`.
