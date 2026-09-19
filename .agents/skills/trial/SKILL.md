---
name: trial
description: >
  Run a full Objective-Vector Trial (Claw-Eval full = 15 tasks + coding-10) for one
  GGUF or a sequential queue of variants. Use whenever the user says trial / Trial
  a model, "trial this X model", "trial all X variants/quants", complete the
  Objective Vector, or wants claw-full + coding-10 measured into results.db —
  even if they do not say the word "skill". Prefer this over ad-hoc llama-server
  or --validation-only when they want real frontier axes.
---

# trial

Operator skill for **full Trials**: one Fingerprint per GGUF basename, Claw full
(15 tasks) + coding-10 (10 tasks/dataset), logged to `results.db` (SQLite). Sequences run
**one at a time**; rejected or crashed items stay in the table and the queue
continues.

## When to use

- "trial this model" / "trial `<gguf>`"
- "trial all X variants" / "trial every quant of X"
- User wants claw-full + coding scores, not smoke `--validation`

Do **not** use this skill for Search neighbor loops (`autoloop`), TPS-only
hill-climbing, or smoke validation alone — those stay on
`docs/discovery/good-enough-tuning.md`. **Never launch `autoloop.py` / the
`/autoresearch` autonomous loop on a user trial/bench/validate request** —
`autoloop.py` is operator-only; the user starts the background hill-climb
themselves.

## Hard gates

- Stop live `model-up` (and any other harness server) before the first Trial —
  single-load.
- Edit only gitignored `autoresearch/core/config.py` Baseline — no CLI flag soup.
- Seed full `ENGINE_DEFAULTS` + `SAMPLER_DEFAULTS` from `docs/models/<card>.md`
  Recommended settings for the job (agentic/general vs coding) before the first
  Trial on that model. No leftover Baseline from a previous model. If the card
  has no Recommended settings, use `UNIVERSAL_FALLBACK_SAMPLER` from
  `config.py.example`.
- One Trial at a time. Venv Python only. Harness CLIs only
  (`benchmark_search.py` / `python -m autoresearch.runners.run`). Never raw
  `llama-server` / `llama-bench` for eval.
- Dense = no shared-mem / expert offload. MoE may use `N_CPU_MOE`. Use configured
  `CTX_SIZE` (floor 2048). `CTX_SIZE` is inviolable: never reduce or suggest
  reducing context size to fit memory, avoid OOM/Circuit Breaker, or resolve failed tasks.
- Do not edit harness / vendor code when a Trial fails. Record the failure, move
  to the next queue item.
- Never push results or tweak branches. Offline only.
- Never commit `models/aliases/` or machine Baseline.

## Inputs

Resolve from the user message; ask in plain chat if missing:

| Input | Meaning |
|---|---|
| Target | One GGUF basename **or** a family/pattern ("all Ornith 35B quants") |
| Profile | agentic/general vs coding — picks which Recommended settings to seed |
| Queue order | Optional; default = discover under `models/`, sort by basename |

Each distinct GGUF basename is its own Trial (quants are not interchangeable).

## Procedure

### 0. Preflight

1. Confirm GGUF(s) resolve under `models/` (flat or `publisher/model/*.gguf`).
2. Stop `model-up` / live servers on the harness port.
3. Ensure `autoresearch/core/config.py` exists (seed from `config.py.example` if
   missing).
4. Build the queue: one entry per basename. Announce the queue before starting.

### 1. For each queue item (serial)

1. **Seed Baseline** in `config.py`:
   - `MODEL` = GGUF basename (portable; no absolute user paths)
   - Engine knobs for the rig (`CTX_SIZE`, `VRAM_LIMIT_MB`, `TPS_FLOOR`,
     `N_CPU_MOE` for MoE, KV/batch/threads, MTP/spec if the card says so).
     `REASONING_PRESERVE` only if the card + `/props` `supports_preserve_reasoning`
     say so for agentic (`None` otherwise; not a Search neighbor).
   - Full sampler from the model card for the chosen profile
2. **Run the full Trial** (no timeout):

```text
.\venv\Scripts\python.exe benchmark_search.py --agentic-full --desc "trial <basename>"
```

   Defaults already mean Claw full = 15 tasks and coding-10
   (`AGENTIC_FULL_TASK_LIMIT=15`, `CODING_TASK_LIMIT` / LCB / BigCode = 10 in
   `autoresearch/benchmarks/bench_config.py`). Do not shrink task counts.
   Optional Night selector (ADR 0013; **off** by default): add `--agentic-coding`.
   Required before claiming a Night pick that uses `agentic_coding`.
3. **Read the latest row from `results.db`** (SQLite) for that basename / Fingerprint.
   Status comes from the store (`on_front` | `dominated` | `incomplete` |
   `rejected`). Low scores are not rejections.
4. **On reject or crash**: append whatever the harness wrote (or a table row
   noting the crash/traceback summary). Do **not** stop the queue. Do **not**
   edit code to "fix" the failure. Continue to the next basename.
5. **Update alias** after each item via the `local-model-alias` skill (machine-
   local `models/aliases/`). Preferred one alias per family; replace config when
   the preferred quant changes. Include sampler flags from `SAMPLER_DEFAULTS`.
   `status: ready` when the Trial produced usable engine metrics; keep
   `untested` / notes when rejected or crashed before a stable measure.
6. Advance to the next queue item.

### 2. Closeout

Print one markdown results table covering the whole queue, then stop.

## Results table (required output)

ALWAYS end with this table (one row per queue item):

```markdown
| # | basename | status | ctx | TPS | agentic | coding | alias | notes |
|---|----------|--------|-----|-----|---------|--------|-------|-------|
| 1 | `model-Q4_K_M.gguf` | on_front | 65536 | 42.1 | 0.6000 | 0.580 | `family-preferred` | |
| 2 | `model-Q3_K_M.gguf` | rejected | 65536 | — | — | — | `family-preferred` | HOST_MEMORY_PREFLIGHT |
```

- Pull numbers from the canonical results store (`results.db` via SQLite or `scripts/rank_results.py`), not memory.
- `status` = Trial Status label from the store.
- `notes` = reject reason, crash summary, or empty.
- Optional follow-up: `.\venv\Scripts\python.exe scripts/rank_results.py` if the
  user asks for Day/Night / front context — not required every time.

## Anti-patterns

- Parallel Trials or GPU burns
- `--validation` when the user asked for a full Trial
- Leaving the previous model's sampler/engine in Baseline
- CLI Baseline overrides instead of editing `config.py`
- Stopping the whole sequence because one variant rejected
- Committing aliases, `results.db`, or Baseline

## References

- Terminology: [CONTEXT.md](../../../CONTEXT.md)
- Manual Trial contract: [docs/discovery/pareto-selection.md](../../../docs/discovery/pareto-selection.md) (issue #9)
- Speed path (not this skill): [docs/discovery/good-enough-tuning.md](../../../docs/discovery/good-enough-tuning.md)
- Gates: [GOLDEN-RULES.md](../../../GOLDEN-RULES.md) §5
- Alias after Trial: `.agents/skills/local-model-alias/SKILL.md` (local, gitignored)
