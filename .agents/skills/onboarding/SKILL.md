---
name: onboarding
description: >
  Guide a human operator through why this repo exists and how to get a working
  local setup — hardware check, venv, llama.cpp release, Baseline seed, optional
  model pick, dashboard. Use whenever the user is new to the clone, asks what
  this project is for, how to start, how to install/setup, "onboarding",
  "primeiro uso", "como começar", "o que é esse repo", "me explica o fluxo",
  or seems stuck before their first successful validate/Trial — even if they
  never say "onboarding". Prefer this over jumping straight into Trials or
  autoloop.
---

# onboarding

Walk-and-run orientation for a **human operator** on a fresh (or half-set-up)
clone of `local-model-autotuning`. You explain **why**, then **do the setup
steps with them** — run safe checks, seed files, start the dashboard — and stop
before expensive Search/Trials unless they explicitly ask to continue.

## Language

Match the user's language. Default to **pt-BR** when unclear (almost all
operators here). Keep domain terms from `CONTEXT.md` in English when they are
canonical (`Trial`, `Baseline`, `Pareto Set`, `Fingerprint`, `CTX_SIZE`) and
gloss them once in the user's language.

## Why this repo exists (say this early)

One short framing, then move to action:

- **Problem:** local GGUF + `llama.cpp` has many runtime flags (KV cache, batch,
  threads, MoE offload, MTP…). Hand-tuning is slow and easy to get wrong on
  limited VRAM/RAM.
- **What this repo does:** autonomous **Search** over those flags for *your*
  hardware. It records Trials in the SQLite results store (`results.db`) and keeps a **Pareto Set** on
  four maximize axes: configured context × TPS × agentic (Claw-Eval full) ×
  coding (coding-10).
- **What it does *not* do:** re-quantize models, train weights, or pick a single
  “best model forever”. Day/Night are **usage profiles** over the Pareto Set,
  not separate frontiers.
- **Default path after setup:** smoke validate → hill-climb **speed**
  (`autoloop --mode tps`) → quality check on the champion. Details:
  `docs/discovery/good-enough-tuning.md`.

Do not dump the whole glossary. Point at `CONTEXT.md` / `README.md` if they want
depth.

## When to use / when to hand off

**Use this skill** for: first clone, “what is this?”, install/setup stuckness,
hardware/runtime/Baseline not ready, wanting the dashboard while learning.

**Hand off** (do not pretend onboarding covers them):

| User wants… | Go to |
|---|---|
| Full Objective Vector Trial(s) | `.agents/skills/trial` |
| Speed hill-climb / good-enough flags | `docs/discovery/good-enough-tuning.md` + `autoloop.py` |
| Which GGUF fits the rig | `docs/discovery/discover-models.md` (llmfit preferred) |
| Alias for daily `model-up` | `.agents/skills/local-model-alias` |

## Hard gates (onboarding must not violate these)

These exist because wrong order burns hardware and time:

1. **Hardware before download.** Run `scripts/check_hardware.py` and confirm the
   numbers with the user **before** `hf download` or a large GGUF load.
   Explain `discrete_gpu` vs `unified_memory` in plain language for *their*
   pool. On unified RAM, leave OS/IDE headroom — reject oversized “#1” picks.
2. **Release binaries first.** Do **not** build `llama.cpp` from source on a
   new clone. Extract a GitHub release under
   `llama.cpp-releases/<engine>/<tag>/build-cuda/bin/` (or CPU/Vulkan layout)
   and set `AUTORESEARCH_LLAMA_CPP_ROOT`. Build from source only when no release
   covers an urgent need (`docs/discovery/agent-onboarding.md`).
3. **Baseline only in gitignored `autoresearch/core/config.py`.** Seed from
   `config.py.example`. No Trial CLI flag soup. Never commit Baseline or
   `models/aliases/`.
4. **Venv Python only.** Prefer `.\venv\Scripts\python.exe` / `./venv/bin/python`.
5. **One heavy job at a time.** No parallel GPU burns. Stop live `model-up` /
   harness servers before any validate/Trial.
6. **Onboarding stops before Search/full Trials.** You may run
   `check_hardware`, setup checks, seed Baseline, `serve-config.py print-cmd`,
   optional `verify_setup.py`, and start the read-only UI. Do **not** start
   `autoloop.py`, full Claw, or coding-10 unless the user explicitly asks after
   setup looks ready. `--validation` smoke is OK only if they ask to prove the
   stack loads **and** a fitting GGUF + Baseline already exist.
7. **Crash during a later loop → stop and report.** Do not silently edit harness
   code to “keep going”.
8. **Privacy.** No committing private paths, hostnames, GPU SKUs, exact MiB
   dumps, or machine Baseline into tracked docs.

## Walk-and-run procedure

Announce the plan in a short numbered list, then execute step by step. After
each step: show what you observed, what it means, ask only when a real choice
exists (plain chat — no form UI).

### 0. Situate

Skim (do not recite) in this order as needed:

1. `README.md` — product pitch + install (pt-BR)
2. `docs/discovery/agent-onboarding.md` — map + essential commands
3. `docs/discovery/good-enough-tuning.md` — what comes *after* onboarding
4. Root `AGENTS.md` — DOX / edit contracts (if they will change code later)

Tell the user where they are: clone ready? venv? runtime? model? Baseline?

### 1. Hardware (required)

```text
.\venv\Scripts\python.exe scripts\check_hardware.py
# ./venv/bin/python scripts/check_hardware.py
```

If venv is missing, create it and install `requirements.txt` first (commands in
`README.md`), then re-run.

Explain memory class + usable pool. Confirm with the user. If detection is
incomplete, guide them to Task Manager / `nvidia-smi` / About This Mac — still
confirm before downloads.

### 2. Runtime release (required before serve)

1. Help them pick the right GitHub release asset for their OS/GPU (README table).
2. Extract into `llama.cpp-releases/<engine>/<tag>/build-cuda/bin/` (or the
   CPU layout the harness expects).
3. Set `AUTORESEARCH_LLAMA_CPP_ROOT` for the session (and remind them to persist
   it in their shell profile if they want).
4. Verify:

```text
.\venv\Scripts\python.exe scripts\serve-config.py print-cmd
```

Fix path/layout issues here — do not fall back to compiling upstream.

### 3. Baseline seed

```text
cp autoresearch/core/config.py.example autoresearch/core/config.py
```

Edit only what onboarding needs: `MODEL` (GGUF basename under `models/` when
they have one), `CTX_SIZE` (floor 2048), `VRAM_LIMIT_MB` / budget knobs for the
rig, `TPS_FLOOR` if relevant. If they already chose a model card under
`docs/models/`, seed full `ENGINE_DEFAULTS` + `SAMPLER_DEFAULTS` from that
card’s Recommended settings (agentic vs coding) — never leave a previous
model’s leftover Baseline.

Remind: autoloop rewrites this file on accepts; it is gitignored; do not commit.

### 4. Optional — model that fits

Only if they want a GGUF now:

1. Prefer **llmfit** over whichllm for sizing (`docs/discovery/discover-models.md`).
2. Filter candidates by the **confirmed** pool from step 1.
3. Download with `hf` CLI into `models/<publisher>/<model>/` (drafts in
   `models/draft/`).
4. Set `MODEL` to the basename in Baseline.

Skip download if they only wanted the conceptual tour or already have a GGUF.

### 5. Optional — prove the stack (smoke)

If they ask to verify load/speed and a fitting model is configured:

- `scripts/verify_setup.py` (server + TPS), and/or
- `benchmark_search.py --validation` (throughput + Claw quick) — **one model**,
  no timeout games, stop live servers first.

Do not start overnight Search from here.

### 6. Dashboard (human watches)

When they want to see Baseline / recent Trials / Trial log:

```text
.\venv\Scripts\python.exe -m ui
```

Open `http://127.0.0.1:18765`. Read-only — no process control from the UI
(`ui/AGENTS.md`).

### 7. Close the loop

End with a crisp “you are here” + **next actions they can choose**:

1. Good-enough speed path → `docs/discovery/good-enough-tuning.md`
2. Full Trial on one GGUF → skill `trial`
3. Pick/discover another model → `docs/discovery/discover-models.md`
4. Course / HTML lessons → `teach/index.html` (if they are in student mode)

Do not start those paths unless they pick one.

## Output shape

Prefer short sections over essays (labels below are for you; render them in
the operator's language — usually pt-BR):

1. **Why it exists** (2–4 sentences)
2. **Ready vs missing** (checklist)
3. **What you just ran** (commands + outcomes)
4. **Risks on this rig** (OOM / unified headroom / dense vs MoE — concrete to them)
5. **Next step** (one recommended default + alternatives)

Terse is fine; do not be cryptic. New operators need the “why” once.

## References (read on demand, do not paste wholesale)

| Doc | When |
|---|---|
| `README.md` | Install, release assets, product caution |
| `docs/discovery/agent-onboarding.md` | Map + command crib |
| `docs/discovery/good-enough-tuning.md` | Post-setup default Search path |
| `docs/discovery/discover-models.md` | Choosing a GGUF |
| `CONTEXT.md` | Domain vocabulary |
| `program.md` | Search protocol (if they will run Search) |
| `GOLDEN-RULES.md` | Safety / validation discipline |
| `teach/index.html` | Student course portal |
