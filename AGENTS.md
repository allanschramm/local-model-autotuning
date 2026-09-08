# AutoResearch Agent Work Contract

<!-- Scope: repo development agents. Research loop agents → read program.md -->

## Purpose
Autonomous multi-objective Search over local LLM runtime flags; Pareto frontier = configured ctx × TPS × agentic × coding. Domain language: [CONTEXT.md](CONTEXT.md).

## Ownership
Repository developers.

## Local Contracts
- Baseline only via gitignored `autoresearch/core/config.py` (seed from `config.py.example`). No Trial CLI flag soup. `program.md` / harnesses fixed unless user asks.
- **Upstream-first (mandatory):** Check upstream `llama.cpp/common/arg.cpp` `add_opt` (`~310` flags) + `docs/llamacpp-toolset.md` + `docs/llamacpp-flags-audit.md` before adding any `ENGINE_*/SAMPLER_*` flag or estimator — `90%` already exists upstream (`fit`/`mlock`/`yarn`/`dry`/`xtc`/`fit-target`/`load-mode`/`warmup`/`cache-ram`/`kv-unified`/`repack`/`yarn` etc.), harness is passthrough mapper in `autoresearch/core/llama_runner.py:895 _build_cmd`; keep only host/WDDM/thermal/Pareto gates (`common/fit.h:14` assumes unlimited host).
- Do not edit vendor trees: `llama.cpp/`, `claw-eval/`, `VITRIOL/`, `llama.cpp-releases/`.
- Never commit private paths, emails, hostnames, GPU SKUs, aliases, or machine Baseline.
- `models/` is a real store directory. NEVER recursive/forced delete (`Remove-Item -Recurse`, `rm -rf`, `rmdir /s`) on the `models/` root — it wipes the GGUF store and private notes. Per-file deletes in subdirs are fine.
- No command timeouts: Never set execution timeouts on commands run by the agent unless explicitly told. Benchmarks and model tests run until completion. This does not prohibit bounded network timeouts inside product code when needed to prevent a stalled service or request from blocking indefinitely.
- **No autonomous autoloop:** When the user asks for a trial, hill climb, bench, validate, or any tuning/eval action, do **NOT** launch `autoloop.py` or the `/autoresearch` autonomous loop. Use the explicit harness (`benchmark_search.py`, `python -m autoresearch.runners.run`, or the Trial skill's `--agentic-full` command). `autoloop.py` is operator-only — if the user wanted the background hill-climb loop, they would start it themselves.

## Work Guidance
- **Runtime (do not rebuild)**: use the prebuilt release `llama.cpp-releases/upstream/b10819` (nightly, CUDA 13.3) via `AUTORESEARCH_LLAMA_CPP_ROOT` — smoke-validated 2026-09-05 (LFM2.5 bench 74 t/s, quick 5/5); carries Nemotron-3.5 embedded-MTP (`nextn_predict_layers`). `./llama.cpp` is source-only; its `build-cuda/` artifacts are stale (`b10099`, pre-MTP) and rebuilding does NOT fix Nemotron (`b10819` supersedes `b10549`, deleted after green smoke; `b10375` remains on disk — Machine-scope env still points there, User scope is `b10819`).
- Method + Trial procedure: `CONTEXT.md`, `docs/adr/`, `docs/discovery/`, `autoresearch/AGENTS.md`, `program.md`.
- Full Trial operator skill (Claw-15 + coding-10, sequential queues): [`.agents/skills/trial/SKILL.md`](.agents/skills/trial/SKILL.md).

## Verification
- `.\venv\Scripts\python.exe -m pytest` (or `scripts/run_validate.py` / pre-commit). Never system-global Python.
- Local Windows pytest does not execute POSIX branches (`fcntl`, Unix-only paths). After every push, wait for `.github/workflows/validate.yml` (`ubuntu-latest`) with `.\venv\Scripts\python.exe scripts\watch_validate.py` until it exits 0. A green local suite is not CI-complete.

# DOX framework

DOX is the repo’s AGENTS.md hierarchy. It is a **project contract**. Agents follow it on every edit.

## Core Contract

- AGENTS.md files are binding work contracts for their subtrees.
- Work products, source materials, instructions, records, assets, and durable docs must stay understandable from the nearest applicable AGENTS.md plus every parent AGENTS.md above it.

## Read Before Editing

1. Read the root AGENTS.md
2. Identify every file or folder you expect to touch
3. Walk from the repository root to each target path
4. Read every AGENTS.md found along each route
5. If a parent AGENTS.md lists a child AGENTS.md whose scope contains the path, read that child and continue from there
6. Use the nearest AGENTS.md as the local contract and parent docs for repo-wide rules
7. If docs conflict, the closer doc controls local work details, but no child doc may weaken DOX

Do not rely on memory. Re-read the applicable DOX chain in the current session before editing.

## Update After Editing

Every meaningful change requires a DOX pass before the task is done.

Update the closest owning AGENTS.md when a change affects:

- purpose, scope, ownership, or responsibilities
- durable structure, contracts, workflows, or operating rules
- required inputs, outputs, permissions, constraints, side effects, or artifacts
- AGENTS.md creation, deletion, move, rename, or index contents

Update parent docs when parent-level structure, ownership, workflow, or child index changes. Update child docs when parent changes alter local rules. Remove stale or contradictory text immediately. Small edits that do not change behavior or contracts may leave docs unchanged, but the DOX pass still must happen.

## Hierarchy

- Root AGENTS.md is the DOX rail: project-wide instructions, durable workflow rules, and the top-level Child DOX Index
- Child AGENTS.md files own domain-specific instructions and their own Child DOX Index
- Each parent explains what its direct children cover and what stays owned by the parent
- The closer a doc is to the work, the more specific and practical it must be

## Child Doc Shape

- Create a child AGENTS.md when a folder becomes a durable boundary with its own purpose, rules, responsibilities, workflow, materials, or quality standards
- Work Guidance must reflect the current standards of the project; if none yet, leave it empty
- Verification must reflect an existing check; if none yet, leave it empty until one exists

Default section order:

- Purpose
- Ownership
- Local Contracts
- Work Guidance
- Verification
- Child DOX Index

## Style

- Keep docs concise, current, and operational
- Document stable contracts, not diary entries
- Put broad rules in parent docs and concrete details in child docs
- Prefer direct bullets with explicit names
- Do not duplicate rules across many files unless each scope needs a local version
- Delete stale notes instead of explaining history
- Trim obvious statements, repeated rules, misplaced detail, and warnings for risks that no longer exist

## Closeout

1. Re-check changed paths against the DOX chain
2. Update nearest owning docs and any affected parents or children
3. Refresh every affected Child DOX Index
4. Remove stale or contradictory text
5. Run existing verification when relevant
6. Report any docs intentionally left unchanged and why

## Child DOX Index

- [autoresearch/AGENTS.md](autoresearch/AGENTS.md) — package (config, runners, benchmarks)
- [ui/AGENTS.md](ui/AGENTS.md) — operator dashboard
- [docs/AGENTS.md](docs/AGENTS.md) — docs tree (`models/`, `adr/`, `discovery/`, `sessions/`)
- [scripts/AGENTS.md](scripts/AGENTS.md) — operator scripts
- [tests/AGENTS.md](tests/AGENTS.md) — test suite
- [teach/AGENTS.md](teach/AGENTS.md) — course materials (gitignored / local-only)
- [models/README.md](models/README.md) — GGUF layout
- [.agents/skills/trial/SKILL.md](.agents/skills/trial/SKILL.md) — full Trial skill (Claw-15 + coding-10; tracked carve-out under otherwise-gitignored `.agents/`)
- [.agents/skills/validation/SKILL.md](.agents/skills/validation/SKILL.md) — model validation skill (download, metadata check, smoke eval; tracked carve-out)
- [.agents/skills/inference-research/SKILL.md](.agents/skills/inference-research/SKILL.md) — inference-performance research skill (flags, engines, quantization, spec decoding; engine-side only, no model search)
- [.pre-commit-config.yaml](.pre-commit-config.yaml) · [.github/workflows/validate.yml](.github/workflows/validate.yml) · [pyproject.toml](pyproject.toml)
- External read-only: `llama.cpp/` · `claw-eval/` · `llama.cpp-releases/` · `VITRIOL/`
