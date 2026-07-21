# AutoResearch Agent Work Contract

<!-- Scope: repo development agents. Research loop agents → read program.md -->

## Purpose
This repository implements an autonomous hill-climbing search system optimizing localized LLM runtime flags.

## Ownership
Repository-wide agent guidelines are owned by the repository developers.

## Local Contracts
- **DO NOT USE PYTHON TO EDIT FILES.** Use the Edit tool for targeted changes. Use Write only for new files or full rewrites the user explicitly requested. Never run a Python script to read-modify-write a data file — it corrupts headers, drops rows, and destroys formatting.
- Respond terse like smart caveman. All technical substance stay. Only fluff die.
- Loop agents: Strictly forbidden from editing code. If error/crash occurs, stop immediately, report error, warn user.
- Results local-only: Never push results, tweaks, or run branches to remote repository. Keep all benchmark runs offline.
- Model downloads: Always use `hf` CLI tool to download models, never web download scripts or browser. Land main GGUFs under `models/<publisher>/<model>/` (LM Studio nested layout). Config keeps basename only; harness resolves via `resolve_model_path`. Drafts stay in `models/draft/`. See [models/README.md](models/README.md).
- Parallel processes: NEVER run multiple validations, benchmarks, or command tasks in parallel. Always run one command/task at a time sequentially.
- Architecture: Never overengineer. Keep it simple. Less is more. Reduce lines of code. Simplify instead of complicate.
- Docs always: Update relevant docs (model cards, ADRs, config comments) whenever any codebase/model/config improvement is found or applied.
- Config surface: Agents and the Search loop change Baseline only via `autoresearch/core/config.py`. Do not drive Trials with CLI flag soup. Never edit `program.md` or harness code from the Search loop.
- **Hard gate (hooks):** Shell allowlist + gate-file protection. Scripts: `scripts/hooks/block-adhoc-eval.ps1`, `scripts/hooks/block-gate-tamper.ps1`. Wiring: `.cursor/hooks.json`, `.claude/settings.json`, `.cursor/rules/harness-trials.mdc`. Trial loop = edit `config.py` → `benchmark_search.py` / `autoloop.py`. **Disable playbook:** [docs/discovery/agent-shell-hard-gates.md](docs/discovery/agent-shell-hard-gates.md) §3 (teach human; wiring edits require unlock).
- Context size: CTX_SIZE default is 131072. User may lower it to trade context for speed. Code minimum is 2048 (llama.cpp practical floor). Always use the user-configured value.
- No timeouts: Never set execution timeouts on commands unless explicitly told to. Benchmarks and model tests run until completion.
- No hardcoded machine paths: Do not commit absolute user or checkout paths in scripts, docs, configs, or durable notes. Resolve them dynamically or keep them repo-relative.
- Ask first, ship never: When user asks "can we do X?", answer yes/no only. Do not implement unless user explicitly says "do it" / "implement" / "go ahead".
- Never assume. When uncertain whether a file is scratch, a decision is right, or a path is safe — ask the user or yourself explicitly before acting.
- NEVER commit and/or push without explicit user command. Wait for "commit", "commit and push", or equivalent. Do not infer intent.

## Work Guidance
- Use `/caveman lite|full|ultra|wenyan` for communication style constraint.
- Prioritize test-driven sanity. Verify logic changes using the test suite.
- Maintain mutable Baseline in local `autoresearch/core/config.py` (gitignored; seed from [config.py.example](autoresearch/core/config.py.example)). Visited memory lives in `.autoresearch_state.json`.
- `program.md` and evaluation harnesses are fixed unless the user explicitly requests a change.

## Verification
- Test with `pytest`. Ensure the full collected test suite passes.
- Inspect `results.tsv` to ensure it is not polluted or modified by agent logic.

## Pre-Task Reading
- Before starting any task, read the frontmatter of **every available tool and skill** — tool names, descriptions, parameter schemas, and skill files. Know what each can do before choosing one.
- Do not rely on memory. Re-read tool/skill descriptions each session. Tool schemas and skill instructions change.

# DOX framework

- DOX is highly performant AGENTS.md hierarchy installed here
- Agent must follow DOX instructions across any edits

## Core Contract

- AGENTS.md files are binding work contracts for their subtrees
- Work products, source materials, instructions, records, assets, and durable docs must stay understandable from the nearest applicable AGENTS.md plus every parent AGENTS.md above it

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
- user preferences about behavior, communication, process, organization, or quality
- AGENTS.md creation, deletion, move, rename, or index contents

Update parent docs when parent-level structure, ownership, workflow, or child index changes. Update child docs when parent changes alter local rules. Remove stale or contradictory text immediately. Small edits that do not change behavior or contracts may leave docs unchanged, but the DOX pass still must happen.

## Hierarchy

- Root AGENTS.md is the DOX rail: project-wide instructions, global preferences, durable workflow rules, and the top-level Child DOX Index
- Child AGENTS.md files own domain-specific instructions and their own Child DOX Index
- Each parent explains what its direct children cover and what stays owned by the parent
- The closer a doc is to the work, the more specific and practical it must be

## Child Doc Shape

- Create a child AGENTS.md when a folder becomes a durable boundary with its own purpose, rules, responsibilities, workflow, materials, or quality standards
- Work Guidance must reflect the current standards of the project or user instructions; if there are no specific standards or instructions yet, leave it empty
- Verification must reflect an existing check; if no verification framework exists yet, leave it empty and update it when one exists

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

## User Preferences

When the user requests a durable behavior change, record it here or in the relevant child AGENTS.md

- **Fair testing across models**: Always keep exactly 10 tasks per dataset for direct-coding evaluations (never 5 tasks) to guarantee fair model comparisons. Claw-Eval quick smoke (5 tasks) is exempt — it is observational smoke, not a cross-model score.
- **README language**: README.md must always be in pt-BR. Agent-facing docs (docs/, AGENTS.md, GOLDEN-RULES.md, CONTEXT.md, program.md) stay in English.
- **Agentic coding migration**: Treat HumanEval+/MBPP+/LiveCodeBench/BigCodeBench as direct-coding preflight benchmarks. Prefer long-horizon agentic targets for future coding-agent quality decisions once adapters exist.
- **Agentic-first Search**: Claw-Eval full is the canonical Val Score; Claw-Eval quick is smoke validation. Direct-coding is optional and, when enabled, uses exactly 10 tasks per dataset.
- **No eval-score floor**: Only the TPS Floor rejects a Trial. Claw-Eval quick/full scores are recorded for keep/discard comparison; low smoke scores must not short-circuit as `MODEL_REJECTED`.
- **config.py is the only mutable Baseline (local)**: Seed with `cp autoresearch/core/config.py.example autoresearch/core/config.py`. Agents and Search edit `ENGINE_DEFAULTS` / `SAMPLER_DEFAULTS` there. File is gitignored — do not commit machine Baseline. `program.md` and harnesses stay fixed. Do not drive Trials with CLI flag soup. `.autoresearch_state.json` is visited memory only.
- **Every requested Trial edits `config.py` first**: For each user-requested test/run, set the Baseline in `config.py` (then invoke harness). Never pass the experiment knobs as CLI flags.
- **No ad-hoc eval scripts**: Do not invent one-off Python/`python -c` Trial loops. Hooks deny them. Use harness CLIs only.
- **Portable agent hard-gates only**: Ship Cursor/Claude project hooks in-repo so any clone benefits. Do **not** require OS ACL (`icacls`), chmod lockdowns, or enterprise managed hooks for normal users (including non-devs).

## Child DOX Index
- [autoresearch/AGENTS.md](autoresearch/AGENTS.md) — Core autotuning package (config, runners, benchmarks).
- [scripts/hooks/block-adhoc-eval.ps1](scripts/hooks/block-adhoc-eval.ps1) — Shell hard-gate (allowlist + cwd).
- [scripts/hooks/block-gate-tamper.ps1](scripts/hooks/block-gate-tamper.ps1) — Gate-file hard-gate (Edit/Write/Delete).
- [.cursor/hooks.json](.cursor/hooks.json) — Cursor `beforeShellExecution` + `preToolUse` wiring.
- [.cursor/rules/harness-trials.mdc](.cursor/rules/harness-trials.mdc) — Always-on Trial/harness rule.
- [.claude/settings.json](.claude/settings.json) — Claude Code permissions.deny + PreToolUse wiring.
- [docs/discovery/agent-shell-hard-gates.md](docs/discovery/agent-shell-hard-gates.md) — Inventory + disable playbook.
- [models/README.md](models/README.md) — Shared GGUF store layout (nested LM Studio + basename resolve).
- [docs/AGENTS.md](docs/AGENTS.md) — Durable documentation contract.
  - [docs/models/](docs/models/) — Per-model GGUF specs (architecture, quant, settings).
  - [docs/adr/](docs/adr/) — Architecture decision records.
 - [docs/discovery/](docs/discovery/) — User-facing guides: model selection workflow, whichllm CLI reference, quantization cascade, agent onboarding, MTP inventory/TPS.
 - [docs/sessions/](docs/sessions/) — Empirical session logs (reproducibility evidence).
 - [docs/architecture.html](docs/architecture.html) — Interactive architecture diagram.
 - [docs/llamacpp-toolset.md](docs/llamacpp-toolset.md) — llama.cpp binary reference (build, bench, server, quantize).
- [scripts/AGENTS.md](scripts/AGENTS.md) — Operator scripts (setup, monitoring, server daemon).
- [tests/AGENTS.md](tests/AGENTS.md) — Unit and integration test suite.
- [teach/AGENTS.md](teach/AGENTS.md) — Course materials (Semana 1 TPS / Semana 2 quality); Dia 1 LM Studio, Dia 2+ this repo.
- External source submodules:
  - [llama.cpp/](llama.cpp/) - Upstream llama.cpp runtime source.
  - [llama.cpp-prismml/](llama.cpp-prismml/) - PrismML fork (bonsai model runtime).
  - [claw-eval/](claw-eval/) - Claw-Eval autonomous-agent benchmark harness source.

