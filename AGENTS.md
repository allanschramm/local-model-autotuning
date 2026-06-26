# AutoResearch Agent Work Contract

<!-- Scope: repo development agents. Research loop agents → read program.md -->

## Purpose
This repository implements an autonomous hill-climbing search system optimizing localized LLM runtime flags.

## Ownership
Repository-wide agent guidelines are owned by the repository developers.

## Local Contracts
- Respond terse like smart caveman. All technical substance stay. Only fluff die.
- Loop agents: Strictly forbidden from editing code. If error/crash occurs, stop immediately, report error, warn user.
- Results local-only: Never push results, tweaks, or run branches to remote repository. Keep all benchmark runs offline.
- Architecture: Never overengineer. Keep it simple. Less is more. Reduce lines of code. Simplify instead of complicate.

## Work Guidance
- Use `/caveman lite|full|ultra|wenyan` for communication style constraint.
- Prioritize test-driven sanity. Verify logic changes using the test suite.
- Maintain config single source of truth in [config.py](file:///home/shark/workspace/autoresearch-public/autoresearch/core/config.py).

## Verification
- Test with `pytest`. Ensure all 101 tests pass.
- Inspect `results.tsv` to ensure it is not polluted or modified by agent logic.

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

- **Fair testing across models**: Always keep exactly 10 tasks per dataset for evaluations (never 5 tasks) to guarantee fair model comparisons.

## Child DOX Index
- [autoresearch/AGENTS.md](autoresearch/AGENTS.md) — Core autotuning package.
- [docs/AGENTS.md](docs/AGENTS.md) — durable documentation (model cards, ADRs, discovery guides, session logs).
  - [docs/models/](docs/models/) — per-model GGUF specs.
  - [docs/adr/](docs/adr/) — architecture decision records.
  - [docs/discovery/](docs/discovery/) — user-facing guides (whichllm, Pareto frontier, model selection).
  - [docs/sessions/](docs/sessions/) — empirical session logs.
- [scripts/AGENTS.md](scripts/AGENTS.md) — Operator scripts (setup, monitoring, configuration server daemon).
- [tests/AGENTS.md](tests/AGENTS.md) — Unit and integration test suite.


<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **local-model-autoresearch** (879 symbols, 1317 relationships, 52 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> Index stale? Run `node .gitnexus/run.cjs analyze` from the project root — it auto-selects an available runner. No `.gitnexus/run.cjs` yet? `npx gitnexus analyze` (npm 11 crash → `npm i -g gitnexus`; #1939).

## Always Do

- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows. For regression review, compare against the default branch: `detect_changes({scope: "compare", base_ref: "main"})`.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `query({query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `context({name: "symbolName"})`.

## Never Do

- NEVER edit a function, class, or method without first running `impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `rename` which understands the call graph.
- NEVER commit changes without running `detect_changes()` to check affected scope.

## Resources

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/local-model-autoresearch/context` | Codebase overview, check index freshness |
| `gitnexus://repo/local-model-autoresearch/clusters` | All functional areas |
| `gitnexus://repo/local-model-autoresearch/processes` | All execution flows |
| `gitnexus://repo/local-model-autoresearch/process/{name}` | Step-by-step execution trace |

## CLI

| Task | Read this skill file |
|------|---------------------|
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus/gitnexus-cli/SKILL.md` |

<!-- gitnexus:end -->
