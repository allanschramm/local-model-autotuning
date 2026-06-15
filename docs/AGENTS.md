# `docs/` — Documentation Contract

<!-- Scope: any agent editing files under /docs. -->

## Purpose
Durable documentation for this repo: model cards, technique notes, architecture decision records (ADRs), and any future per-feature docs. Docs here are **stable contracts** — not diary entries, not run logs, not commit summaries.

## Ownership
- Repo root owns repo-wide rules (`/AGENTS.md`).
- This doc owns the contract for how `/docs/` is organized and what's allowed inside it.
- Each durable boundary under `/docs/` (e.g. `models/`, `adr/`) owns its own `AGENTS.md` and its own index.

## Local Contracts
- **One `AGENTS.md` per durable boundary.** A folder becomes a boundary when it has its own purpose, rules, workflow, or quality standards.
- **DOX hierarchy** is mandatory: walk root → leaf before editing; update nearest owning doc + affected parents after meaningful changes.
- **No run logs, no commit logs, no diary entries** in `/docs/`. Those go to `results.tsv`, git history, or session memory.
- **Stable contracts only.** If a doc will be stale in a week, it doesn't belong here.
- **Markdown only** in this tree (except for media assets in sub-folders when needed).
- **Reference external sources by URL**, not by copy-paste. We're a local repo, not a doc mirror.

## Work Guidance
- New model card? Read `docs/models/AGENTS.md` first; follow its schema; add to its index.
- New architectural decision? Use the existing `docs/adr/` format (numbered filename, status, date, context, decision, consequences).
- New technique or pattern worth preserving? Create a new top-level folder under `/docs/` with its own `AGENTS.md` explaining purpose, scope, and schema.
- TBD / TODO / open questions: mark with `**TBD:**` inline, list them in a closing "Open questions" section. Resolve them when evidence arrives; never silently drop them.

## Verification
- Each model card must pass: filename matches index entry, "Sources / Verification" section has URLs + dates, GGUF metadata matches local file (`general.name`, `block_count`).
- Each ADR must have: status, date, context, decision, consequences.
- No dead links to internal files (run a sanity grep on `[...](...)` before commit).

## Child DOX Index
- [docs/models/AGENTS.md](models/AGENTS.md) — model card schema + per-model index.
- [docs/adr/](adr/) — architecture decision records.
  - 0001 — Deepen llama-server runner.
  - 0002 — Consolidated evaluation harness.
  - 0003 — In-process benchmark orchestration.
