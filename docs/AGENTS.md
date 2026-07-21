# `docs/` — Documentation Contract

<!-- Scope: any agent editing files under /docs. -->

## Purpose
Durable documentation for this repo: model cards, technique notes, architecture decision records (ADRs), per-session logs, and user-facing guides. Docs here are **stable contracts** — not diary entries, not commit logs.

## Ownership
- Repo root owns repo-wide rules (`/AGENTS.md`).
- This doc owns the contract for how `/docs/` is organized and what's allowed inside it.
- Each durable boundary under `/docs/` (e.g. `models/`, `adr/`, `discovery/`, `sessions/`) owns its own `AGENTS.md` and its own index.

## Local Contracts
- **One `AGENTS.md` per durable boundary.** A folder becomes a boundary when it has its own purpose, rules, workflow, or quality standards.
- **DOX hierarchy** is mandatory: walk root → leaf before editing; update nearest owning doc + affected parents after meaningful changes.
- **No run logs, no commit logs, no diary entries** in `/docs/` proper. Run logs go to `results.tsv`, git history, or `docs/sessions/` (only if capturing methodology for reproducibility).
- **Stable contracts only.** If a doc will be stale in a week, it doesn't belong here.
- **Markdown only** in this tree (except media assets and the architecture diagram `architecture.html`).
- **Reference external sources by URL**, not by copy-paste. We're a local repo, not a doc mirror.
- **User-facing guides** in `docs/discovery/` must be runnable by anyone with the documented steps. Avoid hardcoded private paths.

## Work Guidance
- New model card? Read `docs/models/AGENTS.md` first; follow its schema; add to its index.
- New architectural decision? Read `docs/adr/AGENTS.md` first; use the standard ADR format (numbered filename, status, date, context, decision, consequences).
- New technique or pattern worth preserving? Create a new top-level folder under `/docs/` with its own `AGENTS.md` explaining purpose, scope, and schema.
- New session log (single-day empirical capture)? Add to `docs/sessions/` with `YYYY-MM-DD-<topic>.md` naming.
- New user-facing guide? Add to `docs/discovery/` with runnable commands and methodology rationale.
- TBD / TODO / open questions: mark with `**TBD:**` inline, list them in a closing "Open questions" section. Resolve them when evidence arrives; never silently drop them.

## Verification
- Each model card must pass: filename matches index entry, "Sources / Verification" section has URLs + dates, GGUF metadata matches local file (`general.name`, `block_count`).
- Each ADR must have: status, date, context, decision, consequences.
- Each session log must capture: who ran what, on which hardware, with which config, measured numbers, decisions taken, errors encountered.
- Each user-facing guide must be: runnable, hardware-agnostic (or explicitly parameterized), and link to relevant session logs for evidence.
- No dead links to internal files (run a sanity grep on `[...](...)` before commit).

## Child DOX Index
- [docs/models/AGENTS.md](models/AGENTS.md) — model card schema + per-model index.
- [docs/adr/AGENTS.md](adr/AGENTS.md) — architecture decision records contract + index (ADRs 0001–0005).
- [docs/discovery/AGENTS.md](discovery/AGENTS.md) — user-facing guides (tooling, quantization, inference engines, speculative formats).
- [docs/sessions/AGENTS.md](sessions/AGENTS.md) — single-day empirical session logs index (2026-06-19 to 2026-07-20).
- [docs/architecture.html](architecture.html) — interactive architecture diagram (HTML exception under docs/).
- [docs/llamacpp-toolset.md](llamacpp-toolset.md) — llama.cpp toolset reference: build, binaries, flags, bench/server/quantize/perplexity/imatrix commands.
