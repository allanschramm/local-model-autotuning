# `docs/adr/` — Architecture Decision Records Contract

## Purpose
Durable record of architecture decisions for the `local-model-autotuning` project. ADRs record key architectural choices along with context, decision rationale, and consequences.

## Ownership
- Repo root `AGENTS.md` and `docs/AGENTS.md` own top-level architecture rules.
- This doc owns the ADR governance process, file naming convention, and Child DOX Index.
- Each ADR file (`000X-<short-title>.md`) is a leaf document.

## Local Contracts
- **Filename Format**: Sequential 4-digit prefix (`0001-deepen-llama-server-runner.md`, `0002-...`).
- **Required Structure**:
  1. Title (`# ADR 000X: Title`)
  2. Status (`Proposed`, `Accepted`, `Superseded by ADR 000Y`, `Deprecated`)
  3. Date (`YYYY-MM-DD`)
  4. Context & Problem Statement
  5. Decision
  6. Consequences (Positive, Negative, Neutral)
- **Immutability**: Accepted ADRs are immutable. To change a decision, create a new ADR superseding the previous one.

## Work Guidance
- New architectural choice? Create a new numbered file following the schema.
- Update status in this index and in the superseded ADR file if a decision changes.

## Verification
- Every ADR file must have: Status, Date, Context, Decision, Consequences.
- Numbering must be strictly sequential without gaps.

## Child DOX Index
- [`0001-deepen-llama-server-runner.md`](./0001-deepen-llama-server-runner.md) — Deepen llama-server runner.
- [`0002-consolidated-evaluation-harness.md`](./0002-consolidated-evaluation-harness.md) — Consolidated evaluation harness.
- [`0003-in-process-benchmark-orchestration.md`](./0003-in-process-benchmark-orchestration.md) — In-process benchmark orchestration.
- [`0004-agentic-first-search.md`](./0004-agentic-first-search.md) — Agentic-first Search (Val Score); Baseline location superseded by 0005.
- [`0005-config-py-mutable-baseline.md`](./0005-config-py-mutable-baseline.md) — `config.py` as mutable Baseline; state = visited memory only.
