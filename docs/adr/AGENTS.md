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
- [`0004-agentic-first-search.md`](./0004-agentic-first-search.md) — Agentic measurements; Val Score keep superseded by 0006; Baseline location by 0005.
- [`0005-config-py-mutable-baseline.md`](./0005-config-py-mutable-baseline.md) — `config.py` as mutable Baseline; state = visited memory only.
- [`0006-pareto-frontier-search.md`](./0006-pareto-frontier-search.md) — Multi-objective Pareto Set (ctx × TPS × agentic × coding); Day/Night pick; status vocabulary. Day pick superseded by 0007 then 0008.
- [`0007-day-profile-speed-band.md`](./0007-day-profile-speed-band.md) — Day = speed band then IQ (superseded by 0008).
- [`0008-day-iq-epsilon-then-tps.md`](./0008-day-iq-epsilon-then-tps.md) — Day = IQ ε-band then max TPS; Night maximin unchanged.
- [`0009-teach-day-night-agent-harness.md`](./0009-teach-day-night-agent-harness.md) — Teach path: Day/Night usage + Agent Harness arc (`teach/SPEC.md`).
- [`0010-cross-platform-zombie-process-prevention.md`](./0010-cross-platform-zombie-process-prevention.md) — Cross-platform zombie process prevention (Job Objects, `PR_SET_PDEATHSIG`, process groups, pre-flight port checks).
- [`0011-dashboard-ailocal-design.md`](./0011-dashboard-ailocal-design.md) — Dashboard adopts the AILOCAL design language; `ui/` widens to allow static assets (no external deps).
- [`0012-basename-pareto-point.md`](./0012-basename-pareto-point.md) — Global Pareto Point = GGUF basename (max claw/coding/TPS/ctx); Fingerprint for Search/pick hint.
- [`0013-agentic-coding-night-selector.md`](./0013-agentic-coding-night-selector.md) — SWE-lite `agentic_coding` Night selector; Claw stays the agentic axis; four-axis Pareto unchanged.
- [`0014-fingerprint-bus-product-split.md`](./0014-fingerprint-bus-product-split.md) — Fingerprint bus; TPS-then-Pi journey; Pareto Set remains a report; `teach/` frozen.
- [`0015-rocm-first-binary-resolution-windows.md`](./0015-rocm-first-binary-resolution-windows.md) — ROCm-first binary resolution on Windows; Vulkan0 hardcode removal; `--no-reasoning-preserve` fix.
- [`0016-measurement-hygiene-and-morris-screen.md`](./0016-measurement-hygiene-and-morris-screen.md) — Thermal settle, TPS median, crash journal, Morris engine-knob pin; Neighbor Search unchanged.
- [`0017-rank-membership-quality-first.md`](./0017-rank-membership-quality-first.md) — Rank = leaderboard (every complete model once, quality-first, near-tie band); domination is a same-basename config label; cross-model `dominated` ceases (supersedes part of 0006/0009/0013).
