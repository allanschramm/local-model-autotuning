# ADR 0013: Agentic Coding Loop (SWE-lite) as Night Selector

**Date:** 2026-08-13
**Status:** Superseded in part — Night ctx floor demoted to a historical lens note by [0017](0017-rank-membership-quality-first.md) (2026-08-26): `NIGHT_CTX_FLOOR` no longer filters membership anywhere; ctx only breaks ±0.05 near-ties in the Night table (Day tie-break = TPS).
**Supersedes in part:** [0006](0006-pareto-frontier-search.md) Night Usage Profile pick (`min(agentic, coding)` only). Does **not** change Pareto Set membership or the four maximize axes.
**Does not supersede:** Claw-Eval full as the `agentic` Objective Vector axis ([0004](0004-agentic-first-search.md) / [0006](0006-pareto-frontier-search.md)); coding-10 as the `coding` axis; Day pick ([0009](0009-day-profile-tps-floor.md)); Point = GGUF basename ([0012](0012-basename-pareto-point.md)).

## Context & Problem Statement

Claw-Eval full measures office-style JSON tool use against mock HTTP. Coding-10 is one-shot codegen. Neither measures a Pi / Cursor-goal / Claude-loop session: workspace tools, one GitHub-issue-shaped goal, stop when tests pass or the agent is stuck.

Night still ranked `min(claw, coding-10)`. Models can score well on both and still repeat the same tool call or hallucinate paths in a real repo loop. Live GitHub issues as Trial tasks would be non-reproducible.

## Decision

1. **Keep Claw** as the `agentic` axis. Do not replace it with coding-agent scores.
2. **Add `agentic_coding`:** a SWE-lite pack of frozen issue fixtures (issue markdown + pinned mini-workspace + hidden fail-to-pass pytest). One session = one issue. No live `gh` during a Trial. No Docker, no remote judge ([0004](0004-agentic-first-search.md)).
3. **Pass** = hidden tests green **and** no loop/hallucination flag. Fail (task score 0) on: identical `(tool, args)` ≥ 3 times; ≥ N turns with no allowlisted file-hash change; tool name or path outside the allowlist / worktree; `max_turns` with tests still red; `run_tests` on a docs-only issue (restraint).
4. **Pareto `on_front` stays four-axis** (ctx × TPS × claw × coding-10). `agentic_coding` is a TSV column and a **Night selector**, not a fifth domination axis in this ADR.
5. **Night:** among points with `CTX ≥ NIGHT_CTX_FLOOR` **and** a measured `agentic_coding` score, maximize `min(agentic, coding, agentic_coding)`. If no complete point has `agentic_coding` yet, fall back to `min(agentic, coding)` (today’s Night).
6. **Day:** unchanged at 0013 time (`TPS ≥ DAY_TPS_FLOOR` then `min(agentic, coding)`); **historical lens since [0017](0017-rank-membership-quality-first.md)** — the floor no longer gates anything, TPS breaks Day near-ties.
7. **CLI:** `--agentic-coding` / `--no-agentic-coding`. Default **off** until operators opt in for Night claims. Full Trials do not require the column for `on_front`.

## Consequences

### Positive
- Night can reject Claw/coding winners that loop or hallucinate in a workspace.
- Historical TSV rows stay complete; ranking degrades gracefully until the new column is populated.

### Negative
- Night claims without `--agentic-coding` still use the old maximin (Pocket-class false Night picks remain until measured).
- A later ADR is required to promote `agentic_coding` to a fifth Pareto axis (would mark old rows incomplete).

### Neutral
- Score is `passed / N` like Claw. Loop vs tests-red vs hallucination lives in the Trial description, not a blended IQ.
- Visual dashboard issues stay out of v1 (no browser judge).
