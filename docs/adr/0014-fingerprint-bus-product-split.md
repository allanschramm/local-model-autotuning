# ADR 0014: Fingerprint Bus and Product Split (Direction)

**Date:** 2026-08-15
**Status:** Accepted (direction); Benchmark dependency and official Pi evaluation partially superseded by [0018](0018-independent-pi-benchmark-and-context-tuning.md). The phase table below is the original decision snapshot; current staging is in 0018. `teach/` is out of scope until an explicit peel.
**Supersedes in part:** [0006](0006-pareto-frontier-search.md) as the **required product journey and auto picker** (Day/Night maximin is no longer the path that elects what you run in Pi). Does **not** delete the four-axis Pareto Set, TSV columns, or `rank_results.py`.
**Does not supersede:** [0005](0005-config-py-mutable-baseline.md) Baseline location; [0004](0004-agentic-first-search.md) local-only eval (no Docker / remote judge); [0013](0013-agentic-coding-night-selector.md) SWE-lite as an optional numeric pack; [0009](0009-teach-day-night-agent-harness.md) until `teach/` is peeled.

## Context & Problem Statement

Claw-Eval full and coding-10 elect models that can still loop or feel dumb in Pi / OpenCode. Night maximin ranked POCKET-class points over Ornith-35B, which the operator actually enjoys. The repo currently jams four products into one Search loop: TPS hill-climb, IQ benches, `teach/`, and launching a server for a coding agent.

Needed: a portable **Fingerprint** so hill-climb, optional IQ benches, and the launcher/Pi share one GGUF + engine; Pi is the quality that matters; numbers stay for video/compare; course lives elsewhere later.

## Outcome

An operator (or later a student, outside this repo) can:

1. Pick a GGUF (community/portal — **parked**, not this repo).
2. Hill-climb **TPS** under the existing PPL guard. Climb **writes** a Fingerprint file (engine frozen).
3. Launcher loads that file and serves Pi (visual / real work).
4. Optionally pin the **same** file for Claw / SWE-lite / HumanEval-style numbers (YouTube table). Visual tasks never write rank.

## Required path (this repo)

```
pick GGUF → TPS climb (PPL) writes Fingerprint
         → launcher loads Fingerprint → Pi (+ visual pack)
         → optional numeric pack on the same file
```

Sampler is user/card choice, not a TPS Neighbor. Engine change after TPS = new Fingerprint, new Pi run, new optional IQ row.

## Decision

1. **Fingerprint file is the bus.** Portable: GGUF **basename** (no absolute user paths) + `ENGINE_DEFAULTS`. Optional `SAMPLER_DEFAULTS`. Hill-climb writes. Launcher reads. Numeric bench pins. Visual pack uses the same file and **must not** elect a champion or write Pareto/`on_front`.
2. **Search in this product is TPS under PPL.** Not claw×coding domination as the thing you “ship” to Pi. Existing `--mode tps` + 1% PPL guard stays the climb. Four-axis Pareto remains a **report**, not the required picker.
3. **Pi is the IQ that gates daily use.** Claw-Eval, coding-10, SWE-lite are an **optional numeric pack** for comparable scores and videos. They are not a required step before Pi.
4. **Pi cannot replace all evals.** TPS/PPL = `llama-bench` / `llama-perplexity`. Claw-Eval = mock HTTP + JSON tools + `task.yaml` (different exam). coding-10 / HumanEval = one-shot (Pi tools invalidate it). Workspace-shaped tasks (SWE-lite fixtures, visual prompts, later CRUD/API) **may** use a **pinned** Pi as runner; that is not “Claw via Pi.”
5. **One runtime library.** VRAM gates, single-load, release binaries stay in the hill-climb/runtime package. Eval starts an **ephemeral** server and kills it. Daily / visual / Pi uses the **launcher only**. Never two full loads.
6. **Numeric + visual.** Numeric = frozen pass/fail, comparable. Visual = YouTube-style workspace prompts (Luke-style pack as reference, not a vendored tree). Same Fingerprint. Visual may use a short private rubric (app runs, state persists) for the camera; still not rank.
7. **Packages in this git tree later.** Do **not** split TPS / bench / launcher into other remotes until (a) the Fingerprint file exists and (b) one GGUF has walked pick → TPS → optional IQ → Pi visual on that file.
8. **`teach/`:** do not edit. Later: peel to a **private** remote; remove curriculum from this public tree; keep `docs/`. Community-suggest lives on that portal (**parked**).
9. **No auto Night/Day champion** in the launcher until Q6 is unparked. No `daily` flag in v1 of the file.
10. **Do not copy** third-party launcher trees under PolyForm Noncommercial. UX reference only.

## Fingerprint file (contract sketch)

**TBD** on-disk path and exact YAML keys. Constraints that are not TBD:

- Gitignored (machine-local, like `config.py` / aliases).
- `schema_version` required.
- `model` = GGUF basename only.
- `engine` = the `ENGINE_DEFAULTS` used for that climb (ctx, KV, batch, threads, MTP/spec, offload, …).
- `sampler` optional.
- No hostnames, emails, GPU SKUs, absolute user paths, alias names.

Recommended first path (implement later): one file per basename under a gitignored `fingerprints/` directory, written from Baseline after a TPS climb.

## Non-goals (v1)

- Changing `teach/` HTML, `teach/SPEC.md`, or ADR 0009.
- Git-splitting this repo.
- Promoting `agentic_coding` to a fifth Pareto axis.
- Using live Pi / OpenCode / Claude Code as `results.tsv` scorers for Claw or coding-10.
- Full-stack CRUD/API bench (later, workspace-shaped).
- Auto-picking a default model.

## Ship cut

| Phase | What | Status |
|---|---|---|
| 0 | This spec | done |
| 1 | Write/read Fingerprint file from TPS climb + Baseline | not started |
| 2 | Launcher consumes the file (ephemeral eval still uses runtime lib) | not started |
| 3 | Numeric pack pins the file (optional CLI) | not started |
| 4 | Visual pack (same file, no rank write) | not started |
| 5 | Package folders in-tree after one GGUF walks the path | not started |
| 6 | Peel `teach/` to private remote | not started; do not start without an explicit peel |

## Consequences

### Positive

- Engine flags that won TPS are the flags Pi actually sees.
- Quality that matches daily use (Pi) is no longer pretended to be Claw/coding-10.
- Course and Search can diverge without leaking curriculum into the public tree.

### Negative

- No automatic “Night model” until Q6. Operators pick.
- Optional numeric scores will lag Pi experience (already true).
- Fingerprint format is a new machine-local artifact to keep in sync with `config.py` until writers exist.

### Neutral

- `results.tsv` / Pareto Set remain valid as a numeric archive.
- ADR 0013 SWE-lite stays available as optional pack or future pinned-Pi runner.

## Open questions (parked)

- **Q6:** What, if anything, replaces Night pick / a launcher default.
- **Q11:** Where community model suggestions live (private portal with teach).
- On-disk Fingerprint path and YAML schema (phase 1).

## References

- Operator lock: 2026-08-15 design tree (Fingerprint bus, TPS-then-Pi, optional IQ, packages-later, `teach/` frozen).
- [good-enough-tuning.md](../discovery/good-enough-tuning.md) — TPS + PPL climb.
- [0013](0013-agentic-coding-night-selector.md) — why Claw/coding-10 miss Pi loops.
