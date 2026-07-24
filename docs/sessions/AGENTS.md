# AGENTS.md — docs/sessions

## Purpose
Single-day empirical session logs. Captures what was run, on which hardware, with which config, what was measured, decisions taken, and errors encountered. Used as **reproducibility evidence** for the user-facing guides in `docs/discovery/` and as a memory of approaches that did or did not work.

## Ownership
- Owned by: `local-model-autotuning` developers and operators.
- Stable contracts: file naming (`YYYY-MM-DD-<topic>.md`), section shape (Goal / Hardware / Commands / Findings / Errors).

## Local Contracts
- **One file per session or per significant sub-iteration of a session.** Don't merge multiple sessions.
- **Verbatim tool outputs** are preferred over paraphrased summaries. The point is reproducibility.
- **Errors and corrections are first-class.** When an approach was wrong, log it explicitly so future operators don't repeat.
- **Operator-specific paths** (historical session references) are acceptable here because session logs are operator-specific. They are NOT acceptable in `docs/discovery/` or `docs/models/` (user-facing).
- **Do not edit a session log after the session is complete** except to fix typos. Add a follow-up file instead.
- **No external-source URLs in technique claims** (per `docs/models/` rules — methodology names allowed, citations not).

## Work Guidance
- New session → new file with `YYYY-MM-DD-<topic>.md` filename.
- Captured data: tool output (results.tsv excerpts, logs), measured TPS, config tested, errors hit, decisions taken, who approved what.
- Cross-link to related session logs and to model cards / ADRs / discovery guides when relevant.

## Verification
- Each file has: Goal, Hardware, Setup, Commands (reproducible), Findings, Errors, Decisions.
- Hard numbers (TPS, score, VRAM) reported as measured, not estimated.
- "Correções M3" or similar self-correction sections are encouraged.

## Child DOX Index
- [`2026-06-19-alias-system.md`](./2026-06-19-alias-system.md) — Alias system setup and design.
- [`2026-06-19-mtp-baseline.md`](./2026-06-19-mtp-baseline.md) — MTP baseline benchmarking and verification.
- [`2026-06-19-whichllm-coding.md`](./2026-06-19-whichllm-coding.md) — whichllm evaluation on coding benchmarks.
- [`2026-06-19-whichllm-plan.md`](./2026-06-19-whichllm-plan.md) — whichllm search and selection planning.
- [`2026-06-19-whichllm-source-deepdive.md`](./2026-06-19-whichllm-source-deepdive.md) — whichllm source code analysis.
- [`2026-06-23-4bench-integration.md`](./2026-06-23-4bench-integration.md) — 4bench evaluation harness integration.
- [`2026-06-26-ornith-baseline-and-validation.md`](./2026-06-26-ornith-baseline-and-validation.md) — Ornith model baseline validation.
- [`2026-06-29-beellama-tcq-copyspec-dflash-iq3.md`](./2026-06-29-beellama-tcq-copyspec-dflash-iq3.md) — BeeLlama TCQ, CopySpec, and DFlash experiments.
- [`2026-07-01-dense-model-validation.md`](./2026-07-01-dense-model-validation.md) — Dense model execution validation.
- [`2026-07-01-gemma4-v2-q3km-validation.md`](./2026-07-01-gemma4-v2-q3km-validation.md) — Gemma 4 v2 Q3_K_M validation.
- [`2026-07-01-ornith-1.0-9b-analysis.md`](./2026-07-01-ornith-1.0-9b-analysis.md) — Ornith 1.0 9B detailed benchmark analysis.
- [`2026-07-06-windows-model-up.md`](./2026-07-06-windows-model-up.md) — Windows model launcher (`model-up`) validation.
- [`2026-07-20-llama-cli-validation.md`](./2026-07-20-llama-cli-validation.md) — llama-cli execution & validation log.
- [`2026-07-20-root-memory-archive.md`](./2026-07-20-root-memory-archive.md) — Root memory archive and empirical notes.
- [`2026-07-20-small-model-tps-matrix.md`](./2026-07-20-small-model-tps-matrix.md) — Small-model MTP TPS empirical matrix (8 GB).
- [`2026-07-23-nanbeige42-tps-matrix.md`](./2026-07-23-nanbeige42-tps-matrix.md) — Nanbeige4.2-3B arch fork + KV/batch TPS matrix (8 GB).
- [`2026-07-23-lfm2.5-8b-a1b-validation.md`](./2026-07-23-lfm2.5-8b-a1b-validation.md) — LFM2.5-8B-A1B Q4_K_M validation + full-VRAM vs exps→CPU A/B.
- [`2026-07-24-claw-full-smoke-high.md`](./2026-07-24-claw-full-smoke-high.md) — Claw-Eval full queue + historical Val Score ceiling (Laguna 0.6667).
- [`2026-07-24-lfm2.5-1.2b-ctx-kv-matrix.md`](./2026-07-24-lfm2.5-1.2b-ctx-kv-matrix.md) — LFM2.5-1.2B claw-quick ctx/KV matrix (65k f16 preferred).
