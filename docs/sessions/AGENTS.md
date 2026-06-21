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
- **Allan-specific paths** (`/home/shark/...`) are acceptable here because session logs are operator-specific. They are NOT acceptable in `docs/discovery/` or `docs/models/` (user-facing).
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
None — `docs/sessions/` is a leaf.
