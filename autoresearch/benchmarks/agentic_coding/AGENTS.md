# `autoresearch/benchmarks/agentic_coding/` — SWE-lite issue loop

## Purpose
Frozen GitHub-issue fixtures plus the workspace tool loop and loop/hallucination detector that produce the `agentic_coding` Trial column (ADR 0013).

## Ownership
Repository developers. Parent: [`autoresearch/AGENTS.md`](../../AGENTS.md).

## Local Contracts
- Tasks are frozen snapshots (issue markdown + mini-workspace + hidden pytest). No live `gh` during a Trial.
- One session = one issue. Worktrees live in OS `$TEMP`, never under the repo.
- Hidden tests are not in the prompt. Pass = tests green and no detector flag.
- No Docker, no remote judge, no network tools.

## Work Guidance
- Add a task by copying an existing `tasks/<id>/` tree: `task.yaml`, `workspace/`, `hidden_tests/`.
- Detector rules live in `detector.py`; keep them deterministic and unit-tested.
- PATH tools must stay in the worktree **and** on `task.allowlist` (writes = exact file; reads/list/grep may use parent dirs of allowlisted files).
- Stall (`STALL_TURNS`) counts consecutive **mutating** turns with no allowlisted file-hash change. Read-only inspection does not stall.

## Verification
`.\venv\Scripts\python.exe -m pytest tests/test_agentic_coding.py`

## Note on the future replacement
The DM-Code-Agent 30-task real-SWE eval ([`benchmarks/mini_swe_agent/AGENTS.md`](../mini_swe_agent/AGENTS.md)) is the operator's v1 sibling of SWE-lite. Both layers write their own columns in `results.db` (`mini_swe_agent` is observation-only in v1 — the leaderboard does not render it yet). Retiring SWE-lite in favor of DM-Code-Agent is **out of scope for v1** and is a separate ADR (likely 0019); the 5-task frozen pack stays because it is hermetic, no external fixture, runnable in CI.

## Child DOX Index
None
