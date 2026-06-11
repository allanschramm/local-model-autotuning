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

## Child DOX Index
- [ClawBench/AGENTS.md](file:///home/shark/workspace/autoresearch-public/ClawBench/AGENTS.md) — Agency benchmark harness and sandboxing boundaries.

