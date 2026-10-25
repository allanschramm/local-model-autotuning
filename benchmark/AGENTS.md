# Independent Benchmark Harness

## Purpose
Standalone quality measurement through the version-pinned Pi SDK. Local and cloud models are connected endpoints; tuning and model-server management are outside this module.

## Ownership
Repository developers. Parent: [root contract](../AGENTS.md).

## Local Contracts
- This directory must work when copied outside the repository. No imports from `autoresearch`, no mutable Baseline, GGUF discovery, or GPU gates.
- Pi version, tools, prompts, reasoning settings, task bytes, and budgets identify the protocol. Pi results never populate legacy Claw/Coding score columns.
- SQLite `results.db` is canonical. This module owns only `benchmark_runs` and `benchmark_tasks`; existing tables remain untouched.
- Frozen tasks expose only allowlisted workspace files. Hidden verifiers remain inaccessible to model tools and run after the session.
- Pi resources and session state are isolated from personal settings, skills, extensions, and repository instructions. Credentials come from environment, never result artifacts.
- Write E2E checks before implementation; retain repeatable SQLite and trajectory evidence. Fixture verifiers are trusted code, not an OS sandbox for arbitrary datasets.

## Work Guidance
- See [README.md](README.md) for installation, endpoint configuration, and the staged extraction.
- Keep tasks and verifiers unchanged when moving the frozen pack. The legacy runner still consumes that pack for historical protocol compatibility.

## Verification
- `node --check main.mjs` checks syntax. `BENCHMARK_PYTHON=<venv-python> npm test` executes the standalone E2E and retains SQLite/trajectory evidence in the printed temp directory.

## Child DOX Index
None; tasks and E2E verification remain owned here.
