# Independent Benchmark Harness

## Purpose
Standalone quality measurement through the version-pinned Pi SDK. Local and cloud models are connected endpoints; tuning and model-server management are outside this module.

## Ownership
Repository developers. Parent: [root contract](../AGENTS.md).

## Local Contracts
- This directory must work when copied outside the repository. No imports from `autoresearch`, no mutable Baseline, GGUF discovery, or GPU gates.
- Pi version, tools, prompts, reasoning settings, task bytes, and budgets identify the protocol. Pi results never populate legacy Claw/Coding score columns.
- SQLite `results.db` is canonical. This module owns only `benchmark_runs` and `benchmark_tasks`; existing tables remain untouched.
- Workspace packs expose only allowlisted files; Claw packs expose only declared mock API tools. Hidden verifiers/rubrics remain inaccessible to model tools and run after the session.
- `pi-workspace-v2` resolves relative/absolute Pi tool paths against the workspace, then checks exact allowlist membership and realpath equality. Preserve v1 result identity; path-policy changes require a new protocol.
- `--pack` selects a registered frozen corpus and is recorded in run configuration. Default `workspace-smoke` retains its five tasks; `humaneval-base10` calls `check(candidate)` and `mbpp-base10` executes its literal base assertions. Neither is expanded plus coverage. Never pool their scores with legacy coding columns.
- `lcb-v6-stdin10` preserves the ten legacy `test6.jsonl` selections and 40 private stdin cases per task. Hidden programs use fresh cwd/processes and exact stdout after trailing-newline removal. Missing archives are infrastructure errors. No automatic command timeout; select the four easy tasks for initial local smoke.
- `bigcode-hard10` preserves the ten legacy v0.1.4 instruct tasks and original unittest text. Solution and tests share one module namespace inside the private verification cwd. Use Agg for plots, retain the legacy Windows shell-message compatibility, and classify missing verifier dependencies as infrastructure errors. Pack requirements record validated direct dependency versions.
- `claw-quick5`/`claw-full15` use `pi-claw-v1`: Pi runs original prompts/tool schemas against frozen FastAPI apps in a fresh Python ASGI process per task. No listening ports, built-in filesystem tools, retries or compaction. Preserve 4096 output tokens, original turn limits, capped tool output and legacy completion scoring/threshold 0.5. Store completion score separately from pass/fail; forbidden-tool safety checks veto passing. `wrong_data`/`wrong_recipients` remain explicitly unevaluated.
- Claw's last five full-tier tasks need live web. Start those only with explicit `--allow-live-web` and `SERP_DEV_KEY`; otherwise record infrastructure errors with null score/pass. Service children receive a minimal environment and close before checkpointing. Never substitute mock web data for real-web tasks.
- `benchmark_tasks.score`/`score_json` store Claw grading in SQLite through additive migration. Historical and workspace rows retain null weighted scores; never rewrite legacy result tables.
- Pi resources and session state are isolated from personal settings, skills, extensions, and repository instructions. Credentials come from environment, never result artifacts.
- Write E2E checks before implementation; retain repeatable SQLite and trajectory evidence. Fixture verifiers are trusted code, not an OS sandbox for arbitrary datasets.

## Work Guidance
- See [README.md](README.md) for installation, endpoint configuration, and the staged extraction.
- See [MIGRATION.md](MIGRATION.md) for the legacy inventory, verifier limitations, compatibility consumers and retirement criteria. Local endpoints are the validation priority; cloud validation is deferred.
- Keep tasks and verifiers unchanged when moving the frozen pack. The legacy runner still consumes that pack for historical protocol compatibility.

## Verification
- `node --check main.mjs` checks syntax. `BENCHMARK_PYTHON=<venv-python> npm test` executes the standalone E2E and retains SQLite/trajectory evidence in the printed temp directory.
- After changes to the LiveCodeBench importer, run the selected Python's `test/lcb_import_security.py` to check literal-only decoding and malformed-input rejection.
- `BENCHMARK_PYTHON=<venv-python> node test/claw_e2e.mjs` verifies standalone Claw apps, grading, safety veto, fixture integrity, state isolation, disabled live web and SQLite evidence. Install `claw-requirements.txt` in that Python first.

## Child DOX Index
- [packs/AGENTS.md](packs/AGENTS.md) — frozen dataset provenance, oracle isolation and reproduction. Default tasks and E2E execution remain owned by this parent.
