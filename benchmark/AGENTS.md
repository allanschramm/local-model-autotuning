# Independent Benchmark Harness

## Purpose
Standalone quality measurement through the version-pinned Pi SDK. Local and cloud models are connected endpoints; tuning and model-server management are outside this module.

## Ownership
Repository developers. Parent: [root contract](../AGENTS.md).

## Local Contracts
- Original contributions use the local LICENSE and NOTICE.md; ship both when extracting this directory. Preserve every imported pack/dependency license. The undici security override is part of the npm lock; keep Pi SDK 1.4.0 pinned and require npm audit at high severity in CI.
- This directory must work when copied outside the repository. No imports from `autoresearch`, no mutable Baseline, GGUF discovery, or GPU gates.
- Pi version, tools, prompts, reasoning settings, task bytes, and budgets identify the protocol. Pi results never populate legacy Claw/Coding score columns.
- SQLite `results.db` is canonical. This module owns only `benchmark_runs` and `benchmark_tasks`; existing tables remain untouched.
- Workspace packs expose only allowlisted files; Claw packs expose only declared mock API tools. Hidden verifiers/rubrics remain inaccessible to model tools and run after the session.
- `pi-workspace-v3` retains v2's relative/absolute allowlist and realpath policy and requires demonstrated nonempty pytest completion. Keep v1/v2 history separate; verifier/runtime hashes include verifier.mjs and pytest_runner.py.
- `--pack` selects a registered frozen corpus and is recorded in run configuration. Default `workspace-smoke` retains its five tasks; `humaneval-base10` calls `check(candidate)` and `mbpp-base10` executes its literal base assertions. Neither is expanded plus coverage. Never pool their scores with legacy coding columns.
- `lcb-v6-stdin10` preserves the ten legacy `test6.jsonl` selections and 40 private stdin cases per task. Hidden programs use fresh cwd/processes and exact stdout after trailing-newline removal. Missing archives are infrastructure errors. No automatic command timeout; select the four easy tasks for initial local smoke.
- `bigcode-hard10` preserves the ten legacy v0.1.4 instruct tasks and original unittest text. Solution and tests share one module namespace inside the private verification cwd. Use Agg for plots, retain the legacy Windows shell-message compatibility, and classify missing verifier dependencies as infrastructure errors. Pack requirements record validated direct dependency versions.
- `claw-quick5`/`claw-full15` use `pi-claw-v2`: Pi runs original prompts/tool schemas against private ASGI apps per task. Offline apps remain frozen; `claw_web.py` owns the security-maintained live-web adapter. Preserve 4096 output tokens, original turn limits, capped tool output and completion scoring/threshold 0.5. Keep v1 history separate. Store completion score separately from pass/fail; forbidden-tool safety checks veto passing. `wrong_data`/`wrong_recipients` remain explicitly unevaluated.
- Live-web fetch resolves every redirect hop once, rejects non-public/transition IPs and nonstandard ports, and connects to the validated IP with original TLS hostname/certificate checks. No environment proxies or automatic redirects/retries. Bound response bytes, use HTMLParser for plain text and return generic errors. Never replace these checks with URL-prefix validation or suppress CodeQL findings.
- Claw's last five full-tier tasks need live web. Start those only with explicit `--allow-live-web` and `SERP_DEV_KEY`; otherwise record infrastructure errors with null score/pass. Service children receive a minimal environment and close before checkpointing. Never substitute mock web data for real-web tasks.
- `benchmark_tasks.score`/`score_json` store Claw grading in SQLite through additive migration. Historical and workspace rows retain null weighted scores; never rewrite legacy result tables.
- `dm-code30` uses `pi-dm-v2`: preserve all 30 frozen fixtures, metadata and scope constraints, but expose only allowlisted read/write tools. Public tests are readable specifications; no generated Python executes during the agent session. Keep plans in responses. Stage hidden files only in the post-agent verification copy, after the scope snapshot. Preserve v1 history. Store scope/completion details in SQLite `score_json`, leaving weighted score null.
- verifier.mjs/pytest_runner.py own common pytest execution and completion evidence. Passing needs zero exit, nonempty JUnit and every collected test's call phase completed without skips/interruption/failures. This is a completion guard, not an adversarial attestation or OS sandbox. Post-agent Python still has host permissions, with no verifier feedback into model context.
- DM hidden execution adds a harness-created empty pytest configuration to the original command prefix. Generated `pyproject.toml` must not control hidden-test selection; keep the bypass regression in the E2E.
- Reject missing DM contracts, inconsistent step budgets and empty hidden-file lists during task loading; never fall back to the generic workspace grader for an incomplete DM task.
- Pi resources and session state are isolated from personal settings, skills, extensions, and repository instructions. Credentials come from environment, never result artifacts.
- Always scrub the explicitly selected API key, regardless of its variable name or length, including escaped forms and JSON property names in free-form data. Preserve program-owned protocol/score metadata; credential-like public identifiers are not secret-bearing payloads.
- Write E2E checks before implementation; retain repeatable SQLite and trajectory evidence. Fixture verifiers are trusted code, not an OS sandbox for arbitrary datasets.

## Work Guidance
- See [README.md](README.md) for installation, endpoint configuration, and the staged extraction.
- See [MIGRATION.md](MIGRATION.md) for the legacy inventory, verifier limitations, compatibility consumers and retirement criteria. Local endpoints are the validation priority; cloud validation is deferred.
- Keep tasks and verifiers unchanged when moving the frozen pack. The legacy runner still consumes that pack for historical protocol compatibility.

## Verification
- Root CI runs workspace, Claw and DM E2Es in independent matrix jobs with suite-specific Python dependencies and separate SQLite/log/trajectory artifact uploads. All use fixture endpoints without model loading or paid web/API calls.
- `node --check main.mjs` checks syntax. `BENCHMARK_PYTHON=<venv-python> npm test` executes the standalone E2E and retains SQLite/trajectory evidence in the printed temp directory.
- The same E2E retains `integrity/results.db` for arbitrary-name, short/escaped and JSON-key credential leaks, HTTP diagnostics and early-exit rejection.
- After changes to the LiveCodeBench importer, run the selected Python's `test/lcb_import_security.py` to check literal-only decoding and malformed-input rejection.
- `BENCHMARK_PYTHON=<venv-python> node test/claw_e2e.mjs` verifies standalone Claw apps, grading, safety veto, fixture integrity, state isolation, disabled live web and SQLite evidence. Install `claw-requirements.txt` in that Python first.
- Claw E2E also runs `test/claw_web_security.py`: deterministic ASGI/loopback fixtures check private/mixed DNS, IP pinning, redirects, TLS arguments, proxy independence, body limits, HTML extraction and exception/credential handling, including the real bridge. Its `web-security/results.db` and verification/log evidence are uploaded by CI. No external requests or paid search in this check.
- `BENCHMARK_PYTHON=<venv-python> node test/dm_e2e.mjs` checks all 30 initial states, four upstream reference controls, public/hidden separation, scope enforcement, missing fixtures and path restrictions through Pi/pytest/SQLite.

## Child DOX Index
- [packs/AGENTS.md](packs/AGENTS.md) — frozen dataset provenance, oracle isolation and reproduction. Default tasks and E2E execution remain owned by this parent.
