# Independent Model Benchmark

Quality evaluation through a pinned Pi SDK. Connect a local endpoint or cloud provider; no tuning installation, Baseline or GGUF is required. This directory can be copied into its own repository.

Original AILOCAL code is [PolyForm Noncommercial 1.0.0](LICENSE); imported components keep their original licenses. See [NOTICE.md](NOTICE.md) for scope and pack attribution.

## Install and list tasks

Requires Node.js 24+ and Python 3.11+ with pytest for the frozen verifiers. For a fresh installation:

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install pytest
npm ci --prefix benchmark
node benchmark/main.mjs --list
```

On POSIX create the venv with `python3 -m venv venv` and install pytest with `./venv/bin/python -m pip install pytest`. Use that Python path in `--python`. Neither path needs the tuning dependencies.

When using this directory on its own, run `npm ci` and `node main.mjs` instead.

## Evaluate a local or compatible endpoint

Start the model server separately, with the desired context and runtime settings. Supply its actual advertised context; the Benchmark does not verify or change server allocation.

```powershell
node benchmark/main.mjs --provider local --model <served-model-id> --base-url http://127.0.0.1:18080/v1 --context-window 65536 --python .\venv\Scripts\python.exe --task issue-clamp-inclusive
```

Authenticated endpoints additionally accept `--api-key-env MODEL_API_KEY`; set that environment variable outside the command. `--api` selects a Pi-supported protocol for custom endpoints (default: `openai-completions`). Do not put credentials in the URL.

For a built-in Pi cloud provider, omit `--base-url` and use its provider/model IDs plus the provider's API-key environment variable, or select one with `--api-key-env`. The SDK's pinned catalog supplies model metadata. Personal Pi OAuth files/settings/extensions are not loaded.

```powershell
node benchmark/main.mjs --provider <provider-id> --model <model-id> --python .\venv\Scripts\python.exe
```

Without `--pack`, `workspace-smoke` selects the original five frozen tasks. `--pack humaneval-base10` and `--pack mbpp-base10` select their respective ten-task coding subsets. Without `--task`, every task in the selected pack runs sequentially; repeat `--task` to select several. Exit codes: `0` all passed, `1` at least one model/policy/turn-limit failure, `2` infrastructure or configuration error. No automatic retry or model fallback.

## HumanEval base pack

```powershell
node benchmark/main.mjs --pack humaneval-base10 --list
node benchmark/main.mjs --pack humaneval-base10 --provider local --model <served-model-id> --base-url http://127.0.0.1:18080/v1 --context-window 65536 --python .\venv\Scripts\python.exe
```

The frozen pack contains HumanEval/0 through HumanEval/9, the same IDs selected by the legacy coding-10 HumanEval loader. Pi edits only `solution.py`; original assertions run as hidden pytest and explicitly invoke `check(candidate)`. This is original HumanEval base coverage, not expanded HumanEval+ or the full dataset. [Migration inventory](MIGRATION.md) describes the legacy verifier limitation and why its historical scores cannot be compared directly.

`packs/humaneval-base10/source.json` records the EvalPlus source snapshot, IDs and hashes. The MIT license is included. Oracle solutions are outside task workspaces and used only for E2E validation. Running the frozen pack requires no EvalPlus installation or network download.

To reproduce the snapshot, install `evalplus==0.3.1` in an export environment and run `python benchmark/import_coding.py --pack humaneval-base10 --output <new-pack-directory>`. The exporter requires dataset v0.1.10 and the recorded subset digest; it refuses upstream drift and refuses to overwrite an existing directory. The old `import_humaneval.py` command remains a compatibility entrypoint with the same default and byte-identical output.

## MBPP base pack

```powershell
node benchmark/main.mjs --pack mbpp-base10 --list
node benchmark/main.mjs --pack mbpp-base10 --provider local --model <served-model-id> --base-url http://127.0.0.1:18080/v1 --context-window 65536 --python .\venv\Scripts\python.exe
```

This freezes the legacy coding-10 MBPP selection: Mbpp/2, /3, /4, /6, /7, /8, /9, /11, /12 and /14. The source is EvalPlus's MBPP+ v0.2.0 snapshot; verification executes only its original `assertion` field. The module docstring preserves the supplied prompt/example and a placeholder names the required function. Hidden pytest imports that entrypoint and executes the base assertions. Model-written helpers/imports stay in `solution.py`.

This is a legacy-selected smoke subset, including IDs that the original MBPP paper used for prompting; it is not a standard MBPP test-split score, the full corpus, or expanded MBPP+ coverage. Preserve the literal assertions, including their expected result types. Pi protocol and task hashes remain separate from legacy single-shot coding scores.

One inherited limitation is Mbpp/2: its public example compares results through `set(...)`, but the hidden base assertions require an exact tuple. A set with the correct elements still fails. The snapshot preserves this criterion; interpret that failure as a type-sensitive dataset limitation rather than a runtime error.

The pack includes source IDs/hashes, oracle solutions for E2E only, EvalPlus's Apache-2.0 license and original MBPP attribution/license links in `NOTICE.md`. Running the frozen pack needs only Pi's installed dependencies and pytest, without EvalPlus or dataset downloads. Reproduce it with `python benchmark/import_coding.py --pack mbpp-base10 --output <new-pack-directory>` in the pinned `evalplus==0.3.1` export environment.

## LiveCodeBench v6 stdin pack

```powershell
node benchmark/main.mjs --pack lcb-v6-stdin10 --list
node benchmark/main.mjs --pack lcb-v6-stdin10 --provider local --model <served-model-id> --base-url http://127.0.0.1:18080/v1 --context-window 65536 --python .\venv\Scripts\python.exe --task lcb-000-abc387_b
```

This is the exact ten-record stdin selection from the legacy `test6.jsonl` loader, in source order: abc387_b/f/a/c, abc388_d/b/g/c/a/f. It contains 400 private cases and mixed difficulties. The `test6.jsonl` increment is not the complete aggregate `release_v6` benchmark. Select individual tasks for a bounded smoke; the four easy tasks are `lcb-000-abc387_b`, `lcb-002-abc387_a`, `lcb-005-abc388_b`, and `lcb-008-abc388_a`.

Each task exposes only `solution.py` and its original problem statement/examples. Hidden pytest runs the submitted stdin/stdout program against all 40 private cases, with a fresh subprocess and temporary working directory per case. Comparison preserves the legacy rule: exact stdout after stripping trailing newline characters, without trimming spaces or applying numeric tolerance. Public examples remain visible in the statement but are not separately scored. The official LiveCodeBench/Harbor graders differ; do not label this slice an official leaderboard score.

There is no automatic command timeout. Some medium/hard cases have large inputs, and incorrect algorithms can run for a long time. Use explicit task selection for the initial local smoke. The bundled fixtures are trusted; their model-written programs are not enclosed in an OS security sandbox. Missing/corrupt private-case archives produce infrastructure failures with no model score.

Private cases are stored as deterministic gzip JSON, marked binary in Git. The imported source uses a pickle wrapper, but the importer never unpickles it: after checking the pinned whole-file SHA256 it extracts one literal JSON string through a restricted opcode set with a decompression size cap. No pickle bytes are consumed during Benchmark execution. Source IDs, case counts, difficulty, contest metadata, source revision and hashes are in `packs/lcb-v6-stdin10/source.json`. Attribution and the source card's license declaration are preserved in `NOTICE.md`; the dataset does not provide gold solutions.

To reproduce into a fresh directory:

```powershell
hf download livecodebench/code_generation_lite test6.jsonl --repo-type dataset --revision 0fe84c3912ea0c4d4a78037083943e8f0c4dd505
.\venv\Scripts\python.exe benchmark/import_lcb.py --source <downloaded-test6.jsonl> --output <new-pack-directory>
.\venv\Scripts\python.exe benchmark/test/lcb_import_security.py
```

Running the frozen pack needs neither the source JSONL nor Hugging Face dependencies. The standalone E2E rejects every unimplemented stdin task, checks a known solution to 9x9 Sum, rejects sample-only/whitespace/syntax failures, and classifies missing archives as infrastructure. It does not claim oracle validation of all ten algorithms.

## BigCodeBench Hard pack

```powershell
.\venv\Scripts\python.exe -m pip install -r benchmark/packs/bigcode-hard10/requirements.txt
node benchmark/main.mjs --pack bigcode-hard10 --list
node benchmark/main.mjs --pack bigcode-hard10 --provider local --model <served-model-id> --base-url http://127.0.0.1:18080/v1 --context-window 65536 --python .\venv\Scripts\python.exe
```

The pack freezes the first ten legacy selections from BigCodeBench Hard v0.1.4: /13, /15, /17, /19, /34, /37, /82, /89, /92 and /93. It preserves the original instruct prompt and code header. Models write only `solution.py`. Original tests run after generation in a private verification copy, in the same module namespace as the solution, followed by explicit execution of `TestCases` through `unittest`. This retains references to imports/global objects and mocked FTP/process interactions. Plots use the headless Agg backend. The legacy Windows compatibility accepts `not recognized` where the original shell assertion expects `not found`, only during suite execution.

`requirements.txt` pins the direct library versions used for validation, including scientific/plotting and Flask libraries. It is not a complete transitive environment lock or the official leaderboard image. Missing verifier dependencies or fixture files produce infrastructure errors (`passed=NULL`); invalid generated code and failed assertions remain model failures. Install dependencies in the Python passed to `--python`. No automatic execution timeout or OS sandbox is added. This ten-task Pi instruct slice is not the full 148-task Hard score, and legacy single-shot scores remain separate.

Source revision, selected-row hash, original prompt/test hashes, dependency names and test counts are in `source.json`. `oracles.json` contains upstream solutions for E2E only, outside every model workspace. The Apache-2.0 license and attribution are included. Reproduce into a fresh directory with:

```powershell
hf download bigcode/bigcodebench-hard data/v0.1.4-00000-of-00001.parquet --repo-type dataset --revision 298d2cc7b96612e15e47313c3603ee124cee0c1f
.\venv\Scripts\python.exe benchmark/import_bigcode.py --source <downloaded-parquet> --output <new-pack-directory>
```

Reproduction needs `datasets`; running frozen tasks does not. The standalone E2E rejects all ten stubs, accepts all ten upstream oracles and checks missing-library classification using the real Pi loop, hidden verification and SQLite.

## Protocol and evidence

### DM-Code-Agent coding/maintenance

```powershell
node benchmark/main.mjs --pack dm-code30 --list
node benchmark/main.mjs --pack dm-code30 --provider local --model <served-model-id> --base-url http://127.0.0.1:18080/v1 --context-window 65536 --python .\venv\Scripts\python.exe
$env:BENCHMARK_PYTHON = (Resolve-Path .\venv\Scripts\python.exe).Path
node benchmark/test/dm_e2e.mjs
```

`dm-code30` freezes all 15 coding and 15 maintenance tasks from DM-Code-Agent commit `e4530be37e6f972d9735381e524cab4b5f732ddd`, in upstream order. Original setup/hidden file contents, prompts, names, tags, both test commands, step limits and allowed/required changed-file lists are retained. Run one or several IDs with `--task`; without selection it runs all 30. Execution needs the selected Python with pytest, without installing `dm_agent`, mini-swe-agent, Docker or Hugging Face packages.

`pi-dm-v2` uses an isolated Pi session with 4096 output tokens, temperature 0, thinking requested off, and the source task's step count as its turn limit. Models read/write only existing allowlisted files and inspect public tests as specifications. There is no shell, public-test execution tool, retry or continuation. Python verification runs only after the model session, so generated code cannot return host/hidden-file contents through an interactive test tool. Plans stay in responses. New-file requests are policy failures. Keep `pi-dm-v1`, legacy mini-swe-agent and upstream DM agent scores separate.

Hidden files never enter the model workspace. After the session, the harness snapshots changed bytes, excludes only cache/version-control directories, rejects symlinks, copies the workspace to a separate verification directory, then stages every declared hidden file at its original relative path. The trusted runner uses the source pytest arguments with `-c` pointing to an empty configuration, preventing edited `pyproject.toml` from selecting only public tests. Credentials and personal pytest plugins are not inherited. No test output is fed back into the model session. Verification still executes generated Python with host permissions: a separate process is not an OS sandbox, and this protocol does not promise protection from actively malicious code or grader tampering.

Verification records the pre-verifier changed-file list, forbidden edits, missing required edits and hidden-test verdict in SQLite `benchmark_tasks.score_json` and the trajectory artifact. A correct implementation still fails `scope_violation` for a forbidden/missing required edit. Missing hidden fixtures or verifier infrastructure produces `passed=NULL`, never a fake zero. Hidden assertions run even when scope fails, so both diagnoses remain visible. Existing result rows are not rewritten and no weighted Claw score is assigned to this protocol.

The pack's `source.json` records four whole-source hashes and a complete 30-task contract digest. `upstream/` contains read-only MIT source provenance; `oracles.json` holds four reference controls outside model tool access. They are not a 30-task gold solution set. The E2E rejects all 30 initial states, accepts those four references and checks scope/path restrictions, missing fixtures, pytest-selection bypass, early process exit and absence of interactive execution, retaining SQLite/trajectory evidence.

To reproduce, download these four files from `https://raw.githubusercontent.com/hwfengcs/DM-Code-Agent/e4530be37e6f972d9735381e524cab4b5f732ddd/` into a source directory, using their basenames: `dm_agent/benchmarks/tasks.py`, `dm_agent/benchmarks/models.py`, `tests/test_coding_benchmarks.py`, and `LICENSE`. Then run `python benchmark/import_dm.py --source <source-directory> --output <fresh-directory>`. The exporter checks every file hash before executing the stdlib-only data modules and refuses overwrite. The committed `packs/dm-code30/upstream/` directory can also serve as that source directory.

### Claw quick/full

```powershell
.\venv\Scripts\python.exe -m pip install -r benchmark/claw-requirements.txt
node benchmark/main.mjs --pack claw-quick5 --list
node benchmark/main.mjs --pack claw-full15 --provider local --model <served-model-id> --base-url http://127.0.0.1:18080/v1 --context-window 65536 --python .\venv\Scripts\python.exe
$env:BENCHMARK_PYTHON = (Resolve-Path .\venv\Scripts\python.exe).Path
node benchmark/test/claw_e2e.mjs
```

`pi-claw-v2` preserves the five quick and fifteen full legacy-selected IDs in `packs/claw-tiers/source.json`. The full tier contains ten offline tasks and five real-web tasks. The latter require an explicit `--allow-live-web` plus `SERP_DEV_KEY`; otherwise each records `infra_error`, `passed=NULL`, `score=NULL`, without calling the model or internet. Live search can incur provider charges. Local model endpoints remain the priority; no provider-backed live-web validation is claimed.

Pi receives the original task prompt and declared tool names/descriptions/JSON schemas. It has no filesystem or shell tools, rubric, reference solution, personal resources, retries or compaction. The output budget stays 4096 tokens per turn, with original task turn limits. Tool results use the legacy 10,000-character head/tail cap. Final grading prefers the last assistant's visible text and falls back to its thinking text when empty, matching the legacy grader input rule. There is no automatic command timeout or continuation after a generation cap.

Each task starts a fresh Python child with a minimal environment. Frozen upstream FastAPI apps run through TestClient/ASGI, with fixture environment references, reset endpoints, lifecycle cleanup, error injection disabled and the declared mock date. No service listens on fixed TCP ports; no process or state is shared between tasks. Live-web services, when explicitly enabled, retain their actual outbound network behavior. API calls, arguments, results/statuses and the Pi trajectory remain in the task artifact. Missing dependencies, corrupted fixtures, child failures and service HTTP 5xx are infrastructure failures, not model zeroes. HTTP 4xx and application-level error replies remain tool results.

The copied legacy deterministic scorer preserves component weights, keyword/category rules and completion threshold 0.5. Its weighted score and details are stored in SQLite `benchmark_tasks.score`/`score_json`; additive migration leaves historical/workspace weighted scores null. Rule limitations remain: keyword rules accept any listed keyword, category rules accept half, and tool rules count invocations even when a tool result contains an error. This measures that narrow rubric, not comprehensive correctness.

Safety is separate: declared `tool_not_called` and `wrong_action` calls veto a pass (`safety_violation`), while retaining the raw completion score. Semantic `wrong_data` and `wrong_recipients` checks are explicitly reported as unevaluated. The legacy adapter ignored these safety declarations. This single-trial Pi protocol is neither legacy Claw score parity nor upstream Pass^3/full-grader coverage.

Task, fixture and offline-service bytes remain unchanged under `packs/claw-tiers/upstream/` with MIT attribution and per-file hashes. Live web uses the maintained `claw_web.py` derivative; `adapter_sources` records the original server hash instead of shipping its vulnerable executable implementation. The extracted completion scorer has its own pinned hash. Reproduce into a fresh directory using `python benchmark/import_claw.py --source <claw-checkout-containing-pinned-commit> --legacy-scorer benchmark/packs/claw-tiers/legacy_score.py --output <new-directory>`. The original checkout commit includes local upstream integration; the committed snapshot plus adapter supply every runtime file, so execution needs neither that checkout nor `autoresearch`.

Version 2 pins each fetch connection to a verified public IP, validates every redirect, preserves TLS certificate/hostname verification and rejects private, mixed-DNS and IPv6 transition destinations. Only HTTP/HTTPS standard ports are supported; environment proxies, automatic redirects/retries and compressed responses are disabled. Bodies are capped at 2 MiB before HTMLParser extracts plain text. Generic tool errors withhold internal exception details. The adapter is included in task hashes. Keep `pi-claw-v1` history separate; no old score is rewritten.

The dedicated E2E copies only `benchmark/` outside the repo, runs real mock apps through Pi with scripted model responses, checks positive/negative grading, safety veto, fixture integrity, task state isolation, migration of existing SQLite rows and disabled live web, and retains `results.db`/`verification.json`. Scripted controls are not canonical task solutions or model-quality measurements.

### Workspace packs

`pi-workspace-v3` uses Pi 1.4.0 with allowlisted `workspace_read` and `workspace_write` adapters over Pi tools, a fixed system prompt, temperature 0, thinking off, 4096 output tokens per turn, and each task's frozen turn limit. No shell, personal resources, compaction, or automatic recovery. V3 retains v2's path policy and adds a completion guard: zero exit alone is insufficient; pytest must return nonempty JUnit and finish every collected test's call phase, without skips, interruption or failures. Early `os._exit(0)`, partial `pytest.exit` and partial skips cannot pass. Runner/verifier bytes enter task hashes. Keep v1/v2 history separate. This is not an OS sandbox or a guarantee against actively adversarial grader tampering.

Tool requests accept relative or absolute paths resolving to an exact allowlisted workspace file, consistent with Pi's tool schemas. Paths outside that set and symlink redirections remain blocked. Version 1 rejected absolute paths even to allowed files; its historical results retain their protocol and must not be pooled with version 2.

Hidden verifiers are inaccessible to model tools. They run after the session in a separate copy using the selected Python. Verifiers execute model-written Python and are **not an OS security sandbox**; use the trusted bundled fixtures. Additional datasets need their own execution policy.

The current directory's `results.db` is canonical (`--db` selects a different SQLite file). This module creates only `benchmark_runs` and `benchmark_tasks`, preserving historical tables. Each task checkpoints an artifact under `benchmark-artifacts/<run-id>/` containing the transcript, final allowlisted files, verifier output, protocol, Pi version, task hash, and outcome. Infrastructure failures have `passed=NULL`, never a fabricated zero score. Interrupted runs remain `running` with any completed-task checkpoints; they are not complete evaluations.

The task hash includes the manifest, initial workspace, and hidden verifier. Per-run configuration records prompts, tools, output/turn budgets and declared/catalog context; each trajectory also records the effective thinking level selected by Pi. Context metadata is not a measurement of effective long-context capacity. Token usage is kept as normalized by Pi; SDK cost estimates are omitted. No local-runtime TPS or cloud cost is inferred from missing metadata. Pi scores do not enter legacy rankings.

## Repeatable E2E check

```powershell
$env:BENCHMARK_PYTHON = (Resolve-Path .\venv\Scripts\python.exe).Path
node benchmark/test/e2e.mjs
```

The check copies this directory and installed dependencies outside the repo, then uses the real Pi SDK against a simulated streaming HTTP provider. It checks pass/fail, authentication, infrastructure errors, path/hidden-test access, turn limits, fixed request budget, resource isolation, and SQLite/trajectory evidence. No model load or paid request occurs. It prints the retained artifact directory and writes `verification.json` beside `results.db`. `BENCHMARK_E2E_DIR` can select a fresh artifact directory.

## Extraction status

Independent Auto tuning is documented in the parent repo's [`autoresearch/TUNING.md`](../autoresearch/TUNING.md): bounded TPS × context Search under a fixed PPL ceiling, without quality-benchmark imports. It can produce several portable Fingerprints; model-server launch stays outside this Benchmark. Legacy entrypoints remain compatibility paths. The two modules can be extracted into separate repositories after operational validation. See [ADR 0018](../docs/adr/0018-independent-pi-benchmark-and-context-tuning.md) for the decision.
