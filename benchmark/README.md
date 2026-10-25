# Independent Model Benchmark

Quality evaluation through a pinned Pi SDK. Connect a local endpoint or cloud provider; no tuning installation, Baseline or GGUF is required. This directory can be copied into its own repository.

## Install and list tasks

Requires Node.js 24+ and a Python environment with pytest for the frozen verifiers.

```powershell
npm ci --prefix benchmark
node benchmark/main.mjs --list
```

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

Without `--task`, the five frozen tasks run sequentially. Repeat `--task` to select several. Exit codes: `0` all passed, `1` at least one model/policy/turn-limit failure, `2` infrastructure or configuration error. No automatic retry or model fallback.

## Protocol and evidence

`pi-workspace-v2` uses Pi 1.4.0 with allowlisted `workspace_read` and `workspace_write` adapters over Pi tools, a fixed system prompt, temperature 0, thinking off, 4096 output tokens per turn, and each task's frozen turn limit. No shell, personal resources, compaction, or automatic recovery. This measures that constrained Pi configuration, not vanilla Pi or Terminal-Bench.

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
