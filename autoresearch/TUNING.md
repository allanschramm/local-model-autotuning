# Independent Auto tuning

The bounded tuning entrypoint is `python -m autoresearch.tuning`. It imports no quality benchmark, Pi SDK, or legacy evaluation runner. Baseline remains in ignored `core/config.py`; the model, engine, and frozen sampler settings come from that file.

## Run a bounded Search

After a fresh clone, install root `requirements.txt` in the repository venv and seed the local Baseline before importing this module, including `--help`:

```powershell
Copy-Item autoresearch/core/config.py.example autoresearch/core/config.py
.\venv\Scripts\python.exe -m autoresearch.tuning --help
```

On POSIX use `cp` and `./venv/bin/python`. Keep an existing `config.py`; the seed command is for a fresh clone only. Configure its model and engine before executing a Search.

The default requirements contain tuning/runtime and local model tools only. Legacy quality evaluation is an optional `requirements-legacy.txt` installation; developer tools are in `requirements-dev.txt`. Neither optional group is required for this bounded Search or its simulated E2E.

Review the portable [example plan](tuning-plan.example.json) and select the intended harness's minimum context, reachable engine values, and finite Trial budget. Context candidates must be at least the greater of the plan's minimum and the existing Baseline context. Plans cannot change the model, tune sampler settings, or optimize hardware-safety budgets/measurement policy.

```powershell
$env:AUTORESEARCH_LLAMA_CPP_ROOT = (Resolve-Path .\llama.cpp-releases\upstream\b11577).Path
.\venv\Scripts\python.exe -m autoresearch.tuning --plan autoresearch/tuning-plan.example.json
```

The example starts at 65536 tokens. If your Baseline targets a larger context, edit the plan to preserve that target. The command is a finite, sequential Neighbor Search; it never starts `autoloop.py`. `--db` selects the canonical SQLite file, defaulting to repo-root `results.db`. No model download, engine rebuild, automatic model selection, or cloud call occurs.

## Measurement protocol: tuning-runtime-v1

- **PPL:** fixed `data/perplexity_val.txt`, one 2048-token chunk, target model without speculative decoding. Placement/KV/load flags remain explicit and checked against the resolved perplexity binary's help. PPL is a quality guard, not a target-context throughput measurement.
- **TPS:** a monitored llama-server reserves the full candidate `CTX_SIZE` with one slot. `/props` confirms the allocation; after each native `/completion`, `/slots` must confirm the response's `id_slot` and full context. The fixed raw prompt generates 512 tokens, with EOS ignored, cache reuse disabled, and seed 1234. TPS is the median server-reported `predicted_per_second` across Baseline `TPS_REPS`, not total-token wall time or the legacy 4k CLI smoke.
- **Context evidence:** records full configured allocation and observed prompt occupancy separately. A short prompt at a large allocation does not prove long-context retrieval or quality; that belongs to quality Benchmark tasks.
- **PPL ceiling:** the first successful Baseline measurement is the fixed reference for this Search. Every point must have finite positive measurements and PPL no more than 1% above that reference. Accepted degradation cannot compound across Neighbors.
- **Selection:** retain non-dominated TPS × context alternatives. No single scalar winner and no cross-model competition. Exploration is bounded by the supplied Search Space and Trial budget; it is not proof of a global optimum. Equal TPS/context points do not expand the frontier.

RAM, CUDA-free, dedicated/Shared VRAM, single-load, native parent lifetime, DLL environment, and teardown use the same model-process implementation as the server. PPL completes before the full-context server starts. A watchdog kill is recorded separately from a model score. Missing/nonfinite timing, context drift, or incomplete generation produces no TPS value.

## Results and selection

`tuning_runs` stores protocol, plan, reference PPL, status, and the checkpointed frontier. `tuning_trials` stores every attempted configuration, outcomes and evidence. `accepted` means admitted by the measurement/PPL constraints; the final frontier is in `tuning_runs.frontier_json`, not every admitted row.

Each Trial writes JSON under ignored `tuning-artifacts/<run-id>/`, and each final frontier point gets a portable Fingerprint file beside that evidence. The command prints every alternative and its file path. Baseline, aliases, and the existing per-model launcher Fingerprint are not automatically replaced: select the alternative appropriate for your harness before applying it with the existing Fingerprint tooling.

Infrastructure/configuration failures stop the Search after checkpointing; unfinished runs remain `running`. No row is fabricated as zero quality or merged into legacy Claw/Coding rankings. The operator dashboard still displays legacy Trial rows; new tuning evidence is read from its SQLite tables/printed JSON. Legacy `autoloop.py` modes remain compatibility paths and retain their historical measurement semantics.

## Repeatable verification

```powershell
.\venv\Scripts\python.exe scripts/test_tuning_e2e.py
```

The E2E copies only the tuning/runtime source, a public Baseline template, and fixed PPL text into a fresh directory. No Benchmark or legacy runner directories are present, and an import guard rejects them. Real HTTP, SQLite, and guarded Python child processes exercise the pipeline with simulated model/hardware adapters. The printed artifact directory retains `results.db`, trajectories, native logs and `verification.json`. This verifies software behavior, not real GPU throughput.

For a future independent repository, retain `autoresearch/core/`, `autoresearch/tuning.py`, its package initializer, fixed PPL text, public configuration template, declared Python dependencies, and locally provisioned models/runtime. Pi and benchmark datasets are not required.
