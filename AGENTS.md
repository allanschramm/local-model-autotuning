# AutoResearch Agent Work Contract

## Testing Contract

- NEVER write unit tests after you write code.
- Highly prefer E2E tests as the sole testing mechanism. Use them to verify complex features work. At the end of E2E tests, produce a verifiable and repeatable artifact.
- If you must test a system in isolation, FIRST write all the ways it could fail, THEN write the code.

<!-- Scope: repo development agents. Research loop agents → read program.md -->

## Purpose
Local model auto-tuning: configured ctx × TPS under fixed PPL and minimum-context constraints. Quality evaluation and its document archive belong to the independent private [ailocal-benchmark](https://github.com/allanschramm/ailocal-benchmark), under [ADR 0020](docs/adr/0020-tuning-only-distribution.md) and [ADR 0021](docs/adr/0021-independent-repositories-and-documentation.md). Domain language: [CONTEXT.md](CONTEXT.md).

## Ownership
Repository developers.

## Local Contracts
- Teaching is owned by the independent ailocal-teach project; no local course subtree belongs here.
- Installation: requirements.txt contains runtime and local model tools; requirements-dev.txt adds developer checks. Quality evaluators and their installation profile are outside this distribution.
- Original AILOCAL contributions use PolyForm Noncommercial 1.0.0; see LICENSE and NOTICE.md. Imported materials retain their own licenses. Describe this as source-available, not OSI-approved open source.
- Tuning Baseline only via gitignored `autoresearch/core/config.py` (seed from `config.py.example`). No legacy Trial CLI flag soup. `program.md` / harnesses fixed unless user asks.
- **Upstream-first (mandatory):** Check upstream `llama.cpp/common/arg.cpp` `add_opt` (`~310` flags) + `docs/llamacpp-toolset.md` + `docs/llamacpp-flags-audit.md` before adding any `ENGINE_*/SAMPLER_*` flag or estimator — `90%` already exists upstream (`fit`/`yarn`/`dry`/`xtc`/`fit-target`/`warmup`/`cache-ram`/`kv-unified`/`repack` etc.), harness is passthrough mapper in `autoresearch/core/llama_runner.py` `_build_cmd`; keep only host/WDDM/thermal/Pareto gates (`common/fit.h:14` assumes unlimited host).
- **Flag drift is a build-time failure, not a Trial failure:** upstream removes and renames flags across releases (`--mmap`/`--no-mmap`/`--mlock` -> `--load-mode` in #26934), so every command the harness emits is checked against the resolved binary's own `--help` by `assert_flags_supported` (`llama_runner.py`), which raises `FlagDriftError` naming the flag and the `engine@tag` build. Recorded as `INFRA_ERROR` (engine/config problem), never `CODE_ERROR` (our logic). On any engine bump, run a one-Trial bounded validation plan and expect it to pass; if it raises, fix the mapping in `load_mode_flag` / the passthrough — never silence the guard, and never migrate Baseline/store keys for a flag rename.
- Do not edit vendor trees: `llama.cpp/`, `VITRIOL/`, `llama.cpp-releases/`.
- Never commit private paths, emails, hostnames, GPU SKUs, aliases, or machine Baseline.
- `models/` is a real store directory. NEVER recursive/forced delete (`Remove-Item -Recurse`, `rm -rf`, `rmdir /s`) on the `models/` root — it wipes the GGUF store and private notes. Per-file deletes in subdirs are fine.
- No command timeouts: Never set execution timeouts on commands run by the agent unless explicitly told. Benchmarks and model tests run until completion. This does not prohibit bounded network timeouts inside product code when needed to prevent a stalled service or request from blocking indefinitely.
- **Bounded execution only:** use python -m autoresearch.tuning --plan <plan.json> with a finite Trial budget; no autonomous background Search.
- **Results Store is Strictly SQLite (`results.db`):** `results.db` (SQLite) is the sole canonical store for all Trial, validation, and benchmark outcomes. `results.tsv` is deprecated and must not be queried, grepped, or relied upon by agents for metrics or history. Always query `results.db` directly using SQLite or `autoresearch.tuning`.
- **Context is Inviolable (Never reduce context):** Configured context (`CTX_SIZE`) is a core axis of the Pareto frontier (`ctx × TPS`). When encountering RAM pressure, VRAM spill, circuit breaker trips, OOM, or throughput bottlenecks, NEVER reduce or suggest reducing `CTX_SIZE` as a solution. Address resource pressure strictly through host memory hygiene (closing background desktop applications), KV cache quantization (`q4_0`), layer offloading (`N_CPU_MOE`), batch/ubatch tuning, or selecting lighter weight quantizations (`IQ4_XS`, `Q3_K_M`), keeping the target context intact.

## Work Guidance
- Commit meaningful, verified changes as atomic Conventional Commits (`feat`, `fix`, `refactor`, `docs`, etc.). Keep unrelated work out of each commit; push only when explicitly requested.
- **Runtime (do not rebuild)**: use the prebuilt CUDA 13.4 release `llama.cpp-releases/upstream/b11577` via `AUTORESEARCH_LLAMA_CPP_ROOT` (0.6.0-dev, build 11577). Text Trials for MiniCPM5-1B and MiniCPM-V-4.7-1B passed on 2026-10-12. b11577 includes upstream #30097, needed for qwen35/qwen35moe trunk-only GGUFs that declare `nextn_predict_layers` without MTP tensors; b11149 cannot load the MiniCPM-V-4.7-1B export. Keep older releases for comparison. The `llama.cpp/` source submodule is reference-only; do not rebuild its stale `build-cuda/` artifacts to fix runtime compatibility. Pin the release explicitly in harness sessions; verify the active environment path before each evaluation.
- **Windows binary execution**: When invoking `llama-server.exe` or `llama-cli.exe` directly on Windows (outside Python runners), always set the working directory to `build-cuda\bin` where companion DLLs (`llama.dll`, `ggml.dll`, `cudart64_13.dll`) reside. Invoking without `cwd` causes Windows loader failure `0xc0000135` and conhost pipe error `0x800700e8` (`ERROR_NO_DATA`).
- Method + Trial procedure: `CONTEXT.md`, `docs/adr/`, `docs/discovery/`, `autoresearch/AGENTS.md`, `program.md`.

## Testing
- Never write unit tests after you write code.
- Highly prefer E2E tests as the sole testing mechanism. Use them to verify complex features work. At the end of E2E tests, produce a verifiable and repeatable artifact.
- If you must test a system in isolation, first write down all the ways it could fail, then write the code.
- E2E here = a harness run that ends in a `results.db` artifact (`python -m autoresearch.tuning --plan <plan.json>`, tuning E2Es, `scripts/setup-check.sh`). The unit suite in `tests/` is a closed set of exceptions: silent store/rank corruption, security holes, platform branches the rig never executes, and fail-closed hardware gates (contract in `tests/AGENTS.md`).

## Verification
- Clean installation: `.\venv\Scripts\python.exe scripts/test_tuning_install_e2e.py` creates an isolated venv, checks runtime-only dependencies/readiness and retains tuning/dashboard SQLite evidence. It installs public Python packages without loading a model.
- Main merge protection requires `validate`, `Analyze (python)` and the `CodeQL` security result. Keep workflow names and required contexts synchronized.
- CI validates Ruff/pytest, isolated tuning and tuning consumer E2Es; it retains SQLite and JSON evidence on success or failure without model/API spend.
- scripts/test_tuning_consumers_e2e.py checks the tuning-only tree, dashboard and preservation of historical SQLite tables.
- `.\venv\Scripts\python.exe -m pytest` (or `scripts/run_validate.py` / pre-commit). Never system-global Python.
- Independent Auto tuning: `.\venv\Scripts\python.exe scripts/test_tuning_e2e.py` copies only runtime/tuning sources outside the repo and retains SQLite/Fingerprint evidence; no model/GPU load.
- Local Windows pytest does not execute POSIX branches (`fcntl`, Unix-only paths). After every push, wait for `.github/workflows/validate.yml` (`ubuntu-latest`) with `.\venv\Scripts\python.exe scripts\watch_validate.py` until it exits 0. A green local suite is not CI-complete.

# DOX framework

DOX is the repo’s AGENTS.md hierarchy. It is a **project contract**. Agents follow it on every edit.

## Core Contract

- AGENTS.md files are binding work contracts for their subtrees.
- Work products, source materials, instructions, records, assets, and durable docs must stay understandable from the nearest applicable AGENTS.md plus every parent AGENTS.md above it.

## Read Before Editing

1. Read the root AGENTS.md
2. Identify every file or folder you expect to touch
3. Walk from the repository root to each target path
4. Read every AGENTS.md found along each route
5. If a parent AGENTS.md lists a child AGENTS.md whose scope contains the path, read that child and continue from there
6. Use the nearest AGENTS.md as the local contract and parent docs for repo-wide rules
7. If docs conflict, the closer doc controls local work details, but no child doc may weaken DOX

Do not rely on memory. Re-read the applicable DOX chain in the current session before editing.

## Update After Editing

Every meaningful change requires a DOX pass before the task is done.

Update the closest owning AGENTS.md when a change affects:

- purpose, scope, ownership, or responsibilities
- durable structure, contracts, workflows, or operating rules
- required inputs, outputs, permissions, constraints, side effects, or artifacts
- AGENTS.md creation, deletion, move, rename, or index contents

Update parent docs when parent-level structure, ownership, workflow, or child index changes. Update child docs when parent changes alter local rules. Remove stale or contradictory text immediately. Small edits that do not change behavior or contracts may leave docs unchanged, but the DOX pass still must happen.

## Hierarchy

- Root AGENTS.md is the DOX rail: project-wide instructions, durable workflow rules, and the top-level Child DOX Index
- Child AGENTS.md files own domain-specific instructions and their own Child DOX Index
- Each parent explains what its direct children cover and what stays owned by the parent
- The closer a doc is to the work, the more specific and practical it must be

## Child Doc Shape

- Create a child AGENTS.md when a folder becomes a durable boundary with its own purpose, rules, responsibilities, workflow, materials, or quality standards
- Work Guidance must reflect the current standards of the project; if none yet, leave it empty
- Verification must reflect an existing check; if none yet, leave it empty until one exists

Default section order:

- Purpose
- Ownership
- Local Contracts
- Work Guidance
- Verification
- Child DOX Index

## Style

- Keep docs concise, current, and operational
- Document stable contracts, not diary entries
- Put broad rules in parent docs and concrete details in child docs
- Prefer direct bullets with explicit names
- Do not duplicate rules across many files unless each scope needs a local version
- Delete stale notes instead of explaining history
- Trim obvious statements, repeated rules, misplaced detail, and warnings for risks that no longer exist

## Closeout

1. Re-check changed paths against the DOX chain
2. Update nearest owning docs and any affected parents or children
3. Refresh every affected Child DOX Index
4. Remove stale or contradictory text
5. Run existing verification when relevant
6. Report any docs intentionally left unchanged and why

## Child DOX Index
- [CHANGELOG.md](CHANGELOG.md) — versioned release notes, migration and validation limits.
- [.agents/skills/onboarding/SKILL.md](.agents/skills/onboarding/SKILL.md) — bounded tuning setup.
- [LICENSE](LICENSE) · [NOTICE.md](NOTICE.md) — original-contribution license and third-party scope.
- [requirements.txt](requirements.txt) · [requirements-dev.txt](requirements-dev.txt) — runtime and development installation profiles.

- [autoresearch/AGENTS.md](autoresearch/AGENTS.md) — package (runtime and bounded tuning)
- [ui/AGENTS.md](ui/AGENTS.md) — operator dashboard
- [docs/AGENTS.md](docs/AGENTS.md) — docs tree (`models/`, `adr/`, `discovery/`, `sessions/`)
- [scripts/AGENTS.md](scripts/AGENTS.md) — operator scripts
- [tests/AGENTS.md](tests/AGENTS.md) — test suite
- [models/README.md](models/README.md) — GGUF layout
- [.agents/skills/validation/SKILL.md](.agents/skills/validation/SKILL.md) — model validation skill (download, metadata check, smoke eval; tracked carve-out)
- [.agents/skills/inference-research/SKILL.md](.agents/skills/inference-research/SKILL.md) — inference-performance research skill (flags, engines, quantization, spec decoding; engine-side only, no model search)
- [.pre-commit-config.yaml](.pre-commit-config.yaml) · [.github/workflows/validate.yml](.github/workflows/validate.yml) · [pyproject.toml](pyproject.toml)
- External read-only: `llama.cpp/` · `llama.cpp-releases/` · `VITRIOL/`
