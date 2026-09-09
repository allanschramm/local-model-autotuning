# Agent Onboarding Guide

Bootstrap context for agents working on this repo.

## Codebase Map

| File/Path | Role |
|:---|:---|
| `autoloop.py` | Autonomous hill-climbing runner. Mutates configs and loops forever. |
| `autoresearch/core/config.py` | **Only mutable file.** Holds current baseline configs. |
| `autoresearch/core/search.py` | Hill-climbing engine (`SearchStrategy`). Evaluates improvement. |
| `autoresearch/core/llama_runner.py` | Wrapper around `llama-server`. Handles port collision and VRAM telemetry. |
| `autoresearch/benchmarks/benchmark_coding.py` | Evaluates coding capabilities via LCB, HE+, MBPP+, and BigCodeBench. |
| `results.tsv` | Tab-separated database recording trial history. |
| `scripts/rank_results.py` | Rank models from TSV (Pareto / Day / Night / claw / coding). **Use this for rankings — no temp scripts.** |
| `scripts/check_hardware.py` | Cross-OS hardware diagnose (Win/macOS/Linux). Local fit authority before downloads. |

## Local Rules

1. **Be Terse**: Respond in smart caveman style (drop articles, filler, pleasantries).
2. **Loop Rule**: If running `autoloop.py` and a crash or code error occurs, **stop immediately**. Do not edit code to fix bugs during active search unless explicitly requested.
3. **No Pushing**: Never push results or config tweaks to remote branches. Keep all benchmark runs offline.
4. **Sampler seed**: Before first Trial on a model, copy Recommended settings from `docs/models/<card>.md` into `SAMPLER_DEFAULTS` (agentic/general vs coding). Do not start from template `TEMP=0.4` when the card differs.
5. **DOX Framework**: Read the `AGENTS.md` hierarchy path to any file before touching it.
6. **Hardware before download (students / new clones)**: Run `scripts/check_hardware.py` first. Explain `discrete_gpu` vs `unified_memory` in plain language. Confirm RAM/VRAM/pool with the user. **Never** `hf download` or run `benchmark_search --validation` with a large GGUF before that. whichllm/llmfit are candidates only — reject oversized #1 on unified RAM (e.g. ~12 GB model on 16 GB Mac). If detection is incomplete, guide About This Mac / `sysctl`, Task Manager, or `nvidia-smi`.
7. **Runtime = prebuilt release first**: Do **not** build `llama.cpp` from source on a new clone (no VS/CUDA toolkit setup). Point the harness at a prebuilt release instead: extract a GitHub release zip under `llama.cpp-releases/<engine>/<tag>/build-cuda/bin/` (Windows CUDA asset `cudart-llama-bin-win-cuda-12.4-x64.zip` bundles its runtime — no toolkit needed) and set `AUTORESEARCH_LLAMA_CPP_ROOT=<repo>/llama.cpp-releases/<engine>/<tag>` (or pin per-alias `llama_cpp_root` in `models/aliases/<name>/config.yaml`). Keep engine/tag in Trial evidence. Build from source **only** when fixing something urgent that no release covers.

## Student / first-day checklist

1. Run `check_hardware.py` → read memory class + pool
2. Explain to the user (pt-BR if teaching) what unified vs dedicated means for *their* numbers
3. Confirm numbers
4. Optional: whichllm/llmfit for candidates — filter by detected pool
5. Only then download GGUF / `serve-config` / `verify_setup`
6. Do **not** jump to claw validation with an oversized model

## Operator dashboard (agent starts, human watches)

Read-only UI at `http://127.0.0.1:18765` (`ui/`). **Agent** starts it when the human wants to monitor a Trial/Search; **human** watches the browser. Non-goals: no process control, no agent stdout panel, no Baseline/TSV mutation from the UI. See [ui/AGENTS.md](../../ui/AGENTS.md).

```bash
.\venv\Scripts\python.exe -m ui
# ./venv/bin/python -m ui
```

## Essential Commands

```bash
# Hardware diagnose (Win / macOS / Linux) — before any model download
.\venv\Scripts\python.exe scripts\check_hardware.py
# ./venv/bin/python scripts/check_hardware.py

# Runtime (release first — no local build): extract GitHub release zip to
#   llama.cpp-releases/<engine>/<tag>/build-cuda/bin/
# then point the harness at it:
export AUTORESEARCH_LLAMA_CPP_ROOT="<repo>/llama.cpp-releases/<engine>/<tag>"
# Windows PowerShell: $env:AUTORESEARCH_LLAMA_CPP_ROOT="..."
# Verify: .\venv\Scripts\python.exe scripts\serve-config.py print-cmd

# Setup check
bash scripts/setup-check.sh

# Test suite
.\venv\Scripts\python.exe -m pytest tests/

# Default speed path (good enough, fast) — see docs/discovery/good-enough-tuning.md
.\venv\Scripts\python.exe benchmark_search.py --validation --desc "validate <model>"
.\venv\Scripts\python.exe autoloop.py --mode tps --vram-limit-mb=<budget>

# Champion quality check (after TPS is acceptable)
.\venv\Scripts\python.exe benchmark_search.py --agentic-full --desc "champion claw-full"

# Model ranking from the results store (ADR 0017: every complete model, IQ-first) — no ad-hoc filters
.\venv\Scripts\python.exe scripts\rank_results.py
.\venv\Scripts\python.exe scripts\rank_results.py --mode claw
.\venv\Scripts\python.exe scripts\rank_results.py --mode coding

# Leaderboard doc integrity (duplicates + doc/table parity; issue #69)
.\venv\Scripts\python.exe scripts\check_leaderboard_docs.py

# Single manual trial (Baseline already in config.py)
.\venv\Scripts\python.exe benchmark_search.py --desc "Hypothesis details here"
```

## Reading Order

1. `AGENTS.md` (root) — DOX hierarchy, work contracts
2. `program.md` — Search protocol rules
3. `GOLDEN-RULES.md` — Performance flags, safety, validation
4. `CONTEXT.md` — Terminology and definitions
5. `docs/discovery/good-enough-tuning.md` — **default** path: TPS first, quality second
6. `docs/discovery/discover-models.md` — Model selection workflow (which GGUF)
7. `docs/llamacpp-toolset.md` — llama.cpp binary reference
