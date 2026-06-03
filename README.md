# Autoresearch

Autoresearch is a Karpathy-style self-play loop for benchmarking local LLMs on a single consumer GPU.

The core idea: keep the evaluation harness fixed, leave only the search surface editable, and iterate fast.

<br>
## What this repo contains

- Retrieval benchmark (`prepare.py`, `benchmark_search.py`)
- Agency benchmark (`prepare_claw.py`, `benchmark_search_claw.py`)
- Coding benchmark via EvalPlus (`benchmark_coding.py`, `evalplus_wrapper.py`)
- Grid runner (`run_grid.py`)
- Reproducible memory fixture (`data/memory_fixture.json`)

## Hardware target

- RTX 4060 8GB
- Primary harness: local `llama.cpp` server
- Context size: 128k tokens

## Requirements

- Python 3.11+
- `uv` or `pip`
- `llama.cpp` server binary (`llama-server`)
- NVIDIA GPU + drivers

## Setup

1. Clone this repo.
2. Add your GGUF models under `models/`.
3. Point `AUTORESEARCH_LLAMA_CPP_ROOT` to your `llama.cpp` checkout containing `build/bin/llama-server` or `build-cuda/bin/llama-server`.

Example:
```bash
export AUTORESEARCH_MODELS_DIR="$PWD/models"
export AUTORESEARCH_LLAMA_CPP_ROOT="$HOME/repos/llama.cpp"

python benchmark_search.py --model gemma-4-E4B-it-Q4_K_M.gguf
```

## The rules

- `benchmark_search.py`, `benchmark_search_claw.py`, `benchmark_coding.py` are the mutable search surface.
- `prepare.py`, `prepare_claw.py`, `program.md` define the fixed contract.
- Do not add dependencies that require network access during benchmark execution.

## Output shape

Primary metric:
- `val_score`: composite of retrieval + agency + throughput floor (30 TPS)

Secondary metrics:
- `val_retrieval`
- `val_agency`
- `tokens_per_sec`
- `peak_vram_mb`
- `ctx_size`
- `kv_cache`
- `model`

## Notes

- This repo is the public release of the autoresearch loop.
- Large artifacts, logs, and model binaries are intentionally excluded.
