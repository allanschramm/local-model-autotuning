# autoresearch

This is a Karpathy-style autoresearch loop adapted to the local Nexus runtime, focusing on triple-pass evaluation (Retrieval, Agency, and Coding).

The contract is strict:

- `program.md` is fixed (unless explicitly requested by user).
- `prepare.py`, `prepare_claw.py`, and `benchmark_harness.py` are fixed.
- `benchmark_coding.py` and `run_grid.py` are the primary mutable tuning surfaces.

The goal is to push **Qwen 3.5 2B/4B/9B** and **Gemma 4B** as far as possible on a single RTX 4060 8GB using a 128k context window and optimized KV cache configurations.

## Setup

To start a fresh run:

1. **Agree on a run tag**: use a fresh branch tag such as `apr24`.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from `main`.
3. **Read the in-scope files**:
   - `program.md` — the rules for the experiment
   - `prepare.py` — Nexus Retrieval harness (Context Stress)
   - `prepare_claw.py` — ClawBench Agency harness (Tool-Use)
   - `benchmark_coding.py` — Coding harness (EvalPlus) and tuning surface
   - `run_grid.py` — Grid search orchestration script
4. **Verify local assets exist**:
   - GGUF models in `models/`
   - `llama-server` (accessible via `llama_runner.py`)
5. **Initialize grid_results.csv**:
   - Ensure the header matches the current `run_grid.py` output.

Once the setup is clean, begin the loop.

## Evaluation Suite

The runner executes a triple-pass evaluation harness and reports metrics for each domain:

### Pass 1: Nexus Retrieval (`prepare.py`)
Tests context stress with ~50 000 tokens of synthetic history. The model must navigate the needle-in-a-haystack to find the override token, verify it, and unlock the control plane.
- **Metric**: `nexus_val_score` (Accuracy weighted by TPS).

### Pass 2: ClawBench Agency (`prepare_claw.py`)
Tests tool-use (JSON browser calls) and instruction-following using selected dev-tech and office tasks.
- **Metric**: `claw_val_score` (Tool accuracy weighted by TPS).

### Pass 3: Coding Performance (`benchmark_coding.py`)
Uses EvalPlus to evaluate HumanEval+ and MBPP+.
- **Metric**: `val_score` (Average of HE+ and MBPP+ pass@1).

### Throughput & TPS Weighting
The `BenchmarkHarness` applies a speed factor to Nexus and Claw scores:
`speed_factor = 0.5 + 0.5 * min(1.0, current_tps / 30.0)`
Configurations falling significantly below **30 TPS** are aggressively penalized.

### Constraints
- **Hardware target:** RTX 4060 8GB.
- **Context Size:** 128k (131072) is mandatory.
- **VRAM Safety:** Keep `peak_vram_mb` below ~7900 MB.
- **No CPU Offload:** `--n-gpu-layers 999` is mandatory.

## What you CAN do
- Modify `benchmark_coding.py` constants (MODELS, CTX_SIZE, BATCH_SIZE, etc.).
- Modify `run_grid.py` search space (KV_CACHES, MAX_TOKENS_LIST).
- Change model selection and GGUF quantization levels.
- Change generation knobs (TEMP, TOP_P, MIN_P).

## What you CANNOT do
- Modify the fixed evaluation logic in `prepare.py`, `prepare_claw.py`, or `benchmark_harness.py`.
- Add new dependencies.
- Change the context requirement of 128k.

## Output format
Each run (especially via `run_grid.py`) logs to `grid_results.csv`. Successful individual benchmarks should print:

```text
---
val_score:        0.XXXX
nexus_tps:        XX.XX
claw_tps:         XX.XX
peak_vram_mb:     XXXX.X
model:            model_name.gguf
kv_cache:         q4_0
```

## The experiment loop
1. Inspect the current branch and configuration.
2. Edit `benchmark_coding.py` or `run_grid.py` with a new hypothesis (e.g., "Q4_1 KV cache improves accuracy without breaking VRAM").
3. Commit the change.
4. Run: `python run_grid.py` or `python benchmark_coding.py`
5. Analyze `grid_results.csv` or the console output.
6. If results improved or provided new insights, keep the commit. Otherwise, revert or iterate.

## Autonomy rule
Once the loop has started, continue autonomously until manually interrupted.
Do not pause to ask for permission to continue the search.
