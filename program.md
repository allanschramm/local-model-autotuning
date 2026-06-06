# autoresearch

This is a Karpathy-style autoresearch loop adapted to the local Nexus runtime, focusing on triple-pass evaluation (Retrieval, Agency, and Coding).

The contract is strict:

- `program.md` is fixed (unless explicitly requested by user).
- `prepare.py`, `prepare_claw.py`, and `benchmark_harness.py` are fixed.
- `benchmark_coding.py` and `run_grid.py` are the primary mutable tuning surfaces.

The goal is to push any model as far as possible on any hardware using optimized KV cache configurations and runtime parameters.

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
5. **Initialize results.tsv**:
   - Ensure the header matches the unified benchmark runner output.

Once the setup is clean, begin the loop.

## Evaluation Suite

The runner executes a triple-pass evaluation harness and reports metrics for each domain:

### Pass 1: Nexus Retrieval (`prepare.py`)
Tests context stress with synthetic history. The model must navigate the needle-in-a-haystack to find the override token, verify it, and unlock the control plane.
- **Metric**: `nexus_val_score` (Accuracy weighted by TPS).

### Pass 2: ClawBench Agency (`prepare_claw.py`)
Tests tool-use (JSON browser calls) and instruction-following using selected tasks.
- **Metric**: `claw_val_score` (Tool accuracy weighted by TPS).

### Pass 3: Coding Performance (`benchmark_coding.py`)
Uses EvalPlus to evaluate HumanEval+ and MBPP+.
- **Metric**: `val_score` (Average of HE+ and MBPP+ pass@1).

### Throughput & TPS Weighting
The `BenchmarkHarness` applies a speed factor to Nexus and Claw scores:
`speed_factor = 0.5 + 0.5 * min(1.0, current_tps / target_tps)`
Configurations falling significantly below the **target TPS** (default 20.0) are aggressively penalized.

### Constraints
- **Hardware target:** Agnostic (optimize for your local GPU/VRAM).
- **Context Size:** Flexible, defined by the specific experiment.
- **VRAM Safety:** Monitor peak VRAM to ensure stability.
- **CPU Offload:** Partial offload is acceptable if throughput stays above 20 TPS. Prefer full GPU (`--n-gpu-layers 999`) but trade speed for score when needed.

## What you CAN do
- Modify `benchmark_coding.py` constants (MODELS, CTX_SIZE, BATCH_SIZE, etc.).
- Modify `run_grid.py` search space (KV_CACHES, MAX_TOKENS_LIST).
- Change model selection and quantization levels.
- Change generation knobs (TEMP, TOP_P, MIN_P).

## What you CANNOT do
- Modify the fixed evaluation logic in `prepare.py`, `prepare_claw.py`, or `benchmark_harness.py`.
- Add new dependencies.

## Output format
Each run logs to the canonical results file `results.tsv`. Successful individual benchmarks should print:

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

### Autonomous mode (preferred)
Run `python autoloop.py` to start the autonomous hill-climbing loop:
1. Reads current baseline from `config.py`.
2. Runs ALL benchmarks (Nexus + Claw + optionally Coding).
3. Generates single-parameter perturbations (neighbors).
4. Evaluates each neighbor. If improved → writes new baseline to `config.py`.
5. If no improvement found → resets exploration and tries again.
6. Loops forever until `Ctrl+C` (SIGINT). State persists in `config.py` + `results.tsv`.

### Manual mode
1. Edit `config.py` with a hypothesis.
2. Run: `python benchmark_search.py --desc "your hypothesis"`
3. Analyze `results.tsv` or console output.
4. If improved, keep. Otherwise revert `config.py`.

## Autonomy rule
Once the loop has started, continue autonomously until manually interrupted.
Do not pause to ask for permission to continue the search.
