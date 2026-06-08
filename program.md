# AutoResearch Search Protocol

This is an autonomous hill-climbing system adapted to the local Nexus runtime, focusing on a unified Trial covering Retrieval, Agency, and Coding.

The contract is strict:

- `program.md` is fixed (unless explicitly requested by user).
- `autoresearch/benchmarks/*` harnesses are fixed.
- `autoresearch/core/config.py` is the **only** mutable tuning surface for generating Neighbors.

The goal is to push any model as far as possible on any hardware using optimized KV cache configurations and runtime parameters.

## Setup

To start a fresh Search:

1. **Agree on a run tag**: use a fresh branch tag such as `apr24`.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from `main`.
3. **Read the in-scope files**:
   - `program.md` — the rules for the Search (this file)
   - `autoresearch/core/config.py` — tuning surface for the Baseline and Neighbors
4. **Verify local assets exist**:
   - GGUF models in `models/`
   - `llama-server` (accessible via `autoresearch/core/llama_runner.py`)
5. **Initialize results.tsv**:
   - Ensure the header matches the unified benchmark runner output.

Once the setup is clean, begin the Search.

## Evaluation Suite

The runner executes a unified Trial and reports a single Val Score across domains:

### Nexus
Tests context-stress with synthetic history. The model must navigate the needle-in-a-haystack to find the override token, verify it, and unlock the control plane.

### Claw
Tests tool-use (JSON browser calls) and instruction-following using selected tasks.

### Coding
Uses EvalPlus to evaluate HumanEval+ and MBPP+.

### Val Score & Throughput
The system computes the `Val Score` based on a fixed ratio depending on whether Coding is included:
- `80% Coding + 10% Nexus + 10% Claw`
- Without Coding: `60% Claw + 40% Nexus`

A `Speed Factor` applies a soft penalty to Nexus and Claw scores based on throughput.
If a Trial falls below the **TPS Floor** (default 20.0 TPS), the `Val Score` is aggressively zeroed.

### Constraints
- **Hardware target:** Agnostic (optimize for your local GPU/VRAM).
- **VRAM Safety:** Monitor peak VRAM to ensure stability.
- **CPU Offload:** Partial offload is acceptable if throughput stays above the TPS Floor. Prefer full GPU (`--n-gpu-layers 999`) but trade speed for Val Score when needed.
- **Flash Attention:** Must always be `on` (`-fa on`).

## What you CAN do
- Modify `autoresearch/core/config.py` constants to generate a new Neighbor.
- Modify the `Search Space` for autonomous exploration in `autoloop.py` (only if requested).

## What you CANNOT do
- Modify the fixed evaluation logic in any `autoresearch/benchmarks/*` files.
- Add new dependencies.

## Output format
Each Trial logs exclusively to the canonical results file `results.tsv`. No other log files should be committed. The output format in `results.tsv` is tab-separated:
`commit\tval_score\tmemory_gb\tstatus\tdescription`

## The Search Process

### Autonomous mode (preferred)
Run `python autoloop.py` to start the SearchStrategy loop:
1. Reads current Baseline from `autoresearch/core/config.py`.
2. Runs all active benchmarks in a unified Trial.
3. Generates Neighbors by mutating a single parameter within the Search Space.
4. Evaluates each Neighbor via a new Trial. If improved (or won via Pareto Tie-Breaker) → writes new Baseline to `autoresearch/core/config.py`.
5. If at a Local Maxima → triggers a Random Restart.
6. Loops forever until `Ctrl+C` (SIGINT). State persists in `autoresearch/core/config.py` + `results.tsv`.

### Manual mode
1. Edit `autoresearch/core/config.py` with a hypothesis to test a new Neighbor.
2. Run: `python benchmark_search.py --desc "your hypothesis"`
3. Analyze `results.tsv`.
4. If improved, keep as the new Baseline. Otherwise revert `autoresearch/core/config.py`.

## Autonomy rule
Once the Search has started, continue autonomously until manually interrupted.
Do not pause to ask for permission to continue the Search.

## Terminology (Strict)
Use these exact terms in your reasoning and commit messages:
- **Search**: The overall optimization process.
- **Round**: One iteration of the Search.
- **Trial**: One complete execution of all benchmarks against a single configuration.
- **Baseline**: The current best-known configuration in `config.py`.
- **Neighbor**: A configuration derived from the Baseline by changing exactly one parameter.
- **Val Score**: The scalar metric used for keep/discard decisions.
- **Pareto Tie-Breaker**: Keep a Neighbor if it matches Baseline score but improves TPS by >5% or reduces VRAM by >5%.
- **Local Maxima**: When all valid Neighbors fail to improve the score.
- **Random Restart**: Generating a random configuration far from Baseline to escape Local Maxima.
