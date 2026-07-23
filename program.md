# AutoResearch Search Protocol

This is an autonomous hill-climbing system adapted to the local runtime, focusing exclusively on Agentic Coding tasks.

The contract is strict:

- `program.md` is fixed (unless explicitly requested by user).
- `autoresearch/benchmarks/*` harnesses are fixed.
- `autoresearch/core/config.py` is the **only mutable Baseline** (ENGINE = performance, SAMPLER = quality). The Search loop and agents edit this file. It is **gitignored** — seed from `autoresearch/core/config.py.example`.
- `.autoresearch_state.json` stores **visited memory only** (not Baseline).

The goal is to push any model as far as possible on any hardware using optimized KV cache configurations and runtime parameters.

## Setup

To start a fresh Search:

1. **Agree on a run tag**: use a fresh branch tag such as `apr24`.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from `main`.
3. **Read the in-scope files**:
   - `program.md` — the rules for the Search (this file)
   - `autoresearch/core/config.py` — mutable Baseline (ENGINE + SAMPLER)
   - `.autoresearch_state.json` — local visited memory (created on first run)
4. **Verify local assets exist**:
   - GGUF models in `models/`
   - `llama-server` (accessible via `autoresearch/core/llama_runner.py`)
5. **Initialize results.tsv**:
   - Ensure the header matches the unified benchmark runner output.

Once the setup is clean, begin the Search.

## Evaluation Suite

The runner executes a unified Trial and reports a single Val Score based on agentic coding capability. Claw-Eval quick is smoke validation; Claw-Eval full is the canonical Search quality gate. Both use local rule-based grading without Docker, remote APIs, or an LLM judge.

### HumanEval+
Basic algorithmic reasoning and Python proficiency.

### MBPP+
Entry-level to medium complexity programming problems.

### LiveCodeBench v6
Contamination-resistant competitive programming tasks.

### BigCodeBench Hard
Library-call and API-heavy programming tasks.

### Val Score & Throughput
Claw-Eval full supplies the canonical `Val Score`. HumanEval+, MBPP+, LiveCodeBench, and BigCodeBench Hard are optional preflight checks; when enabled, each runs exactly 10 tasks and does not replace the agentic Val Score.

If a Trial falls below the **TPS Floor** (Baseline `TPS_FLOOR`, default 20.0 TPS), the `Val Score` is aggressively zeroed to ensure runtime viability for interactive agent usage. Lower it for large MoE on constrained VRAM when measured speed is still usable.

### Constraints
- **Hardware target:** Agnostic (optimize for your local GPU/VRAM).
- **VRAM Safety:** Monitor peak VRAM to ensure stability.
- **CPU Offload:** Partial offload is acceptable if throughput stays above the TPS Floor. Prefer full GPU (`--n-gpu-layers 999`) but trade speed for Val Score when needed.
- **Flash Attention:** Must always be `on` (`-fa on`).

## What you CAN do
- Mutate the Baseline in `autoresearch/core/config.py` (via `autoloop.py` / `write_baseline`, or by editing ENGINE_/SAMPLER_ defaults) to generate a new Neighbor or seed a hypothesis.
- Modify the `Search Space` for autonomous exploration in `autoloop.py` (only if requested).

## What you CANNOT do
- Modify the fixed evaluation logic in any `autoresearch/benchmarks/*` files.
- Add new dependencies.
- Rewrite `program.md` or harness code from the Search loop.
- Drive Trials via raw `llama-server` / CLI flag soup — change `config.py`, then run the harness.

## Output format
Each Trial logs exclusively to the canonical results file `results.tsv`. No other log files should be committed. The output format in `results.tsv` is tab-separated:
`commit\tmodel\tval_score\tmemory_gb\telapsed_sec\tstatus\tcategory\tdescription`

## The Search Process

### Autonomous mode (preferred)
Run `python autoloop.py` to start the SearchStrategy loop:
1. Reads current Baseline from `autoresearch/core/config.py`.
2. Runs all active benchmarks in a unified Trial (Claw-Eval full = Val Score; quick = smoke).
3. Generates Neighbors by mutating a single parameter within the Search Space.
4. Evaluates each Neighbor via a new Trial. If improved (or won via Pareto Tie-Breaker) → writes new Baseline to `config.py`.
5. If at a Local Maxima → triggers a Random Restart.
6. Loops forever until `Ctrl+C` (SIGINT). Baseline persists in `config.py`; visited memory in `.autoresearch_state.json`; results in `results.tsv`.

### Manual mode
1. Edit `autoresearch/core/config.py` Baseline with a hypothesis.
2. Run: `python benchmark_search.py --desc "your hypothesis"` (or `python -m autoresearch.runners.run --validation --desc "..."`).
3. Analyze `results.tsv`.
4. If improved, keep the `config.py` edit. Otherwise revert `config.py`.

## Autonomy rule
Once the Search has started, continue autonomously until manually interrupted.
Do not pause to ask for permission to continue the Search.
The looping agent is strictly forbidden from editing codebase source code (e.g., `run.py`, benchmarks, tests) under any circumstances — except writing the Baseline via `config.write_baseline` / `config.py`. If any error, bug, or exception occurs in the code, the agent MUST NOT attempt to fix or edit the code. It MUST immediately stop execution, report the error, and warn the user.
Additionally, the looping agent must only save results and tweak commits locally. It is strictly forbidden from pushing any commits, branches, or results (e.g., benchmark scores, config tweaks) to the remote repository. All benchmark runs and results branches must remain completely offline and local-only.

## Terminology (Strict)
Use these exact terms in your reasoning and commit messages:
- **Search**: The overall optimization process.
- **Round**: One iteration of the Search.
- **Trial**: One complete execution of all benchmarks against a single configuration.
- **Baseline**: The current best-known configuration in `autoresearch/core/config.py`.
- **Neighbor**: A configuration derived from the Baseline by changing exactly one parameter.
- **Val Score**: The scalar metric used for keep/discard decisions.
- **Pareto Tie-Breaker**: Keep a Neighbor if it matches Baseline score but improves TPS by >5% or reduces VRAM by >5%.
- **Local Maxima**: When all valid Neighbors fail to improve the score.
- **Random Restart**: Generating a random configuration far from Baseline to escape Local Maxima.

## Model Acquisition & Troubleshooting

### Supported Formats & Models
- **GGUF** via `llama-server`; **directory models** under `models/` via SGLang.
- **Supported Architectures**: Llama (Llama-3/3.1/3.2), Qwen (Qwen-3.5/3.6/3.7), Gemma (Gemma-2/4), Mistral/Mixtral, and derivatives.
- **Recommended Models**: `Qwen3.5-9B-Coder-MTP-Q4_K_M.gguf`, `gemma-4-e2b-it-Q4_K_M.gguf`, or similar.

### How to Get Models
You can download models from HuggingFace. Place them in the `models/` directory.
- **Via HuggingFace CLI**:
  ```bash
  hf download Qwen/Qwen2.5-Coder-7B-Instruct-GGUF qwen2.5-coder-7b-instruct-q4_k_m.gguf --local-dir models/local/qwen2.5-coder-7b-instruct-gguf
  ```
- **Via Agent**: Ask your coding assistant to download a specific GGUF model into the `models/` directory for you.

### Troubleshooting (Wrong Model / Format)
- **Non-GGUF file**: If you download a PyTorch/safetensors weight file (e.g., `.bin`, `.safetensors`), `llama-server` will fail to parse it, log a `FAIL` status in `results.tsv`, and skip the configuration without breaking the loop.
- **Unsupported architecture**: If the GGUF uses a brand new, unsupported neural architecture, `llama-server` will exit during startup, which is caught safely as a trial failure.
- **Corrupt GGUF**: If the model file is corrupt/truncated, the server will fail to load it, log `FAIL` to `results.tsv`, and gracefully continue.
