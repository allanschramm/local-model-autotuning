# AutoResearch

Autonomous hill-climbing system that optimizes local LLM runtime flags by repeatedly benchmarking configurations and keeping improvements.

## Language

### Search Process

**Search**:
The overall optimization process. An indefinite sequence of Rounds that continues until manually stopped.
_Avoid_: loop, sweep, experiment

**Round**:
One iteration of the Search: evaluate the current baseline, then evaluate neighbor configurations until one improves or all are exhausted.
_Avoid_: step, iteration, cycle

**Trial**:
One complete execution of all benchmarks against a single configuration. The atomic unit of work. Produces a score, TPS, and VRAM measurement.
_Avoid_: run, evaluation, pass, execution

**Local Maxima**:
A state where all valid Neighbors from the current Baseline have been evaluated and none improved the score.
_Avoid_: stuck state, convergence

**Pareto Tie-Breaker**:
The logic used to break exact ties in Val Score (diff < 0.0001). A Neighbor is kept if it matches the Baseline's score but improves TPS by >5%, or matches both score and TPS but reduces VRAM by >5%.
_Avoid_: secondary objective, performance score

**Random Restart**:
The mechanism used to escape a Local Maxima. Generates a random configuration far from the current Baseline that isn't in the visited memory, sets it as the new Baseline, and resumes the Search.
_Avoid_: random jump, memory wipe

**SearchStrategy**:
A deep module encapsulating the rules of hill-climbing optimization. It unifies Neighbor generation, Pareto Tie-Breaker logic, and Random Restarts across different search spaces.
_Avoid_: heuristic loop, search script

### Configuration

**Baseline**:
The current best-known configuration. Persisted in `autoresearch/core/config.py`. A Trial must strictly beat the Baseline score to replace it.
_Avoid_: default, current config

**Neighbor**:
A configuration derived from the Baseline by changing exactly one parameter. The Search evaluates Neighbors to find improvements.
_Avoid_: candidate, variant, mutation

**Search Space**:
The set of parameters and their candidate values that the Search explores. Defines which Neighbors are reachable from any Baseline.
_Avoid_: grid, parameter space

### Evaluation

**Validation**:
The 2-step pre-check that runs before every full Trial: (1) `llama-bench` speed test (tg_tps >= threshold), (2) quick coding eval with 2 tasks per dataset (HE+, MBPP+, LCB, BigCodeBench — 8 tasks total). If either step fails, the Trial is logged as FAIL and no extended evaluation runs. The `--validation` flag runs this check alone and exits.

**To validate a single model**: (1) set `MODEL` in `autoresearch/core/config.py`, (2) run `python3 benchmark_search.py --validation --desc "validate <model>"`. One model at a time — never parallel. See GOLDEN-RULES.md §5 for the full step-by-step.
_Avoid_: bench-only, speed check, smoke test

**Val Score**:
The single scalar metric used for keep/discard decisions. When Coding is enabled: **80% Coding + 10% Nexus + 10% Claw**. Without Coding: **60% Nexus + 40% Claw**. Zeroed if TPS falls below the TPS Floor.
_Avoid_: score, result, metric

> **Weighting rule** (consistent with `run.py` implementation):
> - `INCLUDE_CODING=True` → `0.80 × coding + 0.10 × nexus + 0.10 × claw`
> - `INCLUDE_CODING=False` → `0.60 × nexus + 0.40 × claw`
>
> Earlier docs claimed "60% Claw + 40% Nexus" without Coding; this was inconsistent with the implementation. The weights above are the authoritative reference.

**TPS Floor**:
The minimum throughput (tokens per second) a Trial must achieve. Below this, Val Score is forced to zero regardless of accuracy.
_Avoid_: threshold, minimum TPS

**Speed Factor**:
A soft penalty multiplier applied to accuracy scores based on throughput. Ranges from 0.5 (at 0 TPS) to 1.0 (at or above target TPS).
_Avoid_: TPS penalty, speed penalty

### Benchmarks

**Nexus**:
Retrieval benchmark. Tests context-stress with synthetic history — the model must find a needle in a haystack of padding.
_Avoid_: retrieval, context stress

**Claw**:
Agency benchmark. Tests tool-use (JSON browser calls) and instruction-following.
_Avoid_: agency, ClawBench, tool-use benchmark

**Coding**:
Optional benchmark using EvalPlus (HumanEval+ and MBPP+). Measures code generation accuracy.
_Avoid_: EvalPlus, HumanEval

### Runtime

**ServerIntent**:
A pure data object describing the full configuration for a Trial — model path, context size, KV cache types, threads, speculative draft tokens, etc.
_Avoid_: config object, server config

**TurboQuant**:
Hardware-accelerated KV cache compression formats (`turbo2`, `turbo3`, `turbo4`) that fit large contexts within tight VRAM budgets.
_Avoid_: quantized cache, compressed KV

**Multi-Token Prediction (MTP)**:
Speculative decoding using specialized draft heads (built into the model) to predict multiple tokens ahead, improving throughput. Distinct from "speculative decoding with separate draft model", which fails on MoE+SSM models.
_Avoid_: speculative decoding (when referring specifically to MTP)

### Generic Configuration Skeleton

The example below uses portable placeholders. Replace with your actual paths/values.

**Model file**: place in `models/` (relative to repo root), e.g. `models/<model-filename>.gguf`. Symlinks to absolute paths under your home directory are also OK.

**Working llama-server command template**:
```
llama-server \
  -m models/<model-filename>.gguf \
  --host 0.0.0.0 --port 8083 \
  -ngl 999 \
  --n-cpu-moe 32 \
  --cache-type-k q4_0 --cache-type-v q4_0 \
  -c 8192 \
  --override-tensor "v\\..*=CPU" \
  --flash-attn on --no-warmup
```

**MTP flags (only when the GGUF was downloaded from the `-MTP-GGUF` repo variant)**:
```
--spec-type mtp --spec-draft-n-max 2
```
Notes:
- Turboquant and similar forks accept `--spec-type mtp` (NOT `draft-mtp`).
- Upstream `ggml-org/llama.cpp` accepts `--spec-type draft-mtp`.
- `scripts/setup-check.sh` probes your build's `--help` and validates compatibility.
- MTP adds ~1 GB VRAM headroom. Speedup is **1.15–1.25× for MoE**, **1.4–2.0× for dense**.

**Key flags explained**:
- `-ngl 999`: lets auto-fit adjust GPU layers. Avoid combining with explicit `--n-cpu-moe` smaller than auto-fit targets.
- `--n-cpu-moe 32`: first 32 layers' MoE experts on CPU, remaining layers' experts on GPU. Adjust based on your MoE layer count.
- `--override-tensor "v\\..*=CPU"`: value projection weights forced to CPU (saves ~500MB VRAM on multimodal GGUFs).
- `--no-warmup`: required to avoid OOM during empty-run warmup on tight VRAM (e.g., 8GB).
- `--parallel 1`: reduces RS buffer (recurrent state for delta net) significantly on hybrid architectures.

**Performance expectations** (illustrative — depends on hardware + flags):
- q4_0 KV at 8k ctx on RTX 4060 8GB with MoE offload: ~11 tok/s (no MTP), ~13 tok/s (with MTP).
- TurboQuant does not always help — on GQA 8:1 architectures, `turbo4` K cache auto-upgrades to `q8_0` with no speed/VRAM gain over plain `q4_0`. Run `whichllm plan` to inspect your specific model.

**Filesystem caveats**:
- Models on 9p bridges (e.g. `/mnt/c/...`, `/mnt/d/...`) load very slowly via `mmap` (10–50× slower than native ext4). Copy or symlink model files into `models/` (native ext4) for normal speed.
- For WSL2: ensure `vm.overcommit_memory=1` and ample WSL `.wslconfig` memory (≥24 GB) when serving 20+ GB models.

## Discovery Workflow (cross-reference)

For users selecting which model to autotune, see [`docs/discovery/discover-models.md`](discovery/discover-models.md). It documents the **whichllm → Pareto frontier → autoloop handoff** flow that complements the autoloop once a Baseline model has been picked.

## Cached lessons (general, not user-specific)

- **MoE offload is mandatory on 8GB VRAM**: without explicit `--n-cpu-moe`, auto-fit can put 36/42 layers with GATE overflow, dropping throughput to ~0.7 tok/s. Always pair `--n-cpu-moe` with explicit `--override-tensor`.
- **Speculative decoding with separate draft models fails on MoE+SSM**: verification becomes PCIe-bound (MoE expert fetch per token) and SSM layers can't parallelize across a draft window. MTP is a different mechanism and works.
- **`whichllm` score ≠ coding benchmark**: whichllm blends AA Intelligence Index, Aider, LiveBench (intelligence weighted). For Claude Code / Pi Agent loops, cross-reference SWE-bench Verified — Gemma-4-26B-A4B ranks top in whichllm but scores only ~17% on SWE-bench Verified (bad coding agent despite high general intelligence).
- **Pareto frontier beats "highest score"**: for coding agents with high mistake cost, pick the Pareto-optimal tok/s × SWE-bench point, not the highest single-axis leader.
