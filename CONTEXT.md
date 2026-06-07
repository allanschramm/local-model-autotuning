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

**Val Score**:
The single scalar metric used for keep/discard decisions. When Coding is enabled: 80% Coding + 10% Nexus + 10% Claw. Without Coding: 60% Claw + 40% Nexus. Zeroed if TPS falls below the TPS Floor.
_Avoid_: score, result, metric

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
Speculative decoding using specialized draft heads to predict multiple tokens ahead, improving throughput.
_Avoid_: speculative decoding (when referring specifically to MTP)

