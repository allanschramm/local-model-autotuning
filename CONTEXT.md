# AutoResearch

Autonomous multi-objective Search that benchmarks local LLM runtime configurations and maintains a Pareto frontier of non-dominated Trials (context × throughput × agentic × coding) for a given hardware budget.

## Language

### Search Process

**Search**:
The overall optimization process. An indefinite sequence of Rounds that continues until manually stopped.
_Avoid_: loop, sweep, experiment

**Round**:
One iteration of the Search: evaluate Neighbors of the active Baseline until the per-model frontier stops improving or Neighbors are exhausted.
_Avoid_: step, iteration, cycle

**Trial**:
One execution of chosen benchmarks against a single Fingerprint. The atomic unit of work. May be partial (subset of axes) or complete (all four Objective Vector axes measured).
_Avoid_: run, evaluation, pass, execution

**Objective Vector**:
The four maximize axes of a Trial: configured context (`CTX_SIZE`), TPS, agentic (Claw-Eval full), coding (coding-10). Domination compares Objective Vectors.
_Avoid_: Val Score, blended intelligence, single score

**Pareto Set**:
Search-internal concept only: the non-dominated Objective Vectors within ONE model's config comparisons (hill-climb). The global model rank is a plain leaderboard — every complete model, quality-first — not a frontier ([ADR 0017](docs/adr/0017-rank-membership-quality-first.md)).
_Avoid_: leaderboard, keep list

**Domination**:
Same-model config verdict: Trial A dominates B when A is ≥ on every Objective Vector axis and > on at least one. Never applies across different models ([ADR 0017](docs/adr/0017-rank-membership-quality-first.md)). Dominated Trials are not failures.
_Avoid_: cross-model ranking, discard, reject, worse score

**Fingerprint**:
Identity of a configuration for merge and frontier membership: the full `ENGINE_DEFAULTS` + `SAMPLER_DEFAULTS` used for the Trial (model, ctx, KV, batch/threads, MTP/spec, offload, sampler, …).
_Avoid_: model-only key, engine-only key

**Usage Profile**:
A selection lens over the model leaderboard, not a separate frontier. **Day** (supervised): every complete model, IQ-first (`min(agentic, coding)` descending); near-ties (±0.05 band) → higher TPS, then ctx. **Night** (unsupervised long loops): same membership; near-ties → larger ctx, then TPS. Historical lens notes: the old `TPS ≥ DAY_TPS_FLOOR` (ADR 0009) and `CTX_SIZE ≥ NIGHT_CTX_FLOOR` (ADR 0013) admission filters no longer gate anything ([ADR 0017](docs/adr/0017-rank-membership-quality-first.md)). Student-facing **usage** framing (no selection math) lives in [`teach/GLOSSARY.md`](teach/GLOSSARY.md) and [`teach/SPEC.md`](teach/SPEC.md).
_Avoid_: separate day/night frontiers, floors as admission filters, model exclusion by another model

**DAY_TPS_FLOOR**:
Historical lens note (default was 50.0 TPS; [ADR 0009](docs/adr/0009-day-profile-tps-floor.md)). Demoted by [ADR 0017](docs/adr/0017-rank-membership-quality-first.md): no longer filters Day table membership; speed only breaks ±0.05 near-ties in the Day table.
_Avoid_: using as Day rule, Day admission filter

**DAY_IQ_RATIO**:
Legacy Day gate ratio from ADR 0008. Superseded by `DAY_TPS_FLOOR` ([ADR 0009](docs/adr/0009-day-profile-tps-floor.md)); removed from `scripts/rank_results.py` CLI.
_Avoid_: using as Day rule

**NIGHT_CTX_FLOOR**:
Historical lens note (default was 65536). Demoted by [ADR 0017](docs/adr/0017-rank-membership-quality-first.md): no longer filters Night table membership; ctx only breaks ±0.05 near-ties in the Night table.
_Avoid_: ctx axis, hard reject below floor, Night admission filter

**Local Maxima**:
A state where all valid Neighbors from the active Baseline have been evaluated and none join or improve the per-model Pareto Set.
_Avoid_: stuck state, convergence

**Random Restart**:
The mechanism used to escape a Local Maxima. Generates a random configuration far from the active Baseline that isn't in visited memory, sets it as the new Baseline, and resumes the Search.
_Avoid_: random jump, memory wipe

**SearchStrategy**:
A deep module encapsulating Neighbor generation, Pareto Set updates, Usage Profile Baseline pick, and Random Restarts across Search Spaces.
_Avoid_: heuristic loop, search script, Pareto Tie-Breaker

### Configuration

**Baseline**:
The active Neighbor origin for Search — a Fingerprint persisted in `autoresearch/core/config.py` (`ENGINE_DEFAULTS` / `SAMPLER_DEFAULTS`). Chosen by Usage Profile (or manual override) from the per-model front; not “the single best config”. `.autoresearch_state.json` holds visited memory only.
_Avoid_: sole champion, default, current config, state baseline

**Neighbor**:
A configuration derived from the Baseline by changing exactly one parameter. The Search evaluates Neighbors to grow or improve the per-model Pareto Set.
_Avoid_: candidate, variant, mutation

**Search Space**:
The set of parameters and their candidate values that the Search explores. Defines which Neighbors are reachable from any Baseline. Neighbors do not jump across models.
_Avoid_: grid, parameter space

### Evaluation

**Validation**:
The pre-check before an expensive Trial: (1) local backend throughput validation, then (2) Claw-Eval quick. Optional direct-coding preflight always uses exactly 10 tasks per dataset. The `--validation` flag runs throughput plus Claw-Eval quick and exits.

**To validate a single model**: (1) set `MODEL` (and other flags) in `config.py` Baseline, (2) run `python3 benchmark_search.py --validation --desc "validate <model>"` with no CLI flag soup. One model at a time — never parallel. See GOLDEN-RULES.md §5 for the full step-by-step.
_Avoid_: bench-only, speed check, smoke test

**Trial Status**:
Canonical outcome labels: `on_front` (complete vector; store-wide recompute labels every complete basename vector on_front — no model is demoted for another model), `dominated` (same-model config A/B verdict from hill-climb bookkeeping; never a cross-model label, [ADR 0017](docs/adr/0017-rank-membership-quality-first.md)), `incomplete` (missing axes; may merge into a Fingerprint), `rejected` (invalid config, infra/VRAM kill, crash). 
_Avoid_: keep, discard (deleted; not accepted on write), cross-model dominated

**Status de exibição (dashboard)**:
Display convention of Trial Status on the pt-BR operator dashboard: the canonical English labels render localized — `on_front` → "na fronteira", `dominated` → "dominado", `incomplete` → "incompleto", `rejected` → "rejeitado". The canonical labels stay in the data/API; only the read-only UI translates (ADR 0011).
_Avoid_: showing raw English canonical labels in pt-BR panels

**Val Score**:
Legacy scalar (historically Claw-Eval full) retained for display/compat only. Not the Search keep rule. Prefer the Objective Vector.
_Avoid_: score, result, metric (when meaning frontier truth)

**TPS Floor**:
Legacy minimum throughput knob in Baseline `ENGINE_DEFAULTS['TPS_FLOOR']`. Does **not** gate Pareto Set membership or leaderboard membership. The old Day "TPS floor then max IQ" pick ([ADR 0009](docs/adr/0009-day-profile-tps-floor.md)) is a historical lens demoted by [ADR 0017](docs/adr/0017-rank-membership-quality-first.md) — TPS now only breaks ±0.05 near-ties in the Day table. Removable once callers stop depending on it.
_Avoid_: threshold as frontier rule, minimum TPS for on_front, Day = TPS Floor

**TPS median**:
Backend bench gate (`llama-cli` / SGLang) reports the median of `TPS_REPS` generations after GPU cooldown. Floor check uses that median. Coding-generation TPS is unchanged ([ADR 0016](docs/adr/0016-measurement-hygiene-and-morris-screen.md)).
_Avoid_: single-shot bench as Search truth

**gpu_temp_c**:
GPU sensor temperature (°C) recorded at bench/server start. Distinct from sampler `temp` / `TEMP`.
_Avoid_: using TSV `temp` for GPU heat

**Morris Screen**:
Pre-Round elementary-effects probe over **engine** Search Space knobs using llama-cli TPS as y. Low-effect knobs are pinned to the best measured level so Neighbor Search does not step them. Not a replacement for Neighbors. Off in `--mode quality` and `--no-screen` ([ADR 0016](docs/adr/0016-measurement-hygiene-and-morris-screen.md)).
_Avoid_: Taguchi, orthogonal array, replacing Neighbor

**Crash Journal**:
Gitignored `.autoresearch_crash.journal` written immediately before an autoloop Trial and cleared when it returns. If the process dies, the next autoloop start records that Fingerprint as `rejected` / `CRASH` unless `--retry-crashed` ([ADR 0016](docs/adr/0016-measurement-hygiene-and-morris-screen.md)).
_Avoid_: storing crashes as a second scores file, TSV as journal

### Benchmarks

**Nexus**:
Retrieval benchmark. Tests context-stress with synthetic history — the model must find a needle in a haystack of padding.
_Avoid_: retrieval, context stress

**Claw**:
Agency benchmark. Tests tool-use (JSON browser calls) and instruction-following. Claw-Eval full supplies the **agentic** Objective Vector axis; quick remains observational smoke.
_Avoid_: agency, ClawBench, tool-use benchmark

**Coding**:
Benchmark using LiveCodeBench v6, HumanEval+, MBPP+, and BigCodeBench Hard (exactly 10 tasks per dataset when enabled). Supplies the **coding** Objective Vector axis.
_Avoid_: EvalPlus, HumanEval

**Agentic coding**:
SWE-lite workspace loop over frozen GitHub-issue fixtures (`--agentic-coding`). Night selector ([ADR 0013](docs/adr/0013-agentic-coding-night-selector.md)); not a Pareto axis in v1. Pass = hidden tests green and no repeat/stall/hallucination flag.
_Avoid_: live `gh` issues as Trial tasks, SWE-bench vendor dump

### Runtime

**ServerIntent**:
A pure data object describing the full configuration for a Trial — model path, context size, KV cache types, threads, speculative draft tokens, etc.
_Avoid_: config object, server config

**Process Guard**:
A cross-platform process lifecycle module (`autoresearch/core/process_guard.py`) that binds spawned subprocesses (`llama-server`, `llama-cli`, `llama-bench`, `llama-perplexity`, `sglang`, mock services) to OS-native parent lifecycle handles (Windows Job Objects with `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`, Linux `PR_SET_PDEATHSIG`, POSIX process groups) and executes pre-flight port/process cleanup to guarantee zero zombie processes on user machines.
_Avoid_: daemon manager, watchdog script, background worker

**SGLang Backend**:
Directory model paths under `models/` are served through SGLang instead of `llama-server`. SGLang Trials still flow through the harness, run the same Coding benchmark, and use the configured CTX_SIZE.
_Avoid_: raw SGLang run, direct server launch

**TurboQuant**:
Hardware-accelerated KV cache compression formats (`turbo2`, `turbo3`, `turbo4`) that fit large contexts within tight VRAM budgets.
_Avoid_: quantized cache, compressed KV

**Multi-Token Prediction (MTP)**:
Speculative decoding using specialized draft heads (built into the model) to predict multiple tokens ahead, improving throughput. Distinct from "speculative decoding with separate draft model", which fails on MoE+SSM models. MTP is a Search dial (buy ctx or TPS), not a required default.
_Avoid_: speculative decoding (when referring specifically to MTP)

### Monitoring

**Run State**:
The live operational status of the dashboard: `Em execução` when the Trial server log (`autoresearch/runners/llama_server.log`) grew within the last ~10s, otherwise `Idle`. Missing log → `Idle`, never a crash. Primary visual badge of the dashboard (ADR 0011).
_Avoid_: server status, process state, run status

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
- q4_0 KV at 8k ctx on discrete 8 GB-class NVIDIA with MoE offload: ~11 tok/s (no MTP), ~13 tok/s (with MTP).
- TurboQuant does not always help — on GQA 8:1 architectures, `turbo4` K cache auto-upgrades to `q8_0` with no speed/VRAM gain over plain `q4_0`. Run `whichllm plan` to inspect your specific model.

**Filesystem caveats**:
- Models on 9p bridges (e.g. `/mnt/c/...`, `/mnt/d/...`) load very slowly via `mmap` (10–50× slower than native ext4). Copy or symlink model files into `models/` (native ext4) for normal speed.
- For WSL2: ensure `vm.overcommit_memory=1` and ample WSL `.wslconfig` memory (≥24 GB) when serving 20+ GB models.

## Discovery Workflow (cross-reference)

For users selecting which model to autotune, see [`docs/discovery/discover-models.md`](docs/discovery/discover-models.md). It documents the **whichllm → leaderboard → Baseline handoff** flow. Rank membership rules: [ADR 0017](docs/adr/0017-rank-membership-quality-first.md) (Day/Night pick = every complete model, quality-first, ±0.05 near-tie band; see also [pareto-selection.md](docs/discovery/pareto-selection.md)).

## Cached lessons (general, not user-specific)

- **MoE offload is mandatory on 8GB VRAM**: without explicit `--n-cpu-moe`, auto-fit can put 36/42 layers with GATE overflow, dropping throughput to ~0.7 tok/s. Always pair `--n-cpu-moe` with explicit `--override-tensor`.
- **Speculative decoding with separate draft models fails on MoE+SSM**: verification becomes PCIe-bound (MoE expert fetch per token) and SSM layers can't parallelize across a draft window. MTP is a different mechanism and works.
- **`whichllm` score ≠ coding benchmark**: whichllm blends AA Intelligence Index, Aider, LiveBench (intelligence weighted). For Claude Code / Pi Agent loops, cross-reference SWE-bench Verified — Gemma-4-26B-A4B ranks top in whichllm but scores only ~17% on SWE-bench Verified (bad coding agent despite high general intelligence).
- **Leaderboard beats "highest score"**: pick a model from the Day/Night leaderboard (every complete model, IQ-first with a ±0.05 near-tie band — [ADR 0017](docs/adr/0017-rank-membership-quality-first.md)), not the highest single-axis leader. Near-ties: Day = higher TPS, Night = larger ctx.
- **Configured ctx is the ctx axis**: `llama-server` reserves KV for full `CTX_SIZE` even when a short coding prompt uses few tokens. Night loops that fill 65k+ need that reservation; coding-10 alone does not prove lung capacity.
