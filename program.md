# autoresearch

This is a Karpathy-style autoresearch loop adapted to the local Nexus runtime.

The contract is strict:

- `program.md` is fixed (unless explicitly requested by user).
- `prepare.py` and `prepare_claw.py` are fixed.
- `benchmark_search.py` is the only file you are allowed to hack during the search.

The goal is to push **Qwen 3.5 2B/4B/9B (including MTP/Coder)** and **Gemma 4B** as far as possible on a single RTX 4060 8GB using a fixed dual-pass evaluation harness (Retrieval + Agency).

## Setup

To start a fresh run:

1. **Agree on a run tag**: use a fresh branch tag such as `mar15`.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from `main`.
3. **Read the in-scope files**:
   - `program.md` — the rules for the experiment
   - `prepare.py` — Nexus Retrieval harness
   - `prepare_claw.py` — ClawBench Agency harness
   - `benchmark_search.py` — the only mutable tuning surface
4. **Verify local assets exist**:
   - GGUF models in `llm/gguf/`
   - `llama.cpp/build/bin/llama-server` or `llama.cpp/build-cuda/bin/llama-server`
5. **Initialize results.tsv** with only the header row:
   - `commit	val_score	memory_gb	status	description`

Once the setup is clean, begin the loop.

## Evaluation model

Each run starts a local `llama-server`, executes the dual evaluation harness, and reports a composite primary metric:

- **`val_score`**: higher is better — composite of Agency (60%) and Retrieval (40%).

### Pass 1: Nexus Retrieval (weight 0.40)
~50 000 tokens of synthetic Nexus history are injected before the task prompt to test context stress. The model must find the override token in memory, verify it, and unlock the control plane.

### Pass 2: ClawBench Agency (weight 0.60)
Runs 11 ClawBench tasks with 0 context noise. Tests pure tool-use (JSON formatting) and instruction-following logic.

### Throughput Hard Floor (30 TPS)
The combined average `tokens_per_sec` MUST be >= 30.0 TPS. If the throughput drops below this floor, the configuration is aggressively penalized (`val_score = 0.0`) and discarded.

### Secondary metrics
- `val_retrieval` — Retrieval score in isolation
- `val_agency` — Agency score in isolation
- `tokens_per_sec` — throughput from both passes combined
- `total_seconds` — total wall-clock time
- `peak_vram_mb` — maximum VRAM usage

The harness is fixed and deterministic:
- direct local runtime only
- no LiteLLM
- no MCP server dependency
- fixed context requirement of 128k (131072 tokens)

## What you CAN do
- Modify `benchmark_search.py`
- Change model selection
- Change runtime configuration: KV cache type, batch sizes, threads, parallelism, and similar server parameters
- Change generation knobs inside `benchmark_search.py` (TEMP, TOP_P, MIN_P, etc.)

## What you CANNOT do
- Modify `prepare.py` or `prepare_claw.py`
- Modify `program.md` (once baseline is set)
- Add new dependencies
- Change the fixed evaluation fixture or scoring logic
- Use CPU offload as a hidden fallback (Keep `--n-gpu-layers 999`)

## Constraints
- **Hardware target:** RTX 4060 8GB
- **Context Size:** 128k (131072) is mandatory.
- If `peak_vram_mb` approaches or exceeds ~7900 MB, assume the config is unsafe.
- Simpler configs win ties.

## Output format
Each successful run must print a summary block like:

```text
---
val_score:        0.374961
val_retrieval:    0.624950
val_agency:       0.208302
tokens_per_sec:   138.77
total_seconds:    263.3
peak_vram_mb:     7762.0
ctx_size:         131072
kv_cache:         q4_0
model:            Qwopus3.5-9B-Coder-MTP-Q4_K_M.gguf
eval_seconds:     255.425
```

Quick extraction:
```bash
grep '^val_score:\|^val_retrieval:\|^val_agency:\|^tokens_per_sec:\|^peak_vram_mb:' run.log
```

## Logging results
Log every experiment to `results.tsv` as tab-separated data:

```text
commit	val_score	memory_gb	status	description
```
Do not commit `results.tsv`.

## The experiment loop
1. Inspect the current branch and commit.
2. Edit `benchmark_search.py` with one experimental idea.
3. Commit.
4. Run: `python benchmark_search.py > run.log 2>&1`
5. Read results: `grep '^val_score:\|^peak_vram_mb:' run.log`
6. If the run crashed, inspect `tail -n 50 run.log` and fix launch bugs if needed.
7. Append the outcome to `results.tsv`.
8. Keep the commit only if `val_score` improved. Otherwise, reset to the previous best commit.

## Timeout and failure handling
- A normal high-accuracy run should finish in 5 to 15 minutes.
- If a run exceeds 20 minutes, kill it and treat it as a failure.
- Launch bugs are acceptable to fix inside `benchmark_search.py`.
- Harness changes are not allowed.

## Autonomy rule
Once the loop has started, continue autonomously until manually interrupted.
Do not pause to ask if you should keep going.