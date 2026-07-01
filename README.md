# local-model-autoresearch

Autonomous hill-climbing optimizer for local LLM runtime flags via `llama.cpp`. Based on [karpathy/autoresearch](https://github.com/karpathy/autoresearch).

**What it does:** Finds the fastest, most accurate runtime config for your local GGUF model by benchmarking thousands of flag combinations automatically.

**What it doesn't do:** Re-quantize models. Only tunes how the model is served (KV cache, batching, threads, MTP).

---

## Quickstart (Agent-Driven)

Open your coding agent (Claude Code, Codex, Pi Agent, OpenCode) and paste:

> *"Discover the best model for **coding** that fits my PC, download it, and start auto-tuning."*

The agent will:
1. Detect your hardware (GPU/VRAM/RAM)
2. Run `whichllm` to shortlist candidates
3. Cross-check with SWE-bench / Aider / LiveCodeBench
4. Plot Pareto frontier (tok/s vs quality)
5. Edit `autoresearch/core/config.py` with the best model
6. Run `python3 autoloop.py --vram-limit-mb=<budget>` overnight

**Result in the morning:** `results.tsv` with all trials + `config.py` at the best config found.

---

## Prerequisites

Install these **before** asking your agent:

| Dependency | Command | Why |
|---|---|---|
| Python 3.11+ | `sudo apt install python3.11 python3.11-venv` | autoloop runtime |
| CUDA Toolkit | `nvidia-smi` + NVIDIA driver | llama.cpp needs `-DGGML_CUDA=ON` |
| build-essential + cmake >= 3.14 | `sudo apt install build-essential cmake` | compile llama.cpp |
| uvx (or uv) | `pip install uv` | run `uvx whichllm@latest` |
| huggingface_hub[cli] | `pip install huggingface_hub[cli]` | download GGUFs |

Then clone and compile `llama.cpp` (see [Build section](#build-llamacpp-with-cuda) below).

### Verify readiness

```bash
bash scripts/setup-check.sh
```

Green output = ready to run autoloop.

---

## How It Works

### The Loop

1. Read current best config from `autoresearch/core/config.py`
2. Run all enabled benchmarks (Coding: HE+, MBPP+, LCB, BigCodeBench)
3. Compute Val Score (weighted accuracy + TPS floor enforcement)
4. Mutate one param -> generate Neighbor config
5. Evaluate Neighbor -> keep if improved (or Pareto tie-break)
6. If local maxima -> random restart
7. Loop forever until Ctrl+C

### Editing Contract

| File | What | Can agent/loop edit? |
|---|---|---|
| `autoresearch/core/config.py` | Runtime config | **Yes** (constants only) |
| `benchmark_search.py` | CLI runner | **No** |
| `autoresearch/benchmarks/*` | Evaluation logic | **No** |
| `results.tsv` | Trial metrics | **Append only** |

### Val Score

Single scalar metric for keep/discard decisions:
- With Coding: `80% Coding + 10% Nexus + 10% Claw`
- Without Coding: `60% Nexus + 40% Claw`

TPS Floor = 20 tok/s. Below this -> score zeroed.

### Safety

- VRAM pre-flight check before every server start
- Flash attention always on
- All failures logged as `FAIL` in results.tsv, loop continues
- Never pushes to remote

---

## Build llama.cpp with CUDA

Clone inside repo root for auto-detection:

```bash
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

cmake -B build-cuda \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DLLAMA_BUILD_SERVER=ON \
  -DCMAKE_CUDA_ARCHITECTURES=native

cmake --build build-cuda --config Release -j
```

If cloned elsewhere, export the path:

```bash
export AUTORESEARCH_LLAMA_CPP_ROOT="/path/to/llama.cpp"
```

### Forks (TurboQuant / MTP)

For advanced KV cache modes (`turbo2`, `turbo3`, `turbo4`, SPEC MTP):

- **TurboQuant**: `https://github.com/TheTom/llama-cpp-turboquant`
- **MTP & TurboQuant**: `https://github.com/BoFan-tunning/llama.cpp-MTP-TurboQuant`

Clone as `llama.cpp` in repo root. Build commands are identical.

---

## After Tuning: Serve the Model

```bash
# Show the command (don't start)
python3 scripts/serve-config.py print-cmd

# Start llama-server detached
python3 scripts/serve-config.py serve

# Check status
python3 scripts/serve-config.py status

# Stop
python3 scripts/serve-config.py stop
```

Plug into your agent:

```
base_url: http://127.0.0.1:18080/v1
model:    <model-name-from-config>
```

---

## Manual Mode

If you prefer hands-on:

1. Read `program.md` for rules
2. Edit `autoresearch/core/config.py` with a hypothesis
3. Run `python3 benchmark_search.py --desc "your hypothesis"`
4. Check `results.tsv` for results
5. Keep if Val Score improved, revert otherwise

---

## Supported Profiles

| Profile | Benchmarks | Example Models |
|---|---|---|
| **Coding** (default) | SWE-bench, Aider, HE+ | Qwen3.6-27B, Qwen3.6-35B-A3B |
| **Writing** | MMLU-Pro, Chatbot Arena | Qwen3-14B, Gemma3-12B |
| **Vision** | MMMU-Pro, MMBench | Qwen3-VL, Gemma-4-26B-A4B |

Toggle in `autoresearch/core/config.py`:

```python
INCLUDE_CODING = True
INCLUDE_NEXUS = False
INCLUDE_CLAW = False
```

---

## Agent Documentation

For agents working on this repo, read these in order:

1. `AGENTS.md` (root) — DOX hierarchy, work contracts
2. `program.md` — Search protocol rules
3. `GOLDEN-RULES.md` — Performance flags, safety, validation
4. `CONTEXT.md` — Terminology and definitions
5. `docs/discovery/discover-models.md` — Model selection workflow
6. `docs/discovery/whichllm-reference.md` — CLI reference
7. `docs/discovery/quantization-cascade.md` — Quant format selection
8. `docs/llamacpp-toolset.md` — llama.cpp binary reference

---

## License

MIT
