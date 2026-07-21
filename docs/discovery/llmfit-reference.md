# llmfit CLI Reference

Rust-based hardware-aware LLM sizing and discovery CLI/TUI tool ([AlexsJones/llmfit](https://github.com/AlexsJones/llmfit)). Auto-detects GPU, CPU, and RAM to determine which models and quants fit on your hardware across local and provider ecosystems.

**Repository:** [https://github.com/AlexsJones/llmfit](https://github.com/AlexsJones/llmfit)  
**Install Methods:**
- **Cargo (Rust):** `cargo install llmfit`
- **Windows (Scoop):** `scoop install llmfit`
- **macOS/Linux (Homebrew):** `brew install AlexsJones/llmfit/llmfit` (or `brew tap AlexsJones/llmfit && brew install llmfit`)
- **Shell Script:** `curl -fsSL https://llmfit.axjns.dev/install.sh | sh`

---

## Capabilities Overview

- **Hardware Analysis:** Detects GPU model, VRAM capacity, CPU architecture, and system RAM.
- **Model Sizing & Fit:** Calculates memory footprints (weights + KV cache context) for various quantization formats (Q4_K_M, Q8_0, FP16, EXL2).
- **Interactive TUI & CLI:** Provides an interactive TUI for filtering/navigating models or direct CLI output for scripting.
- **Planning Mode:** Simulates RAM/VRAM requirements for target models before downloading.
- **Local Runtime Integration:** Supports launching models with Ollama, `llama.cpp`, or MLX.

---

## Primary Commands

### 1. `llmfit` (Default — Interactive TUI / Model Ranks)

Run without arguments to inspect hardware fit and explore supported models.

```bash
# Launch interactive TUI or default model ranking
llmfit
```

### 2. `llmfit plan <model>`

Estimate memory (VRAM/RAM) footprint and speed for a specific model configuration.

```bash
# Estimate fit for Qwen 3.5 9B
llmfit plan "qwen 3.5 9b"

# Plan specific model with custom context length
llmfit plan "qwen 3.6 35b" --context 131072
```

### 3. `llmfit hardware`

Display detected hardware specs (GPU, VRAM, CPU, RAM).

```bash
llmfit hardware
```

### 4. `llmfit run <model>`

Download and launch runtime execution (Ollama / llama.cpp) for the selected model.

```bash
llmfit run "qwen 3.5 9b"
```

---

## Integration with `local-model-autotuning`

```bash
# 1. Discover hardware fit & candidate models
llmfit plan "qwen 3.5 9b"

# 2. Cross-reference candidate models with coding leaderboards (SWE-bench, Aider, LiveCodeBench)
# 3. Download target GGUF into models/ store using HF CLI:
hf download Qwen/Qwen3.5-9B-Instruct-GGUF qwen3.5-9b-instruct-q4_k_m.gguf --local-dir models/Qwen/Qwen3.5-9B-Instruct-GGUF

# 4. Set Baseline MODEL in autoresearch/core/config.py and execute autoloop
```

---

## Related Docs

- [`discover-models.md`](./discover-models.md) — End-to-end model discovery workflow (`whichllm` + `llmfit`)
- [`whichllm-reference.md`](./whichllm-reference.md) — Reference for `whichllm` discovery CLI
- [`quantization-cascade.md`](./quantization-cascade.md) — Quantization format decision guide
