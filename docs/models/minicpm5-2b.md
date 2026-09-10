# MiniCPM5-2B

## Header Block

- **Source repo**: [`openbmb/MiniCPM5-2B-GGUF`](https://huggingface.co/openbmb/MiniCPM5-2B-GGUF)
- **Base model**: [`openbmb/MiniCPM5-2B`](https://huggingface.co/openbmb/MiniCPM5-2B)
- **License**: Apache 2.0
- **Local file path**: `models/openbmb/MiniCPM5-2B/MiniCPM5-2B-Q4_K_M.gguf`
- **Family**: MiniCPM5 series (~2.6B parameters, `MiniCPM5 2.6B`)
- **Quantization**: `Q4_K_M` (file size: 1,489.0 MiB / 1.56 GB)

---

## Architecture

*Verified directly from GGUF metadata via `gguf.GGUFReader` and `scripts/model_info.py`.*

- **GGUF Architecture**: `llama`
- **Architecture Class**: Dense causal language model (no MoE layers)
- **Block Count (`llama.block_count`)**: 42 transformer layers
- **Native Context Length (`llama.context_length`)**: 131,072 tokens (131k native context)
- **Attention Pattern**: Standard dense causal attention
- **KV Cache Estimate (`f16` @ 64k)**: 1,344.0 MB (~2.6 GB @ 131k)
- **Quantization Fit**: Entire model fits in GPU VRAM (42/42 layers offloaded)

---

## Hardware Requirements

| Parameter | Value |
|---|---|
| Model size | ~2.6B parameters |
| Quantization size | 1.45 GiB (`Q4_K_M`) |
| Physical VRAM required | $\ge 4.0\text{ GB}$ (comfortable at 131k context with `q4_0` KV cache) |
| Runtime Requirement | Supported in standard `llama.cpp` upstream (tested and validated on `upstream@b10867`). |

---

## Recommended Settings

From model card defaults and `UNIVERSAL_FALLBACK_SAMPLER`:

| Setting | Value | Notes |
|---|---|---|
| `TEMP` | 0.8 | General fallback |
| `TOP_P` | 0.95 | Recommended sampling |
| `TOP_K` | 40 | Bounded top-k |
| `MIN_P` | 0.05 | Universal fallback floor |
| `REPEAT_PENALTY` | 1.0 | Neutral |
| `PRESENCE_PENALTY` | 0.0 | Neutral |

---

## MTP Section

- **MTP Tensors in this GGUF**: None (standard dense causal LM without extra prediction heads).
- **Embedded MTP Support**: Not present in `MiniCPM5-2B-Q4_K_M.gguf`.
- **Harness Speculative Decoding**: `--spec-type none`.

---

## MoE Split

- **Architecture**: Dense (not MoE).
- **`N_GPU_LAYERS`**: 99 (all 42 layers offload to GPU).
- **`N_CPU_MOE`**: `None` (dense models must fit entirely in physical VRAM; offload not supported).

---

## Our Config Baseline (Trial Validated — on_front)

Measured on discrete 8 GB-class NVIDIA GPU via `benchmark_search.py --agentic-full` on 2026-09-09:

- **Status**: **`on_front`** (Day #24 / Night #23 Pareto frontier)
- **Engine**: `upstream@b10867` (CUDA 13.3)
- **Context Size**: `131072` (tested at full 131k context)
- **KV Cache**: `q4_0`
- **Batch / Ubatch**: 512 / 128
- **Threads / Threads-Batch**: 8 / 8
- **Flash Attention**: `on`
- **Measured Throughput (`bench_tg` 512)**: **`127.3 t/s`** (combined TPS: **`136.7 t/s`**)
- **Peak VRAM**: **`5.3 GB`** (fits comfortably in physical VRAM at 131k context without shared spill)
- **Agentic Score (Claw-15 full)**: **`0.7333`** (11/15 passed in 913s)
  - Passed: T002, T004, T006, T008, T010, T012, T014, T016, T018, T044, T050
  - Failed: T046, T048, T053, T054
- **Coding Score (Coding-10)**: **`0.3700`**
  - **HumanEval+**: `0.3000` (3/10)
  - **MBPP+**: `0.7000` (7/10)
  - **LiveCodeBench**: `0.3000` (3/10)
  - **BigCode Hard**: `0.1000` (1/10)

---

## Sources / Verification

- **Primary HF Repo**: `https://huggingface.co/openbmb/MiniCPM5-2B-GGUF` (retrieved 2026-09-09)
- **Base HF Repo**: `https://huggingface.co/openbmb/MiniCPM5-2B` (retrieved 2026-09-09)
- **GGUF Inspection**: Verified via `gguf.GGUFReader` and `scripts/model_info.py` on local weights `models/openbmb/MiniCPM5-2B/MiniCPM5-2B-Q4_K_M.gguf` (2026-09-09)
- **Trial Verification**: Verified in canonical `results.db` via `scripts/rank_results.py --mode pareto` (2026-09-09)

---

## Open questions

*(None — fully verified and benchmarked)*
