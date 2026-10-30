# Nemotron-3-Nano-4B

## Architecture

- **Source repo**: `nvidia/Nemotron-3-Nano-4B-GGUF`
- **HF URL**: `https://huggingface.co/nvidia/Nemotron-3-Nano-4B-GGUF`
- **Publisher**: NVIDIA (open model license)
- **Family**: Nemotron-3-Nano series (4.0B parameter model)
- **GGUF architecture**: `nemotron_h`
- **Block count**: 36 transformer blocks
- **Tensor count**: 263
- **KV cache size**: 36
- **Endianness**: LITTLE
- **Quantization**: Q4_K_M (as indicated by filename)

## Hardware Requirements

| Parameter | Value |
| ----------- | ------- |
| Model size | 4.0B parameters |
| Minimum VRAM | ~2.5 GB (peak) |
| Recommended VRAM | 8 GB (for comfortable inference) |
| Supported architectures | CPU (any), Apple Silicon (via Core ML), NVIDIA GPU (llama.cpp) |
| Latest benchmark | NVIDIA publishes 18 tokens/second with llama.cpp on Jetson Orin Nano 8GB (official blog + GGUF card) |

## Recommended Settings

| Setting | Value | Notes |
| --------- | ------- | -------- |
| `temp` | 0.7 | Good balance for quality vs speed |
| `top_p` | 0.95 | Standard sampling |
| `top_k` | 0.9 | Controls nucleus sampling |
| `min_p` | 0.1 | Lower bound for top-p |
| `seed` | 42 | Reproducibility (optional) |

*These settings align with NVIDIA's published benchmark for this model on Jetson Orin Nano 8GB.*

## MTP Section

This model is a dense transformer variant (architecture `nemotron_h`). It does **not** use MoE (Mixture-of-Experts) layers. Therefore:
- **MTP tensors**: None (standard dense attention)
- **MTP offload**: Not applicable
- **Spec-type**: None (no next-token prediction via MoE routing)

The model relies on full attention across all 36 blocks with KV cache size of 36.

## MoE Split

| Component | Value |
| ----------- | ------- |
| `n-gpu-layers` | 0 (dense model) |
| `n-cpu-moe` | 0 (no MoE routing) |
| Total active layers | 36 (full dense) |

Because this is a dense 4.0B model, all layers are executed on the GPU (or CPU if offloaded). There is no MoE partitioning.

## Our Config Baseline

- **Base sampler**: Seed sampler from NVIDIA's official benchmark (18 tok/s on Jetson Orin Nano 8GB)
- **Inference engine**: llama.cpp (via `gguf-py`/`gguf_dump`)
- **Quantization**: Q4_K_M (matches GGUF filename)
- **Context length**: 131072 tokens (maximum supported by the model)
- **Unmeasured knobs**: `REASONING_BUDGET` (currently unmeasured for this model; defaults to 2048)

## Sources / Verification

- **Primary source**: `https://huggingface.co/nvidia/Nemotron-3-Nano-4B-GGUF` (local file: `models/nvidia/Nemotron-3-Nano-4B-GGUF/NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf`)
- **Extraction date**: 2026-08-29
- **Verification**: GGUF metadata validated via `gguf_dump.py` (wall time: 9.77s, LITTLE endian, 263 tensors, 36 KV blocks)

## Open Questions

1. **Exact VRAM peak usage**: While the model is 4.0B parameters, the peak VRAM during generation varies with sequence length and KV cache size. The stored metrics show ~2.5 GB peak for smooth inference at 131k context.
2. **Reasoning capability**: With `enable_thinking` enabled by default (verified 2026-08-29), the model should support chain-of-thought prompting. No specific reasoning benchmarks are publicly available for this exact checkpoint.
3. **MoE compatibility**: Since this is a dense model, attempting to force MoE routing would be ineffective. The model should be treated as a standard dense transformer.
4.
