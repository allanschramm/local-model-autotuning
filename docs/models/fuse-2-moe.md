# Fuse-2-MoE

## Header Block

- **Source repo**: [`Akahsizrr/Fuse-2-MoE-GGUF`](https://huggingface.co/Akahsizrr/Fuse-2-MoE-GGUF)
- **Base model**: `Akahsizrr/Fuse-2` (Fuse4 architecture on `fuse-2-boosted` base checkpoint)
- **License**: Apache 2.0
- **Local file path**: `models/Akahsizrr/Fuse-2-MoE-GGUF/Fuse-2-MoE-Q4_K_M.gguf`
- **Family**: Fuse4 / Qwen3.5 MoE hybrid (~8.87B total parameters, ~4B active)
- **Quantization**: `Q4_K_M` (file size: 5,458.0 MiB / 5.72 GB on disk)
- **Engine fork required**: `llama.cpp-fuse4` (patches for `blk.{i}.ffn_moe_norm` and `build_layer_ffn_fuse4`)

---

## Architecture

*Verified directly from GGUF metadata via `gguf.GGUFReader`.*

- **GGUF Architecture**: `qwen35moe`
- **Architecture Class**: MoE hybrid (8 experts per MoE layer, top-2 routing)
- **Block Count (`qwen35moe.block_count`)**: 32 layers total (layers 3–26 are MoE-augmented, rest are dense)
- **Expert Count (`qwen35moe.expert_count`)**: 8
- **Expert Used Count (`qwen35moe.expert_used_count`)**: 2
- **Shared Expert**: Host FFN acts as active shared expert
- **Bridge Projections**: 2560→5120 and 5120→2560 folded into expert weights
- **Custom Normalization**: `blk.{i}.ffn_moe_norm` applied to MoE output before static scale `0.018421f` and addition to shared expert
- **Native Context Length**: 262,144 tokens (GGUF metadata)

---

## Hardware Requirements

| Parameter | Value |
|---|---|
| Model size | ~8.87B parameters total, ~4B active per token |
| Quantization size | 5.32 GiB tensor data (`Q4_K_M`) |
| Physical VRAM required | $\ge 7.0\text{ GB}$ (measured peak **7.1 GB** @ 32,768 context + `q4_0` KV cache + flash-attn) |
| Runtime Requirement | **Patched `llama.cpp-fuse4` fork** (build with CUDA 13.3 sm_89). Stock builds fail with missing `blk.{i}.ffn_moe_norm.weight`. |

---

## Recommended Settings

From model README and empirical trial:

| Setting | Value | Notes |
|---|---|---|
| `TEMP` | 0.7 | Standard sampling temperature |
| `TOP_P` | 0.95 | Nucleus sampling |
| `TOP_K` | 20 | Bounded top-k |
| `MIN_P` | 0.05 | Minimum probability floor |
| `REPEAT_PENALTY` | 1.05 | Repetition penalty |
| `REASONING` | `off` | Base model without reasoning instructions; auto-reasoning in `llama-cli` stalls |
| `JINJA` | `True` | Jinja chat template rendering |

---

## MTP Section

- **MTP Tensors in this GGUF**: None (0 MTP / nextn tensors in GGUF metadata).
- **Embedded MTP Support**: Not present in `Fuse-2-MoE-Q4_K_M.gguf`.
- **Harness Speculative Decoding**: `--spec-type none`.

---

## MoE Split

- **Architecture**: MoE hybrid (`qwen35moe`).
- **`N_GPU_LAYERS`**: 99 (all layers fit in 8 GB physical VRAM).
- **`N_CPU_MOE`**: 0 (no CPU offloading required; whole model resides in VRAM).

---

## Our Config Baseline (Trial Validated — on_front)

Measured on discrete 8 GB-class NVIDIA GPU via `benchmark_search.py --agentic-full` on 2026-09-12:

- **Status**: **`on_front`** (Trial fingerprint: `732e0d95-e589-455f-8ca2-9d42c9e3c0f5`)
- **Engine**: `llama.cpp-fuse4` (commit `50f068fff` + Fuse4 patches, CUDA 13.3 sm_89)
- **Context Size**: `32768`
- **KV Cache**: `q4_0`
- **Batch / Ubatch**: 512 / 128
- **Threads / Threads Batch**: 8 / 12
- **Flash Attention**: `on`
- **Measured Throughput (`bench_tg`)**: **`56.4 t/s`** (median across reps: 56.3, 56.4, 56.5; coding generation TPS: **`67.5 t/s`**)
- **`llama-bench` Throughput**: `pp512` **`2495.4 t/s`**, `tg128` **`58.3 t/s`**
- **Peak VRAM**: **`7.1 GB`** (fits within 8 GB physical VRAM with 0 Shared GPU spill)
- **Agentic Score (Claw-15 full)**: **`0.0000`** (0/15 passed; raw unaligned base model without tool-calling fine-tuning)
- **Coding Score (Coding-10)**: **`0.0500`**
  - **HumanEval+**: `0.2000` (2/10)
  - **MBPP+**: `0.0000` (0/10)
  - **LiveCodeBench**: `0.0000` (0/10)
  - **BigCode Hard**: `0.0000` (0/10)

---

## Sources / Verification

- **Primary HF Repo**: `https://huggingface.co/Akahsizrr/Fuse-2-MoE-GGUF` (retrieved 2026-09-12)
- **Model README**: `https://huggingface.co/Akahsizrr/Fuse-2-MoE-GGUF/raw/main/README.md` (retrieved 2026-09-12)
- **GGUF Inspection**: Verified via `gguf.GGUFReader` on local weights `models/Akahsizrr/Fuse-2-MoE-GGUF/Fuse-2-MoE-Q4_K_M.gguf` (2026-09-12)

---

## Open Questions

- **Instruction fine-tuning**: Model author released raw base checkpoint; an instruct or tool-tuned fine-tune would be required to unlock agentic capabilities (current Claw-15 score is 0.0000).
- **Extended context (>32k)**: Model claims 262k context, but physical 8 GB VRAM budget limits context to 32k with `q4_0` KV cache; testing 64k requires TurboQuant/KV quantization experiments.
