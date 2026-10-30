# NVFP4 — NVIDIA 4-bit Floating-Point Format (Native Blackwell + Weight-Only Fallback)

NVFP4 is NVIDIA's 4-bit floating-point format (E2M1) built for Blackwell Tensor Cores. Native acceleration is Blackwell-only, but the checkpoint format **loads and runs on older NVIDIA GPUs** via a weight-only Marlin fallback — the distinction matters for this repo's 8 GB-class rig.

## Format structure

E2M1 micro-float, same family as FP4 / MXFP4:

- 1 sign bit, 2 exponent bits, 1 mantissa bit
- Representable range ≈ −6 … +6 (e.g. 0.0, 0.5, 1.0, 1.5, 2, 3, 4, 6, and negatives)
- Each encoded value is reconstructed as `x = x_q × s_block × s_global`

NVFP4's two innovations over MXFP4:

1. **Two-level scaling**
   - Per-16-value micro-block scale `s` in **E4M3 (FP8)** precision — fractional (non-power-of-two), so it can fit the block's actual distribution
   - Second-level per-tensor **FP32 scalar** that rescales the tensor so micro-blocks land in the E4M3 encodable range
2. **Half the MXFP4 block size**: 16 values/block vs 32 → twice as many opportunities to match local dynamic range

E4M3 vs E8M0 scaling: E8M0 snaps the scale to nearest 2ⁿ (cheap, can blow up error on the block maximum); E4M3 picks one fractional scale minimizing collective block error. Example in the source blog: E4M3 blocks average MSE 0.08 vs E8M0. E8M0's only advantage is simplicity (no extra per-tensor scalar), adequate for scale-insensitive weights/activations.

## Memory

- 4 bits/value + 1 FP8 scale per 16 values ≈ **4.5 bits/value**, plus one FP32 per tensor
- ≈ **3.5× less memory than FP16**, ≈ **1.8× less than FP8**

## Accuracy & performance (from NVIDIA's data)

- DeepSeek-R1-0528: ≤1% degradation FP8 → NVFP4 via PTQ across 7 evals; AIME 2024 measured 2% *better* than FP8
- Energy: up to 25× (Blackwell) / 50× (Blackwell Ultra) energy efficiency per token vs H100 baseline on GPT-MoE 1.8T
- Memory + accuracy wins scale to rack systems: GB300 NVL72 = 40 TB memory budget (36 Grace Blackwell Ultra Superchips) — relevant for test-time scaling

## Ecosystem / getting started

- Quantize: TensorRT Model Optimizer, LLM Compressor (PTQ / QAT), Unsloth Dynamic NVFP4
- Export: Unified Hugging Face checkpoint (`hf_quant_config.json` `quant_algo: NVFP4`); ONNX for non-LLM models
- Serve: TensorRT-LLM, vLLM (`modelopt_fp4` / `compressed-tensors` W4A4 NVFP4), and SGLang (`modelopt_fp4`). All three have **native W4A4 on Blackwell** and **automatic Marlin W4A16 fallback on SM75+** — no manual flags. (see [vllm-quantization.md](./vllm-quantization.md) and [vllm-quant-deep-dive.md](./vllm-quant-deep-dive.md) §4 for the full matrix)
- Pre-quantized HF checkpoints: DeepSeek-R1-0528, Llama 3, FLUX.1-dev, Qwen3.6-27B/35B-A3B, Ornith-1.5-35B-A3B-NVFP4 (this prompt's example — `ornith-ai/Ornith-1.5-35B-A3B-NVFP4`)

## Relevance to this repo

- **Two hardware paths (do not conflate):**
  - **Native W4A4 (Blackwell-only for speed):** Blackwell Tensor Cores (datacenter SM100/103: B100/B200/GB200/GB300; consumer SM120: RTX 5090/5080/PRO 6000; DGX Spark SM121 — all need `tcgen05` FP4 instructions). Full memory + **~2× FP8 GEMM** compute win. Gate: CUDA ≥12.8, `ENABLE_NVFP4_SM100/SM120` build.
  - **Marlin weight-only fallback (not Blackwell-only):** `vllm/model_executor/layers/quantization/utils/marlin_utils_fp4.py` + SGLang `PR #19652` — packed FP4 weights load on **SM75+** (Turing and later: Ampere SM80/SM86, Ada SM89 including this repo's discrete 8 GB-class, Hopper SM90); in-kernel dequant → W4A16-class via `MarlinNvFp4LinearKernel` (group size 16 only). Memory win; **no** native FP4 speed edge — throughput is `W4A16` class, and can be slower than native W4A4. Automatic fallback when `!is_blackwell_supported() && is_fp4_marlin_supported()`; override `VLLM_NVFP4_GEMM_BACKEND=marlin` / `SGLANG_FORCE_NVFP4_MARLIN=1`. Prefer **float16** activations on fallback (BF16 + Marlin has reported garbled output).
- **Tester for this rig:** `Ornith-1.5-35B-A3B-NVFP4` *will* load on an 8 GB-class Ada GPU (SM89) via the Marlin fallback (memory: ~4.5 bits/value), but expect **W4A16 speed** — compare to `Ornith-1.5-35B-A3B-GGUF Q4_K_M 21.7GB` (`llama.cpp b10549` `llama-server` path) for a fair `memory vs TPS` read, not `speed vs speed`.
- **Not a GGUF/K-quant**: NVFP4 safetensors (`hf_quant_config.json`) live in the NVIDIA inference stack (TensorRT-LLM, vLLM, SGLang). llama.cpp K-quants / UD / IQ are separate — but Unsloth also ships **NVFP4-GGUF** (Qwen3.6 NVFP4-GGUF) which *is* a GGUF container holding NVFP4-like packed weights; treat it as an NVIDIA-format-in-GGUF, still Blackwell-accelerated only via Unsloth kernels.
- Unsloth ships both NVFP4 safetensors and NVFP4-GGUF (e.g. Qwen3.6 27B / 35B-A3B, MTP-integrated) — see [unsloth-qwen-guides.md](./unsloth-qwen-guides.md) for running them. Unsloth docs explicitly: Blackwell for the speed win, older GPUs → use their standard GGUF.
- **Footprint ≠ “4-bit marketing name”:** NVFP4 only compresses layers the export quantized. Multimodal / hybrid HF packs often leave **vision tower**, **`lm_head`**, and some SSM/gate tensors in BF16. A “9B NVFP4” checkpoint can still be **~8 GB on disk/VRAM**, while a **text-only GGUF Q4** of the same family is often **~5–6 GB**. Size the load against Baseline `VRAM_LIMIT_MB` before serving; do not raise `--gpu-memory-utilization` or enable CPU offload to force an oversized pack onto an 8 GB-class card.
## Sources / Verification

- NVIDIA Developer Blog, *Introducing NVFP4 for Efficient and Accurate Low-Precision Inference*, Jun 24, 2025 — https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/ (format, E2M1, E4M3 vs E8M0, block 16).
- NVIDIA Developer Blog, *NVFP4 Trains with Precision of 16-Bit…*, 2025 — https://developer.nvidia.com/blog/nvfp4-trains-with-precision-of-16-bit-and-speed-and-efficiency-of-4-bit/ (training context, micro-block 16).
- Transformer Engine NVFP4 docs — https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/nvfp4/nvfp4.html (math: `x = x_e2m1 × s_block × s_global`, swizzling, layout) — accessed 2026-08-22.
- Marlin weight-only fallback (not Blackwell-only): vLLM `vllm/model_executor/layers/quantization/utils/marlin_utils_fp4.py` (`apply_fp4_marlin_linear`, SM75+, group 16; `MarlinNvFp4LinearKernel` W4A16) · SGLang `PR #19652` *NVFP4 Marlin fallback for non-Blackwell GPUs (SM75+)* https://github.com/sgl-project/sglang/pull/19652 (auto fallback when `!is_blackwell_supported()`) · vLLM `MarlinNvFp4LinearKernel` https://docs.vllm.ai/en/latest/api/vllm/model_executor/kernels/linear/nvfp4/marlin/ · DGX Spark SM121 fix https://forums.developer.nvidia.com/t/marlin-fix-nvfp4-actually-works-on-sm121-dgx-spark/365119 ( `VLLM_NVFP4_GEMM_BACKEND=marlin` for broken CUTLASS on SM121)
- Unsloth Dynamic NVFP4 (Blackwell speed, fallback note): https://unsloth.ai/docs/basics/nvfp4 · https://unsloth.ai/docs/models/qwen3.6 ("older GPUs → use standard GGUF").
- BF16 + Marlin garbled output caveat: https://github.com/vllm-project/vllm/issues/34694
- Verified against: `docs/discovery/vllm-quant-deep-dive.md` §4 (kernel table) and `docs/sessions/2026-08-02-research-gap-closure.md`.
