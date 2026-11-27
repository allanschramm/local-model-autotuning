# Inkling 2 Mini runtime comparison, November 25, 2026

The [Unsloth llama.cpp setup](2026-11-23-inkling-expert-placement.md) at [100K](2026-11-24-long-context-serving.md) was the control. Windows tests used an 8 GiB-class CUDA device. Strata Terminal 0.2.22 used CUDA engine `strata-inkling` 0.1.69. The smoke compared three requests with a short common text prompt, seeds 41–43 and 512 generated tokens each. The reported metric is median decode TPS from the engine, excluding prefill. The tokenizers or wrappers yielded 44 input tokens on llama.cpp and 46 on Strata. Strata reused a 45-token prefix on the second and third requests. [Strata release](https://github.com/Niko1221/Strata/releases/tag/v0.2.22)

| Runtime | Quantization | Observed context | Median decode TPS |
| --- | --- | ---: | ---: |
| Unsloth llama.cpp | Same `UD-Q4_K_M` GGUF | 100,096 | 19.72 |
| Strata CUDA | Same `UD-Q4_K_M` GGUF | 70,016 | 16.8 |
| Strata CUDA | Native `SQ-10` | 100,096 | 22.2 |

The GGUF path auto-offloaded part of the KV cache to host memory and exposed a smaller usable window. SQ-10 gained 2.48 decode TPS over the control with a different quantization. It reserved an 8.60 GB pageable host expert arena, plus pinned transfer buffers. Total system RAM during generation was unmeasured. Strata disabled DSpark after its own probe measured slower decoding. All timed Strata samples show zero draft tokens. The smoke did not establish matched quality or the vendor's headline speedup at 100K. Local evidence consists of API timings and engine logs.

## What the native packs compare

SQ-10, SQ-15 and SQ-20 are native `.strata` packs of 10.47, 14.35 and 19.35 GB. The sizes cover the complete files, including the bundled draft, and cannot establish per-tensor precision or match a GGUF quantization. SQ-20 is the closest native pack to the original mixed NVFP4/MXFP8 checkpoint per the publisher's comparison. Of the native packs, only SQ-10 ran locally. The publishers' divergence and PPL tables use different references or protocols. Establishing equivalence to `UD-Q4_K_M` requires the same text, reference, tokenizer inputs and evaluation protocol. [Strata pack card](https://huggingface.co/strata-engine/Inkling-2-Mini-Strata) · [Original checkpoint](https://huggingface.co/thinkingmachines/Inkling-2-Mini) · [Unsloth quantization analysis](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF#quantization-analysis)

## Experimental Windows llama.cpp build

The `inkling-fast` CUDA 12.4 binary used the same `UD-Q4_K_M`, target 100,000, K/V `q4_0` and batch/ubatch 512/16. With seven GPU expert blocks the VRAM estimate rejected the launch (`f5b76234`). With six blocks (6–11) the fixed-text PPL reached 53.9543 versus 4.8490 in the earlier six-block Unsloth control at 65,536. The generation stage then failed slot validation with `INFRA_ERROR` and no valid TPS (`448cb855`). The build was rejected and the Unsloth setup was restored and checked at 100,096. The observations establish two failure symptoms under the tested profiles. Their root causes were unisolated.

A different experimental fork reported 34.1 and 32.8 TPS with the same GGUF at roughly 1K and 4.8K prompt tokens and 32,768 configured context. It combined multiple patches with loop-aware expert caching and lacked a ready Windows release when checked. The result is external and leaves fit, speed and PPL at 100K open. A matched retest would hold GGUF and slot capacity constant and gate performance on PPL and placement checks. [External test log](https://github.com/eiselems/llama.cpp/blob/dafda2ad7/docs/inkling-perf/README.md) · [Upstream cache issue](https://github.com/ggml-org/llama.cpp/issues/32830)
