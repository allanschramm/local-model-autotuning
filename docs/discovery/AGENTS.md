# AGENTS.md — docs/discovery

## Purpose
User-facing guides for **discovering, evaluating, selecting, and optimizing** local LLMs and inference runtimes (GPU and CPU) that fit the `local-model-autotuning` workflow. Covers tooling (`whichllm`), evaluation methodology (Pareto frontier, Zellinger economic evaluation), quantization strategies, CPU optimization (AVX-512/AMX/NUMA), and engine architectures, plus harness-side capability extraction (IQ per token).

## Ownership
- Owned by: `local-model-autotuning` developers.
- Stable contracts: whichllm CLI contract, Pareto frontier method, scoring rules.

## Local Contracts
- **Read-mostly**: discovery docs are guides for users to follow. Code under `autoresearch/` is the loop surface.
- **No model-specific claims**: docs reference model families (e.g. "Qwen3.6 MoE") not private paths or single-user hardcoded values.
- **No alias registry in repo**: `model-up` alias names, ports, and `models/aliases/*/config.yaml` are machine-local (`/models/` gitignored). Tracked docs use GGUF basenames + benchmark scores only.
- **No external-source citations in technique claims**: methodology names allowed (Zellinger framework), but no HF/Unsloth/blog URLs in technique descriptions.

## Work Guidance
- New guides go here when they describe a reusable methodology, not a single session's data.
- Single-run session logs → `docs/sessions/` (cavity for empirical capture).
- Per-model GGUF specs and architecture notes → `docs/models/`.

## Verification
- Guides should be **runnable** by a user with the documented steps. Steps referencing a tool should include the install path (`uvx whichllm@latest`, etc).
- Cross-check examples against current `whichllm --help` before claiming a flag/option.

## Child DOX Index

### 1. Tooling & Onboarding
- [`discover-models.md`](./discover-models.md) — end-to-end workflow: discovery (whichllm/llmfit) → Pareto report → pick GGUF → TPS climb writes Fingerprint file → `model-up` → Pi (ADR 0014).
- [`whichllm-reference.md`](./whichllm-reference.md) — full whichllm CLI reference (commands, flags, profiles, examples).
- [`llmfit-reference.md`](./llmfit-reference.md) — full llmfit CLI/TUI reference (hardware sizing, planning, model search, examples).
- [`agent-onboarding.md`](./agent-onboarding.md) — onboarding guide for future agents.
- [`good-enough-tuning.md`](./good-enough-tuning.md) — default speed path: validation smoke → autoloop `--mode tps` (keeps write the Fingerprint file) → `model-up` → Pi; numeric benches optional via apply-to-Baseline (ADR 0014).
- [`agentic-coding-benchmarks.md`](./agentic-coding-benchmarks.md) — Claw tiers + SWE-lite `--agentic-coding` (ADR 0013).
- [`claw-eval-leaderboard.md`](./claw-eval-leaderboard.md) — ranked Claw-Eval full/quick scores on this 8 GB rig + operational lessons.
- [`visual-pack.md`](./visual-pack.md) — Pi visual-pack stub on the Fingerprint bus: same file as daily Pi, camera rubric optional, never writes rank (issue #54, ADR 0014 phase 4).
- [`visual-pack-prompt.md`](./visual-pack-prompt.md) — frozen workspace-shaped sample prompt 01 (notes-CLI fix); do not edit, add numbered files.
- [`thinking-models-claw-harness.md`](./thinking-models-claw-harness.md) — thinking/`reasoning_content` Claw false-fails: symptoms, fix, remasure policy, regression checklist.
- [`coding-leaderboard.md`](./coding-leaderboard.md) — ranked coding-10 (HE/MBPP/LCB/BC) scores on this 8 GB rig.
- [`pareto-leaderboard.md`](./pareto-leaderboard.md) — full model leaderboard (every complete Objective Vector, IQ-first, ±0.05 near-tie band; ADR 0017); tables are the verbatim `scripts/rank_results.py` output — regenerate, never hand-patch (`scripts/check_leaderboard_docs.py` enforces). Numeric report, not the ship picker (ADR 0014).
- [`pareto-selection.md`](./pareto-selection.md) — method background: scalarization citations (maximin / ε-constraint); Night ctx floor + Day TPS floor rules are **historical lenses** demoted by [ADR 0017](../adr/0017-rank-membership-quality-first.md); report lens only under ADR 0014 — not the ship picker.
- [`best-model-8gb-vram.md`](./best-model-8gb-vram.md) — web-sourced selection guide: fastest + smartest model fitting 8 GB VRAM (primary publisher cards only; no local measurements).
- [`../models/README.md`](../../models/README.md) — nested GGUF store shared with LM Studio.

### 2. Quantization & Low-VRAM Optimizations
- [`quantization-cascade.md`](./quantization-cascade.md) — quantization format selection (UD vs standard, VRAM tiers) + HF 4-bit vs GGUF footprint for cross-engine sizing.
- [`quantization-cascade-agent.md`](./quantization-cascade-agent.md) — agent quick reference for quant selection (terse, grog-readable).
- [`nvfp4-quantization.md`](./nvfp4-quantization.md) — NVIDIA NVFP4 4-bit FP format: structure, scaling, memory, native vs Marlin fallback, ecosystem, repo relevance.
- [`advanced-inference-optimizations.md`](./advanced-inference-optimizations.md) — high-performance techniques: CUDA graphs, tcmalloc/jemalloc, KV cache optimizations, and offload bottlenecks.
- [`low-vram-optimizations.md`](./low-vram-optimizations.md) — strategies for VRAM-constrained GPUs: GGUF/EXL2/HQQ quants, KV cache compression, MoE offloading, and preventing system paging.
- [`vitriol-technique.md`](./vitriol-technique.md) — stock `--n-cpu-moe` vs Randozart/VITRIOL DMA fork (study only; Search stays upstream).
- [`local-models-low-vram-configs.md`](./local-models-low-vram-configs.md) — optimal llama.cpp parameters for local and LM Studio models on 8 GB VRAM.

### 3. Inference Engines & Speculative Runtimes
- [`inference-engines-landscape.md`](./inference-engines-landscape.md) — technical comparison & taxonomy guide of LLM inference engines (vLLM, SGLang, TensorRT-LLM, LMDeploy, llama.cpp & fork ecosystem [KoboldCpp, alpaca.cpp, TurboQuant, BeeLlama], Colibrì, TGI).
- [`vllm-quantization.md`](./vllm-quantization.md) — vLLM quantization formats, hardware compatibility matrix, out-of-tree quant plugin API.
- [`vllm-quant-deep-dive.md`](./vllm-quant-deep-dive.md) — per-format quantization mechanics from vLLM source: kernels, group sizes, min SMs, NVFP4 status, config flags, repo relevance.
- [`colibri-inference-engine.md`](./colibri-inference-engine.md) — architectural & performance guide for Colibrì zero-dependency C streaming MoE runtime.
- [`sglang-inference-engine.md`](./sglang-inference-engine.md) — SGLang research guide: RadixAttention, scheduler, structured outputs (XGrammar), speculative decoding, quantization matrix, hardware support, GGUF/Windows limits.
- [`fastest-tps-inference-engine.md`](./fastest-tps-inference-engine.md) — fastest-TPS engine research on the operator host (discrete 8 GB-class NVIDIA / Win): llama.cpp CUDA baseline vs ExLlamaV3/EXL3, TRT-LLM, vLLM/SGLang; EXL3 3 bpw measured on-rig (49.6 vs 38.7/57.3), EXL3-MTP hardware-blocked (248K-vocab head quant >32 GB host RAM; hb16 pack fails 8 GB placement); MoE/MTP speed levers.
- [`speculative-decoding-formats.md`](./speculative-decoding-formats.md) — architectural and performance comparison of speculative formats (MTP vs Eagle vs DFlash vs N-gram).
- [`mtp-baseline-guide.md`](./mtp-baseline-guide.md) — guide on verifying and benchmarking MTP speculative decoding with llama-bench/llama-cli.
- [`small-model-mtp-tps.md`](./small-model-mtp-tps.md) — inventory of local MTP packaging + fair TPS matrix (8 GB, 2026-07-20).
- [`unsloth-qwen-guides.md`](./unsloth-qwen-guides.md) — reference guide on Unsloth dynamic quantization and Qwen fine-tuning mechanics.
- [`speculative-and-multi-gpu-tuning.md`](./speculative-and-multi-gpu-tuning.md) — Universal speculative decoding & multi-GPU playbook: temperature/MTP symbiosis, single-GPU vs layer-split penalty, multi-instance pinning, and reasoning effort controls across model families.
- [`frontier-class-local-ai-guide.md`](./frontier-class-local-ai-guide.md) — frontier-class local AI guide: Qwen3.8-Flash-Next architecture (176B MoE + 51B N-gram lookup table), hybrid GDN+QSA attention, physical-core thread optimization (13.6 vs 24.4 tok/s), and spec-wins adversarial evaluation.

### 4. CPU Inference & Build Optimization
- [`cpu-inference-guide.md`](./cpu-inference-guide.md) — CPU-optimized llama.cpp build flags (AVX-512/AMX), Intel vs AMD notes, NUMA, thread affinity, allocators (tcmalloc/jemalloc), GGUF quant selection for CPU cache, and weight repack (`--repack` / `-nr`) by profile.
- [`amd-rocm-windows-setup.md`](./amd-rocm-windows-setup.md) — AMD Radeon ROCm/HIP setup on Windows: HIP SDK install, PATH, ggml-hip.dll troubleshooting, RDNA 2/3 performance notes.
- [`openvino-genai-cpu-igpu-guide.md`](./openvino-genai-cpu-igpu-guide.md) — OpenVINO GenAI export, INT8/INT4 choices, CPU/iGPU device selection, and reproducible TPS benchmarking.
- [`cpu-only-runtime-tuning.md`](./cpu-only-runtime-tuning.md) — CPU-only runtime tuning: decode bandwidth ceiling, IQ-vs-speed knob classification, hybrid P+E thread split, batch-sweep methodology, and the PPL-guarded TPS hill-climb.
### 5. Harness & Capability Extraction
- [`prefix-cache-reuse.md`](./prefix-cache-reuse.md) — multi-turn KV-cache prefix reuse: `--cache-reuse` 256 already on in pinned build; ~15x prefill speedup on shared-prefix 2nd turn; wiring + A/B validation.
- [`capability-extraction-harness.md`](./capability-extraction-harness.md) — harness-side IQ-per-token techniques: led state, verification loops, test-time compute, J-Space assessment, dead-end registry.
- [`reasoning-levels-mapping.md`](./reasoning-levels-mapping.md) — reading reasoning ladders from embedded chat templates (gguf_dump recipe + tool false-negative caveat); template-gated vs engine levers; store reasoning columns + audit recipe.

