# AGENTS.md — docs/discovery

## Purpose
User-facing guides for **discovering, evaluating, selecting, and optimizing** local LLMs and inference runtimes that fit the `local-model-autotuning` workflow. Covers tooling (`whichllm`), evaluation methodology (Pareto frontier, Zellinger economic evaluation), quantization strategies, and engine architectures.

## Ownership
- Owned by: `local-model-autotuning` developers.
- Stable contracts: whichllm CLI contract, Pareto frontier method, scoring rules.

## Local Contracts
- **Read-mostly**: discovery docs are guides for users to follow. Code under `autoresearch/` is the loop surface.
- **No model-specific claims**: docs reference model families (e.g. "Qwen3.6 MoE") not private paths or single-user hardcoded values.
- **No external-source citations in technique claims**: methodology names allowed (Zellinger framework), but no HF/Unsloth/blog URLs in technique descriptions.

## Work Guidance
- New guides go here when they describe a reusable methodology, not a single session's data.
- Single-run session logs → `docs/sessions/` (cavity for empirical capture).
- Per-model GGUF specs and architecture notes → `docs/models/`.

## Verification
- Guides should be **runnable** by a user with the documented steps. Steps referencing a tool should include the install path (`uvx whichllm@latest`, etc).
- Cross-check examples against current `whichllm --help` before claiming a flag/option.

## Child DOX Index

### 1. Tooling, Onboarding & Hard Gates
- [`discover-models.md`](./discover-models.md) — end-to-end workflow: discovery (whichllm/llmfit) → Pareto frontier → autoloop target selection.
- [`whichllm-reference.md`](./whichllm-reference.md) — full whichllm CLI reference (commands, flags, profiles, examples).
- [`llmfit-reference.md`](./llmfit-reference.md) — full llmfit CLI/TUI reference (hardware sizing, planning, model search, examples).
- [`agent-onboarding.md`](./agent-onboarding.md) — onboarding guide for future agents.
- [`agentic-coding-benchmarks.md`](./agentic-coding-benchmarks.md) — migration guide from direct coding tasks to long-horizon agentic coding benchmarks.
- [`agent-shell-hard-gates.md`](./agent-shell-hard-gates.md) — live gate inventory, disable/rollback playbook (§3), threat model (Cursor + Claude Code).
- [`../models/README.md`](../../models/README.md) — nested GGUF store shared with LM Studio.

### 2. Quantization & Low-VRAM Optimizations
- [`quantization-cascade.md`](./quantization-cascade.md) — quantization format selection guide (UD vs standard, VRAM tiers, decision matrix).
- [`quantization-cascade-agent.md`](./quantization-cascade-agent.md) — agent quick reference for quant selection (terse, grog-readable).
- [`advanced-inference-optimizations.md`](./advanced-inference-optimizations.md) — high-performance techniques: CUDA graphs, tcmalloc/jemalloc, KV cache optimizations, and offload bottlenecks.
- [`low-vram-optimizations.md`](./low-vram-optimizations.md) — strategies for VRAM-constrained GPUs: GGUF/EXL2/HQQ quants, KV cache compression, MoE offloading, and preventing system paging.
- [`local-models-low-vram-configs.md`](./local-models-low-vram-configs.md) — optimal llama.cpp parameters for local and LM Studio models on 8 GB VRAM.

### 3. Inference Engines & Speculative Runtimes
- [`inference-engines-landscape.md`](./inference-engines-landscape.md) — technical comparison & taxonomy guide of LLM inference engines (vLLM, SGLang, TensorRT-LLM, LMDeploy, llama.cpp, Colibrì, TGI).
- [`colibri-inference-engine.md`](./colibri-inference-engine.md) — architectural & performance guide for Colibrì zero-dependency C streaming MoE runtime.
- [`speculative-decoding-formats.md`](./speculative-decoding-formats.md) — architectural and performance comparison of speculative formats (MTP vs Eagle vs DFlash vs N-gram).
- [`mtp-baseline-guide.md`](./mtp-baseline-guide.md) — guide on verifying and benchmarking MTP speculative decoding with llama-bench/llama-cli.
- [`small-model-mtp-tps.md`](./small-model-mtp-tps.md) — inventory of local MTP packaging + fair TPS matrix (8 GB, 2026-07-20).
- [`unsloth-qwen-guides.md`](./unsloth-qwen-guides.md) — reference guide on Unsloth dynamic quantization and Qwen fine-tuning mechanics.
