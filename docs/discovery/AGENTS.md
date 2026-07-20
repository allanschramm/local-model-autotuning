# AGENTS.md — docs/discovery

## Purpose
User-facing guides for **discovering, evaluating, and selecting** local LLMs that fit the `local-model-autotuning` workflow. Covers tooling (`whichllm`), evaluation methodology (Pareto frontier, Zellinger economic evaluation), and integration with the autoloop.

## Ownership
- Owned by: `local-model-autotuning` developers.
- Stable contracts: whichllm CLI contract, Pareto frontier method, scoring rules.

## Local Contracts
- **Read-mostly**: discovery docs are guides for users to follow. Code under `autoresearch/` is the loop surface.
- **No model-specific claims**: docs reference model families (e.g. "Qwen3.6 MoE") not private paths or single-user hardcoded values.
- **No external-source citations in technique claims**: methodology names allowed (Zellinger framework), but no HF/HF/Unsloth/blog URLs in technique descriptions.

## Work Guidance
- New guides go here when they describe a reusable methodology, not a single session's data.
- Single-run session logs → `docs/sessions/` (cavity for empirical capture).
- Per-model GGUF specs and architecture notes → `docs/models/`.

## Verification
- Guides should be **runnable** by a user with the documented steps. Steps referencing a tool should include the install path (`uvx whichllm@latest`, etc).
- Cross-check examples against current `whichllm --help` before claiming a flag/option.

## Child DOX Index
- [`discover-models.md`](./discover-models.md) — end-to-end workflow: whichllm discovery → Pareto frontier → autoloop target selection.
- [`whichllm-reference.md`](./whichllm-reference.md) — full whichllm CLI reference (commands, flags, profiles, examples).
- [`quantization-cascade.md`](./quantization-cascade.md) — quantization format selection guide (UD vs standard, VRAM tiers, decision matrix).
- [`quantization-cascade-agent.md`](./quantization-cascade-agent.md) — agent quick reference for quant selection (terse, grog-readable).
- [`agent-onboarding.md`](./agent-onboarding.md) — onboarding guide for future agents.
- [`agentic-coding-benchmarks.md`](./agentic-coding-benchmarks.md) — migration guide from direct coding tasks to long-horizon agentic coding benchmarks.
- [`mtp-baseline-guide.md`](./mtp-baseline-guide.md) — guide on verifying and benchmarking MTP speculative decoding with llama-bench/llama-cli.
- [`small-model-mtp-tps.md`](./small-model-mtp-tps.md) — inventory of local MTP packaging + fair TPS matrix (8 GB, 2026-07-20).
- [`speculative-decoding-formats.md`](./speculative-decoding-formats.md) — architectural and performance comparison of speculative formats (MTP vs Eagle vs DFlash vs N-gram).
- [`advanced-inference-optimizations.md`](./advanced-inference-optimizations.md) — high-performance techniques: CUDA graphs, tcmalloc/jemalloc, KV cache optimizations, and offload bottlenecks (Fast Gemma Challenge lessons).
- [`low-vram-optimizations.md`](./low-vram-optimizations.md) — strategies for VRAM-constrained GPUs: GGUF/EXL2/HQQ quants, KV cache compression, MoE offloading, and preventing system paging.
