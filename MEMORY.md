# Project Memory — local-model-autotuning

## Project Context

Autonomous hill-climbing optimizer for local LLM runtime flags (KV cache quant, threads, batching, MTP, CPU offload). Forked from karpathy/autoresearch. Targets RTX 4060 8GB. Evaluates configs via coding benchmarks (HE+, MBPP+, LCB, BigCodeBench).

**Core loop**: Load baseline from config.py → run all benchmarks → compute Val Score → mutate 1 param → evaluate neighbor → keep if better → random restart on stagnation → repeat until Ctrl+C.

**Val Score weights**: HumanEval+ 25%, MBPP+ 25%, LiveCodeBench 35%, BigCodeBench Hard 15%. TPS floor = 20 tok/s (score zeroed below).

## Rules

- config.py is the ONLY mutable surface for agent tweaks
- CTX_SIZE frozen at 131072. Min ctx floor = 100k. Never lower.
- Flash attention always on
- Never push results, tweaks, or run branches to remote (local-only)
- Loop agents: never edit code; on error → stop, report, warn user
- Always run harness (`benchmark_search.py` or `autoloop.py`), never raw `llama-server`/`llama-bench`
- README.md must always be in pt-BR. Agent-facing docs (docs/, AGENTS.md, GOLDEN-RULES.md, CONTEXT.md, program.md) stay in English [ses_0e12c7774ffe]

## Architecture Decisions

- **Single mutable surface**: Only `autoresearch/core/config.py` gets edited by agents/harness. All other files are fixed. Rationale: prevents gaming evaluation.
- **Validation 2-step**: llama-bench speed check (512/128, 3 repeats) + quick coding eval (2 tasks/dataset). TPS < 20 → FAIL. `--validation` flag runs steps 1-2 then exits.
- **ExperimentRunner extraction** (2026-06-30): Consolidated trial orchestration into `autoresearch/runners/evaluation.py`. ServerIntent.from_config() normalizes config before trial.
- **GenerationParams dataclass** (2026-07-01): Replaced 5-file **kwargs passthrough chain with typed dataclass. Cleaner API surface.
- **Quantization cascade docs** (2026-07-01): Created `docs/discovery/quantization-cascade.md` (user guide) and `quantization-cascade-agent.md` (agent reference) for quant selection methodology.
- **Human/agent doc split** (2026-07-01): README.md is for humans (concise, pt-BR, quickstart). docs/ is for agents (detailed, English, structured references). Humans paste a prompt to their coding agent; agents read program.md, GOLDEN-RULES.md, CONTEXT.md, and docs/ [ses_0e12c7774ffe].

## Experiment Log (Best per model)

| Commit | Model | KV | Ctx | VRAM (GB) | TPS | Score | Status |
|---|---|---|---|---|---|---|---|
| `bab91e9` | Qwen3.5-9B-MTP | q4_0 | 32768 | 6.9 | 157.3 | 0.772067 | Keep |
| `0c8e5fc` | gemma4-v2 | turbo4 | 65536 | 8.0 | 29.5 | 0.312922 | Keep |
| `a06b9f9` | Ornith-1.0-9B | q4_0 | 131072 | 7.4 | 52.2 | 0.580000 | Keep |
| `d5ddbd1` | Ornith-1.0-35B | q4_0 | 131072 | 7.7 | 31.5 | 0.555000 | Keep |

## Working Configs

*   **Qwen3.5-9B-MTP**: `kv=q4_0`, `ctx=32768`, `threads=12` → **0.7721** (coding=0.8167, retrieval=0.4150)
*   **gemma4-v2**: `kv=turbo4 K+V`, `ctx=65536`, `threads=8`, `temp=0.4`, `batch=128/ubatch=64` → **0.3129** (coding=0.3333, retrieval=0.1499)
*   **Ornith-1.0-9B**: `kv=q4_0`, `ctx=131072`, `threads=8`, `temp=0.4` → **0.5800** (coding=0.5800: lcb=0.40, he=0.80, mbpp=0.90, bigcode=0.10)
*   **Ornith-1.0-35B**: `kv=q4_0`, `ctx=131072`, `threads=8`, `temp=0.4`, `n-cpu-moe=32` (VITRIOL) → **0.5550** (coding=0.5550: lcb=0.40, he=0.80, mbpp=0.80, bigcode=0.10)

## Blocked Configurations

*   **gemma4-v2 temp=0.2 + batch=512/256 + repeat_penalty=1.05**: score dropped to 0.1648. MBPP timeout on task 1/3.
*   **gemma4-v2 ctx<65536 with Nexus**: 50K token padding breaks context. Nexus requires ctx>=65536.
*   **Low Throughput**: g4-opt-it with q4_0 at 16k → 24.3 TPS (below threshold 30.0). Score zeroed.
*   **VRAM Limits**: Models >9B with f16 KV at 64k+ exceed 7.9 GB. Use turbo4.

## Discovered Durable Knowledge

*   **Turbo4 Gemma4 12B dense**: VRAM flat at ~7.9GB up to 128K ctx thanks to aggressive TurboQuant. KV cache cost amortized by nvidia-smi 10Hz sampling.
*   **MTP absent in Gemma4**: Model lacks `nextn_predict_layers`. Speculative decoding doesn't help.
*   **Threads**: 8 threads better than 12 on Gemma4 12B (unregistered baseline issue with 12).
*   **Batch sizing**: batch=128, ubatch=64 more stable on Gemma4 than 512/256 under short trial budget.
*   **Indentation in reasoning models (Ornith/Qwen 3.5)**: `<think>` blocks generate 8+ space indented drafts (inside markdown lists). Truncated output → fallback parser carries extra indentation → IndentationError when concatenating signature. `textwrap.dedent()` + re-indent with exactly 4 spaces fixes parser and doubles HE+ accuracy (0.40 → 0.80 on Ornith 9B).
*   **VITRIOL MoE Streaming 35B**: Keeping all 256 routed experts in CPU/RAM via `--n-cpu-moe 40` enables 35B model loading on 8GB VRAM (RTX 4060) with acceptable throughput (23.6 TPS) and peak VRAM only 4.1 GB.
*   **UBATCH sweet spot**: 256 > 512 on RTX 4060 8GB for Gemma4. llama-bench: ub=256 pp1922 tg49.8 vs ub=512 pp1940 tg41.0. Higher ubatch = better prompt processing but worse token generation.
*   **Category column in results.tsv** (2026-07-01): Added to enable fair cross-model comparisons.
*   **Docs path discipline** (2026-07-01): User-facing docs must not contain hardcoded paths (`/home/shark/...`). Use `./llama.cpp/`, `$LLAMA_CPP`, or `AUTORESEARCH_LLAMA_CPP_ROOT` env var [ses_0e12c7774ffe].

## Patterns

- **Validation pipeline evolution**: llama-bench integration → quick coding checks after bench → lowered TPS threshold. Iterative stabilization over 2026-06-28 to 2026-07-01.
- **Skill cleanup**: Deprecated grill-me, grill-with-docs, GitNexus skills removed. Skills lockfile pruned. Leaner skill set: subagent, debug, tdd, execute, report, review, feedback, plan, worktree, verify, ask, merge, brainstorm, parallel, new-skill.

## Gotchas

- `--n-cpu-moe` value matters: 32 works for 35B MoE, but too high wastes RAM without benefit on dense models
- config.py write_config() preserves section headers and comments — safe to call repeatedly
- GEN_KWARGS filtered to non-None values only before passing to llama-server
- Never run llama-server or llama-bench directly — harness manages paths, lifecycle, VRAM monitoring, result logging
