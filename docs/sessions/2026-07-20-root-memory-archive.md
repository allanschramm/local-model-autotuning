# Root MEMORY.md archive (2026-07-20)

## Goal
Preserve empirical notes formerly kept in root `MEMORY.md` before deleting that file. Canonical runtime history remains `results.tsv`. Stale project-context / Val Score weight claims from MEMORY were **not** copied (superseded by Claw-Eval agentic-first Search — see ADR 0004).

## Hardware
RTX 4060 8GB (operator notes from original MEMORY).

## Setup
N/A — archival pass, not a new Trial session.

## Commands
N/A.

## Findings

### Experiment log (best per model, as recorded in MEMORY)

| Commit | Model | KV | Ctx | VRAM (GB) | TPS | Score | Status |
|---|---|---|---|---|---|---|---|
| `bab91e9` | Qwen3.5-9B-MTP | q4_0 | 32768 | 6.9 | 157.3 | 0.772067 | Keep |
| `0c8e5fc` | gemma4-v2 | turbo4 | 65536 | 8.0 | 29.5 | 0.312922 | Keep |
| `a06b9f9` | Ornith-1.0-9B | q4_0 | 131072 | 7.4 | 52.2 | 0.580000 | Keep |
| `d5ddbd1` | Ornith-1.0-35B | q4_0 | 131072 | 7.7 | 31.5 | 0.555000 | Keep |

### Working configs (as recorded)

* **Qwen3.5-9B-MTP**: `kv=q4_0`, `ctx=32768`, `threads=12` → **0.7721** (coding=0.8167, retrieval=0.4150)
* **gemma4-v2**: `kv=turbo4 K+V`, `ctx=65536`, `threads=8`, `temp=0.4`, `batch=128/ubatch=64` → **0.3129** (coding=0.3333, retrieval=0.1499)
* **Ornith-1.0-9B**: `kv=q4_0`, `ctx=131072`, `threads=8`, `temp=0.4` → **0.5800** (coding=0.5800: lcb=0.40, he=0.80, mbpp=0.90, bigcode=0.10)
* **Ornith-1.0-35B**: `kv=q4_0`, `ctx=131072`, `threads=8`, `temp=0.4`, `n-cpu-moe=32` (VITRIOL) → **0.5550** (coding=0.5550: lcb=0.40, he=0.80, mbpp=0.80, bigcode=0.10)

### Blocked configurations

* **gemma4-v2 temp=0.2 + batch=512/256 + repeat_penalty=1.05**: score dropped to 0.1648. MBPP timeout on task 1/3.
* **gemma4-v2 ctx below 65536 with Nexus**: 50K token padding breaks context. Nexus requires ctx at least 65536.
* **Low Throughput**: g4-opt-it with q4_0 at 16k → 24.3 TPS (below then-threshold 30.0). Score zeroed. (Current TPS Floor is 20.0 — see `GOLDEN-RULES.md`.)
* **VRAM Limits**: Models over 9B with f16 KV at 64k+ exceed 7.9 GB. Use turbo4.

### Discovered durable knowledge

* **Turbo4 Gemma4 12B dense**: VRAM flat at ~7.9GB up to 128K ctx thanks to aggressive TurboQuant. KV cache cost amortized by nvidia-smi 10Hz sampling.
* **MTP absent in Gemma4**: Model lacks `nextn_predict_layers`. Speculative decoding doesn't help.
* **Threads**: 8 threads better than 12 on Gemma4 12B (unregistered baseline issue with 12).
* **Batch sizing**: batch=128, ubatch=64 more stable on Gemma4 than 512/256 under short trial budget.
* **Indentation in reasoning models (Ornith/Qwen 3.5)**: `<think>` blocks generate 8+ space indented drafts (inside markdown lists). Truncated output → fallback parser carries extra indentation → IndentationError when concatenating signature. `textwrap.dedent()` + re-indent with exactly 4 spaces fixes parser and doubles HE+ accuracy (0.40 → 0.80 on Ornith 9B).
* **VITRIOL MoE Streaming 35B**: Keeping routed experts in CPU/RAM via `--n-cpu-moe 40` enables 35B model loading on 8GB VRAM (RTX 4060) with acceptable throughput (23.6 TPS) and peak VRAM only 4.1 GB.
* **UBATCH sweet spot**: 256 beats 512 on RTX 4060 8GB for Gemma4. llama-bench: ub=256 pp1922 tg49.8 vs ub=512 pp1940 tg41.0. Higher ubatch = better prompt processing but worse token generation.
* **Category column in results.tsv** (2026-07-01): Added to enable fair cross-model comparisons.
* **Docs path discipline** (2026-07-01): User-facing docs must not contain hardcoded paths (`/home/<user>/...`). Use `./llama.cpp/`, `$LLAMA_CPP`, or `AUTORESEARCH_LLAMA_CPP_ROOT`.

### Gotchas

* `--n-cpu-moe` value matters: 32 works for 35B MoE; too high wastes RAM without benefit on dense models.
* `config.py` write helpers preserve section headers and comments — safe to call repeatedly.
* GEN_KWARGS filtered to non-None values only before passing to llama-server.
* Never run `llama-server` or `llama-bench` directly — harness manages paths, lifecycle, VRAM monitoring, result logging.

## Errors
None this pass. Original MEMORY also held stale Val Score weights (HE+/MBPP+/LCB/BCB) and an outdated validation description (2 tasks/dataset) — omitted on purpose.

## Decisions
* Root `MEMORY.md` deleted after this archive.
* Living contracts stay in `AGENTS.md`, `program.md`, `CONTEXT.md`, `GOLDEN-RULES.md`, `docs/adr/`.
* Live trial scores: `results.tsv` only.
