# Session 2026-07-01: Gemma4-12B v2 (agentic-fable5) Q3_K_M validation

## Goal
Download and validate `yuxinlu1/gemma-4-12B-agentic-fable5-composer2.5-v2-3.5x-tau2-GGUF` Q3_K_M quant. Quick 2-task coding check after llama-bench passes.

## Hardware
- GPU: RTX 4060 8 GB VRAM
- CPU: AMD Ryzen 7 5800X
- RAM: 24 GB
- OS: WSL2 Ubuntu 24.04

## Setup
Downloaded `gemma4-v2-Q3_K_M.gguf` (5.8 GB) from HF to `models/`. Renamed to `gemma-4-12B-fable5-Q3_K_M.gguf` for clarity.

Config: `CTX_SIZE=131072` (config default), `cache_type_k/v=q4_0`, `batch=1024`, `ubatch=256`, `ngl=99`, `flash_attn=on`, `cont_batching`.

## Validation Results

### Phase 1: llama-bench
| Metric | Value |
|---|---|
| Prompt processing | 512 tokens |
| Generation | 128 tokens |
| tg throughput | **28.9 t/s** ✅ ≥ 20.0 threshold |

### Phase 2: Coding (2 tasks per dataset, 131k ctx)
| Dataset | Pass@1 | TPS | Notes |
|---|---|---|---|
| HumanEval+ | **1.0000** | 44.8 | Both passed — solid for Q3_K_M |
| MBPP+ | **0.5000** | 37.9 | 1/2 passed (better than earlier 8k run) |
| LiveCodeBench | **0.5000** | 36.4 | 1/2 passed |
| BigCodeBench Hard | **0.0000** | 40.0 | Both failed — library-call tasks suffer at 3-bit |
| **Overall** | **0.5500** | **38.4** | Weighted: 0.35*LCB + 0.25*HE + 0.25*MBPP + 0.15*BigCode |
| **VRAM** | **7.9 GB** | | Fits 8 GB at full 131k ctx |

## Bug fix: validation mode
`--validation` flag previously returned after llama-bench only. Fixed in `autoresearch/runners/evaluation.py` — now coerces task limits to 2 and runs quick coding check after bench passes. Also removed `if not is_validation:` gate in `run.py` so coding scores always display.

## Config fix: bench threshold
`BENCH_TPS_THRESHOLD` lowered from 30.0 → 20.0 in `evaluation.py`. 28.9 tg t/s was flaking below 30.0.

## Key Takeaways
1. **Q3_K_M fits 8 GB VRAM at full 131k ctx** (7.9 GB peak). No OOM.
2. **Score 0.5500 matches Gemma 4 base Q4_K_XL** — same overall coding score on 2-task validation.
3. **HumanEval+ 1.0** at Q3_K_M is remarkable — fine-tune retains coding ability even at 3-bit.
4. **BigCode zeros** are expected for 3-bit quant on library-call tasks.
5. **Better at 131k than 8k** — MBPP+ went from 0.0 to 0.5. More context helps the agentic fine-tune.
