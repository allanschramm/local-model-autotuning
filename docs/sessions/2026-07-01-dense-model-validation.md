# Dense Model Validation — 2026-07-01

## Goal
Validate all dense (non-MoE, non-MTP) GGUF models in `models/` with a consistent config and produce a cross-model comparison.

## Hardware
- GPU: NVIDIA RTX 4060 8 GB (8188 MiB total, 7149 MiB free at start)
- CPU: 8 cores
- Build: `llama.cpp/build-cuda` at `0eca4d490` (upstream ggml-org)

## Setup
- Config: `ctx=131072`, `kv=q4_0`, `b=1024`, `ub=256`, `t=8`, `tb=8`, `fa=on`, `cont-batching`, `ngl=99`
- All generation defaults from `config.py` (TEMP=0.4, TOP_P=0.95, TOP_K=20, REPEAT_PENALTY=1.05)
- Validation mode: `--validation` → llama-bench (pp512/tg128) → 2-task coding eval (LCB + HE+ + MBPP+ + BigCode H)
- Runs sequential, 1 model at a time. No timeouts set.

## Models Validated

### Excluded (not dense)
- `models/*-MTP-*.gguf` — speculative draft models
- `models/mtp-gemma-4-12B-it.gguf` — speculative draft
- `models/MTP/`, `models/DFLASH/`, `models/aliases/` — subdirs (draft, MoE flash-attn, launcher configs)
- `ornith-1.0-35b-*` — MoE architecture
- `deepreinforce-ai_Ornith-1.0-35B-IQ3_M.gguf` — MoE architecture

### Included (4 dense models)
| # | Model | File Size |
|---|-------|-----------|
| 1 | `gemma-4-12B-it-qat-UD-Q4_K_XL.gguf` | 6.4 GB |
| 2 | `gemma-4-12B-fable5-Q3_K_M.gguf` | 5.8 GB |
| 3 | `Qwythos-9B-Claude-Mythos-5-1M-Q4_K_M.gguf` | 5.4 GB |
| 4 | `ornith-1.0-9b-Q4_K_M.gguf` | 5.4 GB |

## Commands

```bash
# Each model run individually:
python3 benchmark_search.py --model <model.gguf> --validation --desc "validation <model.gguf>"
```

## Results

### Full comparison

| Model | Score | LCB | HE+ | MBPP+ | BigCode | TPS | tg t/s | VRAM |
|---|---|---|---|---|---|---|---|---|
| **gemma-4-12B-fable5-Q3_K_M** | **0.5500** | 0.50 | **1.00** | 0.50 | 0.00 | **43.0** | 31.2 | 7.4 GB |
| **ornith-1.0-9b-Q4_K_M** | **0.5500** | 0.50 | **1.00** | 0.50 | 0.00 | **50.2** | 43.7 | 7.1 GB |
| gemma-4-12B-it-qat-UD-Q4_K_XL | 0.4250 | 0.50 | 0.50 | 0.50 | 0.00 | 36.0 | 33.4 | 8.0 GB |
| Qwythos-9B-Claude-Mythos-5-1M-Q4_K_M | 0.3000 | 0.50 | 0.50 | 0.00 | 0.00 | 50.4 | 43.7 | 7.1 GB |

### Per-run details

#### 1. gemma-4-12B-it-qat-UD-Q4_K_XL.gguf
- llama-bench: tg 33.4 t/s → PASS (≥20 threshold)
- Coding: HE+ 0.5000, MBPP+ 0.5000, LCB 0.5000, BigCode 0.0000
- Score: 0.4250 (0.35×0.5 + 0.25×0.5 + 0.25×0.5 + 0.15×0.0)
- TPS: 36.0, VRAM: 8.0 GB (maxed)
- Status: DISCARD (prev best 0.550)

#### 2. gemma-4-12B-fable5-Q3_K_M.gguf
- llama-bench: tg 31.2 t/s → PASS
- Coding: HE+ **1.0000**, MBPP+ 0.5000, LCB 0.5000, BigCode 0.0000
- Score: 0.5500
- TPS: 43.0, VRAM: 7.4 GB
- Status: KEEP (first validation for this filename)

#### 3. Qwythos-9B-Claude-Mythos-5-1M-Q4_K_M.gguf
- llama-bench: tg **43.7** t/s → PASS (fastest bench tg)
- Coding: HE+ 0.5000, MBPP+ **0.0000**, LCB 0.5000, BigCode 0.0000
- Score: 0.3000 (MBPP+ collapse)
- TPS: 50.4, VRAM: 7.1 GB
- Status: DISCARD (prev best 0.425)

#### 4. ornith-1.0-9b-Q4_K_M.gguf
- llama-bench: tg 43.7 t/s → PASS
- Coding: HE+ **1.0000**, MBPP+ 0.5000, LCB 0.5000, BigCode 0.0000
- Score: 0.5500
- TPS: 50.2, VRAM: 7.1 GB
- Status: DISCARD (prev best 0.580 from 10-task)

## Findings

### Score ranking (2-task validation)
1. Gemma Fable5 / Ornith 9B — tied at **0.5500**
2. Gemma QAT — **0.4250**
3. Qwythos 9B — **0.3000**

### Speed ranking
1. Qwythos / Ornith — **~50 TPS** (bench tg 43.7)
2. Gemma Fable5 — **43 TPS** (bench tg 31.2 — Q3 is slower kernel)
3. Gemma QAT — **36 TPS** (bench tg 33.4 — larger quantization overhead)

### VRAM ranking
1. Qwythos / Ornith — **7.1 GB** (most headroom)
2. Gemma Fable5 — **7.4 GB**
3. Gemma QAT — **8.0 GB** (pegged)

### Notable
- **BigCode Hard 0/4 across all models** — library-call tasks fail at small sample size (2 tasks) for all models. Not statistically meaningful at this sample size.
- **Qwythos MBPP+ 0/2** — unusual for a 9B model. Possibly a sampling fluke at 2-task validation (high variance).
- **Gemma QAT VRAM pegged at 8.0 GB** — 131k ctx with Q4_K_XL saturates RTX 4060 completely.

## Errors
None — all 4 models passed llama-bench threshold and completed eval cleanly.

## Decisions
- 35B MoE models excluded per user direction
- No hyperparam tuning — validation mode uses config.py defaults
- 2-task validation has high variance (especially BigCode at 2 tasks) — treat absolute scores as directional, not definitive

## Cross-links
- [Gemma-4-12B model card](../models/gemma-4-12b.md)
- [Qwythos model card](../models/qwythos-9b-claude-mythos-5-1m.md)
- [Ornith-1.0-9B model card](../models/ornith-1.0-9b.md)
- [Session: Gemma4 v2/Q3_K_M validation](2026-07-01-gemma4-v2-q3km-validation.md)
