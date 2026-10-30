# Bonsai-27B — Model Card (Local)

Runtime scope: quality results remain in the [archived full card](https://github.com/allanschramm/local-model-autotuning/blob/ea43a71f48571811ec1ba25df4ab71ed272d34f1/docs/models/bonsai-27b.md). Historical performance retains its original protocol.

**Source repo:** https://huggingface.co/PrismML-Eng/Bonsai-27B (models were downloaded from here; see note)
**Canonical card repo:** https://huggingface.co/prism-ml/Bonsai-27B-gguf (**PRIVATE** — 27B repos are gated until launch)
**Run guide:** https://github.com/PrismML-Eng/Bonsai-demo (README.md + SPECULATIVE.md)
**License:** Apache 2.0 (HF `prism-ml/Bonsai-27B-gguf` 2026-08-29); original `PrismML-Eng/Bonsai-27B` source repo was private/gated at time of original download (card retains that historical note)
**Local file:** `models/Bonsai-27B-Q1_0.gguf` (3627.3 MB / 3.53 GiB) — **PURGED from local store; file no longer present (historical card; retain specs from 2026-08-02 GGUF-verified header + 2026-08-29 HF refresh)
**Drafter file:** `models/draft/Bonsai-27B-dspark-Q4_1.gguf` (1704.7 MB) — DSpark speculative drafter (target-specific)
**Family:** Bonsai (PrismML-Eng) — 1-bit ternary-pack
**Quantization:** `Q1_0` (1-bit; ~1.125 bpw)

## Architecture (GGUF-verified 2026-08-02, local `Bonsai-27B-Q1_0.gguf`)
- **Base: Qwen3.5** — `general.architecture='qwen35'`; PrismML's "Qwen3.6-derived" marketing is contradicted by the actual file.
- **Dense, not MoE** — no `expert_count`/`expert_used_count` keys present. Treat as dense, no `--n-cpu-moe`.
- **Hybrid attention:** `full_attention_interval=4` over `block_count=64` → 16 full-attention + 48 linear-attention layers (matches the run-guide "16/64 layers cached"); `ssm.conv_kernel=4, state_size=128, group_count=16, time_step_rank=48, inner_size=6144`.
- **Header values:** `block_count=64`, hidden `5120`, `feed_forward_length=17408`, `attention.head_count=24`, `head_count_kv=4`, key/value length 256, `rope.freq_base=1e7`, ctx 262,144**, 851 tensors, `general.file_type=40`.
- **Embedded recommended sampling:** `TEMP=1.0, TOP_P=0.95, TOP_K=20` (GGUF `general.sampling.*`).
- Context length: 262,144 tokens** — GGUF-verified (`qwen35.context_length = 262144`).
- 27B family is vision-language: accepts images (vision projector +~0.9 GiB when used)

## Hardware Requirements (discrete 8 GB-class NVIDIA)
| Item | Size | Note |
| --- | --- | --- |
| Bonsai-27B Q1_0 weights | 3.53 GiB | 1-bit |
| FP16 KV @ 65K ctx | ~1.1 GiB | hybrid attn (16/64 layers cached) |
| FP16 KV @ 100K ctx | ~6.3 GiB | from run-guide table |
| DSpark drafter extra | ~0.5–2.4 GiB | embeds/head shared with target; naive loader estimate 2426 MiB |
| Runtime overhead | ~0.7–1.2 GiB | +vision projector ~0.9 GiB if used |

**KV sizing (PrismML KV-CACHE guide, 2026-08-02):** FP16 KV ≈ 64 KiB/token (~6.3 GiB @ 100K); experimental 4-bit KV (`--cache-type-k q4_0 --cache-type-v q4_0`) ≈ 18 KiB/token. Footprints: Q1_0 ≈ 4.8 GiB @ 4K / 5.2 GiB @ 10K / 10.8 GiB @ 100K (weights + activations + FP16 KV).

**Verified fit:** target + 65K KV cache loaded and warmed up on 8 GB-class discrete NVIDIA with no OOM (server log).
Full 262K KV (~4.3 GiB) would NOT fit (≈9.3 GiB > 8 GiB) — cap context well below train max.

## Recommended Settings (from SPECULATIVE.md, "tested working on CUDA")
Single deterministic command:
```
llama-server -m models/Bonsai-27B-Q1_0.gguf \
  -md models/draft/Bonsai-27B-dspark-Q4_1.gguf \
  --spec-type draft-dspark --spec-draft-n-max 4 \
  -ngl 999 -ngld 999 -fa on -c 65536 -np 1 \
  --host 127.0.0.1 --port 8080
```
Flag notes (from run guide):
- `--spec-type draft-dspark` + `--spec-draft-n-max 4` — **required; `n-max` must equal drafter block size (4). A smaller value crashes at the first draft round.
- `-ngl 999 -ngld 999` — target and drafter fully on GPU (model has ≤64 layers, so `99` also covers all).
- `-fa on` — flash attention.
- `-c 16384+` — guide recommends ≥16384 (model thinks 1.5–2k tokens before the visible answer); user tested 65536.
- `-np 1` — single slot (speculative disables cross-request prompt-cache; one request at a time).

## Speculative Decoding (DSpark)
- Drafter is **target-specific**: `Bonsai-27B-dspark-Q4_1.gguf` only accelerates `Bonsai-27B-Q1_0.gguf` (Ternary drafter only the Ternary target).
- ~1.8–2× faster decode on CUDA code/reasoning workloads; output at temp 0 identical to normal decode.
- Verify engaged via API `timings.draft_n` / `draft_n_accepted` (missing/zero = not active).
- `llama-cli` has no draft support; one-shot binary is `llama-speculative-simple`.
- Published decode on datacenter GPUs (PrismML, 2026-08-02): ~70 → ~135 tok/s at ~0.9 acceptance. On the operator host the canonical drafter measured 39.2 t/s (2026-07-21) — still below target-only ~41, so max-TPS stays `SPEC_TYPE=None`.

## Reasoning Control
- Publisher evaluates in **thinking mode** (thinking mode used for all benchmark results on HF `prism-ml/Bonsai-27B-gguf`, 2026-08-29).
- Chat template (verified from HF GGUF README, `prism-ml/Bonsai-27B-gguf` 2026-08-29) reads `enable_thinking` — same template family as Qwen3.5/Qwen3.6. Template: `{% if enable_thinking is defined and enable_thinking is false %}` renders empty reasoning block; default renders `<think>\\n\\n</think>\\n\\n` — **thinking ON by default**.
- `--reasoning-effort` is a silent no-op on this template family; use `--reasoning on/off` (Baseline REASONING) and `--reasoning-budget N` (REASONING_BUDGET) to control depth.
- `REASONING_PRESERVE` (server): TBD — not verified from `/props` on this GGUF; local template probe not run (GGUF purged).

## Fork Requirement
- Q2_0 and DSpark need the external **PrismML-Eng/llama.cpp** fork, branch `prism` (tested commit `9fcaed763` = tag `prism-b9596`), built with CUDA. The fork is not vendored in this repository.
- **No Windows CUDA prebuilt binary** exists in the fork releases (only Windows CPU). Windows CUDA must build from `prism` source.
- `Q1_0` is also merged upstream llama.cpp; `Q2_0` ternary needs the fork on CUDA.

## ⚠️ Known Issue — DSpark drafter fails to load
Server log with the exact command above (pre-2026-07-21 drafter):
```
loading draft model 'models/draft/Bonsai-27B-dspark-Q4_1.gguf'
W load: adding 248320 dummy tokens
E llama_model_load: error loading model: invalid vector subscript
```
- Crash is at **drafter model load** (before any draft round) — the command matches the run guide exactly, so this is not a flag error.
- `"adding 248320 dummy tokens"` = drafter GGUF `vocab_size` read as ~248320 (real vocab ≈152K); loader pads then indexes out of bounds → `invalid vector subscript`. Points to a **drafter GGUF / build version mismatch**.
- **Suspected cause:** earlier drafter from `PrismML-Eng/Bonsai-27B` vs canonical PRIVATE `prism-ml/Bonsai-27B-gguf`.
- **2026-07-21:** re-pulled `Bonsai-27B-dspark-Q4_1.gguf` from `prism-ml/Bonsai-27B-gguf` → `models/draft/` (SHA256 `25E73F9F…BEFB1B`, 1787468768 bytes).
- **2026-07-21 Trial (PrismML, dense VRAM guard):** `ctx=65536` `KV=q4_0` `--spec-type draft-dspark` `--spec-draft-n-max 4` `-ngl/-ngld 99`. Preflight est=7066MB ≤7900. **bench_tg=39.2 t/s**, peak VRAM **7.6 GB**, no kill/OOM. Loads OK with canonical drafter. Still **slower than target-only ~41 t/s** → keep max-TPS remains `SPEC_TYPE=None`.

## VITRIOL (MoE split)
**Dense** — do not apply `--n-cpu-moe` / shared-memory offload. Fit in physical VRAM only (dense VRAM guard).

## Max TPS (discrete 8 GB-class NVIDIA, ctx band 65k–131k) — 2026-07-21


| Config | Binary | ctx | KV | bench_tg | Peak VRAM |
| --- | --- | --- | --- | --- | --- |
| Q1_0 target-only | upstream CUDA | 65536 | q4_0 | **40.5** | 6.1 GB |
| Q1_0 target-only | upstream CUDA | 131072 | q4_0 | **41.0–41.2** | 7.2–7.3 GB |
| Q1_0 target-only | prism CUDA | 131072 | q4_0 | 40.5 | 7.3 GB |
| Q1_0 + KV f16 | prism | 65536 | f16 | 11.0 | — |
| Q1_0 + KV q8_0 | prism | 65536 | q8_0 | 40.7 | 6.9 GB |
| Ternary Q2_0 | prism | 32768 | q4_0 | **10.6** | — |
| Ternary Q2_0 | prism | 131072 | q4_0 | 10.6 | — |
| Q1_0 + DSpark (prior) | prism | — | — | 19.2 (−48%) | — |
| Q1_0 + DSpark canonical | prism | 65536 | q4_0 | **39.2** | 7.6 GB |

**Keep (max TPS short-gen):** `Bonsai-27B-Q1_0.gguf`, `CTX_SIZE=131072`, `KV=q4_0`, no speculative (~41 t/s). **Agentic claw-full needs ctx 65k** — 131k VRAM-killed mid-eval (7906>7900).

### Historical runtime evidence (Claw-Eval full (2026-07-24))
- Context 65536, bench_tg 40.2, peak **6.5 GB**. Alias `bonsai` locked to 65k for agentic.

### Historical runtime evidence (Coding-10 (2026-07-26, vs POCKET))
- Context 65536, TEMP 0.6 (fair A/B knobs), bench_tg 41.7, peak **7.5 GB
- Session: [2026-07-26-bonsai-coding-vs-pocket.md](../performance/2026-07-26-bonsai-coding-vs-pocket.md)
- POCKET-35B Q3_K_M on same recipe: ; VRAM **3.7 GB**

## Historical runtime evidence (Config Baseline (agentic / claw-full))
```python
MODEL = 'Bonsai-27B-Q1_0.gguf'
CTX_SIZE = 65536            # 131k OK for short-gen; agentic overflows VRAM_LIMIT 7900
KV_CACHE_K = 'q4_0'
KV_CACHE_V = 'q4_0'
SPEC_TYPE = None
SPEC_DRAFT_N_MAX = 0
BATCH_SIZE = 512
UBATCH_SIZE = 128
THREADS = 8
THREADS_BATCH = 12
FLASH_ATTN = 'on'
NO_MMAP = True
CONT_BATCHING = False
TPS_FLOOR = 15.0
```
Runtime: upstream `llama.cpp/build-cuda` (Q1_0). DSpark / Ternary Q2_0 requires a separate external PrismML checkout.

## Sources / Verification
- Run guide + dspark command: https://github.com/PrismML-Eng/Bonsai-demo (README.md, SPECULATIVE.md) — read 2026-07-18
- Ternary/CUDA status + prebuilt note: same README ("Upstream Status for Ternary", "Pre-built Binary Downloads") — 2026-07-18
- Context length 262144: verified from local GGUF `Bonsai-27B-Q1_0.gguf` (`qwen35.context_length`) — 2026-07-18
- VRAM fit at 65K: observed server log (target + KV allocated, warmed up, no OOM) — 2026-07-18
- Drafter crash: observed server log — 2026-07-18
- Max-TPS matrix (65k–131k): harness cli-bench, this card table — 2026-07-21
- Dense/hybrid/KV/DSpark architecture + published throughput: https://github.com/PrismML-Eng/Bonsai-demo (README.md, SPECULATIVE.md, KV-CACHE.md) + https://prismml.com/news/bonsai-27b — 2026-08-02
- Run guide + dspark command: https://github.com/PrismML-Eng/Bonsai-demo (README.md, SPECULATIVE.md) — read 2026-07-18
- HF metadata JSON (license: Apache 2.0, context: 262144, architecture: qwen35, chat_template with enable_thinking) — https://huggingface.co/api/models/prism-ml/Bonsai-27B-gguf, read 2026-08-29
- Full HF README.md (license: Apache 2.0, publisher generation params: TEMP=0.7, TOP_P=0.95, TOP_K=20, benchmarks: thinking mode, intelligence density) — https://huggingface.co/prism-ml/Bonsai-27B-gguf/blob/main/README.md, read 2026-08-29
- License note: HF repo license is Apache 2.0 (explicit). Original source repo (`PrismML-Eng/Bonsai-27B`) was private/gated at time of original GGUF download (preserve historical note)

## Open Questions
- **DSpark speedup**: canonical drafter loads (39.2 t/s @ 65k / 7.6 GB) but still loses to target-only ~41. Acceptance rate / PrismML kernel path TBD if chasing speculative wins.
- **VITRIOL**: MoE vs dense — treat as **dense** (no shared-memory offload; VRAM guard applies).
- **Ternary variant** `Ternary-Bonsai-27B-Q2_0` — **10.6 t/s** @ ctx 32k, below the 15 t/s floor. See [ternary-bonsai-27b.md](ternary-bonsai-27b.md). Q1_0 preferred.
