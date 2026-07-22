# Bonsai-27B — Model Card (Local)

**Source repo:** https://huggingface.co/PrismML-Eng/Bonsai-27B (models were downloaded from here; see note)
**Canonical card repo:** https://huggingface.co/prism-ml/Bonsai-27B-gguf (**PRIVATE** — 27B repos are gated until launch)
**Run guide:** https://github.com/PrismML-Eng/Bonsai-demo (README.md + SPECULATIVE.md)
**License:** Private / gated (Bonsai-27B preview)
**Local file:** `models/Bonsai-27B-Q1_0.gguf` (3627.3 MB / 3.53 GiB)
**Drafter file:** `models/draft/Bonsai-27B-dspark-Q4_1.gguf` (1704.7 MB) — DSpark speculative drafter (target-specific)
**Family:** Bonsai (PrismML-Eng) — 1-bit ternary-pack
**Quantization:** `Q1_0` (1-bit; ~1.125 bpw)

## Architecture
- Base: **Qwen3.5** (GGUF metadata key prefix `qwen35.*`); hybrid attention (per run guide, keeps KV small for its size)
- Context length: **262,144 tokens** — GGUF-verified (`qwen35.context_length = 262144`)
- Block count / expert counts / head counts: **TBD** — verify with `gguf.GGUFReader` (header read, no model run)
- 27B family is vision-language: accepts images (vision projector +~0.9 GiB when used)

## Hardware Requirements (RTX 4060 8 GB)
| Item | Size | Note |
|---|---|---|
| Bonsai-27B Q1_0 weights | 3.53 GiB | 1-bit |
| FP16 KV @ 65K ctx | ~1.1 GiB | hybrid attn (16/64 layers cached) |
| FP16 KV @ 100K ctx | ~6.3 GiB | from run-guide table |
| DSpark drafter extra | ~0.5–2.4 GiB | embeds/head shared with target; naive loader estimate 2426 MiB |
| Runtime overhead | ~0.7–1.2 GiB | +vision projector ~0.9 GiB if used |

**Verified fit:** target + 65K KV cache loaded and warmed up on RTX 4060 with no OOM (server log).
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
- `--spec-type draft-dspark` + `--spec-draft-n-max 4` — **required**; `n-max` must equal drafter block size (4). A smaller value crashes at the first draft round.
- `-ngl 999 -ngld 999` — target and drafter fully on GPU (model has ≤64 layers, so `99` also covers all).
- `-fa on` — flash attention.
- `-c 16384+` — guide recommends ≥16384 (model thinks 1.5–2k tokens before the visible answer); user tested 65536.
- `-np 1` — single slot (speculative disables cross-request prompt-cache; one request at a time).

## Speculative Decoding (DSpark)
- Drafter is **target-specific**: `Bonsai-27B-dspark-Q4_1.gguf` only accelerates `Bonsai-27B-Q1_0.gguf` (Ternary drafter only the Ternary target).
- ~1.8–2× faster decode on CUDA code/reasoning workloads; output at temp 0 identical to normal decode.
- Verify engaged via API `timings.draft_n` / `draft_n_accepted` (missing/zero = not active).
- `llama-cli` has no draft support; one-shot binary is `llama-speculative-simple`.

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

## Max TPS (RTX 4060 8 GB, ctx band 65k–131k) — 2026-07-21

Harness: `benchmark_search.py --no-agentic-quick --no-agentic-full --no-coding` → `llama-cli` `-n 512`.

| Config | Binary | ctx | KV | bench_tg | Peak VRAM |
|---|---|---|---|---|---|
| Q1_0 target-only | upstream CUDA | 65536 | q4_0 | **40.5** | 6.1 GB |
| Q1_0 target-only | upstream CUDA | 131072 | q4_0 | **41.0–41.2** | 7.2–7.3 GB |
| Q1_0 target-only | prism CUDA | 131072 | q4_0 | 40.5 | 7.3 GB |
| Q1_0 + KV f16 | prism | 65536 | f16 | 11.0 (reject) | — |
| Q1_0 + KV q8_0 | prism | 65536 | q8_0 | 40.7 | 6.9 GB |
| Ternary Q2_0 | prism | 131072 | q4_0 | 10.6 (reject) | — |
| Q1_0 + DSpark (prior) | prism | — | — | 19.2 (−48%) | — |
| Q1_0 + DSpark canonical | prism | 65536 | q4_0 | **39.2** | 7.6 GB |

**Keep (max TPS):** `Bonsai-27B-Q1_0.gguf`, `CTX_SIZE=131072`, `KV=q4_0`, no speculative, upstream `llama.cpp` CUDA (~41 t/s). Canonical DSpark loads and stays under 8GB at ctx=65k but does not beat target-only.

## Config Baseline (max TPS, no draft)
```python
MODEL = 'Bonsai-27B-Q1_0.gguf'
CTX_SIZE = 131072           # 65k–131k band; same short-gen TPS as 65k
KV_CACHE_K = 'q4_0'
KV_CACHE_V = 'q4_0'
SPEC_TYPE = None            # DSpark slower than target-only on Q1_0
SPEC_DRAFT_N_MAX = 0
BATCH_SIZE = 512
UBATCH_SIZE = 128
THREADS = 8
FLASH_ATTN = 'on'
NO_MMAP = True
```
Runtime: upstream `llama.cpp/build-cuda` (Q1_0). DSpark / Ternary Q2_0 requires a separate external PrismML checkout.

## Sources / Verification
- Run guide + dspark command: https://github.com/PrismML-Eng/Bonsai-demo (README.md, SPECULATIVE.md) — read 2026-07-18
- Ternary/CUDA status + prebuilt note: same README ("Upstream Status for Ternary", "Pre-built Binary Downloads") — 2026-07-18
- Context length 262144: verified from local GGUF `Bonsai-27B-Q1_0.gguf` (`qwen35.context_length`) — 2026-07-18
- VRAM fit at 65K: observed server log (target + KV allocated, warmed up, no OOM) — 2026-07-18
- Drafter crash: observed server log — 2026-07-18
- Max-TPS matrix (65k–131k): harness cli-bench, this card table — 2026-07-21

## Open Questions
- **DSpark speedup**: canonical drafter loads (39.2 t/s @ 65k / 7.6 GB) but still loses to target-only ~41. Acceptance rate / PrismML kernel path TBD if chasing speculative wins.
- **Architecture**: block_count, expert_count, head_count_kv — **TBD**, run `gguf.GGUFReader`.
- **VITRIOL**: MoE vs dense — treat as **dense** (no shared-memory offload; VRAM guard applies).
- **Ternary variant** `Ternary-Bonsai-27B-Q2_0` (+ its dspark drafter) — separate card `ternary-bonsai-27b.md` **TBD** (spot: 10.6 t/s @ 131k, reject).
