# Ornith-1.0-9B — Model Card (Local)

Runtime scope: quality results remain in the [archived full card](https://github.com/allanschramm/local-model-autotuning/blob/ea43a71f48571811ec1ba25df4ab71ed272d34f1/docs/models/ornith-1.0-9b.md). Historical performance retains its original protocol.

**Source (Unsloth UD, fetched 2026-08-29):** https://huggingface.co/unsloth/Ornith-1.0-9B-GGUF (api/models/unsloth/Ornith-1.0-9B-GGUF; architecture `qwen35`, context_length 262144, MIT, 228K downloads, lastModified 2026-07-18, base_model: deepreinforce-ai/Ornith-1.0-9B)
**Source (deepreinforce official, fetched 2026-08-29):** https://huggingface.co/deepreinforce-ai/Ornith-1.0-9B-GGUF (api/models/deepreinforce-ai/Ornith-1.0-9B-GGUF; architecture `qwen35`, context_length 262144, MIT, license_link → deepreinforce-ai/Ornith-1.0-9B/blob/main/LICENSE)
**Source (ornith-ai mirror, fetched 2026-08-29):** https://huggingface.co/orinith-ai/Ornith-1.0-9B-GGUF (api/models/ornith-ai/Ornith-1.0-9B-GGUF; 4.3M downloads, 660 likes, MIT, same upstream chat_template; serves `ornith-1.0-9b-Q4_K_M.gguf` etc.)
**Unsloth docs:** https://unsloth.ai/docs/models/qwen35 (model uses Qwen 3.5 architecture; UD 2.0 quant)
**License:** **MIT** (per both Unsloth and ornith-ai `cardData.license` + `license_link`; NOT Apache-2.0 — earlier header was wrong)
**GGUF basenames (each is its OWN Trial/Objective Vector):**
- `Ornith-1.0-9B-UD-Q4_K_XL.gguf` — Unsloth Dynamic 2.0 (5.98 GB; local store: `models/unsloth/Ornith-1.0-9B-GGUF/`)
- `Ornith-1.0-9B-MTP-Q4_K_M.gguf` — MTP pack (`protoLabsAI/Ornith-1.0-9B-MTP-GGUF`; served via `SPEC_TYPE=draft-mtp`)
- `ornith-1.0-9b-Q4_K_M.gguf` — deepreinforce official Q4_K_M (; @ 65k)
**Family:** Ornith (deepreinforce-ai; Qwen 3.5 architecture)
**Quantizations:** Unsloth `UD-Q4_K_XL` ≠ deepreinforce `Q4_K_M` ≠ MTP pack — each is its own Objective Vector (n=18 / n=19 / n=4 in results store).

## Architecture (from GGUF metadata, verified via gguf lib)
- Causal LM (hybrid Attention + SSM)
- **`block_count` = 32 layers**
- Hidden **4096**, vocab 248320, ctx **262144**
- **Hybrid Attention + SSM (Mamba-2 style) layers**:
  - `full_attention_interval = 4` — every 4th layer is full attention
  - Contains `ssm.conv_kernel=4`, `ssm.state_size=128`, `ssm.group_count=16`, `ssm.time_step_rank=32`, `ssm.inner_size=4096`
  - 8 layers of full attention (head count: 16 Q, 4 KV, key/value length 256)
  - 24 layers of SSM / linear path
- `rope.freq_base = 10,000,000`
- **`general.name` = `Ornith 1.0 9B`**, file_type=15 (Q4_K_M), quantization_version=2
- 427 tensors total

## Hardware requirements (per community and size)
| Quant | Total RAM / VRAM |
| --- | --- |
| **Q4_K_M (our pick)** | **~5.6 GB VRAM** (VRAM target is ~5.6 GB + KV cache overhead; HF api `gguf.total` = 8953803264 ≈ 8.95 GB bf16, 4-bit ~5.6 GB; same as Unsloth UD; publisher benchmarks: 5.6 GB target) |
| Q8_0 | ~9.5 GB VRAM |

## Our target
8 GB VRAM (8 GB-class discrete NVIDIA). The 4-bit model size is ~5.6 GB, meaning it fits entirely in GPU VRAM (NGL = 999). However, active KV cache overhead for large contexts can push VRAM usage above 8 GB. Setting safe context size limits is important.

## Publisher HF VRAM recommendations
- HF README (Unsloth, fetched 2026-08-29): model described as "`≈19 GB in bf16`"; 4-bit quant ~5.6 GB fits single 80 GB GPU
- HF README (ornith-ai mirror): same model card; no explicit VRAM limit beyond implicit 80 GB context serving; our 8 GB target remains the binding constraint


## Recommended Settings (based on Qwen 3.5)
- **Temperature:** 0.4
- **Top P:** 0.95
- **Top K:** 20
- **Min P:** 0.0
- **Repeat Penalty:** 1.0 (disabled)

## Reasoning control (verified 2026-08-29 — local `gguf_dump.py --no-tensors` on UD-Q4_K_XL + cardData.chat_template grep; qwen35 family verified per session)
- **Template reads ONLY `enable_thinking`** (NOT `reasoning_effort`) — confirmed in both Unsloth UD and deepreinforce `chat_template`: `enable_thinking` appears in the Jinja generate block (line `enable_thinking is defined and enable_thinking is false`); `reasoning_effort` is absent. → `--reasoning-effort` is a **silent NO-OP** on this GGUF family (same verdict as Ornith-1.5-9B / Qwen3.5 family, documented in reasoning-level mapping 2026-08-29).
- **Working reasoning levers (use these, never `reasoning-effort`):**
  - `--reasoning on` / `--reasoning off` (Baseline `REASONING`; controls whether `<think>` block is emitted; default = on per template — opens with `<think>` block)
  - `--reasoning-budget N` (`REASONING_BUDGET`; server/template-independent budget cap — verified lever, not template-gated)
  - `REASONING_PRESERVE` (`reasoning_preserve`; only where `/props` reports `supports_preserve_reasoning = true` — **not verified** for Ornith-1.0-9B GGUF; mark TBD until prop-checked)
- **Daily alias budget (unmeasured / no Trial claim — DO NOT claim as measured):** `ornith-9b` alias runs budget **2048 + message** (same family pattern as Ornith-1.5-9B; no `results.db` Trial row validates it; code-only, not a benchmarked number).
- **Publisher-recommended sampling (HF README / card, fetched 2026-08-29):** `temperature=0.6`, `top_p=0.95`, `top_k=20`; use `temperature=1.0` only to reproduce reported benchmark setup. **Our local baseline stays 0.4 / 0.95 / 20** (see §7) — different from publisher recommendation; both documented.
- **No `reasoning_effort` ladder** (unlike Qwen3.8-27B UD-IQ1_S which reads `reasoning_effort` with ladder xhigh/medium/low + `high→xhigh` alias + raise-exception; that is a different arch — NOT this card).
- **Verification notes:** `gguf_dump.py --no-tensors --json` on local `Ornith-1.0-9B-UD-Q4_K_XL.gguf` → `tokenizer.chat_template` contains `enable_thinking` 4×, zero `reasoning_effort`; HF API `cardData` has no reasoning keys. Source: unsloth/Ornith-1.0-9B-GGUF (`chat_template` embedded); deepreinforce-ai/Ornith-1.0-9B-GGUF (same).

## MTP (Multi-Token Prediction)
- **Base UD GGUF (`Ornith-1.0-9B-UD-Q4_K_XL.gguf`): NO `nextn`.**
- **MTP GGUF (downloaded 2026-07-20):** `models/Ornith-1.0-9B-MTP-Q4_K_M.gguf` from `protoLabsAI/Ornith-1.0-9B-MTP-GGUF` — embedded `nextn`.
- Enable on MTP file: `SPEC_TYPE=draft-mtp`, `SPEC_DRAFT_N_MAX=4`, no draft model path.
- Fair matrix: base UD **38.7 t/s** → MTP GGUF **56.3 t/s** (**+46%**). [session](../performance/2026-07-20-small-model-tps-matrix.md) · [guide](../discovery/small-model-mtp-tps.md).

## VITRIOL / Split strategy
Since the model is ~5.6 GB and we have 8 GB of VRAM, we can run with maximum GPU offload (`--n-gpu-layers 999`), loading the model completely into GPU VRAM.

## Our config baseline (speed path 2026-07-20)
- Prefer MTP file when throughput matters: `MODEL = 'Ornith-1.0-9B-MTP-Q4_K_M.gguf'`
- `SPEC_TYPE = 'draft-mtp'`, `SPEC_DRAFT_N_MAX = 4`
- Shared matrix knobs: `KV q4_0`, batch 256/128, threads 6/8, `NO_MMAP=True`


## Older verified baseline (2026-06-26, non-MTP)
- `MODEL = 'ornith-1.0-9b-Q4_K_M.gguf'` — deepreinforce official Q4_K_M (separate Trial from UD/MTP)
- Day/Night picks on this front use Unsloth UD / MTP basenames when those vectors are complete — see Pareto leaderboard
- `CTX_SIZE = 131072`
- `KV_CACHE = 'q4_0'`
- `NGL = 99`
- `THREADS = 8`
- `THREADS_BATCH = 8`
- `FLASH_ATTN = 'on'`

## Sources / Verification
- **Unsloth GGUF repo** (fetched 2026-08-29): https://huggingface.co/unsloth/Ornith-1.0-9B-GGUF — HF api: architecture `qwen35`, context_length 262144, MIT, gguf.total=8953803264 (≈8.95 GB bf16), downloads 228K, lastModified 2026-07-18, safetensors.params=null (GGUF-only)
- **deepreinforce-ai base repo** (fetched 2026-08-29): https://huggingface.co/deepreinforce-ai/Ornith-1.0-9B-GGUF — HF api: same architecture/context, MIT, license_link → deepreinforce-ai/Ornith-1.0-9B/blob/main/LICENSE
- **ornith-ai mirror repo** (fetched 2026-08-29): https://huggingface.co/orinith-ai/Ornith-1.0-9B-GGUF — HF api: 4.3M downloads, 660 likes, MIT; same model card + chat_template as deepreinforce
- Local GGUF verified via `GGUFReader` on `Ornith-1.0-9B-UD-Q4_K_XL.gguf` (2026-07-18+): block_count=32, hidden=4096, ctx=262144, rope.freq_base=10M, ssm parameters confirmed
- Reasoning control verified via `gguf_dump.py --no-tensors --json` on UD-Q4_K_XL (2026-08-29): `chat_template` contains `enable_thinking` 4×, zero `reasoning_effort`; qwen35 family ruling applies to Ornith-1.0-9B

## Tuning History (2026-06-29)

### Hyperparam sweep
| Param | TPS |
| ------- | ----- |
| Baseline (0.4/0.95/20) | **52.2** |
| TEMP=0.7 | 51.9 |
| TOP_P=0.9 | 51.3 |
| REPEAT_PENALTY=1.05 | 48.2 |
| TOP_K=40 + REPEAT_PENALTY=1.05 | 51.2 |

### BeeLlama tested (no gains)
- BeeLlama baseline: 41.7 TPS (20% slower than stock fork)
- BeeLlama + CopySpec: 45.1 TPS (marginal, server crashes)
- BeeLlama + TCQ turbo3_tcq: HTTP 500 at 131k ctx
- BeeLlama + turbo3: 31.4 TPS (saves VRAM but kills TPS)

### 2026-07-01 Validation (2-task, b1024/ub256, upstream build-cuda)
| Metric | Value |
| -------- | ------- |
| TPS | 50.2 |
| VRAM | 7.1 GB |
| Bench tg | 43.7 t/s |

### Verdict
- **0.580 / 52.2 TPS is the ceiling** for discrete 8 GB-class NVIDIA

- 9B is the optimal model for this hardware

### 2026-07-19 Update (Unsloth Dynamic 4-bit XL Quant)
- Upgraded local model to the newly released `Ornith-1.0-9B-UD-Q4_K_XL.gguf` (5.98 GB) from Unsloth.
- Alias: `ornith-9b` (INDEX name; old `o9` retired 2026-07-23).

### Historical runtime evidence (Claw-Eval full (2026-07-24))
- Context 65536, bench_tg 42.1, peak **7.5 GB**.

### Objective Vector @ 65k (2026-08-08, `VRAM_LIMIT_MB=8100`, `AUTORESEARCH_SKIP_FREE_CLAMP=1`)
- Fingerprint shared: (peak **7.8 GB; TPS **48.6**) + claw-full.
After agentic fix: , peak **7.7 GB**, TPS **42.1**.

### Fair remeasure @ 4096 floor (2026-08-19, row `29b91359`)
- Same fingerprint as above (65k, TEMP 0.4, `AUTORESEARCH_SKIP_FREE_CLAMP=1`) but agentic `max_tokens` floor 4096 (harness fix).
No 65k ctx truncations (`truncated=0`), no turns hit 4096 (`n_decoded` max ≈ 3.6k).
Note: `VRAM_LIMIT_MB` must be ≥ 8100-class for this quant (Q4_K_XL): at the 8000/keepout-256 ceiling (7932 MB) the runtime monitor killed at 7946 MB (`used > limit`); keepout 64 (ceiling 8124) is the proven config, peak 7.7 GB.

### Historical runtime evidence (Coding-10 (2026-07-24, UD-Q4_K_XL))

| Metric | Value |
| --- | --- |
| ctx | **32768** (65k VRAM-kill mid-HumanEval) |
| bench_tg | 38.4 t/s |
| peak VRAM | 7.4 GB |


### Historical runtime evidence (Coding-10 MTP variant (2026-07-27, `Ornith-1.0-9B-MTP-Q4_K_M.gguf`))
- ctx **32768**, draft-mtp n=4, TEMP 0.4.
- Combined TPS **86.7; bench_tg **64.0; peak **7.4 GB**.

### Historical runtime evidence (deepreinforce `ornith-1.0-9b-Q4_K_M` claw-full (2026-07-28))
- First try @ 65k / `VRAM_LIMIT_MB=7900`: **VRAM kill** mid T054 (used 7930 MB) after 6/15 @ 0.4000 — row `MODEL_REJECTED`.
- Retry @ 32k: early VRAM kill on T002 (7910 MB) — not usable.
- Success @ 65k / `VRAM_LIMIT_MB=8000`: ** , bench_tg **42.5**, peak **7.8 GB**. Weaker than UD . Vector **complete** with (`quality details archived`).

## Open questions
- **REASONING_PRESERVE support:** not verified — `/props` check needed; do NOT claim `reasoning_preserve` works until prop-verified (same rule as Ornith-1.5-9B). `--reasoning-effort` confirmed NO-OP (2026-08-29).
- **Daily budget 2048 + message:** unmeasured — no Trial row; only alias-configured code (same family pattern as Ornith-1.5-9B); do not cite as benchmarked.
- Variant identity: UD-Q4_K_XL, MTP and deepreinforce Q4_K_M are distinct GGUF artifacts; keep their runtime measurements separate.
- **No `reasoning_effort` ladder** applies — this is qwen35 family (enable_thinking only), unlike 27B which has a ladder; no open question on ladder levels.

## Historical runtime provenance

Historical validation recorded **7.9 GB** peak VRAM; consult the archived card for the exact artifact and run conditions.
