# LFM2.5-2.6B-Q8_0 — Model Card (Local)

Runtime scope: quality results remain in the [archived full card](https://github.com/allanschramm/local-model-autotuning/blob/ea43a71f48571811ec1ba25df4ab71ed272d34f1/docs/models/lfm2.5-2.6b.md). Historical performance retains its original protocol.

**Source repo:** https://huggingface.co/LiquidAI/LFM2.5-2.6B
**GGUF repo:** https://huggingface.co/LiquidAI/LFM2.5-2.6B-GGUF
**License:** lfm1.0 (proprietary, `other`) — [HF LICENSE file](https://huggingface.co/LiquidAI/LFM2.5-2.6B-GGUF/blob/main/LICENSE)
**GGUF path:** `models/LiquidAI/LFM2.5-2.6B-GGUF/LFM2.5-2.6B-Q8_0.gguf`
**Family:** Liquid LFM2.5
**Architecture type:** Dense hybrid (`lfm2`) — not MoE
**Quantization:** `Q8_0` (GGUF `general.file_type` = 7)

---

## Architecture (from GGUF metadata, verified 2026-08-29)

- `general.architecture`: `lfm2`
- `general.name`: `LiquidAI_LFM2.5 2.6B`
- `general.size_label`: `2.6B`
- `general.quantization_version`: `2`
- `lfm2.context_length`: **`131072`** *(was 128000 in prior card — corrected to GGUF value)*
- `lfm2.block_count`: `30`
- `lfm2.embedding_length`: `2048` (hidden dim)
- `lfm2.attention.head_count`: `32`
- `lfm2.attention.head_count_kv`: `5` (GQA; 8 KV heads for 8 GQA layers)
- `lfm2.feed_forward_length`: `10752`
- `lfm2.rope.freq_base`: `1e7`
- `lfm2.vocab_size`: `128000`
- `lfm2.shortconv.l_cache`: `3`
- `lfm2.attention.layer_norm_rms_epsilon`: `1e-05`
- `tokenizer.ggml.bos_token_id`: `124894` / `eos_token_id`: `124900`
- MTP / `nextn` / speculative / draft tensors: **`none`** — zero MTP/speculative/draft tensors found in GGUF tensor list (2026-08-29)
- Attention pattern: hybrid — 22 layers use double-gated short convolution; 8 layers use GQA with sliding-window or full attention
- Tensor types: `token_embd.weight` Q8_0; `blk.*.attn_output.weight` Q8_0

Harness: dense → `N_CPU_MOE` must stay `None` (no expert offload).

---

## Hardware requirements (discrete 8 GB-class NVIDIA)

Publisher memory claims (llama.cpp GGUF):
- Q8_0 weights ~2.9 GB (BF16 base = 2.70B params × 2 bytes ≈ 5.4 GB → Q8_0 ≈ 2.9 GB)
- Q4_K_M weights ~1.6 GB (community estimate)
- Publisher CPU inference: 220 tok/s on Apple M5 Max; 113 tok/s on AMD Ryzen AI Max+ 395; 30 tok/s on phone
- Publisher GPU inference: ~15K output tok/s at high concurrency on H100

| ctx | KV type | Fits 8 GB? | Peak VRAM |
| ------ | --------- | ----------- | ----------- |
| 65k | f16 | **Yes** | **~6.0 GB** (measured) |
| 65k | q4_0 | **Yes** | **TBD** |
| 131k | f16 | **Yes** | **TBD** |

> **Note:** `q4_0` KV cache not yet locally validated. F16 KV is the baseline; q4_0 KV is untested.

---

## Recommended settings

### Sampling (publisher defaults — general use)

Publisher generation parameters for general inference ([HF source](https://huggingface.co/LiquidAI/LFM2.5-2.6B), extraction 2026-08-29):
- **TEMP:** `0.1`
- **TOP_K:** `50`
- **REPEAT_PENALTY:** `1.1`

Publisher note: LFM2.5-2.6B is a pure reasoning model that always thinks before answering — it adds a `<think>` tag directly in the chat template. For agentic workloads the thinking is intentional and should be preserved.

### Sampling (agentic / general — local tuned)

Locally validated sampler for agentic + coding workloads:
- **TEMP:** `0.6`
- **TOP_P:** `0.95`
- **TOP_K:** `20`
- **MIN_P:** `0.0`
- **REPEAT_PENALTY:** `1.0` (not specified by publisher; harness default)

### Reasoning control

**`lfm2` template has NO thinking variables** — neither `enable_thinking`, `reasoning_effort`, `reasoning_budget`, nor `thinking_budget` exist in the embedded chat template (verified from GGUF `tokenizer.chat_template` field, 2026-08-29). No `--reasoning`, `--reasoning-effort`, `--reasoning-budget`, or `--enable-thinking` flags produce any effect on this GGUF.

Consequence: reasoning content (`<think>...</think>`) is always rendered when the model generates it. There is no server-side suppression lever.

---

## MTP / Speculative decoding

**This GGUF has no MTP tensors.** Zero `blk.*.nextn.*`, `mtp_*`, `speculative.*`, or `draft.*` tensors found on `GGUFReader` tensor inspection (2026-08-29).

Speculative decoding is available via the separate **LFM2.5-2.6B-DSpark** drafter (328M, separate HF repo). This is **not** bundled in the GGUF and requires a different serving path — it is out of scope for this card.

No MTP flags apply; `verified from common/arg.cpp` (upstream b10549): no `lfm2`-specific reasoning flag aliases exist.

---

## MoE split (VITRIOL split)

Dense architecture — `expert_count` is not present in GGUF metadata; `general.architecture = lfm2` (not `lfm2moe`). `N_CPU_MOE` must remain `None` (harness gate: no expert offload possible).

---

## Our config baseline

```python
MODEL = 'LFM2.5-2.6B-Q8_0.gguf'
CTX_SIZE = 65536
KV_CACHE_K = 'f16'
KV_CACHE_V = 'f16'
BATCH_SIZE = 512
UBATCH_SIZE = 256
THREADS = 8
THREADS_BATCH = 8
FLASH_ATTN = 'on'
N_CPU_MOE = None
TPS_FLOOR = 15.0
```

### Measured (Objective Vector — Store best)

| Metric | Value | Notes |
| -------- | ------- | ------- |
| **Trial Status** | `on_front` |  |
| **Bench tg** | **91.2 t/s** | Store best @65536 |
| **Combined TPS** | **TBD** |  |
| **Peak VRAM** | **~6.0 GB** | @65k, f16 KV |

Historical bench tg increased from 78.8 to 91.2 t/s across the recorded iterations (n=8); quality comparisons are archived.

Evidence: `results.db` / `results.tsv` rows for `LFM2.5-2.6B-Q8_0.gguf`. Claw jump (2026-08-08) is the harness fix (`reasoning_content` / `max_tokens≥2048`), not a model change — see [thinking-models-claw-harness.md](https://github.com/allanschramm/local-model-autotuning/blob/ea43a71f48571811ec1ba25df4ab71ed272d34f1/docs/discovery/thinking-models-claw-harness.md).

---

## Sources / Verification

| Source | Key data extracted | Extraction date |
| -------- | ------------------- | ----------------- |
| [LiquidAI/LFM2.5-2.6B-GGUF — HF API](https://huggingface.co/api/models/LiquidAI/LFM2.5-2.6B-GGUF) | Quantization variants, file size, license lfm1.0, downloads (785K) | 2026-08-29 |
| [LiquidAI/LFM2.5-2.6B — HF API](https://huggingface.co/api/models/LiquidAI/LFM2.5-2.6B) | BF16 params 2.697B, 16 languages, downloads (191K) | 2026-08-29 |
| [LiquidAI/LFM2.5-2.6B — HF README](https://huggingface.co/LiquidAI/LFM2.5-2.6B/blob/main/README.md) | Context 131072, 30 layers (22 conv + 8 GQA), vocab 128K, 34T tokens; generation params temp 0.1/top_k 50/rep_penalty 1.1; benchmarks (IFBench 59.17, Claw-Eval avg 62.85, PinchBench 68.22, AIME25 51.87, LiveCodeBenchv6 59.41); CPU inference speeds; DSpark drafter | 2026-08-29 |
| [LiquidAI/LFM2.5-2.6B-GGUF — HF README](https://huggingface.co/LiquidAI/LFM2.5-2.6B-GGUF/blob/main/README.md) | llama-cli usage example, QAD Q4_0 note | 2026-08-29 |
| Local GGUF `GGUFReader` (2026-08-29) | `lfm2.context_length=131072` (corrected from 128000); `lfm2.block_count=30`; `embedding_length=2048`; `head_count=32`; `head_count_kv=5`; `vocab_size=128000`; zero MTP/speculative/draft tensors | 2026-08-29 |
| Local GGUF `tokenizer.chat_template` | No `enable_thinking`, `reasoning_effort`, `reasoning_budget`, or `thinking_budget` variables — reasoning flags have no effect | 2026-08-29 |
| `results.db` / `results.tsv` | Store best / tg 91.2 t/s @65536 (n=8) | — |

Publisher benchmarks (LFM2.5 family, from base README — comparative table, extraction 2026-08-29):

| Benchmark | LFM2.5-2.6B | gemma-4-E2B-it | gemma-4-E4B-it | Qwen3.5-4B | Qwen3.5-9B |
| --- | ---: | ---: | ---: | ---: | ---: |
| Multi-IF | 80.07 | 69.44 | 77.35 | 55.67 | 62.55 |
| IFStruct | 85.49 | 64.85 | 76.65 | 36.25 | 78.50 |
| ToolSandbox | 77.83 | 52.40 | 65.00 | 75.55 | 76.44 |
| PinchBench | 68.22 | 44.24 | 55.09 | 71.26 | 71.45 |
| LiveCodeBenchv6 | 59.41 | 54.92 | 63.77 | 60.85 | 69.86 |

---

## Open questions

- **TBD:** `q4_0` KV cache — does mixed K/V q4_0 produce a speed cliff (as observed on Qwen3.8-4B GDN hybrid)? Needs a 2-minute ServerIntent fit probe before any KV fingerprinting Trial.
- **TBD:** Peak VRAM at 131k context with f16 KV — the context_length claim is 131072 but no local 131k VRAM measurement has been recorded.
