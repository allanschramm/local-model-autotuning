# Qwen3.5-9B — Model Card (Local)

Runtime scope: quality results remain in the [archived full card](https://github.com/allanschramm/local-model-autotuning/blob/ea43a71f48571811ec1ba25df4ab71ed272d34f1/docs/models/qwen3.5-9b.md). Historical performance retains its original protocol.

**Source repo:** https://huggingface.co/Qwen/Qwen3.5-9B  (base, released 2026-03-02; ctx 262,144 claimed, tag `qwen3_5`; license apache-2.0)
**MTP / GGUF repo:** https://huggingface.co/unsloth/Qwen3.5-9B-GGUF (Dynamic 4-bit / UD; MTP embedded in selected quant filenames; lastModified 2026-03-02)
**Unsloth docs:** https://unsloth.ai/docs/models/qwen3.5 (extraction 2026-08-29; thinking vs instruct settings; 9B small-series reasoning disabled by default)
**License:** Apache-2.0
**Quantization:** Unsloth Dynamic Q4_K_XL with MTP — `UD-Q4_K_XL` (QAT-lossless 4-bit)

## Architecture (from GGUF metadata)
- Causal LM (hybrid Attention + SSM — Qwen 3.5 architecture)
- **`block_count` = 32 layers**
- Hidden **4096**, context **131072**
- **Hybrid Attention + SSM (Mamba-2 style)**:
  - `full_attention_interval = 4` — every 4th layer is full attention
  - SSM: `conv_kernel=4`, `state_size=128`, `group_count=16`, `time_step_rank=32`, `inner_size=4096`
  - 8 full attention layers, 24 SSM layers

## Hardware Requirements (discrete 8 GB-class GPU)
- Fits entirely in GPU VRAM (NGL = 99/999).
- Model size ~5.46 GB (Q4_K_M) / ~6.14 GB (UD Q4_K_XL), leaving adequate headroom for context cache.
- NVIDIA: CUDA backend. AMD: ROCm/HIP backend ([setup guide](../discovery/amd-rocm-windows-setup.md)).

## MTP (Multi-Token Prediction)
- **MTP tensors are embedded in this GGUF** (`qwen35.nextn_predict_layers`, `blk.32.nextn.*` verified 2026-07-20).
- Enable: `SPEC_TYPE = "draft-mtp"`, `SPEC_DRAFT_N_MAX = 4` — **no** `--spec-draft-model`.
- Fair matrix (2026-07-20, `llama-cli` `-n 512`): base **38.7 t/s** → MTP **57.3 t/s** (**+48%**). Evidence: [session](../performance/2026-07-20-small-model-tps-matrix.md).

## MoE split (VITRIOL)

N/A — dense Gated DeltaNet hybrid; `--n-cpu-moe` not applicable.

## Recommended Settings
- **Temperature:** 0.4
- **Top P:** 0.95
- **Top K:** 20
- **Min P:** 0.0
- **Repeat Penalty:** 1.05
### Reasoning control (verified 2026-08-29, qwen35 family — enable_thinking ONLY)
- **Template variable:** `enable_thinking` only (HF `chat_template.jinja` / GGUF tokenizer, 2026-08-29 extraction). `reasoning_effort` / `reasoning_budget` are NOT queried — unlike the Qwen3.8-27B open-source template which reads `reasoning_effort` (ladder xhigh/default/low). **--reasoning-effort is a silent NO-OP on this GGUF.**
- **Small-series default (Unsloth guide):** reasoning **disabled by default** on 9B. To enable: `--chat-template-kwargs '{"enable_thinking":true}'`; to disable/instruct mode: `false`. Source: Unsloth docs (extraction 2026-08-29), section «How to enable or disable reasoning and thinking».
Instruct/non-thinking (false) — general: TEMP 0.7 / TOP_P 0.8 / presence_penalty 1.5; reasoning tasks: TEMP 1.0 / TOP_P 0.95 / presence_penalty 1.5.
- **GDN ngram warmup-death warning:** ngram spec died in warmup on this GDN-hybrid arch; do not re-add without explicit A/B. The daily-driver recipe keeps the flag commented. No live `--spec-type ngram` — verified dead at warmup on this architecture family; do not restore without a fresh A/B trial.
- **Reasoning preserve:** `REASONING_PRESERVE` irrelevant here — `enable_thinking` is a boolean toggle, not a ladder, and the GGUF's embedded template does not expose `preserve_reasoning`; no `/props` claim to seed.

## Config Baseline (2026-07-20 TPS matrix knobs)
- `MODEL = 'Qwen3.5-9B-UD-Q4_K_XL.gguf'`
- `CTX_SIZE = 131072`
- `KV_CACHE = 'q4_0'`
- `KV_CACHE_K = 'q4_0'`
- `KV_CACHE_V = 'q4_0'`
- `NGL = 99`
- `THREADS = 6`
- `THREADS_BATCH = 8`
- `BATCH_SIZE = 256`
- `UBATCH_SIZE = 128`
- `FLASH_ATTN = 'on'`
- `SPEC_TYPE = 'draft-mtp'`
- `SPEC_DRAFT_N_MAX = 4`
- `NO_MMAP = True`

### Status
- **Measured (2026-07-20):** embedded MTP works on upstream CUDA; +48% vs base under fair knobs. Slower absolute than Gemma+draft MTP (122 t/s).

### Historical runtime evidence (Claw-Eval full (2026-07-25) — UD basename)
- Context 32768**, draft-mtp n=4, TEMP 0.4, bench_tg **65.0**, peak **7.5 GB**.

### Historical runtime evidence (Claw-Eval full (2026-07-28) — `Qwen3.5-9B-MTP-Q4_K_M.gguf`)
- First try @ 32k+MTP / limit 7900: **VRAM kill** mid T004 @ 7906 MB.
- Success @ 32k+MTP / limit **8000**: ** , bench_tg **67.5**, peak **7.7 GB**. Vector **complete** with (`quality details archived`). Still weak agentic.

### Historical runtime evidence (Coding-10 — rejected on the operator host (2026-07-27) — UD basename only)
Objective Vector **incomplete** for `Qwen3.5-9B-UD-Q4_K_XL.gguf` (coding axis never landed). All attempts hit harness `VRAM_LIMIT_MB=7900` mid-eval during long codegen:

| config | outcome |
| --- | --- |
| ctx 32768 + MTP n=4 | **VRAM kill** 7913 MB mid HumanEval |
| ctx 65536, no MTP | **VRAM kill** 7931 MB mid HumanEval (preflight est 7584 MB) |

Claw-full @ 32k+MTP fits because tool calls are short; coding-10 long generation does not. **Do not retry** UD coding on discrete 8 GB-class NVIDIA unless `VRAM_LIMIT_MB` or ctx/KV policy changes. Weak agentic + no coding vector on this rig.

`Qwen3.5-9B-MTP-Q4_K_M` already has in TSV (separate Trial).

### AMD ROCm benchmark (2026-08-16) — `Qwen3.5-9B-MTP-Q4_K_M.gguf`

Discrete 8 GB-class AMD (RDNA 2, gfx1032) via ROCm/HIP b10448. Setup: [AMD ROCm Windows guide](../discovery/amd-rocm-windows-setup.md).

| config | pp512 (t/s) | tg128 (t/s) | notes |
| --- | --- | --- | --- |
| base (no MTP), ngl 999, q4_0 KV, FA on | 551 ± 2 | **33.4 ± 0.1** | ROCm backend, full VRAM offload |
| draft-mtp n=4 | 88.3 | **40.4** | +21% over base (llama-cli 512 tokens) |

- Peak VRAM: ~5.5 GB dedicated (model) + KV cache headroom.
- RAM overhead: ~2–4 GB (HIP runtime + VMM:no staging buffers on RDNA 2).
- Prior Vulkan fallback was ~13 t/s; CPU-only was ~5 t/s.
- Prior Vulkan fallback was ~13 t/s; CPU-only was ~5 t/s.

## Sources / Verification (extraction 2026-08-29)
- **Base model:** https://huggingface.co/Qwen/Qwen3.5-9B — HF API (`tags`: `qwen3_5`, `license:apache-2.0`, `model_type=qwen3_5`, `pipeline_tag=image-text-to-text`, `architectures=["Qwen3_5ForConditionalGeneration"]`). `lastModified`: 2026-03-02; `downloads`: 12,507,912. Context claim: 262,144 (GGUF repo `gguf.context_length` 262144, verified 2026-08-29). License file: `LICENSE` (apache-2.0).
- **GGUF / MTP repo:** https://huggingface.co/unsloth/Qwen3.5-9B-GGUF — `unsloth/Qwen3.5-9B-GGUF` API (`gguf.architecture=qwen35`; `gguf.context_length=262144`; siblings show `Q4_K_M`, `UD-Q4_K_XL`, `UD-IQ2_M` etc.; `mmproj-BF16.gguf` present for vision). MTP packaging: embedded in selected quant filenames (`blk.32.nextn.*` verified in local file header 2026-07-20; not flagged separately in GGUF repo metadata beyond quant filenames — rely on local `GGUFReader`).
- **Unsloth docs / settings:** https://unsloth.ai/docs/models/qwen3.5 (extraction 2026-08-29) — 9B is Small-series; reasoning disabled by default; thinking (enable_thinking=true): TEMP 1.0/0.6; instruct (false): TEMP 0.7/1.0; `presence_penalty` 1.5 (general) / 0.0 (coding). No `reasoning_effort` ladder.
- **Reasoning verification (local):** GGUF `tokenizer.chat_template` (extracted 2026-08-29 via `gguf_dump.py --no-tensors --json`) — reads ONLY `enable_thinking` (`is defined` / `is true` / `is false`); no `reasoning_effort` variable queried (unlike Qwen3.8-27B template which uses ladder xhigh/medium/low). **Verified: --reasoning-effort is a silent NO-OP.**
- **MTP packaging / local file header (2026-07-20 verification):** `GGUFReader` shows `qwen35.nextn_predict_layers`, `blk.32.nextn.*` tensors present in selected basenames (`Qwen3.5-9B-MTP-Q4_K_M.gguf`); `UD-Q4_K_XL` file also verified embedded MTP in earlier session (2026-07-28 Trial). `SPEC_DRAFT_N_MAX=4` set; `SPEC_TYPE=draft-mtp`; no `--spec-draft-model`.
- **Measured data preserved unchanged (not overwritten):** MTP-Q4_K_M n=11 tps 67.5 @32768; UD-Q4_K_XL n=14 tps 65.0 @131072; coding-10 rejected (VRAM kill 7913 MB / 7931 MB, preflight est 7584 MB) — lines 63–73 untouched.
- **GDN ngram warmup-death (recorded 2026-08-29):** ngram spec died in warmup on this GDN-hybrid arch; do not re-add without explicit A/B. Confirmed dead at warmup; the daily-driver recipe keeps the flag commented; no restoration without new A/B trial.
- **No truncation noted:** HF API responses fully read (base repo + GGUF repo); Unsloth docs fully read; local GGUF header verified via `GGUFReader` in prior session (not truncated). No missing source URLs.

## Open questions
- None for the operator host — UD coding axis closed as failure; MTP basename vector complete (weak). EXL3 side: 3 bpw base measured 49.6 t/s (+28 % vs 38.7) but packs ship without the MTP head and head self-quant is hardware-blocked on this rig — see [discovery engine doc](../discovery/fastest-tps-inference-engine.md) §3.1 and [session log](../sessions/2026-08-31-qwen35-9b-exl3-probe.md).
