# POCKET-35B — Model Card (Local)

Runtime scope: quality results remain in the [archived full card](https://github.com/allanschramm/local-model-autotuning/blob/ea43a71f48571811ec1ba25df4ab71ed272d34f1/docs/models/pocket-35b.md). Historical performance retains its original protocol.

**Source repo:** https://huggingface.co/FINAL-Bench/POCKET-35B-GGUF
**Publisher blog:** https://huggingface.co/blog/FINAL-Bench/pocket
**Base model:** https://huggingface.co/FINAL-Bench/Darwin-36B-Opus (VIDRAFT Darwin lineage → Qwen3.5-family MoE)
**License:** Apache-2.0
**Local file:** `models/FINAL-Bench/pocket-35b-gguf/POCKET-35B-Q3_K_M.gguf` (15.61 GiB / 16.76 GB) — **GGUF purged (historical); architecture/measured rows below are from the 2026-07-26 `gguf.GGUFReader` capture, preserved verbatim (see Sources).
**Family:** POCKET (FINAL-Bench / VIDRAFT)
**Architecture type:** MoE (`qwen35moe`) — 35B total / ~3B active
**Quantization:** `Q3_K_M` (`general.file_type=12`)

## Architecture (from GGUF metadata)

Verified with `gguf.GGUFReader` on the local file (2026-07-26):

| Key | Value |
| --- | --- |
| `general.architecture` | `qwen35moe` |
| `general.name` | `Model` (publisher left generic) |
| `qwen35moe.block_count` | 40 |
| `qwen35moe.context_length` | 262144 |
| `qwen35moe.embedding_length` | 2048 |
| `qwen35moe.expert_count` | 256 |
| `qwen35moe.expert_used_count` | 8 |
| `qwen35moe.attention.head_count` | 16 Q |
| `qwen35moe.attention.head_count_kv` | 2 KV |
| `qwen35moe.attention.key_length` / `value_length` | 256 / 256 |
| `qwen35moe.rope.freq_base` | 10,000,000 |
| `qwen35moe.full_attention_interval` | 4 (hybrid full-attn + DeltaNet/SSM) |
| `qwen35moe.ssm.*` | conv_kernel=4, state=128, group=16, time_step_rank=32, inner=4096 |
| `qwen35moe.nextn_predict_layers` | **0** (no MTP heads) |
| tensors | 733 |

Harness `is_moe_model` → **True** (`expert_count > 1`). Shared-expert tensors (`*_shexp`) present.

**Note:** Darwin-36B-Opus HF card claims 24Q/4KV; this GGUF matches the standard Qwen3.5-35B-A3B layout (16Q/2KV). Trust the GGUF.

## Hardware requirements (discrete 8 GB-class NVIDIA + ~32 GB RAM)

| Quant | Size | Publisher target | Our pick |
| --- | --- | --- | --- |
| IQ1_M | 8.2 GB | 16 GB RAM box | no — quality cliff |
| Q2_K | 13 GB | mini-PC / no-GPU daily driver | skip — GPU rig can afford better |
| **Q3_K_M** | **~16.8 GB** | PC ~24 GB RAM | **yes** — under 20 GB disk, fits RAM with VITRIOL |
| Q4_K_M | 21 GB | PC 32 GB RAM | optional later quality A/B |

Publisher GPQA-Diamond (greedy): Q4_K_M 68.7% · Q2_K 60.1% (Q3_K_M not published; expect between).

- MoE: experts on CPU via `N_CPU_MOE=None` → auto `--n-cpu-moe 40`.
- Dense spill forbidden — not applicable (MoE).
- Prefer ctx ≤65k + KV `q4_0` on 8 GB until validated higher.

## Recommended settings

POCKET GGUF has **no** embedded `general.sampling.*`. Seed from Darwin-36B-Opus (base) + Qwen3.5-family defaults:

| Param | Agentic / general | Notes |
| --- | --- | --- |
| temperature | **0.6** | Darwin: 0.6–0.7 reasoning; greedy=0.0 for GPQA |
| top_p | 0.95 | Qwen3.5 lineage (Darwin silent) |
| top_k | 20 | Qwen3.5 lineage |
| min_p | 0.0 |  |
| presence_penalty | 0.0 |  |
| repeat_penalty | 1.0 | off |
| chat template | Qwen `< | im_start | >` + thinking | Darwin: `<think>` auto-inserted |

Coding / precise: keep same sampler until a quality pass; Darwin does not publish a separate coding profile.

### Reasoning control (think template)

The `qwen35moe` chat template reads **only `enable_thinking`** — there is **no `reasoning_effort` variable** in this family's template (see Sources, embedded tokenizer + `config.chat_template_jinja`).

| Flag / lever | Effect on POCKET-35B | Seed |
| --- | --- | --- |
| `--reasoning on` | `enable_thinking=true` → default render injects `<think>…</think>` block (thinking ON) | agentic / general — **Baseline `REASONING=True`** |
| `--reasoning off` | `enable_thinking=false` → suppresses thinking, answers directly | coding / precise |
| `--reasoning-budget N` | **server-side, template-independent** — effective on this card | agentic (e.g. 4096) |
| `--reasoning-effort` | **SILENT NO-OP** here (`reasoning_effort` not read by this template — different from Qwen3.8-27B which does read it) | do not use |
| `REASONING_PRESERVE` | `True` where `/props` reports `supports_preserve_reasoning` (Ornith-1.5 GGUFs verified) — **TBD for this card: GGUF purged, `/props` cannot be re-probed** | **TBD** |

Working levers for this GGUF: `--reasoning on/off` + `--reasoning-budget N` (+ `REASONING_PRESERVE` once re-verified).


## MTP

`nextn_predict_layers = 0`. No MTP tensors. Leave `SPEC_TYPE=None`.

## VITRIOL split

Experts do not fit physical VRAM — full expert offload:

```text
--n-gpu-layers 99 --n-cpu-moe 40
```

Baseline `N_CPU_MOE=None` auto-resolves to `block_count` (40). See [vitriol-technique.md](../discovery/vitriol-technique.md).

## Our config baseline (validated 2026-07-26)

```python
MODEL = 'POCKET-35B-Q3_K_M.gguf'
CTX_SIZE = 65536
KV_CACHE_K = 'q4_0'
KV_CACHE_V = 'q4_0'
BATCH_SIZE = 512
UBATCH_SIZE = 128
THREADS = 8
THREADS_BATCH = 8
FLASH_ATTN = 'on'
N_CPU_MOE = None  # → 40
NO_MMAP = True
CONT_BATCHING = True
JINJA = True
SPEC_TYPE = None
TPS_FLOOR = 15.0
TEMP = 0.6
TOP_P = 0.95
TOP_K = 20
```

| Metric | Value | Notes |
| --- | --- | --- |
| Bench tg (`mmap` baseline) | **33.7 t/s** | Peak VRAM 3.4 GB |
| Bench tg (`--no-mmap`) | **37.4 t/s** | +11.0% speedup |
| Bench tg (`--no-mmap --mlock`) | **38.1 t/s** | **+13.1% speedup**, peak VRAM 3.4 GB |
| Session | [2026-07-26-pocket-35b-pipeline.md](../performance/2026-07-26-pocket-35b-pipeline.md) |  |


## Sources / Verification (refreshed 2026-08-29; GGUF file PURGED — historical card)

- HF GGUF repo: https://huggingface.co/FINAL-Bench/POCKET-35B-GGUF (`lastModified=2026-07-29`; API `gguf.architecture=qwen35moe`, `context_length=262144`, `totalFileSize=8239208384`; `license=apache-2.0`; 564K downloads; tags `qwen3_5_moe`/`darwin`/`reasoning`/`advanced-reasoning`/`thinking`/`chain-of-thought`) — extracted 2026-08-29.
- HF README.md (benchmark table + build picks + lineage): https://huggingface.co/FINAL-Bench/POCKET-35B-GGUF/raw/main/README.md (extracted 2026-08-29) — GPQA-Diamond greedy: Q4_K_M 68.7% · Q2_K 60.1% (Q3_K_M between); build table `Q3_K_M`=16 GB / 3.4 bpw; `Q2_K`=13 GB = daily driver; lineage from Darwin-36B-Opus.
- HF blog (measurement method + honest prefill loss): https://huggingface.co/blog/FINAL-Bench/pocket (extracted 2026-08-29) — CPU 27.0 tok/s / GPU 197 tok/s / MacBook M3 Pro 25.4 tok/s; prefill 0.41× vs Bonsai honestly reported.
- Base model / settings (Darwin-36B-Opus, `qwen3_5_moe`): https://huggingface.co/FINAL-Bench/Darwin-36B-Opus (`lastModified=2026-07-25`; tags include `reasoning`, `advanced-reasoning`, `thinking`; `chat_template_jinja` = same qwen35 template; GPQA 88.4% — not verified locally) — source for sampler 0.6–0.7.
- Publisher settings / recommended: https://huggingface.co/blog/FINAL-Bench/pocket (2026-07-23) — Darwin TEMP 0.6–0.7, greedy=0.0 GPQA.
- Think-template / reasoning documentation (embedded tokenizer + `config.chat_template_jinja` from HF + verified local `gguf_dump` notes, 2026-08-29): `qwen35moe` family reads **only `enable_thinking`; `<think>…</think>` auto-inserted when true (default); suppress with `enable_thinking=false`; `preserve_thinking` supported; `reasoning_effort` variable **absent** → `--reasoning-effort` is silent NO-OP on this card (see Reasoning control below; consistent with verified reasoning-template facts for `qwen35` family).
- Local GGUF header via `gguf.GGUFReader` (2026-07-26) — `block_count=40`, `context_length=262144`, `expert_count=256`, `expert_used_count=8`, `nextn_predict_layers=0`, `architecture=qwen35moe`. GGUF file **purged** since 2026-07-26 run; card kept as historical reference.

## Open questions

- Whether partial expert GPU (`N_CPU_MOE < 40`) ever beats full offload here (Ourbox tip: full CPU experts often faster on 8 GB)
- TPS hill-climb (`autoloop --mode tps`) — KV / batch / ctx knobs
- Q4_K_M quality A/B vs this Q3_K_M (RAM headroom exists)
