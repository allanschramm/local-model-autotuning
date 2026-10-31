# KAT-Coder-V2.5-Dev — Model Card (Local)

Runtime scope: quality results remain in the [archived full card](https://github.com/allanschramm/local-model-autotuning/blob/ea43a71f48571811ec1ba25df4ab71ed272d34f1/docs/models/kat-coder-v2.5-dev.md). Historical performance retains its original protocol.

**Source (weights):** https://huggingface.co/Kwaipilot/KAT-Coder-V2.5-Dev
**GGUF:** https://huggingface.co/bartowski/Kwaipilot_KAT-Coder-V2.5-Dev-GGUF
**Base:** Qwen3.6-35B-A3B (post-train SFT+RL)
**License:** Apache-2.0
**Local file:** `models/bartowski/kwaipilot-kat-coder-v2.5-dev-gguf/Kwaipilot_KAT-Coder-V2.5-Dev-IQ4_XS.gguf` (17.51 GiB)
**Family:** KAT-Coder (Kwaipilot)
**Architecture type:** MoE (`qwen35moe`) — 35B total / ~3B active
**Quantization:** `IQ4_XS` (`general.file_type=30`)

## Architecture (from GGUF metadata)

Verified with `gguf.GGUFReader` on the local file (2026-07-27):

| Key | Value |
| --- | --- |
| `general.architecture` | `qwen35moe` |
| `general.name` | `KAT Coder V2.5 Dev` |
| `qwen35moe.block_count` | 40 |
| `qwen35moe.context_length` | 262144 |
| `qwen35moe.embedding_length` | 2048 |
| `qwen35moe.expert_count` | 256 |
| `qwen35moe.expert_used_count` | 8 |
| `qwen35moe.full_attention_interval` | 4 |
| tensors | 733 |

Harness `is_moe_model` → **True**. Same layout class as Ornith-35B / POCKET-35B.

## Hardware requirements (discrete 8 GB-class NVIDIA)

| Quant | Size | Pick |
| --- | --- | --- |
| **IQ4_XS** | **~18.8 GB** | **yes** — under 20 GB disk; MoE + VITRIOL |
| Q4_K_M | ~21.4 GB | over user disk budget |
| Q5+ | ≥24 GB | skip on the operator host |

- Experts: `N_CPU_MOE=None` → auto `--n-cpu-moe 40`.
- Prefer ctx **65k** + KV `q4_0` on 8 GB (same band as Ornith-35B agentic).

## Recommended settings

From Kwaipilot HF card (thinking / agentic default; SWE-bench Verified footnote uses TEMP 1.0 / TOP_P 0.95):

| Param | Agentic / thinking (default) | Instruct / non-thinking |
| --- | --- | --- |
| temperature | **1.0** | 0.7 |
| top_p | **0.95** | 0.8 |
| top_k | **20** | 20 |
| presence_penalty | **1.5** | 1.5 |
| min_p | 0.0 | 0.0 |
| repeat_penalty | 1.0 | 1.0 |
| thinking | on by default | `enable_thinking=False` |

Seed `SAMPLER_DEFAULTS` from **agentic / thinking** for Claw + coding-10 unless an explicit quality pass says otherwise.

## MTP

No `nextn_predict_layers` field in this GGUF header (same 733-tensor layout as Ornith-35B without MTP). Leave `SPEC_TYPE=None`.

## VITRIOL split

```text
--n-gpu-layers 99 --n-cpu-moe 40
```

Baseline `N_CPU_MOE=None` auto-resolves to `block_count` (40).

## Our config baseline (validated 2026-07-27)

```python
MODEL = 'Kwaipilot_KAT-Coder-V2.5-Dev-IQ4_XS.gguf'
CTX_SIZE = 65536
KV_CACHE_K = 'q4_0'
KV_CACHE_V = 'q4_0'
N_CPU_MOE = None  # → 40
TEMP = 1.0
TOP_P = 0.95
TOP_K = 20
PRESENCE_PENALTY = 1.5
TPS_FLOOR = 15.0
VRAM_LIMIT_MB = 7900
```

## Sources / Verification

- HF card `Kwaipilot/KAT-Coder-V2.5-Dev` (https://huggingface.co/Kwaipilot/KAT-Coder-V2.5-Dev) — model card, archived publisher evaluation details, eval config (temp 1.0 / top_p 0.95, 256k ctx, `--reasoning-parser qwen3`; `agent=claude_code@2.1.195`) — **extracted 2026-08-29** (API + README; image/table truncated at ~300 lines, full image referenced by URL `KAT-Coder-V2.5-Dev-Benchmarks.png`)
- HF API `https://huggingface.co/api/models/Kwaipilot/KAT-Coder-V2.5-Dev` (tags: `qwen3_5_moe`, `agentic-coding`, `code`; license `apache-2.0`; base_model `Qwen3.6-35B-A3B`; downloads 49K) — **2026-08-29**
- Bartowski GGUF repo https://huggingface.co/bartowski/Kwaipilot_KAT-Coder-V2.5-Dev-GGUF (quant `IQ4_XS`; file `Kwaipilot_KAT-Coder-V2.5-Dev-IQ4_XS.gguf`) — **2026-08-29**
- Publisher thinking / reasoning claim: `qwen3` reasoning parser (`--reasoning-parser qwen3` in SGLang + vLLM serve commands), 256k context, default agent temp 1.0 / top_p 0.95; vision tower omitted (text-only release). Thinking mode is the publisher's default evaluation profile.
- Local GGUF via `gguf.GGUFReader` 2026-07-27 (GGUF purged since; architecture / block_count / expert_count kept as historical verification)
- Bartowski quant table (IQ4_XS size) — **2026-08-29**
- Local pipeline `results.tsv` 2026-07-27 (store best validated: `n=4`, smoke / **0.6000** full @65536, tb **30.2** t/s, peak VRAM **3.4 GB; IQ4_XS 17.51 GiB file)

## Reasoning control

**Publisher claim:** KAT-Coder-V2.5-Dev is a Qwen3.5-MoE-based thinking / agentic-coding model; publisher serves with `--reasoning-parser qwen3` (SGLang `sglang.launch_server --reasoning-parser qwen3 --context-length 262144`; vLLM `vllm serve --reasoning-parser qwen3 --max-model-len 262144 --language-model-only`) and evaluates at **temp 1.0 / top_p 0.95 / 256k ctx** with `agent=claude_code@2.1.195`. Thinking is the **default profile; no `disable_think` flag is documented.

Do not compare to any pre-2026-08-08 KAT row.

**Reasoning content handling:** like all `qwen35moe` family GGUFs (Ornith-1.5, Qwen3.6-35B-MTP, Qwen3.8-4B-Distill), the embedded tokenizer template reads `enable_thinking`; `--reasoning-effort` is a **silent NO-OP** on these GGUFs — working levers are `REASONING` (`--reasoning on/off`) and `REASONING_BUDGET` (`--reasoning-budget N`). `REASONING_PRESERVE` applies only when `/props` reports `supports_preserve_reasoning` (verified true for Ornith-1.5 GGUFs; **not verified for this purged GGUF** — mark TBD below). For historical reference: publisher's own eval uses `temperature=1.0`, not reasoning-budget control.

**Our baseline (historical, GGUF purged):** `REASONING=on` / `REASONING_BUDGET` unmeasured (default template effort) / `REASONING_PRESERVE=None` (unverified). No `REASONING_EFFORT` key applicable.

## Open questions

- **TBD (2026-08-29):** Re-verify `supports_preserve_reasoning` from a fresh download if the GGUF is re-obtained (current file purged; historical `GGUFReader` 2026-07-27 did not record `/props` preserve flag). Remove TBD + this row once verified.
- **TBD (2026-08-29):** Whether a newer `qwen3` reasoning-parser revision (post-2026-08-29) changes the default `reasoning_effort` ladder (currently xhigh default per 27B open-source template — different family; KAT uses Qwen3.5-MoE template, not 27B). Remove TBD once confirmed.
Historical IQ4_XS runtime evidence: configured context 65k, 30.2 t/s and 3.4 GB peak VRAM. See the archived card for the complete protocol and artifact identity.

## Additional historical runtime evidence

| Axis | Detail |
| --- | --- |
| Historical run runtime | peak VRAM **3.4 GB** |
| Historical run runtime | peak **3.3 GB** |
