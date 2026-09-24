# Qwen3.6-35B-A3B - model card and GGUF inventory

## Status

Publisher inventory refreshed 2026-08-29 (HF fetch). Measured: no-spec Objective Vector @ 100k turbo3 (2026-08-02); Q4 DFlash/MTP/none @ 32k (2026-08-07); Q3 DFlash/MTP/none @ 65k (2026-08-12); store points UD-Q3_K_XL / UD-Q4_K_XL / MTP-GGUF file (verified from results.db, 2026-08-29). Publisher facts and local facts are kept separate.

**Inventory date:** 2026-08-29 (HF fetch); speed matrices 2026-08-07 (Q4) and 2026-08-12 (Q3); store audit 2026-08-29.

**Official source:** [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
**Unsloth base GGUF:** [unsloth/Qwen3.6-35B-A3B-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF)
**Unsloth MTP GGUF:** [unsloth/Qwen3.6-35B-A3B-MTP-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF)
**Unsloth docs:** [Qwen3.6](https://unsloth.ai/docs/models/qwen3.6) and [MTP](https://unsloth.ai/docs/models/mtp)
**Official llama.cpp GGUF:** [ggml-org/Qwen3.6-35B-A3B-GGUF](https://huggingface.co/ggml-org/Qwen3.6-35B-A3B-GGUF)
**Official llama.cpp MTP GGUF:** [ggml-org/Qwen3.6-35B-A3B-MTP-GGUF](https://huggingface.co/ggml-org/Qwen3.6-35B-A3B-MTP-GGUF)
**License:** Apache-2.0 on the official Qwen repo and the Unsloth, ggml-org, and Bartowski base-GGUF repos. Fine-tune license metadata is called out separately below.
**Local basename evidence (2026-08-29, gguf_dump verified):** `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` in **unsloth/Qwen3.6-35B-A3B-GGUF** (non-MTP, 21.9 GB, **no nextn tensors** — 0 `/nextn`) and `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` in **unsloth/Qwen3.6-35B-A3B-MTP-GGUF** (MTP, 22.7 GB, **4 nextn tensors**: `blk.40.nextn.{eh_proj,enorm,hnorm,shared_head_norm}`). The MTP-GGUF repo is the provenance for nextn tensors.
**Local absolute file:** **Not recorded in tracked docs.** Runtime resolves the selected basename; no machine-specific path is stored here.
**Symlink:** **Not recorded in tracked docs.**
**Family:** Qwen3.6 hybrid MoE (text and vision-capable publisher checkpoints; a usable vision GGUF also needs a matching `mmproj` sidecar). 41 GGUF blocks (`qwen35moe`, `block_count=41`), 256 routed experts with 8 active per MoE block, native ctx 262,144; trained with multi-step MTP.
**Quantization coverage:** BF16, Q8_0, Q6/Q5/Q4 K-quants, IQ quants, MXFP4_MOE, Unsloth UD, and specialist MoE mixtures are listed below. None is labelled QAT by the inspected publishers.

## Architecture

### Verified GGUF header evidence (2026-08-02)

The official Qwen card reports 35B total parameters and about 3B active parameters, 40 layers, hidden size 2048, vocabulary 248,320, 256 routed experts with 8 routed experts plus 1 shared expert per MoE block, and a repeating hybrid layout of three gated DeltaNet blocks followed by one gated-attention block (each followed by MoE). DeltaNet uses 32 value heads and 16 query/key heads with head dimension 128; gated attention uses 16 query heads and 2 KV heads with head dimension 256. Rotary dimension is 64. Native context is 262,144 tokens and the card documents extension to 1,010,000 tokens. The model was trained with multi-step MTP.

### Verified GGUF header evidence (2026-08-02)

The project venv `gguf_dump` reported the following header values for the two selected basenames. These are basename-level facts; machine-specific paths are intentionally outside this tracked card. **The two local GGUFs share `general.architecture=qwen35moe`, `block_count=41`, `context_length=262144`, `head_count_kv=2`, `expert_count=256`, `expert_used_count=8`; they differ in whether nextn head tensors are present — see MTP section.**
|---|---|---|
| `general.name` | `Qwen3.6-35B` | `Qwen3.6 35B` |
| `general.architecture` | `qwen35moe` | `qwen35moe` |
| `tensor_count` | 753 | 753 |
| `block_count` | 41 | 41 |
| `context_length` | 262144 | 262144 |
| `head_count_kv` | 2 | 2 |
| `expert_count` | 256 | 256 |
| `expert_used_count` | 8 | 8 |
| `nextn_predict_layers` | 0 | 1 |
| `file_type` | 15 | 15 |
These headers establish MoE (`expert_count=256`) for both inspected basenames. The MTP-preserving claim is established by tensor-name audit only — see MTP and draft packaging.

## Public GGUF inventory

Sizes below are the publisher/Hugging Face display values at extraction time. They are rounded and can change when a repository is rebuilt. `imatrix` and `mmproj` files are sidecars, not standalone language-model targets.

### Unsloth base GGUF (no MTP-specific repository packaging)

Source: [base tree](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/tree/main). The `UD-*` files are Unsloth Dynamic 2.0 post-training quants; the repository does not label them QAT.

| Basename | Quant | Reported size |
|---|---:|---:|
| `Qwen3.6-35B-A3B-MXFP4_MOE.gguf` | MXFP4_MOE | 21.7 GB |
| `Qwen3.6-35B-A3B-Q8_0.gguf` | Q8_0 | 36.9 GB |
| `Qwen3.6-35B-A3B-UD-IQ1_M.gguf` | UD-IQ1_M | 10 GB |
| `Qwen3.6-35B-A3B-UD-IQ2_M.gguf` | UD-IQ2_M | 11.5 GB |
| `Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf` | UD-IQ2_XXS | 10.8 GB |
| `Qwen3.6-35B-A3B-UD-IQ3_S.gguf` | UD-IQ3_S | 13.7 GB |
| `Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf` | UD-IQ3_XXS | 13.2 GB |
| `Qwen3.6-35B-A3B-UD-IQ4_NL.gguf` | UD-IQ4_NL | 18 GB |
| `Qwen3.6-35B-A3B-UD-IQ4_NL_XL.gguf` | UD-IQ4_NL_XL | 19.5 GB |
| `Qwen3.6-35B-A3B-UD-IQ4_XS.gguf` | UD-IQ4_XS | 17.7 GB |
| `Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf` | UD-Q2_K_XL | 12.3 GB |
| `Qwen3.6-35B-A3B-UD-Q3_K_M.gguf` | UD-Q3_K_M | 16.6 GB |
| `Qwen3.6-35B-A3B-UD-Q3_K_S.gguf` | UD-Q3_K_S | 15.4 GB |
| `Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf` | UD-Q3_K_XL | 16.8 GB |
| `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` | UD-Q4_K_M | 22.1 GB |
| `Qwen3.6-35B-A3B-UD-Q4_K_S.gguf` | UD-Q4_K_S | 20.9 GB |
| `Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf` | UD-Q4_K_XL | 22.4 GB |
| `Qwen3.6-35B-A3B-UD-Q5_K_M.gguf` | UD-Q5_K_M | 26.5 GB |
| `Qwen3.6-35B-A3B-UD-Q5_K_S.gguf` | UD-Q5_K_S | 24.9 GB |
| `Qwen3.6-35B-A3B-UD-Q5_K_XL.gguf` | UD-Q5_K_XL | 26.6 GB |
| `Qwen3.6-35B-A3B-UD-Q6_K.gguf` | UD-Q6_K | 29.3 GB |
| `Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf` | UD-Q6_K_XL | 31.8 GB |
| `Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf` | UD-Q8_K_XL | 38.5 GB |

Sidecars in the same repository: `imatrix_unsloth.gguf_file` (192 MB), `mmproj-BF16.gguf` (903 MB), `mmproj-F16.gguf` (899 MB), and `mmproj-F32.gguf` (1.79 GB). The base repository does not expose an `mtp-*` model basename.

### Unsloth MTP GGUF

Source: [MTP tree](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/tree/main). These are the publisher's MTP-preserving targets; there is no separate `mtp-*` file inside this repository. The target GGUF and the `draft-mtp` runtime path must still be checked locally.

| Basename | Quant | Reported size |
|---|---:|---:|
| `Qwen3.6-35B-A3B-MXFP4_MOE.gguf` | MXFP4_MOE | 22.2 GB |
| `Qwen3.6-35B-A3B-Q8_0.gguf` | Q8_0 | 37.8 GB |
| `Qwen3.6-35B-A3B-UD-IQ1_M.gguf` | UD-IQ1_M | 11.4 GB |
| `Qwen3.6-35B-A3B-UD-IQ2_M.gguf` | UD-IQ2_M | 11.9 GB |
| `Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf` | UD-IQ2_XXS | 11.8 GB |
| `Qwen3.6-35B-A3B-UD-IQ3_S.gguf` | UD-IQ3_S | 15.3 GB |
| `Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf` | UD-IQ3_XXS | 14.1 GB |
| `Qwen3.6-35B-A3B-UD-IQ4_NL.gguf` | UD-IQ4_NL | 18.5 GB |
| `Qwen3.6-35B-A3B-UD-IQ4_XS.gguf` | UD-IQ4_XS | 18.2 GB |
| `Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf` | UD-Q2_K_XL | 12.6 GB |
| `Qwen3.6-35B-A3B-UD-Q3_K_M.gguf` | UD-Q3_K_M | 17.1 GB |
| `Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf` | UD-Q3_K_XL | 17.2 GB |
| `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` | UD-Q4_K_M | 22.7 GB |
| `Qwen3.6-35B-A3B-UD-Q4_K_S.gguf` | UD-Q4_K_S | 21.4 GB |
| `Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf` | UD-Q4_K_XL | 22.9 GB |
| `Qwen3.6-35B-A3B-UD-Q5_K_M.gguf` | UD-Q5_K_M | 27.1 GB |
| `Qwen3.6-35B-A3B-UD-Q5_K_S.gguf` | UD-Q5_K_S | 25.5 GB |
| `Qwen3.6-35B-A3B-UD-Q5_K_XL.gguf` | UD-Q5_K_XL | 27.2 GB |
| `Qwen3.6-35B-A3B-UD-Q6_K.gguf` | UD-Q6_K | 30 GB |
| `Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf` | UD-Q6_K_XL | 32.6 GB |
| `Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf` | UD-Q8_K_XL | 39.1 GB |

Sidecars: `imatrix_unsloth.gguf_file` (192 MB), `mmproj-BF16.gguf` (903 MB), `mmproj-F16.gguf` (899 MB), and `mmproj-F32.gguf` (1.79 GB). The Unsloth card recommends upstream `llama-server` with `--spec-type draft-mtp --spec-draft-n-max 2`, `-np 1`; its MTP path does not support `-np > 1` or `--mmproj`.

### Official ggml-org artifacts

Source: [base tree](https://huggingface.co/ggml-org/Qwen3.6-35B-A3B-GGUF/tree/main).

| Basename | Role / quant | Reported size |
|---|---|---:|
| `Qwen3.6-35B-A3B-BF16.gguf` | base BF16 | 69.4 GB |
| `Qwen3.6-35B-A3B-Q4_K_M.gguf` | base Q4_K_M | 20.4 GB |
| `Qwen3.6-35B-A3B-Q8_0.gguf` | base Q8_0 | 36.9 GB |
| `dflash-Qwen3.6-35B-A3B-BF16.gguf` | DFlash draft, BF16 | 783 MB |
| `dflash-Qwen3.6-35B-A3B-Q8_0.gguf` | DFlash draft, Q8_0 | 421 MB |
| `mmproj-Qwen3.6-35B-A3B-BF16.gguf` | vision sidecar | 903 MB |
| `mmproj-Qwen3.6-35B-A3B-Q8_0.gguf` | vision sidecar | 614 MB |
| `mtp-Qwen3.6-35B-A3B-BF16.gguf` | separate MTP artifact, BF16 | 3.74 GB |
| `mtp-Qwen3.6-35B-A3B-Q4_0.gguf` | separate MTP artifact, Q4_0 | 1.06 GB |
| `mtp-Qwen3.6-35B-A3B-Q8_0.gguf` | separate MTP artifact, Q8_0 | 1.99 GB |

Source: [MTP-only tree](https://huggingface.co/ggml-org/Qwen3.6-35B-A3B-MTP-GGUF/tree/main).

| Basename | Role / quant | Reported size |
|---|---|---:|
| `Qwen3.6-35B-A3B-MTP-BF16.gguf` | integrated MTP BF16 target | 71.1 GB |
| `Qwen3.6-35B-A3B-MTP-Q8_0.gguf` | integrated MTP Q8_0 target | 37.8 GB |
| `mmproj-Qwen3.6-35B-A3B-Q8_0.gguf` | vision sidecar | 614 MB |

The official MTP card uses `--spec-type draft-mtp --spec-draft-n-max 2` or `3` and points to [llama.cpp PR #22673](https://github.com/ggml-org/llama.cpp/pull/22673), which is merged. The `dflash-*` files are DFlash artifacts, not MTP files.

### Bartowski GGUF

Source: [bartowski/Qwen_Qwen3.6-35B-A3B-GGUF tree](https://huggingface.co/bartowski/Qwen_Qwen3.6-35B-A3B-GGUF/tree/main). License shown by the repository: Apache-2.0. These are standard K/I quants; no QAT label is supplied.

| Basename | Quant | Reported size |
|---|---:|---:|
| `Qwen_Qwen3.6-35B-A3B-IQ1_M.gguf` | IQ1_M | 9.42 GB |
| `Qwen_Qwen3.6-35B-A3B-IQ2_M.gguf` | IQ2_M | 13 GB |
| `Qwen_Qwen3.6-35B-A3B-IQ2_S.gguf` | IQ2_S | 11.9 GB |
| `Qwen_Qwen3.6-35B-A3B-IQ2_XS.gguf` | IQ2_XS | 11.7 GB |
| `Qwen_Qwen3.6-35B-A3B-IQ2_XXS.gguf` | IQ2_XXS | 10.7 GB |
| `Qwen_Qwen3.6-35B-A3B-IQ3_M.gguf` | IQ3_M | 17.8 GB |
| `Qwen_Qwen3.6-35B-A3B-IQ3_XS.gguf` | IQ3_XS | 17.1 GB |
| `Qwen_Qwen3.6-35B-A3B-IQ3_XXS.gguf` | IQ3_XXS | 15.8 GB |
| `Qwen_Qwen3.6-35B-A3B-IQ4_NL.gguf` | IQ4_NL | 20.8 GB |
| `Qwen_Qwen3.6-35B-A3B-IQ4_XS.gguf` | IQ4_XS | 19.7 GB |
| `Qwen_Qwen3.6-35B-A3B-Q2_K.gguf` | Q2_K | 13.5 GB |
| `Qwen_Qwen3.6-35B-A3B-Q2_K_L.gguf` | Q2_K_L | 14 GB |
| `Qwen_Qwen3.6-35B-A3B-Q3_K_L.gguf` | Q3_K_L | 17.8 GB |
| `Qwen_Qwen3.6-35B-A3B-Q3_K_M.gguf` | Q3_K_M | 17.1 GB |
| `Qwen_Qwen3.6-35B-A3B-Q3_K_S.gguf` | Q3_K_S | 16.4 GB |
| `Qwen_Qwen3.6-35B-A3B-Q3_K_XL.gguf` | Q3_K_XL | 18.2 GB |
| `Qwen_Qwen3.6-35B-A3B-Q4_0.gguf` | Q4_0 | 20.8 GB |
| `Qwen_Qwen3.6-35B-A3B-Q4_1.gguf` | Q4_1 | 22.9 GB |
| `Qwen_Qwen3.6-35B-A3B-Q4_K_L.gguf` | Q4_K_L | 22.7 GB |
| `Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf` | Q4_K_M | 22.3 GB |
| `Qwen_Qwen3.6-35B-A3B-Q4_K_S.gguf` | Q4_K_S | 21.5 GB |
| `Qwen_Qwen3.6-35B-A3B-Q5_K_L.gguf` | Q5_K_L | 26.2 GB |
| `Qwen_Qwen3.6-35B-A3B-Q5_K_M.gguf` | Q5_K_M | 25.9 GB |
| `Qwen_Qwen3.6-35B-A3B-Q5_K_S.gguf` | Q5_K_S | 25.1 GB |
| `Qwen_Qwen3.6-35B-A3B-Q6_K.gguf` | Q6_K | 30.9 GB |
| `Qwen_Qwen3.6-35B-A3B-Q6_K_L.gguf` | Q6_K_L | 31.2 GB |
| `Qwen_Qwen3.6-35B-A3B-Q8_0.gguf` | Q8_0 | 37.8 GB |

Sidecars/drafts: `Qwen_Qwen3.6-35B-A3B-imatrix.gguf` (192 MB), `mmproj-Qwen_Qwen3.6-35B-A3B-bf16.gguf` (903 MB), `mmproj-Qwen_Qwen3.6-35B-A3B-f16.gguf` (899 MB), `mtp-Qwen_Qwen3.6-35B-A3B-Q4_0.gguf` (1.19 GB), and `mtp-Qwen_Qwen3.6-35B-A3B-Q8_0.gguf` (1.99 GB). The `mtp-*` files are separate basenames; inspect GGUF metadata before treating them as integrated MTP targets.

### AesSedai specialist MoE quants

Source: [AesSedai/Qwen3.6-35B-A3B-GGUF](https://huggingface.co/AesSedai/Qwen3.6-35B-A3B-GGUF). The card describes fused/imatrixed specialist MoE mixtures and says its 2026-05-18 update added MTP tensors to the quants. The repository uses one folder and two shards per quant; the sizes below are the card's GiB values.

| Exact shard basenames | Mixture / card size |
|---|---:|
| `Qwen3.6-35B-A3B-IQ3_S-00001-of-00002.gguf`, `Qwen3.6-35B-A3B-IQ3_S-00002-of-00002.gguf` | IQ3_S, 13.48 GiB |
| `Qwen3.6-35B-A3B-IQ4_XS-00001-of-00002.gguf`, `Qwen3.6-35B-A3B-IQ4_XS-00002-of-00002.gguf` | IQ4_XS, 17.23 GiB |
| `Qwen3.6-35B-A3B-Q4_K_M-00001-of-00002.gguf`, `Qwen3.6-35B-A3B-Q4_K_M-00002-of-00002.gguf` | Q4_K_M, 21.45 GiB |
| `Qwen3.6-35B-A3B-Q5_K_M-00001-of-00002.gguf`, `Qwen3.6-35B-A3B-Q5_K_M-00002-of-00002.gguf` | Q5_K_M, 25.28 GiB |
| `Qwen3.6-35B-A3B-Q6_K-00001-of-00002.gguf`, `Qwen3.6-35B-A3B-Q6_K-00002-of-00002.gguf` | Q6_K, 27.93 GiB |

Sidecars: `imatrix.gguf` (192 MB), `mmproj-Qwen3.6-35B-A3B-BF16.gguf` (903 MB), `mmproj-Qwen3.6-35B-A3B-F16.gguf` (899 MB), and `mmproj-Qwen3.6-35B-A3B-F32.gguf` (1.79 GB). License: **apache-2.0** (AesSedai repo card, verified 2026-08-02). MTP keys were confirmed locally on 2026-08-02 (`blk.40.nextn.*` tensors); llama.cpp `draft-mtp` support confirmed via source — see MTP section.

## Fine-tune and uncensored candidates

These are separate model families, not interchangeable with the base Qwen checkpoint. They should not become the baseline without an explicit selection and a complete Objective Vector.

### Hesamation reasoning distill

Source: [hesamation/Qwen3.6-35B-A3B-Claude-4.6-Opus-Reasoning-Distilled-GGUF](https://huggingface.co/hesamation/Qwen3.6-35B-A3B-Claude-4.6-Opus-Reasoning-Distilled-GGUF). The card describes an Apache-2.0 text-only SFT fine-tune and supplies no distinct sampler profile.

| Basename | Quant | Reported size |
|---|---:|---:|
| `Qwen3.6-35B-A3B-Claude-4.6-Opus-Reasoning-Distilled.Q4_K_M.gguf` | Q4_K_M | 21.2 GB |
| `Qwen3.6-35B-A3B-Claude-4.6-Opus-Reasoning-Distilled.Q5_K_M.gguf` | Q5_K_M | 24.7 GB |
| `Qwen3.6-35B-A3B-Claude-4.6-Opus-Reasoning-Distilled.Q6_K.gguf` | Q6_K | 28.5 GB |
| `Qwen3.6-35B-A3B-Claude-4.6-Opus-Reasoning-Distilled.Q8_0.gguf` | Q8_0 | size not surfaced by the inspected tree |

No MTP or draft package is documented for this fine-tune; do not assume MTP merely because the base model supports it. The card's llama.cpp example is ordinary GGUF loading.

### Bahushruth abliterated v4

Source: [Bahushruth/Qwen3.6-35B-A3B-abliterated-v4-GGUF](https://huggingface.co/Bahushruth/Qwen3.6-35B-A3B-abliterated-v4-GGUF). Standard files are converted with `--no-mtp`; the card separately publishes a BF16 MTP file and asks for a recent llama.cpp with `--spec-type draft-mtp --spec-draft-n-max 1`. Sizes are the card's rounded values.

| Basename | Quant / packaging | Reported size |
|---|---|---:|
| `Qwen3.6-35B-A3B-abliterated-v4-BF16.gguf` | BF16, standard | 69.4 GB |
| `Qwen3.6-35B-A3B-abliterated-v4-Q8_0.gguf` | Q8_0, standard | 36.9 GB |
| `Qwen3.6-35B-A3B-abliterated-v4-Q6_K.gguf` | Q6_K, standard | 28.5 GB |
| `Qwen3.6-35B-A3B-abliterated-v4-Q5_K_M.gguf` | Q5_K_M, standard | 24.7 GB |
| `Qwen3.6-35B-A3B-abliterated-v4-Q4_K_M.gguf` | Q4_K_M, standard | 21.2 GB |
| `Qwen3.6-35B-A3B-abliterated-v4-IQ4_XS.gguf` | IQ4_XS, standard | 18.7 GB |
| `Qwen3.6-35B-A3B-abliterated-v4-IQ4_NL.gguf` | IQ4_NL, standard | 19.8 GB |
| `Qwen3.6-35B-A3B-abliterated-v4-Q3_K_M.gguf` | Q3_K_M, standard | 16.8 GB |
| `Qwen3.6-35B-A3B-abliterated-v4-IQ3_M.gguf` | IQ3_M, standard | 15.4 GB |
| `Qwen3.6-35B-A3B-abliterated-v4-IQ3_XXS.gguf` | IQ3_XXS, standard | 13.6 GB |
| `Qwen3.6-35B-A3B-abliterated-v4-Q2_K.gguf` | Q2_K, standard | 12.9 GB |
| `Qwen3.6-35B-A3B-abliterated-v4-IQ2_M.gguf` | IQ2_M, standard | 11.7 GB |
| `Qwen3.6-35B-A3B-abliterated-v4-BF16-MTP.gguf` | BF16, MTP-preserving | about 71 GB (card table) |

The card does not provide a separate sampler recommendation. The BF16 MTP size is rounded differently in the page summary; retain the publisher's uncertainty rather than treating it as a local byte count.

### SC117 Heretic/MPOA APEX

Source: [SC117/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-APEX-GGUF](https://huggingface.co/SC117/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-APEX-GGUF). The card describes an Apache-2.0 uncensored Heretic/MPOA APEX fine-tune with native MTP preserved. The current tree exposes these exact files (HF display values are rounded and differ from an older update table):

| Basename | APEX tier | Current tree size (approx.) |
|---|---|---:|
| `Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-APEX-I-Quality.gguf` | I-Quality | 23.5 GB |
| `Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-APEX-I-Balanced.gguf` | I-Balanced | 26 GB |
| `Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-APEX-I-Compact.gguf` | I-Compact | 17 GB |
| `Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-APEX-I-MINI.gguf` | I-MINI | 14.3 GB |

The card recommends `--spec-type draft-mtp --spec-draft-n-max 2` and publishes the same four sampler profiles used by the MTP Qwen card (thinking/general `1.0/0.95/20`, thinking/coding `0.6/0.95/20`, instruct/general `0.7/0.8/20`, instruct/reasoning `1.0/1.0/40`; presence penalties 1.5, 0, 1.5, and 2 respectively). These are fine-tune recommendations, not base-model measurements.

## Hardware requirements

- Run `scripts/check_hardware.py` before downloading or selecting a model. The whole GGUF, KV cache, runtime allocations, and vision/MTP sidecars must fit with OS/IDE headroom.
- Unsloth's published total-memory guide is approximately: 3-bit 17 GB, 4-bit 23 GB, 6-bit 30 GB, 8-bit 38 GB, and BF16 70 GB. These are sizing hints, not a fit result for this machine.
- The Unsloth Qwen3.6 guidance warns against CUDA 13.2 for this model and suggests BF16 KV cache types when output is degraded; confirm the current runtime/toolchain before applying either recommendation.
- Dense GGUFs must fit physical VRAM on a discrete-GPU machine or the unified RAM pool on UMA. Do not partially offload a dense model and hope shared memory remains usable.
- MoE expert offload is a separate path: `N_CPU_MOE=None` lets the harness infer a safe default from GGUF `block_count`; `N_CPU_MOE=0` is only for a MoE that actually fits in physical VRAM; an explicit positive value is a manual override.
- The listed 10-39 GB model sizes are not a fit guarantee. Context size, KV type, draft/MTP state, `mmproj`, and the selected runtime all change peak memory. The measured point below passed its host/VRAM preflight; that result does not generalize to every quant or context.

## Recommended settings

Seed `SAMPLER_DEFAULTS` from the model card before the first Trial. Use the profile matching the job; do not start from the repository template when a publisher profile exists.

| Profile | Temperature | Top-p | Top-k | Min-p | Presence penalty | Repetition penalty |
|---|---:|---:|---:|---:|---:|---:|
| Thinking / general | 1.0 | 0.95 | 20 | 0 | 1.5 | 1.0 |
| Thinking / precise coding | 0.6 | 0.95 | 20 | 0 | 0 | 1.0 |
| Instruct / non-thinking | 0.7 | 0.80 | 20 | 0 | 1.5 | 1.0 |
| Instruct / reasoning (MTP card) | 1.0 | 1.0 | 40 | 0 | 2.0 | 1.0 |

The Qwen/Unsloth cards use `chat_template_kwargs` with `{"enable_thinking": false}` to disable thinking; `{"preserve_thinking": true}` retains the prior thinking trace in continued conversations (**preserve thinking is on by default**). Harness equivalent: Baseline `REASONING_PRESERVE=True` emits `--reasoning-preserve` (leave `None` unless `GET /props` `chat_template_caps.supports_preserve_reasoning` is true). PowerShell escaping for raw kwargs: `--chat-template-kwargs "{\"enable_thinking\":false}"`. Confirm the cap on the loaded GGUF before a Claw-full seed.
## Reasoning control

**Verified local fact (2026-08-29):** the qwen35moe family (this GGUF's `general.architecture`) reads only `enable_thinking` in the tokenizer template. `--reasoning-effort` is a **silent no-op** on these GGUFs — setting it does nothing; do not use it as a reasoning lever for this family.

Working reasoning levers for Qwen3.6-35B-A3B:
- `--reasoning on` / `--reasoning off` — switches the Baseline `REASONING` path (corresponds to `enable_thinking` toggling in the template).
- `--reasoning-budget N` — the `REASONING_BUDGET` lever; this is the preferred control when a thinking budget is needed (it is template-independent and server-side).

**Daily-driver profile (no Trial claims):** the daily-driver configuration runs a 4096 reasoning budget; this is a profile observation, **not** an Objective-Vector measurement — treat as `**TBD:**` for measured equivalence.

**Template reference (verified 2026-08-29, `gguf_dump --no-tensors --json`):** `enable_thinking` is the only thinking variable queried; the default render injects the xhigh instruction when `enable_thinking` is true and otherwise emits a bare `</think>`. The Qwen/Unsloth cards use `chat_template_kwargs` with `{"enable_thinking": false}` to disable thinking and `{"preserve_thinking": true}` to retain the prior trace in continued conversations. Harness equivalent: Baseline `REASONING_PRESERVE=True` emits `--reasoning-preserve` (leave `None` unless `GET /props` `chat_template_caps.supports_preserve_reasoning` is true). Confirm the cap on the loaded GGUF before a Claw-full seed.

## MTP and draft packaging

- **Local files: MTP provenance is the Unsloth `-MTP-GGUF` repo (2026-08-29 gguf_dump):** the two local files have different MTP state. The non-MTP file `unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` reports `nextn_predict_layers=0` AND has **0 `nextn` tensors** — the MTP head is absent. The MTP file `unsloth/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` reports `nextn_predict_layers=1` AND carries **4 MTP head tensors** `blk.40.nextn.{eh_proj,enorm,hnorm,shared_head_norm}` (UD `eh_proj` is Q8_0 in this UD quant). **Use the MTP file when `--spec-type draft-mtp` is the goal; the non-MTP file's MTP head is not present even though the surrounding base is otherwise identical.** Tensor count is identical at 753; the MTP provenance is the dedicated repo.
- **Integrated MTP targets:** the Unsloth `-MTP-GGUF` repo, ggml-org `Qwen3.6-35B-A3B-MTP-*`, Bahushruth `...-BF16-MTP.gguf`, SC117 APEX files, and AesSedai's updated specialist quants are documented by their publishers as MTP-preserving. Confirm `*.nextn_predict_layers` AND presence of `blk.N.nextn.*` tensors in the actual local GGUF; file naming alone is not proof. **Unsloth MTP repo card (2026-08-29 fetch):** `general.architecture=qwen35moe`, `context_length=262144`, license Apache-2.0, 481k downloads, 883 likes, last modified 2026-05-20. The base Unsloth repo (non-MTP) has 1.2M downloads and 1567 likes (last modified 2026-04-20). Both repos expose the same quant basenames; MTP repo is the provenance for the MTP-head weights.
- **Separate MTP artifacts:** ggml-org base `mtp-*` files and Bartowski `mtp-*` files are separate basenames that need explicit pairing/metadata inspection.
- **DFlash:** ggml-org ships `dflash-Qwen3.6-35B-A3B-{Q8_0,BF16}.gguf`. Upstream llama.cpp accepts `--spec-type draft-dflash` with `--spec-draft-model`. **Dead end on 8 GB-class + `--n-cpu-moe`:** DFlash needs the 35B target fully on GPU. Q8 draft header: `dflash.block_size=16`, `target_layers=[2,7,12,17,23,28,33,38]` (eight target-layer extracts per decode). **Q4 @ 32k (2026-08-07):** DFlash n=15 **17.5** vs no-spec **27.2**. **Q3 @ 65k (2026-08-12):** DFlash n=15 **12.5** vs no-spec **24.6** vs embedded MTP n=1 **29.5**. Do not seed DFlash for max TPS on this path. Primary DFlash docs still target vLLM/SGLang ([z-lab/dflash](https://github.com/z-lab/dflash)).
- **llama.cpp MTP:** `--spec-type draft-mtp --spec-draft-n-max N` ([PR #22673](https://github.com/ggml-org/llama.cpp/pull/22673)). **Q4 @ 32k:** n_max=2 → **27.5** (≈ flat vs 27.2). **Q3 @ 65k:** n_max=1 → **29.5** (beats no-spec 24.6). Prefer embedded MTP over DFlash.
- **Local runtime probe (2026-08-02, source-verified):** the checked-out llama.cpp `common/speculative.cpp` `common_speculative_type_from_name_map` accepts: `none`, `draft-simple`, `draft-eagle3`, `draft-mtp`, `draft-dflash`, `draft-dspark`, `ngram-simple`, `ngram-map-k`, `ngram-map-k4v`, `ngram-mod`, `ngram-cache`. **Bare `mtp` is NOT accepted in this tree** — keep `--spec-type draft-mtp`. Fork nuance (GOLDEN-RULES.md): TurboQuant/custom fork builds historically accept `mtp`; the harness probes `--help` at runtime and picks per build.
- **TurboQuant/custom builds:** GGUF format compatibility does not prove MTP, DFlash, UD, IQ, or specialist-mixture support. The project runtime and release tag must be recorded and probed before relying on any speculative flag.

## MoE split (VITRIOL-compatible policy)

The repository policy is to keep dense models fully resident and to use expert CPU offload only for MoE models. For this family, seed `N_CPU_MOE=None` and let the harness derive the initial value from the inspected GGUF's block count; do not hardcode `40` until local metadata confirms 40 blocks. Tune `--n-gpu-layers` and `--n-cpu-moe N` only after the host-memory preflight passes. See [vitriol-technique.md](./vitriol-technique.md) for the stock split policy; this is not the Randozart DMA fork.

## Context ladder (2026-09-02, serving-path probes)

Operator ask: ≥200k window on the 8 GB box. Same harness preflight wall as ornith-1.5-35b (same `qwen35moe` geometry): est @ 204800 q4_0 ≈ 8.6 GB (`VRAM_MOE_NON_EXPERT_FRAC` 0.28 × 21.6 GB file over-reads the real ~1.8 GB GPU-resident weights ~3×) vs the hard 7676 keepout clamp — **every ctx > ~136k is preflight-rejected regardless of KV type**. Ladder ran via `model-up` alias probes per the `use-harness-not-raw-llama` carve-out. Device-wide nvidia-smi; q4_0 KV, draft-mtp n-max 2 unless noted, single-shot prompts:

| Config | idle+smoke | peak @ fill | pp | tg |
|---|---|---|---|---|
| codacus cache48 + MTP @ 204800 | **7710 — over at idle** | 7760 @ 4k | 258 | 32.5 @ 4k |
| codacus cache32 + MTP @ 204800 | 7599 | 7717 @ 4k — over | 621 | (early-stop) |
| plain b10549 + MTP @ 204800 | 5376 | 5443 @ 4k | 807 | **34.0** @ 512-tok gen |
| plain b10549, MTP off @ 204800 | 4587 | — | — | 30.3 @ 472-tok gen |
| **plain b10549 + MTP @ 262144 (winner)** | 6035 | **6093 @ 255k fill** | 582 @ fill | 30.5 shallow / 14.7 @ ~255k |

**Winner: plain upstream b10549, ctx 262144 (full native GGUF window), q4_0 KV, draft-mtp n-max 2 kept** — daily alias re-pointed, expert-cache flags removed. MTP A/B on plain upstream: +12% decode (34.0 vs 30.3) for +789 MB — kept, unlike ornith-1.5-35b where MTP is a net loss on the CPU-offloaded MoE; this model's trained head stays positive on plain upstream. Deep-fill acceptance: 255014-tok prefill (pp_avg 582, wall 441 s), tg@fill 14.7, peak 6093 = 1.58 GB under keepout. The fork cache+MTP stack's 42 t/s @131k shallow number does not survive the 200k window trade (cache48 is over keepout at idle; at deep fill cache buys nothing). No-Trial-claims (single-shot serving probes, not TPS_REPS medians).

## Our config baseline

- `CTX_SIZE`: repository default `131072`. Speed smokes (2026-08-07) used `32768`. Objective Vector point below used `100000`. **Serving alias now 262144 (2026-09-02 ladder)**; harness Trials remain est-capped ~131k until the MoE preflight frac is recalibrated.
- `SAMPLER_DEFAULTS`: thinking/general `TEMP=1.0`, `TOP_P=0.95`, `TOP_K=20`, `MIN_P=0`, `REPEAT_PENALTY=1.0`, `PRESENCE_PENALTY=1.5`.
- `N_CPU_MOE=None`: harness auto-resolved to `41` for the measured basename.
- Objective Vector engine knobs (2026-08-02): TurboQuant `tqp-v0.3.0`, KV K/V `turbo3/turbo3`, batch/ubatch `32/16`, threads `6/8`, `SPEC_TYPE=None`.
- Speed-smoke knobs (2026-08-07): upstream `b10286`, KV `q4_0/q4_0`, batch/ubatch `512/128`, threads `8/8`. Spec matrix: none / `draft-dflash`+Q8 draft n=15 / `draft-mtp` n=2 — see Measured Trial evidence.
- Max-TPS note: DFlash lost on both Q4@32k and Q3@65k with expert CPU offload. Embedded MTP can beat no-spec on the Q3 Fingerprint. Do not retry DFlash knobs (`n_max`, draft GPU vs CPU) expecting a sign flip.
- A point is not `on_front` until it has the same Fingerprint and complete Claw-full plus coding-10 Objective Vector. Fine-tunes and base quants are distinct Trial families.

## Measured Trial evidence

### Objective Vector (2026-08-02) — no-spec @ 100k turbo3

Trial `76f6f780-dda3-4ba7-8a42-e6a267d95b1e` for basename `Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf`:

| Field | Measured value |
|---|---|
| Engine | TurboQuant `tqp-v0.3.0` |
| Context | `100000` |
| KV cache K/V | `turbo3/turbo3` |
| Batch / ubatch | `32/16` |
| Threads / batch threads | `6/8` |
| Speculative decoding | none (`SPEC_TYPE=None`) |
| Sampler | thinking/general: `TEMP=1.0`, `TOP_P=0.95`, `TOP_K=20`, `MIN_P=0`, `REPEAT_PENALTY=1.0`, `PRESENCE_PENALTY=1.5` |
| `N_CPU_MOE` | `None` requested; auto-resolved `41` |
| Preflight | VRAM `7496 < 7900`; host `23189 < 27790` |
| Benchmark throughput | `24.5 TPS` |
| Coding generation | `29.4` |
| Peak memory | `4.9 GB` |
| Coding scores | LCB `0.4`, HumanEval `0.2`, MBPP `0.9`, BigCode `0.1`; composite `0.43` |
| Claw full | `0.5333` (`8/15`) |
| Elapsed | `7454 s` |
| Result | `OK` |

### Validation speed matrix (2026-08-07) — CTX 32k, upstream `b10286`, same Q4 basename

Incomplete vectors (smoke only). Session: [2026-08-07-qwen36-35b-dflash-tps.md](../sessions/2026-08-07-qwen36-35b-dflash-tps.md).

| Trial | Spec | Bench tg | Peak VRAM |
|---|---|---:|---:|
| `ef3094b2-d634-4308-b6f4-cc300c4a6d2b` | none | **27.2** | 4.1 GB |
| `3810c77b-f108-4874-8ffe-21c0ade7209a` | `draft-mtp` n_max=2 | **27.5** | 4.6 GB |
| `06dce572-7122-45a3-a075-901c7460dda8` | `draft-dflash` n_max=15 + Q8 draft | **17.5** | 5.6 GB |

### Validation speed matrix (2026-08-12) — CTX 65k, Q3 basename, `N_CPU_MOE=40`

Incomplete vectors (smoke only). Session: [2026-08-12-qwen36-dflash-tps.md](../sessions/2026-08-12-qwen36-dflash-tps.md).

| Trial | Spec | Bench tg | Peak VRAM |
|---|---|---:|---:|
| `512c52ee-5f0c-423e-b348-28ddd9ed59b3` | none | **24.6** | 4.7 GB |
| `6fe4189f-7329-4c44-87b0-6325ed173b9b` | `draft-mtp` n_max=1 | **29.5** | 5.7 GB |
| `54e972a6-c0ec-4687-ba60-56a941125e3e` | `draft-dflash` n_max=15 + Q8 draft | **12.5** | 6.6 GB |
| `c409f082-8061-41d9-96da-015a1edb0504` | `draft-dflash` n_max=15 | **9.5** | rejected (`TPS_FLOOR`) |

## Sources / verification

- Official architecture, license, sampler, and file tree: [Qwen model card](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) and [official files](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/tree/main), extracted 2026-08-02; re-verified 2026-08-29 (35.95B BF16 params, 4.99M downloads, 26 shards, Apache-2.0).
- Unsloth sizing and runtime guidance: [Qwen3.6 docs](https://unsloth.ai/docs/models/qwen3.6) and [MTP docs](https://unsloth.ai/docs/models/mtp), extracted 2026-08-02.
- **Unsloth base GGUF inventory** (non-MTP provenance): [base tree](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/tree/main), API + tree fetched 2026-08-29 — arch `qwen35moe`, ctx 262144, license Apache-2.0, 1.2M downloads / 1567 likes, last modified 2026-04-20; the local non-MTP file is `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` (21.9 GB) with **0 `nextn` tensors**.
- **Unsloth MTP GGUF inventory (nextn provenance)**: [MTP tree](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/tree/main), API + tree fetched 2026-08-29 — arch `qwen35moe`, ctx 262144, license Apache-2.0, 481k downloads / 883 likes, last modified 2026-05-20; the local MTP file is `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` (22.7 GB) with **4 `blk.40.nextn.*` tensors** (`eh_proj`, `enorm`, `hnorm`, `shared_head_norm`). The `-MTP-GGUF` repo is the provenance for nextn tensors; both repos expose identical quant basenames, so MTP state is verified via GGUF tensor audit, not file naming.
- Official llama.cpp artifacts and command: [base tree](https://huggingface.co/ggml-org/Qwen3.6-35B-A3B-GGUF/tree/main), [MTP tree](https://huggingface.co/ggml-org/Qwen3.6-35B-A3B-MTP-GGUF/tree/main), and [PR #22673](https://github.com/ggml-org/llama.cpp/pull/22673), extracted 2026-08-02.
- Bartowski inventory: [model card/tree](https://huggingface.co/bartowski/Qwen_Qwen3.6-35B-A3B-GGUF/tree/main), extracted 2026-08-02.
- Specialist/fine-tune inventories: [AesSedai](https://huggingface.co/AesSedai/Qwen3.6-35B-A3B-GGUF), [Hesamation](https://huggingface.co/hesamation/Qwen3.6-35B-A3B-Claude-4.6-Opus-Reasoning-Distilled-GGUF), [Bahushruth](https://huggingface.co/Bahushruth/Qwen3.6-35B-A3B-abliterated-v4-GGUF), and [SC117](https://huggingface.co/SC117/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-APEX-GGUF), extracted 2026-08-02.
- DFlash pairing and runtime scope: [z-lab/dflash](https://github.com/z-lab/dflash), extracted 2026-08-02.
- GGUF header verification: project venv `gguf_dump` with `PYTHONUTF8=1`, exact basenames and fields recorded in the Architecture table above, extracted 2026-08-02 and refreshed 2026-08-29; the card records basenames and header fields only.
- Reasoning-control verification (template Jinja, 2026-08-29): `qwen35moe` chat template reads `enable_thinking` only — `reasoning_effort` is absent; `--reasoning-effort` is a silent no-op — see § Reasoning control.
- Trial evidence (Objective Vector): run `76f6f780-dda3-4ba7-8a42-e6a267d95b1e`, basename `Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf`, measured 2026-08-02; flags, preflight, scores, and elapsed time are recorded above.
- Trial evidence (DFlash / MTP speed smokes): Q4 runs `ef3094b2-…`, `06dce572-…`, `3810c77b-…` (2026-08-07, CTX 32768) — [2026-08-07-qwen36-35b-dflash-tps.md](../sessions/2026-08-07-qwen36-35b-dflash-tps.md). Q3 runs `512c52ee-…`, `6fe4189f-…`, `54e972a6-…`, `c409f082-…` (2026-08-12, CTX 65536) — [2026-08-12-qwen36-dflash-tps.md](../sessions/2026-08-12-qwen36-dflash-tps.md).
- Results store points (2026-08-29): `Qwen3.6-35B-A3B-UD-Q3_K_XL` n=14, agentic 0.8, 34.8 tps @65536; `Qwen3.6-35B-A3B-UD-Q4_K_XL` n=8, 27.5 tps @100000; MTP-GGUF file `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` (MTP) n=1, 32.1 tps @65536 — verified from `results.db`, not overwritten by the older trial rows above.

## Open questions

- Should the SC117 uncensored candidate receive a separate Trial in addition to the measured Unsloth point?
- What does the hardware preflight allow for the SC117 candidate, MTP state, `mmproj`, and other contexts?
- Which candidate, if any, is authorized for a full Claw + coding-10 Objective Vector on a **spec** Fingerprint? Speed smokes stay `incomplete`.
- No QAT-labelled Qwen3.6-35B-A3B artifact was found in the inspected official Qwen, Unsloth, or ggml-org inventories. Treat QAT as **not identified**, not as a claim that no third party can publish one later.
- Unmeasured max-TPS ladder: Q3/IQ quant, `N_CPU_MOE` sweep, lower CTX, EXL3 study — see session brainstorm.

## Daily SWE eval (DM-Code-Agent 30-task via `mini-swe-agent`)

The operator's daily-SWE eval against this alias runs via `mini-swe-agent` (env-only wire-up to the harness `llama-server`) on the DM-Code-Agent 30-task frozen benchmark. Layered next to SWE-lite (ADR 0013) as another Night selector; **not** a Pareto axis in v1.

- **Run:** `python -m autoresearch.runners.run --mini-swe-agent --desc "msa-baseline-qwen3.6-35b-a3b"`
- **Schema:** writes `mini_swe_agent_val` (passed/total) and `mini_swe_agent_detail` (per-task `<id>=<verdict>`) into `results.db`. Backward-compatible `ALTER TABLE` migration; existing rows read `NULL` for the new columns.
- **Operator guide:** [`../discovery/mini-swe-agent-benchmark.md`](../discovery/mini-swe-agent-benchmark.md).
- **Acceptance gate:** ≥ 18/30 (~60 %) on this alias; per-task wall-time ≤ 4× `esagduyu`'s 3.3 min baseline on Qwen3.6-35B-A3B; trajectory JSONL non-empty for every task; hidden tests never in the prompt.
- **Deferral:** promotion to a fifth Pareto axis is ADR 0018 territory. v1 lives in the new column as an observation; Day/Night Usage Profile rules (ADR 0017) stay unchanged.
