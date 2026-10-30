# Speculative Decoding Formats Guide (llama.cpp)

This guide documents the speculative decoding formats supported by `llama.cpp` (`llama-cli`/`llama-server`), their architectural requirements, and their availability for the models in this repository.

---

## 1. Overview of Speculative Decoding Formats

Speculative decoding speeds up inference by using a fast **drafting method** to propose a block of tokens, which the larger **target model** then validates in a single forward pass.

In `llama.cpp`, the format is specified using the `--spec-type` flag. These formats are **not** software configurations that can be applied to any model; they must match the **mathematical and architectural format** of the draft model file (`-md` / `--spec-draft-model`).

| Format (`--spec-type`) | Description | Drafter Type | VRAM Cost | Availability for our Models |
| :--- | :--- | :--- | :---: | :--- |
| **`draft-mtp`** | Multi-Token Prediction (MTP) | Neural (Assistant) | Medium | **High** (Native for Qwen3.5/3.6 and Gemma-4. Pre-trained drafts available). |
| **`draft-eagle3`** | Eagle 3 tree-based drafting | Neural (Eagle Head) | Medium | **Available (since 2026-08, now stale "None")** — upstream `docs/speculative.md` (master, 2026-08-22) lists `RedHatAI/gemma-4-31B-it-speculator.eagle3`, `RedHatAI/gemma-4-26B-A4B-it-speculator.eagle3`, `Tengyunw/qwen3_8b_eagle3` / `qwen3_30b_moe_eagle3`, `AngelSlim/Qwen3-{1.7B,4B,8B,14B,32B,a3B}_eagle3`, plus `yuhuili/EAGLE3-LLaMA3.*`, `RedHatAI/gpt-oss-20b-speculator.eagle3`, `nvidia/gpt-oss-120b-Eagle3-long-context`. Convert with `python convert_hf_to_gguf.py <eagle3> --target-model-dir <target> --outfile draft.gguf` then `--spec-type draft-eagle3 -md draft.gguf`. Repo has a matching target: Gemma-4-26B-A4B (26B-A4B card). **On this 8 GB rig, still predicted loss on MoE+`n-cpu-moe` fingerprints** (same expert-union verify-cost rule as DFlash/DSpark dead ends, §4b) — falsifiable only with target fully on GPU, which 26B does not fit at 65k. No Eagle3 draft exists for this repo's E2B/E4B or Qwen3.5-9B dense targets as of 2026-08-22. |
| **`draft-dflash`** / **`dspark`** | DeepSeek parallel block drafters | Neural (DeepSpec) | Medium | **Dead end on 8 GB-class MoE+`n-cpu-moe`.** Qwen3.6-35B DFlash loads but is slower than no-spec; needs the 35B target fully on GPU. Embedded `draft-mtp` can still win. Evidence: [2026-08-12](../sessions/2026-08-12-qwen36-dflash-tps.md), [2026-08-07](../performance/2026-08-07-qwen36-35b-dflash-tps.md). Native `draft-dspark` is in upstream since 2026-07-28 ([#25173](https://github.com/ggml-org/llama.cpp/pull/25173), i.e. inside pinned b10375) for **Qwen3-backbone dense-format** drafts; **speculators-format** DSpark (SpecForge/RedHat) merged 2026-08-17 ([#26275](https://github.com/ggml-org/llama.cpp/pull/26275)) — **post-b10375** — and adds Gemma-4-backbone drafts (`makora-ai/gemma4-26b-a4b-dspark`, `RedHatAI/gemma-4-31B-it-speculator.dspark`). |
| **`ngram-cache`** | N-gram → next-token statistics (3-level cache: context, dynamic, static) | Statistical | None (0 MB VRAM; host RAM grows with context) | **Universal** (any model, no extra files). |
| **`ngram-simple`** | History match: last n-gram → following m-gram | Statistical | None (0 MB) | **Universal** (any model, no extra files). |
| **`ngram-map-k` / `ngram-map-k4v`** | Hash-map of n-grams in the context window; k4v tracks up to 4 value m-grams | Statistical | None (0 MB) | **Universal** (any model, no extra files). |
| **`ngram-mod`** | LCG rolling-hash pool: n-gram hash → next token, shared across server slots | Statistical | None (0 MB VRAM; ~16 MB host RAM, constant) | **Universal** (any model, no extra files). `--spec-default` enables it. |
| **`none`** | Standard autoregressive decoding | None | None | **Universal** (Speculative decoding disabled). |

---

## 2. Can MTP and other Speculative Decoding methods be used together?

**Partial — updated 2026-08-18.** `--spec-type` is a **comma-separated list**; upstream combines a draft-model implementation with a draftless one. The earlier "one spec-type only" claim in this guide was stale.

1.  **MTP is already a form of Speculative Decoding:**
    *   Speculative decoding is the general concept of using a fast drafting method to propose tokens and a target model to verify them.
    *   Multi-Token Prediction (MTP) is simply a *specific implementation* of this concept, where the "draft model" is built directly into the target model as auxiliary prediction heads (e.g. Qwen's MTP heads or Gemma-4 assistant models). When you enable MTP, you *are* using speculative decoding.
2.  **Mixing rules (upstream docs/speculative.md, master; MTP merge [#22673](https://github.com/ggml-org/llama.cpp/pull/22673) 2026-05-16):**
    *   `--spec-type` accepts a comma-separated list, e.g. `--spec-type draft-mtp,ngram-mod` (also valid as repeated `--spec-type draft-mtp --spec-type ngram-mod` flags — `--spec-type` is exempt from the duplicate-argument warning).
    *   **An implementation with a draft model can be mixed with an implementation without one**: neural (MTP/DFlash/DSpark/Eagle) + statistical (ngram-*) is supported. The MTP PR documents exactly this combination (`--spec-type draft-mtp --spec-draft-n-max 3 --spec-type ngram-mod --spec-ngram-mod-n-match 24 --spec-ngram-mod-n-min 48 --spec-ngram-mod-n-max 64`, or `--spec-default --spec-type draft-mtp`), labeled *experimental, suitable for non-CUDA systems*.
    *   **If a draft model is combined with a draftless decoding, the draftless decoding has higher precedence** (upstream docs).
    *   **Two neural drafters simultaneously is still not supported** (one draft model at a time — a single `-md`). Chaining (Eagle drafts, then MTP drafts those drafts) remains unsupported.
3.  **Choosing the Best Method:**
    *   Measure per Fingerprint. For Gemma-4 / Qwen3.5 dense on this 8 GB rig, **MTP is usually best when it actually accelerates**; Mythos MTP GGUF is a counterexample (~+1%). See [small-model-mtp-tps.md](./small-model-mtp-tps.md).
    *   A cheap composite worth probing on dense MTP models: `draft-mtp` + `ngram-mod` (ngram fills where the MTP head hesitates; 0 MB VRAM). On small-active MoE (A3B), external measurements say all spec modes lose, and this rig's 2026-08-22 MTP sweep on `--n-cpu-moe 41` agrees (see §4b).

---

## 3. Deep-Dive into Formats

### A. Multi-Token Prediction (MTP) — `draft-mtp`
*   **How it works:** Native to Qwen2.5/3.5/3.6 and Gemma-4. Two packaging forms:
    1. **Embedded `nextn` heads** inside the main GGUF (Qwen UD MTP builds; community `*-MTP*.gguf`).
    2. **External assistant draft** (Gemma-4): Google trained `*-assistant` models; Unsloth ships tiny draft GGUFs. The **main** Gemma UD file does **not** contain `nextn` tensors.
*   **Why it's best for us:** When acceptance is healthy, MTP hits **~1.5x–1.8x** on this 8 GB rig (see §4 matrix). Not automatic — Mythos MTP GGUF measured ~+1%.
*   **In llama.cpp:**
    - Qwen3.5-9B UD: MTP **embedded** — `--spec-type draft-mtp --spec-draft-n-max N` only (no `-md`).
    - Gemma-4 E4B: pass `--spec-draft-model models/draft/mtp-gemma-4-E4B-it.gguf` (path relative to `models/` in harness: `draft/mtp-gemma-4-E4B-it.gguf`).
    - **Ornith-1.0-9B UD:** no MTP. Use Hub `protoLabsAI/Ornith-1.0-9B-MTP-GGUF` (local: `Ornith-1.0-9B-MTP-Q4_K_M.gguf`).
    - **Mythos 5-1M:** Hub `mradermacher/Qwythos-9B-Claude-Mythos-5-1M-MTP-GGUF` (local MTP Q4_K_M). Loads; short-gen gain negligible in 2026-07-20 matrix.
    - **Qwythos-9B-v2:** no useful CUDA GGUF MTP on Hub (MLX-only) as of 2026-07-20.
*   **Detect:** metadata keys `nextn` / `blk.*.nextn.*` (embedded) or `gemma4-assistant` (draft). See [small-model-mtp-tps.md](./small-model-mtp-tps.md). **Caveat:** low-bit quants may drop the head — verify per-file. Example: `Qwen3.8-27B-Q1Q-XYZ-v2.gguf` declares `nextn_predict_layers` value 0 and zero `nextn.*` tensors despite pack docs claiming embedded MTP (2026-08-18 validation).
*   **Acceptance vs draft length (measured upstream + community):** acceptance is prompt- and head-quality-dependent, and it is the lever the `--spec-draft-n-max` sweep tunes.
    - Upstream MTP merge [#22673](https://github.com/ggml-org/llama.cpp/pull/22673) (Qwen3.6-27B q8_0, DGX Spark): `n_max=3` aggregate accept 0.72 @ ~2.6× vs baseline; `n_max=2` accept 0.83 but fewer tokens/pass (17.4 vs 21.6 t/s on code). Per-task accept ranged 0.54 (translation) to 0.91 (code) — the same target, same head, big spread.
    - Community rule of thumb (dev.to, 2026-05-18): acceptance below ~0.6 means MTP **costs** wall-clock; temperature >~0.7 and aggressively quantized MTP heads both depress acceptance ("the MTP head suffered more than the main weights").
    - **Draft probability threshold (`--spec-draft-p-min P`):** upstream supports early-abort of speculative draft passes when cumulative token probability drops below $P$ (default 0.00). Setting $P \in [0.1, 0.4]$ prevents drafting low-confidence branches on noisy heads.
    - **Unified KV buffer (`-kvu` / `--kv-unified`):** shares a single unified KV buffer across all sequences and slots (default enabled on auto slot counts), reducing multi-slot fragmentation.
    - Implication for this repo: Mythos 5-1M MTP (+1%) and the Qwen3.6-35B-A3B n=2 (+1%) cases are consistent with quant-damaged/low-yield heads or draft-length mismatch, not engine failure — the falsifiable probe is a `SPEC_DRAFT_N_MAX` / `SPEC_DRAFT_P_MIN` sweep on the same Fingerprint (see [small-model-mtp-tps.md](./small-model-mtp-tps.md) open questions).

### B. Eagle 3 — `draft-eagle3`
*   **How it works:** A tree-based speculative decoder that trains a small recurrent neural network head directly on top of the target model's hidden states.
*   **Availability (re-checked 2026-08-22, upstream `docs/speculative.md` master):** upstream lists `RedHatAI/gemma-4-31B-it-speculator.eagle3`, `RedHatAI/gemma-4-26B-A4B-it-speculator.eagle3` (matches this repo's Gemma-4-26B-A4B target, but predicted loss on 8 GB `n-cpu-moe` — see table), `Tengyunw/qwen3_8b_eagle3` / `qwen3_30b_moe_eagle3`, `AngelSlim/Qwen3-{1.7B,4B,8B,14B,32B,a3B}_eagle3`, `yuhuili/EAGLE3-LLaMA3.*`, `RedHatAI/gpt-oss-20b-speculator.eagle3`, `nvidia/gpt-oss-120b-Eagle3-long-context` plus `lmsys/EAGLE3-gpt-oss-120b-bf16`. None match this repo's Qwen3.5-9B or Gemma-4 E2B/E4B targets; the earlier "None (No pre-trained Eagle drafts exist for Gemma-4/Qwen)" claim is now stale for Gemma-4 31B/26B-A4B and Qwen3 (not Qwen3.5). Convert with `python convert_hf_to_gguf.py <eagle3> --target-model-dir <target> --outfile draft.gguf` then `--spec-type draft-eagle3 -md draft.gguf`. Community examples that remain valid: `Ex0bit/Qwen3.6-27B-PRISM-PRO-DQ`, `thoughtworks/Gemma-4-31B-Eagle3`.
*   **Error Case:** If you try to pass an MTP draft model to `--spec-type draft-eagle3`, it will fail to load with:
    `failed to initialize speculative decoding context: draft model is not eagle3`

### C. DFlash & DSpark — `draft-dflash` (DFlash) / `dspark` (DSpark)
*   **How it works:** Open-source research released by DeepSeek (the **DeepSpec** framework).
    - **DFlash:** A block-parallel drafter utilizing a diffusion-like block process to predict blocks of tokens in a single step (e.g. `spiritbuun/Qwen3.6-27B-DFlash-GGUF` or `williamliao/gemma-4-31B-it-DFlash-GGUF`).
    - **DSpark:** A semi-autoregressive model combining a parallel backbone with a lightweight serial head to reduce "suffix decay" (loss of coherence at the end of draft sequences).
*   **Availability:** The `Bonsai-27B` model (a quantized Qwen3.6-27B fork) can use DSpark speculative decoding (`Bonsai-27B-dspark-Q4_1.gguf`) with the external PrismML fork, which is not vendored in this repository. There are also official DeepSeek DSpark releases for Gemma-4 (e.g. `deepseek-ai/dspark_gemma4_12b_block7` and community GGUF conversions like `ankk98/dspark-gemma4-12b-block7-Q4_0-GGUF`). Other target models require custom training via DeepSpec or finding matching community GGUF drafts.
- **Official DSpark integration (2026):** DeepSeek ships DSpark inside SGLang with Gemma-4 drafts (`deepseek-ai/dspark_gemma4_12b_block7`; community GGUF `ankk98/dspark-gemma4-12b-block7-Q4_0-GGUF`). Vendor claims 60-85% faster per-user generation / up to 6.6x throughput on datacenter-class hardware; measured 8 GB-class verdicts stay negative (see 4b). SGLang is one Gemma-4 path; since 2026-08-17 llama.cpp master also loads Gemma-4-backbone drafts via the speculators format (next bullet) — no longer SGLang-only, but that llama.cpp path is post-b10375.
- **Speculators-format path (post-b10375):** `speculators`-format (SpecForge/RedHat, vLLM `speculators` project) DSpark checkpoints gained support 2026-08-17 ([#26275](https://github.com/ggml-org/llama.cpp/pull/26275)) — **after** the b10375 build, so not in the pinned engine. PR verification loads Gemma-4-backbone drafts in llama.cpp: `makora-ai/gemma4-26b-a4b-dspark`, `RedHatAI/gemma-4-31B-it-speculator.dspark`, plus `RadixArk/Qwen3.8-27B-DSpark` (2.09× coding / 1.93× overall on its 27B target; **external / unverified on this rig**). The format supports `sample_from_anchor` block layout (true = anchor-first dense-style, false = DFlash 1+N infill) and pruned draft vocabularies with a draft-to-target (`d2t`) remap. Post-b10375 spec-type auto-detect from draft metadata ([#26814](https://github.com/ggml-org/llama.cpp/pull/26814)) covers both dense and speculators formats.
- **DSpark-for-LFM2 (new, post-b10375):** upstream merged `model : support DSpark for LFM2 models` ([#27383](https://github.com/ggml-org/llama.cpp/pull/27383)) — inside b10549 but not b10375. This is the engine path for a future `LFM2.5-8B-A1B` DSpark draft. No such draft is published as of 2026-08-22; even if one appears, the durable rule (§4b, MoESD expert-union cost + `n_cpu_moe` PCIe-expert fetch) predicts net-negative on this small-active MoE class despite the target fitting on GPU — **external A3B-class prior**: all spec modes lost on RTX 3090 / A100 even at 100% draft acceptance. Falsifiable only by a dense on-GPU target; LFM2 is therefore a high-skepticism candidate.

### D. N-gram Decoders — `ngram-cache` / `ngram-simple` / `ngram-map-*` / `ngram-mod`

Five draftless (statistical) variants ship in upstream llama.cpp. All search the **token history** (not the KV cache proper — the earlier "searches the KV cache" phrasing here was loose) for repeating n-grams and draft the continuation; the target verifies the draft in the usual batched pass. **0 MB VRAM**, no extra model file, universal across architectures. Mechanism per upstream `docs/speculative.md` + `common/` source (vendored `llama.cpp/` tree, matching master):

- **`ngram-cache`** — maintains statistics of short n-grams (`LLAMA_NGRAM_MIN=1` … `MAX=4`): `n-gram → following-token → count`, built from the prompt and updated per accepted token. Drafts from the empirical distribution, with early-abort thresholds (min sample size / percentage, lax and strict tables). External statistics can be loaded/saved from files (`--spec-lookup-cache-static` / `--spec-lookup-cache-dynamic`, built/merged by the `lookup` example tools). Host-RAM cost grows with context; the dynamic cache persists across sessions to disk. Refs: #5479, #6828, #6848.
- **`ngram-simple`** — looks for the last n-gram in history matching the current context and drafts the **m tokens that followed it**. Defaults `--spec-ngram-simple-size-n 12` (lookup; docs say 12 suits code, 8 suits text), `--spec-ngram-simple-size-m 48` (draft length), `--spec-ngram-simple-min-hits 1`. Ref: #18471.
- **`ngram-map-k`** — same idea with an internal hash map of the n-grams in the current context window (`--spec-ngram-map-k-size-n/m`, `--spec-ngram-map-k-min-hits`); accepted-token counts tracked per n-gram. Ref: #18471.
- **`ngram-map-k4v`** — experimental; per key n-gram, tracks **up to 4 value m-grams** with occurrence stats and drafts the most frequent. Docs example for "a lot of longer repetitions": `--spec-ngram-map-k4v-size-n 8 --spec-ngram-map-k4v-size-m 8 --spec-ngram-map-k4v-min-hits 2`. Ref: #18471.
- **`ngram-mod`** — LCG rolling-hash pool: n-gram hash → next token, stored in a fixed-size table **shared across all server slots** (requests learn from each other). Constant ~16 MB host RAM, constant cost per step, variable draft length. Defaults `--spec-ngram-mod-n-match 24`, `--spec-ngram-mod-n-min 48`, `--spec-ngram-mod-n-max 64`; docs: *small n not recommended, MoEs require long drafts, dense models can reduce n-min/n-max*. `--spec-default` enables it. Ref: #19164.

**When it helps:** repetition in the generated text — code completion/rewriting, iterating over a block (llama.vim), reasoning models re-stating thinking, summarization. **When it doesn't:** creative text, or any target where the verify pass is the bottleneck (see §4b: small-active MoE pays the expert-union verify cost and loses even at 100% acceptance).

**Community-measured prior (external benchmarks):** mixed results; not always a net speedup, especially on quantized models; reported positive when stacked with other methods or on high-active MoE/dense. See §4b for the measured A3B case and §2 for the `draft-mtp,ngram-mod` composite.

- The 2026-07-20 small-model TPS matrix has **no ngram row** — candidate Search neighbor on coding-10 (0 MB, universal); prior is negative on A3B-class per [external measurement](../performance/capability-extraction-harness.md) §3/§8.

### Empirical finding: `ngram-cache` throughput regression on fast dense models (Trial 0593e117, 2026-09-07)

Measured on coding-10 + Claw-Eval full (15 tasks) vs Baseline `--spec-type none` (Trial 6069530a). Full run log and task breakdown: [session log](https://github.com/allanschramm/local-model-autotuning/blob/ea43a71f48571811ec1ba25df4ab71ed272d34f1/docs/sessions/2026-09-07-issue-57-ngram-cache-trial.md).

- **TPS:** -23.5% slower (72.1 vs 94.2 t/s; bench_tg 60.3 vs 74.9 t/s).
- **Coding score:** -7.8% (0.5900 vs 0.6400).
- **Agentic score:** 0.8667 vs 0.8667 (tied).
- **Draft acceptance:** Low in practice (~1.7% to 9.5% across coding and agentic tasks, max 33.9% on repetitive boilerplate).
- **Host memory footprint:** `ngram-cache` maintains an unbounded dynamic n-gram table (`std::map<common_ngram, common_ngram_cache_part>`) in host RAM that grows continuously across requests without LRU eviction, risking host memory exhaustion and RAM watchdog termination on extended runs.

**Mechanism & Pareto Verdict:** When baseline GPU generation is already fast (>70–90 t/s), llama.cpp verification passes must evaluate drafted tokens and rewind on rejection. Because statistical n-gram acceptance is low on reasoning and coding tasks, verification overhead exceeds latency savings from accepted drafts. Historical candidate `0593e117` had lower TPS than its baseline at the same Fingerprint. This is evidence against that n-gram configuration, not a current tuning result.

### Empirical finding: `ngram-simple` stacked with MTP produces warmup instability (2026-08-20)

**Evaluated 2026-08-20 (Qwen3.8-27B via Hermes, IQ3_S @ 131k, ts 70,30,
MTP `draft_n_max=2` delivering ~20 t/s).** Stacking
`--spec-type draft-mtp,ngram-simple` produced:

- Server **died during warmup** — no health response, process terminated during initial state setup.
- The same target without `ngram-simple` warmed up cleanly.

**Mechanism & Verdict:** Do not stack `ngram-simple` on top of an active MTP path. While the `--spec-type` comma-list syntax is supported upstream, combining it with recurrent/hybrid layers triggers warmup stalls without performance gain. `ngram-mod` (the LCG rolling-hash variant enabled by `--spec-default`) is a separate implementation with distinct characteristics.


---

## 4. Local Performance Comparison

### 4a. Fair small-model matrix (2026-07-20) — canonical

Upstream CUDA `llama-cli`, `-n 512`, shared knobs (`q4_0` KV, batch 256/128, threads 6/8, `draft-mtp` n_max=4). Full write-up: [session](../performance/2026-07-20-small-model-tps-matrix.md) · [operator guide](./small-model-mtp-tps.md).

| Model | Base t/s | MTP t/s | Gain | MTP form |
|---|---:|---:|---:|---|
| Gemma-4 E4B | 67.6 | **122.0** | **+80%** | external draft (~60 MB) |
| Qwen3.5-9B | 38.7 | 57.3 | +48% | embedded `nextn` |
| Ornith-1.0-9B | 38.7 | 56.3 | +46% | Hub MTP GGUF |
| Mythos 5-1M | 40.8 | 41.2 | +1% | Hub MTP GGUF (not worth it) |
| Qwythos-9B-v2 | 40.1 | — | — | no CUDA MTP |

**Default speed Baseline:** Gemma-4 E4B + draft MTP.

### 4b. Earlier spot checks (still useful)

Short `-n 128` / mixed prompts (pre-matrix):

### Gemma-4 E4B:
*   **Baseline (`none`, -n 128):** **69.9 t/s**
*   **MTP (`draft-mtp` n_max=4, -n 128):** **136.6 t/s** (+95.4%, draft ~60 MB VRAM)
*   **MTP sustained (-n 512):** **113.4 t/s** (earlier); matrix **122.0 t/s** under fixed knobs
*   *Note: MTP draft + `--spec-type draft-eagle3` → init error / silent fallback to non-spec.*

### Qwen3.5-9B:
*   **Baseline (`none`):** **39.1 t/s** (earlier spot); matrix **38.7 t/s**
*   **DFlash (`draft-dflash`):** **51.3 t/s** (+31.2% speedup, cost: ~765 MB VRAM)
*   **MTP (`draft-mtp`):** earlier spot **69.1 t/s**; matrix sustained `-n 512` **57.3 t/s** (+48% vs matrix base)

### Qwen3.6-35B-A3B (MoE, 8 GB-class + `--n-cpu-moe`):
*   **Q3 @ 65k (2026-08-12, harness `results.tsv`):** no-spec **24.6** t/s; embedded `draft-mtp` n=1 **29.5** t/s; `draft-dflash` n=15 **12.5** t/s (9.5 rejected at TPS floor).
*   **Q4 @ 32k (2026-08-07):** no-spec **27.2**; MTP n=2 **27.5**; DFlash n=15 **17.5**.
*   **Rule:** DFlash needs the 35B target fully on GPU. Expert CPU offload makes DFlash slower. Do not disable all spec — embedded MTP can still beat no-spec. Eagle-3 historically also lost on this split.
*   **External A3B-class prior (RTX 3090, 2026-04, [HF discussion #14](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/discussions/14), N=3-reproducible):** every spec-decode mode was net-negative on Qwen3.6-35B-A3B — baseline 135.7 t/s; `ngram-mod` n=8..24 129–131 t/s; `ngram-cache` 119.1 t/s (min 65.3, bimodal); 0.8B draft −39 to −60% **despite 100% draft acceptance**. Mechanism: small-active MoE (3B active, 8-of-256, sparsity 0.031) pays the expert-union verify cost — [MoESD arXiv:2505.19645](https://arxiv.org/html/2505.19645), [Utility-Driven SD arXiv:2506.20675](https://arxiv.org/pdf/2506.20675) — confirmed hardware-class-independent (A100 NVLink same magnitude) and acceptance-independent. Counter-example: Qwen3.5-122B-A10B (10B active) gets +15–45% from the same ngram machinery ([#20075](https://github.com/ggml-org/llama.cpp/pull/20075)). This repo's own MTP/DFlash numbers on A3B sit on the same side of the line; the open ngram-coding-10 Neighbor carries a **negative prior on MoE**, positive prior on dense.

*   **Measured MTP sweep (2026-08-22, harness Trials, b10549 / harness v0.2.0, `--n-cpu-moe 41`):** contrast with dense (+46–80%): 65k ctx no-spec **27.8** → `draft-mtp` n=1 **27.6** (−0.7%, Trial 1c1bc293) → n=2 **24.6** (−11%, e568c5e0); 131k ctx n=4 **18.1** (−34%, 67cb12d9). Acceptance collapsed **0.54 → 0.11** (n=1 @ 65k → n=4 @ 131k). The 131k n=4 run lands **below the harness `TPS_FLOOR` (20)** — a rejected config, not merely slower. [INFERENCE] Not directly comparable to the 2026-08-12/08-07 spots above: those ran under the pre-fix VRAM estimator (next bullet).
*   **VRAM estimator fix (b10549 / harness v0.2.0):** the workspace estimator reported **~9104 MB** for MoE MTP; direct `llama-server` measurement is **4243 MB**. The estimator was fixed so MoE MTP workspace is accounted as **0**. Trust direct server measurement over the estimator when sizing MoE MTP.
*   **Verdict — why MTP doubles dense but hurts MoE with `--n-cpu-moe` (durable rule, 2026-08-22):** dense targets keep the whole model on GPU, so the verify pass is GPU-compute-bound and near-free and accepted draft tokens cost a fraction of a step → **+46–80%** on this rig (Ornith 38.7→56.3, Qwen3.5-9B 38.7→57.3, Gemma-4 E4B 67.6→122.0), matching external **~1.9×** on dense ([PR #22673](https://github.com/ggml-org/llama.cpp/pull/22673), [Frontier Lab 260-run study](https://thefrontierlab.ai/mtp-defaults-are-a-trap/)). MoE with `n_cpu_moe` instead offloads routed-expert FFNs to CPU RAM (attention/KV/routing/shared experts stay on GPU — [Aliteq](https://aliteq.com/n-cpu-moe-llama-cpp-what-it-actually-does)): **every draft and every verify token copies the offloaded expert weights over PCIe (~144 MB/token, 36-layer top-4 — [issue #20757](https://github.com/ggml-org/llama.cpp/issues/20757))** on top of MTP's separate KV cache (~2.5 GiB) and D2H embedding transfers ([PR #22673](https://github.com/ggml-org/llama.cpp/pull/22673)). Batched verify is free only when compute-bound; over PCIe it is bandwidth-bound, so more draft tokens = more paid fetches, and rejected drafts pay for tokens never used. Acceptance collapses with context depth (0.54 @ 65k n=1 → 0.11 @ 131k n=4), so the waste grows exactly where deep-context use lives. **Net: MTP is a dense win to tune (`n_max` 3–4) and a MoE+CPU-offload loss to avoid (n ≤ 1 or `none` at deep context)** — the same shape as the earlier DFlash/Eagle-3 losses on this split.
*   **Evidence (external, checked 2026-08-22):** dense ~1.9× and ~76% acceptance @ n=3 ([PR #22673](https://github.com/ggml-org/llama.cpp/pull/22673), merged 2026-05-16); dense +61% @ n=3 over 260 runs ([Frontier Lab](https://thefrontierlab.ai/mtp-defaults-are-a-trap/)); MoE 35B-A3B ~+10% @ n=2, ~flat @ n=3, negative beyond; the old n=16 default collapsed ~54→~13 t/s and net loss occurred at every config on Metal ([issue #23752](https://github.com/ggml-org/llama.cpp/issues/23752)); `--spec-draft-n-max` default cut 16→3 ([PR #23269](https://github.com/ggml-org/llama.cpp/pull/23269)); expert caching proposed as the fix ([discussion #24528](https://github.com/ggml-org/llama.cpp/discussions/24528)); draft length can also change committed tokens at n-max 3 ([issue #23302](https://github.com/ggml-org/llama.cpp/issues/23302), [#23335](https://github.com/ggml-org/llama.cpp/issues/23335)); MTP heads are graph-wired only for Qwen3 `qwen35`/`qwen35moe` on stock builds — GLM-5.2 loads the tensors but has no wiring (no speedup) ([llama-ext.h](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-ext.h), [qwen3next.cpp](https://github.com/ggml-org/llama.cpp/blob/master/src/models/qwen3next.cpp)).
*   **Falsifiable experiment:** same Fingerprint (quant, `--n-cpu-moe 41`), sweep `SPEC_DRAFT_N_MAX` ∈ {0, 1, 2, 4} at 65k and 131k, record acceptance + t/s. Predictions: (1) t/s declines monotonically with `n` at fixed context on MoE+CPU-offload; (2) acceptance declines with context depth; (3) moving the same MoE target fully on GPU (drop `n_cpu_moe`) flips MTP toward dense behavior (+10–80%), isolating the PCIe expert fetch as the cause.
*   **Uncertainty:** no published benchmark covers MTP + `--n-cpu-moe 41`; the PCIe-fetch attribution for this rig's steeper drop (−0.7/−11/−34% vs external +10% @ n=2) is **inference from confirmed mechanisms, not a cited measurement**; acceptance is prompt- and head-quality-dependent; external figures come from different rigs/builds.

### Bonsai-27B (Sparse MoE):
*   **Baseline (`none`):** **37.3 t/s** (Target model `Bonsai-27B-Q1_0.gguf` fully on GPU, ~3.80 GB VRAM)
*   **DSpark (`draft-dspark`):** **19.2 t/s** (**-48.5% performance slowdown!** with draft `Bonsai-27B-dspark-Q4_1.gguf`, ~1.79 GB VRAM)
*   *Note: Under extreme low-bit target quantizations (like 1-bit Q1_0), the main model's forward passes run incredibly fast on GPU (37.3 t/s for 27B parameters). Because the draft model is in a heavier precision (4-bit Q4_1, 1.79 GB), its forward passes are slower per token, making speculation slower than simply running the target model directly (Quantization-Speed Inversion).*

---

## 5. Key Takeaways & Trade-offs (VRAM vs Context Size)

For consumer GPU rigs with constrained VRAM (like our discrete 8 GB-class NVIDIA):

1.  **VRAM and Context Trade-Off:**
    *   Every megabyte saved on the model/draft weights is a megabyte gained for the active KV cache context window.
    *   For a 9B model using a `q4_0` KV cache quantization, saving **~385 MB** of VRAM (MTP vs DFlash) translates directly to an extra **~12,000 tokens of context depth**.
2.  **MTP is usually optimal (but measure):**
    *   On the 2026-07-20 matrix: Gemma draft **+80%**, Qwen/Ornith embedded/Hub MTP **~+46–48%**, Mythos Hub MTP **~+1%** (skip).
    *   Lowest VRAM overhead for Gemma is the tiny assistant draft (~60 MB), not a second full model.
    *   Qwen embeds heads in the main GGUF — no secondary file to track.
3.  **DFlash and Eagle-3 Standby Value:**
    *   While MTP is superior, DFlash and Eagle-3 remain valuable fallback architectures when testing custom fine-tuned models that do not support or were not trained with MTP layers.

4.  **MoE + CPU expert offload: MTP is a measured loss (2026-08-22):**
    *   Dense MTP wins (+46–80% here, ~1.9× external); MoE with `--n-cpu-moe` loses — 65k n=1 −0.7%, n=2 −11%, 131k n=4 −34% (below `TPS_FLOOR` 20), acceptance 0.54→0.11. Every draft/verify token pays a PCIe expert fetch (~144 MB/token) plus a ~2.5 GiB separate KV cache. Keep n ≤ 1 or `none` on MoE+CPU-offload. Details, evidence, falsifiable probe: §4b.
