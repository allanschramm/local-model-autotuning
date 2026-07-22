# Speculative Decoding Formats Guide (llama.cpp)

This guide documents the speculative decoding formats supported by `llama.cpp` (`llama-cli`/`llama-server`), their architectural requirements, and their availability for the models in this repository.

---

## 1. Overview of Speculative Decoding Formats

Speculative decoding speeds up inference by using a fast **drafting method** to propose a block of tokens, which the larger **target model** then validates in a single forward pass. 

In `llama.cpp`, the format is specified using the `--spec-type` flag. These formats are **not** software configurations that can be applied to any model; they must match the **mathematical and architectural format** of the draft model file (`-md` / `--spec-draft-model`).

| Format (`--spec-type`) | Description | Drafter Type | VRAM Cost | Availability for our Models |
| :--- | :--- | :--- | :---: | :--- |
| **`draft-mtp`** | Multi-Token Prediction (MTP) | Neural (Assistant) | Medium | **High** (Native for Qwen3.5/3.6 and Gemma-4. Pre-trained drafts available). |
| **`draft-eagle3`** | Eagle 3 tree-based drafting | Neural (Eagle Head) | Medium | **None** (No pre-trained Eagle drafts exist for Gemma-4/Qwen). |
| **`draft-dflash`** / **`dspark`** | DeepSeek parallel block drafters | Neural (DeepSpec) | Medium | **Limited** (Used only for `Bonsai-27B` in the PrismML fork). |
| **`ngram-cache`** | KV Cache N-gram statistics | Statistical | None (0 MB) | **Universal** (Runs on any model without extra files). |
| **`ngram-simple`** | Basic sliding window N-grams | Statistical | None (0 MB) | **Universal** (Runs on any model without extra files). |
| **`none`** | Standard autoregressive decoding | None | None | **Universal** (Speculative decoding disabled). |

---

## 2. Can MTP and other Speculative Decoding methods be used together?

The short answer is **no, they are mutually exclusive at runtime**.

1.  **MTP is already a form of Speculative Decoding:**
    *   Speculative decoding is the general concept of using a fast drafting method to propose tokens and a target model to verify them.
    *   Multi-Token Prediction (MTP) is simply a *specific implementation* of this concept, where the "draft model" is built directly into the target model as auxiliary prediction heads (e.g. Qwen's MTP heads or Gemma-4 assistant models). When you enable MTP, you *are* using speculative decoding.
2.  **No Chaining or Multi-Speculation:**
    *   During inference, you can only set **one** `--spec-type` flag. You cannot run `--spec-type draft-mtp` and `--spec-type draft-eagle3` at the same time.
    *   The engine must follow a single path: either it uses the MTP heads to propose draft tokens, or it uses the external Eagle/DFlash model. Chaining them (e.g. using Eagle to draft tokens, then MTP to draft those drafts, and then verifying both) is not supported by any current inference engine and would introduce catastrophic synchronization overhead.
3.  **Choosing the Best Method:**
    *   Since they cannot be combined, you must choose the single best speculative type. For Gemma-4 / Qwen3.5 on this 8 GB rig, **MTP is usually best when it actually accelerates** (measure). Mythos MTP GGUF is a counterexample (~+1%). See [small-model-mtp-tps.md](./small-model-mtp-tps.md).

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
*   **Detect:** metadata keys `nextn` / `blk.*.nextn.*` (embedded) or `gemma4-assistant` (draft). See [small-model-mtp-tps.md](./small-model-mtp-tps.md).

### B. Eagle 3 — `draft-eagle3`
*   **How it works:** A tree-based speculative decoder that trains a small recurrent neural network head directly on top of the target model's hidden states.
*   **Availability:** While Unsloth does not bundle Eagle drafts for our current targets, the community hosts Eagle-3 GGUF draft models for target models on Hugging Face (e.g. `Ex0bit/Qwen3.6-27B-PRISM-PRO-DQ`, `thoughtworks/Gemma-4-31B-Eagle3`, or `RedHatAI/gemma-4-31B-it-speculator.eagle3`). They must be paired using `--spec-type draft-eagle3` and `-md <path>`.
*   **Error Case:** If you try to pass an MTP draft model to `--spec-type draft-eagle3`, it will fail to load with:
    `failed to initialize speculative decoding context: draft model is not eagle3`

### C. DFlash & DSpark — `draft-dflash` (DFlash) / `dspark` (DSpark)
*   **How it works:** Open-source research released by DeepSeek (the **DeepSpec** framework).
    - **DFlash:** A block-parallel drafter utilizing a diffusion-like block process to predict blocks of tokens in a single step (e.g. `spiritbuun/Qwen3.6-27B-DFlash-GGUF` or `williamliao/gemma-4-31B-it-DFlash-GGUF`).
    - **DSpark:** A semi-autoregressive model combining a parallel backbone with a lightweight serial head to reduce "suffix decay" (loss of coherence at the end of draft sequences).
*   **Availability:** The `Bonsai-27B` model (a quantized Qwen3.6-27B fork) can use DSpark speculative decoding (`Bonsai-27B-dspark-Q4_1.gguf`) with the external PrismML fork, which is not vendored in this repository. There are also official DeepSeek DSpark releases for Gemma-4 (e.g. `deepseek-ai/dspark_gemma4_12b_block7` and community GGUF conversions like `ankk98/dspark-gemma4-12b-block7-Q4_0-GGUF`). Other target models require custom training via DeepSpec or finding matching community GGUF drafts.

### D. N-gram Decoders — `ngram-cache` / `ngram-simple`
*   **How it works:** Instead of loading a neural network, these search the model's own active KV cache to find repeating patterns of words (N-grams) and predict subsequent tokens based on past context.
*   **Why to use it:** 0 MB VRAM footprint. Good for code completion or summarizing highly repetitive text.
*   **Performance:** ~5-10% speedup on repetitive text; 0% speedup on creative/non-repetitive text.

---

## 4. Local Performance Comparison

### 4a. Fair small-model matrix (2026-07-20) — canonical

Upstream CUDA `llama-cli`, `-n 512`, shared knobs (`q4_0` KV, batch 256/128, threads 6/8, `draft-mtp` n_max=4). Full write-up: [session](../sessions/2026-07-20-small-model-tps-matrix.md) · [operator guide](./small-model-mtp-tps.md).

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

### Qwen3.6-35B-A3B (MoE):
*   **Baseline (`none` with `--n-cpu-moe 40`):** **19.0 - 22.1 t/s**
*   **Eagle-3 (`draft-eagle3` with `--n-cpu-moe 40`):** **8.3 t/s** (**-60% performance slowdown!**)
*   *Note: For sparse MoE models where active experts are offloaded to CPU (to fit in 8 GB VRAM), speculative decoding causes severe bottlenecks. The draft model runs on GPU but requires sequential CPU synchronization for routing/experts on every draft token proposal, collapsing throughput. When running MoE models with CPU expert offloading, speculative decoding should be disabled (`none`).*

### Bonsai-27B (Sparse MoE):
*   **Baseline (`none`):** **37.3 t/s** (Target model `Bonsai-27B-Q1_0.gguf` fully on GPU, ~3.80 GB VRAM)
*   **DSpark (`draft-dspark`):** **19.2 t/s** (**-48.5% performance slowdown!** with draft `Bonsai-27B-dspark-Q4_1.gguf`, ~1.79 GB VRAM)
*   *Note: Under extreme low-bit target quantizations (like 1-bit Q1_0), the main model's forward passes run incredibly fast on GPU (37.3 t/s for 27B parameters). Because the draft model is in a heavier precision (4-bit Q4_1, 1.79 GB), its forward passes are slower per token, making speculation slower than simply running the target model directly (Quantization-Speed Inversion).*

---

## 5. Key Takeaways & Trade-offs (VRAM vs Context Size)

For consumer GPU rigs with constrained VRAM (like our RTX 4060 8 GB):

1.  **VRAM and Context Trade-Off:**
    *   Every megabyte saved on the model/draft weights is a megabyte gained for the active KV cache context window.
    *   For a 9B model using a `q4_0` KV cache quantization, saving **~385 MB** of VRAM (MTP vs DFlash) translates directly to an extra **~12,000 tokens of context depth**.
2.  **MTP is usually optimal (but measure):**
    *   On the 2026-07-20 matrix: Gemma draft **+80%**, Qwen/Ornith embedded/Hub MTP **~+46–48%**, Mythos Hub MTP **~+1%** (skip).
    *   Lowest VRAM overhead for Gemma is the tiny assistant draft (~60 MB), not a second full model.
    *   Qwen embeds heads in the main GGUF — no secondary file to track.
3.  **DFlash and Eagle-3 Standby Value:**
    *   While MTP is superior, DFlash and Eagle-3 remain valuable fallback architectures when testing custom fine-tuned models that do not support or were not trained with MTP layers.
