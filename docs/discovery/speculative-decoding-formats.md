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
    *   Since they cannot be combined, you must choose the single best speculative type. For our Qwen/Gemma-4 target models, **MTP is always the optimal choice** (providing the highest speedup at the lowest VRAM footprint).

---

## 3. Deep-Dive into Formats

### A. Multi-Token Prediction (MTP) — `draft-mtp`
*   **How it works:** Native to Qwen2.5/3.5/3.6 and Gemma-4. Google trained official "assistant" models (e.g. `google/gemma-4-E4B-it-assistant`) that share the base model's vocabulary and embeddings. Unsloth quantized these into standalone GGUF drafts.
*   **Why it's best for us:** MTP has a very high token acceptance rate because it is specifically trained by the model creators. It achieves **1.6x–2.0x speedups** locally.
*   **In llama.cpp:** 
    - Qwen models (like Qwen3.5-9B and Qwen3.6-35B) have the MTP draft head **embedded directly in the main GGUF file** (no separate `-md` file needed).
    - Gemma-4 models have the MTP draft head in a **separate file** (e.g. `models/draft/mtp-gemma-4-E4B-it.gguf`). You must pass `--spec-draft-model <path>`.
    - **Ornith-1.0 models:** As they are built on top of Gemma-4/Qwen, they support MTP. However, the standard Unsloth UD files we use do **not** have MTP heads. To run MTP on Ornith-1.0, you must download community GGUFs with MTP heads grafted onto them (e.g. repositories by `satgeze/ornith-35b-1m`, `wang-yang`, or `SC117`).

### B. Eagle 3 — `draft-eagle3`
*   **How it works:** A tree-based speculative decoder that trains a small recurrent neural network head directly on top of the target model's hidden states.
*   **Availability:** While Unsloth does not bundle Eagle drafts for our current targets, the community hosts Eagle-3 GGUF draft models for target models on Hugging Face (e.g. `Ex0bit/Qwen3.6-27B-PRISM-PRO-DQ`, `thoughtworks/Gemma-4-31B-Eagle3`, or `RedHatAI/gemma-4-31B-it-speculator.eagle3`). They must be paired using `--spec-type draft-eagle3` and `-md <path>`.
*   **Error Case:** If you try to pass an MTP draft model to `--spec-type draft-eagle3`, it will fail to load with:
    `failed to initialize speculative decoding context: draft model is not eagle3`

### C. DFlash & DSpark — `draft-dflash` (DFlash) / `dspark` (DSpark)
*   **How it works:** Open-source research released by DeepSeek (the **DeepSpec** framework).
    - **DFlash:** A block-parallel drafter utilizing a diffusion-like block process to predict blocks of tokens in a single step (e.g. `spiritbuun/Qwen3.6-27B-DFlash-GGUF` or `williamliao/gemma-4-31B-it-DFlash-GGUF`).
    - **DSpark:** A semi-autoregressive model combining a parallel backbone with a lightweight serial head to reduce "suffix decay" (loss of coherence at the end of draft sequences).
*   **Availability:** In our repository, the `Bonsai-27B` model (a quantized Qwen3.6-27B fork) utilizes DSpark speculative decoding (`Bonsai-27B-dspark-Q4_1.gguf`) in the PrismML fork. There are also official DeepSeek DSpark releases for Gemma-4 (e.g. `deepseek-ai/dspark_gemma4_12b_block7` and community GGUF conversions like `ankk98/dspark-gemma4-12b-block7-Q4_0-GGUF`). Other target models require custom training via DeepSpec or finding matching community GGUF drafts.

### D. N-gram Decoders — `ngram-cache` / `ngram-simple`
*   **How it works:** Instead of loading a neural network, these search the model's own active KV cache to find repeating patterns of words (N-grams) and predict subsequent tokens based on past context.
*   **Why to use it:** 0 MB VRAM footprint. Good for code completion or summarizing highly repetitive text.
*   **Performance:** ~5-10% speedup on repetitive text; 0% speedup on creative/non-repetitive text.

---

## 4. Local Performance Comparison

Tested on our local rig (RTX 4060 8 GB VRAM) on a standard short prompt:

### Gemma-4 E4B:
*   **Baseline (`none`):** **61.6 t/s**
*   **N-gram Cache (`ngram-cache`):** **65.4 t/s** (+6.1% speedup, 0 VRAM cost)
*   **N-gram Simple (`ngram-simple`):** **65.6 t/s** (+6.4% speedup, 0 VRAM cost)
*   **Multi-Token Prediction (`draft-mtp`):** **103.7 t/s** (**+68.3% speedup!**, cost: ~60 MB VRAM)
*   *Note: Attempting to load the MTP draft file with `--spec-type draft-eagle3` throws a model initialization error and falls back silently to the non-speculative baseline (64.4 t/s).*

### Qwen3.5-9B:
*   **Baseline (`none`):** **39.1 t/s**
*   **DFlash (`draft-dflash`):** **51.3 t/s** (+31.2% speedup, cost: ~765 MB VRAM)
*   **Multi-Token Prediction (`draft-mtp`):** **69.1 t/s** (**+76.7% speedup!**, cost: ~380 MB VRAM)

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
2.  **MTP is the Optimal Choice:**
    *   **Highest Speed:** +76.7% speedup vs +31.2% for DFlash.
    *   **Lowest VRAM Footprint:** ~380 MB VRAM overhead vs ~765 MB for DFlash (saving nearly 50% VRAM overhead).
    *   **Zero File Complexity (for Qwen):** Qwen's MTP draft heads are embedded directly in the main GGUF file, requiring no secondary `-md` flag or file tracking.
3.  **DFlash and Eagle-3 Standby Value:**
    *   While MTP is superior, DFlash and Eagle-3 remain valuable fallback architectures when testing custom fine-tuned models that do not support or were not trained with MTP layers.
