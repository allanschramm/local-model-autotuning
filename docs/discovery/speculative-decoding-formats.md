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

## 2. Deep-Dive into Formats

### A. Multi-Token Prediction (MTP) — `draft-mtp`
*   **How it works:** Native to Qwen2.5/3.5/3.6 and Gemma-4. Google trained official "assistant" models (e.g. `google/gemma-4-E4B-it-assistant`) that share the base model's vocabulary and embeddings. Unsloth quantized these into standalone GGUF drafts.
*   **Why it's best for us:** MTP has a very high token acceptance rate because it is specifically trained by the model creators. It achieves **1.6x–2.0x speedups** locally.
*   **In llama.cpp:** 
    - Qwen models (like Qwen3.5-9B and Qwen3.6-35B) have the MTP draft head **embedded directly in the main GGUF file** (no separate `-md` file needed).
    - Gemma-4 models have the MTP draft head in a **separate file** (e.g. `models/draft/mtp-gemma-4-E4B-it.gguf`). You must pass `--spec-draft-model <path>`.

### B. Eagle 3 — `draft-eagle3`
*   **How it works:** A tree-based speculative decoder that trains a small recurrent neural network head directly on top of the target model's hidden states.
*   **Availability:** While Unsloth does not bundle Eagle drafts for our current targets, the community uploads Eagle-3 GGUF draft models for some target models on Hugging Face (e.g. `Ex0bit/Qwen3.6-27B-PRISM-PRO-DQ`). They must be paired using `--spec-type draft-eagle3` and `-md <path>`.
*   **Error Case:** If you try to pass an MTP draft model to `--spec-type draft-eagle3`, it will fail to load with:
    `failed to initialize speculative decoding context: draft model is not eagle3`

### C. DFlash & DSpark — `draft-dflash` (DFlash) / `dspark` (DSpark)
*   **How it works:** Open-source research released by DeepSeek (the **DeepSpec** framework).
    - **DFlash:** A block-parallel drafter utilizing a diffusion-like block process to predict blocks of tokens in a single step (e.g. community files like `spiritbuun/Qwen3.6-27B-DFlash-GGUF`).
    - **DSpark:** A semi-autoregressive model combining a parallel backbone with a lightweight serial head to reduce "suffix decay" (loss of coherence at the end of draft sequences).
*   **Availability:** These require specialized draft models trained via the DeepSpec pipeline. In our repository, the `Bonsai-27B` model (a quantized Qwen3.6-27B fork) utilizes DSpark speculative decoding (`Bonsai-27B-dspark-Q4_1.gguf`) in the PrismML fork to accelerate local inference. Other models (like Qwen3.5-9B or Gemma-4) require custom training through DeepSpec.

### D. N-gram Decoders — `ngram-cache` / `ngram-simple`
*   **How it works:** Instead of loading a neural network, these search the model's own active KV cache to find repeating patterns of words (N-grams) and predict subsequent tokens based on past context.
*   **Why to use it:** 0 MB VRAM footprint. Good for code completion or summarizing highly repetitive text.
*   **Performance:** ~5-10% speedup on repetitive text; 0% speedup on creative/non-repetitive text.

---

## 3. Local Performance Comparison (Gemma-4 E4B)

Tested on our local rig (RTX 4060 8 GB VRAM) on a standard short prompt:

*   **Baseline (`none`):** **61.6 t/s**
*   **N-gram Cache (`ngram-cache`):** **65.4 t/s** (+6.1% speedup, 0 VRAM cost)
*   **N-gram Simple (`ngram-simple`):** **65.6 t/s** (+6.4% speedup, 0 VRAM cost)
*   **Multi-Token Prediction (`draft-mtp`):** **103.7 t/s** (**+68.3% speedup!**, cost: ~60 MB VRAM)

*Note: Attempting to load the MTP draft file with `--spec-type draft-eagle3` throws a model initialization error and falls back silently to the non-speculative baseline (64.4 t/s).*
