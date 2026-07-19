# Low VRAM Local LLM Optimization Guide

This guide documents the design strategies, model formats, and configuration settings required to run large or complex models efficiently on consumer GPUs with limited VRAM (such as an RTX 4060 8 GB).

---

## 1. The VRAM Allocation Equation

VRAM consumption in local inference is determined by three main factors:
$$\text{Total VRAM} = \text{Model Weights} + \text{KV Cache Context} + \text{Inference Overhead (CUDA/System)}$$

To prevent driver-level memory paging (which swaps memory to system RAM via PCIe and slows inference to a crawl of ~2–3 t/s), the total allocation must stay strictly under the physical hardware limit.

---

## 2. Advanced Quantization Formats

Choosing the right quantization format is the most critical step to fit large models into limited VRAM.

### A. GGUF (K-Quants & IQ-Quants) — *Best for Hybrid Offloading*
*   **How it works:** Splits model layers dynamically between VRAM and system RAM.
*   **Low-VRAM Tip:** Utilize Importance Matrix (imatrix) quantized GGUFs like `IQ3_XXS` or `IQ2_XS` (2-bit to 3-bit precision). These quants retain high semantic intelligence while shrinking massive models (e.g., 30B+) to under 8 GB.

### B. EXL2 (ExLlamaV2) — *Best for Pure GPU Speed*
*   **How it works:** A GPU-only format supporting **fractional bits-per-weight (bpw)** (e.g., 3.65 bpw or 3.85 bpw).
*   **Low-VRAM Tip:** Instead of stepping down from 4-bit to 3-bit, you can fine-tune the bit-rate to the exact megabyte needed to fit your target model and KV cache perfectly into 8 GB VRAM. Runs at extreme speeds (100+ t/s) by keeping weights entirely on the GPU.

### C. HQQ (Half-Quadratic Quantization) — *Best for Calibration-Free Compression*
*   **How it works:** A highly efficient low-bit (1-bit to 4-bit) quantization format that does not require calibration data.
*   **Low-VRAM Tip:** HQQ is highly compatible with modern, fast backends (like `torchao` and `Marlin`), enabling fast 2-bit or 3-bit inference on GPUs.

---

## 3. KV Cache Optimization

The Key-Value (KV) cache grows linearly with context size and batch size. At 65k context depth, the KV cache alone can exceed **4 GB of VRAM** for a 9B model in FP16 precision.

*   **KV Quantization:** Compress the KV cache representation to **4-bit (`q4_0`)** or **8-bit (`q8_0`)** using:
    `-ctk q4_0 -ctv q4_0`
    This reduces the KV cache size by **75%**, saving gigabytes of VRAM and reducing memory bandwidth bottlenecks.
*   **Chunked Prefill:** During prompt evaluation (prefill), large batches of input tokens are processed in parallel, causing temporary VRAM spikes. Setting a small prefill batch size (e.g., `-b 512 -ub 128` in `llama.cpp`) limits these memory spikes.

---

## 4. Sparse Mixture-of-Experts (MoE) Routing

For MoE models (like Qwen3.6-35B-A3B or Gemma-4 26B-A4B), you do not need to keep all weights in VRAM since only a fraction of experts are active for any given token.

*   **Expert Offloading:** Keep the model's attention heads and routing layers on the GPU, while offloading the heavy experts to CPU RAM using `--n-cpu-moe <N>` (e.g. `--n-cpu-moe 30` or `40`).
*   **Trade-off:** This allows 35B models to fit comfortably in 8 GB VRAM, but you must **disable speculative decoding** to avoid CPU-GPU roundtrip synchronization bottlenecks (as documented in our benchmark guide).

---

## 5. Preventing Driver-Level Paging (Shared System Memory)

On Windows, when VRAM usage approaches 100%, the NVIDIA driver automatically redirects allocations to Shared System Memory (System RAM). This prevents Out-Of-Memory (OOM) crashes but degrades token throughput from ~40+ t/s to ~2 t/s.

*   **Solution:** 
    - Set the number of GPU offloaded layers (`-ngl` / `--n-gpu-layers`) conservatively, leaving at least **500 MB to 1 GB of headroom** for the OS and context growth.
    - If using MoEs, allocate CPU experts (`--n-cpu-moe`) to keep physical VRAM usage around **7.0 GB** on an 8 GB card.
