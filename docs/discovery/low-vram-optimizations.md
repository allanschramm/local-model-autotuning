# Low VRAM Local LLM Optimization Guide

This guide documents the design strategies, model formats, and configuration settings required to run large or complex models efficiently on consumer GPUs with limited VRAM (such as an discrete 8 GB-class NVIDIA).

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

*   **Expert offload (this repo / upstream llama.cpp):** Keep attention + routing on the GPU; force routed experts to CPU RAM with `--n-cpu-moe <N>` (often `N = block_count`). Harness: `N_CPU_MOE=None` → auto full offload. Detail: [vitriol-technique.md](../models/vitriol-technique.md) Path A.
*   **Trade-off:** 35B-class MoE fits in 8 GB VRAM; speculative decoding + heavy CPU experts can sync-stall — validate per model ([local-models-low-vram-configs.md](./local-models-low-vram-configs.md)).
*   **`--cpu-moe` vs `--n-cpu-moe` — one ruler, two ends:** both select the same tensors (`LLM_FFN_EXPS_REGEX`, `common.h:1130`); `-cmoe` applies it to every block, `-ncmoe N` instantiates it per block for the first N only (`common.h:1142-1149`). So `-cmoe` ≡ `-ncmoe <block_count>`, and manual `-ot` regexes that send `*_exps` to CPU compile to the same thing (measured ≡ within noise, 2026-08-28). Partial offload (`N < block_count`) keeps the LAST blocks' experts on GPU: on a 24 GB-class card that bought ~+8 % decode for ~2× expert VRAM (125B-class community run, PR-branch arch); on 8 GB-class the keepout reverses it — full offload (`N = block_count`) measured best on this rig. Pick N from free VRAM after KV + cache, not by folklore.
*   **Not our default:** Randozart/VITRIOL keeps experts in page-locked host RAM and runs MoE matmuls on the **GPU over PCIe DMA** (custom fork). Useful ideas (PCIe width, pin-vs-`n-cpu-moe`, MTP N sweeps) are absorbed in the technique note; Search stays on official `llama.cpp`.
*   **Expert profile cache (fork study, not upstream):** profile which experts the router picks (~512 generated tokens × two merged prompts), then pin that fixed set per layer in VRAM slots at load time; non-resident experts compute on CPU via a two-pass zero-add trick. Key insight: long-run expert usage is flat (no hot set) yet an ~80% hit rate holds because *temporal* locality beats frequency. **Measured on this rig (fork `0ac3d9b`, 2026-08-24)** — direction depends entirely on the offload split point: at FULL expert offload (`-ncmoe = block_count --no-mmap`, 48 slots ≈ 19 % coverage) the cache gains **+7 % decode alone and +46 % stacked with embedded-MTP speculative decoding** (28.5 → 42.0 t/s), at a ~11 % prefill tax; at partial offload it *lost* 9 % (slot budget starves to the dead zone). Cache activates via `llama-server` only (`llama-cli` accepts but never plumbs the flags), output is **not** token-identical to baseline, and MTP requires the separate `-MTP-GGUF` packaging (plain-GGUF files carry no draft head despite card wording). Creator's +66 % needs ~50 % coverage — unreachable for 256-expert 35B models on 8 GB-class cards.
*   **Profile-cache workflow** (fork build, server only): capture with `MOE_TRACE_OUT=<csv> llama-moe-trace -m <model>.gguf -ngl 99 -ncmoe <N> -fa 1 -c 4096 -n 512 -p "<prompt>"` for one code-ish and one chatty prompt, concatenate the CSVs; serve with `--moe-cache-profile <merged.csv> --moe-cache-slots <N>` and verify the startup line `expert cache: <layers> layers x <N> slots` (raise `-lv` if silent). Slots are **per layer**, raised until VRAM spill; stack with MTP kept at `--spec-draft-n-max 2–3` (higher inflates the per-step expert union and lands slower than no speculation). Reported decode ladder on a 12 GB card at ~50 % coverage: 45 → 6 naive graph ordering → 44 after reorder → 70 with MTP → 75 peak 80 with concurrent CPU/GPU chains.
*   **Coverage scaling rule** (predicts every gain): expected benefit scales with the fraction of the expert set that fits in VRAM. ~50% coverage → +66%; ~15–20% or less → wrong tool entirely. Measured on this rig's GGUFs: `Ling-3.0-tiny-Q4_K_M` experts are 3.90 GiB of 4.57 GiB weights (128 exp × 24 layers) — the whole expert set fits free VRAM on 8 GB-class at Day ctx, so prefer plain full offload (`-ncmoe 0`) on the pinned build over any cache; `Ornith-1.5-35B-Q4_K_M` / `Qwen3.6-35B-A3B-UD-Q4_K_M` expert sets are 18.65 / 18.22 GiB (256 exp × 41/40 layers) → slot budget on 8 GB caps near ~20–25 % coverage, i.e. the modest-gain regime per this rule (measured: +14 % decode at full offload, bullet above).
*   **Host-register prefill lever (already upstream in the pinned release):** `GGML_CUDA_REGISTER_HOST=1` page-locks mmap'd host expert weights so PCIe copies go straight DMA (~6–7 → ~20 GB/s). Measured on this rig (2026-08-24): **+78 % prefill at full expert offload** (`-ncmoe = block_count`, 704 → 1257 t/s on a 256-expert 35B) — but **neutral alone and harmful combined at partial offload**: the companion second-stream expert prefetch env var dropped prefill ~40 % when some layers' experts already live in VRAM (second stream fights compute; only pays when every layer crosses PCIe). The profile-cache flags are fork-only — evaluation builds go in their own vendor directory, never repointing `AUTORESEARCH_LLAMA_CPP_ROOT`. Evidence: [session log](../sessions/2026-08-24-codacus-fork-validation.md).
*   **Fork failure lessons (generalize beyond this fork):**
    - Dynamic LRU expert caching lost to static pinning — the best hit-rate variant was slowest. On PCIe, missing isn't what hurts — **loading is**; never load weights while running.
    - The scheduler merges alternating GPU/CPU graph segments and can drag pinned VRAM experts back across PCIe every layer (~80 MB/layer). Fix is graph ordering — one unbroken GPU run, then CPU run, then a single add (6 → 44 t/s with no math change).
    - Two agreeing measurements can both be wrong ("flat distribution" trace + "cache made it worse" bench): trace what actually executed, not what you meant to execute.

    Source attribution: Codacus video notes (<https://youtu.be/k_LostFpatg>, extracted 2026-08-23) and the companion fork `thecodacus/llama.cpp` (`perf` branch); verified against pinned-release binaries as described above.

---

## 5. Preventing Driver-Level Paging (Shared System Memory)

On Windows, when VRAM usage approaches 100%, the NVIDIA driver automatically redirects allocations to Shared System Memory (System RAM). This prevents Out-Of-Memory (OOM) crashes but degrades token throughput from ~40+ t/s to ~2 t/s.

*   **Solution:** 
    - Set the number of GPU offloaded layers (`-ngl` / `--n-gpu-layers`) conservatively, leaving at least **500 MB to 1 GB of headroom** for the OS and context growth.
    - If using MoEs, allocate CPU experts (`--n-cpu-moe`) to keep physical VRAM usage around **7.0 GB** on an 8 GB card.

---

## 6. Host Memory Pinning & Page-Fault Elimination (`--no-mmap` & `--mlock`)

> **Flags renamed (2026-09-28).** `--no-mmap` and `--mlock` were removed upstream ([#26934](https://github.com/ggml-org/llama.cpp/pull/26934)) and are rejected by the current build (`error: invalid argument`). The Baseline knobs `NO_MMAP` / `MLOCK` are unchanged and are the contract here; the harness translates them to `--load-mode` (`none` / `mlock` / `mmap` / `mmap+mlock`). The measured rows below are the historical `--no-mmap` / `--mlock` comparison and are kept as provenance.

When running MoE models with expert CPU offloading (`--n-cpu-moe N`), standard `mmap` lazy loading causes OS page-fault overhead as active experts cycle per token.

*   **`--no-mmap`:** Forces `llama.cpp` to allocate system memory explicitly and load all host weights upfront into RAM. Eliminates disk I/O stuttering during generation.
*   **`--mlock`:** Locks allocated host memory using system calls (`mlock()` / `VirtualLock()`), preventing the OS from paging out idle expert blocks to swap (`pagefile.sys`).
*   **Empirical Result (POCKET-35B MoE @ 65k context):**
    - `mmap` baseline: **33.7 t/s**
    - `--no-mmap`: **37.4 t/s (+11.0% speedup)**
    - `--no-mmap --mlock`: **38.1 t/s (+13.1% speedup)**
*   **Safety Guard:** Always verify host RAM headroom via `HOST_MEMORY_PREFLIGHT` before enabling `--mlock` to avoid system OOM.
