# MoE on small VRAM — stock `--n-cpu-moe` vs Randozart/VITRIOL

**Repo policy:** Default Search / Trials / Day-Night use upstream `llama.cpp`. Alternate llama.cpp engines and architecture forks must be versioned prebuilt releases under `llama.cpp-releases/`; upstream is the only llama.cpp source clone. Randozart [VITRIOL](https://github.com/Randozart/VITRIOL) is a separate gitignored study repository, not a llama.cpp fork or default Trial engine.

Local study clone: `VITRIOL/` at the pinned commit recorded below.

## Naming (read this first)

| Name in our docs | What it actually is |
|---|---|
| **“VITRIOL split”** / MoE offload (model cards §) | Upstream flags: `--n-gpu-layers` + `--n-cpu-moe N`. Experts **compute on CPU**. |
| **Randozart/VITRIOL** | Custom ggml host buffer (`mmap` → `mlock` → `cudaHostRegister`) + patched `llama-server`. Experts stay in page-locked RAM; **GPU** matmuls over **PCIe DMA**. Optional Chimera (CUDA MoE + Vulkan dense). |

Same product goal (big MoE on ≤8 GB). Opposite compute path. Do not treat Codacus YouTube flags as “running VITRIOL.”

- YouTube (Codacus, stock-style flags): https://www.youtube.com/watch?v=ZwNCsUTNWOA  
- Upstream project: https://github.com/Randozart/VITRIOL (studied 2026-07-28 @ `576ef69`)

---

## Path A — what this repo runs (upstream llama.cpp)

### Core insight
MoE activates few experts per token. Put **attention + shared + routing** on GPU; keep **routed experts** in CPU/RAM via `--n-cpu-moe`. Active compute is small; CPU bottleneck often acceptable on a strong desktop CPU.

### The 2-knob split

```
-ncmoe 40            # --n-cpu-moe 40 — MoE experts of first N layers on CPU
-ngl 99              # --n-gpu-layers 99 — max attention/shared on GPU
-cache-type-k q4_0   # K (and usually V) KV quant — measure quality
-c <CTX_SIZE>        # Baseline CTX_SIZE (user-configured; code floor 2048)
```

- `--n-gpu-layers N` — how much non-expert path sits on GPU  
- `--n-cpu-moe N` — how many layers force MoE experts to CPU (`N = block_count` ⇒ all experts CPU)

### Harness default
`N_CPU_MOE=None` → `ServerIntent.from_config` reads GGUF `block_count` → `--n-cpu-moe {block_count}`.  
`N_CPU_MOE=0` only when MoE fits **physical** VRAM. Explicit `N>0` overrides. Dense: never partial offload / Windows shared-memory spill.

### Preflight
`estimate_vram_mb` must receive `n_cpu_moe`. Without it, estimator charges full GGUF to VRAM and falsely rejects 14–20 GB MoE on 8 GB. Offload fraction ≈ `min(1, N / block_count)`.

### Architecture class
MoE vs dense = GGUF metadata only (`expert_count > 1` or arch name contains `moe`). No filename heuristics.

### When it helps
- Helps: large MoE (35B+ total, ≤~5B active) on ≤16 GB VRAM with a decent CPU.  
- Skip / set `0`: MoE that already fits physical VRAM (e.g. LFM2.5-8B-A1B).  
- Dense: never use as a “spill” strategy.

### Our rig vs Codacus-style stock numbers

| | Codacus (video) | The operator host (upstream) |
|---|---|---|
| GPU | GTX 1070 8 GB | discrete 8 GB-class NVIDIA |
| Path | stock-ish flags | `--n-cpu-moe` = `block_count` |
| Example | ~18 t/s @ ~132k (claim) | Qwen3.6-35B ~22 t/s @ 65k; Ornith-35B / Laguna similar band |

Exact rows: `results.tsv` + model cards. `TPS_FLOOR` default 20 — lower for heavy MoE if needed.

### Upstream knobs still worth hill-climbing
- `--n-cpu-moe` sweep (partial vs full) — VRAM ↔ TPS trade  
- KV quant (`cache-type-k/v`) — measure; do not cargo-cult VITRIOL’s “V must be f16” onto stock without a Trial  

### Host RAM Pinning (`--no-mmap` & `--mlock`) — Empirical MoE Impact (2026-08-04)

When MoE experts sit in CPU RAM (`--n-cpu-moe N`), standard OS `mmap` lazy-loading introduces page-fault overhead during token-by-token expert activation. Pre-loading and locking host memory eliminates this bottleneck:

| Strategy | POCKET-35B-Q3_K_M (`N_CPU_MOE=40`) @65k | Delta |
|---|---|---|
| Baseline (`mmap`, no mlock) | 33.7 t/s | Baseline |
| **`--no-mmap`** | 37.4 t/s | **+11.0%** (pre-loads experts upfront, no disk I/O) |
| **`--no-mmap --mlock`** | **38.1 t/s** | **+13.1%** (pins 15.6 GB in physical RAM, zero OS paging) |

*Rule:* Use `NO_MMAP=True` and `MLOCK=True` in `config.py` when system host RAM has sufficient headroom (validated via `HOST_MEMORY_PREFLIGHT`). Those Baseline keys are the contract; the harness emits `--load-mode` (`none` / `mlock`), since `--no-mmap` / `--mlock` were removed upstream (#26934) and the table above is the historical measurement. On small dense models (e.g. LFM2.5 1.2B), `NO_MMAP=True` yields ~3% speedup.
- MTP / `--spec-type draft-mtp` — on **this** path, speculative + heavy CPU experts can **hurt** (sync); validate per model (see `local-models-low-vram-configs.md`)  
- `--n-cpu-moe-draft` when draft is MoE  
- Fit-first: `N_CPU_MOE=0` when preflight says full GPU fits  

---

## Path B — Randozart/VITRIOL (fork; optional future experiment)

**Not wired into `benchmark_search` / `autoloop`.** For operators who later want to try GPU-over-PCIe experts on Linux (or a Windows port of the DMA path).

### Mechanism (compressed)
1. Custom ggml CUDA buffer type: anonymous `mmap` → `MADV_HUGEPAGE` → `mlock` → `cudaHostRegister`  
2. `is_host=true` so scheduler keeps expert tensors in host; `MUL_MAT_ID` fast paths (MMQ/MMVQ) read `src0->data` over PCIe DMA  
3. Dense / attention may stay VRAM or, in **Chimera**, move to Vulkan via `VK_EXT_external_memory_host`  
4. Needs `CAP_IPC_LOCK` on the server binary (`vitriol setup`) so `mlock` works  

CPU is mostly orchestrator — weak/old CPUs (no AVX2) can still run; **PCIe width/bandwidth** dominates.

### Claimed peak (their hardware, not ours)
GTX 1070 Ti, PCIe 3.0 **x16**, ~15 GB RAM, Linux: Qwen3.6-35B IQ2_M + Chimera + MTP N=2 + pin 8 → **~23.3 t/s** (their 2026-05-22 server note). Mellum2-12B-A2.5B Q4_K_M → **~27.1 t/s**. Removing a second GPU that forced **x8** was their largest single win (~5.7 → ~9 t/s before Chimera).

### Ideas worth remembering on *upstream* (no fork)

| VITRIOL finding | Transfer to official llama.cpp? |
|---|---|
| PCIe x16 vs x8 kills streaming-expert throughput | Yes if ever measuring host↔GPU weight traffic; less critical for pure CPU-expert `--n-cpu-moe`, still matters for any DMA/spill |
| Pin first N expert layers in VRAM | Upstream has no `--pin-layers`; closest = lower `--n-cpu-moe` so late layers’ experts stay GPU |
| Chimera CUDA+Vulkan | Fork-only |
| Predictive expert prefetch / LRU VRAM expert pool | Fork-only today; watch upstream PRs (they cite ggml PRs #11397, #6387, #11571 lineage) |
| MTP N=2 often best; larger N wastes drafts if accept≈1/N | Yes — sweep `--spec-draft-n-max` on MTP GGUFs |
| MTP “flat” on some Pascal sweeps | Hardware-specific; always measure on the operator host |
| `--cache-type-v` q4/q8 “corrupts” under VITRIOL | **Their** stack; do not assume for upstream — Trial both |
| Secondary GPU dropping primary to x8 | Hardware hygiene for any PCIe-bound path |
| Memory shim / Hebbian SQLite | App-layer, orthogonal to Search harness |
| Kernel module / ALKA SSD→GPU | Experimental; out of scope |

### Platform notes (from their README)
- **Linux + NVIDIA:** verified path  
- **Windows + NVIDIA:** `cudaHostRegister` “likely”; CLI is bash — needs a wrapper; untested by them  
- **Apple unified memory / Intel SYCL:** little or no benefit  
- Requires their **pinned submodule** (`llama.cpp` branch `vitriol-mellum2`), not our submodule  

### If someone tries it later
1. Clone recursive: `https://github.com/Randozart/VITRIOL`  
2. Build their `llama.cpp` with CUDA (+ optional Vulkan for Chimera)  
3. `./vitriol setup` then `./vitriol config` / `run` / `serve`  
4. Keep results **offline** — do not mix fork TPS into Pareto rows without labeling engine = VITRIOL fork  
5. Compare A/B against the **same** GGUF on upstream `--n-cpu-moe` on the operator host before changing policy  

---

## See also
- [Qwen3.6-35B-A3B](qwen3.6-35b-a3b.md) — Codacus demo model family  
- [Gemma-4-26B-A4B](gemma-4-26b-a4b.md) — same stock split, `n-cpu-moe 30`  
- [low-vram-optimizations.md](../discovery/low-vram-optimizations.md) — MoE + anti-paging  
- [local-models-low-vram-configs.md](../discovery/local-models-low-vram-configs.md) — per-family 8 GB recipes  
- [inference-engines-landscape.md](../discovery/inference-engines-landscape.md) — where forks sit vs default  
)
