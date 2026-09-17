# Qwen3.8-35B-A3B — Model Card and GGUF Inventory

## Status

Validated and trialed on local rig (2026-09-17). Measured full Objective Vector (`agentic-full` = Claw 15 + Coding 10) at context 65,536 (`q4_0`). **Status:** `on_front` (Pareto frontier pick #17 DAY, #14 NIGHT).

- **Inventory date:** 2026-09-17
- **Official base repository:** [empero-ai/Qwen3.8-35B-A3B-Distill](https://huggingface.co/empero-ai/Qwen3.8-35B-A3B-Distill)
- **Official GGUF repository:** [empero-ai/Qwen3.8-35B-A3B-Distill-GGUF](https://huggingface.co/empero-ai/Qwen3.8-35B-A3B-Distill-GGUF)
- **Base architecture:** [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
- **License:** Apache-2.0
- **Family:** Qwen3.6-class hybrid MoE (30 Gated DeltaNet layers + 10 full-attention layers). 41 GGUF blocks (`qwen35moe`), 256 routed experts with 8 active per token (~3B active parameters out of 35B total). Embedded-MTP nextn tensors present (`nextn_predict_layers: 1`).

---

## Architecture & GGUF Metadata

Verified via harness read-only inspection (`scripts/model_info.py`) on `Qwen3.8-35B-A3B-Q4_K_M.gguf`:

| Header Key | Value | Description |
|---|---|---|
| `general.architecture` | `qwen35moe` | Hybrid MoE (Gated DeltaNet + Gated Attention) |
| `block_count` | 41 | 40 transformer layers + shared head / nextn block |
| `expert_count` | 256 | 256 routed experts per MoE block |
| `expert_used_count` | 8 | 8 active experts per token (3.125% sparsity) |
| `context_length` | 262144 | Native publisher context limit (evaluated at 65536) |
| `head_count_kv` | 2 | Grouped-Query Attention (GQA) with head dimension 256 |
| `nextn_tensors` | 4 | `blk.40.nextn.{eh_proj,enorm,hnorm,shared_head_norm}` |
| `file_size_mib` | 20707.6 | 20.22 GiB (weights file) |
| `kv_f16_mb@65536` | 2624.0 | FP16 KV size; at `q4_0` consumes ~734.7 MiB |

---

## Hardware Recipe (RTX 4060 8GB + 32GB RAM)

Because 35B MoE models require the entire ~20.2 GB weights file to be resident across RAM/VRAM:
- **Default offload (`N_CPU_MOE = None` / 41):** Leaves ~4 GB of GPU VRAM idle and loads all 19.1 GB of experts into CPU RAM, resulting in physical RAM exhaustion and triggering the RAM Circuit Breaker.
- **Optimal offload (`N_CPU_MOE = 31`):** Keeps **10 expert layers on the RTX 4060 VRAM** and offloads 31 to CPU RAM.
  - **Peak VRAM:** 7.7 GB (fits comfortably under the 7,932 MB keepout limit, zero WDDM shared memory spill).
  - **Host RAM saved:** ~4.2 GB relieved from system RAM, keeping physical RAM usage well within safety bounds.
  - **Circuit Breaker Floor:** `FREE_RAM_FLOOR_MB = 128`.
  - **Context:** 65,536 tokens fully preserved (`q4_0` KV cache).

---

## Measured Benchmark Results

Full Objective Vector Trial logged to [`results.tsv`](file:///D:/Dev/ailocal-nexus-system/ailocal-model-autotuning/results.tsv) (`trial_id: 01acfbac-90da-4e06-855b-79531fc06051`):

| Metric | Measured Value | Threshold / Target | Status |
|---|---|---|---|
| **Context Length (`CTX_SIZE`)** | **65,536** | Inviolable frontier axis | **PASS** |
| **Throughput (`bench_tg` 512)** | **33.5 t/s** (33.0, 33.5, 34.0) | $\ge 20.0\text{ t/s}$ | **PASS** |
| **Combined Generation TPS** | **42.8 t/s** | Pareto frontier metric | **PASS** |
| **Peak VRAM** | **7.7 GB** | $\le 7.93\text{ GB}$ (Dedicated VRAM) | **PASS** |
| **Coding Score** | **0.4900** | Coding-10 benchmark suite | **PASS** |
| ↳ `LiveCodeBench (LCB)` | 0.5000 (5/10) | Hard competitive coding | Completed |
| ↳ `MBPP+` | 0.8000 (8/10) | Python synthesis | Completed |
| ↳ `HumanEval+` | 0.4000 (4/10) | Standard function synthesis | Completed |
| ↳ `BigCode Hard` | 0.1000 (1/10) | Edge case coding | Completed |
| **Claw Full Agentic (15 tasks)** | **0.7333** (11/15 passed) | Multi-turn agentic evaluation | **PASS** |
| **Trial Status** | **`on_front`** | Frontier ranking (#17 DAY / #14 NIGHT) | **PASS** |

### Agentic Task Breakdown (Claw-15):
- **PASS:** T002 (email triage: 0.50), T004 (calendar scheduling: 1.00), T006 (email reply draft: 1.00), T008 (todo management: 1.00), T010 (contact lookup: 1.00), T012 (expense report: 0.75), T014 (meeting notes: 0.70), T016 (kb search: 0.50), T018 (ticket triage: 0.85), T044 (outage research: 1.00), T046 (cve research: 1.00).
- **FAIL:** T048 (oss comparison: 0.12), T050 (regulatory research: 0.12), T053 (finance merger: 0.20), T054 (nflx arppu: 0.00).

---

## Recommended Serving Baseline

From [`models/aliases/qwen3.8-35b-a3b/config.yaml`](file:///D:/Dev/ailocal-nexus-system/ailocal-model-autotuning/models/aliases/qwen3.8-35b-a3b/config.yaml):

```yaml
alias: qwen3.8-35b-a3b
model: models/empero-ai/Qwen3.8-35B-A3B-Distill-GGUF/Qwen3.8-35B-A3B-Q4_K_M.gguf
port: 18080
llama_cpp_root: llama.cpp-releases/upstream/b10867
flags:
  - --jinja
  - --ctx-size 65536
  - --parallel 1
  - --n-gpu-layers 99
  - --n-cpu-moe 31
  - --cache-type-k q4_0
  - --cache-type-v q4_0
  - --flash-attn on
  - --cache-reuse 256
  - --temp 0.6
  - --top-p 0.95
  - --top-k 20
  - --min-p 0.0
  - --repeat-penalty 1.0
  - --presence-penalty 0.0
status: ready
```
