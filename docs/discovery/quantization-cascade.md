# Quantization Format Selection Guide

Best Choice Cascade for selecting a GGUF quantization format. Prioritizes **quality first**, then **size efficiency**, then **hardware constraints**.

## The Golden Rule

Choose a quantization by measured quality, actual bytes, and throughput at the target context. A publisher prefix (`UD`, `GSQ-RCO`) or nominal bit-depth is not a universal quality ranking. The GSQ-RCO evidence below directly contradicts an unconditional UD-first rule. [S3–S5]

## Selection Procedure

1. Fix the base checkpoint, evaluation scope, configured context, and runtime before comparing quantization recipes.
2. Inspect actual file bytes, tensor allocation, and the bpw denominator. Include optional heads/projectors and all required shards in sizing.
3. Check full memory requirements: resident weights, KV cache, runtime workspace, and host RAM. Dense models must fit physical VRAM; MoE expert offload follows the repo's supported gates.
4. Validate each eligible artifact, then compare complete Objective Vectors at matched budgets. Keep sampler, reasoning, KV, and speculative settings fixed for the quantization comparison.
5. Select using measured quality and throughput. Higher nominal precision, UD, and GSQ-RCO are candidates rather than universal winners. Preserve the target context when addressing resource pressure. [S3?S5]

## HF 4-bit (AWQ / GPTQ / NVFP4) vs GGUF Q4

Same parameter count does **not** imply the same VRAM. Before comparing llama.cpp to vLLM/SGLang:

| Pack type | What usually gets ~4-bit | Typical extras left fat | Rough 9B-class text footprint |
| :-- | :-- | :-- | :-- |
| **GGUF Q4 / UD-Q4** (llama.cpp) | Most text tensors (incl. aggressive embd/output in many UD packs) | Often **text-only** GGUF (no ViT in VRAM) | ~5–6 GB weights |
| **HF AWQ / GPTQ / NVFP4** (vLLM/SGLang) | Mostly `Linear` layers the recipe targeted | Vision tower, `lm_head`, hybrid SSM/gates often **BF16** | Often **~8 GB+** for multimodal hybrids |

Rules:

1. Compare **engine-reported weight load** (or disk size of the actual shard) to `VRAM_LIMIT_MB` — not the “Q4 / NVFP4” label alone.
2. **Same-bytes** comparison (e.g. GGUF load on every engine that supports it) isolates the engine. **Native-stack** comparison (GGUF vs AWQ vs NVFP4) is a **compound** claim — label it as such.
3. Reject oversized packs under the repo no-spill contract. Never “nudge util until it loads.”

## GSQ + RCO: quantization recipe versus deployment format

**GSQ improves a tensor's quantized values; RCO chooses which precision each tensor receives under a budget.** Neither name is a new GGUF storage type, a speculative decoder, or a KV-cache setting. The released GGUFs assemble ordinary GGML tensor types, including mixed IQ/K types and retained floating-point tensors. [S1–S5]

| Component | Role | What it does not establish |
| :-- | :-- | :-- |
| GSQ — Gumbel-Softmax Quantization | Post-training scalar quantization: jointly optimizes discrete grid assignments and group scales through a differentiable Gumbel-Softmax relaxation; projects the result into the deployment format. | A new inference kernel, guaranteed task quality, or automatic speedup. |
| RCO — Riemannian Constrained Optimization | Searches assignments of compression options to groups against the joint model loss, subject to the cost budget. For these releases, options are tensor quantization types. | Every tensor at the suffix's precision, or a VRAM limit covering KV/runtime allocations. |
| GGUF / GGML tensor types | Container and stored representation that the runtime loads and computes. | Which optimizer produced the quantized values. |

RCO relaxes assignments into softmax probabilities. The expected-cost equality defines a manifold in logit space; tangent projection, binary-search retraction, and momentum transport wrap an Adam step. A budget-constrained discrete solver with Gumbel straight-through estimation supplies feasible hard assignments. The exact-budget claim concerns this optimization problem, not exact runtime memory, nor a proof that every packaged byte equals a rounded target. [S2, S4]

### Differences from GPTQ and AWQ

GPTQ uses approximate second-order information for post-training weight quantization. AWQ protects salient weights using activation-aware scaling. GSQ instead refines scalar assignments and scales with the relaxed discrete optimization; its reference pipeline can initialize from GPTQ, so these methods are not necessarily mutually exclusive. RCO is the allocation search over candidate representations, a different level from producing a candidate quantized tensor. [S1, S4, S6, S7]

Optimizing quantizer parameters on calibration data remains post-training quantization in the GSQ paper; it is not evidence of model pretraining or a separate QAT checkpoint. A QAT provenance claim needs the base checkpoint's own training documentation. GSQ/RCO names do not supply it. [S1, S4]

### Read actual bpw and allocation

The suffix can name a budget class while individual tensors use lower/higher IQ/K types or BF16/F32. An inspected publisher allocation dump contains IQ1 through IQ4, Q2_K/Q4_K, and floating-point tensors in the same nominal IQ2 release. A count histogram is not a parameter-weighted bpw histogram: a small norm tensor and a large matrix count equally. [S3, S8]

Record the denominator before comparing bpw. A publisher may report whole-file average precision, or average over only the transformer weights searched by RCO. Fixed lookup embeddings, vision projectors, MTP heads, metadata, and extra shards can be outside that search. Total required bytes and loaded memory remain the sizing authorities. [S3, S5]

```powershell
# Install the HF CLI if absent: uv tool install huggingface_hub
$env:PYTHONIOENCODING = 'utf-8'
hf models card <publisher>/<release>
hf models list <publisher>/<release> -R --json
# Copy the exact small text path listed above; inspect allocation without weights.
hf download <publisher>/<release> tensor-allocation/<allocation-name>.txt
```

Record the repository commit from model metadata (`hf models info <publisher>/<release> --json`) and use `--revision <commit>` on file listing/download commands. The card command reads the current card; use a pinned README download for reproducible extraction. Inspect README and allocation text before downloading weights. The paper/code GitHub organization is `IST-DASLab`; the example HF publisher is `ISTA-DASLab`. [S3–S5, S8]

### Runtime, MTP, and KV cache

The publisher describes these exports as standard GGUFs usable in unmodified llama.cpp. This is a **publisher compatibility claim, unverified locally for GSQ-RCO**: the engine must still support the architecture and every stored tensor type. A recipe suffix alone does not require a new build; validate the exact artifact against the pinned release before changing the runtime. [S3, S5]

MTP is separate packaging and execution: optional MTP heads add weights and require runtime speculative support. Hold MTP disabled for the first quantization comparison, then test it as a separate factor; count the extra head in the memory budget. Quantized weights do not quantize KV automatically: llama.cpp exposes distinct K/V cache types, whose support and effect need their own validation. [S3, S9]

### Claims and fair local comparison

Publisher quality tables are **external / unverified on this rig**. A reported task average near BF16, recovery above 100%, or “task-lossless” means results on those tasks under that protocol; it does not mean identical weights, logits, general capability, or a guaranteed improvement. Published comparisons already show metric-dependent wins and losses against another recipe. Quantization kernel choice can also trade quality for throughput at similar sizes. [S3, S5]

1. Use the same base checkpoint and text/vision scope; pin artifact revisions, exact bytes, tensor allocation, engine tag, and templates.
2. Match actual weight budget as closely as possible; report residual size differences. Keep configured context, KV precision, offload, batch/ubatch, sampler, reasoning/token budgets, prompts, and benchmark revisions fixed. Compare complete Objective Vectors, not publisher averages.
3. Run one validation/Trial at a time through the explicit harness, with Baseline in `autoresearch/core/config.py`; record canonical outcomes in `results.db`. No autonomous loop or context reduction.
4. Record prefill/decode separately where available, peak memory, failures, and per-task outcomes. Repeat apparent near-ties before claiming dominance. Change MTP, KV, or engine only in subsequent comparisons with one factor varied.

### Open questions

- **TBD:** exact GSQ-RCO artifact load and every tensor-type kernel on the pinned release; answer with harness validation before a full Trial.
- **TBD:** quality/throughput dominance at matched bytes and unchanged context on this rig; answer with sequential complete Trials. No GSQ-RCO local result was generated by this research pass.
- **TBD:** calibration distribution and exact generation protocol behind a particular publisher score; establish them before interpreting small deltas or transferring claims to agentic/coding-10.

## Sources / Verification

Primary sources checked **2026-10-12**. HF cards/file listings and the small allocation text were read with `hf`; GitHub READMEs with `gh api`; papers through arXiv. No weights downloaded or engine launched for this research.

- **S1:** [GSQ paper, v2 (2026-05-15)](https://arxiv.org/abs/2604.18556v2).
- **S2:** [RCO paper, v3 (2026-09-27)](https://arxiv.org/abs/2605.00649v3).
- **S3:** [Publisher GGUF recipe, packaging, and external comparison table](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF/blob/d562806dbafae37109975e970aae91b43e73b440/README.md).
- **S4:** [GSQ reference implementation](https://github.com/IST-DASLab/GSQ) and [RCO reference implementation](https://github.com/IST-DASLab/RCO).
- **S5:** [Publisher MoE release: search denominator, fixed lookup shard, external speed/quality tradeoff](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF).
- **S6:** [GPTQ paper, v2 (2023-03-22)](https://arxiv.org/abs/2210.17323v2).
- **S7:** [AWQ paper](https://arxiv.org/abs/2306.00978).
- **S8:** [Inspected per-tensor allocation, pinned revision](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF/blob/d562806dbafae37109975e970aae91b43e73b440/tensor-allocation/Qwen3.8-27B-GSQ-RCO-IQ2_XS.rco-allocation.txt).
- **S9:** [Upstream llama.cpp server options, separate K/V cache types](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md).

## Related Docs

- `docs/models/` — per-model GGUF specs and architecture notes
- `docs/discovery/discover-models.md` — end-to-end model selection workflow
- `docs/discovery/nvfp4-quantization.md` — NVIDIA NVFP4 (native Blackwell vs Marlin SM75+ fallback; not a GGUF quant)
- `docs/discovery/vllm-quantization.md` — vLLM quant matrix (AWQ/GPTQ/GGUF/NVFP4)
- `docs/discovery/inference-engines-landscape.md` — engine taxonomy + cross-engine comparison levels
