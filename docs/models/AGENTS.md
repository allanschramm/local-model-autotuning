# Model runtime cards

## Purpose
Model architecture, GGUF identity, runtime configuration, memory fit and measured performance. Quality scores and model-quality rankings belong to the preserved benchmark history.

## Ownership
Repository developers; parent docs/AGENTS.md. Each card is a leaf.

## Local Contracts
- One card per model; basename/quant identity is explicit. No private paths, alias inventory, GPU SKU or machine identifiers.
- Classify dense/MoE, layer counts and MTP from GGUF metadata through scripts/model_info.py, never filename heuristics.
- Keep publisher sampler recommendations as frozen configuration inputs; this is not sampler optimization or an endorsement of model quality.
- Retain architecture, memory fit, KV/offload, batch/thread, quantization and throughput findings with original dates/protocols. Historical CLI smoke TPS is not proof of full-context tuning-runtime-v1 performance.
- Quality tables, task success rates and Day/Night ranks are archived. Mixed cards link to their complete original; no numeric runtime value is promoted from estimate to measurement.
- Mark unknowns as dated TBDs. Preserve source URLs and verification dates; do not invent values.

## Work Guidance
- Update only affected sections after verified runtime changes. Compare the exact GGUF artifact and protocol before reusing a recorded configuration.
- Preserve the target context; use KV/offload/batch or lighter quants for resource pressure.

## Verification
- Architecture claims follow metadata evidence; numeric performance keeps its units and protocol; source and internal links resolve.

## Child DOX Index
- [README.md](README.md) — card index.
