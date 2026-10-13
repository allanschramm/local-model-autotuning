# Quant Selection ? Agent Quick Reference

Read the source guide: [quantization-cascade.md](./quantization-cascade.md), including Sources / Verification.

## Rules

1. Compare the same base checkpoint at the same configured context and evaluation scope.
2. Inspect actual bytes, tensor types, and bpw denominator. Quant labels and publisher prefixes do not prove fit or quality.
3. Include weights, KV, workspace, optional heads/projectors, and host RAM. Dense requires physical VRAM fit; MoE expert offload follows the harness gates.
4. Keep flash attention on. Never reduce configured context to resolve memory pressure.
5. Use the explicit harness sequentially; canonical outcomes belong in results.db.

## GSQ / RCO

- GSQ optimizes scalar assignments and group scales; RCO allocates compression options among tensors under a budget.
- These are quantization recipes. GGUF is the container; IQ/K types are stored tensor representations.
- A nominal IQ2 artifact can contain IQ1?IQ4, K-quants, and BF16/F32. Compare parameter-weighted precision and actual required bytes.
- Publisher results and stock-runtime compatibility claims remain unverified locally until validation and complete Trials.
- MTP, QAT provenance, and KV precision are separate factors. Hold them fixed in the first comparison.

## Decision

Inspect ? apply hardware gates at fixed context ? validate ? complete Trial ? compare measured quality and throughput.

UD, standard, and GSQ-RCO have no universal ranking. Use matched budgets and repeat near-ties before claiming superiority.
