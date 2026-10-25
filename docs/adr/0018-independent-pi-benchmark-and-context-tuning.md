# ADR 0018: Independent Pi Benchmark and TPS × Context Auto tuning

**Status:** Accepted. The stages below are the decision snapshot; current bounded tuning protocol and verification are documented in [TUNING.md](../../autoresearch/TUNING.md).
**Date:** 2026-10-25
**Supersedes in part:** [0014](0014-fingerprint-bus-product-split.md) for the Benchmark's dependency on a tuned Fingerprint and Pi's role as only an informal quality check; [0004](0004-agentic-first-search.md) for local-only model connections in the new Benchmark. Historical evaluation protocols remain unchanged.

## Context & Problem Statement

Search, local runtime, benchmark selection, mutable Baseline, and persistence are coupled through `autoresearch.runners`. The operator wants two independently usable modules, initially in one repository and extractable into separate repositories later. A Benchmark must also evaluate models never tuned here, including cloud models.

## Decision

1. **Auto tuning** owns local runtime optimization, Baseline, Search Space, Neighbors, hardware protection, and Fingerprint publication. Optimize TPS × configured context as a Pareto Set, subject to the intended harness's minimum context and the PPL quality ceiling. Preserve multiple non-dominated alternatives, not one scalar winner. Never reduce the operator's target context to resolve resource pressure.
2. **Benchmark** owns tasks, verification, measurement protocol, trajectories, and quality results. Its official agentic harness is version-pinned Pi. A model endpoint/provider is sufficient; Search, Baseline, GGUF metadata and prior tuning are not prerequisites. Native Pi providers and compatible custom endpoints use Pi's adapters.
3. **Runtime stays with Auto tuning.** Benchmark does not start, tune, or stop model servers. A local launcher may serve a Fingerprint and expose the same kind of endpoint as a cloud provider. No second implementation of GPU/RAM/process safeguards.
4. **Two modules in-tree first.** `benchmark/` must run when copied outside the checkout with only its declared dependencies. Keep existing tuning/legacy paths operational during extraction; physical separation into two remotes is a later packaging step, not required to prove independence.
5. **Protocol identity is mandatory.** Pi version, tools, prompts, task/verifier hashes, output budget, reasoning and recovery policies identify results. New Pi results never overwrite or merge into legacy Claw, Coding, SWE-lite or model rankings. In this checkout, new tables coexist in canonical SQLite `results.db`.
6. **Port tasks before adding datasets.** The five frozen SWE-lite tasks move intact to `benchmark/tasks`; the legacy runner reads those same files. Initial Pi protocol `pi-workspace-v1` exposes allowlisted read/write tools, 4096 output tokens per turn, task-specific turn limits, thinking off, no compaction or automatic recovery. Hidden pytest verification runs after the agent, outside its accessible workspace. This is a constrained workspace protocol, not vanilla Pi or Terminal-Bench.

## Implementation Stages

- **First slice:** standalone `benchmark/`, Pi SDK integration, frozen workspace pack, isolated resources, SQLite checkpoints and E2E evidence using a simulated endpoint. This proves wiring and independence, not real-model quality.
- **Next:** separate tuning orchestration/persistence from the legacy benchmark CLI. Establish target-context throughput measurement, minimum-context admission and TPS × context Search without IQ-pack dependencies. Keep hardware gates fail-closed.
- **Then:** migrate additional workspace-shaped packs to explicit Pi protocols; add datasets only after existing packs are reproducible. Extract repositories after module isolation is demonstrated.

The existing TPS smoke caps allocation at 4096 and PPL uses 2048. TPS/PPL-only returns before a monitored full-context server. Those measurements must not be advertised as proof of target-context throughput or long-context capability. Context allocation and actual prompt occupancy are distinct measurements. The existing Round-relative PPL guard also needs a fixed-reference policy before claiming a non-accumulating quality ceiling.

## Consequences

- **Positive:** Benchmark can compare local/cloud models without importing tuning, changing Baseline, or carrying a local runtime. The same task bytes survive migration.
- **Negative:** Pi adds a pinned Node dependency; existing Python-only installations do not automatically include it. SDK upgrades or tool changes require explicit protocol review.
- **Neutral:** Legacy entrypoints, store tables, numeric protocols, and leaderboards remain available during extraction. `teach/` and vendor trees are unchanged.

## References

- [Pi SDK](https://github.com/earendil-works/pi/blob/v1.4.0/packages/coding-agent/docs/sdk.md)
- [Pi model connections](https://github.com/earendil-works/pi/blob/v1.4.0/packages/coding-agent/docs/models.md)
