# Capability Extraction Harness — Raising IQ per Token Without Engine Changes

## Purpose

Engine TPS has a hard ceiling per model+quant Fingerprint. The remaining local lever is **how many useful tokens the harness spends per completed task** — the ratio between a model's measured skill and its realized task completion is called *capability-realization loss* in the 2026 literature. This guide catalogs harness-side capability-extraction techniques (led state, verification loops, test-time compute) and maps each to a falsifiable experiment on this repo's agentic / coding axes.

Reference: [thinking-models-claw-harness.md](./thinking-models-claw-harness.md) — the in-repo precedent: a harness bug (ignored `reasoning_content`, `max_tokens=512`) produced false "dumb model" floors (Ornith 0.333 → 0.933). That fix was pure IQ recovery at identical engine TPS. This guide generalizes that lesson.

## 1. Core insight

- Wall-clock per task = `tokens_spent / TPS`. At fixed engine TPS, raising agentic/coding = raising **IQ per token**.
- Capability-realization loss sources: interface mismatch (persona/schema drift between training and harness), goal-fade on long loops, blank retries without failure diagnosis, runaway reasoning chains, under-budgeted `max_tokens`.
- **score/token** (and score/time) is the metric that separates "harness extracted IQ" from "engine got faster". It is observational — derived from `results.tsv` + mock logs, **not** a Pareto axis. Pareto axes stay ctx × TPS × agentic × coding (ADR 0006).

## 2. Inference-engine / ctx survey (August 2026) — what's real

- **Pinned engine is usable, but no longer the latest release (re-checked 2026-08-22).** Prebuilt `llama.cpp-releases/upstream/b10375` (build 2026-08-13) postdates the Gemma-4 KV-cache fix (Apr 2026), upstream MTP merge (May 2026), and MMVQ rewrites. **Latest 2026-08-22:** semver **v0.2.0** (nightly **b10566**, tagged **b10549** 2026-08-21T09:23Z) — **174 commits ahead** of b10375, win CUDA-12.4/13.3 assets ([releases/tag/b10549](https://github.com/ggml-org/llama.cpp/releases/tag/b10549), [compare b10375...b10549](https://github.com/ggml-org/llama.cpp/compare/b10375...b10549)). Verdict remains **optional bump**: since the 2026-08-19 check (b10488), new items include (a) **general CUDA decode** — MVQ→MMQ crossover switch points per HW + quant ([#26079](https://github.com/ggml-org/llama.cpp/pull/26079)) and CUDA-graphs kept for more `mul_mat_id` shapes ([#26802](https://github.com/ggml-org/llama.cpp/pull/26802)) — first general decode-path change for this class since the prior "RWKV-7-only" verdict; (b) spec auto-detect from draft GGUF ([#26814](https://github.com/ggml-org/llama.cpp/pull/26814)/[#27005](https://github.com/ggml-org/llama.cpp/pull/27005)) and DSpark-for-LFM2 ([#27383](https://github.com/ggml-org/llama.cpp/pull/27383)); (c) correctness — recurrent-state rollback for `ggml_ssm_scan` ([#26623](https://github.com/ggml-org/llama.cpp/pull/26623)); (d) flag migration — `--mmap`/`--no-mmap` → `--load-mode` ([#26934](https://github.com/ggml-org/llama.cpp/pull/26934)) and `--models-dir` MTP-assistant discovery ([#24431](https://github.com/ggml-org/llama.cpp/pull/24431)). None are required for current Pareto fingerprints; **the (d) flag migration has since landed** — the bump to `b11149` broke the harness (`--mmap` rejected, exit 1) until `NO_MMAP`/`MLOCK` were routed through `load_mode_flag` -> `--load-mode` ([`llamacpp-flags-audit.md`](../llamacpp-flags-audit.md)); `assert_flags_supported` now fails closed on the next rename. Falsifiable: same Fingerprint bench_tg on b10375 vs latest.
- **N-gram speculation on coding text.** Five draftless variants (`ngram-cache`, `ngram-simple`, `ngram-map-k`, `ngram-map-k4v`, `ngram-mod`) are universal (any model), 0 MB VRAM (~16 MB host RAM for `ngram-mod`), and composable with a neural drafter (`--spec-type draft-mtp,ngram-mod`). Mechanism + flags: [speculative-decoding-formats.md](./speculative-decoding-formats.md) §2/§3D. **External measured prior on this repo's own A3B family is negative** — RTX 3090 matrix (2026-04, N=3-reproducible, cross-checked on A100): `ngram-cache` −12%, `ngram-mod` −3–5%, 0.8B draft −39–60% despite 100% acceptance (MoESD expert-union verify cost, [speculative-decoding-formats.md](./speculative-decoding-formats.md) §4b). The 2026-07-20 small-model TPS matrix has **no ngram row** — candidate Search neighbor on coding-10; measured 2026-09-07 (§7) as net-negative on fast dense GPU models due to low acceptance and verification overhead.
- **`--model-draft` external drafts** merged upstream May 2026: equivalent to MTP for dense targets; known-fail on MoE + CPU-expert offload (see [speculative-decoding-formats.md](./speculative-decoding-formats.md) §4b). Skip.
- **DFlash KV ideas are 3× this rig's VRAM scale.** The DFlash 256K-ctx-on-24-GB-class demo (TQ3_0 3.5 bpv KV, 4096-slot target-feature ring buffer, 2048-token sliding-window flash attention) is a concept reference only — no transfer to a discrete 8 GB-class rig.

## 3. dflash / dspark — vendor claims vs in-repo measurement

| Technique | Vendor claim (2026) | In-repo measurement (8 GB-class) | Verdict |
|---|---|---|---|
| DFlash (block-diffusion drafter, NVIDIA/DeepSeek DeepSpec) | 3–6× lossless; up to 15× Blackwell | 12.5 vs 24.6 t/s (−49%) on Qwen3.6-35B-A3B with `n-cpu-moe`; needs 35B target fully on GPU | Dead end on MoE + CPU offload |
| DSpark (semi-autoregressive drafter + confidence head + hardware-aware scheduler, DeepSeek) | 60–85% faster per-user, up to 6.6× throughput | 19.2 vs 37.3 t/s (−48.5%) on Bonsai-27B Q1_0 (quantization-speed inversion: Q4 draft slower than Q1 target) | Dead end under extreme low-bit targets |

**Durable rule:** neural speculation wins only when target **and** draft fit on GPU and the draft is faster per token than the target. On this rig's MoE/CPU-offload fingerprints it is a proven loss.

**DSpark reachability (updated 2026-08-19):** upstream llama.cpp merged native `draft-dspark` on 2026-07-28 ([#25173](https://github.com/ggml-org/llama.cpp/pull/25173)) — inside the pinned b10375 — for **Qwen3-backbone dense-format** drafts (`deepseek-ai/dspark_qwen3_{4b,8b,14b}_block7`). `speculators`-format (SpecForge/RedHat) DSpark checkpoints merged 2026-08-17 ([#26275](https://github.com/ggml-org/llama.cpp/pull/26275)) — **post-b10375** — and its own verification loads **Gemma-4-backbone** drafts in llama.cpp (`makora-ai/gemma4-26b-a4b-dspark`, `RedHatAI/gemma-4-31B-it-speculator.dspark`, `RadixArk/Qwen3.8-27B-DSpark`): the "Gemma-4 drafts are SGLang-only" claim is now stale at master. The official dense-format Gemma-4 drafts (`deepseek-ai/dspark_gemma4_12b_block7`; community GGUF `ankk98/...`) have no verified llama.cpp path in the sources gathered — status **TBD** pending a conversion check. On the pinned b10375, Gemma-4-backbone drafts remain unreachable regardless (needs the bump). Any path still requires a dense target fully on GPU to beat the measured dead ends. **TBD:** only if that path is ever exercised.

## 4. J-Space report (Tiger3807861189) — assessment

Facts: report-only repo (README + LICENSE + CITATION, no code — the plugin lives in a separate repo); CC BY-ND license (do not copy text into tracked docs); **single-run results, no confidence intervals** (the report itself states single runs do not represent a stable distribution); target models are API-only (DeepSeek V4-Flash 284B / V4-Pro 1.6T — nothing local-runnable); 354 stars / 19 forks on a 3-file repo.

Verdict: **technique reference, not a dependency.** The diagnosis is sound; the numbers are unverified marketing-grade evidence.

Transferable mechanisms — each falsifiable via a harness flag switched on/off on one Fingerprint:

- **Led state**: persistent `Goal / Core / Verified / Open / Next` block the model maintains across tool turns → attacks goal-fade on long (65k) loops.
- **Checkpoint with verifier + coverage**: resume from the last *verified* state, not the last action; wire hidden-test results into retry decisions. This repo's `agentic_coding` pass already has hidden tests — the missing piece is feeding them back into continuation.
- **Fail-diagnosis retry**: retry with the failure reason in context, never a blank retry.
- **fast/full/loop gating**: spend think budget proportional to task depth — maps to the Day (fast) vs Night (deep) profile split.

Caveats (from the report itself): first-person word frequency (`we`/`let me`) is a behavioral probe, not a quality mechanism; gains must be reproduced on the same model + harness (switch on/off) — exactly this repo's Trial protocol.

## 5. Test-time compute catalog (harness-side, evidence-backed)

- **Self-consistency / best-of-N + self-verification**: sample N diverse continuations for grader-critical turns, verify, keep best. No second model, no extra VRAM — small models verify their own outputs. Cost: N× tokens at same TPS → trades wall-clock for IQ. Strongest evidence base in the catalog.
- **Small verifiers**: verification is structured reasoning, not scale; small efficient models rank outputs adequately. Enables verifier-in-the-loop without a second on-GPU model.
- **SELFCOMPACT adaptive compaction**: the model decides when to summarize its own context → lower KV pressure for the same effective working memory. The only item in this catalog that touches **both** the IQ and the CTX axis.
- **Constrained/structured decoding for tool calls** (XGrammar on the SGLang path; llama.cpp grammars): malformed tool JSON is simultaneously a TPS waste and an IQ waste — one broken tool call costs a full retry round-trip.
- **MCTS / LATS**: real but token-hungry; only worth it on `agentic_coding`, and only after the cheaper items above are exhausted.

## 6. What NOT to chase (dead ends with evidence)

- **Engine bump**: pinned release already current (see §2).
- **dflash / dspark re-runs** on 8 GB-class MoE: measured losses (§3); vendor claims are datacenter/Blackwell conditions.
- **J-Space code integration**: CC BY-ND, API-only models, no code in the report repo (§4).
- **Consumer GUI engines** (Atomic Chat, LM Studio etc.): no harness path, no measured edge over llama.cpp CUDA on this rig — see [fastest-tps-inference-engine.md](./fastest-tps-inference-engine.md).
- **Colibrì / TRT-LLM / OpenVINO / vLLM / LMDeploy**: already documented and classified in [inference-engines-landscape.md](./inference-engines-landscape.md).

## 7. Candidate experiments (falsifiable, one Fingerprint each)

1. **ngram-cache as Search neighbor** — engine-only, 0 MB VRAM, universal. Measured 2026-09-07 (Trial 0593e117, Issue #57; [session log](../sessions/2026-09-07-issue-57-ngram-cache-trial.md)) on Qwen3.8-4B-Distill Q4_K_M @ 131k: TPS -23.5% (72.1 vs 94.2), coding 0.5900 vs 0.6400, continuous host RAM growth. Strictly **dominated** under Pareto rules vs `--spec-type none` at same Fingerprint; search keeps `--spec-type none`.
2. **Led-state + fail-diagnosis retry harness flag** — `agentic_coding` switch on/off on one Fingerprint. Harness change → operator decision (ADR if adopted).
3. **Best-of-N self-consistency** on grader-critical turns — measure agentic/coding gain per token budget.
4. **score/token observational metric** in Night loops — derived from existing results + mock logs; not a Pareto axis.
5. **SELFCOMPACT adaptive compaction** in the agentic runner — operator decision; only item affecting the CTX axis.

Protocol rule: harness is fixed unless the operator approves a change; every candidate must be measured same Fingerprint, same tasks, switch on/off.

## 8. Open questions

- **Resolved (2026-09-07):** ngram gain on coding-10 — measured (Issue #57, Trial 0593e117; [session log](../sessions/2026-09-07-issue-57-ngram-cache-trial.md)); throughput regresses on fast dense targets (-23.5% TPS) due to low draft acceptance (1.7-9.5%) and verification pass overhead; candidate is Pareto-dominated.
- **TBD:** DSpark on discrete GPU architectures — llama.cpp-native path exists in pinned b10375 for Qwen3-backbone dense-format drafts; Gemma-4-backbone speculators-format drafts need a bump past b10375 ([#26275](https://github.com/ggml-org/llama.cpp/pull/26275)); official dense-format Gemma-4 drafts unverified in llama.cpp. All require a dense on-GPU target to beat the measured dead ends (quantization-speed inversion).
- **TBD:** best-of-N token cost vs agentic gain — harness-side (parked until a harness fork); no local measurement yet.
- **Engine bump (re-checked 2026-08-22):** latest is **v0.2.0 / b10549** (2026-08-21T09:23Z, 174 commits ahead of b10375, [releases/tag/b10549](https://github.com/ggml-org/llama.cpp/releases/tag/b10549)); new since b10488: **general CUDA decode** MVQ→MMQ per-HW/quant crossover ([#26079](https://github.com/ggml-org/llama.cpp/pull/26079)) + CUDA-graph retention for `mul_mat_id` ([#26802](https://github.com/ggml-org/llama.cpp/pull/26802)) — first general decode-path change for this class; plus recurrent-state rollback for spec on SSM hybrids ([#26623](https://github.com/ggml-org/llama.cpp/pull/26623)), `--mmap`→`--load-mode` migration ([#26934](https://github.com/ggml-org/llama.cpp/pull/26934)), spec auto-detect ([#26814](https://github.com/ggml-org/llama.cpp/pull/26814)/[#27005](https://github.com/ggml-org/llama.cpp/pull/27005)), DSpark-for-LFM2 ([#27383](https://github.com/ggml-org/llama.cpp/pull/27383)). Verdict stays **optional**: falsifiable bench_tg on same Fingerprint b10375 vs latest before adopting a bump. A bump that lands a flag rename additionally requires the `load_mode_flag` passthrough and a green `assert_flags_supported` (already done for `b11149`).

## Sources

- [nvidia.com — DFlash up to 15× on Blackwell](https://developer.nvidia.com/blog/boost-inference-performance-up-to-15x-on-nvidia-blackwell-using-dflash-speculative-decoding/)
- [deepseek.ai — Inside DeepSeek DSpark (lossless inference)](https://deepseek.ai/blog/inside-deepseek-dspark-lossless-inference)
- [baseten.co — DFlash faster LLM inference](https://www.baseten.co/blog/dflash-faster-llm-inference/)
- [note.com — DFlash 256K ctx on 24 GB, TQ3_0 KV, ring buffer](https://note.com/samehadaonsen/n/nde52cc429dfd)
- [dev.to — why MTP doesn't speed up and how to fix it (SWA caveat, n-gram)](https://dev.to/alanwest/why-mtp-doesnt-speed-up-your-llamacpp-inference-and-how-to-actually-fix-it-2m2m)
- [n1n.ai — Gemma-4 local inference KV-cache fix (Apr 2026)](https://explore.n1n.ai/blog/gemma-4-local-inference-llama-cpp-kv-cache-fix-npu-benchmarks-2026-04-05)
- [arxiv 2606.23525 — adaptive context compaction (SELFCOMPACT)](https://arxiv.org/pdf/2606.23525)
- [arxiv 2501.14304 — LATS (Language Agent Tree Search)](https://arxiv.org/html/2501.14304v1)
- [arxiv 2607.05391 — small models as verifiers](https://arxiv.org/html/2607.05391v1)
- [Tiger3807861189/DeepSeek-V4-J-Space-Capability-Realization-Report](https://github.com/Tiger3807861189/DeepSeek-V4-J-Space-Capability-Realization-Report)
- [xiaobright/dsh-anchored-standard](https://github.com/xiaobright/dsh-anchored-standard)
- [yjh051108/dsh-routing-suite](https://github.com/yjh051108/dsh-routing-suite)
