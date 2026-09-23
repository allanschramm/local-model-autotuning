# 2026-09-22 — External Findings (HF + GH sweep)

> Research note (not a technique doc): what we found outside the repo via `hf` + `gh` CLIs after the operator critique that "the benchmark is lame, it didn't reflect a real usage for a SWE." Pairs with [`mini-swe-agent-bridge.md`](./mini-swe-agent-bridge.md) and [`real-swe-benchmarks.md`](./real-swe-benchmarks.md).

## Context

Operator daily-use verdict: **Qwen3.8-4B (Day #1) and Ternary-Bonsai-2-27B (current Baseline) are garbage for daily SWE; Qwen3.6-35B-A3B-MTP is the actual daily driver; the harness's `coding-10` (LiveCodeBench + HumanEval+ + MBPP+ + BigCodeBench Hard) does not predict real Claude Code / Pi Agent / SWE-bench Verified performance.** Operator directive: "find outside this repo, the repo is only a harness for you to prove if the model is good or not, but my own taste will tell the true in the end."

This doc records what `hf models ls/info` and `gh search repos` returned on 2026-09-22, filtered for fit on the operator rig (discrete 8 GB-class NVIDIA, Ada CUDA, llama.cpp `upstream@b10867`, 32 GB-class host RAM).

## Method

Both CLIs were already on PATH:

| Tool | Version | Path |
|---|---|---|
| `hf` | huggingface_hub 1.21.0 (1.32.0 available) | `C:\Users\allan\.local\bin\hf.exe` |
| `gh` | 2.96.0 (2026-07-02) | `C:\Program Files\GitHub CLI\gh.exe` |

Searches issued:

```bash
# Hugging Face — discovery by tag, author, last_modified
hf models ls --search "qwen3" --sort last_modified --filter text-generation --num-parameters min:4B,max:35B --limit 15
hf models ls --search "deepseek flash" --sort last_modified --num-parameters min:3B,max:35B --limit 10
hf models ls --search "dflash" --sort last_modified --num-parameters min:4B,max:40B --limit 10
hf models ls --author deepseek-ai --sort last_modified --limit 10
hf models ls --author Qwen --sort last_modified --limit 8
hf models ls --author perplexity-ai --sort last_modified --limit 8

# Per-repo metadata
hf models info Qwen/Qwen3.8-Flash-Next
hf models info Qwen/Qwen3.8-Flash-Next-FP8
hf models info deepseek-ai/DeepSeek-V4.1-Flash
hf models info deepseek-ai/DeepSeek-V4-Flash-DSpark
hf models info kernelpool/DeepSeek-V4.1-Flash-MXFP4-GGUF
hf models info JigSawPT/DeepSeek-V4.1-Flash-DSpark-GGUF
hf models info Qwen/Qwen3.6-27B-FP8

# GitHub — search repos
gh search repos "swe-bench-verified" --sort stars --limit 15
gh search repos "swe-rebench" --sort stars --limit 15
gh search repos "agentic coding benchmark" --sort stars --limit 15
gh search repos "dflash speculative" --sort stars --limit 8
gh search repos "pi-coding-agent" --sort stars --limit 8

# Per-repo README
gh api repos/mrguo6221/swe-bench-88-90/readme
gh api repos/SWE-agent/mini-swe-agent/readme
gh api repos/SWE-rebench/SWE-rebench-V2/readme
gh api repos/esagduyu/qwen-pi-plan-vs-haiku-benchmark/readme
gh api repos/augmentcode/augment-swebench-agent/readme
```

## Headline findings

### 1. The harness gap is bigger than the model gap

[mrguo6221/swe-bench-88-90](https://github.com/mrguo6221/swe-bench-88-90) is the single most important reference for the operator's complaint. Same model, **Qwen3.6-27B-Thinking (FP8)**, evaluated three ways:

| Harness | Score | Δ vs baseline |
|---|---:|---:|
| `mini-swe-agent` baseline (`r14k_500`) | 67.8 % | — |
| `qwen-cli` + 47 K LoC custom proxy (ablation) | 87.4 % headline / 85.4 % strict | **+19.6 / +17.6 pt** |
| `claude-cli -p` + same proxy (main submission) | **90.0 %** headline / **88.0 %** strict | **+22.2 / +20.2 pt** |

**22.2 pt of SWE-bench Verified uplift from the same model, by changing the harness.** That is roughly the size of the gap between SOTA open models (79.2 % Sonar) and the absolute floor (e.g. SWE-agent-LM-32B at 37.3 %). The operator's own architecture — `llama-server` + OpenAI-compatible surface + custom tooling — can capture a meaningful slice of that uplift on 8 GB VRAM with **mini-swe-agent** (see [harness doc](./mini-swe-agent-bridge.md)).

### 2. `Qwen3.8-Flash-Next` is the Qwen4-architecture preview

Released 2026-08-27 / 2026-08-31, **`Qwen/Qwen3.8-Flash-Next-FP8` 529 k downloads / 5 591 likes**. Architecture:

- `model_type: qwen4_exp` (new Qwen4 experimental arch — not qwen35)
- `architectures: ["Qwen4ExpForConditionalGeneration"]`
- Multi-modal (`pipeline_tag: image-text-to-text`)
- 131-shard safetensors, total ≈ 420 GB BF16 / ≈ 210 GB FP8 → **too large for the operator rig as-is**
- New chat template reads `reasoning_effort` (xhigh/medium/low) and supports `<|vision_start|><|image_pad|><|vision_end|>` tokens
- 1 M-token context class per the chat template

The family has at least one **distilled small variant** that the operator rig should investigate:

- `IsValorum/Qwen3.8-35B-A3B-Distill-MTP-APEX-I-Mini` (6 046 downloads, 12 likes, 2026-09-22) — distill of the Qwen4 arch into a 35B MoE envelope, MTP packaged
- `IsValorum/Qwen3.6-35B-A3B-MTP-APEX-I-MiniPlus-V...` — Qwen3.6 lineage, MTP-packaged

The HF card read failed (`Error: Model not found`) on the exact basename in `hf models info` — the `ls` returned a truncated form. Verify the exact ID via `hf models ls IsValorum` before download.

### 3. `DeepSeek-V4.1-Flash` + `DSpark` is the most interesting new MoE for the rig

Released 2026-09-10 (`deepseek-ai/DeepSeek-V4.1-Flash`), the **DSpark packagings** add a built-in draft:

| Repo | Size | Format | Arch | Notes |
|---|---:|---|---|---|
| `kernelpool/DeepSeek-V4.1-Flash-MXFP4-GGUF` | 14 GB | MXFP4 GGUF (split into `.part1` + `.part2`) | `deepseek41-dspark` | 2026-09-17, 2 246 downloads |
| `JigSawPT/DeepSeek-V4.1-Flash-DSpark-GGUF` | 8.0 GB (`totalFileSize`) | GGUF, `library: llama.cpp` | `dflash` | 2026-09-14, 2 983 downloads, **1 048 576 ctx**, **MIT**, base_model `deepseek-ai/DeepSeek-V4.1-Flash`, tags `speculative-decoding`, `dspark` |

The 14 GB figure is misleading (it counts both `.part1` + `.part2`); the real single-file GGUF is 8 GB per JigSawPT. Active param count (`num_experts_per_tok`) needs verification on first download — if it lands in the 3B-active envelope like `Qwen3.6-35B-A3B`, the operator's existing `--n-cpu-moe 99` recipe applies directly.

**Caveat:** MXFP4 (block-fp4) support in llama.cpp upstream `b10867` needs verification before the kernelpool variant will load. The JigSawPT variant is the safer first download (standard GGUF).

### 4. `perplexity-ai/pplx-computer-qwen-3-8-27b-dflash2-gguf` is Perplexity's official DFlash2 drafter

6 368 downloads (highest DFlash-related download count in the search), released 2026-08-26. Pairs with the `Qwen/Qwen3.8-27B` target the operator already has a card for (currently `qwen3.8-27b.md:46` lists agentic + coding as `TBD`). DFlash2 (successor to the `draft-dflash` path mentioned in the operator's `qwen3.8-27b.md` card) is the modern alternative to MTP-n2 for dense 27B on this rig.

### 5. `mini-swe-agent` is the daily-driver harness

[SWE-agent/mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) — **7 885 stars** (2026-09-21 last push):

- **100 lines of Python** for the agent class; ~500 LoC total including env / model / run-script
- **>74 % on SWE-bench Verified** (bash-only tools — no tool-calling interface required)
- Beats Claude Code and Codex on DeepSWE (`deepswe.datacurve.ai`)
- Powers Ramp SWE-Bench in production
- Used by Meta, NVIDIA, Essential AI, IBM, Nebius, Anyscale, Princeton, Stanford
- **Speaks litellm → any OpenAI-compatible endpoint**. The operator's existing `llama-server` on port 18080 (`qwen3.6-35b-a3b-mtp` alias) is one env var away.
- Linear history (just appends to messages) → trivial trajectory inspection, trivial FT/RL data export
- `subprocess.run` per action → trivial swap to docker / podman / singularity sandboxing

Bridge setup is documented in [`mini-swe-agent-bridge.md`](./mini-swe-agent-bridge.md).

### 6. The `plan-first` skill from `esagduyu/qwen-pi-plan-vs-haiku-benchmark`

Same Qwen3.6-35B-A3B the operator runs daily, same `llama.cpp` build, on a single RTX 2080 Ti (11 GB). Reproducible spec (Python CLI todo manager). Outcome:

| Run | Wall time | Tests | Spec adherence |
|---|---:|---|---|
| Baseline Qwen3 (no plan) | ~9 min | 5/5 | 6/10 |
| Qwen3 + `plan-first` skill | **3.3 min** | **6/6** | **9/10** |
| Claude Haiku 4.5 (cloud) | 86 s | 18/18 | 10/10 |

**The skill is a markdown file** (`skills/plan-first.md` in that repo) that asks the model to plan into `TODO.md` first, then execute. Drop-in for any OpenAI-compatible surface — read it, copy it into `models/aliases/qwen3.6-35b-a3b-mtp/skills/` (or anywhere the harness reads it).

### 7. Real-SWE benchmarks: SWE-rebench-V2 + DM-Code-Agent 30-task

Two benchmarks the operator's complaint about "the leaderboard doesn't reflect real SWE" maps to:

- **[SWE-rebench/SWE-rebench-V2](https://github.com/SWE-rebench/SWE-rebench-V2)** — arXiv 2602.23866, **language-agnostic**, anti-contamination by construction (March 2026). Sample dataset on HF (`ibragim-bad/SWE-rebench-V2-sample`).
- **[hwfengcs/DM-Code-Agent](https://github.com/hwfengcs/DM-Code-Agent)** — 152 stars, local-first Python code agent + **30-task hidden-test benchmark** + SWE-bench Verified harness. The hidden-test design is exactly the "real SWE" the operator wants — the model cannot game the test by reading it.
- **[FreakyAdy/EvalGuard](https://github.com/FreakyAdy/EvalGuard)** — wraps SWE-bench / Terminal-Bench / OpenEnv **without modifying the harness**; adds sandbox-boundary audit, reward-hack AST detection, and contamination probes. Pairs with whichever benchmark wins.

Setup recipe is documented in [`real-swe-benchmarks.md`](./real-swe-benchmarks.md).

### 8. Other repos of record

- **[augmentcode/augment-swebench-agent](https://github.com/augmentcode/augment-swebench-agent)** — 885 stars, calls itself "#1 open-source SWE-bench Verified implementation" (65.4 % with Sonnet 3.7 + o1 ensembler). Anthropic-derived system prompt + majority-vote ensembler.
- **[JARVIS-Xs/SE-Agent](https://github.com/JARVIS-Xs/SE-Agent)** — 289 stars, "SOTA performance" on SWE-bench Verified via trajectory-level self-evolution (Revision / Recombination / Refinement).
- **[china-qijizhifeng/agentic-harness-engineering](https://github.com/china-qijizhifeng/agentic-harness-engineering)** — 901 stars, NexAU-AHE lifts GPT-5.4 69.7→77.0 % on Terminal-Bench 2 over 10 iters, beats Codex/ACE/Training-Free GRPO; the frozen harness transfers to SWE-bench-Verified.
- **[AutoCodeRoverSG/auto-code-rover](https://github.com/AutoCodeRoverSG/auto-code-rover)** — 3 100 stars, project-structure-aware autonomous SE agent; 37.3 % on SWE-bench lite, 46.2 % on SWE-bench Verified at <$0.7/task.
- **[EvalGuard](https://github.com/FreakyAdy/EvalGuard)** — see §7.

## Fit assessment (per rig)

| Candidate | On disk? | File size | RAM | Active path | Verdict |
|---|:---:|---:|---|---|---|
| `IsValorum/Qwen3.8-35B-A3B-Distill-MTP-APEX-I-Mini` | no | ~21 GB (UD-Q4_K_M est) | 32 GB OK | MoE 3B active → fits 8 GB with `--n-cpu-moe` | **Strong fit** — first MoE Trial |
| `JigSawPT/DeepSeek-V4.1-Flash-DSpark-GGUF` | no | 8 GB | fits native | TBD (`num_experts_per_tok`) | **Strong fit** — 1 M ctx + DSpark if active path small |
| `kernelpool/DeepSeek-V4.1-Flash-MXFP4-GGUF` | no | 14 GB split | fits native | TBD | **Caveat:** MXFP4 in b10867 needs verification |
| `perplexity-ai/pplx-computer-qwen-3-8-27b-dflash2-gguf` | no | drafter (small) + 27B target | target ≈ 5–10 GB GGUF | full dense 27B → **doesn't fit 8 GB alone** | **Conditional** — needs partial layer offload or IQ1_S target |
| `Qwen/Qwen3.8-Flash-Next-FP8` | no | ≈ 210 GB | impossible | — | **Out of scope** for 8 GB rig |
| `Qwen/Qwen3.6-27B-FP8` (no GGUF; FP8 only) | no | ≈ 30 GB FP8 / ≈ 14 GB GGUF est | fits RAM | dense 27B → **doesn't fit 8 GB alone** | **Conditional** — needs quant + layer tuning |

## What this changes for the operator

1. **Stop treating `coding-10` as the SWE benchmark.** It's single-turn code completion. For real SWE agentic usability, the harness is at least as important as the model (see §1).
2. **The next-step priority is harness, not model.** `mini-swe-agent` + `plan-first` skill against the existing `qwen3.6-35b-a3b-mtp` alias is the cheapest, highest-confidence improvement.
3. **One MoE Trial is worth the disk cost.** `JigSawPT/DeepSeek-V4.1-Flash-DSpark-GGUF` is the best fit: 8 GB GGUF, 1 M ctx, DSpark, MIT-licensed. If the active path lands ≤ 3 B params, it slots into the existing `--n-cpu-moe 99` recipe and gives the operator a longer-context + faster daily driver than the Qwen3.6-35B-A3B baseline.

## Open questions

- **Qwen3.8-Flash-Next distilled variants**: confirm exact basenames via `hf models ls IsValorum` — `hf models info` failed on the truncated name.
- **DeepSeek-V4.1-Flash active params**: `num_experts_per_tok` not in the GGUF metadata API response we captured — must verify after first download via `gguf.GGUFReader` or `scripts/model_info.py`.
- **MXFP4 support in llama.cpp b10867**: must probe with `--help` on the binary before downloading the kernelpool split variant.
- **DFlash2 vs MTP-n2 head-to-head on Qwen3.8-27B**: no published A/B on Ada CUDA 8 GB. The card already lists MTP as `draft-mtp`; DFlash2 is a separate path.
- **`agent-psychometrics` (Dariakryvosheieva, 5 stars)**: paper claims task-level performance prediction in agentic benchmarks — would let us skip running full SWE for ranking. Worth a single read-through but not a Trial.

## Sources

All URLs fetched **2026-09-22** via `hf` + `gh` CLIs. Date in parentheses is upstream release.

- https://huggingface.co/Qwen/Qwen3.8-Flash-Next (2026-08-27)
- https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8 (2026-08-31)
- https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash (2026-09-10)
- https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-DSpark (2026-07-04)
- https://huggingface.co/kernelpool/DeepSeek-V4.1-Flash-MXFP4-GGUF (2026-09-17)
- https://huggingface.co/JigSawPT/DeepSeek-V4.1-Flash-DSpark-GGUF (2026-09-14)
- https://huggingface.co/Qwen/Qwen3.6-27B-FP8 (2026-08-14)
- https://huggingface.co/perplexity-ai/pplx-computer-qwen-3-8-27b-dflash2-gguf (2026-08-26)
- https://github.com/mrguo6221/swe-bench-88-90 (last push 2026-05-12)
- https://github.com/SWE-agent/mini-swe-agent (last push 2026-09-21)
- https://github.com/SWE-rebench/SWE-rebench-V2 (last push 2026-03-12)
- https://github.com/esagduyu/qwen-pi-plan-vs-haiku-benchmark (last push 2026-05-01)
- https://github.com/augmentcode/augment-swebench-agent (last push 2026-08-21)
- https://github.com/FreakyAdy/EvalGuard (last push 2026-09-17)
- https://github.com/hwfengcs/DM-Code-Agent (last push 2026-08-27)
- https://github.com/JARVIS-Xs/SE-Agent (last push 2025-09-23)
- https://github.com/china-qijizhifeng/agentic-harness-engineering (last push 2026-08-03)
- https://github.com/AutoCodeRoverSG/auto-code-rover (last push 2025-04-24)
- https://arxiv.org/abs/2602.23866 (SWE-rebench V2 paper, 2026-03)
