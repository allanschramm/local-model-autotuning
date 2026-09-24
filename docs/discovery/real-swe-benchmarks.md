# Real-SWE Benchmarks — integration options

> Why `coding-10` doesn't reflect daily SWE usability, and which benchmark(s) to swap or layer in. Pairs with the [2026-09-22 external findings](./2026-09-22-external-findings.md) and [mini-swe-agent-bridge](./mini-swe-agent-bridge.md).

## The problem (recap)

The operator's complaint, paraphrased: "the harness benchmark is lame, it didn't reflect a real usage for a SWE." The repository's current coding-10 axis (`autoresearch/benchmarks/coding/`) is **LiveCodeBench v6 + HumanEval+ + MBPP+ + BigCodeBench Hard — exactly 10 tasks per dataset**. These are:

- **Single-turn code completion**, not multi-step agentic loops
- **Stated problems with a known answer key**, not real GitHub issues with held-out tests
- **No tool-use, no file editing, no pytest loop**, no `git` history

This is correct for "is this model capable at one-shot coding?" and almost useless for "will this model survive a 2-hour Claude Code session on a real repo?" Two distinct questions, two distinct benchmarks.

The harness's `agentic_coding` SWE-lite axis (`autoresearch/benchmarks/agentic_coding/`) is closer — multi-step agentic loops over frozen GitHub-issue fixtures (ADR 0013) — but it's **not a Pareto axis in v1** per `CONTEXT.md:133`, so model rankings ignore it.

## What real SWE agentic evaluation looks like (the landscape)

| Benchmark | Format | Anti-contamination | Agent | Source | Why it's on this list |
|---|---|---|---|---|---|
| **SWE-bench Verified** | 500 real GitHub issues, held-out test pass/fail | partial (training-overlap concern, all models affected) | bash + custom tools | OpenAI 2024 | The de facto standard; mrguo6221 measured 90 % headline on Qwen3.6-27B-FP8 |
| **SWE-rebench V2** | new tasks, language-agnostic | **yes — by construction** (March 2026) | bash | arXiv 2602.23866 | Direct fix for SWE-bench Verified's training-overlap problem |
| **Terminal-Bench 2** | long-horizon terminal tasks | n/a (curated) | bash | tbench.ai | Captures multi-hour, multi-step agentic reasoning |
| **DM-Code-Agent 30-task hidden** | 30 real coding tasks, **tests hidden from the model** | yes (by design — sealed verifier) | local Python agent | hwfengcs | Smallest, cheapest "real SWE" — fast to run on 8 GB |
| **LiveCodeBench Pro** | contamination-free single-turn | yes (rolling window) | n/a | livecodebench.github.io | For the single-turn axis (keeps the existing capability surface) |
| **Aider polyglot** | multi-language edit benchmark | n/a (curated) | aider CLI | aider.chat | Edit-task quality (closer to "refactor this file" than "fix this issue") |

Source priority for ranking models on **daily SWE usability** (per `discover-models.md:48-57`): SWE-bench Verified → Aider polyglot → LiveCodeBench → Artificial Analysis Intelligence Index → Chatbot Arena ELO. **Add: SWE-rebench V2 → DM-Code-Agent 30-task → Terminal-Bench 2**, in that order of preference for the operator's specific complaint.

## Three integration paths (ordered by operator cost)

### Path A — Layer in DM-Code-Agent 30-task hidden (cheapest, fastest signal)

**Cost:** ~1 hour setup, 30 min/eval, fits on 8 GB. **Signal-to-noise:** high — sealed verifier rules out format-gaming, hidden tests rule out test-gaming.

```bash
gh repo clone hwfengcs/DM-Code-Agent C:\dev\dm-code-agent
cd C:\dev\dm-code-agent
pip install -e .   # their local-first agent + benchmark

# Wire to the operator's existing llama-server alias (same env vars as mini-swe-agent)
export DMCODE_LLM_BASE_URL="http://127.0.0.1:18080/v1"
export DMCODE_LLM_MODEL="qwen3.6-35b-a3b"
python scripts/run_bench.py --task-count 30
```

Output: a 30-task pass/fail report + per-task trajectory. Maps cleanly to the operator's existing `results.db` schema (one row per task with `task_id`, `pass`, `trajectory_tokens`, `wall_time`).

### Path B — Stand up SWE-rebench V2 locally (medium cost, anti-contamination leaderboard)

**Cost:** ~4 hours setup (Docker images + eval script), 6–12 hours/eval (500 tasks × Docker warm-up). **Signal:** high — the closest thing to SWE-bench Verified that doesn't have the training-overlap problem.

```bash
git clone https://github.com/SWE-rebench/SWE-rebench-V2
cd SWE-rebench-V2
pip install -r requirements.txt

# Build per-task Docker images (one-time, ~1 hour):
python scripts/build_base_images.py --dockerfiles-dir base_dockerfiles
python scripts/build_instance_images.py --json sample.json --template combine.Dockerfile.j2 --output-dir dockerfiles

# Run on the 20-task sample first (sanity):
python scripts/eval.py --hf-dataset ibragim-bad/SWE-rebench-V2-sample --hf-config default --hf-split train --max-workers 8 --golden-eval --report-json eval_report.json
```

To run the full set, swap `--hf-dataset` to the full `SWE-rebench/SWE-rebench-V2` (gated, may require HF auth). The eval driver is OpenAI-compatible endpoint friendly — point it at the operator's `llama-server` on 18080.

### Path C — Drop in `EvalGuard` around the existing benchmark (audit layer, no new eval)

**Cost:** ~30 min setup, runs alongside any SWE-bench-family eval. **Signal:** orthogonal — detects reward-hacking and contamination probes the harness wouldn't otherwise catch.

```bash
git clone https://github.com/FreakyAdy/EvalGuard
cd EvalGuard
pip install -r requirements.txt

# Wraps any SWE-bench / Terminal-Bench / OpenEnv run without modifying the harness:
evalguard wrap --benchmark swe-rebench-v2 --report audit_report.html
```

Useful as a guard rail: if a model suddenly jumps 5 pt on SWE-rebench-V2, EvalGuard's AST tamper detection flags whether the test files were touched (a common agent failure mode).

## Recommended action for the operator

**Path A first.** 30 minutes of operator time buys 30 SWE-style tasks evaluated against the operator's actual daily driver (`qwen3.6-35b-a3b`). The numbers from that run immediately settle the question "is the IQ-min Pareto sort actually picking models I want to ship?" — and provide a baseline to compare against when a new model Trial lands.

**Path B as the second move.** Once Path A has run on the current daily driver, the next time a new model candidate lands (e.g. `JigSawPT/DeepSeek-V4.1-Flash-DSpark-GGUF`), the eval cost is paid once for the new model and the SWE-rebench-V2 number is directly comparable to the literature.

**Path C always-on.** Wrap whichever benchmark wins, after Path A settles. Audit is cheap insurance against silent regressions.

## How this interacts with the existing Pareto Set

Per ADR 0017, the leaderboard (`pareto-leaderboard.md`) is every complete Objective Vector, IQ-first, ±0.05 near-tie band. Adding a new axis (Path A or B) is **a new Pareto axis**, not a replacement. The existing `agentic` axis (Claw-Eval full) stays; the new axis is added under a new name.

Proposed naming:
- `agentic` → `claw` (Claw-Eval full, current axis)
- `coding-10` → `coding` (single-turn, current axis)  
- New: `swe` (DM-Code-Agent 30-task or SWE-rebench-V2 — pick one before ranking)

Then `iq_min = min(claw, coding, swe)` and the ±0.05 band logic from ADR 0017 applies. This is a leaderboard contract change → requires ADR 0018.

## Open questions

- **Which benchmark wins, Path A or Path B?** DM-Code-Agent is faster and the hidden tests align with the operator's "real SWE" framing; SWE-rebench-V2 is the literature-grade anti-contamination standard. Both should be tracked; the leaderboard uses whichever the operator wants to gate on.
- **30-task hidden is small** — single bad run on a 3-task cluster could swing the score by 0.05 (the ADR 0017 near-tie band). Need a confidence interval in the leaderboard table.
- **Trajectory data ownership**: DM-Code-Agent, SWE-rebench-V2, and mini-swe-agent all export trajectories. Whether to write them to `results.db` (operator's canonical store) or a sidecar `swe_trajectories/` is a schema decision.
- **Existing SWE-lite (`autoresearch/benchmarks/agentic_coding/`)** uses frozen GitHub-issue fixtures per ADR 0013. Path A or B would replace or complement it; decide whether to retire SWE-lite or keep all three.

## Sources

- https://github.com/SWE-rebench/SWE-rebench-V2 (arXiv 2602.23866, 2026-03)
- https://github.com/hwfengcs/DM-Code-Agent (152 stars, 2026-08-27)
- https://github.com/FreakyAdy/EvalGuard (4 stars, 2026-09-17)
- https://github.com/SWE-agent/mini-swe-agent (7 885 stars, 2026-09-21)
- https://github.com/mrguo6221/swe-bench-88-90 (Qwen3.6-27B-FP8 → 90.0 % SWE-bench Verified)
- https://github.com/augmentcode/augment-swebench-agent (885 stars, "the #1 open-source SWE-bench Verified implementation")
- https://github.com/JARVIS-Xs/SE-Agent (289 stars, SOTA on SWE-bench Verified via self-evolution)
- https://docs/adr/0017-rank-membership-quality-first.md (rank-membership rules — any new axis lands here)
- https://docs/adr/0013-agentic-coding-night-selector.md (the existing SWE-lite Night selector)
- https://docs/discovery/discover-models.md (source priority for coding-quality decision-making)
- https://CONTEXT.md (Objective Vector definition)
