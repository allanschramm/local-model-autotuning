# Discover Models for Your Hardware

End-to-end workflow: **find models that fit your rig, filter for coding quality, run the autoloop on the Pareto-optimal pick.**

## Step 1 — Find candidates with `whichllm` or `llmfit`

```bash
# Option A: whichllm (Python / uvx)
uvx whichllm@latest

# Option B: llmfit (Rust CLI/TUI)
llmfit
# or plan a specific model:
llmfit plan "qwen 3.5 9b"
```

Auto-detects GPU/CPU/RAM. Outputs a ranked list and memory footprint breakdown. See [`whichllm-reference.md`](./whichllm-reference.md) and [`llmfit-reference.md`](./llmfit-reference.md) for full CLI docs.

Key flags for this workflow:

| Flag / Command | Tool | Use |
|---|---|---|
| `--gpu-only` / `--fit gpu` | `whichllm` | Only models that fit FULL GPU (faster, no offload penalty) |
| `--speed usable` | `whichllm` | Hide models too slow to be practical |
| `--gpu "RTX 4090"` | `whichllm` | Simulate different hardware before buying |
| `--profile coding` | `whichllm` | Rank by coding-agent quality |
| `whichllm plan "qwen 3.6 35b"` | `whichllm` | VRAM/quant options + estimated tok/s for one model |
| `llmfit plan "qwen 3.6 35b"` | `llmfit` | Interactive memory fit & footprint planning per quant |

**Caveat**: discovery tool "scores" (e.g. intelligence index) are broad quality blends. They are **NOT** coding-agent benchmarks. Gemma 4 26B A4B ranks top on general intelligence lists but scores only 17.4% on SWE-bench Verified — bad for coding agents despite the high score.

Always cross-check coding quality on real benchmarks before committing.

## Step 2 — Cross-check coding benchmarks

Source priority for coding-agent decision-making:

1. **SWE-bench Verified** — multi-step agentic coding, closest to Claude Code / Pi Agent workload.
2. **Aider polyglot** — single-turn cross-language code editing. YAML at [aider/.../polyglot_leaderboard.yml](https://raw.githubusercontent.com/Aider-AI/aider/main/aider/website/_data/polyglot_leaderboard.yml).
3. **LiveCodeBench** — single-turn coding, contamination-free.
4. **Artificial Analysis Intelligence Index** — broad 10-benchmark intelligence (includes code but also math/science/agentic).
5. **Chatbot Arena ELO** — frozen 2025-07, useful as long-tail coverage of older models.

Cross-reference these directly. Treat whichllm output as a **candidate list**, not a score.

## Step 3 — Pareto frontier

Plot **tok/s (measured)** vs **coding quality (SWE-bench Verified or equivalent)** for each candidate that fits your hardware. Pareto-optimal models are NOT dominated — better in at least one axis without being worse in the other.

```
coding quality (SWE-bench %)
   80 ┤
77.2 ●──── Qwen3.6-27B (12-15 tok/s, dense)
      ╲
73.4 ●─────────● Qwen3.6-35B-A3B (22.5 tok/s, MoE 3B-active)
      │                ← Pareto-optimal sweet spot
   65 ●─────────────● Qwen3.5-9B-MTP (134+ tok/s, dense)
      │
   17 ●─────── Gemma-4-26B-A4B (36 tok/s whichllm est.)
   ──┴──────────────────────────── tok/s →
```

Dominance rule: model X dominates Y if X.tps ≥ Y.tps AND X.quality ≥ Y.quality (at least one strict). Drop dominated models from your shortlist.

### Why not just pick the highest score?

For Claude Code / Pi Agent loops, error cost (debugging a wrong code change) is high, latency cost is moderate, abstention is rare. Per [Zellinger & Thomson (arXiv 2507.03834, 2025)](https://arxiv.org/abs/2507.03834), expected cost = `C_a · P(abstain) + C_m · P(wrong) + C_l · latency`. For `C_m > $0.01`, use the most powerful model that runs at acceptable speed. For coding agents, **Pareto-optimal beats highest-score** because quality is roughly sigmoid in score — the gap from 65% to 73% SWE-bench matters more than 73% to 77%.

## Step 4 — Pick ONE Pareto-optimal model to autotune

Don't try to autotune every candidate. Pick the Pareto-optimal point that matches your tolerance:

| Preference | Pick | Rationale |
|---|---|---|
| Coding quality matters most | Best-quality Pareto point | Higher SWE-bench = fewer wrong code edits |
| Speed matters most | Fastest Pareto point | More iterations per minute, snappier UX |
| Balanced (default) | Sweet spot (middle of frontier) | Best quality-per-tok/s ratio |

Once picked, download the GGUF and place it where `local-model-autotuning` expects.

## Step 5 — Hand off to the autoloop

Set the single source of truth:

```bash
# Edit autoresearch/core/config.py
MODEL = '<your-chosen-model-filename>.gguf'
```

Then run the autoloop overnight:

```bash
# Optional: point at a non-default llama.cpp tree (upstream submodule is default)
export AUTORESEARCH_LLAMA_CPP_ROOT=/path/to/your/llama.cpp

cd local-model-autotuning
python3 autoloop.py --vram-limit-mb=<your-VRAM-budget-in-MB>
```

The autoloop hill-climbs around the baseline, saves the best `config.py`, and appends every trial to `results.tsv` (gitignored, stays local).

**Expected behavior overnight**:
- Trial every ~2-5 minutes (model load + Nexus + Claw + Coding)
- Each trial writes 1 row to `results.tsv` with score / VRAM / status
- On keep, `config.py` rewrites with the better config
- SIGINT handler saves state — kill it any time, resume tomorrow
- TPS Floor (`TPS_FLOOR` in Baseline `config.py`, default 20): configs below the floor are auto-discarded; lower it for large MoE on constrained VRAM

## Quick checklist

- [ ] `uvx whichllm@latest` or `llmfit` — shortlist of viable models
- [ ] Cross-reference SWE-bench Verified / Aider for each
- [ ] Plot Pareto frontier on tok/s vs coding-quality axes
- [ ] Pick the Pareto-optimal point matching your preference
- [ ] Download GGUF, place in models/, set `MODEL` in config.py
- [ ] Set `AUTORESEARCH_LLAMA_CPP_ROOT` if using a non-upstream llama.cpp fork
- [ ] Run `python3 autoloop.py` and let it cook
- [ ] Read `results.tsv` in the morning

## Common pitfalls

1. **Picking by whichllm score alone** — Gemma-4-26B-A4B ranks top but is bad at coding agents.
2. **Picking the densest model that fits** — Qwen3.6-27B fits partial but is slower than the MoE alternative.
3. **Trying to autotune every candidate** — 24h × 1 model beats 8h × 3 models (each gets a full search).
4. **Wrong `AUTORESEARCH_LLAMA_CPP_ROOT`** — autoloop silently fails to find llama-server.
5. **Not watching VRAM at startup** — use `--vram-limit-mb=7500` (or whatever your budget) to skip configs that would OOM.

## Related docs

- `docs/models/` — per-model GGUF specs and architecture notes
- `docs/sessions/` — empirical session logs (yours and others)
- `docs/adr/` — architecture decisions (why certain conventions exist)
- `docs/AGENTS.md` — top-level documentation index
