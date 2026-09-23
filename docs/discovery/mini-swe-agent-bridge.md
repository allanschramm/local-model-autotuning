# mini-swe-agent → local llama-server (bridge)

> Why the harness matters more than the model for daily SWE — and how to wire [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) into the operator's existing `llama-server` alias in under five minutes. Pairs with the [2026-09-22 external findings](./2026-09-22-external-findings.md).

## Why this exists

The operator's `models/aliases/qwen3.6-35b-a3b-mtp/config.yaml` already runs an OpenAI-compatible `llama-server` on port 18080. That surface is enough for `mini-swe-agent` to talk to. The point of this doc is to capture:

1. **The economic case** — mrguo6221 measured +22.2 pt SWE-bench Verified uplift from `mini-swe-agent` baseline → `claude-cli -p` + custom proxy on the **same model**. We don't need their 47 K LoC proxy; we just need the harness shape.
2. **The wiring recipe** — env vars, smoke test, and the `plan-first` skill from [`esagduyu/qwen-pi-plan-vs-haiku-benchmark`](https://github.com/esagduyu/qwen-pi-plan-vs-haiku-benchmark) that already proved +2.7× wall-time, +30 % spec-adherence lift on Qwen3.6-35B-A3B specifically.
3. **The sandbox shape** — `subprocess.run` per action in mini-swe-agent makes the swap to Docker / Podman trivial; the rig already has `autoresearch/runners/process_guard.py` for OS-native process binding (Windows Job Objects / Linux `PR_SET_PDEATHSIG`).

## The economic case (one paragraph)

mrguo6221 measured Qwen3.6-27B-Thinking (FP8) at **67.8 % on SWE-bench Verified** with `mini-swe-agent` baseline (no proxy, no skill — just `mini`). With `qwen-cli` + their 47 K LoC custom proxy: 87.4 % headline / 85.4 % strict. With `claude-cli -p` + same proxy: **90.0 % headline / 88.0 % strict**. Same model — +22.2 pt. The `plan-first` skill alone (a markdown file the model reads) gave a 2.7× wall-time cut + 6→9/10 spec-adherence on the **same Qwen3.6-35B-A3B the operator already runs**. Combined: the harness+skill delta is plausibly the single highest-ROI change the operator can make before buying a new model. Source: [mrguo6221/swe-bench-88-90](https://github.com/mrguo6221/swe-bench-88-90) README, 2026-05-11.

## Pre-flight

1. **Confirm the alias is serving.**
   ```bash
   .\venv\Scripts\python.exe scripts\model_up.py qwen3.6-35b-a3b-mtp serve
   # In another shell:
   curl http://127.0.0.1:18080/v1/models
   ```
   Expect a JSON body with `id: qwen3.6-35b-a3b` (or whatever the alias uses). If empty, re-check `AUTORESEARCH_LLAMA_CPP_ROOT` and the alias config.

2. **Install mini-swe-agent** (use `uvx` so it doesn't pollute the venv — its deps are isolated):
   ```bash
   uvx mini-swe-agent --help
   # or
   pipx ensurepath && pipx run mini-swe-agent --help
   ```
   Expected first-run: prints the same `--help` as `mini --help` (uvx spawns an ephemeral venv, then drops into the `mini` CLI).

3. **Grab the `plan-first` skill** from `esagduyu/qwen-pi-plan-vs-haiku-benchmark`:
   ```bash
   gh api repos/esagduyu/qwen-pi-plan-vs-haiku-benchmark/contents/skills/plan-first.md \
     --jq '.content' | ForEach-Object { [System.Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($_)) } \
     > models/aliases/qwen3.6-35b-a3b-mtp/skills/plan-first.md
   ```
   (Path under `models/aliases/<alias>/skills/` is the convention; the harness doesn't read it, but it's the convention used by `pi-coding-agent`.)

## Wire-up (env-only — no code changes)

Set three env vars and call `mini`:

```bash
# Tell mini-swe-agent where the local llama-server is.
export OPENAI_BASE_URL="http://127.0.0.1:18080/v1"
export OPENAI_API_KEY="sk-no-auth-required"   # llama-server doesn't enforce auth by default
export MSWEA_MODEL_NAME="qwen3.6-35b-a3b"     # whatever /v1/models reports

uvx mini-swe-agent
```

For non-interactive runs (the analog of `pi -p` from `esagduyu`):

```bash
mkdir -p /tmp/swe-run && cd /tmp/swe-run
cp /path/to/spec/SPEC.md .
# Baseline (no skill):
uvx mini-swe-agent -p "$(cat SPEC.md)"
# With plan-first (recommended):
uvx mini-swe-agent --skill /path/to/skills/plan-first.md -p "Read SPEC.md. Apply the plan-first skill."
```

## Daily-driver recipe (the operator's working stack)

The minimum that should always be on:

| Layer | Component | Source |
|---|---|---|
| Model | `qwen3.6-35b-a3b` (alias `qwen3.6-35b-a3b-mtp`) | already on disk |
| Server | `llama-server` (upstream b10867) + MTP n2 + q4_0 KV + 262 144 ctx | already in alias config |
| Agent | `mini-swe-agent` (uvx ephemeral) | install one-shot |
| Skill | `plan-first.md` (7-line prompt) | copy from esagduyu |
| Sandbox | `subprocess.run` per action (mini default) — for stricter isolation, swap to Docker | mini-swe-agent env class |

**Expected delta vs the operator's current daily driver** (qualitative, based on the upstream ablations):

- **Wall time** — 2–3× faster on small specs (esagduyu measured 9 min → 3.3 min on a Python CLI todo spec)
- **Spec adherence** — 6/10 → 9/10 on the same model (esagduyu)
- **Long-task stability** — bash-only tool surface removes the tool-call-format failure mode that breaks thinking models mid-loop; linear history makes mid-loop stalls inspectable
- **Sandbox safety** — `subprocess.run` per action means a single bad command can't poison the persistent shell (the SWE-agent team's "big deal" callout)

## What this does NOT solve

- **Single-model ceiling.** Mini-swe-agent inherits the model's coding ability. The +22.2 pt mrguo6221 measured was on **the same model** — the proxy closed the gap between mini-swe-agent's 67.8 % and a 90 % ceiling. Without the proxy, expect the operator's daily-driver score on SWE-rebench-V2 / DM-Code-Agent-30 to land in the 60–75 % band, not 90 %. The remaining gap is real model-quality headroom, not a harness issue.
- **Tool-use beyond bash.** Mini-swe-agent is bash-only by design. Claude Code / Pi expose custom tools (Edit, file viewing, structured output) that aren't reachable from `mini`. For those, the operator's existing daily Claude Code / Pi loop is the right surface.
- **Latency hiding.** The bash-only tool surface means every "edit a file" costs a `cat > file << EOF` round-trip — visible on chat UX. Fine for SWE tasks, awkward for interactive coding.

## Smoke test

After wire-up, verify the loop end-to-end on a 30-line micro-spec:

```bash
mkdir -p /tmp/swe-smoke && cd /tmp/swe-smoke
cat > SPEC.md <<'EOF'
# SPEC.md

Write a Python script `wordcount.py` that:
1. Reads `input.txt` (one word per line) from the current directory.
2. Prints the top 5 most frequent words with their counts, sorted descending by count then ascending by word.
3. Handles missing `input.txt` by exiting non-zero with a clean message (no traceback).
4. Ships a `test_wordcount.py` with pytest covering happy path, empty input, and missing-file error.

Constraints: standard library only.
EOF

uvx mini-swe-agent -p "$(cat SPEC.md)"
# After run:
ls test_wordcount.py && pytest -q test_wordcount.py
```

Expected: ~3 min wall, 3/3 tests pass on Qwen3.6-35B-A3B + plan-first (esagduyu baseline).

## Open questions

- **Trajectory export**: mini-swe-agent uses a linear message history. Capturing it as JSONL for FT/RL is documented (`mini --trajectory-output ...` per their docs); needs an integration test with the operator's `autoresearch/benchmarks/agentic_coding/` shape.
- **Sandbox choice for the operator rig**: `subprocess.run` on Windows spawns a console-less subprocess per action — fine for SWE tasks but will flash terminal windows unless `--windows-no-window` is set in the launcher. The operator's `autoresearch/core/process_guard.py` already binds `llama-server` to a Windows Job Object; mini-swe-agent's child processes should inherit the same handling.
- **Sandboxing against model-induced damage**: bash-only with no approval is the point of mini, but for an 8 GB daily driver the cost of a bad `rm -rf` is real. DM-Code-Agent's 30-task hidden benchmark is a safer eval surface than `mini -p` on a real repo.
- **Plan-first skill format compatibility**: the esagduyu skill is plain markdown with frontmatter. `pi-coding-agent` reads these natively; mini-swe-agent reads them as `--skill <file>` and prepends them as a system message. Worth verifying the operator's existing skill-discovery conventions don't conflict.

## Sources

- https://github.com/SWE-agent/mini-swe-agent (7 885 stars, 2026-09-21)
- https://github.com/mrguo6221/swe-bench-88-90 (2026-05-11, +22.2 pt harness uplift on same model)
- https://github.com/esagduyu/qwen-pi-plan-vs-haiku-benchmark (2026-05-01, +2.7× wall-time, +30 % spec-adherence on Qwen3.6-35B-A3B)
- https://github.com/SWE-rebench/SWE-rebench-V2 (arXiv 2602.23866, anti-contamination benchmark for the eval half)
- https://github.com/hwfengcs/DM-Code-Agent (30-task hidden-test benchmark for the eval half)
