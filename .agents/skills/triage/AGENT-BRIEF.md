---
name: agent-brief
description: Universal pre-task contract template — what an agent must read, how it must operate, and how to document outcomes before, during, and after any task.
---

# Agent Brief — Universal Pre-Task Contract

An agent brief is the authoritative contract an agent works from. It covers **three phases**: what to read before starting, how to operate during the task, and what to produce when done.

This template applies to any scope: an issue (build from scratch), a PR (finish existing diff), a research task, or a tuning campaign. Same principles throughout.

---

## 1. Pre-Task Reading Protocol

Before starting **any** task, the agent MUST read in this order:

1. **Tool/Skill frontmatter** — Read name, description, parameter schema of every available tool and skill. Re-read each session. Tool schemas and skill instructions change. Do not rely on memory.

2. **DOX chain walk** — Starting from repo root, identify every file or folder you expect to touch. Walk from root to each target path, reading every `AGENTS.md` found along each route. If a parent AGENTS.md lists a child AGENTS.md whose scope contains the path, read that child too. Use the nearest AGENTS.md as the local contract.

3. **Docs chain** — Read `program.md` (user control channel, never edit), `CONTEXT.md` (terminology), `GOLDEN-RULES.md` (performance rules, validation protocol), and `README.md` (project overview).

4. **Code surface** — Read the files you plan to edit plus their callers/callees. Understand the full flow before changing anything.

---

## 2. Tool Election Hierarchy (MUST)

The agent MUST follow this tool hierarchy. Lower = worse. Only fall down when the higher tool cannot do the job.

| Priority | Tool | When |
|----------|------|------|
| 1 | `ctx_batch_execute` | Multi-command research, parallel queries, gather + analyze in one round trip |
| 2 | `ctx_execute` | Single command whose output might be large (test runs, grep, git log, API calls, CLI output) |
| 3 | `ctx_execute_file` | Read/analyze a file without loading its bytes into context (JSON, CSV, logs, source code) |
| 4 | `ctx_search` | Query previously indexed content (docs, session memory, prior decisions) |
| 5 | `read` | Only when you need exact text for editing (edit/oldText must match) |
| 6 | `bash` | Only whitelist: file mutations (mkdir, mv, cp, rm, touch, chmod), git writes (add, commit, checkout, branch, merge), navigation (cd, pwd), process control (kill, pkill), package mgmt (npm install, pip install), simple output (echo, printf) |
| 7 | `edit` / `write` | File mutations only. Never for reading. |

**Exceptions**: web pages → `ctx_fetch_and_index` then `ctx_search`. Index docs → `ctx_index`. Stats → `ctx_stats`. Doctor → `ctx_doctor`. Upgrade → `ctx_upgrade`. Purge → `ctx_purge`.

---

## 3. Harness & Tuning Contract

### Autotuning repo only — edit surface

| Scope | Editable? | Notes |
|-------|-----------|-------|
| `autoresearch/core/config.py` | **Yes** — constants only | The only tuning surface. Change MODEL, KV_CACHE, THREADS, etc. |
| `benchmark_search.py` | **Yes** — CL arg overrides | For single-run hypotheses with `--desc` |
| `autoloop.py` | Search space only | `SEARCH_SPACE` dict if requested |
| `program.md` | **Never** | User control channel. Fixed unless user explicitly asks. |
| `autoresearch/benchmarks/*` | **Never** | Fixed evaluation harness |
| `results.tsv` | **Never** | Append-only by harness. Agent reads only. |

### Mandatory rules

- **Never run `llama-server` or `llama-bench` directly.** The harness (`benchmark_search.py`, `autoloop.py`) resolves paths, translates config flags to CLI args, manages server lifecycle, monitors VRAM, and logs results. Bypassing the harness produces unlogged, unreproducible trials.
- **All tuning goes through `config.py` constants.** The harness reads config.py and generates the correct `llama-server` command. Never override flags via raw CLI arguments.
- **Push nothing to remote.** Benchmark results, config tweaks, and search branches stay local-only.

---

## 4. Validation Protocol

Every Trial runs a **2-step validation** before full eval:

1. **llama-bench speed check** — `prompt=512`, `gen=128`, 3 repeats. If `tg_tps < TPS Floor` (20.0 t/s), Trial FAILs immediately. No server spin-up, no coding eval.

2. **Quick coding eval** — 2 tasks per dataset (HE+, MBPP+, LCB, BigCodeBench). Validates the model generates coherent code under the config — not just fast garbage.

### How to validate a single model

1. Set `MODEL = '<filename>.gguf'` in `autoresearch/core/config.py`
2. Run: `python3 benchmark_search.py --validation --desc "validate <filename>"`
3. One model at a time. Never parallel. Never chain multiple in one command.
4. Result goes to `results.tsv` with category `validation`.

Use `--validation` flag to run steps 1-2 and exit (no extended eval, no keep/discard). Quick sanity check before committing to a full Trial.

---

## 5. Session Hygiene

| Type | Location | Rules |
|------|----------|-------|
| Durable docs | `docs/models/`, `docs/adr/`, `docs/discovery/`, `docs/sessions/` | Commit. Update on every meaningful change. |
| Results | `results.tsv` | Append-only by harness. Never edit manually. |
| Session state | `run.log`, intermediate files | Ephemeral. Gitignored. |
| Discovery caches | `models/.cache/`, `llm/gguf/.cache/` | Gitignored. |

After every meaningful change: run a DOX pass (update closest owning AGENTS.md, affected parents/children, remove stale text, verify).

---

## 6. Acceptance Criteria

The agent brief must include concrete, testable criteria. Each independently verifiable.

- **Good:** "Running `python3 benchmark_search.py --validation` with model X reports tg_tps ≥ 20.0"
- **Bad:** "Tuning should work"

---

## Template

```markdown
## Agent Brief

**Category:** bug / enhancement / tuning / research
**Summary:** one-line description

**Current behavior / baseline:**
What happens now. For bugs: broken behavior. For tuning: current config + score.

**Desired behavior / target:**
What should happen after. Edge cases, error conditions, target score.

**Pre-task reading:**
- Tools/skills to read frontmatter for
- DOX chain: AGENTS.md files to walk
- Docs: program.md, CONTEXT.md, GOLDEN-RULES.md sections to review
- Code surface: files to inspect before editing
- Test files: relevant tests to check

**Tool hierarchy note:**
- Research phase: ctx_batch_execute / ctx_execute / ctx_execute_file
- Edit phase: matching oldText via read, then edit/write

**Key interfaces:**
- Type, function, config shape to change — by contract, not file path

**Tuning surface (if applicable):**
- Which constants in config.py to modify
- Which harness flags to use

**Acceptance criteria:**
- [ ] Criterion 1 — specific, testable, verifiable
- [ ] Criterion 2
- [ ] Verification command to run

**Out of scope:**
- Things NOT to change
- Adjacent features that seem related but aren't
```

## Examples

### Good brief — tuning task

```markdown
## Agent Brief

**Category:** tuning
**Summary:** Find optimal KV cache type for Gemma-4-12B on RTX 4060 8GB

**Current behavior:**
Running with q4_0 across K/V. tg_tps=48.1 on llama-bench, val_score=0.72.

**Desired behavior:**
Try turbo3/turbo4/f16 for kv_cache_k, keep kv_cache_v at q4_0. Target: tg_tps ≥ 25.0, val_score ≥ current.

**Pre-task reading:**
- GOLDEN-RULES.md §1 (KV Cache Quant)
- autoresearch/core/config.py (current values)
- autoresearch/runners/evaluation.py (run_llama_bench_validation)

**Tool hierarchy note:**
- ctx_execute for benchmark_search.py runs
- ctx_execute_file for results.tsv analysis
- edit for config.py changes

**Tuning surface:**
- KV_CACHE_K, KV_CACHE_V in config.py
- Run: `python3 benchmark_search.py --validation --desc "turbo3 K cache"`

**Acceptance criteria:**
- [ ] Each candidate runs through validation (llama-bench + 2-task coding)
- [ ] Best config persisted in config.py
- [ ] Results logged in results.tsv

**Out of scope:**
- Model file changes (no re-quantization)
- Benchmark harness modifications
```

### Good brief — code change

```markdown
## Agent Brief

**Category:** enhancement
**Summary:** Add `--dry-run` flag to serve-config.py for printing llama-server command without starting it

**Current behavior:**
`serve-config.py serve` starts llama-server immediately. To see what command would run, user must inspect config.py manually.

**Desired behavior:**
`serve-config.py dry-run` prints the exact llama-server command that would be run to stdout, then exits. No server start, no port binding.

**Pre-task reading:**
- tools: review service launcher skill if active
- DOX: root AGENTS.md, scripts/AGENTS.md
- scripts/serve-config.py (existing serve/stop/status implementation)
- scripts/AGENTS.md (operator script contracts)

**Key interfaces:**
- `scripts/serve-config.py` — add `dry-run` subcommand
- The command-building logic lives in `_build_cmd()` inside `serve-config.py`
- Flags must match what llama_runner.py would produce for the same config

**Acceptance criteria:**
- [ ] `python3 scripts/serve-config.py dry-run` prints the llama-server command and exits
- [ ] Command matches what `serve` would run (same args, flags, model path)
- [ ] No llama-server process is started
- [ ] `python3 scripts/serve-config.py serve` still works unchanged

**Out of scope:**
- Changes to config.py or llama_runner.py
- Adding a `--validate` flag (separate concern)
```

### Bad brief

```markdown
## Agent Brief

**Summary:** Fix the tuning

**What to do:**
Open evaluation.py and change the llama-bench parameters.
Look at the function around line 200.

**Files to change:**
- autoresearch/runners/evaluation.py (line 200)
- tests/test_run.py (line 50)
```

This is bad because:
- No category
- Vague description
- References file paths and line numbers that go stale
- Proposes editing FIXED harness (evaluation.py) — violates tuning contract
- No acceptance criteria
- No scope boundaries
- No pre-task reading step
- No tool hierarchy consideration
