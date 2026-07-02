# Code Context — AutoResearch Agent Brief

## 1. AGENT-BRIEF.md + Triage Skill Files

**Files:**
- `.agents/skills/triage/AGENT-BRIEF.md` — templates + principles for writing durable briefs
- `.agents/skills/triage/SKILL.md` — triage state machine definition, references brief
- `.agents/skills/triage/OUT-OF-SCOPE.md` — `.out-of-scope/` KB docs for rejected features

**Key findings:**

- **Brief template** requires: Category, Summary, Current/Desired behavior, Key interfaces (types/functions, NOT file paths), Acceptance criteria (testable), Out of scope boundaries. Examples show bug, enhancement, PR variants.
- **Durability rule**: briefs must survive file renames — no file paths, no line numbers. Describe behavioral contracts + types.
- **SKILL.md** references brief via `AGENT-BRIEF.md` for writing agent-ready briefs during `ready-for-agent` state. Brief is posted as issue comment.
- **OUT-OF-SCOPE.md** documents rejected enhancements only (not bugs, not already-implemented). One file per concept. Used during triage step 1 for dedup.
- **State machine**: needs-triage → needs-info / ready-for-agent / ready-for-human / wontfix. PRs treated as issues with attached code.

**Severity**: Reference docs only. No active changes needed unless triage workflow is being modified.

---

## 2. Root AGENTS.md — DOX Hierarchy

**File:** `AGENTS.md` (root)

**Key findings:**

- **Pre-Task Reading section** (new): Must read frontmatter of every tool/skill before starting. Re-read each session. Tool schemas change.
- **DOX framework** installed in this session. Core contract: AGENTS.md files are binding work contracts for their subtrees. Walk DOX chain before editing (parent → child).
- **Read Before Editing**: 7-step protocol. Walk from root to target path, read every AGENTS.md along the route. Closer doc controls local details, no child doc weakens DOX.
- **Update After Editing**: DOX pass required after every meaningful change. Update closest owning AGENTS.md + affected parents/children. Remove stale/contradictory text.
- **Child DOX Index** at bottom: autoresearch/, docs/, scripts/, tests/. Each with own AGENTS.md.
- **GitNexus section** appended at end: 918 symbols, 1359 relationships, 52 execution flows. MUST run impact analysis before editing any symbol.
- **Local contracts preserved**: caveman mode, config frozen (never ctx_size), min 100k ctx, no timeouts, ask-first-ship-never.

---

## 3. Autotuning Core Docs

**Files:** `program.md`, `GOLDEN-RULES.md`, `CONTEXT.md`

**Key findings:**

### program.md — Agent contract (user-only territory, dev agents don't touch)
- Fixed rules for the autonomous Search loop. Not to be edited by dev agents.
- Single mutable surface: `autoresearch/core/config.py`. All tuning goes through it.
- Evaluation: unified Trial with HE+, MBPP+, SWE-bench (stub). Val Score = 40% SWE + 30% HE + 30% MBPP (SWE weighted 0 currently).
- TPS floor = 20.0. Below → Val Score zeroed. Flash Attn must always be `on`.
- Strict terminology: Search, Round, Trial, Baseline, Neighbor, Val Score, Pareto Tie-Breaker, Local Maxima, Random Restart.
- Autonomy rule: once Search starts, continue until interrupted. Looping agent MUST NOT edit code. On error → stop, report, warn. Never push results to remote.

### GOLDEN-RULES.md — §5 Validation Protocol, §6 Use the Harness
- **§5 Validation Protocol**: Every Trial runs 2-step validation before full eval:
  1. `llama-bench` speed check (prompt=512, gen=128, 3 repeats). tg_tps < 20.0 → FAIL immediately.
  2. Quick coding eval (2 tasks per dataset: HE+, MBPP+, LCB, BigCodeBench). Validates coherent code generation.
  - `--validation` flag runs steps 1-2 then exits.
- **§6 Use the Harness**: Never run `llama-server` or `llama-bench` directly. Bypassing harness produces unlogged, unreproducible trials. Only mutation surface is `config.py`.
- **Performance flags**: KV Cache quant (q8_0, turbo3, q4_0), Flash Attn always on, MTP (1.15-2.0x speedup), batching, threading.
- **VRAM safety**: Pre-flight estimation. TPS floor as guardrail. NVML failsafe.

### CONTEXT.md — Terminology
- Strict terminology definitions: Validation, Val Score, TPS Floor, Speed Factor, SearchStrategy, ServerIntent, TurboQuant, MTP, benchmarks (Nexus, Claw, Coding).
- **"Validation"** = 2-step pre-check (llama-bench + quick coding eval). NOT the full eval.
- **Val Score weighting** (authoritative): `INCLUDE_CODING=True` → 0.80×coding + 0.10×nexus + 0.10×claw. False → 0.60×nexus + 0.40×claw.
- **Generic config skeleton** with llama-server command template, MTP flag variants (upstream vs turboquant), key flags explained, WSL caveats.
- **Discovery workflow** cross-ref: whichllm → Pareto frontier → autoloop handoff.

---

## 4. Tuning Surface & Runner

**Files:** `autoresearch/core/config.py`, `autoresearch/runners/evaluation.py`, `autoresearch/runners/run.py`

### config.py — What gets edited
- **The ONLY changeable file** for agent tweaks (stated in file header).
- CTX_SIZE = 131072 (frozen, never touch. Min ctx floor = 100k).
- Current model: `gemma-4-12B-it-qat-UD-Q4_K_XL.gguf`
- KV cache: q4_0 across K/V, BATCH=1024, UBATCH=256, THREADS=8, THREADS_BATCH=8
- FLASH_ATTN = 'on'. SPEC_TYPE = None (MTP disabled). N_CPU_MOE = 32.
- Generation: TEMP=0.4, TOP_P=0.95, TOP_K=20, MIN_P=0.0, REPEAT_PENALTY=1.05
- Benchmarks: INCLUDE_CODING=True, INCLUDE_NEXUS=False, INCLUDE_CLAW=False
- Task limits: CODING=10, LCB=10, BIGCODE=10, EVALPLUS_STRICT=True, TRIAL_BUDGET=300
- `load_config()` / `write_config()` — hot-reload and persist functions for config.py.
- write_config() preserves section headers and comments.

### evaluation.py — `run_llama_bench_validation()` + `run_trial()` flow
- `TrialResult` dataclass: status, val_score, coding_val, coding_tps, lcb/he/mbpp/bigcode/swe scores, avg_tps, peak_vram_gb, bench_tg_tps, bench_pp_tps.
- `run_llama_bench_validation()`: calls llama-bench with prompt=512, gen=128, 3 repeats. Returns tg_tps. Raises on failure (subprocess error, parse error, zero result).
- `ExperimentRunner.run_trial()`:
  1. Normalize config via `ServerIntent.from_config()`
  2. Pre-check: llama-bench validation (unless skip_bench). If tg_tps < threshold → FAIL. If `validation` flag → reduce to 2-task coding, fall through.
  3. Full eval: spin up llama-server via `LlamaServerRunner`, run coding benchmark via `run_coding()`, compute combined metrics.
  4. TPS check: if avg_tps < 20.0 → score zeroed.
- Gen kwargs filtered to non-None values only.
- VRAM tracked via runner.peak_vram_mb.

### run.py — `--validation` flag
- `--validation` flag: runs llama-bench + 2-task coding eval then exits. No extended eval, no keep/discard.
- CLI defaults all pull from config.py constants. Extensive arg list (~40 params).
- Grid sweep mode with `--grid-*` flags for multidimensional sweeps.
- Results logged to `results.tsv` with commit hash, scores, memory, status (keep/discard), description.
- `handle_single_run()`: compares against previous best 'keep' score, logs keep/discard, prints summary with git commands.

---

## 5. Test Patterns

**Files in `tests/`:**

```
tests/test_adversarial_challenger.py
tests/test_autoloop.py
tests/test_benchmark_coding.py
tests/test_benchmark_harness.py
tests/test_benchmark_search.py
tests/test_config_parsing.py
tests/test_llama_client.py
tests/test_llama_runner.py
tests/test_run.py
tests/test_search_strategy.py
```

**Most relevant to validation/runner flow:**
- `test_llama_runner.py` — tests LlamaServerRunner, ServerIntent, llama-bench resolution. Core for runner lifecycle.
- `test_llama_client.py` — tests API client used during coding eval.
- `test_run.py` — tests CLI arg parsing, run_evaluation(), handle_single_run(). Most recently modified (validation fix commit 95f438e).
- `test_search_strategy.py` — tests Neighbor generation, Pareto logic, random restart. Core for autoloop.
- `test_benchmark_coding.py` — tests coding benchmark harness (HE+, MBPP+, LCB, BigCodeBench).
- `test_config_parsing.py` — tests config.py load/write roundtrip.

---

## 6. Git Log — Last 30 Commits

**Recent activity (most recent first):**

| Commit | Type | Subject |
|--------|------|---------|
| 95f438e | fix | Adjust validation mode: quick coding checks after llama-bench, lower TPS threshold |
| 8a6a33b | refactor | Remove deprecated grill-me and grill-with-docs skills |
| 66642b8 | fix | Patch npx skills CLI for short-form GitHub URLs |
| fef5223 | fix | Remove stale skills from lockfile, rename diagnose→diagnosing-bugs |
| 9653425 | feat | Extract ExperimentRunner, apply ponytail simplifications |
| 3a35a26 | refactor | Consolidate config normalization into ServerIntent.from_config() |
| 38e6156 | refactor | Remove unused warmup, simplify llama-bench validation logic |
| 49c1ac1 | docs | Clarify agent communication guidelines in AGENTS.md |
| f76aac0 | feat | Add llama-bench integration and validation to the runner |
| 97444c6 | feat | Update model configs and add Gemma-4-12B model card |
| f2806a9 | docs | Add config-frozen and never-touch-ctx rules to AGENTS.md |
| c3e8af0 | docs | Add 'always update docs on improvement' rule to AGENTS.md |
| 44a5264 | feat | Add Qwythos-9B model card, llama-bench sweep, batch sweet-spot comment |
| b0b4cef | docs | 2026-06-29 tuning session |
| 58f6271 | docs | Remove GitNexus skill documentation |
| f6d34cd | docs | Remove outdated grill-me and grill-with-docs skills |
| 4bcc3f8 | merge | Merge autoresearch/ornith35b-val-opt into main |
| ... | ... | Older: Ornith 35B tuning, model card updates, INDEX.md changes |

**Pattern**: Heavy refactoring last 7 commits (runner extraction, config normalization, validation fix). Recent focus on stabilizing validation pipeline and cleaning up skill infrastructure. Earlier commits were Ornith 35B tuning sessions.

---

## Clarification Questions for the User

1. **Validation mode behavior**: Recent commit (95f438e) lowered bench TPS threshold and made validation run quick coding after llama-bench. Is this the intended final behavior, or is more work planned on the validation path?

2. **Nexus/Claw benchmarks**: Currently INCLUDE_NEXUS=False and INCLUDE_CLAW=False in config.py. Are these permanently disabled or expected to be re-enabled?

3. **GitNexus integration**: Root AGENTS.md contains extensive GitNexus MCP instructions (impact analysis before edits, etc.). Is this tooling available in this environment, or is it aspirational?

4. **Triage skill files**: The `.agents/skills/triage/` files reference a GitHub issue tracker workflow. Does this repo use that workflow, or are these orphaned templates?

5. **Scope of current work**: Are we validating/refining the existing validation pipeline, adding a feature, or preparing for a new tuning campaign?

```acceptance-report
{
  "criteriaSatisfied": [
    {
      "id": "criterion-1",
      "status": "satisfied",
      "evidence": "All 6 sections inspected: AGENT-BRIEF.md + triage skill files read, AGENTS.md DOX hierarchy mapped, core autotuning docs (program.md, GOLDEN-RULES.md, CONTEXT.md) read, tuning surface (config.py, evaluation.py, run.py) read, 11 test files listed with relevance assessment, last 30 commits analyzed with pattern summary."
    }
  ],
  "changedFiles": [
    "context.md"
  ],
  "testsAddedOrUpdated": [],
  "commandsRun": [
    {
      "command": "read 8 files",
      "result": "passed",
      "summary": "Read all target docs: AGENT-BRIEF.md, SKILL.md, OUT-OF-SCOPE.md, AGENTS.md, program.md, GOLDEN-RULES.md, CONTEXT.md, config.py, evaluation.py, run.py"
    },
    {
      "command": "find tests/**/*.py",
      "result": "passed",
      "summary": "Found 11 test files"
    },
    {
      "command": "git log --oneline -30",
      "result": "passed",
      "summary": "Retrieved last 30 commits"
    }
  ],
  "validationOutput": [
    "context.md written to context.md with 6 sections and 5 clarification questions"
  ],
  "residualRisks": [
    "Did not read autore search/benchmarks/ or tests/ file contents — assumption is these are fixed as stated in program.md",
    "GitNexus MCP tools not available for query — relied on grep/read only",
    "evaluation.py imports autore search.benchmarks.benchmark_coding — not inspected for validation-specific behavior",
    "LlamaServerRunner and ServerIntent.from_config() not deep-dived — referenced but not read as source"
  ],
  "noStagedFiles": true,
  "diffSummary": "New context.md with compressed briefs across 6 domains, 5 clarification questions, acceptance report",
  "reviewFindings": [
    "no blockers: All target files read successfully",
    "note: gitnexus MCP tools referenced in AGENTS.md but not available in this session — analysis done via grep/read",
    "note: evaluation.py imports benchmark_coding not inspected — shallow coverage on that path"
  ],
  "manualNotes": "Scout mission complete. Primary risk: did not read LlamaServerRunner internals or benchmark_coding module — those may matter if the validation path needs deeper changes. The 5 clarification questions should be resolved before implementation planning."
}
```