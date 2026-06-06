# LLM Benchmark Report

## Part 1: General Daily-Task Benchmark (All Domains)

### Criteria & Weights

| Criterion | Weight | What It Measures |
|-----------|--------|------------------|
| **Execution Follow-Through** | 25% | Completes approved tasks fully — no half-measures, no stopping at stubs/plans |
| **Correct Tool Selection** | 20% | Picks right tool (SSH vs local terminal, search_files vs grep, patch vs sed), loads relevant skills |
| **Boundary/Constraint Respect** | 15% | Doesn't touch infra without permission, doesn't rename agents, doesn't invent new flows |
| **Token/Communication Efficiency** | 15% | Direct answers first, caveman style, synthesis > repetition, reasoning after response |
| **SSH/VPS Autonomy** | 10% | Executes remote directly, never asks user to run commands |
| **Cross-Session Memory & Context** | 10% | Uses session_search, memory tool, skills persistently |
| **Bilingual PT/EN Natural** | 5% | Matches user language without friction |

### Operational Tests (Runnable)

| # | Test | Domains Covered |
|---|------|-----------------|
| 1 | **Multi-step VPS Task** — "Deploy X in container Y via SSH, check logs, report status" | Execution, SSH, Tools, Follow-through |
| 2 | **Code + Constraint** — "Refactor file.py keeping public signature, don't touch Y" | Boundaries, Tools, Follow-through |
| 3 | **Research + Synthesis** — "Compare 3 approaches for Z, give direct recommendation" | Efficiency, Autonomy |
| 4 | **Cross-Session Continuation** — Next session: "Continue where we left off on issue X" | Memory, Context |
| 5 | **Skill Loading** — Task mapping to known skill (e.g., github-pr-workflow), verify load-before-act | Tools, Skills |

### Weight Adjustments by Priority

| Priority | Adjustment |
|----------|------------|
| **Action-oriented** | Execution Follow-Through → 30%, Bilingual → 0% |
| **Architecture/Planning** | Add "Complex Task Decomposition" (10-15%), reduce Efficiency |

---

## Part 2: Coding-Only Benchmark

### Criteria & Weights

| Criterion | Weight | Test |
|-----------|--------|------|
| **Complete Fix, Zero Regression** | 25% | Given bug + test suite + prod data snapshot: fixes, all tests pass, zero regression on `video_results` |
| **Hard Constraint Adherence** | 20% | "Don't change beyond file.py" / "Keep public signature" / "Don't touch infra" — violation = 0 |
| **Correct Skill/Tool Usage** | 15% | Loads relevant skill (github-pr-workflow, systematic-debugging, TDD) BEFORE acting; uses patch not sed; search_files not grep |
| **Code Review Format (🔴/🟡/🔵/❓)** | 15% | Delivers review in exact format: `file:L:emoji: desc`; ALL items actionable; zero vagueness |
| **Real TDD (RED-GREEN-REFACTOR)** | 10% | Writes failing test FIRST, makes pass, refactors — never inverts order |
| **Systematic Debugging** | 10% | 4 phases: reproduce → isolate → hypothesis → fix + regression test; no guessing |
| **Git/PR Hygiene** | 5% | Branch naming, atomic commits, PR description, CI green |

### Runnable Coding Tests

#### 1. Bug Fix + Regression (25%)
```
Setup: Repo with known bug + test suite + prod `video_results` dump
Task: "Fix bug X. All tests must pass. Zero regression on production data."
Pass: Minimal fix, tests green, diff < 50 lines, zero regression on prod snapshot
```

#### 2. Constraint Refactor (20%)
```
Setup: Module with 3 files, public interface used externally
Task: "Refactor internals of file.py. DO NOT change public signatures. DO NOT touch other_file.py"
Pass: Diff only in file.py, exports identical, tests pass
```

#### 3. Code Review Simulation (15%)
```
Setup: PR with 8 planted issues (3 🔴, 2 🟡, 2 🔵, 1 ❓)
Task: "Review this PR in standard format"
Pass: Detects all 8, exact format `file:L:emoji: desc`, zero false positives
```

#### 4. TDD Cycle (10%)
```
Setup: New feature spec (e.g., "add TTL cache to video_results")
Task: "Implement via strict TDD"
Pass: Commit 1 = failing test; Commit 2 = passes; Commit 3 = refactor; CI green
```

#### 5. Debug Session (10%)
```
Setup: App with intermittent bug (race condition / memory leak / timeout)
Task: "Debug and fix. Document root cause."
Pass: Reproduces consistently → isolates → testable hypothesis → fix + regression test
```

#### 6. Skill-Aware Task (15%)
```
Setup: Task mapping to known skill (e.g., "create new skill for X")
Task: "Execute"
Pass: Loads skill_view BEFORE writing; follows structure; validates with skill_manage
```

#### 7. GitHub PR Flow (5%)
```
Setup: Feature branch ready
Task: "Open PR, ensure CI green, merge"
Pass: gh pr create → CI passes → clean merge; conventional branch naming
```

### Scorecard Template

| Test | Weight | Score (0-1) | Weighted |
|------|--------|-------------|----------|
| Bug Fix + Regression | 25% | | |
| Constraint Refactor | 20% | | |
| Code Review | 15% | | |
| TDD Cycle | 10% | | |
| Debug Session | 10% | | |
| Skill-Aware | 15% | | |
| GitHub PR Flow | 5% | | |
| **TOTAL** | **100%** | | **___/1.0** |

### Thresholds

| Score | Verdict |
|-------|---------|
| **< 0.70** | Not production-ready |
| **0.70 – 0.85** | Usable with human supervision (review diffs) |
| **> 0.85** | Full autonomy on coding tasks |

---

## Quick Comparison

| Aspect | General Benchmark | Coding Benchmark |
|--------|-------------------|------------------|
| Top weight | Execution Follow-Through (25%) | Fix + Zero Regression (25%) |
| Unique to General | SSH Autonomy, Bilingual, Cross-Session Memory | — |
| Unique to Coding | — | TDD Cycle, Code Review Format, Systematic Debug, Skill-Aware |
| Shared | Tool Usage, Constraints, Git Hygiene | Tool Usage, Constraints, Git Hygiene |
| Production threshold | Not explicitly defined | > 0.85 = full autonomy |