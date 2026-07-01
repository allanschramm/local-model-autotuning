# Research: Agent Brief Patterns & Best Practices

## Search Strategy

Will run 4 queries covering:
1. Agent contract / brief templates for autonomous coding agents
2. Pre-task reading patterns in AI-assisted development
3. Harness-based evaluation patterns (agent edits config, harness runs)
4. AGENTS.md / CLAUDE.md style specifications

## Results

### Finding 1: AGENTS.md / CLAUDE.md as Agent Contract Files

**Pattern:** A single markdown file in the project root that acts as a binding contract for the agent. It specifies behavioral rules, scope boundaries, immutable constraints, and acceptance criteria — not implementation steps.

**Where it applies:** Any repo where multiple autonomous agents or agent sessions work. The file sits in the root (standard) or per-directory for sub-team boundaries.

**Concrete example:** This very repo's `AGENTS.md` hierarchy (root → child per directory) uses:
- **Purpose** section — why the directory exists
- **Ownership** — who owns what
- **Local Contracts** — behavioral rules with MUST/NEVER/SHOULD keywords
- **Work Guidance** — which files to change first, coding style
- **Verification** — commands to run before finishing
- **Child DOX Index** — what sub-folders have their own contracts

The key behavioral spec lines from this repo's contracts: *"Loop agents: Strictly forbidden from editing code. If error/crash occurs, stop immediately, report error, warn user."* — this is a behavioral constraint, not a procedural instruction.

**Anti-pattern:** Writing procedural step-by-step instructions instead of behavioral rules. Agents follow constraints better than recipes. An agent told "run prepare.py then benchmark_search.py" will do that blindly. An agent told "You may edit benchmark_search.py but never prepare.py" will self-correct.

**Template structure from observed practice:**
```markdown
# [Scope] Agent Contract
## Purpose
— why this module exists
## Ownership
— who owns/responsible
## Rules
— MUST: always do X
— NEVER: do Y
— If Z happens: stop, report, do not recover
## Fixed files (do not edit)
— list
## Editable files
— list
## Verification
— commands to run
## Child contracts
— sub-directories with their own contracts
```

**Sources (observed patterns, not single origin):**
- Anthropic's CLAUDE.md specification (per-project instructions)
- This repo's DOX hierarchy with AGENTS.md chain
- "Defining Charters for AI Agents" — behavioral boundaries over procedural scripts

---

### Finding 2: Pre-Task Reading Protocol as Tool/Manifest Manifest

**Pattern:** A structured protocol where the agent reads every available tool/skill manifest (name, description, parameter schema) before starting the task. This is not optional — it's enforced as a precondition, often stated in the agent contract itself.

**Where it applies:** Multi-tool environments where agents can choose among many capabilities. Particularly common in agent frameworks with plugin/skill architectures.

**Concrete example from this repo's contract:**
```
## Pre-Task Reading
- Before starting any task, read the frontmatter of **every available tool and skill**
- Tool names, descriptions, parameter schemas, and skill files
- Know what each can do before choosing one
- Do not rely on memory. Re-read tool/skill descriptions each session.
- Tool schemas and skill instructions change.
```

This is a binding contract clause. The pattern is: **"Re-read the interface before acting, every session."** It prevents stale knowledge and tool hallucination.

**Anti-pattern:** Caching tool descriptions across sessions. Agents drift as tools get updated. The pattern explicitly forbids memory reliance.

**Related variant — "Read before editing" chain:**
```markdown
## Read Before Editing
1. Read the root AGENTS.md
2. Identify every file you expect to touch
3. Walk from repo root to each target path
4. Read every AGENTS.md found along each route
5. Use the nearest AGENTS.md as the local contract
6. If docs conflict, the closer doc controls local work details
```

This establishes a **document hierarchy walk** as a precondition for any edit. The pattern is: "read context from broadest to narrowest before acting."

---

### Finding 3: Harness-Based Evaluation — Agent Edits Config, Harness Runs Tests

**Pattern:** A clear separation of concerns where the agent is allowed to modify configuration/parameters but a fixed harness executes the evaluation. The agent never runs tools directly — it writes configs, and the evaluation loop picks them up.

**Where it applies:** Hyperparameter optimization, automated benchmarking, prompt engineering pipelines, any hill-climbing or search system where you want to prevent the agent from gaming the evaluation.

**Concrete example pattern (synthesized from this project's structure + known prior art):**
- `program.md` — defines fixed rules, immutable
- `prepare.py` — fixed evaluation harness, immutable
- `benchmark_search.py` — the editable surface where agent can modify search strategy
- Agent writes config → harness runs benchmark → results feedback loop

**Key properties of this pattern:**

1. **Immutable harness, mutable search space.** The agent can only touch what defines the search (params, strategy), never the measurement apparatus.
2. **File-level access control.** List immutable files separately from editable files. Enforce via explicit error if agent tries to edit fixed files.
3. **Results are append-only.** Agent reads results (e.g., `results.tsv`) but never modifies them. The harness writes.
4. **Agent never executes the evaluation.** The evaluation is triggered externally or by the harness itself. This prevents shortcut-taking where agent would mark a run as "best" without running it.
5. **The agent is a config generator, not an evaluator.**

**Anti-patterns:**

- **Agent-in-the-loop evaluation** where the agent both runs tests and interprets results. Creates incentive to skip expensive runs or fabricate results.
- **Mutable evaluation harness.** If the agent can modify the harness to get better scores, the benchmark loses all meaning.
- **No guardrail on execution time.** Without explicit constraints, agents can run unlimited evaluations until they find a good result by luck.
- **Agent modifying results files.** Agents can fake success by writing to the results file.

**Guardrails observed:**
- Read-only access to results
- Immutable harness with documentation of what is immutable
- Explicit error handling: "If error/crash occurs, stop immediately, report error, warn user"
- Config frozen: "Never change any config value without explicit user permission"

---

### Finding 4: Specification Pattern — Behavioral over Procedural, Acceptance Criteria over Instructions

**Pattern:** Specifications for autonomous agents emphasize **what behavior is acceptable** and **how to verify success** rather than **how to do the task**. This is distinct from traditional developer documentation which is usually procedural.

**Where it applies:** Any system where the agent has autonomy in choosing *how* to accomplish a task.

**Concrete patterns observed across multiple sources:**

1. **DOX (Durable Object X) framework** — This repo uses a hierarchical contract system where AGENTS.md files are binding work contracts. Key innovation: "Update After Editing" pass — every meaningful change requires a DOX pass before completion, which catches stale documentation.

2. **Acceptance criteria pattern** (used in this very task's output format):
   - Specific criterion with required evidence
   - Explicit fields: changed-files, tests-added, commands-run
   - "No staged files" check (no uncommitted changes left behind)
   - Residual risks declared explicitly

3. **Behavioral constraint language** uses explicit keywords:
   - MUST / MUST NOT — hard constraints
   - NEVER — absolute prohibition
   - SHOULD / SHOULD NOT — preference
   - Prefer — comparative guidance
   - "If X happens: do Y" — error handling protocol

**Anti-patterns:**
- **Vague requirements** like "be careful" — agents need binary or enumerated constraints
- **Negative-only specs** ("don't do X") without positive guidance on what to do instead
- **Contradictory constraints** (e.g., "be thorough" + "be fast" without tradeoff guidance)
- **Implicit context** — assuming the agent knows things not written in the contract

---

### Finding 5: Session Hygiene — Ephemeral State, Durable Artifacts

**Pattern:** Clear separation between what is ephemeral (session state, search progress, intermediate results) and what is durable (results files, model cards, ADRs). Agents write to specific durable paths; session state is not preserved across sessions.

**Where it applies:** Any multi-session project where agents run iteratively.

**Concrete example from this project:**
- Durable: `docs/models/`, `docs/adr/`, `docs/discovery/`, `docs/sessions/`
- Ephemeral: `run.log`, intermediate results, session state
- Git-ignored: `llm/gguf/*.gguf`, `run.log`, `results.tsv`
- Rule: "Never push results, tweaks, or run branches to remote repository"

**The pattern:**
- Durable paths are explicitly declared
- Ephemeral paths are in .gitignore
- Agent contract states what must be documented (model cards, ADRs, discovery guides)
- Session logs go in a dedicated directory with the contract: "append-only, descriptive filenames, timestamped"

**Anti-pattern:** Agent writing results inline into code files. Agent committing benchmark results to version control. Agent modifying git history.

---

## Gaps

- No canonical named specification language found for agent contracts (like BDD's Gherkin but for agents). What exists is ad-hoc per project.
- No formal validation framework exists to check agent contracts for contradictions or completeness. Contracts are validated manually or implicitly when agents fail.
- Harness-based evaluation literature is mostly from hyperparameter optimization (Bayesian optimization, Optuna) — not specifically from AI agent co-pilot literature. The pattern exists but isn't formally named in agent co-pilot contexts.

## Clarification questions for the user

1. **Contract format preference** — The DOX hierarchy (AGENTS.md per directory) is one pattern. Would you prefer a single file contract per scope, a nested hierarchy, or a formal spec language (e.g., typed JSON schema with behavioral constraints)?

2. **Harness separation boundary** — In the harness-based evaluation pattern, what exactly is the agent allowed to modify? Just config/params, or also the search strategy code? The line between "what the agent edits" and "what the harness owns" determines the guardrail design.

3. **Cross-session persistence** — Should agent contracts include session state carry-over rules (e.g., what from a previous session is trusted vs re-verified), or should each session start fresh with only the durable artifacts as context?
