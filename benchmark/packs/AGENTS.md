# Frozen Benchmark Packs

## Purpose
Versioned dataset snapshots for the independent Pi Benchmark. Parent: [Benchmark contract](../AGENTS.md).

## Ownership
Benchmark maintainers.

## Local Contracts
- Keep upstream prompts, source IDs and original verifier text traceable through `source.json` hashes and the upstream license.
- `humaneval-base10` contains the first ten HumanEval IDs used by legacy coding-10. It runs original base assertions by calling `check(candidate)`; it is neither HumanEval+ nor the full HumanEval dataset.
- `mbpp-base10` preserves the ten non-contiguous IDs selected by legacy coding-10 and executes the source `assertion` field. It includes original prompting IDs; do not label it a standard MBPP test split or expanded MBPP+ score.
- `lcb-v6-stdin10` preserves source order and all 400 private stdin cases. Its strict newline-only comparison is legacy-compatible, not the official full-dataset grader. Public examples stay in prompts. The dataset has no gold solutions; do not invent oracle provenance.
- `bigcode-hard10` freezes the first ten v0.1.4 Hard instruct tasks. Preserve unittest globals, mocked services and full TestCases invocation; oracle code stays outside workspaces. Install its requirements in the selected verifier environment.
- Coding oracles live outside task workspaces and are for E2E verification only. Those models access only `solution.py`; hidden assertions and oracle sources never enter model context.
- `claw-tiers` freezes 15 legacy-selected tasks and their upstream mock apps/fixtures at a recorded checkout commit. `source.json` records every copied hash plus the extracted legacy completion scorer hash. Treat `upstream/` as read-only source material; change adapters outside it. Claw models see original prompt/tool schemas, never task YAML, fixtures, rubric or reference solution.
- `dm-code30` preserves the pinned 30-task source order, setup and hidden file contents, both verifier commands, max steps, tags and scope constraints. Only setup files enter model workspaces. Its four reference solutions come from pinned upstream tests and stay outside tool access; they do not cover every task. `upstream/` is read-only provenance, not a runtime dependency.
- Frozen task changes alter task hashes. Do not overwrite historical packs or merge their results with legacy coding scores.
- Keep pack text in LF as enforced by `.gitattributes`, so checkout platform does not change the frozen bytes.
- Gzip case archives are binary (`*.gz -text`), with deterministic headers and recorded compressed/logical hashes.
- Adding a pack must preserve the default `workspace-smoke` selection and needs an explicit CLI registration.

## Work Guidance
- `../import_coding.py --pack <pack>` reproduces either pinned snapshot into a new directory and refuses to overwrite an existing pack. `../import_humaneval.py` keeps its HumanEval default for compatibility.
- `../import_lcb.py --source <test6.jsonl>` requires the pinned source hash before parsing, validates literal-only serialization and stdin schemas, and refuses overwrite. `../stdin_verifier.py` is copied into each frozen task; editing the template does not mutate existing packs.
- `../import_bigcode.py --source <v0.1.4-parquet>` checks the selected-row hash/order and refuses overwrite. `../bigcode_verifier.py` is copied into each task; keep original tests separate and unchanged.
- `../import_claw.py --source <claw-checkout> --legacy-scorer <agentic_runner.py>` reads pinned Git objects, checks the scorer hash and refuses overwrite. Its frozen selection is five quick tasks and 15 full tasks; five full tasks use live web, not fixtures.
- `../import_dm.py --source <four-source-file-directory>` checks whole-file hashes and the full 30-task contract digest before executing the stdlib-only data modules. It refuses overwrite and needs neither `dm_agent` nor mini-swe-agent installed. See the parent README for pinned download URLs.

## Verification
- Parent `node test/e2e.mjs` verifies every HumanEval/MBPP stub fails and every upstream oracle passes through Pi, hidden pytest and SQLite in a standalone copy.
- The same E2E covers stdin stubs, a known first-task solution, sample-only and formatting errors, and missing private archives. `python test/lcb_import_security.py` checks malformed/executable serialization rejection. Actual full-pack model quality is separate from these checks.
- BigCodeBench E2E requires the pack's Python dependencies; every stub must fail, every upstream oracle must pass, and a missing dependency must produce `passed=NULL`.
- Parent `test/claw_e2e.mjs` verifies actual offline mock API execution with scripted Pi responses. These are harness controls, not gold solutions or model-quality scores. No live-web execution occurs in that check.
- Parent `test/dm_e2e.mjs` verifies all initial states fail, the four upstream reference controls pass and correct implementations still fail forbidden/missing required edits. Public tests run before hidden staging; syntax/policy failures and missing fixtures retain distinct outcomes.

## Child DOX Index
None; individual datasets remain owned here.
