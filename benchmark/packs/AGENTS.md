# Frozen Benchmark Packs

## Purpose
Versioned dataset snapshots for the independent Pi Benchmark. Parent: [Benchmark contract](../AGENTS.md).

## Ownership
Benchmark maintainers.

## Local Contracts
- Keep upstream prompts, source IDs and original verifier text traceable through `source.json` hashes and the upstream license.
- `humaneval-base10` contains the first ten HumanEval IDs used by legacy coding-10. It runs original base assertions by calling `check(candidate)`; it is neither HumanEval+ nor the full HumanEval dataset.
- `mbpp-base10` preserves the ten non-contiguous IDs selected by legacy coding-10 and executes the source `assertion` field. It includes original prompting IDs; do not label it a standard MBPP test split or expanded MBPP+ score.
- Oracles live outside task workspaces and are for E2E verification only. Models can access only `solution.py`; hidden assertions and oracle sources never enter model context.
- Frozen task changes alter task hashes. Do not overwrite historical packs or merge their results with legacy coding scores.
- Keep pack text in LF as enforced by `.gitattributes`, so checkout platform does not change the frozen bytes.
- Adding a pack must preserve the default `workspace-smoke` selection and needs an explicit CLI registration.

## Work Guidance
- `../import_coding.py --pack <pack>` reproduces either pinned snapshot into a new directory and refuses to overwrite an existing pack. `../import_humaneval.py` keeps its HumanEval default for compatibility.

## Verification
- Parent `node test/e2e.mjs` verifies every HumanEval/MBPP stub fails and every upstream oracle passes through Pi, hidden pytest and SQLite in a standalone copy.

## Child DOX Index
None; individual datasets remain owned here.
