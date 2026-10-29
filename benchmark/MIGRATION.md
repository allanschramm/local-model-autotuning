# Evaluation migration

Runtime optimization belongs to `autoresearch/tuning.py`; quality evaluation belongs here. A local model endpoint is the primary validation target. Cloud validation is deferred. Both modules remain in this repository until the operator moves it to the AILOCAL organization.

## Existing evaluations and next owners

| Evaluation | Existing entrypoint and tasks | Migration requirements | Status |
| --- | --- | --- | --- |
| SWE-lite | `autoresearch/benchmarks/agentic_coding/runner.py`; five frozen workspaces | Same task bytes and hidden pytest; Pi protocol differs from the old tool loop | Available as default `workspace-smoke` |
| HumanEval slice | `benchmark_coding.py::EvalplusTask`; first ten HumanEval IDs | Preserve original prompts/base assertions; invoke `check(candidate)`; record source/version/hashes; keep oracle inaccessible | Available as `humaneval-base10` |
| MBPP slice | `benchmark_coding.py::EvalplusTask`; first ten MBPP IDs | Preserve non-contiguous IDs, module prompt/example, entrypoint and literal base `assertion` text | Available as `mbpp-base10` |
| LiveCodeBench v6 | `benchmark_coding.py::LiveCodeBenchTask`; ten selected problems | Preserve statement and public/private stdin/stdout cases; review encoded private-case deserialization | Legacy only |
| BigCodeBench Hard | `benchmark_coding.py::BigCodeBenchTask`; ten selected library tasks | Preserve unittest invocation, package requirements and isolated working directory | Legacy only |
| Claw quick/full | `agentic_benchmarks.py` and `agentic_runner.py`; 5/15 selected tasks | Preserve upstream mock services, tool schemas, scoring components and service lifecycle; needs a separate Pi protocol | Legacy only |
| DM-Code-Agent | `mini_swe_agent/runner.py`; 15 coding + 15 maintenance tasks | Preserve pinned revision, setup/hidden files, visible/hidden commands, step limits and allowed/required changed files; pinned package must be installed | Legacy only |

The first ten HumanEval IDs are the HumanEval portion of coding-10, not ten tasks spread across its four datasets. `humaneval-base10` is a subset using the original base assertions; neither the 164-task HumanEval aggregate nor expanded HumanEval+ is measured.

## Verifier correction and score identity

The legacy HumanEval path returns the dataset's `test` text and concatenates it after model code. In the inspected snapshot that text defines `check(candidate)` but never calls it. Merely defining that function can exit successfully even for an incorrect implementation. It also does not execute the dataset's `plus_input` cases.

The migrated hidden pytest keeps the original assertion text and explicitly calls `check` on the function imported from `solution.py`. Every unimplemented stub must fail and every upstream canonical solution must pass in the standalone E2E. Oracle sources are outside the model's allowlisted workspace.

MBPP's inspected legacy path already executes its `assertion` field; it does not execute `plus_input`. The migrated pack imports the same entrypoint from `solution.py` and wraps those literal assertions in pytest. Its first ten IDs are not contiguous and include original prompting examples, so label it a legacy-selected base smoke subset, not a standard test-split score. The same fail-stub/pass-oracle E2E applies to all ten tasks.

Preserved dataset limitation: Mbpp/2's public prompt compares sets, while its hidden base assertions require a literal tuple. The pack does not normalize that mismatch. Keep task-failure interpretation distinct from model-runtime or harness failures.

New results retain `pi-workspace-v2` because the Pi tools, prompts and budgets are unchanged. Run configuration identifies the pack; task hashes identify its exact prompts and verifiers. Do not pool these results with legacy HumanEval columns or claim retrospective validation of historical coding scores. Existing SQLite rows are not rewritten.

## Compatibility and retirement

`autoresearch/runners/evaluation.py`, CLI dispatch in `autoresearch/runners/run.py`, compatibility exports in `autoresearch/benchmarks/__init__.py`, `scripts/lcb_only.py`, existing tests, dashboard consumers and legacy rankings still depend on the old paths. They remain available during migration. The independent runner never imports them, and the tuning module never invokes Pi.

Before retiring a legacy path: port all selected tasks and verifier behavior, publish its protocol identity, run a local endpoint end to end with SQLite/trajectory evidence, and check every caller plus the commands documented for operators. Historical result tables/protocols remain readable. A passing small pack alone does not authorize deleting all old evaluations.

## Open-source approaches considered

- [EvalPlus data loader](https://github.com/evalplus/evalplus/blob/master/evalplus/data/humaneval.py) distinguishes original `test`, `base_input`, expanded `plus_input`, and oracle solutions. Use the existing pinned snapshot; do not relabel its base tests as plus coverage.
- [Harbor HumanEvalFix adapter](https://github.com/harbor-framework/harbor/tree/main/adapters/humanevalfix) demonstrates dataset-to-workspace adaptation with separate verification and oracle assets. Its repair dataset and container runtime differ from our existing generation tasks; adopting its whole execution stack is unnecessary for this slice.
- [EvalPlus MBPP loader](https://github.com/evalplus/evalplus/blob/master/evalplus/data/mbpp.py) preserves source IDs and deserializes inputs; [EvoAgentX's MBPP adapter](https://github.com/ANative-Lab/EvoAgentX/blob/main/evoagentx/benchmark/mbpp.py) converts raw MBPP to HumanEval-shaped prompts and checks. Our legacy already consumes EvalPlus's normalized snapshot, so reuse that exact data and the existing Pi workspace runner without another dataset remapping.
- [Pi SDK 1.4.0](https://github.com/earendil-works/pi/blob/v1.4.0/packages/coding-agent/docs/sdk.md) supplies the existing agent loop and file tools. Reuse that loop with a frozen pack rather than adding a second agent implementation.

The pack includes the original [HumanEval MIT license](https://github.com/openai/human-eval/blob/master/LICENSE). Frozen provenance and reproduction instructions are in [README.md](README.md).
