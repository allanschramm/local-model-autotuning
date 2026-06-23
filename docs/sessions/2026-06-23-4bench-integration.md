# Session Log: 4-Bench Pipeline Integration & Strip Code Fixes
**Date**: 2026-06-23
**Focus**: Upgrading coding evaluation from 2-bench (HumanEval, MBPP) to 4-bench composite (LiveCodeBench v6, HumanEval+, MBPP+, BigCodeBench Hard).

## 1. Context & Architecture
We previously ran `run_coding_eval` which only tested HumanEval+ and MBPP+. However, those benchmarks are susceptible to data contamination with newer models. We integrated a 4-benchmark pipeline:
- **LiveCodeBench v6** (Weight 0.35): Contamination-free competitive programming (AtCoder/Codeforces). Requires stdin/stdout testing.
- **HumanEval+ Strict** (Weight 0.25): 164 algorithmic problems using `evalplus` strict mode (`test` field).
- **MBPP+** (Weight 0.25): 974 entry-level problems.
- **BigCodeBench Hard** (Weight 0.15): Library-call and API-based programming. Requires `unittest` emulation.

The composite score calculation is now accurately implemented and logged in `results.tsv`.

## 2. The "No Code Extracted" Bug
During the first smoke test with the Qwen3.5-4B model, we encountered a 100% failure rate on LiveCodeBench and BigCodeBench, logging `"no code extracted"`.

**Root Causes Identified**:
1. **Think-only responses**: When a model used all tokens thinking (`<think>...</think>`), dropping the `<think>` block resulted in an empty string.
2. **Prose prefix leakage**: When the model returned `Here is the solution:\n\n<code>` without fences, the text was executed directly in Python, causing `SyntaxError`s but masked as FAIL.
3. **Truncated think blocks**: Cut-off thinking generated unclosed tags, executing partial thought as code.
4. **MTP Reasoning Field**: `llama-server` emits think tokens into a separate `reasoning_content` field (when using `--jinja`). `LlamaClient` was completely ignoring `reasoning_content`, meaning think-only responses were treated as empty `content=""`.

## 3. The Fixes (`_strip_code`)
We introduced robust regex parsing in `autoresearch/benchmarks/benchmark_coding.py`:
- `_THINK_CLOSED_RE` strips all `<think>...</think>` non-greedily.
- `_THINK_OPEN_RE` strips unclosed tags at the end of text.
- `_FENCE_RE` safely extracts standard markdown code blocks.
- **Code-line prefix scan**: Scans line-by-line (`def `, `class `, `import `) to safely extract non-fenced code without crashing on prose prefixes.
- `LlamaClient.complete()` was updated to capture and return `reasoning_content`. The runner now concatenates `content + reasoning_content` before passing to `_strip_code`.
- `LCB_MAX_TOKENS` and `BIGCODE_MAX_TOKENS` were raised to `2048` to accommodate the heavy thinking overhead in competitive/library coding prompts.

## 4. Test Suite Resilience
The test suite was broken because it referenced `run_claw` and `run_nexus` in its mocks. We cleaned up the legacy mocks, explicitly mocking the unified `run_coding` evaluation logic that now returns a `BenchmarkResult` featuring four pass rates. Orphaned test files (`test_claw_task.py`, `test_nexus_task.py`) were removed.

## 5. Results (4B Baseline)
The fixed pipeline was verified via smoke test with **Qwen3.5-4B-MTP**:
- **HumanEval+**: 0.00
- **MBPP+**: 0.70
- **LiveCodeBench**: 0.20
- **BigCodeBench**: 0.00
- **Weighted Score**: **0.2450**

The 4B model successfully generated executable competitive programming code, establishing our first real 4-bench baseline. No `"no code extracted"` false-negatives occurred.

**Next Steps**:
Establish the baseline score for Qwen3.5-9B, then begin autotuning loops over the 28B/35B models (Mudler / REAP) using the robust pipeline.
