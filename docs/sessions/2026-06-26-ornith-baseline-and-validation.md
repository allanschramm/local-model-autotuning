# Session Log: Ornith-1.0-9B Baseline & Fast Validation Mode

**Date**: 2026-06-26
**Focus**: Establishing a reliable baseline for the new model `Ornith-1.0-9B` and creating a fast validation mode to test new models' VRAM and TPS.

---

## 1. Zero-Score Diagnoses & Bug Fixes

Prior attempts to evaluate `Ornith-1.0-9B` on the 4-bench composite returned `0.0000` scores on HumanEval+ and BigCodeBench. In this session, we systematically diagnosed and patched the following execution pipeline issues:

1. **Stop Sequence Generation Cuts**: 
   - **Issue:** The reasoning model frequently outputs thinking tokens like `"Task: "` inside its `<think>` blocks. The default `stop` sequence list in [llama_client.py](file:///home/shark/workspace/Nexus-System/local-model-autotuning/autoresearch/core/llama_client.py) included `"Task:"`, leading to premature generation halts.
   - **Fix:** Removed `"Task:"` from the default `stop` list in [llama_client.py](file:///home/shark/workspace/Nexus-System/local-model-autotuning/autoresearch/core/llama_client.py).
2. **Thinking block response merging**: 
   - **Issue:** The runner concatenated `content` and `reasoning_content` blindly, appending plain English thoughts directly into the Python code, corrupting executable block parses.
   - **Fix:** In [benchmark_coding.py](file:///home/shark/workspace/Nexus-System/local-model-autotuning/autoresearch/benchmarks/benchmark_coding.py#L487-L495), updated response combining logic to fall back to `reasoning_content` *only* if the main `content` field is empty.
3. **Missing Function Signatures**: 
   - **Issue:** As instruction-followers, the models generated *only* the function bodies. The test runner evaluated them as-is without prepending the signature definition, causing `NameError` on evaluation.
   - **Fix:** Prepended signature prompts (`entry.get("prompt", "")`) to the extracted function body when signature headers (`def ...`) are missing.
4. **Indentation Mismatches in `_strip_code`**: 
   - **Issue:** The code block parser called `.strip()` directly. This stripped the leading whitespace of the *first* line of code (destroying its 4-space indentation) but left the rest of the body indented, causing an `IndentationError` when prepended with the signature.
   - **Fix:** Created a helper `_strip_empty_lines` that strips leading/trailing empty lines from code block text but preserves horizontal space indentation. Replaced `.strip()` calls in `_strip_code` with this helper.

---

## 2. Ornith-1.0-9B Verified Baseline (10 Tasks)

After applying the fixes, we executed a complete baseline sweep with exactly 10 tasks per dataset. The model achieved a new repository record:

- **Model**: `ornith-1.0-9b-Q4_K_M.gguf` (Offloaded to GPU with `NGL=99`)
- **Coding Score**: **`0.4800`** (Up from previous best `0.3800`)
  - **HumanEval+**: `0.4000` (up from `0.0000`!)
  - **MBPP+**: `0.9000`
  - **LiveCodeBench**: `0.4000`
  - **BigCodeBench Hard**: `0.1000`
- **Peak VRAM**: `7.9 GB`
- **TPS**: `49.4`
- **Status**: `KEEP` (committed locally to `main`)

---

## 3. Fast Validation Mode (`--validation`)

To enable quick profiling of VRAM usage and TPS for newly downloaded models without running full 10-task evaluations (which can take 5+ minutes), we introduced a **Validation Mode**:

- **Command flag**: `--validation`
- **Behavior**: Overrides task limits for all coding datasets to exactly 2 tasks.
- **Run isolation**: Results are automatically written to `results.tsv` under status `discard` with a `[validation]` description prefix. This ensures validation scores are ignored for previous-best KEEP checks.
- **Determinism**: Slicing logic is fully deterministic (takes the first 2 tasks sequentially), ensuring fair future comparisons.

### Execution Command:
```bash
python3 benchmark_search.py --desc "validation run test" --validation
```

---

## 4. Test Suite Sanity

- Added unit tests in [test_run.py](file:///home/shark/workspace/Nexus-System/local-model-autotuning/tests/test_run.py) and [test_benchmark_coding.py](file:///home/shark/workspace/Nexus-System/local-model-autotuning/tests/test_benchmark_coding.py) verifying the validation task overrides and indentation stripping fixes.
- Verified all 89 unit tests pass.
- All code changes are committed offline to the local repository. No changes pushed to remote repositories per repository rules.
