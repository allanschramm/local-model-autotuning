# AutoResearch Roadmap

Decomposed tasks from codebase audit. Each task is atomic, testable, and independently shippable.

---

## Phase 1: Critical — Tool Calling Pipeline

### 1.1 Fix `system_prefix` dead code
- **File**: `autoresearch/benchmarks/benchmark_harness.py:37`
- **What**: `_run_task_loop` receives `system_prefix` via kwargs but never prepends it to prompt
- **Fix**: Add `prompt = kwargs.pop("system_prefix", "") + prompt` before line 37
- **Impact**: Thinking models (`<|think|>`) get their trigger token. Affects Nexus + Claw.
- **Test**: Verify `system_prefix` appears in prompt sent to `client.complete()`

### 1.2 Rewrite `LlamaClient` to use `/v1/chat/completions`
- **File**: `autoresearch/core/llama_client.py`
- **What**: Currently uses raw `/completion` endpoint which returns no structured `tool_calls`
- **Fix**: Switch to OpenAI-compatible `/v1/chat/completions` with `messages` format
- **Changes**:
  - Endpoint: `/completion` → `/v1/chat/completions`
  - Payload: `{"prompt": str}` → `{"messages": [{"role": "user", "content": str}], "tools": [...], ...}`
  - Response: parse `choices[0].message.tool_calls` properly
  - Keep backward compat for generation params (temp, top_p, etc.)
- **Impact**: Nexus scores from 0.0 → real values. Claw multi-turn works.
- **Prerequisite**: llama-server started with `--jinja` (already in config/search space)
- **Test**: `test_llama_client.py` — mock `/v1/chat/completions` response with tool_calls
- **Note**: evalplus already uses `/v1` successfully, proving endpoint works

### 1.3 Remove temp double-passing in benchmarks
- **Files**: `autoresearch/benchmarks/prepare.py:261`, `autoresearch/benchmarks/prepare_claw.py:171`
- **What**: `harness.evaluate(..., temp=temp, **kwargs)` — `temp` in both explicit and kwargs
- **Fix**: Pop `temp` from kwargs before passing, or pass only via kwargs
- **Impact**: Eliminates duplicate key ambiguity

---

## Phase 2: Critical — Config & Search Space

### 2.1 Remove invalid Flash Attn search values
- **File**: `autoloop.py:41`
- **What**: `"FLASH_ATTN": ["on", "off", "auto"]` — GOLDEN-RULES.md says must be `"on"`
- **Fix**: `"FLASH_ATTN": ["on"]` (single value, effectively removed from search)
- **Impact**: Stops search from exploring forbidden configs

### 2.2 Fix dual `config.py` write
- **File**: `autoloop.py:115`
- **What**: Writes to both root `config.py` (gitignored) and `autoresearch/core/config.py`
- **Fix**: Write only to `autoresearch/core/config.py`
- **Impact**: Eliminates config desync risk

### 2.3 Add `--no-coding` flag
- **File**: `autoresearch/runners/run.py:50`
- **What**: `action="store_true", default=True` — always on, no way to disable
- **Fix**: Use `BooleanOptionalAction` or add `--no-coding` as `store_false`
- **Impact**: Allows Nexus-only or Claw-only trials

---

## Phase 3: Simplify — Remove Duplication

### 3.1 Delete `tune_search.py` (Done)
- **File**: `autoresearch/runners/tune_search.py` (192 lines)
- **What**: Standalone tuner duplicates autoloop.py logic (search space, baseline eval, neighbor loop)
- **Fix**: Delete file. Update imports if any test references it.
- **Test**: Run `tests/test_tune_search.py` — move useful tests to autoloop tests
- **Impact**: -192 lines, single source of truth for search logic

### 3.2 Simplify `run_evaluation` args passing (Done)
- **Files**: `autoresearch/runners/run.py:125-156`, `autoloop.py:119-159`
- **What**: `run_evaluation()` takes 15+ kwargs. `config_to_args()` builds fake Args class (30 attrs). run.py has 25 `isinstance()` checks.
- **Fix**: Make `run_evaluation(cfg: dict, **overrides)` accept config dict directly. Delete `config_to_args()`.
- **Impact**: -80 lines, eliminates fragile isinstance chain
- **Test**: Update `test_run.py` mocks to pass dicts instead of MagicMock args

### 3.3 Extract shared `build_context_padding`
- **Files**: `autoresearch/benchmarks/prepare.py:230-240`, `autoresearch/benchmarks/prepare_claw.py:124-147`
- **What**: Nearly identical padding generators in two files
- **Fix**: Move to `autoresearch/benchmarks/benchmark_harness.py` or new `common.py`
- **Impact**: -20 lines, single source for padding logic
- **Constraint**: program.md says benchmarks/* is fixed — check if extraction counts as "modifying" or just moving

### 3.4 Fix `benchmark_coding.py` import spaghetti (Done)
- **File**: `autoresearch/benchmarks/benchmark_coding.py:17-35, 89-90`
- **What**: Imports `benchmark_search` at module level, sets globals, then resets some at line 89. `parse_args()` mutates globals.
- **Fix**: Import from `config` directly. Remove global mutation. Pass config via args.
- **Impact**: Cleaner imports, no hidden coupling to benchmark_search

---

## Phase 4: Robustness & Quality

### 4.1 Add retry with backoff to server health check (Done)
- **File**: `autoresearch/core/llama_runner.py:269-281`
- **What**: Fixed 100ms sleep, 0.5s timeout, no backoff
- **Fix**: Exponential backoff: 50ms → 100ms → 200ms → 400ms
- **Impact**: More resilient to slow GPU init

### 4.2 Document VRAM estimation constants (Done)
- **File**: `autoresearch/core/llama_runner.py:70-96`
- **What**: Magic numbers: 80KB/token f16, 300MB overhead, quant factors
- **Fix**: Extract to named constants with docstring explaining calibration
- **Impact**: Readability, maintainability

### 4.3 Don't inject metadata keys into config dicts (Done)
- **File**: `autoresearch/core/search.py:33-35`
- **What**: `get_neighbors()` adds `_changed`, `_old`, `_new` to config dicts
- **Fix**: Return `(config, metadata)` namedtuple or dataclass instead
- **Impact**: Eliminates fragile `.pop("_changed")` calls in autoloop.py:370-372

### 4.4 Coding benchmark TPS = 0
- **File**: `autoresearch/benchmarks/benchmark_coding.py:224`
- **What**: `avg_tps=0.0` always — no throughput measurement for coding
- **Fix**: Parse evalplus stdout for timing info, or add token counting wrapper
- **Impact**: Coding-only trials get TPS data

### 4.5 Avoid reinstalling evalplus every run (Done)
- **File**: `autoresearch/benchmarks/benchmark_coding.py:117-118`
- **What**: `uv run --with evalplus` reinstalls every subprocess call (2x per trial)
- **Fix**: Check `import evalplus` first, use `uv run` only as fallback
- **Impact**: Faster coding benchmark execution

---

## Phase 5: Cleanup

### 5.1 Delete dead scripts (Done)
- **Files**: `scripts/robust_run.py` (references non-existent `run_grid.py`), `scripts/test_imports.py` (debug artifact)
- **Fix**: Delete both

### 5.2 Normalize line endings (Done)
- **What**: Mixed `\r\n` (run.py, llama_runner.py) and `\n` (search.py, prepare.py)
- **Fix**: Normalize to `\n`. Add `.gitattributes: *.py text eol=lf`

### 5.3 Add `coding_results/` cleanup (Done)
- **What**: EvalPlus generates files indefinitely in `coding_results/`
- **Fix**: Add cleanup on startup or gitignore pattern (partially there)

---

## Execution Order

```
Phase 1 (unblocks Nexus/Claw):  1.1 → 1.2 → 1.3
Phase 2 (config correctness):   2.1 → 2.2 → 2.3
Phase 3 (code reduction):       3.1 → 3.2 → 3.4 → 3.3
Phase 4 (robustness):           4.1 → 4.2 → 4.3 → 4.4 → 4.5
Phase 5 (cleanup):              5.1 → 5.2 → 5.3
```

## Impact Summary

| Phase | Lines Saved | Bugs Fixed | Perf Wins |
|-------|------------|------------|-----------|
| 1     | ~10        | 3 critical | 1 (Nexus unblocked) |
| 2     | ~5         | 3 high     | 0 |
| 3     | ~250       | 0          | 0 |
| 4     | ~10        | 0          | 3 |
| 5     | ~70        | 0          | 1 |
| **Total** | **~345** | **6** | **5** |
