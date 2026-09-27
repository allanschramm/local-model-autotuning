# 2026-09-27 — Rust Phase 0: Criterion speedup + PyO3 marshalling honest finding

> Phase 0 closure session. Captures per-call speedup of the Rust kernel
> vs CPython for the three ported CPU-bound modules
> (`autoresearch.core.{pareto,state,fingerprint}` →
> `autoresearch_core.{pareto,state,fingerprint}` via PyO3 0.22 + maturin
> 1.15) and the honest measurement of PyO3 dict↔serde_json::Value
> marshalling overhead against CPython. Verifies the Phase-0 acceptance
> end-to-end on the operator host before Sprint 5 (CI) and Sprint 6
> (handoff + report) close-out.

## Goal

1. Confirm the Rust kernel is faster than CPython for the three hot paths
   in isolation (no GIL, no PyO3 overhead).
2. Confirm `smoke_bindings.py`, `smoke_fp.py`, `smoke_inspect.py` and
   the 67-pytest Phase-0 suite all stay green with the new
   `autoresearch_core` wheel installed.
3. Document the wall-clock speedup (or lack thereof) when going through
   PyO3, to justify Phase-1 architecture (`autoresearch-loop` binary
   bypassing PyO3 for the trial loop, keeping PyO3 only for surfaces
   the autoloop / harness must call).
4. Provide evidence for the wiki entity `Rust-Refactoring-Strategy` and
   ADR 0018 acceptance criteria.

## Hardware

- **Class:** discrete_gpu host. Phase 0 does not load llama.cpp; the
  Rust kernel ports are CPU-only and were timed on the operator host's
  general-purpose cores.
- **VRAM_LIMIT_MB:** n/a (Phase 0 has no engine context).
- **OS family:** Windows.
- **llama.cpp engine/tag:** n/a for Phase 0 timing (the Python autoloop
  binary path that *consumes* these ports is unchanged; Phase 1 will
  re-time against real engines).
- **Toolchain:** Rust stable from `rust-toolchain.toml` (MSRV 1.82);
  Python 3.12 (CI matrix target) / 3.11 (abi3-py311 wheel); maturin 1.15.

## Setup

- Working tree branch `allanschramm/rust`.
- One-time maturin install per fresh venv:
  `.\.venv\Scripts\python.exe -m pip install maturin==1.15`
- Develop build (re-installs the wheel into the active venv):
  `cd rust && ..\venv\Scripts\python.exe -m maturin develop --release`
- The `autoresearch_core` module is registered as a `#[pymodule]` with
  three `add_submodule` children — `pareto`, `state`, `fingerprint` —
  so `from autoresearch_core.pareto import …` resolves without a
  Python `__init__.py`.

## Commands (reproducible)

```powershell
# Rust kernel: isolated speedup.
cd rust
cargo test                                          # 50/50 unit + parity
cargo bench --bench pareto -- --quick               # Criterion pareto
cargo bench --bench fingerprint -- --quick          # Criterion fingerprint
cargo bench --bench state -- --quick                # Criterion state

# Python-side parity sweep (Rust vs hand-rolled canonical JSON).
.\venv\Scripts\python.exe scripts\bench_rust_vs_python.py
.\venv\Scripts\python.exe scripts\smoke_inspect.py    # surface sanity
.\venv\Scripts\python.exe scripts\smoke_fp.py         # fingerprint surface
.\venv\Scripts\python.exe scripts\smoke_bindings.py   # pareto surface + 100-config parity

# Acceptance suite (must remain 100 % green).
.\venv\Scripts\python.exe -m pytest tests\test_pareto.py tests\test_state.py tests\test_fingerprint.py tests\test_fingerprint_apply.py tests\test_fingerprint_trial.py tests\test_classify.py
```

## Findings

### Rust kernel — isolated (Criterion, sample_size=50, --quick)

| Operation | Per-call | Notes |
|---|---:|---|
| `pareto::dominates` | ~234 ps | 4-axis strict dominance, `Option<f64>` per axis |
| `pareto::fingerprint_hash` | 3.76 µs | SHA-256 over canonical JSON (`{"engine": …, "sampler": …}`, dict shape, sorted keys, compact separators) |
| `pareto::pareto_set` (1 000 vectors) | 239 µs | preserves insertion order of non-dominated subset |
| `pareto::merge` (500 trials) | 42.9 µs | per-fingerprint merge, max-of-each-axis |

All four are kernel-only numbers; no GIL, no Python interpreter, no
PyO3 marshalling. They establish the lower bound on what the Rust
binary can deliver in Phase 1 when the trial loop bypasses PyO3.

### Python parity (the part that mattered for Phase 0 closure)

- `smoke_bindings.py`: 100/100 SHA-256 hex digests identical between
  Rust `fingerprint()` and `hashlib.sha256(canonical_json(...))`. The
  canonical payload shape is `{"engine": …, "sampler": …}` (dict), per
  ADR 0018 §Strategies #5 (a prior smoke draft compared against
  `[engine, sampler]` list shape, which never bit — fixed in this
  session).
- `smoke_fp.py`: `dump`, `load`, `mismatch_reason`, `path_for`,
  `FingerprintError`, `SCHEMA_VERSION` all importable and present on
  `autoresearch_core.fingerprint`.
- `smoke_inspect.py`: the `pareto`/`state`/`fingerprint` submodules
  register into `sys.modules` correctly.
- **67/67 pytest** in the Phase-0 suite (`-q` summary:
  `67 passed in 5.30s`). The Python shims in
  `autoresearch/core/{pareto,state,fingerprint}.py` continue to
  re-export 100 % of the names `autoloop.py` and the rest of the
  harness import — zero-ruptura contract honored.

### PyO3 marshalling — honest finding

`scripts/bench_rust_vs_python.py` (also committed in `faeea9c`)
measures wall-clock for paths that *cross* the PyO3 boundary
(`pyo3::types::PyDict` ↔ `serde_json::Value` marshalling, plus GIL
acquire/release per call).

For microsecond-class operations like `fingerprint_hash` the
overhead caps wall-clock speedup at **~1.5× vs CPython** — the
kernel is faster, but the marshalling nullifies most of the win at
this layer. This is **not** a regression; it is the documented
trade-off of PyO3 for Python-side hot paths.

The architecture decision for Phase 1 follows directly: keep the
PyO3 binding for **consumer-side** surfaces the autoloop must call,
but route the **trial-loop hot path** (`pareto.compute`,
`fingerprint.mismatch_reason`, `state.write_atomic`, the per-neighbor
Pareto recompute) through a dedicated `autoresearch-loop` binary that
calls the Rust kernel directly with no Python interpreter in the loop.
This unlocks:

- no PyO3 / no GIL in the hot loop (full kernel speedup is observable)
- real parallelism (`rayon` over Pareto fronts, `tokio` for spawn/
  health-check of `llama-server`)
- single-binary distribution (one `autoresearch-loop.exe` per target)

## Errors

1. **`scripts/smoke_bindings.py` written against wrong contract** (found
   in this session, fixed):
   - `v.complete()` (method call) → `v.complete` (Python `@property`
     returns bool; the Rust `#[getter]` mirrors that). The dataclass
     Python original `autoresearch/core/pareto.py` re-exported via
     shim never exposed `complete()` as callable; the binding is
     correct, the smoke was wrong.
   - `Trial("m", "00", 1024, 50, 0.5, 0.5)` (6 positional args) →
     `Trial("m", ObjectiveVector(...))` (canonical 2-arg form: `fp`
     + `ObjectiveVector`).
   - Hard-coded `[engine, sampler]` list shape for the parity sweep
     → dict shape `{"engine": engine, "sampler": sampler}` to match
     ADR 0018 §Strategies #5.
   - Importing `autoresearch.core.pareto` for "parity" comparison
     actually compared Rust → Rust via the re-export shim, which is
     trivially correct but says nothing about parity. Replaced with
     hand-rolled `hashlib.sha256(canonical_json(...))` baseline so
     the parity sweep is independent of the `autoresearch` package
     being importable.

   Rewrite landed; `smoke_bindings.py` now ends in `ALL OK` with
   `100/100 hashes identical`.

## Decisions

- **Phase 0 closed.** All 4 sprints completed (0.1 bootstrap, 0.2
  fingerprint, 0.3 state, 0.4 pareto) and acceptance verified locally
  (50/50 cargo + 67/67 pytest + 100/100 parity-sweep hashes).
- **Sprint 5 (CI) and Sprint 6 (handoff + speedup report) finish in
  this session** to leave Phase 0 fully closed before opening Phase 1.
- **T1 (`llama_runner.py` 1323 LOC)** stays open; the Phase-1
  architecture (`autoresearch-loop` binary + PyO3 surface binding)
  follows from this measurement, not from T1 alone.
- **T2 / T3 / T6 / T7** stay open and queue for Phase 1 sprint kickoff.
- **Smoke scripts commit-ready.** `scripts/smoke_{inspect,fp,bindings}.py`
  are durable E2E evidence surfaces; they will move to CI
  (`.github/workflows/rust-ci.yml`) once Allan merges.
