# Rust native core — `autoresearch-core`

Native implementation of the CPU-bound hot-path modules of
`ailocal-model-autotuning`:

| Module    | Python source                         | LOC  | Surface |
|-----------|---------------------------------------|-----:|---------|
| fingerprint | `autoresearch/core/fingerprint.py`  |  300 | SHA‑256 over JSON + 6 scrub rules (Windows/POSIX/UNC paths, e‑mail, URL, GPU SKU, hostname, file extension) |
| state       | `autoresearch/core/state.py`        |   93 | Atomic JSON persistence of visited memory + Morris pin dictionary |
| pareto      | `autoresearch/core/pareto.py`       |   93 | SHA‑256 fingerprint, 4‑axis dominance, Pareto set, merge |

All three are exposed via PyO3 (`pymodule autoresearch_core`) so the Python
side keeps its existing import surface (`from autoresearch.core.pareto import
…`). The shim in `autoresearch/core/<module>.py` re-exports every symbol from
the binding.

The binary `autoresearch-loop` and the port of `llama_runner.py` are
**deferred** to Phase 1 (deferred decision T1 of the carry-over).

## Build

Prerequisites:
- Rust 1.82+ (toolchain auto-installed via `rust-toolchain.toml`)
- `maturin >= 1.7` (install with `pip install maturin`)
- Python 3.11+ virtual environment (see top-level `README.md`)

```powershell
# 1. Install maturin into the venv (one-time)
.\venv\Scripts\python.exe -m pip install maturin

# 2. Develop build (debug or release) — installs the wheel into the active venv.
#    IMPORTANT: cwd must point at the subcrate, not at the cargo workspace
#    root. maturin 1.15 chokes on a workspace-only Cargo.toml (no [package]
#    field) and reports a TOML parse error. Use the subcrate as cwd:
cd rust\autoresearch-core
..\..\venv\Scripts\python.exe -m maturin develop --release
cd ..\..

# 3. Run the unit tests (Rust + parity)
cd rust
cargo test

# 4. Run the original Python suite — must remain green
.\venv\Scripts\python.exe -m pytest tests\test_pareto.py tests\test_state.py tests\test_fingerprint.py tests\test_fingerprint_apply.py tests\test_fingerprint_trial.py tests\test_crash_journal.py tests\test_classify.py tests\test_recompute.py tests\test_autoloop.py
```

Linux / macOS:

```bash
python3 -m pip install maturin
cd rust/autoresearch-core
python3 -m maturin develop --release
cd ../..
cargo test --manifest-path rust/Cargo.toml
```

## Layout

```
rust/
├── Cargo.toml                    # workspace root
├── rust-toolchain.toml           # channel = "stable"
└── autoresearch-core/
    ├── Cargo.toml                # lib (cdylib + rlib), three [[bench]] targets
    ├── pyproject.toml            # maturin backend
    ├── src/
    │   ├── lib.rs                # native facade + pymodule init
    │   ├── fingerprint/          # 6 scrub rules + dump/load/apply/mismatch
    │   ├── state/                # atomic JSON store + baseline Python bridge
    │   ├── pareto/               # fingerprint hash + dominance + set + merge
    │   ├── error.rs              # FingerprintError + PyO3 exception mapping
    │   └── py/                   # #[pymodule] autoresearch_core + bindings
    ├── benches/                  # criterion isolated speedup measurement
    └── tests/                    # unit + parity golden tests
```

## Acceptance gates (Phase 0)

* `cargo fmt --check`
* `cargo clippy --all-targets -- -D warnings`
* `cargo test` (unit + parity golden)
* `maturin develop --release` no warnings
* `pytest tests/…` original Python suite stays 100 % green

Targets (Criterion isolated):

| Bench                              | Target  |
|------------------------------------|---------|
| `fingerprint.mismatch_reason` 10k | ≥ 3×   |
| `pareto.fingerprint_hash` 10k     | ≥ 5×   |
| `state.persist` 1k                | ≥ 1× (no regression) |
| `pareto.compute_pareto` 10k       | ≥ 5×   |

## Development shell session

```powershell
# Activate venv
.\venv\Scripts\Activate.ps1

# Build wheel only (without installing)
maturin build --release --manifest-path rust\autoresearch-core\Cargo.toml

# Bench locally
cargo bench --manifest-path rust\Cargo.toml --bench fingerprint
```

## See also

* Plan: `…/artifacts/plan.md` (current session)
* Wiki: `Obsidian Vault/wiki/entities/Rust-Refactoring-Strategy.md`
* AGENTS.md — DOX hierarchy and per-folder contracts.
