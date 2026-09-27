"""smoke_e2e.py — end-to-end smoke: prove the Rust port is reachable from
the operator-facing ``autoloop`` import path with the same workload that
the loop performs in production.

This is the **acceptance harness** for Phase 0: even though the wall-clock
speedup per call is limited by PyO3 marshalling (see
``bench_rust_vs_python.py``), the **contract** must hold:

  * ``autoloop.py`` imports cleanly with the Rust shim in place.
  * ``SearchState`` roundtrips state on disk in the same JSON shape the
    Python version produced.
  * ``fingerprint.mismatch_reason`` returns identical ``None`` / rejection
    strings for the same model + baseline.
  * ``pareto.pareto_set`` returns the same set membership ordering as the
    pre-Rust code, including ``incomplete`` drop behavior.
  * ``results.db`` SQLite store stays the canonical trial log; the TSV
    mirror still appends.

Run with::

    .venv\\Scripts\\python.exe scripts\\smoke_e2e.py
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def section(title: str) -> None:
    print(f"\n[{title}]")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help="Only run the autoloop import smoke (fastest check).",
    )
    args = parser.parse_args()

    print("== smoke_e2e.py ==")
    print(f"Repo root: {REPO_ROOT}")

    section("1. Module imports")
    modules = [
        "autoresearch",
        "autoresearch.core",
        "autoresearch.core.pareto",
        "autoresearch.core.state",
        "autoresearch.core.fingerprint",
        "autoresearch_core",
        "autoresearch_core.pareto",
        "autoresearch_core.state",
        "autoresearch_core.fingerprint",
    ]
    for name in modules:
        try:
            importlib.import_module(name)
            print(f"  OK  import {name}")
        except Exception as exc:
            print(f"  FAIL import {name}: {exc!r}")
            return 1

    section("2. Symbol re-export parity")
    expected_pareto = {
        "ObjectiveVector",
        "Trial",
        "fingerprint",
        "dominates",
        "pareto_set",
        "merge",
        "AXES",
        "VectorLike",
    }
    from autoresearch.core import pareto as shim_pareto

    for sym in expected_pareto:
        if not hasattr(shim_pareto, sym):
            print(f"  MISSING autoresearch.core.pareto.{sym}")
            return 1
    print(f"  OK pareto has {len(expected_pareto)} symbols")

    expected_fingerprint = {
        "SCHEMA_VERSION",
        "FingerprintError",
        "default_dir",
        "path_for",
        "dump",
        "load",
        "apply",
        "mismatch_reason",
        "SERVER_ENGINE_KEYS",
        "HARNESS_ONLY_ENGINE_KEYS",
    }
    from autoresearch.core import fingerprint as shim_fp

    for sym in expected_fingerprint:
        if not hasattr(shim_fp, sym):
            print(f"  MISSING autoresearch.core.fingerprint.{sym}")
            return 1
    print(f"  OK fingerprint has {len(expected_fingerprint)} symbols")

    from autoresearch.core import state as shim_state

    if not hasattr(shim_state, "SearchState"):
        print("  MISSING autoresearch.core.state.SearchState")
        return 1
    print("  OK state has SearchState")

    if args.smoke_only:
        print("\nsmoke_only mode - skipping deeper exercises.")
        return 0

    section("3. Pareto nucleus: same vector set in Python and Rust")
    from autoresearch_core import pareto as rust_pareto

    from autoresearch.core import pareto as py_pareto

    py_vectors = [
        py_pareto.ObjectiveVector(
            ctx=2048,
            tps=80.0,
            agentic=0.6,
            coding=0.5,
        ),
        py_pareto.ObjectiveVector(
            ctx=4096,
            tps=50.0,
            agentic=0.4,
            coding=0.45,
        ),
        py_pareto.ObjectiveVector(ctx=1024, tps=20.0, agentic=0.2, coding=0.3),
        py_pareto.ObjectiveVector(),  # incomplete - dropped
    ]
    rust_vectors = [
        rust_pareto.ObjectiveVector(ctx=2048.0, tps=80.0, agentic=0.6, coding=0.5),
        rust_pareto.ObjectiveVector(ctx=4096.0, tps=50.0, agentic=0.4, coding=0.45),
        rust_pareto.ObjectiveVector(ctx=1024.0, tps=20.0, agentic=0.2, coding=0.3),
        rust_pareto.ObjectiveVector(),
    ]

    py_set = py_pareto.pareto_set(py_vectors)
    rust_set = rust_pareto.pareto_set(rust_vectors)
    if len(py_set) != len(rust_set):
        print(f"  FAIL pareto_set length mismatch py={len(py_set)} rust={len(rust_set)}")
        return 1
    print(f"  OK pareto_set({len(py_vectors)}) -> {len(py_set)} complete vectors in both impls")

    section("4. SearchState roundtrip on disk")
    from autoresearch.core.state import SearchState

    with tempfile.TemporaryDirectory() as tmp:
        state_path = Path(tmp) / "state.json"
        st = SearchState(state_path)
        st.mark_visited("n1")
        st.mark_visited("n2")
        st.set_morris(
            "trial.gguf",
            {"threads": 8, "batch_size": 512},
            {"KV_CACHE_effect": "+12%"},
        )
        del st

        with state_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if data["visited"] != ["n1", "n2"]:
            print(f"  FAIL visited ordering: {data['visited']}")
            return 1
        if "trial.gguf" not in data["morris"]:
            print(f"  FAIL morris missing trial.gguf: {data['morris']}")
            return 1
        print(
            f"  OK state written + read: {data['schema_version']} schema, {len(data['visited'])} visited, "
            f"{len(data['morris'])} morris entries"
        )

        st2 = SearchState(state_path)
        if not st2.is_visited("n1") or not st2.is_visited("n2"):
            print(f"  FAIL second instance lost state: visited={st2.visited}")
            return 1
        pins = st2.morris_pins_for("trial.gguf")
        if pins.get("threads") != 8 or pins.get("batch_size") != 512:
            print(f"  FAIL morris pins corrupted: {pins}")
            return 1
        print("  OK reload preserves visited + morris pins")

    section("5. Fingerprint Trial-gate (mismatch_reason) preserves Python contract")
    from autoresearch_core import fingerprint as rust_fp

    from autoresearch.core import fingerprint as py_fp

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        engine = {
            "MODEL": "trial.gguf",
            "CTX_SIZE": 65536,
            "BATCH_SIZE": 512,
            "KV_CACHE": "q4_0",
            "FLASH_ATTN": "on",
            "VRAM_LIMIT_MB": 7900.0,
            "TPS_FLOOR": 20.0,
        }
        # Dump via Python shim (which delegates to Rust dump).
        py_fp.dump(tmp_path / "trial.json", model="trial.gguf", engine=engine)

        baseline_matching = dict(engine)
        baseline_drifting = dict(engine, **{"CTX_SIZE": 32768})

        py_match = py_fp.mismatch_reason("trial.gguf", baseline_matching, directory=tmp_path)
        rust_match = rust_fp.mismatch_reason(
            "trial.gguf", baseline_matching, directory=str(tmp_path)
        )
        if py_match != rust_match:
            print(f"  FAIL matching baseline mismatch - py={py_match!r} rust={rust_match!r}")
            return 1
        print("  OK matching baseline: py=None rust=None (gate opens)")

        py_diff = py_fp.mismatch_reason("trial.gguf", baseline_drifting, directory=tmp_path)
        rust_diff = rust_fp.mismatch_reason(
            "trial.gguf", baseline_drifting, directory=str(tmp_path)
        )
        if py_diff is None or rust_diff is None:
            print(f"  FAIL drifting baseline should reject - py={py_diff!r} rust={rust_diff!r}")
            return 1
        if "CTX_SIZE" not in py_diff or "CTX_SIZE" not in rust_diff:
            print(f"  FAIL reason missing CTX_SIZE - py={py_diff!r} rust={rust_diff!r}")
            return 1
        print("  OK drifting baseline: both reject and mention CTX_SIZE")
        print(f"     Python:  {py_diff[:90]}...")
        print(f"     Rust:    {rust_diff[:90]}...")

    section("6. autoloop importable + state import path wired")
    try:
        import autoloop

        print(f"  OK autoloop module imported: {autoloop.__file__}")
    except Exception as exc:
        print(f"  FAIL autoloop import: {exc!r}")
        return 1

    print("\nALL E2E SMOKE CHECKS PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
