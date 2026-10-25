"""Bounded tuning E2E: real HTTP/SQLite, simulated model and hardware; no GPU load.

Failure modes: short-context TPS accepted as target-context TPS; non-finite or
missing timings; PPL drift accumulated across Neighbors; inferior points kept;
below-floor context; benchmark imports; simultaneous PPL/server model loads;
watchdog failures scored as success; broken native process cleanup; native CPU
commands depending on server-only attributes while selecting memory monitoring.
"""

from __future__ import annotations

import importlib.abc
import json
import math
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if not os.environ.get("TUNING_E2E_ISOLATED"):
    standalone = Path(tempfile.mkdtemp(prefix="tuning-standalone-"))
    shutil.copytree(
        ROOT / "autoresearch/core",
        standalone / "autoresearch/core",
        ignore=shutil.ignore_patterns("config.py", "__pycache__"),
    )
    shutil.copyfile(
        ROOT / "autoresearch/core/config.py.example", standalone / "autoresearch/core/config.py"
    )
    shutil.copyfile(ROOT / "autoresearch/__init__.py", standalone / "autoresearch/__init__.py")
    shutil.copyfile(ROOT / "autoresearch/tuning.py", standalone / "autoresearch/tuning.py")
    (standalone / "scripts").mkdir()
    shutil.copyfile(Path(__file__), standalone / "scripts/test_tuning_e2e.py")
    (standalone / "data").mkdir()
    shutil.copyfile(ROOT / "data/perplexity_val.txt", standalone / "data/perplexity_val.txt")
    completed = subprocess.run(
        [sys.executable, str(standalone / "scripts/test_tuning_e2e.py")],
        cwd=standalone,
        check=True,
        env={**os.environ, "TUNING_E2E_ISOLATED": "1"},
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        capture_output=True,
        text=True,
    )
    print(completed.stdout, end="")
    assert "Tuning E2E passed:" in completed.stdout, completed.stderr
    print(f"Standalone tuning source: {standalone}")
    raise SystemExit(0)
sys.path.insert(0, str(ROOT))


class NoBenchmarks(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.startswith(("autoresearch.benchmarks", "autoresearch.runners")):
            raise AssertionError(f"Tuning imported legacy benchmark code: {fullname}")


sys.meta_path.insert(0, NoBenchmarks())
from autoresearch.core import trial_store  # noqa: E402
from autoresearch.core.llama_runner import NativeModelProcess, ServerIntent  # noqa: E402
from autoresearch.tuning import TuningRunner, measure_perplexity, run_search  # noqa: E402


def main() -> None:
    artifact = (
        Path(os.environ["TUNING_E2E_DIR"])
        if os.environ.get("TUNING_E2E_DIR")
        else Path(tempfile.mkdtemp(prefix="tuning-e2e-"))
    )
    artifact.mkdir(parents=True, exist_ok=True)
    if (artifact / "results.db").exists():
        raise ValueError("TUNING_E2E_DIR must be a fresh evidence directory")
    active = []
    contexts = []

    class Server:
        def __init__(self, intent, **kwargs):
            self.intent = intent
            self.vram_killed = self.ram_killed = False
            self.vram_kill_reason = ""
            self.peak_vram_mb = 1024

        def __enter__(self):
            assert not active
            active.append(self)
            intent = self.intent

            class Handler(BaseHTTPRequestHandler):
                def do_GET(self):
                    body = {
                        "default_generation_settings": {"n_ctx": intent.ctx_size},
                        "total_slots": 1,
                    }
                    if intent.batch_size == 2048:
                        body["default_generation_settings"]["n_ctx"] = 4096
                    self.respond(body)

                def do_POST(self):
                    request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                    assert request["n_predict"] == 512
                    assert request["cache_prompt"] is False
                    contexts.append(intent.ctx_size)
                    rate = (60 if intent.ctx_size == 65536 else 45) + (
                        10 if intent.batch_size == 256 else 0
                    )
                    if intent.batch_size == 4096:
                        rate = math.nan
                    self.respond(
                        {
                            "generation_settings": {"n_ctx": intent.ctx_size},
                            "timings": {"predicted_per_second": rate, "predicted_n": 512},
                            "tokens_evaluated": 16,
                            "truncated": False,
                        }
                    )

                def respond(self, body):
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(json.dumps(body).encode())

                def log_message(self, *args):
                    pass

            self.http = HTTPServer(("127.0.0.1", 0), Handler)
            self.port = self.http.server_port
            self.thread = threading.Thread(target=self.http.serve_forever, daemon=True)
            self.thread.start()
            return self

        def __exit__(self, *args):
            self.http.shutdown()
            self.http.server_close()
            self.thread.join()
            active.remove(self)

    def perplexity(intent, **kwargs):
        assert not active, "PPL and server loaded simultaneously"
        return 10.11 if intent.batch_size == 1024 else 10.05 if intent.batch_size == 256 else 10.0

    baseline = {
        "MODEL": "fixture.gguf",
        "CTX_SIZE": 65536,
        "N_CPU_MOE": None,
        "N_GPU_LAYERS": 0,
        "BATCH_SIZE": 512,
        "UBATCH_SIZE": 128,
        "SPEC_DRAFT_N_MAX": 0,
        "TPS_REPS": 1,
    }
    plan = {
        "minimum_context": 65536,
        "max_trials": 20,
        "search_space": {"CTX_SIZE": [65536, 131072], "BATCH_SIZE": [256, 512, 1024]},
    }
    with (
        patch("autoresearch.tuning.preflight_vram_for_intent", return_value=(True, 1000, "")),
        patch(
            "autoresearch.tuning.preflight_host_memory_for_intent",
            return_value=(True, 1000, 2000, ""),
        ),
    ):
        runner = TuningRunner(
            artifact, server_factory=Server, perplexity=perplexity, thermal_wait=False
        )
        for unsafe_key in ("VRAM_LIMIT_MB", "RAM_WATCHDOG_RESERVE_MB"):
            try:
                run_search(
                    baseline,
                    {**plan, "search_space": {unsafe_key: [1]}},
                    runner,
                    artifact / "forbidden.db",
                )
            except ValueError:
                pass
            else:
                raise AssertionError("Search Space must not optimize hardware safety budgets")
        result = run_search(baseline, plan, runner, artifact / "results.db")
        front = {(p["ctx"], p["tps"]) for p in result["frontier"]}
        assert front == {(65536, 70.0), (131072, 55.0)}, front
        assert result["reference_ppl"] == 10.0
        for batch in (2048, 4096):
            measurement = runner.run_trial({**baseline, "BATCH_SIZE": batch}, minimum_context=65536)
            assert measurement["outcome"] == "INFRA_ERROR", measurement
            assert measurement["tps"] is None
        before = len(contexts)
        rejected = runner.run_trial({**baseline, "CTX_SIZE": 4096}, minimum_context=65536)
        assert rejected["outcome"] == "INVALID_CONFIG"
        assert len(contexts) == before
    assert min(contexts) >= 65536
    assert not active
    with sqlite3.connect(artifact / "results.db") as db:
        rows = db.execute("SELECT ctx,tps,ppl,accepted FROM tuning_trials").fetchall()
        assert any(ppl > 10.1 and not accepted for _, _, ppl, accepted in rows)
        assert all(ctx >= 65536 for ctx, *_ in rows)
    # Native model lifecycle uses a tiny Python child, not an inference binary.
    intent = ServerIntent(Path("fixture.gguf"), 2048, "q4_0", "on", ngl=0)
    with (
        patch(
            "autoresearch.core.llama_runner.preflight_vram_for_intent", return_value=(True, 0, "")
        ),
        patch(
            "autoresearch.core.llama_runner.preflight_host_memory_for_intent",
            return_value=(True, 0, 1, ""),
        ),
        patch("autoresearch.core.llama_runner.resolve_vram_limit_mb", return_value=10000.0),
        patch("autoresearch.core.llama_runner.enforce_single_load"),
        patch("autoresearch.core.llama_runner.sweep_leftover_processes"),
        patch("autoresearch.core.llama_runner.ram_resident_bytes_mb", return_value=(0, "fixture")),
        patch("autoresearch.core.llama_runner.circuit_breaker.preflight_ram"),
        patch("autoresearch.core.llama_runner.circuit_breaker.start_ram_watchdog"),
    ):
        with NativeModelProcess(
            intent, [sys.executable, "-c", "print('native-child-ok')"], artifact / "native.log"
        ) as child:
            child.wait()
            assert "native-child-ok" in child.output()
        assert child._server_proc is None
        for index, output in enumerate(
            ("Final estimate: PPL = 10.05", "[1]10.0,\nFinal estimate: PPL = nan")
        ):

            def native_fixture(actual_intent, command, log_path, _output=output):
                assert "--load-mode" in command
                assert actual_intent.ctx_size == 2048
                return NativeModelProcess(
                    actual_intent, [sys.executable, "-c", f"print({_output!r})"], log_path
                )

            with (
                patch(
                    "autoresearch.tuning.resolve_llama_perplexity",
                    return_value=Path(sys.executable),
                ),
                patch("autoresearch.tuning.assert_flags_supported"),
                patch("autoresearch.tuning.NativeModelProcess", side_effect=native_fixture),
            ):
                try:
                    value = measure_perplexity(intent, log_path=artifact / f"ppl-{index}.log")
                except ValueError:
                    assert index == 1
                else:
                    assert index == 0 and value == 10.05
    with patch.object(trial_store, "_engine_version", return_value="e2e-runtime"):
        trial_store.write_row(
            artifact / "results.tsv",
            "e2e",
            0,
            0,
            0,
            0,
            0,
            "incomplete",
            "isolated storage E2E",
            model="fixture.gguf",
            ctx=65536,
            tps=60,
        )
    with sqlite3.connect(artifact / "results.db") as db:
        assert db.execute("SELECT model FROM trials").fetchall() == [("fixture.gguf",)]
    # Existing store callers no longer import a benchmark CLI.
    assert callable(trial_store.write_row)
    assert not any(
        name.startswith(("autoresearch.benchmarks", "autoresearch.runners")) for name in sys.modules
    )
    summary = {
        "status": "passed",
        "cases": [
            "shared_sqlite_store",
            "native_ppl_finite_estimate",
            "pareto",
            "fixed_ppl_reference",
            "full_context_http",
            "context_mismatch",
            "invalid_timing",
            "minimum_context",
            "native_lifecycle",
            "no_benchmark_imports",
        ],
        "result": result,
        "measured_contexts": contexts,
    }
    (artifact / "verification.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Tuning E2E passed: {artifact / 'verification.json'}")


if __name__ == "__main__":
    main()
