"""Legacy consumer E2E: isolated imports, real CLI/HTTP readers and SQLite evidence.

Failure modes: catalog/config imports eagerly load quality runners; public
exports or star imports change identity; historical rows disappear or mix with
Pi/tuning protocols; SQLite-only rankings require a deprecated log; read-only
consumers mutate outcomes; independent tuning imports legacy evaluation.
"""

from __future__ import annotations

import ast
import contextlib
import importlib
import importlib.abc
import io
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
import urllib.request
from http.server import HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NO_WINDOW = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
RUNNERS = (
    "autoresearch.runners",
    "autoresearch.benchmarks.agentic_runner",
    "autoresearch.benchmarks.agentic_coding",
    "autoresearch.benchmarks.benchmark_coding",
    "autoresearch.benchmarks.mini_swe_agent",
)


class NoEvaluation(importlib.abc.MetaPathFinder):
    def __init__(self, prefixes):
        self.prefixes = prefixes

    def find_spec(self, fullname, path, target=None):
        if any(fullname == prefix or fullname.startswith(prefix + ".") for prefix in self.prefixes):
            raise AssertionError(f"Read-only consumer imported evaluation: {fullname}")


def probe_catalog():
    sys.meta_path.insert(0, NoEvaluation(RUNNERS))
    from autoresearch.benchmarks import bench_config, format_claw_tiers

    assert bench_config.load_config()["CODING_TASK_LIMIT"] == 10
    text = format_claw_tiers()
    assert "Quick tier (5 tasks" in text and "Full tier (15 tasks" in text
    assert "T002_email_triage" in text
    assert not any(name.startswith(RUNNERS) for name in sys.modules)
    return {"case": "catalog-and-config", "quality_runners_loaded": False}


def probe_readers():
    sys.meta_path.insert(0, NoEvaluation(("autoresearch.benchmarks", "autoresearch.runners")))
    from autoresearch.core import results_db
    from scripts import rank_results
    from ui.server import DashboardHandler
    from ui.trial_reader import read_last_50_trials

    database = ROOT / "results.db"
    rows = []
    for suffix, protocol in (
        ("historical", "coding-legacy-unversioned"),
        ("modern", "coding-v2-32768-final-only"),
    ):
        rows.append(
            {
                "trial_id": suffix,
                "timestamp": "2026-01-01T00:00:00Z",
                "model": f"fixture-{suffix}.gguf",
                "status": "on_front",
                "outcome": "OK",
                "category": "full-suite",
                "agentic": "0.6",
                "coding": "0.7",
                "tps": "50",
                "ctx": "65536",
                "config_json": json.dumps({"coding_protocol": protocol, "CTX_SIZE": 65536}),
            }
        )
    results_db.upsert_rows(database, rows)
    with sqlite3.connect(database) as db:
        db.executescript("""
            CREATE TABLE benchmark_runs (run_id TEXT, protocol TEXT, model TEXT, status TEXT);
            CREATE TABLE benchmark_tasks (run_id TEXT, task_id TEXT, passed INTEGER);
            INSERT INTO benchmark_runs VALUES ('pi', 'pi-dm-v1', 'fixture-pi.gguf', 'complete');
            INSERT INTO benchmark_tasks VALUES ('pi', 'fixture-task', 1);
            CREATE TABLE tuning_runs (run_id TEXT, model TEXT, status TEXT);
            CREATE TABLE tuning_trials (run_id TEXT, ctx INTEGER, tps REAL);
            INSERT INTO tuning_runs VALUES ('tuning', 'fixture-tuning.gguf', 'complete');
            INSERT INTO tuning_trials VALUES ('tuning', 65536, 999);
        """)
        before = db.execute("SELECT * FROM trials ORDER BY trial_id").fetchall()
    assert {r["model"] for r in read_last_50_trials()} == {r["model"] for r in rows}
    assert not (ROOT / "results.tsv").exists()
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        code = rank_results.main(["--mode", "all"])
    assert code == 0, "Ranking CLI rejected SQLite-only store: " + output.getvalue()
    text = output.getvalue()
    assert "fixture-historical.gguf" in text and "fixture-modern.gguf" in text
    assert "coding-legacy-unversioned" in text and "coding-v2-32768-final-only" in text
    assert "fixture-pi" not in text and "fixture-tuning" not in text
    (ROOT / "ranking.txt").write_text(text, encoding="utf-8")
    server = HTTPServer(("127.0.0.1", 0), DashboardHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        with urllib.request.urlopen(base + "/") as response:
            assert 'lang="pt-BR"' in response.read().decode()
        with urllib.request.urlopen(base + "/api/status") as response:
            payload = json.load(response)
        assert len(payload["trials"]) == 2
        assert {float(r["coding"]) for r in payload["trials"]} == {0.7}
        (ROOT / "dashboard.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    finally:
        server.shutdown()
        thread.join()
        server.server_close()
    with sqlite3.connect(database) as db:
        assert db.execute("SELECT * FROM trials ORDER BY trial_id").fetchall() == before
        assert db.execute("SELECT COUNT(*) FROM benchmark_runs").fetchone()[0] == 1
        assert db.execute("SELECT COUNT(*) FROM tuning_runs").fetchone()[0] == 1
    assert not any(
        name.startswith(("autoresearch.benchmarks", "autoresearch.runners")) for name in sys.modules
    )
    return {
        "case": "SQLite-readers-and-HTTP",
        "legacy_rows": 2,
        "protocols_isolated": True,
        "rows_unchanged": True,
    }


def probe_exports():
    import autoresearch.benchmarks as package

    expected = {
        "BenchmarkResult": "benchmark_harness",
        "run_benchmark": "benchmark_coding",
        "run_coding_eval": "benchmark_coding",
        "run_agentic_eval": "agentic_runner",
        "run_agent_loop": "agentic_runner",
        "run_agentic_coding_eval": "agentic_coding.runner",
        "score_task": "agentic_runner",
        "discover_claw_tasks": "agentic_benchmarks",
        "get_quick_tier_tasks": "agentic_benchmarks",
        "get_full_tier_tasks": "agentic_benchmarks",
        "format_agentic_benchmarks": "agentic_benchmarks",
        "format_claw_tiers": "agentic_benchmarks",
        "AgenticBenchmarkSpec": "agentic_benchmarks",
        "AGENTIC_BENCHMARKS": "agentic_benchmarks",
    }
    assert set(package.__all__) == set(expected)
    assert set(expected).issubset(dir(package))
    for name, owner in expected.items():
        value = getattr(package, name)
        assert value is getattr(importlib.import_module(f"autoresearch.benchmarks.{owner}"), name)
        assert getattr(package, name) is value
    namespace = {}
    exec("from autoresearch.benchmarks import *", namespace)
    assert all(namespace[name] is getattr(package, name) for name in expected)
    assert package.bench_config is importlib.import_module("autoresearch.benchmarks.bench_config")
    try:
        _ = package.not_a_public_api
    except AttributeError:
        pass
    else:
        raise AssertionError("Unknown package attributes must fail normally")
    return {"case": "public-exports", "identical_symbols": len(expected), "star_import": True}


def main():
    if os.environ.get("LEGACY_CONSUMERS_E2E_PROBE"):
        sys.path.insert(0, str(ROOT))
        probes = {"catalog": probe_catalog, "readers": probe_readers, "exports": probe_exports}
        print(json.dumps(probes[os.environ["LEGACY_CONSUMERS_E2E_PROBE"]]()))
        return
    isolated = Path(tempfile.mkdtemp(prefix="legacy-consumers-e2e-"))
    tracked = subprocess.check_output(
        [
            "git",
            "ls-files",
            "--",
            "autoresearch",
            "scripts",
            "ui",
            "benchmark_search.py",
            "autoloop.py",
        ],
        cwd=ROOT,
        text=True,
        creationflags=NO_WINDOW,
    ).splitlines()
    inventory = []
    for name in tracked:
        path = ROOT / name
        if path.suffix != ".py" and not name.endswith("config.py.example"):
            continue
        if "data" in path.relative_to(ROOT).parts or "logs" in path.relative_to(ROOT).parts:
            continue
        destination = isolated / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        if path.suffix == ".py":
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                    "autoresearch.benchmarks"
                ):
                    inventory.append(
                        {
                            "path": name,
                            "line": node.lineno,
                            "module": node.module,
                            "names": [alias.name for alias in node.names],
                        }
                    )
    shutil.copyfile(
        ROOT / "autoresearch/core/config.py.example", isolated / "autoresearch/core/config.py"
    )
    shutil.copyfile(Path(__file__), isolated / "scripts/test_legacy_consumers_e2e.py")
    provenance = json.loads(
        (ROOT / "benchmark/packs/claw-tiers/source.json").read_text(encoding="utf-8")
    )
    for name in provenance["full"]:
        target = isolated / "claw-eval/tasks" / name / "task.yaml"
        target.parent.mkdir(parents=True)
        shutil.copyfile(
            ROOT / "benchmark/packs/claw-tiers/upstream/tasks" / name / "task.yaml", target
        )
    print(f"Consumer E2E artifacts: {isolated}", flush=True)
    cases = []
    for probe in ("catalog", "readers", "exports"):
        result = subprocess.run(
            [sys.executable, str(isolated / "scripts/test_legacy_consumers_e2e.py")],
            cwd=isolated,
            env={
                **os.environ,
                "PYTHONPATH": str(isolated),
                "PYTHONIOENCODING": "utf-8",
                "LEGACY_CONSUMERS_E2E_PROBE": probe,
            },
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=NO_WINDOW,
        )
        (isolated / f"{probe}.log").write_text(result.stdout + result.stderr, encoding="utf-8")
        assert result.returncode == 0, result.stdout + result.stderr
        cases.append(json.loads(result.stdout.strip().splitlines()[-1]))
    for args in (
        ["benchmark_search.py", "--help"],
        ["benchmark_search.py", "--list-claw-tiers"],
        ["-m", "autoresearch.runners.run", "--list-agentic-benchmarks"],
        ["-m", "autoresearch.tuning", "--help"],
    ):
        result = subprocess.run(
            [sys.executable, *args],
            cwd=isolated,
            env={**os.environ, "PYTHONPATH": str(isolated), "PYTHONIOENCODING": "utf-8"},
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=NO_WINDOW,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        cases.append({"case": "CLI " + " ".join(args), "exit": result.returncode})
    (isolated / "imports.json").write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    summary = {
        "status": "passed",
        "cases": cases,
        "database": str(isolated / "results.db"),
        "static_import_inventory": "imports.json",
        "standalone": str(isolated),
    }
    (isolated / "verification.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
