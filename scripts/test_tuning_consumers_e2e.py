"""Tuning-only distribution and read-only dashboard E2E with SQLite evidence.

Failure modes: retired evaluators/CLIs remain shipped or imported; dashboard
shows old quality rows instead of tuning; admitted dominated points look like
frontier points; reading changes historical tables; empty/legacy-only store
crashes the UI. No model or external service is used.
"""

from __future__ import annotations

import importlib.abc
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
RETIRED = (
    "autoresearch/benchmarks",
    "autoresearch/runners",
    "benchmark",
    "autoloop.py",
    "benchmark_search.py",
    "scripts/rank_results.py",
    "autoresearch/core/trial_store.py",
    "autoresearch/core/classify.py",
)


def main():
    for name in RETIRED:
        assert not (ROOT / name).exists(), f"Retired quality surface still shipped: {name}"
    if not os.environ.get("TUNING_CONSUMERS_ISOLATED"):
        isolated = Path(tempfile.mkdtemp(prefix="tuning-consumers-e2e-"))
        for directory in ("autoresearch", "ui"):
            shutil.copytree(
                ROOT / directory,
                isolated / directory,
                ignore=shutil.ignore_patterns("config.py", "__pycache__", "logs"),
            )
        shutil.copyfile(
            ROOT / "autoresearch/core/config.py.example", isolated / "autoresearch/core/config.py"
        )
        (isolated / "scripts").mkdir()
        shutil.copyfile(Path(__file__), isolated / "scripts/test_tuning_consumers_e2e.py")
        result = subprocess.run(
            [sys.executable, isolated / "scripts/test_tuning_consumers_e2e.py"],
            cwd=isolated,
            env={**os.environ, "TUNING_CONSUMERS_ISOLATED": "1", "PYTHONPATH": str(isolated)},
            text=True,
            capture_output=True,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        (isolated / "consumer.log").write_text(result.stdout + result.stderr, encoding="utf-8")
        assert result.returncode == 0, result.stdout + result.stderr
        print(f"Tuning consumer artifacts: {isolated}")
        print(result.stdout, end="")
        return
    sys.path.insert(0, str(ROOT))

    class NoQuality(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname.startswith(("autoresearch.benchmarks", "autoresearch.runners")):
                raise AssertionError(f"Quality import: {fullname}")

    sys.meta_path.insert(0, NoQuality())
    from ui.server import DashboardHandler
    from ui.trial_reader import read_last_50_trials

    assert read_last_50_trials() == []
    database = ROOT / "results.db"
    with sqlite3.connect(database) as db:
        db.executescript("""
            CREATE TABLE trials (trial_id TEXT, agentic REAL, coding REAL);
            INSERT INTO trials VALUES ('historical-quality', 0.9, 0.8);
            CREATE TABLE benchmark_runs (run_id TEXT, model TEXT);
            INSERT INTO benchmark_runs VALUES ('pi', 'quality-only.gguf');
        """)
    assert read_last_50_trials() == []
    with sqlite3.connect(database) as db:
        db.executescript("""
            CREATE TABLE tuning_runs (run_id TEXT PRIMARY KEY, protocol TEXT, plan_json TEXT,
                status TEXT, reference_ppl REAL, frontier_json TEXT);
            CREATE TABLE tuning_trials (trial_id TEXT PRIMARY KEY, run_id TEXT, config_json TEXT,
                ctx INTEGER, tps REAL, ppl REAL, outcome TEXT, accepted INTEGER, artifact TEXT);
        """)
        db.execute(
            "INSERT INTO tuning_runs VALUES (?,?,?,?,?,?)",
            (
                "tuning",
                "tuning-runtime-v1",
                "{}",
                "complete",
                10.0,
                json.dumps([{"trial_id": "frontier"}]),
            ),
        )
        for name, accepted, tps, ppl in (
            ("dominated", 1, 30, 10.0),
            ("frontier", 1, 40, 10.0),
            ("ppl-rejected", 0, 50, 11.0),
        ):
            db.execute(
                "INSERT INTO tuning_trials VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    name,
                    "tuning",
                    json.dumps({"MODEL": "tuning-only.gguf"}),
                    65536,
                    tps,
                    ppl,
                    "OK",
                    accepted,
                    "fixture.json",
                ),
            )
        before = db.execute("SELECT * FROM trials").fetchall()
    rows = read_last_50_trials()
    assert len(rows) == 3
    statuses = {row["trial_id"]: row["status"] for row in rows}
    assert statuses == {
        "dominated": "dominated",
        "frontier": "on_front",
        "ppl-rejected": "rejected",
    }
    server = HTTPServer(("127.0.0.1", 0), DashboardHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        with urllib.request.urlopen(base + "/api/status") as response:
            payload = json.load(response)
        assert len(payload["trials"]) == 3
        assert all(row["model"] == "tuning-only.gguf" for row in payload["trials"])
        assert all(
            "ppl" in row and "agentic" not in row and "coding" not in row
            for row in payload["trials"]
        )
        with urllib.request.urlopen(base + "/") as response:
            page = response.read().decode()
        assert "PPL" in page and '<th class="num">coding</th>' not in page
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    with sqlite3.connect(database) as db:
        assert db.execute("SELECT * FROM trials").fetchall() == before
        assert db.execute("PRAGMA integrity_check").fetchone() == ("ok",)
    summary = {
        "status": "passed",
        "tuning_rows": 3,
        "historical_rows_unchanged": True,
        "retired_surfaces_absent": list(RETIRED),
    }
    (ROOT / "verification.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
