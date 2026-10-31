"""Fresh tuning install -> readiness check -> existing HTTP/SQLite E2Es.

Failure modes: default installation pulls evaluation/dev dependencies; readiness
requires optional packages or uses global Python; tuning/dashboard metadata readers
fail without evaluation dependencies. Uses simulated model processes, no GPU.
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NO_WINDOW = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0


def main():
    output = Path(tempfile.mkdtemp(prefix="tuning-install-e2e-"))
    print(f"Install E2E artifacts: {output}", flush=True)
    python = output / ("venv/Scripts/python.exe" if os.name == "nt" else "venv/bin/python")
    env = {**os.environ, "PYTHONPATH": "", "PYTHONIOENCODING": "utf-8"}

    def run(name, command, *, extra=None, cwd=ROOT, expected=0):
        result = subprocess.run(
            list(map(str, command)),
            cwd=cwd,
            env={**env, **(extra or {})},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=NO_WINDOW,
        )
        (output / f"{name}.log").write_text(result.stdout, encoding="utf-8")
        assert result.returncode == expected, f"{name}: {result.stdout}"
        return result.stdout

    run("create-venv", [sys.executable, "-m", "venv", output / "venv"])
    run(
        "install",
        [
            python,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "-r",
            ROOT / "requirements.txt",
        ],
    )
    run("dependency-check", [python, "-m", "pip", "check"])
    packages = json.loads(run("installed-packages", [python, "-m", "pip", "list", "--format=json"]))
    run(
        "runtime-only",
        [
            python,
            "-c",
            """
import importlib.util
import yaml, huggingface_hub, gguf, psutil
for name in ('evalplus', 'datasets', 'fastapi', 'uvicorn', 'pypdf', 'pytest', 'ruff', 'pre_commit'):
    assert importlib.util.find_spec(name) is None, f'Unexpected default dependency: {name}'
""",
        ],
        cwd=output,
    )

    # Readiness runs against this venv, a fake --help binary and no model files.
    (output / "scripts").mkdir()
    shutil.copyfile(ROOT / "scripts/setup-check.sh", output / "scripts/setup-check.sh")
    shutil.copyfile(ROOT / "requirements.txt", output / "requirements.txt")
    (output / "models").mkdir()
    tools = output / "fixture-tools"
    server = tools / "build-cpu/bin/llama-server"
    server.parent.mkdir(parents=True)
    server.write_text(
        "#!/bin/sh\nprintf '%s\\n' '--spec-type --cache-type-k --n-cpu-moe'\n", encoding="utf-8"
    )
    server.chmod(0o755)
    uvx = tools / "uvx"
    uvx.write_text("#!/bin/sh\nprintf '%s\\n' 'fixture discovery tool'\n", encoding="utf-8")
    uvx.chmod(0o755)
    for name in ("python", "python3"):
        global_python = tools / name
        global_python.write_text("#!/bin/sh\nexit 99\n", encoding="utf-8")
        global_python.chmod(0o755)
    bash = shutil.which("bash")
    if os.name == "nt":
        git_bash = Path(os.environ.get("PROGRAMFILES", "")) / "Git/bin/bash.exe"
        if git_bash.is_file():
            bash = str(git_bash)
    assert bash, "bash is required for setup-check.sh"
    ready = run(
        "readiness",
        [bash, output / "scripts/setup-check.sh"],
        extra={
            "AUTORESEARCH_LLAMA_CPP_ROOT": str(tools),
            "PATH": os.pathsep.join((str(tools), str(python.parent), env.get("PATH", ""))),
        },
        cwd=output,
    )
    assert "Core tuning dependencies installed" in ready, ready

    missing = output / "missing-deps"
    run("empty-venv", [sys.executable, "-m", "venv", "--without-pip", missing / "venv"])
    (missing / "scripts").mkdir()
    (missing / "models").mkdir()
    shutil.copyfile(ROOT / "scripts/setup-check.sh", missing / "scripts/setup-check.sh")
    shutil.copyfile(ROOT / "requirements.txt", missing / "requirements.txt")
    rejected = run(
        "missing-runtime-rejected",
        [bash, missing / "scripts/setup-check.sh"],
        extra={
            "AUTORESEARCH_LLAMA_CPP_ROOT": str(tools),
            "PATH": os.pathsep.join((str(tools), str(python.parent), env.get("PATH", ""))),
        },
        cwd=missing,
        expected=1,
    )
    assert "Missing tuning dependencies" in rejected, rejected

    run(
        "tuning",
        [python, ROOT / "scripts/test_tuning_e2e.py"],
        extra={"TUNING_E2E_DIR": str(output / "tuning")},
    )
    consumer_temp = output / "consumers"
    consumer_temp.mkdir()
    run(
        "consumers",
        [python, ROOT / "scripts/test_tuning_consumers_e2e.py"],
        extra={"TMPDIR": str(consumer_temp), "TEMP": str(consumer_temp), "TMP": str(consumer_temp)},
    )
    databases = list(output.rglob("results.db"))
    assert len(databases) >= 2, databases
    for database in databases:
        with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as db:
            assert db.execute("PRAGMA integrity_check").fetchone() == ("ok",)
    summary = {
        "status": "passed",
        "installed_distributions": len(packages),
        "databases": [str(p.relative_to(output)) for p in databases],
    }
    (output / "verification.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
