"""Shared validate: ruff check + pytest (CI + local agents).

Usage (repo root):
  ./venv/Scripts/python.exe scripts/run_validate.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_RUFF_TARGETS = (
    "autoresearch",
    "tests",
    "scripts",
    "ui",
    "autoloop.py",
    "benchmark_search.py",
)


def _python() -> Path:
    for rel in ("venv/Scripts/python.exe", "venv/bin/python"):
        candidate = ROOT / rel
        if candidate.is_file():
            return candidate
    return Path(sys.executable)


def main() -> int:
    py = str(_python())
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    print("+", py, "-m ruff check", *_RUFF_TARGETS, flush=True)
    code = subprocess.run(
        [py, "-m", "ruff", "check", *_RUFF_TARGETS],
        cwd=str(ROOT),
        env=env,
        check=False,
    ).returncode
    if code != 0:
        return code
    # Fail closed on throwaway files in the repo root, before the slow suite.
    print("+", py, "scripts/check_no_root_litter.py", flush=True)
    code = subprocess.run(
        [py, str(ROOT / "scripts" / "check_no_root_litter.py")],
        cwd=str(ROOT),
        env=env,
        check=False,
    ).returncode
    if code != 0:
        return code
    # Reuse pre-commit pytest entry (same flags / venv rules).
    print("+", py, "scripts/run_pytest_hook.py", flush=True)
    return subprocess.run(
        [py, str(ROOT / "scripts" / "run_pytest_hook.py")],
        cwd=str(ROOT),
        env=env,
        check=False,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
