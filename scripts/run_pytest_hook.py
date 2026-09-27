"""pre-commit local hook: run pytest via the project venv (Win/macOS/Linux)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _venv_python() -> Path | None:
    # Both `venv/` and `.venv/` are accepted. `.venv/` is the canonical name
    # this repo standardizes on; `venv/` is kept as a fallback for older
    # checkouts.
    for rel in (
        ".venv/Scripts/python.exe",
        ".venv/bin/python",
        "venv/Scripts/python.exe",
        "venv/bin/python",
    ):
        candidate = ROOT / rel
        if candidate.is_file():
            return candidate
    return None


def main() -> int:
    py = _venv_python() or Path(sys.executable)
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    # Avoid .pytest_cache writes that can trip pre-commit's "files modified" check on Windows.
    cmd = [
        str(py),
        "-m",
        "pytest",
        "tests/",
        "-q",
        "-p",
        "no:cacheprovider",
    ]
    return subprocess.call(cmd, cwd=ROOT, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
