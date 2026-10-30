"""Copied into frozen stdin tasks; each program runs in a fresh private cwd."""

import gzip
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


def test_private_stdin_cases():
    directory = Path(__file__).resolve().parent
    try:
        with gzip.open(directory / "private_cases.json.gz", "rt", encoding="utf-8") as handle:
            cases = json.load(handle)
        if (
            not isinstance(cases, list)
            or not cases
            or any(
                not isinstance(case, dict)
                or case.get("testtype") != "stdin"
                or not isinstance(case.get("input"), str)
                or not isinstance(case.get("output"), str)
                for case in cases
            )
        ):
            raise ValueError("Invalid private stdin cases")
    except (OSError, ValueError, EOFError) as exc:
        pytest.exit(f"Invalid frozen stdin fixtures: {exc}", returncode=3)
    source = directory.parent / "solution.py"
    for index, case in enumerate(cases):
        with tempfile.TemporaryDirectory(prefix="pi-stdin-case-") as workdir:
            result = subprocess.run(
                [sys.executable, str(source)],
                input=case["input"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=workdir,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
        successful = result.returncode == 0
        assert successful, f"private case {index}: exit={result.returncode}; {result.stderr[:2000]}"
        matches = result.stdout.rstrip("\n") == case["output"].rstrip("\n")
        assert matches, f"private case {index}: stdout mismatch"
