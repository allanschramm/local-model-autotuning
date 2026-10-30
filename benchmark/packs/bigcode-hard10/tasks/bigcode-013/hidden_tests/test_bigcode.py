"""Frozen BigCodeBench unittest bridge; copied into each task before export."""

import importlib
import io
import json
import os
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import pytest


def test_original_bigcode_suite():
    directory = Path(__file__).resolve().parent
    os.environ["MPLBACKEND"] = "Agg"
    try:
        dependencies = json.loads((directory / "dependencies.json").read_text(encoding="utf-8"))
        tests = (directory / "original_tests.txt").read_text(encoding="utf-8")
        for name in dependencies:
            importlib.import_module(name)
    except (OSError, ValueError, ImportError) as exc:
        pytest.exit(f"BigCodeBench verifier environment unavailable: {exc}", returncode=3)
    source = directory.parent / "solution.py"
    module = types.ModuleType("__test__")
    module.__file__ = str(source)
    exec(
        compile(source.read_text(encoding="utf-8") + "\n" + tests, str(source), "exec"),
        module.__dict__,
    )
    suite = unittest.TestLoader().loadTestsFromTestCase(module.TestCases)
    if not suite.countTestCases():
        pytest.exit("Frozen BigCodeBench suite has no test cases", returncode=3)
    original_assert_in = unittest.TestCase.assertIn

    def compatible_assert_in(self, member, container, msg=None):
        if os.name == "nt" and member == "not found" and isinstance(container, str):
            if "not recognized" in container or "not found" in container:
                return
        return original_assert_in(self, member, container, msg=msg)

    log = io.StringIO()
    with patch.object(unittest.TestCase, "assertIn", compatible_assert_in):
        result = unittest.TextTestRunner(stream=log, verbosity=1).run(suite)
    assert result.wasSuccessful(), log.getvalue()
