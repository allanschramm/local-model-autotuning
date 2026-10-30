"""Report pytest collection, completed calls and JUnit evidence to the parent."""

import json
import sys
from pathlib import Path
from xml.etree import ElementTree

import pytest


class Completion:
    def __init__(self):
        self.collected = 0
        self.finished = 0
        self.interrupted = False

    def pytest_collection_finish(self, session):
        self.collected = len(session.items)

    def pytest_runtest_logreport(self, report):
        if report.when == "call" and not report.skipped:
            self.finished += 1

    def pytest_keyboard_interrupt(self, excinfo):
        self.interrupted = True


report = Path(sys.argv[1])
xml = report.with_suffix(".xml")
completion = Completion()
status = int(pytest.main([*sys.argv[2:], f"--junitxml={xml}"], plugins=[completion]))
details = {"complete": False, "tests": 0, "executed": 0, "failed": 0}
try:
    cases = list(ElementTree.parse(xml).iter("testcase"))
    details.update(
        complete=bool(cases),
        tests=len(cases),
        executed=sum(case.find("skipped") is None for case in cases),
        failed=sum(
            case.find("failure") is not None or case.find("error") is not None for case in cases
        ),
    )
except (OSError, ElementTree.ParseError):
    pass
details["complete"] = bool(
    details["complete"]
    and not completion.interrupted
    and completion.finished == completion.collected
    and completion.collected > 0
)
details["success"] = bool(
    status == 0 and details["complete"] and details["executed"] and not details["failed"]
)
details.update(
    collected=completion.collected, finished=completion.finished, interrupted=completion.interrupted
)
report.write_text(json.dumps(details), encoding="utf-8")
raise SystemExit(status if status else 0 if details["success"] else 1)
