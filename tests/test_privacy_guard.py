"""Privacy guard: tracked files must carry no operator PII.

Public repo. Root ``AGENTS.md``: "Never commit private paths, emails, hostnames,
GPU SKUs, aliases, or machine Baseline"; ``docs/AGENTS.md`` repeats it. E2E
cannot catch this class — no harness run inspects the repo for PII — and the
leak recurred (operator first name in a code comment, a personal profile path in
two session logs and discovery note, the operator GPU SKU in model cards).
Fail-closed guard, same shape as the ``fingerprint.py`` private-value check.
"""

from __future__ import annotations

import re
import subprocess
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Third-party calibration corpora: foreign names are expected there.
SKIP_DIRS = ("autoresearch/data/", "data/")
# This file holds the patterns themselves.
SKIP_FILES = ("tests/test_privacy_guard.py",)

TEXT_SUFFIXES = frozenset(
    {
        ".py", ".md", ".sh", ".ps1", ".cmd", ".bat", ".yml", ".yaml", ".json",
        ".toml", ".cfg", ".ini", ".txt", ".example", ".tsv", ".txt",
    }
)

PATTERNS = {
    "windows user profile path": re.compile(r"[A-Za-z]:[\\/]Users[\\/]", re.I),
    # `user` is the sanitized placeholder the repo's own docs/tests use.
    "posix user profile path": re.compile(r"/(?:home|Users)/(?!user(?:/|\b))[a-z][\w.-]*/"),
    "AppData": re.compile(r"AppData"),
    "operator first name": re.compile(r"(?<![a-z])allan(?![a-z])", re.I),
    # `RTX 4060 Ti` in an external citation is a different card.
    "operator GPU SKU": re.compile(r"RTX\s*4060(?!\s*Ti)"),
}


def tracked_text_files() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files"], cwd=REPO, check=True, capture_output=True, text=True
    ).stdout
    return [
        Path(rel)
        for rel in out.splitlines()
        if rel not in SKIP_FILES
        and not rel.startswith(SKIP_DIRS)
        and Path(rel).suffix.lower() in TEXT_SUFFIXES
    ]


class TestPrivacyGuard(unittest.TestCase):
    def test_no_operator_pii_in_tracked_files(self):
        hits = []
        for rel in tracked_text_files():
            text = (REPO / rel).read_text(encoding="utf-8", errors="replace")
            for line_no, line in enumerate(text.splitlines(), 1):
                for label, pattern in PATTERNS.items():
                    if pattern.search(line):
                        hits.append(f"{rel}:{line_no}: {label}: {line.strip()[:110]}")
        self.assertEqual([], hits, "operator PII in tracked files:\n" + "\n".join(hits))


if __name__ == "__main__":
    unittest.main()