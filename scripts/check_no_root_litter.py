#!/usr/bin/env python3
"""check_no_root_litter.py — fail closed on throwaway files in the repo root.

The repo root is a **closed list**. Everything that belongs to the project is
either a tracked file or an entry in :data:`ALLOWED_DOTFILES`. Anything else —
a throwaway probe script, a redirected log, a scratch note — is agent litter
that accumulated because nothing failed when it was left behind.

Why this exists (measured, not hypothetical): during the Rust refactor a
single session left 17 such files in the root (``.cmp.py``, ``.ctx.log``,
``.mread.py``, ``.t.log``, ``.w.py``, …). ``*.log`` was already gitignored, so
``git status`` stayed clean and nothing failed — which is exactly why the
litter was invisible until a human looked at the directory. A guard that
cannot see the problem is not a guard.

Same philosophy as ``check_no_tsv_readers.py`` (allowlist) and the
anti-agent containment gate (fail closed, remediation written for the agent
that tripped it). No Rust toolchain needed: pure filesystem + stdlib.

Usage::

    python scripts/check_no_root_litter.py
    python scripts/check_no_root_litter.py --json

Exit codes: 0 = clean · 1 = litter found (fail closed) · 2 = bad usage.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Root files that are *not* litter: the project's own entry points.
ALLOWED_ROOT_PY: frozenset[str] = frozenset({"autoloop.py", "benchmark_search.py"})

#: Hidden root files that legitimately exist. Anything hidden and not here is
#: litter. (VCS dirs are directories, never matched — this only sees files.)
ALLOWED_DOTFILES: frozenset[str] = frozenset(
    {
        ".gitignore",
        ".gitattributes",
        ".pre-commit-config.yaml",
        ".env",
        ".env.example",
        ".python-version",
    }
)

#: Suffixes that identify a throwaway artifact. Dotfiles only — a tracked
#: source file with a real name is handled by the rest of the toolchain.
LITTER_SUFFIXES: tuple[str, ...] = (".py", ".log", ".err", ".txt", ".json", ".tmp")

#: Where scratch work belongs. Gitignored, already in use by the wayfinder map.
SCRATCH_DIR = ".scratch/tmp"

REMEDIATION = (
    f"Move it to {SCRATCH_DIR}/ (gitignored) or delete it. "
    f"Do not leave throwaway artifacts in the repo root — `*.log` being "
    f"gitignored hid this litter from `git status`, so nothing failed when "
    f"it accumulated."
)


def _force_utf8() -> None:
    """The Windows console is cp1252 and raises on non-ASCII output.

    Without this the guard crashes *at the moment it reports a violation* —
    i.e. exactly when it most needs to speak. Crashing while finding a problem
    is fail-open wearing a fail-closed costume.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass


def _is_litter(path: Path) -> str | None:
    """Return the rule name if this root file is litter, else None."""
    name = path.name
    if path.is_dir():
        return None
    if name.startswith("."):
        if name in ALLOWED_DOTFILES:
            return None
        if path.suffix.lower() in LITTER_SUFFIXES:
            return "root-dotfile-litter"
        return None
    if path.suffix == ".py" and name not in ALLOWED_ROOT_PY:
        return "root-py-litter"
    return None


def scan() -> list[dict[str, str]]:
    """Return a list of ``{path, rule, remediation}`` for every litter file."""
    findings: list[dict[str, str]] = []
    for path in sorted(REPO_ROOT.iterdir(), key=lambda p: p.name):
        rule = _is_litter(path)
        if rule is not None:
            findings.append({"path": path.name, "rule": rule, "remediation": REMEDIATION})
    return findings


def main(argv: list[str] | None = None) -> int:
    _force_utf8()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)

    findings = scan()
    if args.json:
        print(json.dumps({"ok": not findings, "findings": findings}, indent=2))
        return 0 if not findings else 1

    if not findings:
        print(f"OK: repo root is clean (scratch belongs in {SCRATCH_DIR}/).")
        return 0

    print("ERROR: throwaway artifacts in the repo root:", file=sys.stderr)
    for f in findings:
        print(f"  {f['path']}: [{f['rule']}]", file=sys.stderr)
        print(f"    → {f['remediation']}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
