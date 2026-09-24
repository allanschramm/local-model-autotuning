"""DM-Code-Agent 30-task scoreboard loader and revision guard."""

from __future__ import annotations

import hashlib
import importlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

DEFAULT_SUITE = "all"
PINNED_DM_CODE_AGENT_COMMIT = "e4530be37e6f972d9735381e524cab4b5f732ddd"


def _git_dir_for_package() -> Path | None:
    try:
        package = importlib.import_module("dm_agent")
    except ImportError:
        return None

    roots: list[Path] = []
    package_file = getattr(package, "__file__", None)
    if package_file:
        roots.append(Path(package_file).resolve().parent)
    roots.extend(Path(item).resolve() for item in getattr(package, "__path__", ()))

    for root in roots:
        for parent in (root, *root.parents):
            dotgit = parent / ".git"
            if dotgit.is_dir():
                return dotgit
            if not dotgit.is_file():
                continue
            try:
                value = dotgit.read_text(encoding="utf-8").strip()
            except OSError:
                continue
            if not value.startswith("gitdir:"):
                continue
            target = Path(value.split(":", 1)[1].strip())
            if not target.is_absolute():
                target = parent / target
            return target.resolve()
    return None


def _read_git_commit(git_dir: Path) -> str | None:
    try:
        head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not head.startswith("ref:"):
        return head if head else None

    ref = head.split(":", 1)[1].strip()
    ref_path = git_dir / ref
    try:
        value = ref_path.read_text(encoding="utf-8").strip()
        if value:
            return value
    except OSError:
        pass

    try:
        packed = (git_dir / "packed-refs").read_text(encoding="utf-8")
    except OSError:
        return None
    for line in packed.splitlines():
        if not line or line.startswith(("#", "^")):
            continue
        parts = line.split()
        if len(parts) == 2 and parts[1] == ref:
            return parts[0]
    return None


def upstream_commit() -> str | None:
    git_dir = _git_dir_for_package()
    return _read_git_commit(git_dir) if git_dir is not None else None


def validate_upstream_revision() -> str:
    actual = upstream_commit()
    if actual is None:
        raise FileNotFoundError(
            "DM-Code-Agent must be installed from the pinned editable checkout "
            f"at {PINNED_DM_CODE_AGENT_COMMIT}; its git revision could not be determined."
        )
    if actual.lower() != PINNED_DM_CODE_AGENT_COMMIT:
        raise RuntimeError(
            f"DM-Code-Agent checkout is at {actual}, expected "
            f"{PINNED_DM_CODE_AGENT_COMMIT}; refusing an unpinned benchmark."
        )
    return actual


def _require_dm_agent():
    try:
        module = importlib.import_module("dm_agent.benchmarks.tasks")
        get_benchmark_tasks = module.get_benchmark_tasks
    except (ImportError, AttributeError) as exc:
        raise FileNotFoundError(
            "dm_agent is not importable; install the pinned DM-Code-Agent checkout "
            f"at {PINNED_DM_CODE_AGENT_COMMIT} with `pip install -e <checkout>`."
        ) from exc
    validate_upstream_revision()
    return get_benchmark_tasks


def list_suites() -> list[str]:
    validate_upstream_revision()
    try:
        module = importlib.import_module("dm_agent.benchmarks.tasks")
        suites = module.BENCHMARK_SUITES
    except (ImportError, AttributeError) as exc:
        raise FileNotFoundError(
            f"dm_agent is not importable; install the pinned checkout at "
            f"{PINNED_DM_CODE_AGENT_COMMIT}."
        ) from exc
    return sorted(suites.keys())


def discover_tasks(suite: str = DEFAULT_SUITE) -> list[str]:
    get_benchmark_tasks = _require_dm_agent()
    return [str(task.task_id) for task in get_benchmark_tasks(suite)]


def _task_dict(task: Any, suite: str) -> dict[str, Any]:
    return {
        "id": str(task.task_id),
        "name": str(getattr(task, "name", "")),
        "issue": str(getattr(task, "prompt", "")),
        "max_steps": getattr(task, "max_steps", None),
        "tags": list(getattr(task, "tags", []) or []),
        "visible_test_command": list(
            getattr(task, "visible_test_command", ["{python}", "-m", "pytest", "-q"]) or []
        ),
        "hidden_test_command": list(
            getattr(task, "hidden_test_command", ["{python}", "-m", "pytest", "-q"]) or []
        ),
        "allowed_changed_files": list(getattr(task, "allowed_changed_files", []) or []),
        "required_changed_files": list(getattr(task, "required_changed_files", []) or []),
        "_setup_files": dict(getattr(task, "setup_files", {}) or {}),
        "_hidden_files": dict(getattr(task, "hidden_files", {}) or {}),
        "_suite": suite,
    }


def load_task(task_id: str, suite: str | None = None) -> dict[str, Any]:
    get_benchmark_tasks = _require_dm_agent()
    suites_to_try = [suite] if suite else list_suites()
    for current_suite in suites_to_try:
        for task in get_benchmark_tasks(current_suite):
            if str(task.task_id) == task_id:
                return _task_dict(task, current_suite)
    raise FileNotFoundError(f"DM-Code-Agent task {task_id!r} not found in suites {suites_to_try}")


def suite_signature(tasks: Sequence[dict[str, Any]]) -> str:
    fields = (
        "id",
        "name",
        "issue",
        "max_steps",
        "tags",
        "visible_test_command",
        "hidden_test_command",
        "allowed_changed_files",
        "required_changed_files",
        "_setup_files",
        "_hidden_files",
        "_suite",
    )
    payload = {
        "upstream_commit": PINNED_DM_CODE_AGENT_COMMIT,
        "tasks": [
            {field: task.get(field) for field in fields}
            for task in sorted(tasks, key=lambda item: str(item.get("id", "")))
        ],
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
