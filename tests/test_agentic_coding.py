"""SWE-lite agentic coding detector + stubbed agent loop (ADR 0013)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from unittest.mock import patch
from urllib.request import Request

from autoresearch.benchmarks.agentic_coding.detector import (
    REPEAT_LIMIT,
    DetectorState,
    resolve_in_worktree,
)
from autoresearch.benchmarks.agentic_coding.runner import (
    discover_tasks,
    load_task,
    run_agentic_coding_eval,
    run_task_loop,
)
from autoresearch.core.llama_client import LlamaClient

TASKS = (
    Path(__file__).resolve().parents[1] / "autoresearch" / "benchmarks" / "agentic_coding" / "tasks"
)


def test_discover_five_frozen_tasks():
    ids = discover_tasks()
    assert "issue-43-static-route" in ids
    assert "issue-35-trial-status" in ids
    assert "loop-trap-retry" in ids
    assert len(ids) == 5


def test_resolve_rejects_escape(tmp_path):
    assert resolve_in_worktree(tmp_path, "..") is None
    assert resolve_in_worktree(tmp_path, "C:/Windows/notepad.exe") is None
    inside = resolve_in_worktree(tmp_path, "ok.py")
    assert inside is not None
    assert inside.parent == tmp_path.resolve()


def test_repeat_calls_fail():
    state = DetectorState(worktree=Path("."), allowlisted=("a.py",))
    args = {"path": "a.py"}
    for _ in range(REPEAT_LIMIT - 1):
        assert state.record_tool("read_file", args) is None
    assert state.record_tool("read_file", args) == "loop:repeat_3:read_file"


def test_unknown_tool_is_hallucination():
    state = DetectorState(worktree=Path("."), allowlisted=())
    assert state.record_tool("bash", {}).startswith("hallucination:unknown_tool")


def test_docs_only_run_tests_is_restraint():
    state = DetectorState(worktree=Path("."), allowlisted=("GLOSSARY.md",), docs_only=True)
    assert state.record_tool("run_tests", {}) == "restraint:run_tests_on_docs_only"


def test_path_outside_allowlist_fails(tmp_path):
    (tmp_path / "ok.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "other.py").write_text("y = 2\n", encoding="utf-8")
    state = DetectorState(worktree=tmp_path, allowlisted=("ok.py",))
    assert state.record_tool("read_file", {"path": "ok.py"}) is None
    assert state.record_tool("list_dir", {"path": "."}) is None
    reason = state.record_tool("read_file", {"path": "other.py"})
    assert reason == "hallucination:path_not_allowlisted:other.py"
    state2 = DetectorState(worktree=tmp_path, allowlisted=("ok.py",))
    assert state2.record_tool("write_file", {"path": "other.py", "contents": "z"}) == (
        "hallucination:path_not_allowlisted:other.py"
    )


def test_read_only_turns_do_not_stall(tmp_path):
    (tmp_path / "a.py").write_text("n = 0\n", encoding="utf-8")
    state = DetectorState(worktree=tmp_path, allowlisted=("a.py",))
    state.last_hash = state.snapshot_hash()
    for _ in range(5):
        assert state.note_turn_hash(mutated=False) is None


def test_mutating_turns_without_hash_change_stall(tmp_path):
    (tmp_path / "a.py").write_text("n = 0\n", encoding="utf-8")
    state = DetectorState(worktree=tmp_path, allowlisted=("a.py",))
    state.last_hash = state.snapshot_hash()
    reason = None
    for i in range(3):
        assert (
            state.record_tool(
                "str_replace",
                {
                    "path": "a.py",
                    "old_string": f"missing-{i}",
                    "new_string": "x",
                },
            )
            is None
        )
        reason = state.note_turn_hash(mutated=True)
    assert reason == "stall:3_turns_no_file_change"


def _completion(tool_calls=None, content=""):
    msg = {"content": content, "tool_calls": tool_calls or []}
    return {"choices": [{"message": msg}]}


class _Resp:
    def __init__(self, payload: dict):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(self._payload).encode()


def test_looping_stub_fails_repeat_detector(tmp_path):
    task = load_task("loop-trap-retry")
    tree = tmp_path / "ws"
    src = TASKS / "loop-trap-retry" / "workspace"
    hidden = TASKS / "loop-trap-retry" / "hidden_tests"
    tree.mkdir()
    (tree / "retry.py").write_text((src / "retry.py").read_text(encoding="utf-8"), encoding="utf-8")
    dest_h = tree / "_hidden_tests"
    dest_h.mkdir()
    (dest_h / "test_retry.py").write_text(
        (hidden / "test_retry.py").read_text(encoding="utf-8"), encoding="utf-8"
    )

    call = {
        "id": "c1",
        "function": {"name": "read_file", "arguments": json.dumps({"path": "retry.py"})},
    }
    payloads = [_completion([call]) for _ in range(5)]

    def fake_open(req: Request, timeout=None):
        del req, timeout
        return _Resp(payloads.pop(0) if payloads else _completion(content="done"))

    client = LlamaClient(9)
    with patch("autoresearch.benchmarks.agentic_coding.runner.urllib.request.urlopen", fake_open):
        out = run_task_loop(client, task, worktree=tree)
    assert out["passed"] is False
    assert out["fail_reason"]
    assert out["fail_reason"].startswith("loop:repeat_") or out["fail_reason"].startswith("stall:")


def test_fixing_stub_passes_loop_trap(tmp_path):
    task = load_task("loop-trap-retry")
    tree = tmp_path / "ws"
    src = TASKS / "loop-trap-retry" / "workspace"
    hidden = TASKS / "loop-trap-retry" / "hidden_tests"
    tree.mkdir()
    (tree / "retry.py").write_text((src / "retry.py").read_text(encoding="utf-8"), encoding="utf-8")
    dest_h = tree / "_hidden_tests"
    dest_h.mkdir()
    (dest_h / "test_retry.py").write_text(
        (hidden / "test_retry.py").read_text(encoding="utf-8"), encoding="utf-8"
    )

    write = {
        "id": "c1",
        "function": {
            "name": "write_file",
            "arguments": json.dumps({"path": "retry.py", "contents": "N = 1\n"}),
        },
    }
    payloads = [_completion([write]), _completion(content="fixed")]

    def fake_open(req: Request, timeout=None):
        del req, timeout
        return _Resp(payloads.pop(0))

    client = LlamaClient(9)
    with patch("autoresearch.benchmarks.agentic_coding.runner.urllib.request.urlopen", fake_open):
        out = run_task_loop(client, task, worktree=tree)
    assert out["passed"] is True
    assert out["fail_reason"] is None
    assert (tree / "retry.py").read_text(encoding="utf-8").strip() == "N = 1"


def test_run_eval_score_with_stub_pack(tmp_path):
    pack = tmp_path / "pack"
    src = TASKS / "loop-trap-retry"
    dest = pack / "loop-trap-retry"
    dest.mkdir(parents=True)
    (dest / "task.yaml").write_text(
        (src / "task.yaml").read_text(encoding="utf-8"), encoding="utf-8"
    )
    shutil.copytree(src / "workspace", dest / "workspace")
    shutil.copytree(src / "hidden_tests", dest / "hidden_tests")

    def fake_open(req: Request, timeout=None):
        del req, timeout
        return _Resp(_completion(content="I am done"))

    client = LlamaClient(9)
    with patch("autoresearch.benchmarks.agentic_coding.runner.urllib.request.urlopen", fake_open):
        out = run_agentic_coding_eval(client, tasks_root=pack)
    assert out["total"] == 1
    assert out["passed"] == 0
    assert out["score"] == 0.0
    assert "tests_red" in (out["detail"] or "")
