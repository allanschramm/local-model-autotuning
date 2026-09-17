"""SWE-lite agentic coding eval: frozen issues, workspace tools, loop detector."""

from __future__ import annotations

import json
import shutil
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any

import yaml

from autoresearch.benchmarks.agentic_coding.detector import (
    ALLOWED_TOOLS,
    HIDDEN_DIR_NAME,
    MUTATING_TOOLS,
    DetectorState,
    resolve_in_worktree,
)
from autoresearch.core.llama_client import GenerationParams, LlamaClient

TASKS_ROOT = Path(__file__).resolve().parent / "tasks"
MAX_TURNS_DEFAULT = 20
RUN_TESTS_TIMEOUT_SEC = 30

TOOL_DEFS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a UTF-8 file in the workspace.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write UTF-8 contents to a workspace file (creates parents).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "contents": {"type": "string"},
                },
                "required": ["path", "contents"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "str_replace",
            "description": "Replace one exact occurrence of old_string with new_string.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_string": {"type": "string"},
                    "new_string": {"type": "string"},
                },
                "required": ["path", "old_string", "new_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Search a path for a substring. path defaults to workspace root.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "pattern": {"type": "string"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List files under a workspace directory (not hidden tests).",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_tests",
            "description": "Run the task's hidden pytest suite. Do not use on docs-only issues.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

SYSTEM_MSG = (
    "You are a coding agent in a local workspace. Fix the single GitHub issue. "
    "Use tools to read and edit files. Call run_tests to check hidden tests when "
    "the issue is a code change. When done, stop calling tools and reply with a "
    "short summary. Do not invent tools. Stay inside the workspace. "
    "Do not edit _hidden_tests."
)


def discover_tasks(root: Path | None = None) -> list[str]:
    base = root or TASKS_ROOT
    if not base.is_dir():
        return []
    ids = []
    for child in sorted(base.iterdir()):
        if child.is_dir() and (child / "task.yaml").is_file():
            ids.append(child.name)
    return ids


def load_task(task_id: str, root: Path | None = None) -> dict[str, Any]:
    task_dir = (root or TASKS_ROOT) / task_id
    data = yaml.safe_load((task_dir / "task.yaml").read_text(encoding="utf-8"))
    data["_dir"] = task_dir
    data["id"] = task_id
    return data


def _prepare_worktree(task: dict[str, Any]) -> Path:
    src = Path(task["_dir"]) / "workspace"
    dest = Path(tempfile.mkdtemp(prefix=f"ac-{task['id']}-"))
    shutil.copytree(src, dest, dirs_exist_ok=True)
    hidden_src = Path(task["_dir"]) / "hidden_tests"
    if hidden_src.is_dir():
        shutil.copytree(hidden_src, dest / HIDDEN_DIR_NAME, dirs_exist_ok=True)
    return dest


def _run_hidden_tests(worktree: Path) -> tuple[bool, str]:
    import subprocess
    import sys

    hidden = worktree / HIDDEN_DIR_NAME
    if not hidden.is_dir():
        return False, "no hidden tests"
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", str(hidden)],
        cwd=str(worktree),
        capture_output=True,
        text=True,
        timeout=RUN_TESTS_TIMEOUT_SEC,
        check=False,
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode == 0, out[-4000:]


def _exec_tool(
    name: str,
    arguments: dict[str, Any],
    worktree: Path,
    detector: DetectorState,
) -> dict[str, Any]:
    if name == "run_tests":
        ok, log = _run_hidden_tests(worktree)
        return {"ok": ok, "output": log}
    rel = str(arguments.get("path") or ".")
    path = resolve_in_worktree(worktree, rel)
    if path is None:
        return {"error": "path rejected"}
    if name == "read_file":
        if not path.is_file():
            return {"error": f"not found: {rel}"}
        return {"contents": path.read_text(encoding="utf-8")}
    if name == "write_file":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(arguments.get("contents") or ""), encoding="utf-8")
        return {"ok": True}
    if name == "str_replace":
        if not path.is_file():
            return {"error": f"not found: {rel}"}
        text = path.read_text(encoding="utf-8")
        old = str(arguments.get("old_string") or "")
        new = str(arguments.get("new_string") or "")
        count = text.count(old)
        if count != 1:
            return {"error": f"str_replace expected 1 match, got {count}"}
        path.write_text(text.replace(old, new, 1), encoding="utf-8")
        return {"ok": True}
    if name == "list_dir":
        if not path.exists():
            return {"error": f"not found: {rel}"}
        target = path if path.is_dir() else path.parent
        names = []
        for child in sorted(target.iterdir()):
            if child.name == HIDDEN_DIR_NAME:
                continue
            names.append(child.name + ("/" if child.is_dir() else ""))
        return {"entries": names}
    if name == "grep":
        pattern = str(arguments.get("pattern") or "")
        hits: list[str] = []
        if path.is_file():
            files = [path]
        else:
            files = [p for p in path.rglob("*") if p.is_file() and HIDDEN_DIR_NAME not in p.parts]
        for fp in files:
            try:
                body = fp.read_text(encoding="utf-8")
            except OSError:
                continue
            if pattern in body:
                rel_fp = fp.relative_to(worktree).as_posix()
                hits.append(rel_fp)
        return {"hits": hits[:50]}
    return {"error": f"unhandled tool {name}"}


def _assistant_visible_text(msg: dict) -> str:
    content = (msg.get("content") or "").strip()
    if content:
        return msg.get("content") or ""
    return msg.get("reasoning_content") or ""


def _assistant_history_message(msg: dict, *, tool_calls: list | None = None) -> dict:
    out: dict = {"role": "assistant", "content": _assistant_visible_text(msg)}
    reasoning = msg.get("reasoning_content")
    if reasoning:
        out["reasoning_content"] = reasoning
    if tool_calls:
        out["tool_calls"] = tool_calls
    return out


def run_task_loop(
    client: LlamaClient,
    task: dict[str, Any],
    gen_params: GenerationParams | None = None,
    *,
    gen: GenerationParams | None = None,
    worktree: Path | None = None,
) -> dict[str, Any]:
    """Run one issue session. Caller may pass a prepared worktree (tests)."""
    gen = gen_params or gen or GenerationParams(max_tokens=2048)
    max_turns = int(task.get("max_turns", MAX_TURNS_DEFAULT))
    allowlisted = tuple(task.get("allowlist") or [])
    docs_only = bool(task.get("docs_only", False))
    own_tree = worktree is None
    tree = worktree or _prepare_worktree(task)
    detector = DetectorState(
        worktree=tree,
        allowlisted=allowlisted
        or tuple(
            p.relative_to(tree).as_posix()
            for p in tree.rglob("*")
            if p.is_file() and HIDDEN_DIR_NAME not in p.parts
        ),
        docs_only=docs_only,
    )
    detector.last_hash = detector.snapshot_hash()

    issue_text = str(task.get("issue") or "")
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": issue_text},
    ]
    t_start = time.time()
    fail_reason: str | None = None
    tests_ok = False
    tests_log = ""

    try:
        for turn in range(max_turns):
            if detector.fail_reason:
                fail_reason = detector.fail_reason
                break
            stop_val = gen.stop if gen and gen.stop is not None else ["</s>"]
            if isinstance(stop_val, str):
                stop_val = [stop_val]
            elif isinstance(stop_val, (tuple, set)):
                stop_val = list(stop_val)
            payload = {
                "messages": messages,
                "tools": TOOL_DEFS,
                "stream": False,
                "max_tokens": gen.max_tokens,
                "temperature": gen.temp,
                "stop": stop_val,
            }
            for key in ("top_p", "top_k", "repeat_penalty"):
                val = getattr(gen, key, None)
                if val is not None:
                    payload[key] = val

            if getattr(gen, "reasoning_effort", None) is not None:
                payload["reasoning_effort"] = gen.reasoning_effort
                chat_kwargs = dict(
                    getattr(gen, "chat_template_kwargs", None)
                    or getattr(gen, "chat_template_args", None)
                    or {}
                )
                chat_kwargs["reasoning_effort"] = gen.reasoning_effort
                payload["chat_template_kwargs"] = chat_kwargs
                payload["chat_template_args"] = chat_kwargs
            elif getattr(gen, "chat_template_kwargs", None) is not None:
                payload["chat_template_kwargs"] = gen.chat_template_kwargs
                payload["chat_template_args"] = gen.chat_template_kwargs
            elif getattr(gen, "chat_template_args", None) is not None:
                payload["chat_template_kwargs"] = gen.chat_template_args
                payload["chat_template_args"] = gen.chat_template_args

            url = f"{client.base_url}/v1/chat/completions"
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            )
            turn_timeout = max(float(getattr(client, "timeout", 420.0) or 420.0), 420.0)
            try:
                with urllib.request.urlopen(req, timeout=turn_timeout) as resp:
                    raw = json.loads(resp.read().decode())
            except Exception as exc:
                fail_reason = f"request_failed:{exc}"
                break

            choice = (raw.get("choices") or [{}])[0]
            msg = choice.get("message", {})
            tool_calls = msg.get("tool_calls") or []

            if tool_calls:
                messages.append(_assistant_history_message(msg, tool_calls=tool_calls))
                mutated = False
                for tc in tool_calls:
                    func = tc.get("function", {})
                    tool_name = func.get("name", "")
                    try:
                        args = json.loads(func.get("arguments") or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    if not isinstance(args, dict):
                        args = {}
                    if tool_name in MUTATING_TOOLS:
                        mutated = True
                    reason = detector.record_tool(tool_name, args)
                    if reason:
                        fail_reason = reason
                        result: dict[str, Any] = {"error": reason}
                    elif tool_name not in ALLOWED_TOOLS:
                        result = {"error": f"unknown tool {tool_name}"}
                    else:
                        result = _exec_tool(tool_name, args, tree, detector)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.get("id", f"call_{turn}"),
                            "content": json.dumps(result),
                        }
                    )
                    if fail_reason:
                        break
                stall = detector.note_turn_hash(mutated=mutated)
                if stall:
                    fail_reason = stall
                    break
                if fail_reason:
                    break
                continue

            messages.append(_assistant_history_message(msg))
            tests_ok, tests_log = _run_hidden_tests(tree)
            if detector.fail_reason:
                fail_reason = detector.fail_reason
            elif not tests_ok:
                fail_reason = "tests_red"
            break
        else:
            tests_ok, tests_log = _run_hidden_tests(tree)
            if detector.fail_reason:
                fail_reason = detector.fail_reason
            elif not tests_ok:
                fail_reason = "max_turns_tests_red"
    finally:
        if own_tree:
            shutil.rmtree(tree, ignore_errors=True)

    elapsed = time.time() - t_start
    passed = tests_ok and fail_reason is None
    return {
        "passed": passed,
        "fail_reason": fail_reason,
        "tests_ok": tests_ok,
        "tests_log": tests_log,
        "elapsed_sec": elapsed,
        "worktree": str(tree),
    }


def run_agentic_coding_eval(
    client: LlamaClient,
    task_ids: list[str] | None = None,
    gen_params: GenerationParams | None = None,
    *,
    tasks_root: Path | None = None,
) -> dict[str, Any]:
    """Run the frozen pack. Returns score passed/total plus per-task detail."""
    ids = task_ids if task_ids is not None else discover_tasks(tasks_root)
    results: list[dict[str, Any]] = []
    passed = 0
    for task_id in ids:
        task = load_task(task_id, tasks_root)
        print(f"    [agentic-coding] {task_id}")
        one = run_task_loop(client, task, gen_params=gen_params)
        one["id"] = task_id
        results.append(one)
        if one["passed"]:
            passed += 1
            print(f"    [agentic-coding] {task_id} PASS")
        else:
            print(f"    [agentic-coding] {task_id} FAIL {one.get('fail_reason')}")
    total = len(ids)
    score = passed / total if total else 0.0
    detail = ",".join(
        f"{r['id']}={'pass' if r['passed'] else r.get('fail_reason') or 'fail'}" for r in results
    )
    return {
        "score": score,
        "passed": passed,
        "total": total,
        "detail": detail,
        "tasks": results,
    }
