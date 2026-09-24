"""mini-swe-agent driver for the DM-Code-Agent scoreboard."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from autoresearch.benchmarks.mini_swe_agent.dm_code_agent import (
    DEFAULT_SUITE,
    PINNED_DM_CODE_AGENT_COMMIT,
    discover_tasks,
    load_task,
    suite_signature,
)
from autoresearch.core.llama_client import GenerationParams
from autoresearch.core.process_guard import ProcessGuard

SKILLS_DIR = Path(__file__).resolve().parent / "skills"
PLAN_FIRST_SKILL = SKILLS_DIR / "plan-first.md"
CONFIG_FILENAME = ".mini-swe-agent-config.yaml"
TASK_TRAJECTORY_FILENAME = ".mini-swe-agent-task.traj.json"
SYSTEMIC_NO_PROGRESS_LIMIT = 2
LOG_DIR = Path(__file__).resolve().parents[2] / "runners" / "logs"
LOG_KEEP = 20
IGNORED_WORKTREE_PARTS = frozenset(
    {".git", ".pytest_cache", "__pycache__", ".mypy_cache", ".ruff_cache"}
)


def _prune_glob(directory: Path, pattern: str, keep: int) -> None:
    if not directory.exists():
        return
    paths = sorted(directory.glob(pattern), key=lambda path: path.stat().st_mtime)
    for path in paths[:-keep] if keep > 0 else paths:
        try:
            path.unlink()
        except OSError:
            pass


def _new_trajectory_path(model_filename: str) -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    _prune_glob(LOG_DIR, "mini-swe-agent-*.jsonl", LOG_KEEP - 1)
    stem = Path(model_filename).stem if model_filename else "unknown"
    return LOG_DIR / f"mini-swe-agent-{time.strftime('%Y%m%d-%H%M%S')}-{stem}.jsonl"


def _subprocess_no_window_kwargs() -> dict[str, Any]:
    if sys.platform != "win32":
        return {}
    return {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)}


def _ensure_uvx_available() -> None:
    if shutil.which("uvx") is None:
        raise FileNotFoundError(
            "`uvx` is not on PATH. Install it via `pip install uv` or "
            "`pipx install uv`; mini-swe-agent needs uvx for its ephemeral venv."
        )


def _ensure_skill_present() -> None:
    if not PLAN_FIRST_SKILL.is_file():
        raise FileNotFoundError(
            f"plan-first skill missing at {PLAN_FIRST_SKILL}. "
            "Re-create it from the recipe in `docs/discovery/mini-swe-agent-benchmark.md`."
        )


def _normalize_model_name(value: str | None) -> str:
    name = str(value or "").strip()
    if not name:
        return ""
    if name.lower().startswith("openai/"):
        return name
    return f"openai/{name}"


def _resolve_model_name(env: dict[str, str], fallback: str | None) -> str:
    candidate = fallback or env.get("MSWEA_MODEL_NAME") or os.environ.get("MSWEA_MODEL_NAME")
    return _normalize_model_name(candidate)


def _generation_kwargs(gen_params: GenerationParams | None) -> dict[str, Any]:
    params = {} if gen_params is None else gen_params.to_payload()
    return {
        "drop_params": True,
        "parallel_tool_calls": False,
        "tool_choice": "required",
        **params,
    }


def _task_trajectory_status(path: Path) -> str:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    info = payload.get("info") if isinstance(payload, dict) else None
    status = info.get("exit_status") if isinstance(info, dict) else None
    return str(status) if status else ""


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _run_guarded(
    command: Sequence[str],
    *,
    cwd: Path,
    env: dict[str, str] | None,
    timeout_sec: int,
) -> tuple[int | None, str, str, bool]:
    guard = ProcessGuard()
    try:
        proc = guard.spawn(
            list(command),
            cwd=str(cwd),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            **_subprocess_no_window_kwargs(),
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout_sec)
        except subprocess.TimeoutExpired as exc:
            guard.teardown()
            guard = None
            stdout, stderr = proc.communicate()
            return (
                None,
                _as_text(stdout or exc.stdout),
                _as_text(stderr or exc.stderr),
                True,
            )
        return proc.returncode, _as_text(stdout), _as_text(stderr), False
    finally:
        if guard is not None:
            guard.teardown()


def _resolve_command(command: Sequence[str] | str, worktree: Path) -> list[str]:
    parts = shlex.split(command) if isinstance(command, str) else [str(part) for part in command]
    resolved: list[str] = []
    for part in parts:
        if part == "{python}":
            resolved.append(sys.executable)
        elif part in {"{worktree}", "{workspace}"}:
            resolved.append(str(worktree))
        else:
            resolved.append(part)
    return resolved


def _run_hidden_tests(
    worktree: Path,
    command: Sequence[str] | str | None = None,
    timeout_sec: int = 60,
) -> tuple[bool, str]:
    raw_command = command or ["{python}", "-m", "pytest", "-q"]
    resolved = _resolve_command(raw_command, worktree)
    try:
        return_code, stdout, stderr, timed_out = _run_guarded(
            resolved,
            cwd=worktree,
            env=None,
            timeout_sec=timeout_sec,
        )
    except OSError as exc:
        return False, f"pytest could not start: {exc}"
    output = f"{stdout}{stderr}"
    if timed_out:
        return False, f"pytest timeout after {timeout_sec}s: {output[-2000:]}"
    return return_code == 0, output[-4000:]


def _safe_join(dest: Path, rel_path: str) -> Path | None:
    from autoresearch.benchmarks.agentic_coding import resolve_in_worktree

    return resolve_in_worktree(dest, str(rel_path))


def _prepare_worktree(task: dict[str, Any]) -> Path:
    dest = Path(tempfile.mkdtemp(prefix=f"msa-{task['id']}-"))
    for rel_path, contents in task.get("_setup_files", {}).items():
        target = _safe_join(dest, rel_path)
        if target is None:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(str(contents), encoding="utf-8")
    return dest


def _stage_hidden_tests(worktree: Path, task: dict[str, Any]) -> bool:
    files = task.get("_hidden_files", {}) or {}
    staged = 0
    for rel_path, contents in files.items():
        target = _safe_join(worktree, rel_path)
        if target is None:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(str(contents), encoding="utf-8")
        staged += 1
    return staged == len(files)


def _snapshot_workspace(worktree: Path) -> dict[str, bytes]:
    snapshot: dict[str, bytes] = {}
    for path in worktree.rglob("*"):
        if path.is_symlink() or not path.is_file():
            continue
        relative = path.relative_to(worktree).as_posix()
        if relative in {CONFIG_FILENAME, TASK_TRAJECTORY_FILENAME}:
            continue
        if any(part in IGNORED_WORKTREE_PARTS for part in Path(relative).parts):
            continue
        try:
            snapshot[relative] = path.read_bytes()
        except OSError:
            continue
    return snapshot


def _diff_workspace(before: dict[str, bytes], after: dict[str, bytes]) -> list[str]:
    return sorted(path for path in set(before) | set(after) if before.get(path) != after.get(path))


def _normalized_path(value: Any) -> str:
    return str(value).replace("\\", "/").strip("/")


def _scope_failure(task: dict[str, Any], changed_files: Sequence[str]) -> str | None:
    allowed = {
        _normalized_path(path)
        for path in task.get("allowed_changed_files", []) or []
        if str(path).strip()
    }
    if allowed:
        violations = [path for path in changed_files if _normalized_path(path) not in allowed]
        if violations:
            return "changed files outside allowed set: " + ", ".join(violations)

    required = {
        _normalized_path(path)
        for path in task.get("required_changed_files", []) or []
        if str(path).strip()
    }
    missing = sorted(path for path in required if _normalized_path(path) not in changed_files)
    if missing:
        return "required files were not changed: " + ", ".join(missing)
    return None


def _task_step_limit(task: dict[str, Any]) -> int:
    try:
        value = int(task.get("max_steps") or 0)
    except (TypeError, ValueError):
        return 0
    return value if value > 0 else 0


def _task_has_scope(task: dict[str, Any]) -> bool:
    return bool(task.get("allowed_changed_files") or task.get("required_changed_files"))


def _mini_swe_agent_config(
    worktree: Path,
    task: dict[str, Any],
    *,
    model_name: str,
    gen_params: GenerationParams | None,
    max_wall_sec: int,
) -> Path:
    import yaml

    if _task_has_scope(task):
        allowed = ", ".join(f"`{path}`" for path in task.get("allowed_changed_files", []) or [])
        required = ", ".join(f"`{path}`" for path in task.get("required_changed_files", []) or [])
        if allowed:
            scope_text = f"You may create or modify only these files: {allowed}."
        else:
            scope_text = f"These files must be changed: {required}."
        instance_template = (
            "Please solve this issue: {{task}}\n\n"
            "Keep a short plan in your response before editing. Do not create TODO.md "
            f"or any other planning file. {scope_text} Run the project's tests and finish "
            "with exactly: `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`."
        )
    else:
        instance_template = (
            "Please solve this issue: {{task}}\n\n"
            "Before editing, write TODO.md at the workspace root with 2-5 concrete steps. "
            "Execute the plan, run pytest when available, and finish with exactly: "
            "`echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`."
        )

    config: dict[str, Any] = {
        "agent": {
            "agent_class": "default",
            "system_template": (
                "You are a helpful coding agent working in a temporary repository. "
                "Inspect the files, make the smallest correct change, and verify it."
            ),
            "instance_template": instance_template,
            "step_limit": _task_step_limit(task),
            "wall_time_limit_seconds": max(0, int(max_wall_sec)),
            "max_consecutive_format_errors": 1,
            "output_path": str(worktree / TASK_TRAJECTORY_FILENAME),
        },
        "model": {
            "model_name": model_name,
            "model_kwargs": _generation_kwargs(gen_params),
        },
    }

    dest = worktree / CONFIG_FILENAME
    dest.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False, width=10_000),
        encoding="utf-8",
    )
    return dest


def _compose_prompt(task: dict[str, Any]) -> str:
    task_id = str(task.get("id") or task.get("name") or "unknown")
    prompt = str(task.get("issue") or "").strip()
    if not prompt:
        prompt = f"Complete DM-Code-Agent task {task_id} in the current worktree."
    if _task_has_scope(task):
        return prompt
    try:
        plan_first_text = PLAN_FIRST_SKILL.read_text(encoding="utf-8")
    except OSError:
        plan_first_text = ""
    return (
        prompt if not plan_first_text else prompt + "\n\n# Skill: plan-first\n\n" + plan_first_text
    )


def _run_one_task(
    task: dict[str, Any],
    *,
    model_name: str,
    base_url: str,
    api_key: str,
    trajectory_path: Path,
    gen_params: GenerationParams | None = None,
    suite_sig: str = "",
    max_wall_sec: int = 600,
) -> dict[str, Any]:
    t_start = time.time()
    worktree = _prepare_worktree(task)
    task_id = str(task.get("id") or task.get("name") or "unknown")
    fail_reason: str | None = None
    tests_ok = False
    tests_log = ""
    changed_files: list[str] = []
    mini_returncode: int | None = None
    mini_exit_status = ""
    mini_stdout_tail = ""
    mini_stderr_tail = ""
    elapsed = 0.0

    try:
        prompt = _compose_prompt(task)
        env = os.environ.copy()
        env["OPENAI_BASE_URL"] = base_url
        env["OPENAI_API_KEY"] = api_key
        env["MSWEA_CONFIGURED"] = "1"
        env["MSWEA_COST_TRACKING"] = "ignore_errors"
        env["MSWEA_MODEL_NAME"] = _normalize_model_name(model_name)
        mini_cfg = _mini_swe_agent_config(
            worktree,
            task,
            model_name=_normalize_model_name(model_name),
            gen_params=gen_params,
            max_wall_sec=max_wall_sec,
        )
        before = _snapshot_workspace(worktree)
        command = [
            "uvx",
            "mini-swe-agent",
            "--task",
            prompt,
            "--model",
            _normalize_model_name(model_name),
            "-c",
            str(mini_cfg),
            "--output",
            str(worktree / TASK_TRAJECTORY_FILENAME),
        ]
        try:
            (
                mini_returncode,
                mini_stdout_tail,
                mini_stderr_tail,
                timed_out,
            ) = _run_guarded(
                command,
                cwd=worktree,
                env=env,
                timeout_sec=max_wall_sec,
            )
            mini_stdout_tail = mini_stdout_tail[-2000:]
            mini_stderr_tail = mini_stderr_tail[-2000:]
            if timed_out:
                fail_reason = f"mini_timeout_{max_wall_sec}s"
            elif mini_returncode not in (0, None):
                fail_reason = f"mini_exit_{mini_returncode}"
        except FileNotFoundError as exc:
            fail_reason = f"mini_missing:{exc}"
        except Exception as exc:
            fail_reason = f"mini_error:{str(exc)[:200]}"

        mini_exit_status = _task_trajectory_status(worktree / TASK_TRAJECTORY_FILENAME)
        if fail_reason is None and mini_exit_status == "RepeatedFormatError":
            fail_reason = "mini_repeated_format_error"

        after = _snapshot_workspace(worktree)
        changed_files = _diff_workspace(before, after)
        if fail_reason is None:
            fail_reason = _scope_failure(task, changed_files)

        try:
            staged = _stage_hidden_tests(worktree, task)
        except OSError as exc:
            staged = False
            tests_log = f"hidden test staging failed: {exc}"
        if not staged and fail_reason is None:
            fail_reason = "hidden_stage_failed"
        try:
            tests_ok, verifier_log = _run_hidden_tests(
                worktree,
                command=task.get("hidden_test_command"),
            )
            tests_log = verifier_log
        except Exception as exc:
            tests_ok = False
            tests_log = f"hidden test runner failed: {str(exc)[:200]}"
        if fail_reason is None and not tests_ok:
            fail_reason = "tests_red"
    finally:
        elapsed = time.time() - t_start
        line = {
            "task_id": task_id,
            "elapsed_sec": round(elapsed, 3),
            "passed": bool(tests_ok and fail_reason is None),
            "fail_reason": fail_reason,
            "mini_returncode": mini_returncode,
            "mini_exit_status": mini_exit_status,
            "changed_files": changed_files,
            "suite_signature": suite_sig,
            "upstream_commit": PINNED_DM_CODE_AGENT_COMMIT,
            "stdout_tail": mini_stdout_tail,
            "stderr_tail": mini_stderr_tail,
            "tests_log_tail": tests_log[-2000:] if tests_log else "",
        }
        try:
            with trajectory_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(line) + "\n")
        except OSError:
            pass
        try:
            shutil.rmtree(worktree, ignore_errors=True)
        except OSError:
            pass

    return {
        "id": task_id,
        "passed": bool(tests_ok and fail_reason is None),
        "fail_reason": fail_reason,
        "elapsed_sec": elapsed,
        "tests_log": tests_log[-2000:] if tests_log else "",
        "changed_files": changed_files,
        "mini_returncode": mini_returncode,
        "mini_exit_status": mini_exit_status,
        "suite_signature": suite_sig,
        "upstream_commit": PINNED_DM_CODE_AGENT_COMMIT,
    }


def _is_no_progress_format_error(result: dict[str, Any]) -> bool:
    return (
        result.get("mini_returncode") in (0, None)
        and result.get("mini_exit_status") == "RepeatedFormatError"
        and not result.get("changed_files")
    )


def run_mini_swe_agent_eval(
    *,
    base_url: str,
    api_key: str = "sk-no-auth-required",
    model_name: str | None = None,
    model_filename: str = "",
    task_ids: list[str] | None = None,
    suite: str = DEFAULT_SUITE,
    gen_params: GenerationParams | None = None,
    max_wall_sec: int = 600,
) -> dict[str, Any]:
    _ensure_uvx_available()
    _ensure_skill_present()
    resolved_model = _resolve_model_name({}, model_name or model_filename)
    if not resolved_model:
        raise ValueError(
            "MSWEA_MODEL_NAME must be set (env or arg) so mini-swe-agent "
            "knows which model to call. Set it to the model name reported "
            "by GET /v1/models."
        )

    ids = list(task_ids) if task_ids is not None else discover_tasks(suite)
    if not ids:
        raise FileNotFoundError(
            f"No DM-Code-Agent tasks resolved for suite={suite!r}. "
            f"Is the pinned dm_agent checkout at {PINNED_DM_CODE_AGENT_COMMIT} installed?"
        )

    tasks: list[dict[str, Any]] = []
    task_suite = None if task_ids is not None and suite == DEFAULT_SUITE else suite
    for task_id in ids:
        try:
            tasks.append(load_task(task_id, suite=task_suite))
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"DM-Code-Agent task {task_id!r} could not be loaded from suite={suite!r}"
            ) from exc
    suite_sig = suite_signature(tasks)
    trajectory_path = _new_trajectory_path(model_filename or resolved_model)
    results: list[dict[str, Any]] = []
    passed = 0
    no_progress_format_errors = 0

    print(
        f"  [mini-swe-agent] DM-Code-Agent {len(tasks)} task(s) via mini-swe-agent "
        f"(suite={suite}, signature={suite_sig[:12]})"
    )
    print(f"  [mini-swe-agent] trajectory: {trajectory_path}")
    print(f"  [mini-swe-agent] model={resolved_model} base_url={base_url}")
    for task in tasks:
        task_id = str(task["id"])
        print(f"    [mini-swe-agent] {task_id}")
        one = _run_one_task(
            task,
            model_name=resolved_model,
            base_url=base_url,
            api_key=api_key,
            trajectory_path=trajectory_path,
            gen_params=gen_params,
            suite_sig=suite_sig,
            max_wall_sec=max_wall_sec,
        )
        results.append(one)
        if one["passed"]:
            passed += 1
            print(f"    [mini-swe-agent] {task_id} PASS ({one['elapsed_sec']:.1f}s)")
        else:
            print(f"    [mini-swe-agent] {task_id} FAIL {one.get('fail_reason') or 'unknown'}")
        if _is_no_progress_format_error(one):
            no_progress_format_errors += 1
        else:
            no_progress_format_errors = 0
        if no_progress_format_errors >= SYSTEMIC_NO_PROGRESS_LIMIT:
            task_ids = [str(result["id"]) for result in results[-SYSTEMIC_NO_PROGRESS_LIMIT:]]
            raise RuntimeError(
                "mini-swe-agent hit RepeatedFormatError with no workspace changes on "
                f"{SYSTEMIC_NO_PROGRESS_LIMIT} consecutive tasks ({', '.join(task_ids)}); "
                "aborting before the full-suite time sink"
            )

    total = len(tasks)
    score = passed / total if total else 0.0
    detail = ",".join(
        f"{result['id']}={'pass' if result['passed'] else result.get('fail_reason') or 'fail'}"
        for result in results
    )
    return {
        "score": score,
        "passed": passed,
        "total": total,
        "detail": detail,
        "tasks": results,
        "trajectory_path": str(trajectory_path),
        "suite_signature": suite_sig,
        "upstream_commit": PINNED_DM_CODE_AGENT_COMMIT,
    }
