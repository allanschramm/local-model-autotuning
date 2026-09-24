"""mini-swe-agent driver and DM-Code-Agent loader tests."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml

from autoresearch.benchmarks.mini_swe_agent import (
    discover_tasks,
    list_suites,
    load_task,
    run_mini_swe_agent_eval,
)
from autoresearch.benchmarks.mini_swe_agent import dm_code_agent as loader
from autoresearch.benchmarks.mini_swe_agent import runner as msa_runner
from autoresearch.benchmarks.mini_swe_agent.runner import PLAN_FIRST_SKILL
from autoresearch.core.llama_client import GenerationParams


@pytest.fixture(autouse=True)
def _sandbox_log_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(msa_runner, "LOG_DIR", tmp_path / "logs")


def _fake_task(task_id: str = "config_precedence") -> dict:
    return {
        "id": task_id,
        "name": task_id,
        "issue": "Fix config precedence: later keys must win. See config.py.",
        "max_steps": 10,
        "tags": [],
        "visible_test_command": ["{python}", "-m", "pytest", "-q"],
        "hidden_test_command": ["{python}", "-m", "pytest", "-q"],
        "allowed_changed_files": [],
        "required_changed_files": [],
        "_setup_files": {"config.py": "PRECEDENCE = True\n"},
        "_hidden_files": {"test_config.py": "def test_ok():\n    assert True\n"},
        "_suite": "maintenance",
    }


def _scoped_task() -> dict:
    task = _fake_task()
    task["allowed_changed_files"] = ["config.py"]
    task["required_changed_files"] = ["config.py"]
    return task


def test_skill_file_is_present():
    assert PLAN_FIRST_SKILL.is_file()
    text = PLAN_FIRST_SKILL.read_text(encoding="utf-8")
    assert "Step" in text or "## " in text


def test_skill_dir_is_harness_local():
    text = PLAN_FIRST_SKILL.read_text(encoding="utf-8")
    assert "esagduyu" in text
    assert "qwen-pi-plan-vs-haiku-benchmark" in text


def test_list_suites_exposes_coding_maintenance_all():
    pytest.importorskip("dm_agent.benchmarks.tasks")
    suites = list_suites()
    assert "coding" in suites
    assert "maintenance" in suites
    assert "all" in suites


def test_discover_tasks_returns_known_task_ids():
    pytest.importorskip("dm_agent.benchmarks.tasks")
    tasks = discover_tasks("maintenance")
    assert isinstance(tasks, list)
    ids = set(tasks)
    assert "config_precedence" in ids
    assert "patch_summary_name_status" in ids
    assert "retry_regression_tests" in ids


def test_load_task_preserves_upstream_contract(monkeypatch):
    task = SimpleNamespace(
        task_id="contract",
        name="Contract",
        prompt="Fix it",
        setup_files={"src.py": "x = 1\n"},
        hidden_files={"tests/test_hidden.py": "def test_ok(): pass\n"},
        visible_test_command=["{python}", "-m", "pytest", "-q", "tests"],
        hidden_test_command=["{python}", "-m", "pytest", "-q", "tests/test_hidden.py"],
        max_steps=17,
        tags=["contract"],
        allowed_changed_files=["src.py"],
        required_changed_files=["src.py"],
    )
    monkeypatch.setattr(loader, "_require_dm_agent", lambda: lambda _suite: [task])
    loaded = loader.load_task("contract", suite="maintenance")
    assert loaded["visible_test_command"] == task.visible_test_command
    assert loaded["hidden_test_command"] == task.hidden_test_command
    assert loaded["allowed_changed_files"] == ["src.py"]
    assert loaded["required_changed_files"] == ["src.py"]
    assert loaded["max_steps"] == 17


def test_suite_signature_changes_with_task_contract():
    first = [_fake_task()]
    second = [_fake_task()]
    second[0]["_hidden_files"] = {"test_config.py": "def test_changed(): pass\n"}
    assert loader.suite_signature(first) == loader.suite_signature([_fake_task()])
    assert loader.suite_signature(first) != loader.suite_signature(second)


def test_upstream_revision_is_validated_against_pin(tmp_path, monkeypatch):
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (git_dir / "refs" / "heads").mkdir(parents=True)
    (git_dir / "refs" / "heads" / "main").write_text(
        loader.PINNED_DM_CODE_AGENT_COMMIT + "\n", encoding="utf-8"
    )
    monkeypatch.setattr(loader, "_git_dir_for_package", lambda: git_dir)
    assert loader.upstream_commit() == loader.PINNED_DM_CODE_AGENT_COMMIT
    assert loader.validate_upstream_revision() == loader.PINNED_DM_CODE_AGENT_COMMIT
    (git_dir / "refs" / "heads" / "main").write_text("0" * 40 + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="refusing an unpinned benchmark"):
        loader.validate_upstream_revision()


def test_load_task_missing_raises():
    pytest.importorskip("dm_agent.benchmarks.tasks")
    with pytest.raises(FileNotFoundError):
        load_task("does-not-exist")


def test_run_mini_swe_agent_requires_uvx():
    with patch.object(msa_runner.shutil, "which", return_value=None):
        with pytest.raises(FileNotFoundError, match="uvx"):
            run_mini_swe_agent_eval(
                base_url="http://127.0.0.1:18080/v1",
                api_key="sk-x",
                model_name="m",
                task_ids=["any"],
            )


def test_run_mini_swe_agent_requires_model_name():
    with (
        patch.object(msa_runner.shutil, "which", return_value="uvx"),
        patch.dict(os.environ, {"MSWEA_MODEL_NAME": ""}),
    ):
        with pytest.raises(ValueError, match="MSWEA_MODEL_NAME"):
            run_mini_swe_agent_eval(
                base_url="http://127.0.0.1:18080/v1",
                api_key="sk-x",
                model_name=None,
                model_filename="",
                task_ids=["any"],
            )


def test_run_mini_swe_agent_missing_skill(tmp_path):
    missing_skill = tmp_path / "no-such-skill.md"
    with (
        patch.object(msa_runner.shutil, "which", return_value="uvx"),
        patch.object(msa_runner, "PLAN_FIRST_SKILL", missing_skill),
    ):
        with pytest.raises(FileNotFoundError, match="plan-first skill"):
            run_mini_swe_agent_eval(
                base_url="http://127.0.0.1:18080/v1",
                api_key="sk-x",
                model_name="m",
                model_filename="m",
                task_ids=["any"],
            )


def test_model_name_explicit_argument_wins_and_prefixes_once(monkeypatch):
    monkeypatch.setenv("MSWEA_MODEL_NAME", "openai/stale")
    assert msa_runner._resolve_model_name({}, "fresh") == "openai/fresh"
    assert msa_runner._resolve_model_name({}, "openai/already") == "openai/already"
    assert msa_runner._resolve_model_name({}, None) == "openai/stale"


def test_config_contains_step_limit_and_generation_parameters(tmp_path):
    config_path = msa_runner._mini_swe_agent_config(
        tmp_path,
        _fake_task(),
        model_name="openai/m",
        gen_params=GenerationParams(
            temp=0.25,
            top_p=0.9,
            top_k=7,
            reasoning_effort="low",
            max_tokens=1234,
        ),
        max_wall_sec=60,
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["agent"]["step_limit"] == 10
    assert config["agent"]["wall_time_limit_seconds"] == 60
    assert config["agent"]["max_consecutive_format_errors"] == 1
    assert Path(config["agent"]["output_path"]) == tmp_path / ".mini-swe-agent-task.traj.json"
    assert config["model"]["model_name"] == "openai/m"
    kwargs = config["model"]["model_kwargs"]
    assert kwargs["temperature"] == 0.25
    assert kwargs["top_p"] == 0.9
    assert kwargs["top_k"] == 7
    assert kwargs["reasoning_effort"] == "low"
    assert kwargs["max_tokens"] == 1234
    assert kwargs["parallel_tool_calls"] is False
    assert kwargs["tool_choice"] == "required"


def test_constrained_task_does_not_require_plan_file(tmp_path):
    config_path = msa_runner._mini_swe_agent_config(
        tmp_path,
        _scoped_task(),
        model_name="openai/m",
        gen_params=None,
        max_wall_sec=60,
    )
    template = yaml.safe_load(config_path.read_text(encoding="utf-8"))["agent"]["instance_template"]
    assert "Do not create TODO.md" in template
    assert "`config.py`" in template


def _patched_guarded(returncodes: dict[str, int], agent_hook=None):
    def side_effect(command, *args, **kwargs):
        if command[:2] == ["uvx", "mini-swe-agent"]:
            if agent_hook is not None:
                agent_hook(Path(kwargs["cwd"]))
            return returncodes["mini"], "agent stdout", "agent stderr", False
        if command[:3] == [sys.executable, "-m", "pytest"]:
            return returncodes["pytest"], "1 passed\n", "", False
        return 0, "", "", False

    return side_effect


def test_run_one_task_pass_path():
    with (
        patch.object(msa_runner.shutil, "which", return_value="uvx"),
        patch.object(msa_runner, "load_task", return_value=_fake_task()),
        patch.object(
            msa_runner,
            "_run_guarded",
            side_effect=_patched_guarded({"mini": 0, "pytest": 0}),
        ),
    ):
        result = run_mini_swe_agent_eval(
            base_url="http://127.0.0.1:18080/v1",
            api_key="sk-x",
            model_name="m",
            model_filename="m",
            task_ids=["config_precedence"],
            max_wall_sec=60,
        )
    assert result["passed"] == 1
    assert result["total"] == 1
    assert result["score"] == 1.0
    assert "config_precedence=pass" in result["detail"]
    assert result["suite_signature"]


def test_run_one_task_scope_violation_sets_zero():
    def write_forbidden(worktree: Path):
        (worktree / "forbidden.py").write_text("x = 1\n", encoding="utf-8")

    with (
        patch.object(msa_runner.shutil, "which", return_value="uvx"),
        patch.object(msa_runner, "load_task", return_value=_scoped_task()),
        patch.object(
            msa_runner,
            "_run_guarded",
            side_effect=_patched_guarded({"mini": 0, "pytest": 0}, agent_hook=write_forbidden),
        ),
    ):
        result = run_mini_swe_agent_eval(
            base_url="http://127.0.0.1:18080/v1",
            api_key="sk-x",
            model_name="m",
            model_filename="m",
            task_ids=["config_precedence"],
            max_wall_sec=60,
        )
    assert result["passed"] == 0
    assert "outside allowed set" in result["detail"]


def test_run_one_task_pytest_fail_sets_zero():
    with (
        patch.object(msa_runner.shutil, "which", return_value="uvx"),
        patch.object(msa_runner, "load_task", return_value=_fake_task()),
        patch.object(
            msa_runner,
            "_run_guarded",
            side_effect=_patched_guarded({"mini": 0, "pytest": 1}),
        ),
    ):
        result = run_mini_swe_agent_eval(
            base_url="http://127.0.0.1:18080/v1",
            api_key="sk-x",
            model_name="m",
            model_filename="m",
            task_ids=["config_precedence"],
            max_wall_sec=60,
        )
    assert result["passed"] == 0
    assert "tests_red" in result["detail"]


def test_run_one_task_uses_custom_hidden_test_command():
    seen: list[list[str]] = []

    def capture(command, *args, **kwargs):
        seen.append(list(command))
        if command[:2] == ["uvx", "mini-swe-agent"]:
            return 0, "", "", False
        return 0, "passed", "", False

    task = _fake_task()
    task["_hidden_files"] = {"tests/test_hidden.py": "def test_hidden():\n    assert True\n"}
    task["hidden_test_command"] = [
        "{python}",
        "-m",
        "pytest",
        "-q",
        "tests/test_hidden.py",
    ]
    with (
        patch.object(msa_runner.shutil, "which", return_value="uvx"),
        patch.object(msa_runner, "load_task", return_value=task),
        patch.object(msa_runner, "_run_guarded", side_effect=capture),
    ):
        result = run_mini_swe_agent_eval(
            base_url="http://127.0.0.1:18080/v1",
            api_key="sk-x",
            model_name="m",
            model_filename="m",
            task_ids=["config_precedence"],
            max_wall_sec=60,
        )
    assert result["passed"] == 1
    assert seen[-1][-1] == "tests/test_hidden.py"


def test_run_one_task_subprocess_crash_records_fail_reason():
    with (
        patch.object(msa_runner.shutil, "which", return_value="uvx"),
        patch.object(msa_runner, "load_task", return_value=_fake_task()),
        patch.object(
            msa_runner,
            "_run_guarded",
            side_effect=_patched_guarded({"mini": 7, "pytest": 0}),
        ),
    ):
        result = run_mini_swe_agent_eval(
            base_url="http://127.0.0.1:18080/v1",
            api_key="sk-x",
            model_name="m",
            model_filename="m",
            task_ids=["config_precedence"],
            max_wall_sec=60,
        )
    assert result["passed"] == 0
    assert "mini_exit_7" in result["detail"]


def test_run_mini_swe_agent_aborts_after_two_no_progress_format_errors():
    tasks = [_fake_task("first"), _fake_task("second")]
    mini_calls = 0
    commands: list[list[str]] = []

    def repeated_format_error(command, *args, **kwargs):
        nonlocal mini_calls
        if command[:2] == ["uvx", "mini-swe-agent"]:
            mini_calls += 1
            commands.append(list(command))
            trajectory = Path(kwargs["cwd"]) / ".mini-swe-agent-task.traj.json"
            trajectory.write_text(
                json.dumps({"info": {"exit_status": "RepeatedFormatError"}}),
                encoding="utf-8",
            )
            return 0, "", "", False
        return 0, "tests failed", "", False

    with (
        patch.object(msa_runner.shutil, "which", return_value="uvx"),
        patch.object(
            msa_runner,
            "load_task",
            side_effect=lambda task_id, suite=None: tasks[0 if task_id == "first" else 1],
        ),
        patch.object(msa_runner, "_run_guarded", side_effect=repeated_format_error),
    ):
        with pytest.raises(RuntimeError, match="2 consecutive tasks"):
            run_mini_swe_agent_eval(
                base_url="http://127.0.0.1:18080/v1",
                api_key="sk-x",
                model_name="m",
                model_filename="m",
                task_ids=["first", "second"],
                max_wall_sec=60,
            )
    assert mini_calls == 2
    for command in commands:
        output_index = command.index("--output")
        assert Path(command[output_index + 1]).name == ".mini-swe-agent-task.traj.json"


def test_hidden_tests_staged_only_after_agent_exit():
    seen: dict[str, bool] = {}

    def capture(command, *args, **kwargs):
        cwd = Path(kwargs["cwd"])
        if command[:2] == ["uvx", "mini-swe-agent"]:
            seen["hidden_present_during_agent"] = (cwd / "test_config.py").exists()
            return 0, "", "", False
        seen["hidden_present_during_pytest"] = (cwd / "test_config.py").exists()
        return 0, "1 passed\n", "", False

    with (
        patch.object(msa_runner.shutil, "which", return_value="uvx"),
        patch.object(msa_runner, "load_task", return_value=_fake_task()),
        patch.object(msa_runner, "_run_guarded", side_effect=capture),
    ):
        result = run_mini_swe_agent_eval(
            base_url="http://127.0.0.1:18080/v1",
            api_key="sk-x",
            model_name="m",
            model_filename="m",
            task_ids=["config_precedence"],
            max_wall_sec=60,
        )
    assert result["passed"] == 1
    assert seen["hidden_present_during_agent"] is False
    assert seen["hidden_present_during_pytest"] is True


def test_trajectory_file_is_appended(tmp_path):
    captured: dict[str, Path] = {}

    def fake_new_path(_model_filename: str) -> Path:
        path = tmp_path / f"traj-{time.time_ns()}.jsonl"
        captured["path"] = path
        return path

    with (
        patch.object(msa_runner.shutil, "which", return_value="uvx"),
        patch.object(msa_runner, "load_task", return_value=_fake_task()),
        patch.object(
            msa_runner,
            "_run_guarded",
            side_effect=_patched_guarded({"mini": 0, "pytest": 0}),
        ),
        patch.object(msa_runner, "_new_trajectory_path", side_effect=fake_new_path),
    ):
        result = run_mini_swe_agent_eval(
            base_url="http://127.0.0.1:18080/v1",
            api_key="sk-x",
            model_name="m",
            model_filename="m",
            task_ids=["config_precedence"],
            max_wall_sec=60,
        )
    trajectory = Path(result["trajectory_path"])
    assert trajectory.is_file()
    assert trajectory == captured["path"]
    lines = [
        json.loads(line) for line in trajectory.read_text(encoding="utf-8").strip().splitlines()
    ]
    assert len(lines) == 1
    assert lines[0]["task_id"] == "config_precedence"
    assert lines[0]["suite_signature"] == result["suite_signature"]


def test_run_guarded_tears_down_on_timeout(tmp_path, monkeypatch):
    class FakeProcess:
        def communicate(self, timeout=None):
            if timeout is not None:
                raise subprocess.TimeoutExpired(["cmd"], 1)
            return "partial", "stderr"

    class FakeGuard:
        def __init__(self):
            self.torn_down = False

        def spawn(self, command, **kwargs):
            assert command == ["cmd"]
            return FakeProcess()

        def teardown(self):
            self.torn_down = True

    guard = FakeGuard()
    monkeypatch.setattr(msa_runner, "ProcessGuard", lambda: guard)
    return_code, stdout, stderr, timed_out = msa_runner._run_guarded(
        ["cmd"], cwd=tmp_path, env=None, timeout_sec=1
    )
    assert return_code is None
    assert timed_out is True
    assert stdout == "partial"
    assert stderr == "stderr"
    assert guard.torn_down is True


def test_run_mini_swe_agent_no_dm_agent_raises_actionable_error():
    saved = {
        name: sys.modules.pop(name, None)
        for name in ("dm_agent", "dm_agent.benchmarks", "dm_agent.benchmarks.tasks")
    }
    sys.modules["dm_agent"] = None
    sys.modules["dm_agent.benchmarks"] = None
    sys.modules["dm_agent.benchmarks.tasks"] = None
    try:
        with pytest.raises(FileNotFoundError, match="dm_agent is not importable"):
            discover_tasks("maintenance")
    finally:
        for name, module in saved.items():
            if module is not None:
                sys.modules[name] = module
            else:
                sys.modules.pop(name, None)
