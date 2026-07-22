"""Tests for autoresearch.benchmarks.agentic_runner — Claw-Eval runner and scoring."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from autoresearch.benchmarks.agentic_runner import (
    run_agentic_eval,
    score_task,
)
from autoresearch.core.llama_client import LlamaClient


@pytest.fixture
def dummy_task_dir(tmp_path: Path) -> Path:
    """Fixture providing a temporary task directory path."""
    return tmp_path / "dummy_task"


@pytest.fixture
def mock_llama_client() -> MagicMock:
    """Fixture providing a mocked LlamaClient per tests/AGENTS.md standards."""
    return MagicMock(spec=LlamaClient)


@pytest.fixture
def sample_task() -> dict:
    """Fixture providing a sample task dictionary with scoring components."""
    return {
        "scoring_components": [
            {
                "name": "check_tool",
                "weight": 1.0,
                "check": {
                    "type": "tool_called",
                    "tool_name": "fetch_data",
                    "min_calls": 1,
                },
            },
            {
                "name": "check_keyword",
                "weight": 1.0,
                "check": {
                    "type": "keywords_present",
                    "keywords": ["success"],
                },
            },
        ]
    }


def test_score_task_tool_called_and_keywords(sample_task: dict, dummy_task_dir: Path):
    """Test score_task with tool_called and keywords_present checks."""
    tool_calls = [{"tool": "fetch_data", "arguments": {}, "result": {}, "turn": 1}]
    final_text = "Operation completed with success."

    result = score_task(sample_task, final_text, tool_calls, dummy_task_dir)

    assert result["score"] == 1.0
    assert result["tool_calls_count"] == 1
    assert result["tools_used"] == ["fetch_data"]
    assert "check_tool: PASS" in result["details"]
    assert "check_keyword: PASS" in result["details"]


def test_score_task_llm_judge_skip(dummy_task_dir: Path):
    """Test that llm_judge tasks return score 0.0 with skipped message."""
    task = {
        "scoring_components": [
            {
                "name": "judge_check",
                "weight": 1.0,
                "check": {"type": "llm_judge"},
            }
        ]
    }
    result = score_task(task, "some text", [], dummy_task_dir)
    assert result["score"] == 0.0
    assert "skipped: llm_judge" in result["details"]


def test_score_task_categories_present(dummy_task_dir: Path):
    """Test categories_present check type."""
    task = {
        "scoring_components": [
            {
                "name": "cats",
                "weight": 1.0,
                "check": {
                    "type": "categories_present",
                    "categories": ["speed", "accuracy", "reliability"],
                },
            }
        ]
    }
    text = "Detailed info on speed and accuracy in benchmark."
    result = score_task(task, text, [], dummy_task_dir)
    assert result["score"] == 1.0


def test_score_task_min_length(dummy_task_dir: Path):
    """Test min_length check type."""
    task = {
        "scoring_components": [
            {
                "name": "len",
                "weight": 1.0,
                "check": {
                    "type": "min_length",
                    "field": "final_text",
                    "min_length": 20,
                },
            }
        ]
    }
    result_fail = score_task(task, "Too short", [], dummy_task_dir)
    assert result_fail["score"] == 0.0


def test_run_agentic_eval_missing_task(mock_llama_client: MagicMock):
    """Test run_agentic_eval handles non-existent task gracefully with mocked LlamaClient."""
    res = run_agentic_eval(mock_llama_client, ["non_existent_task_xyz_123"])

    assert res["passed"] == 0
    assert res["total"] == 1
    assert res["score"] == 0.0
    assert len(res["task_results"]) == 1
    assert res["task_results"][0]["details"] == "missing"


def test_run_agentic_eval_successful_task(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mock_llama_client: MagicMock,
):
    """Test run_agentic_eval full orchestration with mocked task and agent loop."""
    task_id = "test_task_001"
    task_dir = tmp_path / task_id
    task_dir.mkdir(parents=True)
    yaml_path = task_dir / "task.yaml"

    task_yaml_content = """
tools:
  - name: get_weather
    description: Get weather
    input_schema:
      type: object
      properties:
        city:
          type: string
tool_endpoints:
  - tool_name: get_weather
    url: http://127.0.0.1:8080/weather
scoring_components:
  - name: check_success
    weight: 1.0
    check:
      type: keywords_present
      keywords: ["success"]
"""
    yaml_path.write_text(task_yaml_content, encoding="utf-8")

    monkeypatch.setattr("autoresearch.benchmarks.agentic_runner.TASKS_DIR", tmp_path)

    dummy_svc_mgr = MagicMock()
    dummy_svc_mgr.__enter__.return_value = dummy_svc_mgr
    dummy_svc_mgr.__exit__.return_value = False

    monkeypatch.setattr(
        "autoresearch.benchmarks.agentic_runner.ServiceManager",
        lambda tdir, tdict: dummy_svc_mgr,
    )
    monkeypatch.setattr(
        "autoresearch.benchmarks.agentic_runner.run_agent_loop",
        lambda client, task, gen_params, max_turns: (
            "Operation completed with success.",
            [{"tool": "get_weather", "arguments": {"city": "Paris"}, "result": {}, "turn": 1}],
            0.1,
        ),
    )

    res = run_agentic_eval(mock_llama_client, [task_id], trials=1)

    assert res["passed"] == 1
    assert res["total"] == 1
    assert res["score"] == 1.0
    assert len(res["task_results"]) == 1
    assert res["task_results"][0]["score"] == 1.0
    assert "check_success: PASS" in res["task_results"][0]["details"]
