"""Tests for autoresearch.benchmarks.agentic_runner — Claw-Eval runner and scoring."""

import email.message
import io
import json
import sys
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from autoresearch.benchmarks.agentic_runner import (
    ServiceManager,
    _assistant_history_message,
    _assistant_visible_text,
    run_agent_loop,
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
    assert res["task_results"][0]["final_text_length"] == 0
    assert res["task_results"][0]["tool_calls_count"] == 0
    assert res["task_results"][0]["length_stops"] == 0
    assert res["task_results"][0]["http_errors"] == []


def test_service_manager_does_not_sweep_before_start(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """Regression: ServiceManager.start must not run the harness-port orphan sweep.

    The full harness sweep (name∩harness-port intersection) would kill the live
    llama-server on :18080 mid-Trial, causing WinError 10061 and a 0.0 agentic
    score. Pre-flight sweep lives in LlamaServerRunner, not here.
    """
    sweep = MagicMock()
    monkeypatch.setattr("autoresearch.core.llama_runner.sweep_leftover_processes", sweep)
    guard = MagicMock()
    monkeypatch.setattr("autoresearch.benchmarks.agentic_runner.ProcessGuard", lambda: guard)
    monkeypatch.setattr(ServiceManager, "_wait_healthy", lambda *_: None)

    mgr = ServiceManager(
        tmp_path,
        {
            "services": [
                {
                    "name": "web",
                    "port": 9113,
                    "command": "python mock_services/web/server.py",
                }
            ]
        },
    )
    mgr.start()

    sweep.assert_not_called()
    guard.spawn.assert_called_once()
    assert mgr._guard is guard


def test_service_manager_spawns_through_guard(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Issue #39: mock services are spawned via the Process Guard, not raw Popen."""
    guard = MagicMock()
    monkeypatch.setattr("autoresearch.benchmarks.agentic_runner.ProcessGuard", lambda: guard)
    monkeypatch.setattr(ServiceManager, "_wait_healthy", lambda *_: None)

    mgr = ServiceManager(
        tmp_path,
        {
            "services": [
                {
                    "name": "web",
                    "port": 9113,
                    "command": "python mock_services/web/server.py",
                },
                {
                    "name": "db",
                    "port": 9114,
                    "command": "python mock_services/db/server.py",
                },
            ]
        },
    )
    mgr.start()

    assert guard.spawn.call_count == 2
    assert len(mgr._procs) == 2
    for call in guard.spawn.call_args_list:
        assert call.args[0][0] == sys.executable


def test_service_manager_stop_tears_down_guard(monkeypatch: pytest.MonkeyPatch):
    """Issue #39: ServiceManager.stop tears the Process Guard down and clears procs."""
    guard = MagicMock()
    monkeypatch.setattr("autoresearch.benchmarks.agentic_runner.ProcessGuard", lambda: guard)
    mgr = ServiceManager(Path("dummy"), {"services": []})
    proc = MagicMock()
    mgr._procs.append(proc)

    mgr.stop()

    guard.teardown.assert_called_once()
    assert mgr._procs == []


def test_service_manager_starts_mock_with_utf8(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    popen = MagicMock(return_value=MagicMock())
    monkeypatch.setattr("autoresearch.benchmarks.agentic_runner.subprocess.Popen", popen)
    monkeypatch.setattr(ServiceManager, "_wait_healthy", lambda *_: None)

    ServiceManager(
        tmp_path,
        {
            "services": [
                {
                    "name": "web",
                    "port": 9113,
                    "command": "python mock_services/web/server.py",
                }
            ]
        },
    ).start()

    assert popen.call_args.kwargs["env"]["PYTHONUTF8"] == "1"


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
            {"length_stops": 2, "http_errors": ["HTTP 500: boom"]},
        ),
    )

    res = run_agentic_eval(mock_llama_client, [task_id], trials=1)

    assert res["passed"] == 1
    assert res["total"] == 1
    assert res["score"] == 1.0
    assert len(res["task_results"]) == 1
    assert res["task_results"][0]["score"] == 1.0
    assert "check_success: PASS" in res["task_results"][0]["details"]
    # Observability fields flow into the sidecar JSON.
    row = res["task_results"][0]
    assert row["final_text_length"] == len("Operation completed with success.")
    assert row["tool_calls_count"] == 1
    assert row["elapsed_sec"] == 0.1
    assert row["length_stops"] == 2
    assert row["http_errors"] == ["HTTP 500: boom"]


def test_assistant_visible_text_falls_back_to_reasoning_content():
    """Thinking models often leave content empty; graders need reasoning_content."""
    assert _assistant_visible_text({"content": "final", "reasoning_content": "think"}) == "final"
    assert _assistant_visible_text({"content": "", "reasoning_content": "needs reply FYI"}) == (
        "needs reply FYI"
    )
    assert _assistant_visible_text({"content": None, "reasoning_content": None}) == ""


def test_assistant_history_preserves_reasoning_content():
    msg = {"content": "", "reasoning_content": "plan…"}
    hist = _assistant_history_message(msg, tool_calls=[{"id": "c1"}])
    assert hist["reasoning_content"] == "plan…"
    assert hist["tool_calls"] == [{"id": "c1"}]
    assert hist["content"] == ""


def test_run_agent_loop_uses_reasoning_when_content_empty(monkeypatch, mock_llama_client):
    """Final turn with empty content + reasoning_content must not score as blank."""
    mock_llama_client.base_url = "http://127.0.0.1:18080"
    final_payload = {
        "choices": [
            {
                "message": {
                    "content": "",
                    "reasoning_content": "categories: needs reply, FYI, spam. Summary follows.",
                    "tool_calls": [],
                }
            }
        ]
    }

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            import json

            return json.dumps(final_payload).encode()

    monkeypatch.setattr(
        "autoresearch.benchmarks.agentic_runner.urllib.request.urlopen",
        lambda *a, **k: _Resp(),
    )
    text, calls, _elapsed, _meta = run_agent_loop(
        mock_llama_client,
        {"prompt": {"text": "triage"}, "tools": [], "tool_endpoints": []},
        max_turns=2,
    )
    assert "needs reply" in text
    assert calls == []
    assert _meta == {"length_stops": 0, "http_errors": []}


def test_run_agent_loop_logs_http_error_body(monkeypatch, mock_llama_client, capsys):
    """HTTP 400/500 bodies must survive post-mortem (printed + in loop_meta)."""
    mock_llama_client.base_url = "http://127.0.0.1:18080"

    def _raise(*a, **k):
        raise urllib.error.HTTPError(
            "http://127.0.0.1:18080/v1/chat/completions",
            400,
            "Bad Request",
            email.message.Message(),
            io.BytesIO(b'{"error":{"message":"reasoning content after final"}}'),
        )

    monkeypatch.setattr("autoresearch.benchmarks.agentic_runner.urllib.request.urlopen", _raise)
    text, calls, _elapsed, meta = run_agent_loop(
        mock_llama_client,
        {"prompt": {"text": "triage"}, "tools": [], "tool_endpoints": []},
        max_turns=2,
    )
    assert text == ""
    assert calls == []
    assert "HTTP 400" in capsys.readouterr().out
    assert meta["http_errors"] == [
        'HTTP 400: {"error":{"message":"reasoning content after final"}}'
    ]


def test_run_agent_loop_counts_finish_reason_length(monkeypatch, mock_llama_client, capsys):
    """finish_reason == 'length' (max_tokens exhausted) must be counted and surfaced."""
    mock_llama_client.base_url = "http://127.0.0.1:18080"
    payload = {
        "choices": [
            {
                "finish_reason": "length",
                "message": {"content": "partial", "tool_calls": []},
            }
        ]
    }

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps(payload).encode()

    monkeypatch.setattr(
        "autoresearch.benchmarks.agentic_runner.urllib.request.urlopen",
        lambda *a, **k: _Resp(),
    )
    text, calls, _elapsed, meta = run_agent_loop(
        mock_llama_client,
        {"prompt": {"text": "triage"}, "tools": [], "tool_endpoints": []},
        max_turns=2,
    )
    assert text == "partial"
    assert calls == []
    assert meta["length_stops"] == 1
    assert "finish_reason=length" in capsys.readouterr().out


def test_run_agent_loop_tool_call_length_stop_not_counted(monkeypatch, mock_llama_client):
    """llama.cpp reports finish_reason='length' on tool-call boundary stops too.

    Those are not max_tokens exhaustion — they must not inflate length_stops.
    """
    mock_llama_client.base_url = "http://127.0.0.1:18080"
    payload = {
        "choices": [
            {
                "finish_reason": "length",
                "message": {
                    "content": "",
                    "tool_calls": [
                        {"id": "c1", "function": {"name": "gmail_list_messages", "arguments": "{}"}}
                    ],
                },
            }
        ]
    }

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps(payload).encode()

    monkeypatch.setattr(
        "autoresearch.benchmarks.agentic_runner.urllib.request.urlopen",
        lambda *a, **k: _Resp(),
    )
    _, calls, _elapsed, meta = run_agent_loop(
        mock_llama_client,
        {"prompt": {"text": "triage"}, "tools": [], "tool_endpoints": []},
        max_turns=2,
    )
    assert calls and calls[0]["tool"] == "gmail_list_messages"
    assert meta["length_stops"] == 0


def test_run_agent_loop_model_specific_stop_tokens(monkeypatch, mock_llama_client):
    """K2-Horizon stop token <|ifm|im_end|> must be respected in agent turn payload."""
    from autoresearch.core.llama_client import GenerationParams

    mock_llama_client.base_url = "http://127.0.0.1:18080"
    captured_payloads = []

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps(
                {"choices": [{"message": {"content": "finished", "tool_calls": []}}]}
            ).encode()

    def _mock_urlopen(req, timeout=None):
        captured_payloads.append(json.loads(req.data.decode()))
        return _Resp()

    monkeypatch.setattr(
        "autoresearch.benchmarks.agentic_runner.urllib.request.urlopen", _mock_urlopen
    )

    gen = GenerationParams(stop=["<|ifm|im_end|>", "</s>"], reasoning_effort="low")
    run_agent_loop(
        mock_llama_client,
        {"prompt": {"text": "test prompt"}, "tools": [], "tool_endpoints": []},
        gen_params=gen,
        max_turns=1,
    )

    assert len(captured_payloads) == 1
    assert captured_payloads[0]["stop"] == ["<|ifm|im_end|>", "</s>"]
    assert captured_payloads[0]["reasoning_effort"] == "low"
    assert captured_payloads[0]["chat_template_kwargs"] == {"reasoning_effort": "low"}
    assert captured_payloads[0]["chat_template_args"] == {"reasoning_effort": "low"}


def test_run_agent_loop_enforces_420s_timeout_floor(monkeypatch, mock_llama_client):
    """Agent turns must enforce 420s timeout floor to avoid cancelling long reasoning turns."""
    mock_llama_client.base_url = "http://127.0.0.1:18080"
    mock_llama_client.timeout = 100.0  # sub-floor timeout on client
    captured_timeouts = []

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps(
                {"choices": [{"message": {"content": "done", "tool_calls": []}}]}
            ).encode()

    def _mock_urlopen(req, timeout=None):
        captured_timeouts.append(timeout)
        return _Resp()

    monkeypatch.setattr(
        "autoresearch.benchmarks.agentic_runner.urllib.request.urlopen", _mock_urlopen
    )

    # Calling with sub-floor turn_timeout=30.0
    run_agent_loop(
        mock_llama_client,
        {"prompt": {"text": "test prompt"}, "tools": [], "tool_endpoints": []},
        max_turns=1,
        turn_timeout=30.0,
    )
    assert len(captured_timeouts) == 1
    assert captured_timeouts[0] >= 420.0

    # Calling with turn_timeout=None also respects >= 420.0 floor
    captured_timeouts.clear()
    run_agent_loop(
        mock_llama_client,
        {"prompt": {"text": "test prompt"}, "tools": [], "tool_endpoints": []},
        max_turns=1,
        turn_timeout=None,
    )
    assert len(captured_timeouts) == 1
    assert captured_timeouts[0] >= 420.0


def test_run_agent_loop_string_stop_token_normalized(monkeypatch, mock_llama_client):
    """String stop token should be safely normalized to list in request payload."""
    mock_llama_client.base_url = "http://127.0.0.1:18080"
    captured_payloads = []

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps(
                {"choices": [{"message": {"content": "done", "tool_calls": []}}]}
            ).encode()

    def _mock_urlopen(req, timeout=None):
        captured_payloads.append(json.loads(req.data.decode()))
        return _Resp()

    monkeypatch.setattr(
        "autoresearch.benchmarks.agentic_runner.urllib.request.urlopen", _mock_urlopen
    )

    run_agent_loop(
        mock_llama_client,
        {"prompt": {"text": "test prompt"}, "tools": [], "tool_endpoints": []},
        stop="<|ifm|im_end|>",
        max_turns=1,
    )
    assert len(captured_payloads) == 1
    assert captured_payloads[0]["stop"] == ["<|ifm|im_end|>"]


def test_run_agentic_eval_forwards_turn_timeout_and_stop(monkeypatch, tmp_path):
    """run_agentic_eval should pass turn_timeout and stop to run_agent_loop."""
    import yaml

    from autoresearch.benchmarks.agentic_runner import run_agentic_eval

    # Create dummy claw task
    task_dir = tmp_path / "dummy_task"
    task_dir.mkdir()
    task_file = task_dir / "task.yaml"
    task_file.write_text(
        yaml.dump(
            {
                "prompt": {"text": "hello"},
                "environment": {"max_turns": 2},
            }
        )
    )

    mock_client = MagicMock()
    mock_client.port = 8080
    mock_client.timeout = 420.0

    captured_kwargs = {}

    def _mock_run_agent_loop(client, task, **kwargs):
        captured_kwargs.update(kwargs)
        return "done", [], 1.0, {"length_stops": 0, "http_errors": []}

    monkeypatch.setattr(
        "autoresearch.benchmarks.agentic_runner.run_agent_loop", _mock_run_agent_loop
    )
    monkeypatch.setattr(
        "autoresearch.benchmarks.agentic_runner.score_task",
        lambda *a, **k: {
            "score": 1.0,
            "details": "ok",
            "tool_calls_count": 0,
            "tools_used": [],
            "final_text_length": 4,
        },
    )
    monkeypatch.setattr("autoresearch.benchmarks.agentic_runner.TASKS_DIR", tmp_path)

    run_agentic_eval(
        mock_client,
        ["dummy_task"],
        turn_timeout=500.0,
        stop=["<|ifm|im_end|>", "</s>"],
    )

    assert captured_kwargs.get("turn_timeout") == 500.0
    assert captured_kwargs.get("stop") == ["<|ifm|im_end|>", "</s>"]


def test_format_tool_content_truncates_large_payload():
    """Tool content exceeding MAX_TOOL_OUTPUT_CHARS must be truncated safely."""
    from autoresearch.benchmarks.agentic_runner import _format_tool_content

    small_payload = {"status": "ok", "count": 42}
    assert _format_tool_content(small_payload, max_chars=100) == json.dumps(small_payload)

    large_payload = {"data": "x" * 200}
    formatted = _format_tool_content(large_payload, max_chars=50)
    assert len(formatted) < len(json.dumps(large_payload))
    assert "... [truncated" in formatted


def test_prune_messages_for_context():
    """Messages list must be pruned progressively to avoid context blowouts."""
    from autoresearch.benchmarks.agentic_runner import _prune_messages_for_context

    messages = [
        {"role": "system", "content": "You are an assistant."},
        {"role": "user", "content": "Do task."},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "old thought",
            "tool_calls": [{"id": "1"}],
        },
        {"role": "tool", "content": "a" * 2000, "tool_call_id": "1"},
        {"role": "assistant", "content": "", "reasoning_content": "recent thought"},
    ]

    # Pass 1: Pruning large tool responses
    assert _prune_messages_for_context(messages) is True
    assert len(messages[3]["content"]) < 1200
    assert "... [pruned for context" in messages[3]["content"]

    # Pass 2: Pruning old reasoning content from earlier assistant turns
    assert _prune_messages_for_context(messages) is True
    assert "reasoning_content" not in messages[2]
    assert messages[4].get("reasoning_content") == "recent thought"


def test_score_task_categories_present_synonyms(dummy_task_dir: Path):
    """categories_present check should accept natural synonyms (e.g. notifications <-> fyi)."""
    task = {
        "scoring_components": [
            {
                "name": "triage_categories",
                "weight": 1.0,
                "check": {
                    "type": "categories_present",
                    "categories": ["fyi", "action required"],
                },
            }
        ]
    }
    # Text uses "notifications" instead of "fyi", and "action items" instead of "action required"
    text = "Emails sorted into: Notifications and Action Items."
    result = score_task(task, text, [], dummy_task_dir)
    assert result["score"] == 1.0
    assert "triage_categories: PASS" in result["details"]


def test_run_agent_loop_retries_on_context_error(monkeypatch, mock_llama_client, capsys):
    """Context blowout HTTP 400 must trigger pruning and retry instead of immediately aborting."""
    mock_llama_client.base_url = "http://127.0.0.1:18080"

    calls = 0

    class _MockResp:
        def __init__(self, data):
            self._data = data

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps(self._data).encode()

    def _mock_urlopen(req, *a, **k):
        nonlocal calls
        calls += 1
        if calls == 1:
            # Turn 1: model returns a tool call
            return _MockResp(
                {
                    "choices": [
                        {
                            "finish_reason": "tool_calls",
                            "message": {
                                "content": "",
                                "tool_calls": [
                                    {
                                        "id": "c1",
                                        "function": {"name": "fetch_info", "arguments": "{}"},
                                    }
                                ],
                            },
                        }
                    ]
                }
            )
        elif calls == 2:
            # Turn 2: server rejects with 400 context blowout
            raise urllib.error.HTTPError(
                "http://127.0.0.1:18080/v1/chat/completions",
                400,
                "Bad Request",
                email.message.Message(),
                io.BytesIO(b'{"error":{"message":"request exceeds context size (32768)"}}'),
            )
        else:
            # Turn 2 retry: server returns final response
            return _MockResp(
                {
                    "choices": [
                        {"finish_reason": "stop", "message": {"content": "Final triage completed."}}
                    ]
                }
            )

    monkeypatch.setattr(
        "autoresearch.benchmarks.agentic_runner.urllib.request.urlopen", _mock_urlopen
    )
    monkeypatch.setattr(
        "autoresearch.benchmarks.agentic_runner._call_mock_endpoint",
        lambda ep, args: {"data": "x" * 2500},
    )

    task = {
        "prompt": {"text": "Process emails"},
        "tools": [{"name": "fetch_info"}],
        "tool_endpoints": [{"tool_name": "fetch_info", "url": "http://127.0.0.1:9999"}],
    }

    text, tool_calls, elapsed, meta = run_agent_loop(
        mock_llama_client,
        task,
        max_turns=4,
    )
    assert "context limit reached" in capsys.readouterr().out
    assert text == "Final triage completed."
    assert calls == 3
