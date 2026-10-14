"""STORE-SILENT: output caps, reasoning extraction and protocol mixing corrupt scores.

Failure modes: 2k overrides, premature HTTP timeout, lost completion usage/finish
reason, reasoning executed as code, missing test stderr/HTTP failures in artifacts,
and old/new partial vectors silently becoming complete through store/rank merges.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from autoresearch.benchmarks import benchmark_coding as bc
from autoresearch.core import classify, recompute
from autoresearch.core.llama_client import GenerationParams, LlamaClient
from autoresearch.runners import run
from scripts import rank_results

PROTOCOL = "coding-v2-32768-final-only"


def row(protocol=None, **axes):
    cfg = {"model": "M.gguf", "ctx_size": 131072, "vram_limit_mb": 8000}
    if protocol:
        cfg["coding_protocol"] = protocol
    return {
        "trial_id": protocol or "old",
        "model": "M.gguf",
        "status": "incomplete",
        "config_json": json.dumps(cfg),
        "ctx": "131072",
        "tps": "40",
        "outcome": "OK",
        "category": "10-task",
        **axes,
    }


def test_historical_unknown_budget_is_not_mislabeled_as_2048():
    from autoresearch.core.evaluation_protocol import coding_protocol_from_config

    assert coding_protocol_from_config("{}") == "coding-legacy-unversioned"


def test_protocol_recorded_and_hash_isolated():
    cfg = {"MODEL": "M.gguf", "CTX_SIZE": 131072}
    recorded = json.loads(run.tsv_fields_from_cfg(cfg)["config_json"])
    assert recorded["coding_protocol"] == PROTOCOL
    assert classify.fp_from_baseline(cfg) != classify.fp_from_baseline(
        dict(cfg, CODING_PROTOCOL=PROTOCOL)
    )


def test_partial_protocols_never_complete_each_other():
    rows = [row(agentic="0.9", coding=""), row(PROTOCOL, agentic="", coding="0.5")]
    assert [r["status"] for r in recompute.recompute_rows(rows)] == ["incomplete", "incomplete"]
    complete, _ = rank_results.build_vectors(rows)
    assert not complete
    status, flips = classify.plan_write(
        rows[:1],
        fp="new",
        vector=classify.ObjectiveVector(131072, 40, None, 0.5),
        bucket_gb=8,
        model="M.gguf",
        coding_protocol=PROTOCOL,
    )
    assert status == "incomplete"
    assert not flips


def test_known_vectors_do_not_stitch_protocols():
    rows = [row(agentic="0.9", coding=""), row(PROTOCOL, agentic="", coding="0.5")]
    assert classify.known_vectors(rows, model="M.gguf", bucket_gb=8) == []


def test_search_known_set_excludes_legacy_protocol():
    import autoloop

    old = row(agentic="0.9", coding="0.9")
    current = row(PROTOCOL, agentic="0.6", coding="0.4")
    with patch.object(autoloop, "read_rows", return_value=[old, current]):
        vectors = autoloop._seed_known_vectors("M.gguf", 8)
    assert len(vectors) == 1
    assert vectors[0].coding == 0.4


def test_client_preserves_completion_diagnostics_and_large_budget_timeout():
    raw = {
        "choices": [
            {"finish_reason": "length", "message": {"content": "", "reasoning_content": "thinking"}}
        ],
        "usage": {"total_tokens": 33000, "completion_tokens": 32768, "prompt_tokens": 232},
    }
    response = MagicMock()
    response.__enter__.return_value = response
    response.read.return_value = json.dumps(raw).encode()
    with patch("urllib.request.urlopen", return_value=response) as request:
        res = LlamaClient(1234).complete("prompt", gen=GenerationParams(max_tokens=32768))
    assert request.call_args.kwargs["timeout"] >= 3600
    assert res["usage"] == raw["usage"]
    assert res["finish_reason"] == "length"
    assert res["choices"][0]["finish_reason"] == "length"


@pytest.mark.parametrize(
    "content,reasoning,finish,tokens,passed,error",
    [
        ("```python\ndef f():\n    return 42\n```", "deliberation", "stop", 100, True, ""),
        ("", "```python\ndef f():\n    return 42\n```", "length", 32768, False, "reasoning_only"),
        ("def f():\n    return 0", "", "stop", 100, False, "AssertionError"),
        ("def f():\n    return", "", "length", 32768, False, "budget_exhausted"),
    ],
)
def test_real_eval_artifact(tmp_path, content, reasoning, finish, tokens, passed, error):
    client = MagicMock(spec=LlamaClient)
    client.complete.return_value = {
        "content": content,
        "reasoning_content": reasoning,
        "finish_reason": finish,
        "usage": {"completion_tokens": tokens, "total_tokens": tokens + 10},
    }
    entry = {"prompt": "def f():", "entry_point": "f", "test": "assert f() == 42"}
    artifact = tmp_path / "coding.json"
    with (
        patch.object(bc, "_load_problems", side_effect=[{"HE/0": entry}, {}]),
        patch.object(bc, "_load_livecodebench", return_value=[]),
        patch.object(bc, "_load_bigcodebench_hard", return_value=[]),
    ):
        result = bc.run_benchmark(client, task_limit=1, artifact_path=artifact, model_name="M.gguf")
    assert client.complete.call_args.kwargs["gen"].max_tokens == 32768
    assert bool(result.val_pass2) == passed
    data = json.loads(artifact.read_text())
    assert data["coding_protocol"] == PROTOCOL
    task = data["tasks"][0]
    assert task["content"] == content
    assert task["reasoning_content"] == reasoning
    assert task["finish_reason"] == finish
    assert task["completion_tokens"] == tokens
    assert task["budget_exhausted"] == (tokens == 32768)
    assert task["passed"] == passed
    assert error in task["error"]
    if not content:
        assert task["code"] == ""
    else:
        assert "def f" in task["code"]


def test_http_failure_is_artifact_not_missing_task(tmp_path):
    client = MagicMock(spec=LlamaClient)
    client.complete.side_effect = RuntimeError("HTTP 500")
    artifact = tmp_path / "coding.json"
    with (
        patch.object(bc, "_load_problems", side_effect=[{"HE/0": {"prompt": "def f():"}}, {}]),
        patch.object(bc, "_load_livecodebench", return_value=[]),
        patch.object(bc, "_load_bigcodebench_hard", return_value=[]),
    ):
        result = bc.run_benchmark(client, task_limit=1, artifact_path=artifact)
    assert result.val_pass2 == 0
    assert "HTTP 500" in json.loads(artifact.read_text())["tasks"][0]["error"]
    assert client.complete.call_count == 1
