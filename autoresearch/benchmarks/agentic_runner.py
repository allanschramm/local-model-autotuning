"""Agentic benchmark runner — Claw-Eval mock services + agent loop + rule-based scoring.

Single-file adapter. No external deps beyond what the repo already has (requests,
PyYAML). Starts mock services as subprocesses, runs an agent loop against a local
llama-server endpoint with OpenAI-compatible tool calling, then scores via
deterministic rule checks from task.yaml.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

import yaml

from autoresearch.core.llama_client import LlamaClient, GenerationParams

# ── Paths ────────────────────────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CLAW_DIR = _PROJECT_ROOT / "claw-eval"
TASKS_DIR = CLAW_DIR / "tasks"
MOCK_SERVICES_DIR = CLAW_DIR / "mock_services"

# ── Service Manager ──────────────────────────────────────────────────────────


class ServiceManager:
    """Launch and tear down mock service subprocesses for a single task."""

    def __init__(self, task_dir: Path, task: dict):
        self.task_dir = task_dir
        self.task = task
        self._procs: list[subprocess.Popen] = []

    def start(self) -> None:
        """Start all mock services declared in the task's services block."""
        services = self.task.get("services", [])
        for svc in services:
            name = svc["name"]
            port = svc["port"]
            cmd_parts = svc["command"].split()
            # Resolve relative path from claw-eval root
            script = CLAW_DIR / cmd_parts[1] if len(cmd_parts) > 1 else CLAW_DIR / cmd_parts[0]
            python_exe = sys.executable

            env = os.environ.copy()
            env["ERROR_RATE"] = "0"  # disable error injection for deterministic scoring
            for k, v in svc.get("env", {}).items():
                env[k] = str(v)
            env["PORT"] = str(port)

            print(f"    [service] starting {name} on :{port} ({script})")
            proc = subprocess.Popen(
                [python_exe, str(script)],
                cwd=str(CLAW_DIR),
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self._procs.append(proc)

        # Wait for health checks
        for svc in services:
            self._wait_healthy(svc)

    def _wait_healthy(self, svc: dict) -> None:
        """Poll health check endpoint until ready or timeout."""
        url = svc.get("health_check", "")
        method = svc.get("health_check_method", "GET")
        timeout = svc.get("ready_timeout", 10)
        deadline = time.time() + timeout
        name = svc["name"]

        while time.time() < deadline:
            try:
                req = urllib.request.Request(url, method=method)
                req.add_header("X-Health-Check", "1")
                with urllib.request.urlopen(req, timeout=2) as resp:
                    if resp.status < 500:
                        return
            except Exception:
                pass
            time.sleep(0.3)
        print(f"    [service] WARNING: {name} on :{svc['port']} not ready after {timeout}s")

    def reset_all(self) -> None:
        """Reset all service states between trials."""
        for svc in self.task.get("services", []):
            reset_url = svc.get("reset_endpoint", "")
            if not reset_url:
                continue
            try:
                req = urllib.request.Request(reset_url, method="POST", data=b"{}")
                urllib.request.urlopen(req, timeout=5)
            except Exception:
                pass

    def stop(self) -> None:
        """Kill all mock service processes."""
        for proc in self._procs:
            try:
                proc.terminate()
                proc.wait(timeout=3)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


# ── Agent Loop ───────────────────────────────────────────────────────────────


def _build_tool_defs(task: dict) -> list[dict]:
    """Convert task.yaml tool definitions to OpenAI tool format."""
    tools = []
    for tdef in task.get("tools", []):
        tools.append({
            "type": "function",
            "function": {
                "name": tdef["name"],
                "description": tdef.get("description", ""),
                "parameters": tdef.get("input_schema", {"type": "object", "properties": {}, "required": []}),
            },
        })
    return tools


def _build_tool_endpoint_map(task: dict) -> dict[str, dict]:
    """Build mapping: tool_name -> {url, method}."""
    return {
        ep["tool_name"]: {"url": ep["url"], "method": ep.get("method", "POST")}
        for ep in task.get("tool_endpoints", [])
    }


def _call_mock_endpoint(endpoint: dict, arguments: dict) -> dict:
    """Call a mock service endpoint and return the JSON response."""
    url = endpoint["url"]
    method = endpoint["method"]
    data = json.dumps(arguments).encode() if arguments else b"{}"
    req = urllib.request.Request(url, method=method, data=data)
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        return {"error": str(e)}


def run_agent_loop(
    client: LlamaClient,
    task: dict,
    gen_params: GenerationParams | None = None,
    max_turns: int = 20,
) -> tuple[str, list[dict], float]:
    """Run one agent loop for a task. Returns (final_text, tool_calls_made, elapsed_sec).

    Uses llama-server's native /v1/chat/completions with tool calling.
    """
    gen = gen_params or GenerationParams(max_tokens=2048)
    tool_defs = _build_tool_defs(task)
    endpoint_map = _build_tool_endpoint_map(task)

    prompt_text = task.get("prompt", {}).get("text", "")
    system_msg = (
        "You are an AI assistant with access to tools. Use the tools to complete the task. "
        "When you have gathered all needed information, provide a final answer without tool calls. "
        "Call tools one at a time, wait for results, then decide the next step."
    )

    messages: list[dict] = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": prompt_text},
    ]

    all_tool_calls: list[dict] = []
    t_start = time.time()

    for turn in range(max_turns):
        payload = {
            "messages": messages,
            "tools": tool_defs,
            "stream": False,
            "max_tokens": gen.max_tokens,
            "temperature": gen.temp,
            "stop": ["</s>"],
        }
        for key in ("top_p", "top_k", "repeat_penalty"):
            val = getattr(gen, key, None)
            if val is not None:
                payload[key] = val

        url = f"{client.base_url}/v1/chat/completions"
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )

        try:
            with urllib.request.urlopen(req, timeout=client.timeout) as resp:
                raw = json.loads(resp.read().decode())
        except Exception as e:
            print(f"    [agent] turn {turn+1} request failed: {e}")
            break

        choice = (raw.get("choices") or [{}])[0]
        msg = choice.get("message", {})

        # Check for tool calls
        tool_calls = msg.get("tool_calls") or []
        content = msg.get("content") or ""

        if tool_calls:
            # Model wants to use tools
            messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})

            for tc in tool_calls:
                func = tc.get("function", {})
                tool_name = func.get("name", "")
                try:
                    args = json.loads(func.get("arguments", "{}"))
                except json.JSONDecodeError:
                    args = {}

                ep = endpoint_map.get(tool_name)
                if ep:
                    result = _call_mock_endpoint(ep, args)
                else:
                    result = {"error": f"Unknown tool: {tool_name}"}

                all_tool_calls.append({
                    "tool": tool_name,
                    "arguments": args,
                    "result": result,
                    "turn": turn + 1,
                })

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id", f"call_{turn}"),
                    "content": json.dumps(result),
                })
        else:
            # Model produced final answer
            messages.append({"role": "assistant", "content": content})
            elapsed = time.time() - t_start
            return content, all_tool_calls, elapsed

    elapsed = time.time() - t_start
    # If we hit max_turns, use last assistant message content
    last_content = ""
    for m in reversed(messages):
        if m.get("role") == "assistant":
            last_content = m.get("content", "")
            break
    return last_content, all_tool_calls, elapsed


# ── Rule-Based Scorer ────────────────────────────────────────────────────────


def score_task(
    task: dict,
    final_text: str,
    tool_calls: list[dict],
    task_dir: Path,
) -> dict:
    """Apply rule-based scoring from task.yaml. Returns {score, details}.

    Supports check types: tool_called, keywords_present, categories_present, min_length.
    LLM judge tasks are skipped (return score=0, reason=llm_judge).
    """
    components = task.get("scoring_components", [])

    # Check for LLM judge tasks
    for comp in components:
        if comp.get("check", {}).get("type") == "llm_judge":
            return {"score": 0.0, "details": "skipped: llm_judge task not supported"}

    if not components:
        return {"score": 0.0, "details": "no scoring components"}

    total_weight = 0.0
    weighted_score = 0.0
    details_parts = []

    tool_names_called = {tc["tool"] for tc in tool_calls}

    for comp in components:
        name = comp.get("name", "?")
        weight = comp.get("weight", 0.0)
        check = comp.get("check", {})
        check_type = check.get("type", "")

        passed = False
        if check_type == "tool_called":
            target_tool = check.get("tool_name", "")
            min_calls = check.get("min_calls", 1)
            calls = sum(1 for tc in tool_calls if tc["tool"] == target_tool)
            passed = calls >= min_calls
            details_parts.append(f"{name}: {'PASS' if passed else 'FAIL'} ({target_tool} called {calls}/{min_calls})")

        elif check_type == "keywords_present":
            keywords = check.get("keywords", [])
            text_lower = final_text.lower()
            found = [kw for kw in keywords if kw.lower() in text_lower]
            passed = len(found) > 0
            details_parts.append(f"{name}: {'PASS' if passed else 'FAIL'} (keywords found: {found})")

        elif check_type == "categories_present":
            categories = check.get("categories", [])
            text_lower = final_text.lower()
            found = [cat for cat in categories if cat.lower() in text_lower]
            passed = len(found) >= len(categories) * 0.5  # at least half
            details_parts.append(f"{name}: {'PASS' if passed else 'FAIL'} (categories: {found}/{categories})")

        elif check_type == "min_length":
            field = check.get("field", "final_text")
            min_len = check.get("min_length", 0)
            text = final_text if field == "final_text" else ""
            passed = len(text) >= min_len
            details_parts.append(f"{name}: {'PASS' if passed else 'FAIL'} (len={len(text)}/{min_len})")

        else:
            details_parts.append(f"{name}: SKIP (unknown check type: {check_type})")

        if passed:
            weighted_score += weight
        total_weight += weight

    score = weighted_score / total_weight if total_weight > 0 else 0.0
    return {
        "score": round(score, 4),
        "details": " | ".join(details_parts),
        "tool_calls_count": len(tool_calls),
        "tools_used": sorted(tool_names_called),
        "final_text_length": len(final_text),
    }


# ── Top-level runner ─────────────────────────────────────────────────────────


def run_agentic_eval(
    client: LlamaClient,
    task_ids: list[str],
    gen_params: GenerationParams | None = None,
    trials: int = 1,
) -> dict:
    """Run agentic evaluation on selected Claw-Eval tasks.

    Returns dict with keys: passed, total, score, task_results, elapsed_sec.
    """
    gen = gen_params or GenerationParams(max_tokens=2048, temp=0.4)
    results: list[dict] = []
    passed = 0
    t_start = time.time()

    for tid in task_ids:
        task_dir = TASKS_DIR / tid
        yaml_path = task_dir / "task.yaml"
        if not yaml_path.exists():
            print(f"  [agentic] SKIP {tid}: task.yaml not found")
            results.append({"task_id": tid, "score": 0.0, "details": "missing"})
            continue

        with open(yaml_path, "r", encoding="utf-8") as f:
            task = yaml.safe_load(f)

        task_best = 0.0
        task_detail = ""

        for trial in range(trials):
            with ServiceManager(task_dir, task) as svc:
                final_text, tool_calls, elapsed = run_agent_loop(
                    client, task, gen_params=gen,
                    max_turns=task.get("environment", {}).get("max_turns", 20),
                )
                scoring = score_task(task, final_text, tool_calls, task_dir)
                svc.reset_all()

            if scoring["score"] > task_best:
                task_best = scoring["score"]
                task_detail = scoring["details"]

            status = "PASS" if scoring["score"] >= 0.5 else "FAIL"
            print(
                f"  [agentic] {tid} trial{trial+1}: {status} "
                f"score={scoring['score']:.2f} calls={scoring['tool_calls_count']} "
                f"({scoring['details']})"
            )

        if task_best >= 0.5:
            passed += 1

        results.append({
            "task_id": tid,
            "score": task_best,
            "details": task_detail,
        })

    total = len(task_ids)
    overall = passed / total if total > 0 else 0.0
    elapsed = time.time() - t_start

    print(f"  [agentic] {passed}/{total} passed (score={overall:.4f}) in {elapsed:.0f}s")

    return {
        "passed": passed,
        "total": total,
        "score": round(overall, 4),
        "task_results": results,
        "elapsed_sec": round(elapsed, 1),
    }
