"""Agentic benchmark runner — Claw-Eval mock services + agent loop + rule-based scoring.

Single-file adapter. No external deps beyond what the repo already has (requests,
PyYAML). Starts mock services as subprocesses, runs an agent loop against a local
llama-server endpoint with OpenAI-compatible tool calling, then scores via
deterministic rule checks from task.yaml.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import yaml

from autoresearch.core.llama_client import GenerationParams, LlamaClient
from autoresearch.core.process_guard import ProcessGuard

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
        self._guard = ProcessGuard()

    def start(self) -> None:
        """Start all mock services declared in the task's services block.

        No full harness-port orphan sweep here: it would kill the live
        llama-server mid-Trial. Pre-flight sweep lives in LlamaServerRunner.
        """
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
            env["PYTHONUTF8"] = "1"
            for k, v in svc.get("env", {}).items():
                env[k] = str(v)
            env["PORT"] = str(port)

            print(f"    [service] starting {name} on :{port} ({script})")
            proc = self._guard.spawn(
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
        """Poll health check endpoint until ready or the service exits."""
        url = svc.get("health_check", "")
        method = svc.get("health_check_method", "GET")
        for _ in range(100):
            if any(proc.poll() is not None for proc in self._procs):
                raise RuntimeError(f"Mock service exited before readiness on :{svc['port']}")
            try:
                req = urllib.request.Request(url, method=method)
                req.add_header("X-Health-Check", "1")
                # Bound each probe so a half-open service cannot stall a trial.
                with urllib.request.urlopen(req, timeout=2.0) as resp:
                    if resp.status < 500:
                        return
            except urllib.error.HTTPError as e:
                if e.code < 500:
                    return
            except Exception:
                pass
            time.sleep(0.3)
        raise RuntimeError(f"Mock service health check timed out on :{svc['port']}")

    def reset_all(self) -> None:
        """Reset all service states between trials."""
        for svc in self.task.get("services", []):
            reset_url = svc.get("reset_endpoint", "")
            if not reset_url:
                continue
            try:
                req = urllib.request.Request(reset_url, method="POST", data=b"{}")
                urllib.request.urlopen(req, timeout=2.0)
            except Exception:
                pass

    def stop(self) -> None:
        """Kill all mock service processes via the Process Guard."""
        self._guard.teardown()
        self._procs.clear()

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
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": tdef["name"],
                    "description": tdef.get("description", ""),
                    "parameters": tdef.get(
                        "input_schema", {"type": "object", "properties": {}, "required": []}
                    ),
                },
            }
        )
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
        with urllib.request.urlopen(req, timeout=10.0) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        return {"error": str(e)}


def _assistant_visible_text(msg: dict) -> str:
    """Surface text for graders: prefer content, fall back to reasoning_content.

    Thinking models (Qwen3.5 / Ornith family) often park the turn in
    ``reasoning_content`` while ``content`` is empty — coding-10 already
    merges this; agentic must too or keyword graders see len=0.
    """
    content = (msg.get("content") or "").strip()
    if content:
        return msg.get("content") or ""
    return msg.get("reasoning_content") or ""


def _assistant_history_message(msg: dict, *, tool_calls: list | None = None) -> dict:
    """Build an assistant history item, preserving reasoning_content when present.

    Dropping reasoning mid-conversation makes later turns HTTP 400 on servers
    that expect thinking-model message continuity.
    """
    out: dict = {
        "role": "assistant",
        "content": msg.get("content") or "",
    }
    reasoning = msg.get("reasoning_content")
    if reasoning:
        out["reasoning_content"] = reasoning
    if tool_calls:
        out["tool_calls"] = tool_calls
    return out


def run_agent_loop(
    client: LlamaClient,
    task: dict,
    gen_params: GenerationParams | None = None,
    max_turns: int = 20,
    turn_timeout: float = 420.0,
    stop: list[str] | None = None,
) -> tuple[str, list[dict], float, dict[str, Any]]:
    """Run one agent loop for a task.

    Uses llama-server's native /v1/chat/completions with tool calling.
    Returns (final_text, tool_calls_made, elapsed_sec, loop_meta) where
    loop_meta = {"length_stops": int, "http_errors": list[str]}.
    """
    gen = gen_params or GenerationParams(max_tokens=4096)
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
    length_stops = 0
    http_errors: list[str] = []
    t_start = time.time()

    # Enforce 420s turn timeout floor to prevent premature cancellation
    effective_timeout = max(
        float(turn_timeout or getattr(client, "timeout", 420.0) or 420.0), 420.0
    )
    stop_tokens = stop if stop is not None else (gen.stop if gen.stop is not None else ["</s>"])
    if isinstance(stop_tokens, str):
        stop_tokens = [stop_tokens]
    elif isinstance(stop_tokens, (tuple, set)):
        stop_tokens = list(stop_tokens)

    for turn in range(max_turns):
        payload = {
            "messages": messages,
            "tools": tool_defs,
            "stream": False,
            "max_tokens": gen.max_tokens,
            "temperature": gen.temp,
            "stop": stop_tokens,
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

        try:
            # Bound idle network waits while allowing long model generations
            # (reasoning models: <think> traces + max_tokens=4096 at ~27 t/s
            # ≈ 152 s plus heavy prefill on 30-50k-token contexts; slow
            # CPU-offloaded MoE rigs (~8-15 t/s effective) need up to ~420 s
            # for a full 4096-token turn).
            with urllib.request.urlopen(req, timeout=effective_timeout) as resp:
                raw = json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            try:
                body = e.read().decode("utf-8", errors="replace")[:2000]
            except Exception:
                body = ""
            msg = f"HTTP {e.code}: {body}"
            print(f"    [agent] turn {turn + 1} request failed: {msg}")
            http_errors.append(msg)
            break
        except Exception as e:
            print(f"    [agent] turn {turn + 1} request failed: {e}")
            http_errors.append(str(e))
            break

        choice = (raw.get("choices") or [{}])[0]
        msg = choice.get("message", {})

        # Check for tool calls
        tool_calls = msg.get("tool_calls") or []

        # llama.cpp reports finish_reason="length" for THREE cases on this
        # stack (verified 2026-08-19/20): max_tokens exhaustion, tool-call
        # boundary stops, AND stops on the "</s>" stop string (not the EOS
        # token) — i.e. every complete final answer. The only reliable cap
        # signal is n_decoded == max_tokens in the per-run server log.
        if choice.get("finish_reason") == "length" and not tool_calls:
            length_stops += 1
            print(
                f"    [agent] turn {turn + 1} finish_reason=length, no tool_calls "
                "(stop-string stop or cap hit — confirm n_decoded in server log)"
            )
        content = _assistant_visible_text(msg)

        if tool_calls:
            # Model wants to use tools — keep reasoning_content for next turns.
            messages.append(_assistant_history_message(msg, tool_calls=tool_calls))

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

                all_tool_calls.append(
                    {
                        "tool": tool_name,
                        "arguments": args,
                        "result": result,
                        "turn": turn + 1,
                    }
                )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.get("id", f"call_{turn}"),
                        "content": json.dumps(result),
                    }
                )
        else:
            # Model produced final answer
            messages.append(_assistant_history_message(msg))
            elapsed = time.time() - t_start
            return (
                content,
                all_tool_calls,
                elapsed,
                {
                    "length_stops": length_stops,
                    "http_errors": http_errors,
                },
            )

    elapsed = time.time() - t_start
    # If we hit max_turns, use last assistant message visible text
    last_content = ""
    for m in reversed(messages):
        if m.get("role") == "assistant":
            last_content = _assistant_visible_text(m)
            break
    return (
        last_content,
        all_tool_calls,
        elapsed,
        {
            "length_stops": length_stops,
            "http_errors": http_errors,
        },
    )


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
            details_parts.append(
                f"{name}: {'PASS' if passed else 'FAIL'} ({target_tool} called {calls}/{min_calls})"
            )

        elif check_type == "keywords_present":
            keywords = [str(kw) for kw in check.get("keywords", [])]
            text_lower = str(final_text or "").lower()
            found = [kw for kw in keywords if kw.lower() in text_lower]
            passed = len(found) > 0
            details_parts.append(
                f"{name}: {'PASS' if passed else 'FAIL'} (keywords found: {found})"
            )

        elif check_type == "categories_present":
            categories = [str(cat) for cat in check.get("categories", [])]
            text_lower = str(final_text or "").lower()
            found = [cat for cat in categories if cat.lower() in text_lower]
            passed = len(found) >= len(categories) * 0.5  # at least half
            details_parts.append(
                f"{name}: {'PASS' if passed else 'FAIL'} (categories: {found}/{categories})"
            )

        elif check_type == "min_length":
            field = check.get("field", "final_text")
            min_len = check.get("min_length", 0)
            text = str(final_text or "") if field == "final_text" else ""
            passed = len(text) >= min_len
            details_parts.append(
                f"{name}: {'PASS' if passed else 'FAIL'} (len={len(text)}/{min_len})"
            )

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
    turn_timeout: float = 420.0,
    stop: list[str] | None = None,
) -> dict:
    """Run agentic evaluation on selected Claw-Eval tasks.

    Returns dict with keys: passed, total, score, task_results, elapsed_sec.
    """
    gen = gen_params or GenerationParams(max_tokens=4096, temp=0.4)
    results: list[dict] = []
    passed = 0
    t_start = time.time()

    for tid in task_ids:
        task_dir = TASKS_DIR / tid
        yaml_path = task_dir / "task.yaml"
        if not yaml_path.exists():
            print(f"  [agentic] SKIP {tid}: task.yaml not found")
            results.append(
                {
                    "task_id": tid,
                    "score": 0.0,
                    "details": "missing",
                    "final_text_length": 0,
                    "tool_calls_count": 0,
                    "elapsed_sec": 0.0,
                    "length_stops": 0,
                    "http_errors": [],
                }
            )
            continue

        with open(yaml_path, encoding="utf-8") as f:
            task = yaml.safe_load(f)

        task_best = 0.0
        task_detail = ""

        for trial in range(trials):
            with ServiceManager(task_dir, task) as svc:
                loop_kwargs: dict[str, Any] = {
                    "turn_timeout": turn_timeout,
                }
                if stop is not None:
                    loop_kwargs["stop"] = stop
                try:
                    final_text, tool_calls, elapsed, loop_meta = run_agent_loop(
                        client,
                        task,
                        gen_params=gen,
                        max_turns=task.get("environment", {}).get("max_turns", 20),
                        **loop_kwargs,
                    )
                except TypeError:
                    final_text, tool_calls, elapsed, loop_meta = run_agent_loop(
                        client,
                        task,
                        gen_params=gen,
                        max_turns=task.get("environment", {}).get("max_turns", 20),
                    )
                scoring = score_task(task, final_text, tool_calls, task_dir)
                svc.reset_all()

            if scoring["score"] > task_best:
                task_best = scoring["score"]
                task_detail = scoring["details"]

            status = "PASS" if scoring["score"] >= 0.5 else "FAIL"
            length_note = (
                f" length_stops={loop_meta['length_stops']}"
                if loop_meta["length_stops"] > 0
                else ""
            )
            print(
                f"  [agentic] {tid} trial{trial + 1}: {status} "
                f"score={scoring['score']:.2f} calls={scoring['tool_calls_count']} "
                f"len={len(final_text)}{length_note} ({scoring['details']})"
            )

        if task_best >= 0.5:
            passed += 1

        results.append(
            {
                "task_id": tid,
                "score": task_best,
                "details": task_detail,
                "final_text_length": len(final_text),
                "tool_calls_count": len(tool_calls),
                "elapsed_sec": round(elapsed, 1),
                "length_stops": loop_meta["length_stops"],
                "http_errors": loop_meta["http_errors"],
            }
        )

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
