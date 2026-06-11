"""
Strict ClawBench evaluation harness for Nexus models.
Evaluates agent trajectory against the real ClawBench eval_schema.
"""

from __future__ import annotations

import json
import re
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List

from autoresearch.core.llama_client import LlamaClient
from autoresearch.benchmarks.benchmark_harness import BenchmarkHarness, EvalTask, BenchmarkResult

# Paths
ROOT_DIR = Path(__file__).resolve().parent
CLAW_ROOT = Path(__file__).resolve().parent.parent.parent / "ClawBench"
V1_CASES = CLAW_ROOT / "test-cases" / "v1"

@dataclass
class ClawTaskData:
    id: int
    instruction: str
    category: str
    url_pattern: str | None = None
    method: str | None = None
    
    @classmethod
    def from_path(cls, path: Path) -> ClawTaskData | None:
        task_json = path / "task.json"
        if not task_json.exists():
            return None
        try:
            data = json.loads(task_json.read_text())
            meta = data.get("metadata", {})
            eval_s = data.get("eval_schema", {})
            return cls(
                id=meta.get("task_id", 0),
                instruction=data.get("instruction", ""),
                category=meta.get("metaclass", "general"),
                url_pattern=eval_s.get("url_pattern"),
                method=eval_s.get("method")
            )
        except Exception:
            return None

class ClawEvalTask(EvalTask):
    """Adapter for ClawBench tasks."""
    def __init__(self, data: ClawTaskData):
        self.data = data
        self.id = str(data.id)
        self._last_response: str = ""

    def get_initial_prompt(self, pass_num: int, padding: str = "") -> str:
        if pass_num == 1:
            system = "System: Available tools: [browser(url, method, body)]. Output tool calls in JSON format."
            history = f"History:\n{padding}" if padding else ""
            return f"{system}\n{history}\nTask: {self.data.instruction}\nResponse:"
        else:
            # Pass 2 is raw throughput on a clean prompt
            return f"Task: {self.data.instruction}\nResponse:"

    def get_tools(self, pass_num: int) -> List[Dict[str, Any]]:
        # ClawBench uses a single browser tool
        return [
            {
                "type": "function",
                "function": {
                    "name": "browser",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {"type": "string"},
                            "method": {"type": "string", "enum": ["GET", "POST", "PUT", "DELETE"]},
                            "body": {"type": "string"}
                        },
                        "required": ["url"]
                    }
                }
            }
        ]

    def process_step(self, pass_num: int, response: str, tool_calls: List[Dict[str, Any]]) -> Optional[str]:
        # Simple one-step browser interaction for now
        self._last_response = response
        if tool_calls:
            results = []
            for call in tool_calls:
                results.append({"role": "tool", "tool_call_id": call.get("id"), "name": "browser", "content": "{\"status\": 200, \"content\": \"Success\"}"})
            return "\n".join([json.dumps(r) for r in results])
        return None

    def get_final_score(self, pass_num: int) -> float:
        score = 0.0
        response = self._last_response
        if self.data.url_pattern:
            if re.search(self.data.url_pattern, response):
                score += 0.5
            if self.data.method and self.data.method.upper() in response.upper():
                score += 0.3
            if "browser" in response.lower() or "{" in response:
                score += 0.2
        return score

def discover_tasks() -> list[ClawTaskData]:
    tasks = []
    target_patterns = ["dev-tech", "office"]
    if not V1_CASES.exists():
        return []
    
    for case_dir in V1_CASES.iterdir():
        if not case_dir.is_dir():
            continue
        if any(p in case_dir.name for p in target_patterns):
            t = ClawTaskData.from_path(case_dir)
            if t:
                tasks.append(t)
    return sorted(tasks, key=lambda x: x.id)

def build_context_padding(target_tokens: int = 50_000) -> str:
    """Generate deterministic synthetic history (~50k tokens)."""
    ops = ["browser_view", "scroll", "click", "type", "back", "wait"]
    statuses = ["success", "success", "success", "partial", "cached", "skipped"]
    templates = [
        "Viewed product page for research.",
        "Scrolled down to find checkout button.",
        "Clicked on the shopping cart icon.",
        "Typed address into the delivery field.",
        "Waiting for page to load assets.",
        "Navigated back to search results.",
    ]
    rng = random.Random(42)
    target_chars = int(target_tokens * 3.5)
    blocks: list[str] = []
    current_chars = 0
    while current_chars < target_chars:
        op = rng.choice(ops)
        status = rng.choice(statuses)
        task = rng.choice(templates)
        block = f"[Nexus-Log] op={op} | status={status} | activity={task}\n"
        blocks.append(block)
        current_chars += len(block)
    return "".join(blocks)

def run_benchmark(
    client: LlamaClient,
    max_tokens: int = 512,
    system_prefix: str = "",
    context_tokens: int = 50000,
    temp: float = 0.2,
    target_tps: float = 20.0,
    **kwargs
) -> BenchmarkResult:
    """Unified entry point for Claw benchmark."""
    tasks_data = discover_tasks()
    if not tasks_data:
        raise RuntimeError("No ClawBench tasks found.")

    tasks = [ClawEvalTask(d) for d in tasks_data]
    padding = build_context_padding(context_tokens)
    
    harness = BenchmarkHarness(client, target_tps=target_tps)
    kwargs.pop("temp", None)
    kwargs.pop("maxtok", None)
    return harness.evaluate(
        tasks,
        context_padding=padding,
        system_prefix=system_prefix,
        temp=temp,
        maxtok=max_tokens,
        **kwargs
    )

if __name__ == "__main__":
    tasks = discover_tasks()
    print(f"Discovered {len(tasks)} target tasks (dev-tech/office).")
    for t in tasks[:5]:
        print(f"  [{t.id}] {t.category}: {t.instruction[:60]}... (Pattern: {t.url_pattern})")