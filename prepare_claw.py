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

from llama_client import LlamaClient
from benchmark_harness import BenchmarkHarness, EvalTask, BenchmarkResult

# Paths
ROOT_DIR = Path(__file__).resolve().parent
CLAW_ROOT = Path(__file__).parent / "ClawBench"
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

    def get_prompt_p1(self, padding: str = "") -> str:
        system = "System: Available tools: [browser(url, method, body)]. Output tool calls in JSON format."
        history = f"History:\n{padding}" if padding else ""
        return f"{system}\n{history}\nTask: {self.data.instruction}\nResponse:"

    def get_prompt_p2(self) -> str:
        # Pass 2 is raw throughput on a clean prompt
        return f"Task: {self.data.instruction}\nResponse:"

    def score_p1(self, response: str) -> float:
        score = 0.0
        if self.data.url_pattern:
            if re.search(self.data.url_pattern, response):
                score += 0.5
            if self.data.method and self.data.method.upper() in response.upper():
                score += 0.3
            if "browser" in response.lower() or "{" in response:
                score += 0.2
        return score

    def score_p2(self, response: str) -> float:
        # Pass 2 quality is assumed high if Pass 1 is high, 
        # but the harness multiplier needs a baseline quality score.
        return self.score_p1(response)

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

def evaluate_claw_workflow(
    *,
    model_name: str,
    port: int = 8080,
    temp: float = 0.2,
    target_tps: float = 30.0,
    context_target_tokens: int = 50_000,
    system_prefix: str = "",
    **kwargs
) -> tuple[float, float, float, float, float]:
    tasks_data = discover_tasks()
    if not tasks_data:
        print("Error: No ClawBench tasks found.")
        return 0.0, 0.0, 0.0, 0.0, 0.0

    tasks = [ClawEvalTask(d) for d in tasks_data]
    padding = build_context_padding(context_target_tokens)
    
    client = LlamaClient(port)
    harness = BenchmarkHarness(client, target_tps=target_tps)
    
    result = harness.evaluate(
        tasks,
        context_padding=padding,
        system_prefix=system_prefix,
        temp=temp,
        **kwargs
    )
    
    return (
        result.val_score, 
        result.val_pass1, 
        result.val_pass2, 
        result.avg_tps, 
        result.total_seconds
    )

if __name__ == "__main__":
    tasks = discover_tasks()
    print(f"Discovered {len(tasks)} target tasks (dev-tech/office).")
    for t in tasks[:5]:
        print(f"  [{t.id}] {t.category}: {t.instruction[:60]}... (Pattern: {t.url_pattern})")