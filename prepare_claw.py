
"""
Strict ClawBench evaluation harness for Nexus models.
Evaluates agent trajectory against the real ClawBench eval_schema.
"""

from __future__ import annotations

import json
import re
import random
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Paths
ROOT_DIR = Path(__file__).resolve().parent
CLAW_ROOT = Path(__file__).parent / "ClawBench"
V1_CASES = CLAW_ROOT / "test-cases" / "v1"

# Throughput target for scoring
TARGET_TPS = 30.0

@dataclass
class ClawTask:
    id: int
    instruction: str
    category: str
    url_pattern: str | None = None
    method: str | None = None
    
    @classmethod
    def from_path(cls, path: Path) -> ClawTask | None:
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

def discover_tasks() -> list[ClawTask]:
    tasks = []
    # Targeted categories: dev-tech and office
    target_patterns = ["dev-tech", "office"]
    if not V1_CASES.exists():
        return []
    
    for case_dir in V1_CASES.iterdir():
        if not case_dir.is_dir():
            continue
        if any(p in case_dir.name for p in target_patterns):
            t = ClawTask.from_path(case_dir)
            if t:
                tasks.append(t)
    return sorted(tasks, key=lambda x: x.id)

def _call_llama_server(port: int, prompt: str, **kwargs) -> dict[str, Any]:
    url = f"http://127.0.0.1:{port}/completion"
    payload = {
        "prompt": prompt,
        "n_predict": kwargs.get("maxtok", 512),
        "temperature": kwargs.get("temp", 0.1), # Lower temp for strict following
        "stream": False,
        "stop": ["</s>", "Instruction:", "User:", "Task:"]
    }
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=180) as res:
        return json.loads(res.read().decode())

# ---------------------------------------------------------------------------
# Context padding (Pass 1 only)
# ---------------------------------------------------------------------------

_OPS = ["browser_view", "scroll", "click", "type", "back", "wait"]
_STATUSES = ["success", "success", "success", "partial", "cached", "skipped"]
_TASK_TEMPLATES = [
    "Viewed product page for research.",
    "Scrolled down to find checkout button.",
    "Clicked on the shopping cart icon.",
    "Typed address into the delivery field.",
    "Waiting for page to load assets.",
    "Navigated back to search results.",
]

def build_context_padding(target_tokens: int = 50_000) -> str:
    """Generate deterministic synthetic history (~50k tokens)."""
    import random
    rng = random.Random(42)
    target_chars = int(target_tokens * 3.5)
    blocks: list[str] = []
    current_chars = 0
    while current_chars < target_chars:
        op = rng.choice(_OPS)
        status = rng.choice(_STATUSES)
        task = rng.choice(_TASK_TEMPLATES)
        block = f"[Nexus-Log] op={op} | status={status} | activity={task}\n"
        blocks.append(block)
        current_chars += len(block)
    return "".join(blocks)

def evaluate_claw_workflow(
    *,
    model_name: str,
    port: int = 8080,
    temp: float = 0.2,
    target_tps: float = TARGET_TPS,
    context_target_tokens: int = 50_000,
    system_prefix: str = "",
    **kwargs
) -> tuple[float, float, float, float, float]:
    tasks = discover_tasks()
    if not tasks:
        print("Error: No ClawBench tasks found.")
        return 0.0, 0.0, 0.0, 0.0, 0.0

    # Select ALL available tasks for a high-accuracy long run (~10-15 minutes)
    eval_tasks = tasks
    
    t_total_start = time.time()
    pass1_scores = []
    pass2_speeds = []
    total_tokens = 0
    
    print(f"  [eval] Running {len(eval_tasks)} ClawBench tasks with {context_target_tokens} tokens noise...")
    padding = build_context_padding(context_target_tokens)

    for i, task in enumerate(eval_tasks):
        # --- PASS 1: Strict Agency & Context Stress ---
        system_noise = f"{system_prefix}System: Available tools: [browser(url, method, body)]. Output tool calls in JSON format."
        context_noise = f"History:\n{padding}"
        prompt = f"{system_noise}\n{context_noise}\nTask: {task.instruction}\nResponse:"
        
        t0 = time.time()
        try:
            res = _call_llama_server(port, prompt, **kwargs)
            elapsed = time.time() - t0
            content = res.get("content", "")
            
            # Strict Validation: Search for tool calls matching the eval_schema
            score = 0.0
            found_call = False
            
            # Simple regex to find browser(url, method, body) or JSON tool calls
            # Looking for the URL pattern defined in ClawBench task.json
            if task.url_pattern:
                if re.search(task.url_pattern, content):
                    score += 0.5 # Matched the URL target
                if task.method and task.method.upper() in content.upper():
                    score += 0.3 # Matched the HTTP method
                if "browser" in content.lower() or "{" in content:
                    score += 0.2 # Attempted a tool call/structure
            
            pass1_scores.append(score)
            
            # --- PASS 2: Throughput ---
            # Measure raw speed on a clean prompt
            clean_prompt = f"Task: {task.instruction}\nResponse:"
            t_s = time.time()
            res_speed = _call_llama_server(port, clean_prompt, maxtok=128)
            dur = time.time() - t_s
            toks = res_speed.get("tokens_predicted", 1)
            pass2_speeds.append(toks / dur if dur > 0 else 0)
            total_tokens += toks
            
            print(f"    - Task {task.id}: score={score:.2f} speed={toks/dur:.1f}tps")
            
        except Exception as e:
            print(f"    - Task {task.id}: ERROR {e}")
            pass1_scores.append(0.0)
            pass2_speeds.append(0.0)

    val_pass1 = sum(pass1_scores) / len(pass1_scores) if pass1_scores else 0.0
    avg_tps = sum(pass2_speeds) / len(pass2_speeds) if pass2_speeds else 0.0
    
    # Pass 2 quality is tied to speed
    speed_factor = 0.5 + 0.5 * min(1.0, avg_tps / target_tps)
    val_pass2 = val_pass1 * speed_factor # Quality * Speed
    
    val_score = round(0.55 * val_pass1 + 0.45 * val_pass2, 6)
    total_seconds = time.time() - t_total_start
    
    return val_score, val_pass1, val_pass2, avg_tps, total_seconds

if __name__ == "__main__":
    tasks = discover_tasks()
    print(f"Discovered {len(tasks)} target tasks (dev-tech/office).")
    for t in tasks[:5]:
        print(f"  [{t.id}] {t.category}: {t.instruction[:60]}... (Pattern: {t.url_pattern})")