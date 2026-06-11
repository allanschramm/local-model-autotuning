from __future__ import annotations
import time
import json
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Protocol, Tuple, Optional
from autoresearch.core.llama_client import LlamaClient

@dataclass
class BenchmarkResult:
    val_score: float
    val_pass1: float
    val_pass2: float
    avg_tps: float
    total_seconds: float
    metadata: Dict[str, Any] = field(default_factory=dict)

class EvalTask(Protocol):
    """Polymorphic interface for an evaluation unit (supports agentic multi-turn)."""
    id: str
    def get_initial_prompt(self, pass_num: int, padding: str = "") -> str: ...
    def get_tools(self, pass_num: int) -> List[Dict[str, Any]]: ...
    def process_step(self, pass_num: int, response: str, tool_calls: List[Dict[str, Any]]) -> Optional[str]: 
        """Return tool result to continue loop, or None to finish."""
        ...
    def get_final_score(self, pass_num: int) -> float: ...

class BenchmarkHarness:
    """Deep module for Dual-Pass evaluation orchestration with agentic support."""
    def __init__(self, client: LlamaClient, target_tps: float = 20.0, p1_weight: float = 0.55):
        self.client = client
        self.target_tps = target_tps
        self.p1_weight = p1_weight
        self.p2_weight = 1.0 - p1_weight

    def _run_task_loop(self, task: EvalTask, pass_num: int, padding: str = "", max_steps: int = 8, **kwargs) -> Tuple[float, int]:
        timeout_at = kwargs.get("timeout_at")
        prompt = task.get_initial_prompt(pass_num, padding)
        prompt = kwargs.pop("system_prefix", "") + prompt
        tools = task.get_tools(pass_num)
        step = 0
        total_tokens = 0
        
        while step < max_steps:
            if timeout_at and time.time() > timeout_at:
                raise TimeoutError("Trial time budget exceeded")
            step += 1
            try:
                # kwargs can override maxtok/temp/etc
                res = self.client.complete(prompt, tools=tools, **kwargs)
                content = res.get("content", "")
                usage = res.get("usage", {})
                total_tokens += int(usage.get("total_tokens", 0) or 0)
                
                choice = res.get("choices", [{}])[0]
                message = choice.get("message", {})
                tool_calls = message.get("tool_calls") or []
                
                tool_result = task.process_step(pass_num, content, tool_calls)
                if tool_result is None:
                    break
                
                # Continue loop
                prompt += f"{content}\n{tool_result}\n"
            except Exception as e:
                if isinstance(e, TimeoutError):
                    raise
                print(f"    - Pass {pass_num} Step {step} FAIL: {e}")
                break
        
        return task.get_final_score(pass_num), total_tokens

    def evaluate(
        self, 
        tasks: List[EvalTask], 
        context_padding: str = "",
        system_prefix: str = "",
        **kwargs
    ) -> BenchmarkResult:
        t_start = time.time()
        timeout_at = kwargs.get("timeout_at")
        p1_scores = []
        p2_scores = []
        p2_total_tokens = 0
        p2_total_time = 0.0

        print(f"  [harness] Evaluating {len(tasks)} tasks...")

        # --- PASS 1: Accuracy (Context Stress) ---
        for task in tasks:
            if timeout_at and time.time() > timeout_at:
                raise TimeoutError("Trial time budget exceeded")
            score_p1, _ = self._run_task_loop(task, pass_num=1, padding=context_padding, system_prefix=system_prefix, **kwargs)
            p1_scores.append(score_p1)

        # --- PASS 2: Throughput (Clean measures) ---
        for task in tasks:
            if timeout_at and time.time() > timeout_at:
                raise TimeoutError("Trial time budget exceeded")
            t0 = time.time()
            
            # Use p2_maxtok/p2_max_steps if provided, otherwise defaults. 
            # We copy kwargs to avoid modifying the caller's dict or Pass 1 state.
            p2_kwargs = kwargs.copy()
            p2_kwargs["maxtok"] = p2_kwargs.pop("p2_maxtok", 128)
            p2_max_steps = p2_kwargs.pop("p2_max_steps", 6)
            
            score_p2_raw, tokens = self._run_task_loop(
                task, 
                pass_num=2, 
                padding="", 
                system_prefix=system_prefix, 
                max_steps=p2_max_steps, 
                **p2_kwargs
            )
            dur = time.time() - t0
            
            p2_total_tokens += tokens
            p2_total_time += dur
            
            tps = tokens / dur if dur > 0 else 0
            speed_factor = 0.5 + 0.5 * min(1.0, tps / self.target_tps)
            p2_scores.append(score_p2_raw * speed_factor)

        avg_p1 = sum(p1_scores) / len(p1_scores) if p1_scores else 0.0
        avg_p2 = sum(p2_scores) / len(p2_scores) if p2_scores else 0.0
        avg_tps = p2_total_tokens / p2_total_time if p2_total_time > 0 else 0.0
        
        val_score = round(self.p1_weight * avg_p1 + self.p2_weight * avg_p2, 6)
        total_seconds = time.time() - t_start

        return BenchmarkResult(
            val_score=val_score,
            val_pass1=avg_p1,
            val_pass2=avg_p2,
            avg_tps=avg_tps,
            total_seconds=total_seconds
        )


def build_context_padding(target_tokens: int = 50_000, is_claw: bool = False) -> str:
    if is_claw:
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
        fmt = "[Nexus-Log] op={op} | status={status} | activity={task}\n"
    else:
        ops = ["scanning", "sync", "validating", "dispatching", "pruning", "archiving"]
        statuses = ["ok", "warning", "pending", "critical", "recovering", "retrying"]
        templates = [
            "Synchronized local telemetry with master node",
            "Dispatched control packet to edge clusters",
            "Scanned system memory for orphaned fragments",
            "Re-validated session handshake for worker-42",
            "Pruned stale operational logs from Q1-2026",
            "Archived encrypted operational snapshot to cold storage",
        ]
        fmt = "NEXUS-OPS | operation={op} | status={status} | context: {task}\n"

    rng = random.Random(42)
    target_chars = int(target_tokens * 3.5)
    blocks = []
    current_chars = 0
    while current_chars < target_chars:
        op = rng.choice(ops)
        status = rng.choice(statuses)
        task = rng.choice(templates)
        block = fmt.format(op=op, status=status, task=task)
        blocks.append(block)
        current_chars += len(block)
    return "".join(blocks)