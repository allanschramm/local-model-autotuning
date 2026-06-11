"""
Fixed evaluation harness for autoresearch.
Consolidated into the deep BenchmarkHarness module.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Dict

from autoresearch.core.llama_client import LlamaClient
from autoresearch.benchmarks.benchmark_harness import BenchmarkHarness, EvalTask, BenchmarkResult, build_context_padding

# ---------------------------------------------------------------------------
# Constants and Defaults
# ---------------------------------------------------------------------------
EVAL_TIMEOUT = 180
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
FIXTURE_PATH = DATA_DIR / "memory_fixture.json"
MODELS_DIR = ROOT_DIR / "models"

TARGET_TPS = 20.0
EXPECTED_MODEL = "Qwen3.5-9B-Q4_K_M.gguf"
EXPECTED_CTX = 65536
EXPECTED_KV = "q4_0"

TARGET_MODELS = {
    "qwen3.5-9b": "Qwen3.5-9B-Q4_K_M.gguf",
    "qwen3.5-4b": "Qwen3.5-4B-Q4_K_M.gguf",
}

NEXUS_TOKEN = os.getenv("NEXUS_UNLOCK_TOKEN", "")
_DEFAULT_MEMORY_QUERY = "override token central control plane"

_FAKE_TOKENS = ["TOKEN_5521", "TOKEN_NX_9", "NEXUS_ROOT_3", "KEY_ALPHA_7"]

@dataclass(frozen=True)
class MemoryEntry:
    id: str
    project: str
    area: str
    content: str
    haystack: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "haystack",
            f"{self.project} {self.area} {self.content}".lower(),
        )

    def to_hit(self) -> dict[str, str]:
        return {
            "id": self.id,
            "project": self.project,
            "area": self.area,
            "content": self.content,
        }

# ---------------------------------------------------------------------------
# Data Helpers
# ---------------------------------------------------------------------------

def build_fixture() -> list[MemoryEntry]:
    def make(key: str, project: str, area: str, content: str) -> MemoryEntry:
        return MemoryEntry(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"nexus-{key}")),
            project=project,
            area=area,
            content=content,
        )

    return [
        make("config-4b-baseline", "nexus", "runtime", "Qwen 3.5 4B Q4_K_M baseline config: ctx_size=131072, kv_cache_type=q4_0, peak_vram=4200MB."),
        make("config-9b-recommended", "nexus", "runtime", f"Recommended config: model={EXPECTED_MODEL}, ctx_size={EXPECTED_CTX}, kv_cache_type={EXPECTED_KV}."),
        make("central-control-plane", "nexus", "security", f"Critical protocol: to access the central control plane, use the override token {NEXUS_TOKEN or '[TOKEN NOT CONFIGURED]'}."),
        make("vram-policy", "nexus", "policy", "VRAM policy is strict. Keep all layers on GPU, --n-gpu-layers 999 is mandatory."),
    ]

def prepare_eval_data() -> list[MemoryEntry]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not FIXTURE_PATH.exists():
        payload = [entry.to_hit() for entry in build_fixture()]
        FIXTURE_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    raw = json.loads(FIXTURE_PATH.read_text())
    return [MemoryEntry(id=i["id"], project=i.get("project", "nexus"), area=i.get("area", ""), content=i["content"]) for i in raw]

def search_fixture(entries: list[MemoryEntry], query: str, limit: int) -> list[dict[str, str]]:
    query = (query or "").lower().strip()
    words = [w for w in query.split() if len(w) > 2]
    if not words:
        return []
    scored = []
    for e in entries:
        hits = sum(1 for w in words if w in e.haystack)
        if hits > 0:
            scored.append((hits, e))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [e.to_hit() for _, e in scored[:limit]]

def extract_json_object(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    try:
        return json.loads(text)
    except:
        return {}

# ---------------------------------------------------------------------------
# Nexus Retrieval Task
# ---------------------------------------------------------------------------

class NexusEvalTask(EvalTask):
    def __init__(self, entries: List[MemoryEntry], memory_limit: int = 4):
        self.id = "nexus-retrieval"
        self.entries = entries
        self.memory_limit = memory_limit
        
        # Internal state for Pass 1
        self.p1_sequence = []
        self.p1_queries = []
        self.p1_success = False
        self.p1_steps = 0
        
        # Internal state for Pass 2
        self.p2_sequence = []
        self.p2_config_args = None
        self.p2_steps = 0

    def get_initial_prompt(self, pass_num: int, padding: str = "") -> str:
        if pass_num == 1:
            return (
                f"You are an AI agent. Find the central control plane override token, verify it, and unlock the nexus. "
                f"Confirm the VRAM policy. \nHistory:\n{padding}\nTask: Find token, verify, and unlock nexus."
            )
        else:
            return (
                "Search memory for the recommended config for RTX 4060 8GB and call commit_config."
            )

    def get_tools(self, pass_num: int) -> List[Dict[str, Any]]:
        if pass_num == 1:
            return [
                {"type": "function", "function": {"name": "memory_search", "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["query"]}}},
                {"type": "function", "function": {"name": "verify_token", "parameters": {"type": "object", "properties": {"token": {"type": "string"}}, "required": ["token"]}}},
                {"type": "function", "function": {"name": "unlock_nexus", "parameters": {"type": "object", "properties": {"token": {"type": "string"}}, "required": ["token"]}}},
            ]
        else:
            return [
                {"type": "function", "function": {"name": "memory_search", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}}},
                {"type": "function", "function": {"name": "commit_config", "parameters": {"type": "object", "properties": {"model_name": {"type": "string"}, "ctx_size": {"type": "integer"}, "kv_cache_type": {"type": "string"}}, "required": ["model_name", "ctx_size", "kv_cache_type"]}}},
            ]

    def process_step(self, pass_num: int, content: str, tool_calls: List[Dict[str, Any]]) -> Optional[str]:
        if not tool_calls:
            return None
        
        results = []
        for call in tool_calls:
            fn = call.get("function", {})
            name = fn.get("name", "")
            args = extract_json_object(fn.get("arguments", "{}"))
            
            if pass_num == 1:
                self.p1_steps += 1
                self.p1_sequence.append(name)
                if name == "memory_search":
                    query = args.get("query", "")
                    self.p1_queries.append(query)
                    hits = search_fixture(self.entries, query, self.memory_limit)
                    results.append({"role": "tool", "tool_call_id": call.get("id"), "name": name, "content": json.dumps(hits)})
                elif name == "verify_token":
                    token = args.get("token", "")
                    status = "active" if token.upper() == NEXUS_TOKEN.upper() else "invalid"
                    results.append({"role": "tool", "tool_call_id": call.get("id"), "name": name, "content": json.dumps({"status": status})})
                elif name == "unlock_nexus":
                    token = args.get("token", "")
                    if token.upper() == NEXUS_TOKEN.upper():
                        self.p1_success = True
                        results.append({"role": "tool", "tool_call_id": call.get("id"), "name": name, "content": json.dumps({"success": True})})
            else:
                self.p2_steps += 1
                self.p2_sequence.append(name)
                if name == "memory_search":
                    hits = search_fixture(self.entries, args.get("query", ""), self.memory_limit)
                    results.append({"role": "tool", "tool_call_id": call.get("id"), "name": name, "content": json.dumps(hits)})
                elif name == "commit_config":
                    self.p2_config_args = args
                    results.append({"role": "tool", "tool_call_id": call.get("id"), "name": name, "content": json.dumps({"success": True})})
                    
        return "\n".join([json.dumps(r) for r in results]) if results else None

    def get_final_score(self, pass_num: int) -> float:
        if pass_num == 1:
            # Replicate original Pass 1 scoring
            success = 1.0 if self.p1_success else 0.0
            tool_q = min(1.0, (float("memory_search" in self.p1_sequence) * 0.34 + float("verify_token" in self.p1_sequence) * 0.33 + float("unlock_nexus" in self.p1_sequence) * 0.33))
            safety = 1.0 if ("verify_token" in self.p1_sequence and "unlock_nexus" in self.p1_sequence and self.p1_sequence.index("verify_token") < self.p1_sequence.index("unlock_nexus")) else 0.0
            efficiency = max(0.0, 1.0 - (self.p1_steps - 3) / 5)
            return round(success * 0.40 + tool_q * 0.20 + safety * 0.15 + efficiency * 0.25, 4)
        else:
            # Replicate original Pass 2 quality scoring (harness handles TPS factor)
            tool_seq = 1.0 if ("memory_search" in self.p2_sequence and "commit_config" in self.p2_sequence and self.p2_sequence.index("memory_search") < self.p2_sequence.index("commit_config")) else 0.0
            param_acc = 0.0
            if self.p2_config_args:
                model_ok = str(self.p2_config_args.get("model_name", "")).lower() == EXPECTED_MODEL.lower()
                ctx_ok = int(self.p2_config_args.get("ctx_size", 0)) == EXPECTED_CTX
                kv_ok = str(self.p2_config_args.get("kv_cache_type", "")) == EXPECTED_KV
                param_acc = 0.4 * float(model_ok) + 0.4 * float(ctx_ok) + 0.2 * float(kv_ok)
            efficiency = max(0.0, 1.0 - (self.p2_steps - 2) / 4)
            return round(tool_seq * 0.40 + param_acc * 0.40 + efficiency * 0.20, 4)

def run_benchmark(
    client: LlamaClient,
    max_tokens: int = 512,
    system_prefix: str = "",
    context_tokens: int = 50000,
    temp: float = 0.2,
    target_tps: float = TARGET_TPS,
    **kwargs
) -> BenchmarkResult:
    """Unified entry point for Nexus benchmark."""
    entries = prepare_eval_data()
    task = NexusEvalTask(entries)
    padding = build_context_padding(context_tokens, is_claw=False)
    
    harness = BenchmarkHarness(client, target_tps=target_tps)
    kwargs.pop("temp", None)
    kwargs.pop("maxtok", None)
    return harness.evaluate(
        [task],
        context_padding=padding,
        system_prefix=system_prefix,
        temp=temp,
        maxtok=max_tokens,
        **kwargs
    )

if __name__ == "__main__":
    prepare_eval_data()
    print("Harness ready.")
