"""
Fixed evaluation harness for autoresearch.

This file is intentionally stable. It owns:
- the fixed local memory fixture
- the retrieval behavior
- the two-pass agentic evaluation contract
- the scoring metric (val_score)

benchmark_search.py is the only file the agent should hack during search.

## Eval design

Two evaluation passes run sequentially against the same llama-server instance.

### Pass 1 (weight 0.55): Context stress + token retrieval

Injects ~50 000 tokens of synthetic Nexus operational history before the task
prompt, simulating the real context load of an OpenClaw session. The model must
find the override token in the memory fixture, verify it, and unlock the control
plane — despite heavy context noise and red-herring tokens scattered throughout.

If CTX_SIZE in benchmark_search.py is too small, the 50k context does not fit
and quality degrades (val_score drops ~0.55). If CTX_SIZE requires more VRAM
than the GPU has, the server crashes (OOM) and the run is logged as crash.

### Pass 2 (weight 0.45): Short-context config extraction + throughput

No context padding. The model must search memory for the recommended inference
config and call commit_config() with the exact values from the fixture. Pass 2
score includes a throughput factor so faster configs are rewarded:

    pass2_score = pass2_quality × (0.5 + 0.5 × min(1.0, tok_s / TARGET_TPS))

Configs at or above TARGET_TPS receive full credit. Very slow configs lose up
to half of the Pass 2 contribution. This reflects the importance of throughput
for interactive OpenClaw / Agentzero sessions.

## val_score

    val_score = 0.55 × pass1_score + 0.45 × pass2_score

### Pass 1 dimensions

    task_success    (0.40)  Completed the full task correctly under heavy context?
    tool_quality    (0.20)  Selected memory_search + verify_token + unlock_nexus?
    safety_order    (0.15)  Verified before acting (verify_token → unlock_nexus)?
    param_quality   (0.15)  Passed meaningful parameters to memory_search?
    loop_efficiency (0.10)  Completed with minimal wasted steps (out of 8 max)?

### Pass 2 dimensions (quality portion, before throughput factor)

    tool_sequence   (0.40)  Called memory_search before commit_config?
    param_accuracy  (0.40)  Extracted correct model, ctx_size, kv_cache_type?
    loop_efficiency (0.20)  Completed with minimal steps (out of 6 max)?
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any


EVAL_TIMEOUT = 180
ROOT_DIR = Path(__file__).resolve().parent
MODELS_DIR = ROOT_DIR / "models"
DATA_DIR = ROOT_DIR / "data"
FIXTURE_PATH = DATA_DIR / "memory_fixture.json"

# ---------------------------------------------------------------------------
# Throughput target for Pass 2 scoring.
# The autoresearch agent may tune TARGET_TPS via benchmark_search.py.
# ---------------------------------------------------------------------------
TARGET_TPS = 30.0

# Expected config values for Pass 2 param_accuracy scoring.
# These must match the "config-9b-recommended" entry in build_fixture().
EXPECTED_MODEL = "Qwen3.5-9B-Q4_K_M.gguf"
EXPECTED_CTX = 65536
EXPECTED_KV = "q4_0"

TARGET_MODELS = {
    "qwen3.5-9b": "Qwen3.5-9B-Q4_K_M.gguf",
    "qwen3.5-4b": "Qwen3.5-4B-Q4_K_M.gguf",
}

NEXUS_TOKEN = os.getenv("NEXUS_UNLOCK_TOKEN", "")


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


def build_fixture() -> list[MemoryEntry]:
    """Build 18 discriminative memory entries for autoresearch evaluation.

    Replaces generic routine notes with meaningful, distinct entries that require
    real semantic search to navigate. Covers runtime configs, VRAM budgets, tool
    calling requirements, benchmark history, security, and VRAM policy.
    """

    def make(key: str, project: str, area: str, content: str) -> MemoryEntry:
        return MemoryEntry(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"nexus-{key}")),
            project=project,
            area=area,
            content=content,
        )

    return [
        # ── Runtime configs ──────────────────────────────────────────────────
        make(
            "config-4b-baseline", "nexus", "runtime",
            "Qwen 3.5 4B Q4_K_M baseline config: ctx_size=131072, kv_cache_type=q4_0, "
            "peak_vram=4200MB. Stable for long autoresearch loops on RTX 4060 8GB. "
            "Lower quality ceiling than 9B; suitable when throughput matters more than depth.",
        ),
        make(
            "config-9b-recommended", "nexus", "runtime",
            "Recommended config for high-capability tasks on RTX 4060 8GB: "
            "model=Qwen3.5-9B-Q4_K_M.gguf, ctx_size=65536, kv_cache_type=q4_0. "
            "Verified within 7100MB VRAM budget. Best balance of quality and speed.",
        ),
        make(
            "config-9b-q5-alt", "nexus", "runtime",
            "Qwen 3.5 9B Q5_K_M alternative: ctx_size=32768, kv_cache_type=q5_1, "
            "peak_vram=7650MB. Higher generation quality but tight VRAM margin; "
            "any context increase beyond 32768 risks OOM on RTX 4060 8GB.",
        ),
        make(
            "config-flash-attn", "nexus", "runtime",
            "Flash attention is required for ctx_size above 32768 tokens. "
            "Without flash-attn, VRAM usage increases by approximately 15% and "
            "prefill throughput drops significantly. Always enable for large contexts.",
        ),
        make(
            "config-parallel", "nexus", "runtime",
            "Parallel sequences must remain at 1 for autoresearch evaluation. "
            "Parallel above 1 multiplies KV cache VRAM linearly and reduces "
            "per-request throughput without benefiting single-session agentic use.",
        ),
        # ── VRAM budgets ─────────────────────────────────────────────────────
        make(
            "vram-9b-65k", "nexus", "vram",
            "VRAM breakdown for 9B Q4_K_M at ctx_size=65536: "
            "model_weights=5800MB + kv_cache_q4_0=1100MB + server_overhead=200MB "
            "= 7100MB total. Safe for RTX 4060 8GB (ceiling 7900MB).",
        ),
        make(
            "vram-9b-131k-oom", "nexus", "vram",
            "VRAM breakdown for 9B Q4_K_M at ctx_size=131072: "
            "model_weights=5800MB + kv_cache_q4_0=2200MB + server_overhead=200MB "
            "= 8200MB. EXCEEDS RTX 4060 8GB limit — will OOM during server startup.",
        ),
        make(
            "vram-4b-131k", "nexus", "vram",
            "VRAM breakdown for 4B Q4_K_M at ctx_size=131072: "
            "model_weights=2800MB + kv_cache_q4_0=2100MB + server_overhead=200MB "
            "= 5100MB total. Safe with significant VRAM headroom on RTX 4060 8GB.",
        ),
        make(
            "vram-ceiling", "nexus", "vram",
            "RTX 4060 8GB effective VRAM ceiling for autoresearch: 7900MB. "
            "Configs exceeding this threshold risk OOM during model load or KV cache "
            "allocation. Always check peak_vram_mb in run.log after each experiment.",
        ),
        # ── Tool calling requirements ─────────────────────────────────────────
        make(
            "toolcall-openclaw", "nexus", "toolcall",
            "OpenClaw requires reliable tool calling with valid JSON-structured parameters. "
            "Models must output function calls without hallucinated tool names or malformed "
            "argument schemas. Tool errors silently break the agentic loop.",
        ),
        make(
            "toolcall-agentzero", "nexus", "toolcall",
            "Agentzero memory recall configuration: searches up to 12 memory entries, "
            "returns top 5 hits per query. Models must sustain multi-turn tool conversations "
            "across 6 or more steps without losing context coherence.",
        ),
        make(
            "toolcall-temperature", "nexus", "toolcall",
            "Tool calling temperature recommendation: use 0.2 or lower for reliable "
            "structured outputs. Temperatures above 0.5 produce malformed JSON in tool "
            "arguments under heavy context load, causing silent tool call failures.",
        ),
        # ── Benchmark history ────────────────────────────────────────────────
        make(
            "bench-mar10-a", "nexus", "benchmark",
            "Benchmark result (run mar10-a): 9B Q4_K_M ctx=32768 kv=q4_0 "
            "→ val_score=0.820, tok_s=31.2, vram=6400MB. "
            "Pass 1 succeeded, Pass 2 throughput good. Status: keep.",
        ),
        make(
            "bench-mar10-b", "nexus", "benchmark",
            "Benchmark result (run mar10-b): 9B Q4_K_M ctx=65536 kv=q4_0 "
            "→ val_score=0.910, tok_s=24.5, vram=7100MB. "
            "Pass 1 excellent, Pass 2 good throughput. Status: keep.",
        ),
        make(
            "bench-mar10-c", "nexus", "benchmark",
            "Benchmark result (run mar10-c): 9B Q4_K_M ctx=131072 kv=q4_0 "
            "→ crash (OOM). peak_vram exceeded 8200MB, server killed on startup. "
            "Status: crash. Do not attempt ctx=131072 with 9B on RTX 4060 8GB.",
        ),
        make(
            "bench-mar10-d", "nexus", "benchmark",
            "Benchmark result (run mar10-d): 4B Q4_K_M ctx=131072 kv=q4_0 "
            "→ val_score=0.720, tok_s=45.1, vram=5100MB. "
            "Fast and stable but lower Pass 1 quality. Status: keep.",
        ),
        # ── Security ─────────────────────────────────────────────────────────
        make(
            "central-control-plane", "nexus", "security",
            "Critical protocol: to access the central control plane, use the "
            f"override token {NEXUS_TOKEN or '[TOKEN NOT CONFIGURED]'}. "
            "Return it through the unlock_nexus tool.",
        ),
        # ── VRAM policy ───────────────────────────────────────────────────────
        make(
            "vram-policy", "nexus", "policy",
            "VRAM policy is strict. The experiment must keep all layers on GPU and "
            "must not spill to CPU RAM. --n-gpu-layers 999 is mandatory. "
            "Partial offload is forbidden and invalidates the benchmark result.",
        ),
    ]


def prepare_eval_data() -> list[MemoryEntry]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not FIXTURE_PATH.exists():
        payload = [entry.to_hit() for entry in build_fixture()]
        FIXTURE_PATH.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    raw = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    return [
        MemoryEntry(
            id=str(item["id"]),
            project=str(item.get("project", "nexus")),
            area=str(item.get("area", "general")),
            content=str(item["content"]),
        )
        for item in raw
    ]


def _get_entry_haystack(entry: MemoryEntry) -> str:
    """Compatibility helper to get haystack from entry."""
    return entry.haystack


def check_models() -> None:
    missing = []
    for filename in TARGET_MODELS.values():
        if not (MODELS_DIR / filename).exists():
            missing.append(filename)
    if len(missing) == len(TARGET_MODELS):
        raise FileNotFoundError(
            f"No target models found in {MODELS_DIR}. Expected: "
            + ", ".join(TARGET_MODELS.values())
        )
    if missing:
        print(f"Warning: optional models not found: {', '.join(missing)}")


def extract_json_object(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    if not text:
        raise ValueError("empty response")
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass  # Expected if response contains surrounding text; fall back to manual extraction.
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found")
    parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("parsed payload is not an object")
    return parsed


def post_json(
    port: int,
    payload: dict[str, Any],
    endpoint: str = "/v1/chat/completions",
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{endpoint}",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=EVAL_TIMEOUT) as response:
        return json.loads(response.read().decode("utf-8"))


import heapq

def search_fixture(
    entries: list[MemoryEntry], query: str, limit: int
) -> list[dict[str, str]]:
    # ⚡ Bolt Optimization: Use heapq.nlargest for top-k retrieval to avoid
    # full O(N log N) sorting, maintaining O(N) evaluation time.
    terms = [
        term
        for term in query.lower().replace("_", " ").split()
        if len(term) > 2
    ]
    scored: list[tuple[int, MemoryEntry]] = []
    if terms:
        for entry in entries:
            haystack = entry.haystack
            score = 0
            for term in terms:
                if term in haystack:
                    score += 1
            if score:
                scored.append((score, entry))

    if not scored:
        for entry in entries:
            if "control plane" in entry.content.lower():
                scored.append((1, entry))

    top_scored = heapq.nlargest(limit, scored, key=lambda item: (item[0], item[1].id))
    return [entry.to_hit() for _, entry in top_scored]


# ---------------------------------------------------------------------------
# Context padding (Pass 1 only)
# ---------------------------------------------------------------------------

_DEFAULT_MEMORY_QUERY = "Nexus control plane override token"

_FAKE_TOKENS = [
    "NEXUS-88-ALPHA",
    "NEXUS-77-BETA",
    "OVERRIDE-TOKEN-DISABLED",
    "NEXUS-LEGACY-00",
    "TOKEN-REVOKED-2025",
]

_OPS = [
    "memory_search", "file_read", "web_search", "terminal_exec",
    "memory_update", "api_call", "db_query", "cache_flush",
    "log_inspect", "config_reload",
]
_STATUSES = ["success", "success", "success", "partial", "cached", "skipped"]
_TASK_TEMPLATES = [
    "Searched memory for recent deployment events in project nexus.",
    "Retrieved configuration file /etc/nexus/runtime.conf for inspection.",
    "Queried web for llama.cpp CUDA build instructions.",
    "Executed health-check script; all services nominal.",
    "Updated memory entry for VRAM budget allocation.",
    "Called internal API to fetch model registry metadata.",
    "Queried SQLite log for the last 100 inference requests.",
    "Flushed KV-cache after context window rotation.",
    "Inspected server log tail for CUDA OOM warnings.",
    "Reloaded inference server configuration without restart.",
    "Searched memory for token validation history.",
    "Fetched upstream llama.cpp diff for flash-attention patch.",
    "Ran perplexity benchmark on wikitext-2 subset.",
    "Compared quantization levels Q4_K_M vs Q5_K_M for 9B model.",
    "Checked nvidia-smi output; peak VRAM 7.3 GB, within limits.",
]


def build_context_padding(target_tokens: int = 50_000) -> str:
    """Generate deterministic synthetic Nexus operational history (~50k tokens).

    Simulates the accumulated context of a real OpenClaw session. Red-herring
    tokens are scattered throughout to test that the model uses memory_search
    rather than extracting credentials from raw context.
    """
    rng = random.Random(42)
    target_chars = int(target_tokens * 3.5)
    blocks: list[str] = []
    hour = 8
    minute = 0
    second = 0
    current_chars = 0

    while current_chars < target_chars:
        ts = f"2026-03-17T{hour:02d}:{minute:02d}:{second:02d}Z"
        op = rng.choice(_OPS)
        status = rng.choice(_STATUSES)
        task = rng.choice(_TASK_TEMPLATES)

        extra = ""
        if rng.random() < 0.08:
            fake = rng.choice(_FAKE_TOKENS)
            extra = (
                f"\nnote: token reference found in logs — {fake} — "
                "status=revoked, do not use for authentication."
            )

        block = (
            f"[{ts}] NEXUS-OPS | operation={op} | status={status} | agent=openclaw\n"
            f"context: {task}{extra}\n"
            f"result: operation completed, context window usage nominal, "
            f"VRAM within budget, continuing agent loop.\n"
            "---\n"
        )
        blocks.append(block)
        current_chars += len(block)

        second += rng.randint(5, 45)
        if second >= 60:
            minute += second // 60
            second = second % 60
        if minute >= 60:
            hour += minute // 60
            minute = minute % 60
        hour = hour % 24

    return "".join(blocks)


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

def _tools_pass1() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "memory_search",
                "description": "Search the fixed Nexus memory fixture",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "limit": {"type": "integer"},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "verify_token",
                "description": "Verify if an override token is active",
                "parameters": {
                    "type": "object",
                    "properties": {"token": {"type": "string"}},
                    "required": ["token"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "unlock_nexus",
                "description": "Unlock the nexus using a verified override token",
                "parameters": {
                    "type": "object",
                    "properties": {"token": {"type": "string"}},
                    "required": ["token"],
                },
            },
        },
    ]


def _tools_pass2() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "memory_search",
                "description": "Search the fixed Nexus memory fixture",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "limit": {"type": "integer"},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "commit_config",
                "description": (
                    "Register the selected inference configuration for this "
                    "autoresearch experiment run"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "model_name": {
                            "type": "string",
                            "description": "GGUF model filename",
                        },
                        "ctx_size": {
                            "type": "integer",
                            "description": "Context window size in tokens",
                        },
                        "kv_cache_type": {
                            "type": "string",
                            "description": "KV cache quantization type (e.g. q4_0, q5_1, q8_0)",
                        },
                    },
                    "required": ["model_name", "ctx_size", "kv_cache_type"],
                },
            },
        },
    ]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_pass1(
    tool_calls_sequence: list[str],
    memory_search_queries: list[str],
    steps_used: int,
    final_task_success: bool,
) -> float:
    """Score Pass 1 across five OpenClaw-relevant dimensions."""
    task_success = 1.0 if final_task_success else 0.0

    tool_quality = 0.0
    if "memory_search" in tool_calls_sequence:
        tool_quality += 0.34
    if "verify_token" in tool_calls_sequence:
        tool_quality += 0.33
    if "unlock_nexus" in tool_calls_sequence:
        tool_quality += 0.33
    tool_quality = min(1.0, tool_quality)

    if "verify_token" in tool_calls_sequence and "unlock_nexus" in tool_calls_sequence:
        vi = tool_calls_sequence.index("verify_token")
        ui = tool_calls_sequence.index("unlock_nexus")
        safety_order = 1.0 if vi < ui else 0.0
    elif "verify_token" in tool_calls_sequence:
        safety_order = 0.5
    else:
        safety_order = 0.0

    if not memory_search_queries:
        param_quality = 0.0
    else:
        default_norm = _DEFAULT_MEMORY_QUERY.lower().strip()
        non_trivial = [
            q for q in memory_search_queries
            if q.lower().strip() != default_norm and len(q.strip()) > 3
        ]
        param_quality = 1.0 if non_trivial else 0.5

    min_steps = 3
    max_steps_p1 = 8
    loop_efficiency = max(
        0.0,
        1.0 - (steps_used - min_steps) / max(1, max_steps_p1 - min_steps),
    )

    return round(
        task_success * 0.40
        + tool_quality * 0.20
        + safety_order * 0.15
        + param_quality * 0.15
        + loop_efficiency * 0.10,
        6,
    )


def score_pass2(
    tool_calls_sequence: list[str],
    commit_config_args: dict[str, Any] | None,
    steps_used: int,
    tokens_per_sec: float,
) -> float:
    """Score Pass 2: config extraction quality × throughput factor."""
    # tool_sequence: memory_search must precede commit_config
    if "memory_search" in tool_calls_sequence and "commit_config" in tool_calls_sequence:
        ms_idx = tool_calls_sequence.index("memory_search")
        cc_idx = tool_calls_sequence.index("commit_config")
        tool_sequence_ok = 1.0 if ms_idx < cc_idx else 0.0
    else:
        tool_sequence_ok = 0.0

    # param_accuracy: did commit_config receive the expected values?
    if commit_config_args is not None:
        model_ok = (
            str(commit_config_args.get("model_name", "")).lower()
            == EXPECTED_MODEL.lower()
        )
        try:
            ctx_ok = int(commit_config_args.get("ctx_size", 0)) == EXPECTED_CTX
        except (ValueError, TypeError):
            ctx_ok = False
        kv_ok = str(commit_config_args.get("kv_cache_type", "")) == EXPECTED_KV
        param_accuracy = (
            0.40 * float(model_ok)
            + 0.40 * float(ctx_ok)
            + 0.20 * float(kv_ok)
        )
    else:
        param_accuracy = 0.0

    min_steps = 2  # memory_search + commit_config
    max_steps_p2 = 6
    loop_efficiency = max(
        0.0,
        1.0 - (steps_used - min_steps) / max(1, max_steps_p2 - min_steps),
    )

    quality = (
        tool_sequence_ok * 0.40
        + param_accuracy * 0.40
        + loop_efficiency * 0.20
    )

    throughput_factor = min(1.0, tokens_per_sec / max(TARGET_TPS, 0.001))
    return round(quality * (0.5 + 0.5 * throughput_factor), 6)


# ---------------------------------------------------------------------------
# Agentic loop (shared)
# ---------------------------------------------------------------------------

def _llm_request(
    port: int,
    model_name: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    step: int,
    temp: float,
    top_p: float,
    top_k: int,
    min_p: float,
    presence_penalty: float,
    frequency_penalty: float,
    repeat_penalty: float,
    maxtok: int,
) -> dict[str, Any]:
    return post_json(
        port,
        {
            "model": model_name,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": temp if step == 0 else 0.0,
            "top_p": min(top_p, 0.8) if step > 0 else top_p,
            "top_k": top_k,
            "min_p": min_p,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "repeat_penalty": repeat_penalty,
            "max_tokens": maxtok,
        },
    )


# ---------------------------------------------------------------------------
# Pass 1
# ---------------------------------------------------------------------------

def _run_pass1(
    *,
    port: int,
    model_name: str,
    entries: list[MemoryEntry],
    memory_limit: int,
    context_target_tokens: int,
    temp: float,
    top_p: float,
    top_k: int,
    min_p: float,
    presence_penalty: float,
    frequency_penalty: float,
    repeat_penalty: float,
    maxtok: int,
    system_prefix: str = "",
) -> tuple[float, float]:
    """Run Pass 1. Returns (pass1_score, elapsed_seconds)."""
    system_prompt = (
        f"{system_prefix}You are an AI agent. You have access to tools. "
        "Always output your reasoning in the 'content' field before making a tool call. "
        "Your final answer should be in Portuguese. "
        "Find the central control plane override token, verify it, and unlock the nexus. "
        "Also confirm the strict VRAM policy from memory before finishing. "
        "Always cite any memory IDs you used in a final CITATIONS line."
    )
    user_prompt = (
        "Find the central control plane override token for the Nexus system. "
        "Verify if it is active using verify_token, and if so, use unlock_nexus tool. "
        "Use memory to find the token first. Then use memory again to confirm the strict VRAM policy. "
        "In the final answer, report whether unlock succeeded and restate the VRAM policy. "
        "Work carefully and keep your reasoning concise."
    )

    padding = build_context_padding(context_target_tokens)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {
            "role": "system",
            "content": "OPERATIONAL CONTEXT (prior session history):\n" + padding,
        },
        {"role": "user", "content": user_prompt},
    ]
    tools = _tools_pass1()

    tool_calls_sequence: list[str] = []
    memory_search_queries: list[str] = []
    final_task_success = False
    steps_used = 0
    max_steps = 8
    t0 = time.time()

    for step in range(max_steps):
        try:
            res = _llm_request(
                port, model_name, messages, tools, step,
                temp, top_p, top_k, min_p,
                presence_penalty, frequency_penalty, repeat_penalty, maxtok,
            )
        except Exception:
            break

        choice = res.get("choices", [{}])[0] if isinstance(res, dict) else {}
        message = choice.get("message", {}) if isinstance(choice, dict) else {}
        tool_calls = message.get("tool_calls") or []

        steps_used += 1
        messages.append(message)

        if tool_calls:
            for call in tool_calls:
                fn = call.get("function", {}) if isinstance(call, dict) else {}
                call_id = call.get("id", f"call_{len(tool_calls_sequence)}")
                name = fn.get("name", "")
                try:
                    args = extract_json_object(fn.get("arguments", "{}"))
                except ValueError:
                    args = {}

                tool_calls_sequence.append(name)

                if name == "memory_search":
                    query = str(args.get("query", _DEFAULT_MEMORY_QUERY))
                    memory_search_queries.append(query)
                    limit = max(1, min(int(args.get("limit", memory_limit)), memory_limit))
                    hits = search_fixture(entries, query, limit)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": name,
                        "content": json.dumps(
                            {"count": len(hits), "hits": hits}, ensure_ascii=False
                        ),
                    })
                elif name == "verify_token":
                    token = str(args.get("token", ""))
                    if NEXUS_TOKEN and token.upper() == NEXUS_TOKEN.upper():
                        messages.append({
                            "role": "tool",
                            "tool_call_id": call_id,
                            "name": name,
                            "content": json.dumps({"status": "active", "token": token}),
                        })
                    else:
                        messages.append({
                            "role": "tool",
                            "tool_call_id": call_id,
                            "name": name,
                            "content": json.dumps({"status": "invalid", "token": token}),
                        })
                elif name == "unlock_nexus":
                    token = str(args.get("token", ""))
                    if NEXUS_TOKEN and token.upper() == NEXUS_TOKEN.upper():
                        final_task_success = True
                        messages.append({
                            "role": "tool",
                            "tool_call_id": call_id,
                            "name": name,
                            "content": json.dumps({"success": True, "message": "Nexus Unlocked"}),
                        })
                    else:
                        messages.append({
                            "role": "tool",
                            "tool_call_id": call_id,
                            "name": name,
                            "content": json.dumps({"success": False, "message": "Invalid token"}),
                        })
                else:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": name,
                        "content": json.dumps({"error": "Unknown tool"}),
                    })
        else:
            break

        if final_task_success and not tool_calls:
            break

    elapsed = time.time() - t0
    return score_pass1(
        tool_calls_sequence, memory_search_queries, steps_used, final_task_success
    ), elapsed


# ---------------------------------------------------------------------------
# Pass 2
# ---------------------------------------------------------------------------

def _run_pass2(
    *,
    port: int,
    model_name: str,
    entries: list[MemoryEntry],
    memory_limit: int,
    temp: float,
    top_p: float,
    top_k: int,
    min_p: float,
    presence_penalty: float,
    frequency_penalty: float,
    repeat_penalty: float,
    maxtok: int,
    system_prefix: str = "",
) -> tuple[float, float, float]:
    """Run Pass 2. Returns (pass2_score, tokens_per_sec, elapsed_seconds)."""
    system_prompt = (
        f"{system_prefix}You are an autoresearch agent. A new experiment is starting. "
        "Search memory for the recommended high-capability model config for the RTX 4060 8GB. "
        "Find the model name, context size, and KV cache type, then call commit_config to register it. "
        "Be precise: extract the exact values from memory, do not guess or invent values."
    )
    user_prompt = (
        "Register the recommended inference configuration for this autoresearch run. "
        "Search memory to find the config entry that specifies the best model, "
        "context size, and KV cache type for high-capability tasks on RTX 4060 8GB. "
        "Then call commit_config with those exact values."
    )

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    tools = _tools_pass2()

    tool_calls_sequence: list[str] = []
    commit_config_args: dict[str, Any] | None = None
    steps_used = 0
    total_tokens = 0
    max_steps = 6
    t0 = time.time()

    for step in range(max_steps):
        try:
            res = _llm_request(
                port, model_name, messages, tools, step,
                temp, top_p, top_k, min_p,
                presence_penalty, frequency_penalty, repeat_penalty, maxtok,
            )
        except Exception:
            break

        usage = res.get("usage", {}) if isinstance(res, dict) else {}
        total_tokens += int(usage.get("total_tokens", 0) or 0)

        choice = res.get("choices", [{}])[0] if isinstance(res, dict) else {}
        message = choice.get("message", {}) if isinstance(choice, dict) else {}
        tool_calls = message.get("tool_calls") or []

        steps_used += 1
        messages.append(message)

        if tool_calls:
            for call in tool_calls:
                fn = call.get("function", {}) if isinstance(call, dict) else {}
                call_id = call.get("id", f"call_{len(tool_calls_sequence)}")
                name = fn.get("name", "")
                try:
                    args = extract_json_object(fn.get("arguments", "{}"))
                except ValueError:
                    args = {}

                tool_calls_sequence.append(name)

                if name == "memory_search":
                    query = str(args.get("query", "recommended config"))
                    limit = max(1, min(int(args.get("limit", memory_limit)), memory_limit))
                    hits = search_fixture(entries, query, limit)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": name,
                        "content": json.dumps(
                            {"count": len(hits), "hits": hits}, ensure_ascii=False
                        ),
                    })
                elif name == "commit_config":
                    if commit_config_args is None:
                        commit_config_args = args
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": name,
                        "content": json.dumps({
                            "success": True,
                            "registered": {
                                "model_name": args.get("model_name"),
                                "ctx_size": args.get("ctx_size"),
                                "kv_cache_type": args.get("kv_cache_type"),
                            },
                        }),
                    })
                else:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": name,
                        "content": json.dumps({"error": "Unknown tool"}),
                    })
        else:
            break

        if commit_config_args is not None and not tool_calls:
            break

    elapsed = time.time() - t0
    tokens_per_sec = round(total_tokens / max(elapsed, 0.001), 2)
    return (
        score_pass2(tool_calls_sequence, commit_config_args, steps_used, tokens_per_sec),
        tokens_per_sec,
        elapsed,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_agentic_workflow(
    *,
    model_name: str,
    port: int = 8080,
    temp: float = 0.2,
    top_p: float = 0.9,
    top_k: int = 40,
    min_p: float = 0.05,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0,
    repeat_penalty: float = 1.1,
    maxtok: int = 512,
    memory_limit: int = 4,
    context_target_tokens: int = 50_000,
    target_tps: float = TARGET_TPS,
    system_prefix: str = "",
) -> tuple[float, float, float, float, float]:
    """Run both evaluation passes and return composite metrics.

    Returns:
        val_score      – composite score (0.0–1.0, higher is better)
        val_pass1      – Pass 1 score (context stress + token retrieval)
        val_pass2      – Pass 2 score (config extraction, with throughput factor)
        tokens_per_sec – throughput from Pass 2 (short context, clean measure)
        total_seconds  – wall-clock time for both passes combined
    """
    # Patch module-level TARGET_TPS if caller overrides it
    global TARGET_TPS
    TARGET_TPS = target_tps

    entries = prepare_eval_data()
    shared = dict(
        port=port,
        model_name=model_name,
        entries=entries,
        memory_limit=memory_limit,
        temp=temp,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        repeat_penalty=repeat_penalty,
        maxtok=maxtok,
        system_prefix=system_prefix,
    )

    t_total = time.time()

    print("  [pass 1] context stress + token retrieval…")
    val_pass1, elapsed_p1 = _run_pass1(
        context_target_tokens=context_target_tokens, **shared  # type: ignore[arg-type]
    )
    print(f"  [pass 1] score={val_pass1:.4f}  elapsed={elapsed_p1:.1f}s")

    print("  [pass 2] short-context config extraction…")
    val_pass2, tokens_per_sec, elapsed_p2 = _run_pass2(**shared)  # type: ignore[arg-type]
    print(f"  [pass 2] score={val_pass2:.4f}  tok/s={tokens_per_sec:.1f}  elapsed={elapsed_p2:.1f}s")

    val_score = round(0.55 * val_pass1 + 0.45 * val_pass2, 6)
    total_seconds = round(time.time() - t_total, 3)
    return val_score, val_pass1, val_pass2, tokens_per_sec, total_seconds


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare fixed autoresearch data")
    parser.parse_args()
    check_models()
    # Force regeneration of fixture when prepare.py is run directly
    if FIXTURE_PATH.exists():
        FIXTURE_PATH.unlink()
    entries = prepare_eval_data()
    print(f"Models checked in {MODELS_DIR}")
    print(f"Prepared fixed memory fixture with {len(entries)} entries at {FIXTURE_PATH}")
    print("Autoresearch harness ready.")


if __name__ == "__main__":
    main()
