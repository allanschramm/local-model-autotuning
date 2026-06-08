#!/usr/bin/env python3
"""
Unified AutoResearch Benchmark Runner (Tuning Surface).
Modify the constants in the Search Surface below, then run this file.
"""
import os
import sys
from pathlib import Path

# Set default llama.cpp root directory
os.environ.setdefault("AUTORESEARCH_LLAMA_CPP_ROOT", "/home/shark/workspace/Nexus-System/llama.cpp")

# Import search surface defaults from config.py (single source of truth)
from autoresearch.core import config

MODEL = config.MODEL
CTX_SIZE = config.CTX_SIZE
KV_CACHE = config.KV_CACHE
KV_CACHE_K = config.KV_CACHE_K
KV_CACHE_V = config.KV_CACHE_V
BATCH_SIZE = config.BATCH_SIZE
UBATCH_SIZE = config.UBATCH_SIZE
THREADS = config.THREADS
THREADS_BATCH = config.THREADS_BATCH
FLASH_ATTN = config.FLASH_ATTN
SPEC_TYPE = config.SPEC_TYPE
SPEC_DRAFT_N_MAX = config.SPEC_DRAFT_N_MAX
NO_MMAP = config.NO_MMAP
JINJA = config.JINJA
REASONING_BUDGET = config.REASONING_BUDGET
REASONING_BUDGET_MESSAGE = config.REASONING_BUDGET_MESSAGE
REASONING = config.REASONING
CONT_BATCHING = config.CONT_BATCHING

# Generation options
TEMP = config.TEMP
TOP_P = config.TOP_P
MIN_P = config.MIN_P
TOP_K = config.TOP_K
REPEAT_PENALTY = config.REPEAT_PENALTY
PRESENCE_PENALTY = config.PRESENCE_PENALTY
FREQUENCY_PENALTY = config.FREQUENCY_PENALTY

# Benchmarks to run
INCLUDE_CODING = config.INCLUDE_CODING

# ---------------------------------------------------------------------------
# Plumbing & CLI Overrides
# ---------------------------------------------------------------------------
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Unified Tuning Runner")
    # Base params defaulting to the search surface constants
    parser.add_argument("--model", type=str, default=MODEL)
    parser.add_argument("--ctx-size", "-c", type=int, default=CTX_SIZE)
    parser.add_argument("--kv", type=str, default=KV_CACHE)
    parser.add_argument("--kv-k", "--cache-type-k", "-ctk", dest="kv_k", type=str, default=KV_CACHE_K)
    parser.add_argument("--kv-v", "--cache-type-v", "-ctv", dest="kv_v", type=str, default=KV_CACHE_V)
    parser.add_argument("--batch-size", "-b", type=int, default=BATCH_SIZE)
    parser.add_argument("--ubatch-size", "-ub", type=int, default=UBATCH_SIZE)
    parser.add_argument("--threads", "-t", type=int, default=THREADS)
    parser.add_argument("--threads-batch", type=int, default=THREADS_BATCH)
    parser.add_argument("--flash-attn", "-fa", nargs="?", const="on", default=FLASH_ATTN, choices=["on", "off", "auto"])
    parser.add_argument("--spec-type", type=str, default=SPEC_TYPE)
    parser.add_argument("--spec-draft-n-max", type=int, default=SPEC_DRAFT_N_MAX)
    parser.add_argument("--no-mmap", action="store_true", default=NO_MMAP)
    parser.add_argument("--jinja", action="store_true", default=JINJA)
    parser.add_argument("--reasoning-budget", type=int, default=REASONING_BUDGET)
    parser.add_argument("--reasoning-budget-message", type=str, default=REASONING_BUDGET_MESSAGE)
    parser.add_argument("--reasoning", type=str, choices=["on", "off", "auto"], default=REASONING)
    parser.add_argument("--cont-batching", action="store_true", default=CONT_BATCHING)
    
    parser.add_argument("--temp", type=float, default=TEMP)
    parser.add_argument("--top-p", type=float, default=TOP_P)
    parser.add_argument("--min-p", type=float, default=MIN_P)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--repeat-penalty", type=float, default=REPEAT_PENALTY)
    parser.add_argument("--presence-penalty", type=float, default=PRESENCE_PENALTY)
    parser.add_argument("--frequency-penalty", type=float, default=FREQUENCY_PENALTY)
    
    parser.add_argument("--include-coding", action="store_true", default=True, dest="include_coding")
    parser.add_argument("--include-nexus", action="store_true", default=getattr(config, "INCLUDE_NEXUS", False), dest="include_nexus")
    parser.add_argument("--include-claw", action="store_true", default=getattr(config, "INCLUDE_CLAW", False), dest="include_claw")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--context-tokens", type=int, default=8192)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--desc", type=str, default="Tuner sweep run")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Import the execution logic from run.py
    from autoresearch.runners import run
    
    run.handle_single_run(args)
