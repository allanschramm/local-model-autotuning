#!/usr/bin/env python3
"""
Unified AutoResearch Benchmark Runner (Tuning Surface).
Modify the constants in the Search Surface below, then run this file.
"""
import os
import sys
from pathlib import Path

# Import search surface defaults from config.py (single source of truth)
from autoresearch.core import config
from autoresearch.benchmarks.agentic_benchmarks import format_agentic_benchmarks

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

def parse_args():
    from autoresearch.runners import run
    args = run.parse_args()
    if getattr(args, "desc", None) is None:
        args.desc = "Tuner sweep run"
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.list_agentic_benchmarks:
        print(format_agentic_benchmarks())
        sys.exit(0)
    
    # Import the execution logic from run.py
    from autoresearch.runners import run
    
    run.handle_single_run(args)
