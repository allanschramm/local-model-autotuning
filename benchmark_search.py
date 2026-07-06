#!/usr/bin/env python3
"""
Unified AutoResearch Benchmark Runner.
Delegates to autoresearch.runners.run for all execution logic.
"""
import sys

from autoresearch.benchmarks import format_agentic_benchmarks, format_claw_tiers

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
    if getattr(args, "list_claw_tiers", False):
        print(format_claw_tiers())
        sys.exit(0)
    
    # Import the execution logic from run.py
    from autoresearch.runners import run
    
    run.handle_single_run(args)
