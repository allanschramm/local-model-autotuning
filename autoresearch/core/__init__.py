"""Running Project — stable public API for model server lifecycle, inference, and search.

This is the contract boundary. Benchmarks consume this API and NEVER the reverse.
The running project MUST NOT import from autoresearch.benchmarks or autoresearch.runners.
"""

from autoresearch.core.llama_runner import (
    LlamaServerRunner,
    ServerIntent,
    ROOT_DIR,
    estimate_vram_mb,
    resolve_llama_server,
    resolve_llama_bench,
)
from autoresearch.core.llama_client import (
    LlamaClient,
    GenerationParams,
)
from autoresearch.core.sglang_runner import (
    SGLangServerRunner,
    run_sglang_bench_validation,
)
from autoresearch.core.search import (
    SearchStrategy,
    Neighbor,
)

__all__ = [
    # Server lifecycle
    "LlamaServerRunner",
    "ServerIntent",
    "SGLangServerRunner",
    # Inference
    "LlamaClient",
    "GenerationParams",
    # Search
    "SearchStrategy",
    "Neighbor",
    # Utilities
    "estimate_vram_mb",
    "resolve_llama_server",
    "resolve_llama_bench",
    "run_sglang_bench_validation",
    "ROOT_DIR",
]
