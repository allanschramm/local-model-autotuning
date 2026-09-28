"""Running Project — stable public API for model server lifecycle, inference, and search.

This is the contract boundary. Benchmarks consume this API and NEVER the reverse.
The running project MUST NOT import from autoresearch.benchmarks or autoresearch.runners.
"""

from autoresearch.core.hardware import detect_hardware_capabilities
from autoresearch.core.llama_client import (
    GenerationParams,
    LlamaClient,
)
from autoresearch.core.llama_runner import (
    ROOT_DIR,
    FlagDriftError,
    LlamaServerRunner,
    ServerIntent,
    engine_version_tag,
    estimate_vram_mb,
    load_mode_flag,
    preflight_vram,
    preflight_vram_for_intent,
    resolve_llama_bench,
    resolve_llama_server,
    resolve_vram_limit_mb,
)
from autoresearch.core.model_arch import is_dense_model, is_moe_model
from autoresearch.core.search import (
    Neighbor,
    SearchStrategy,
)
from autoresearch.core.sglang_runner import (
    SGLangServerRunner,
    run_sglang_bench_validation,
)
from autoresearch.core.state import (
    SearchState,
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
    "SearchState",
    # Utilities
    "estimate_vram_mb",
    "preflight_vram",
    "preflight_vram_for_intent",
    "resolve_vram_limit_mb",
    "is_dense_model",
    "is_moe_model",
    "detect_hardware_capabilities",
    "resolve_llama_server",
    "resolve_llama_bench",
    "engine_version_tag",
    "load_mode_flag",
    "FlagDriftError",
    "run_sglang_bench_validation",
    "ROOT_DIR",
]
