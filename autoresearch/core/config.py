# config.py
# The ONLY changeable file for agent tweaks
# NOTE: CTX_SIZE is frozen at 131072. Min ctx floor = 100k. Never lower it.

from typing import Any
from pathlib import Path

MODEL = 'ornith-1.0-9b-Q4_K_M.gguf'
CTX_SIZE = 131072
KV_CACHE = 'q4_0'
KV_CACHE_K = 'q4_0'
KV_CACHE_V = 'q4_0'
# Sweet spot for RTX 4060 8GB (llama-bench 2026-07-05): ub=128 pp1861 t/s tg43.4 t/s, ub=256 pp1768 t/s tg40.8 t/s
BATCH_SIZE = 1024
UBATCH_SIZE = 128
THREADS = 8
THREADS_BATCH = 8
FLASH_ATTN = 'on'
SPEC_TYPE = None
SPEC_DRAFT_N_MAX = 0
NO_MMAP = False
JINJA = False
REASONING_BUDGET = None
REASONING_BUDGET_MESSAGE = None
REASONING = None
CONT_BATCHING = True
N_CPU_MOE = 32

# Generation options (Unsloth-corrected for Qwen3.5 thinking mode)
TEMP = 0.4
TOP_P = 0.95
TOP_K = 20
MIN_P = 0.0
REPEAT_PENALTY = 1.05
PRESENCE_PENALTY = 0.0
FREQUENCY_PENALTY = None



# ── Config persistence ────────────────────────────────────────────────────


def load_config(params: list[str] | None = None) -> dict[str, Any]:
    """Hot-reload config.py and return current values as dict."""
    import importlib
    import sys

    mod = sys.modules[__name__]
    importlib.reload(mod)
    if params is not None:
        return {p: getattr(mod, p, None) for p in params}
    return {k: v for k, v in vars(mod).items() if k.isupper() and not k.startswith('_')}


def write_config(cfg: dict[str, Any], path: str | Path | None = None) -> None:
    """Persist config dict back to config.py."""
    from pathlib import Path

    if path is None:
        path = Path(__file__).resolve()
    path = Path(path)

    lines = [
        "# config.py",
        "# The ONLY changeable file for agent tweaks",
        "# NOTE: CTX_SIZE is frozen at 131072. Min ctx floor = 100k. Never lower it.",
        "",
    ]

    server_params = [
        "MODEL", "CTX_SIZE", "KV_CACHE", "KV_CACHE_K", "KV_CACHE_V",
        "BATCH_SIZE", "UBATCH_SIZE", "THREADS", "THREADS_BATCH",
        "FLASH_ATTN", "SPEC_TYPE", "SPEC_DRAFT_N_MAX", "NO_MMAP",
        "JINJA", "REASONING_BUDGET", "REASONING_BUDGET_MESSAGE",
        "REASONING", "CONT_BATCHING", "N_CPU_MOE",
    ]
    for p in server_params:
        lines.append(f"{p} = {repr(cfg.get(p))}")

    lines.append("")
    lines.append("# Generation options (Unsloth-corrected for Qwen3.5 thinking mode)")
    gen_params = [
        "TEMP", "TOP_P", "MIN_P", "TOP_K",
        "REPEAT_PENALTY", "PRESENCE_PENALTY", "FREQUENCY_PENALTY",
    ]
    for p in gen_params:
        lines.append(f"{p} = {repr(cfg.get(p))}")

    path.write_text("\n".join(lines), encoding="utf-8")


