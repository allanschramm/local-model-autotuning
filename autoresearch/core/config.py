# config.py
# The ONLY changeable file for agent tweaks
# NOTE: CTX_SIZE is frozen at 131072. Min ctx floor = 100k. Never lower it.

MODEL = 'gemma-4-12B-it-qat-UD-Q4_K_XL.gguf'
CTX_SIZE = 131072
KV_CACHE = 'q4_0'
KV_CACHE_K = 'q4_0'
KV_CACHE_V = 'q4_0'
# Sweet spot for RTX 4060 8GB (llama-bench 2026-06-30): ub=256 pp1922 t/s tg49.8 t/s, ub=512 pp1940 t/s tg41.0 t/s
BATCH_SIZE = 1024
UBATCH_SIZE = 256
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

# Benchmarks to run
INCLUDE_CODING = True
INCLUDE_NEXUS = False
INCLUDE_CLAW = False
CODING_TASK_LIMIT = 10      # tasks per dataset for HE+ / MBPP+
LCB_TASK_LIMIT = 10         # LiveCodeBench v6 sample (contamination-free competitive prog)
BIGCODE_TASK_LIMIT = 10     # BigCodeBench Hard sample (library-call tasks)
EVALPLUS_STRICT = True
TRIAL_BUDGET = 300


