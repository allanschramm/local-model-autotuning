# config.py
# The ONLY changeable file for agent tweaks

MODEL = 'gemma-4-26B-A4B-it-UD-Q4_K_M.gguf'
CTX_SIZE = 8192
KV_CACHE = 'q4_0'
KV_CACHE_K = 'q4_0'
KV_CACHE_V = 'q4_0'
BATCH_SIZE = 512
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
CONT_BATCHING = False

# Generation options (Unsloth-corrected for Qwen3.5 thinking mode)
TEMP = 0.4
TOP_P = 0.95
TOP_K = 20
MIN_P = 0.0
REPEAT_PENALTY = 1.0
PRESENCE_PENALTY = 0.0
FREQUENCY_PENALTY = None

# Benchmarks to run
INCLUDE_CODING = True
INCLUDE_NEXUS = True
INCLUDE_CLAW = False
CODING_TASK_LIMIT = 30  # Tasks per dataset (HumanEval/MBPP). 0 = full dataset.
