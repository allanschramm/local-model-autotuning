# config.py
# The ONLY changeable file for agent tweaks

MODEL = 'g4-opt-it-Q4_K_M.gguf'
CTX_SIZE = 16384
KV_CACHE = 'q4_0'
KV_CACHE_K = 'q4_0'
KV_CACHE_V = 'q4_0'
BATCH_SIZE = 512
UBATCH_SIZE = 128
THREADS = 12
THREADS_BATCH = None
FLASH_ATTN = 'on'
SPEC_TYPE = None
SPEC_DRAFT_N_MAX = 1
NO_MMAP = False
JINJA = False
REASONING_BUDGET = None
REASONING_BUDGET_MESSAGE = None
REASONING = None
CONT_BATCHING = False

# Generation options
TEMP = 0.2
TOP_P = None
MIN_P = None
TOP_K = None
REPEAT_PENALTY = None
PRESENCE_PENALTY = None
FREQUENCY_PENALTY = None

# Benchmarks to run
INCLUDE_CODING = True
CODING_TASK_LIMIT = 30  # Tasks per dataset (HumanEval/MBPP). 0 = full dataset.
