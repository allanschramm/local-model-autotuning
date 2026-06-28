# config.py
# The ONLY changeable file for agent tweaks

MODEL = 'ornith-1.0-35b-Q4_K_M.gguf'
CTX_SIZE = 131072
KV_CACHE = 'q4_0'
KV_CACHE_K = 'q4_0'
KV_CACHE_V = 'q4_0'
BATCH_SIZE = 512
UBATCH_SIZE = 128
THREADS = 12
THREADS_BATCH = 12
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
REPEAT_PENALTY = 1.0
PRESENCE_PENALTY = 0.0
FREQUENCY_PENALTY = None

# Benchmarks to run
INCLUDE_CODING = True
INCLUDE_NEXUS = False
INCLUDE_CLAW = False
CODING_TASK_LIMIT = 10      # tasks per dataset for HE+ / MBPP+
LCB_TASK_LIMIT = 10         # LiveCodeBench v6 sample (contamination-free competitive prog)
BIGCODE_TASK_LIMIT = 10     # BigCodeBench Hard sample (library-call tasks)
EVALPLUS_STRICT = True      # use evalplus `test` field (strict mode) for HE+/MBPP+
TRIAL_BUDGET = 300          # 5 minutes budget max for a single run


