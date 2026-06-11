import sys
import evalplus.provider.base
from evalplus.codegen import main

# Monkeypatch default max_new_tokens to 2048 for thinking models
# DecoderBase.__init__(self, name, batch_size=1, temperature=0.8, max_new_tokens=768, ...)
# We can't easily change defaults for positional-or-keyword args this way.
# Let's wrap the __init__ method.

import os
import time
import atexit
import json
from pathlib import Path

original_init = evalplus.provider.base.DecoderBase.__init__

def patched_init(self, *args, **kwargs):
    if 'max_new_tokens' not in kwargs:
        # Check environment variable for max tokens, default to 2048
        max_tokens = int(os.getenv("EVALPLUS_MAX_TOKENS", 2048))
        kwargs['max_new_tokens'] = max_tokens
    original_init(self, *args, **kwargs)

evalplus.provider.base.DecoderBase.__init__ = patched_init

total_completion_tokens = 0
total_generation_time = 0.0

try:
    import evalplus.provider.openai
    original_make_auto_request = evalplus.provider.openai.openai_request.make_auto_request

    def patched_make_auto_request(*args, **kwargs):
        global total_completion_tokens, total_generation_time
        t_start = time.time()
        ret = original_make_auto_request(*args, **kwargs)
        duration = time.time() - t_start
        total_generation_time += duration
        if ret and hasattr(ret, 'usage') and ret.usage:
            total_completion_tokens += getattr(ret.usage, 'completion_tokens', 0) or 0
        return ret

    evalplus.provider.openai.openai_request.make_auto_request = patched_make_auto_request
except Exception:
    pass

# Parse args to find root path and dataset
root_path = None
dataset = "unknown"
try:
    for i, arg in enumerate(sys.argv):
        if arg == "--root" and i + 1 < len(sys.argv):
            root_path = sys.argv[i+1]
        elif arg == "--dataset" and i + 1 < len(sys.argv):
            dataset = sys.argv[i+1]
except Exception:
    pass

def save_stats():
    if root_path:
        try:
            stats_file = Path(root_path) / f"stats_{dataset}.json"
            with open(stats_file, "w") as f:
                json.dump({
                    "total_tokens": total_completion_tokens,
                    "total_seconds": total_generation_time
                }, f)
        except Exception:
            pass

atexit.register(save_stats)

if __name__ == "__main__":
    # Remove this script from argv
    sys.argv.pop(0)
    # Run evalplus.codegen main
    from evalplus.codegen import codegen
    import evalplus.codegen
    
    # We need to replicate what 'python3 -m evalplus.codegen' does
    # which is calling main()
    evalplus.codegen.main()
