import sys
import evalplus.provider.base
from evalplus.codegen import main

# Monkeypatch default max_new_tokens to 2048 for thinking models
# DecoderBase.__init__(self, name, batch_size=1, temperature=0.8, max_new_tokens=768, ...)
# We can't easily change defaults for positional-or-keyword args this way.
# Let's wrap the __init__ method.

import os

original_init = evalplus.provider.base.DecoderBase.__init__

def patched_init(self, *args, **kwargs):
    if 'max_new_tokens' not in kwargs:
        # Check environment variable for max tokens, default to 2048
        max_tokens = int(os.getenv("EVALPLUS_MAX_TOKENS", 2048))
        kwargs['max_new_tokens'] = max_tokens
    original_init(self, *args, **kwargs)

evalplus.provider.base.DecoderBase.__init__ = patched_init

if __name__ == "__main__":
    # Remove this script from argv
    sys.argv.pop(0)
    # Run evalplus.codegen main
    from evalplus.codegen import codegen
    import evalplus.codegen
    
    # We need to replicate what 'python3 -m evalplus.codegen' does
    # which is calling main()
    evalplus.codegen.main()
