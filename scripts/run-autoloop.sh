#!/bin/bash
# run-autoloop.sh — Start the autoloop with all required env setup.
# Uso: bash scripts/run-autoloop.sh [model-filename.gguf]
# If no model given, uses whatever MODEL is currently set in autoresearch/core/config.py.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"

MODEL="${1:-}"

echo "=================================================================="
echo "  Starting autoloop"
echo "=================================================================="

# 1. Verify setup before launching
if [[ -x "$SCRIPT_DIR/setup-check.sh" ]]; then
    if ! bash "$SCRIPT_DIR/setup-check.sh"; then
        echo ""
        echo "ERROR: setup-check failed. Fix issues before running autoloop."
        exit 1
    fi
fi

# 2. Set llama.cpp path (env var > ./llama.cpp > sibling)
if [[ -z "${AUTORESEARCH_LLAMA_CPP_ROOT:-}" ]]; then
    if [[ -d "$REPO_ROOT/llama.cpp" ]]; then
        export AUTORESEARCH_LLAMA_CPP_ROOT="$REPO_ROOT/llama.cpp"
    elif [[ -d "$REPO_ROOT/../llama.cpp" ]]; then
        export AUTORESEARCH_LLAMA_CPP_ROOT="$REPO_ROOT/../llama.cpp"
    fi
fi
echo ""
if [[ -n "${AUTORESEARCH_LLAMA_CPP_ROOT:-}" ]]; then
    echo "AUTORESEARCH_LLAMA_CPP_ROOT=$AUTORESEARCH_LLAMA_CPP_ROOT"
else
    echo "WARNING: AUTORESEARCH_LLAMA_CPP_ROOT not set (will rely on PATH or in-repo llama.cpp)"
fi

# 3. Edit config.py MODEL if specified
if [[ -n "$MODEL" ]]; then
    CONFIG="$REPO_ROOT/autoresearch/core/config.py"
    if [[ ! -f "$CONFIG" ]]; then
        echo "ERROR: $CONFIG not found"
        exit 1
    fi
    if grep -q "^MODEL = " "$CONFIG"; then
        sed -i.bak "s/^MODEL = .*/MODEL = '$MODEL'/" "$CONFIG"
        echo "Updated MODEL = '$MODEL' in $CONFIG (backup: ${CONFIG}.bak)"
    else
        echo "WARNING: MODEL line not found in config.py"
    fi
fi

# 4. Detect VRAM budget
vram_budget=""
if command -v nvidia-smi &>/dev/null; then
    vram_total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
    if [[ -n "$vram_total" && "$vram_total" -gt 0 ]]; then
        # Leave 512MB headroom
        vram_budget=$((vram_total - 512))
        echo "VRAM: ${vram_total}MB total, ${vram_budget}MB budget (--vram-limit-mb)"
    fi
fi

# 5. Start autoloop
echo ""
echo "Starting autoloop (Ctrl+C to stop, state persists)..."
echo ""

if [[ -n "$vram_budget" ]]; then
    exec python3 autoloop.py --vram-limit-mb="$vram_budget"
else
    exec python3 autoloop.py
fi
