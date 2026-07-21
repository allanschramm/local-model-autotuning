#!/bin/bash
# discover-models.sh — Find candidate models for your hardware.
# Uso: bash scripts/discover-models.sh [profile]
# Profiles: coding (default) | writing | vision | ocr
# Output: hardware summary + whichllm recommendations + next-step prompt.

set -u

profile="${1:-coding}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=================================================================="
echo "  Model Discovery for profile: $profile"
echo "=================================================================="
echo ""

# Hardware detection
echo "--- Hardware ---"
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version \
        --format=csv,noheader 2>/dev/null | while IFS=, read -r name mem_total mem_free driver; do
        echo "  GPU: $name (${mem_total}MB total, ${mem_free}MB free, driver $driver)"
    done
elif [[ "$(uname)" == "Darwin" ]]; then
    echo "  GPU: Apple Silicon (Metal)"
    sysctl -n machdep.cpu.brand_string 2>/dev/null | head -1 | sed 's/^/  CPU: /'
else
    echo "  GPU: not detected (NVIDIA / Apple / AMD)"
fi

echo ""
echo "  CPU: $(lscpu 2>/dev/null | grep 'Model name' | head -1 | sed 's/.*: *//')"
echo "  RAM: $(free -h 2>/dev/null | grep Mem: | awk '{print $2 " total, " $3 " used, " $4 " available"}')"
echo "  Disk: $(df -h "$REPO_ROOT" 2>/dev/null | tail -1 | awk '{print $4 " free (" $5 " used)"}')"
echo ""

# Discovery tools (llmfit / whichllm)
if command -v llmfit &>/dev/null; then
    echo "--- llmfit recommendations ---"
    echo ""
    llmfit 2>/dev/null
    echo ""
fi

if command -v uvx &>/dev/null; then
    echo "--- whichllm recommendations ---"
    echo ""
    uvx whichllm@latest 2>/dev/null
elif ! command -v llmfit &>/dev/null; then
    echo "ERROR: Neither llmfit nor uvx (for whichllm) is installed."
    echo "Install llmfit (cargo install llmfit / brew install AlexsJones/llmfit/llmfit) or uvx (pip install uv)."
    exit 1
fi

echo ""
echo "=================================================================="
echo "  IMPORTANT: whichllm score is INTELLIGENCE-INDEX, not coding."
echo "  Always cross-reference with real benchmarks for coding profile."
echo "=================================================================="
echo ""
echo "Suggested next step (agent-driven):"
echo ""
echo "  Ask your coding agent:"
echo "    \"Cross-reference these candidates with SWE-bench Verified,"
echo "     Aider polyglot, and LiveCodeBench for the '$profile' profile."
echo "     Plot a Pareto frontier of tok/s vs coding quality."
echo "     Pick the Pareto-optimal point and configure autoresearch/core/config.py\""
echo ""
echo "Manual cross-reference sources:"
echo "  - SWE-bench Verified: https://www.swebench.com/"
echo "  - Aider polyglot:     https://aider.chat/docs/leaderboards/"
echo "  - LiveCodeBench:      https://livecodebench.github.io/leaderboard.html"
echo "  - Artificial Analysis: https://artificialanalysis.ai/"
echo ""
echo "After picking the model:"
echo "  1. Download GGUF:  hf download <repo-id> --local-dir models/<dir>"
echo "  2. Set in config:  bash scripts/run-autoloop.sh <filename.gguf>"
echo ""
