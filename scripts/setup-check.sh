#!/bin/bash
# setup-check.sh — Verifica todos os pré-requisitos antes do autoloop rodar.
# Uso: bash scripts/setup-check.sh
# Exit 0 se tudo OK, não-zero se algum pré-requisito crítico falhou.

set -u

checks_failed=0
warnings=0

# Colors (if terminal supports)
if [[ -t 1 ]]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[0;33m'
    NC='\033[0m'
else
    RED=''; GREEN=''; YELLOW=''; NC=''
fi

ok()   { printf "${GREEN}OK${NC}    %s\n" "$1"; }
warn() { printf "${YELLOW}WARN${NC}  %s\n" "$1"; warnings=$((warnings+1)); }
fail() { printf "${RED}FAIL${NC}  %s\n" "$1"; checks_failed=$((checks_failed+1)); }

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=== Setup Check for $REPO_ROOT ==="
echo ""

# 1. Python 3.11+
echo "--- Python ---"
if command -v python3 &>/dev/null; then
    py_version=$(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
    py_major=$(echo "$py_version" | cut -d. -f1)
    py_minor=$(echo "$py_version" | cut -d. -f2)
    if [[ "$py_major" -ge 3 && "$py_minor" -ge 11 ]]; then
        ok "Python $py_version (>= 3.11)"
    else
        fail "Python $py_version (need >= 3.11)"
    fi
else
    fail "python3 not found"
fi

# 2. CUDA / NVIDIA driver
echo ""
echo "--- CUDA ---"
if command -v nvidia-smi &>/dev/null; then
    gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    vram_mb=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    ok "NVIDIA GPU: $gpu_name (${vram_mb} MB)"
elif [[ "$(uname)" == "Darwin" ]]; then
    ok "Apple Silicon detected (Metal backend)"
elif command -v rocm-smi &>/dev/null; then
    ok "AMD GPU detected (ROCm)"
else
    fail "No GPU detected (nvidia-smi/Metal/ROCm)"
fi

# 3. llama.cpp build
echo ""
echo "--- llama.cpp build ---"
if [[ -n "${AUTORESEARCH_LLAMA_CPP_ROOT:-}" ]]; then
    if [[ -x "$AUTORESEARCH_LLAMA_CPP_ROOT/build-cuda/bin/llama-server" ]] || \
       [[ -x "$AUTORESEARCH_LLAMA_CPP_ROOT/build/bin/llama-server" ]]; then
        ok "llama-server at $AUTORESEARCH_LLAMA_CPP_ROOT (env var)"
    else
        fail "AUTORESEARCH_LLAMA_CPP_ROOT set but llama-server not found"
    fi
elif [[ -x "$REPO_ROOT/llama.cpp/build-cuda/bin/llama-server" ]]; then
    ok "llama-server at $REPO_ROOT/llama.cpp/build-cuda/"
elif [[ -x "$REPO_ROOT/../llama.cpp/build-cuda/bin/llama-server" ]]; then
    ok "llama-server at $REPO_ROOT/../llama.cpp/build-cuda/ (sibling)"
else
    fail "llama-server not found. Clone llama.cpp (or fork) and build with CUDA."
fi

# 4. llama-server flags
echo ""
echo "--- llama.cpp flags ---"
LLAMA_BIN=""
for candidate in \
    "$AUTORESEARCH_LLAMA_CPP_ROOT/build-cuda/bin/llama-server" \
    "$REPO_ROOT/llama.cpp/build-cuda/bin/llama-server" \
    "$REPO_ROOT/../llama.cpp/build-cuda/bin/llama-server"; do
    if [[ -x "$candidate" ]]; then
        LLAMA_BIN="$candidate"
        break
    fi
done

if [[ -n "$LLAMA_BIN" ]]; then
    help_out=$("$LLAMA_BIN" --help 2>&1)
    if echo "$help_out" | grep -q "\-\-spec-type"; then
        ok "--spec-type supported (MTP/speculative decoding)"
    else
        warn "--spec-type not in --help (no MTP/speculative on this build)"
    fi
    if echo "$help_out" | grep -q "\-\-cache-type-k"; then
        ok "--cache-type-k supported (KV quant)"
    else
        warn "--cache-type-k not in --help"
    fi
    if echo "$help_out" | grep -q "\-\-n-cpu-moe\|n_cpu_moe\|ncmoe"; then
        ok "--n-cpu-moe supported (MoE expert streaming)"
    else
        warn "--n-cpu-moe not in --help (no MoE offload)"
    fi
fi

# 5. uvx (whichllm)
echo ""
echo "--- Discovery tools ---"
if command -v uvx &>/dev/null; then
    uvx_ver=$(uvx --version 2>&1)
    ok "uvx ($uvx_ver) - whichllm ready"
else
    fail "uvx not found (pip install uv)"
fi

# 6. huggingface CLI
if command -v hf &>/dev/null; then
    hf_ver=$(hf --version 2>&1 | head -1)
    ok "hf CLI ($hf_ver)"
else
    warn "hf CLI not on PATH (pip install huggingface_hub[cli])"
fi

# 7. requirements.txt deps
echo ""
echo "--- Python dependencies ---"
if [[ -f "$REPO_ROOT/requirements.txt" ]]; then
    if python3 -c "import yaml, requests, huggingface_hub" 2>/dev/null; then
        ok "Core Python deps installed"
    else
        warn "Some Python deps missing (pip install -r requirements.txt)"
    fi
else
    warn "requirements.txt not found"
fi

# 8. Models directory
echo ""
echo "--- Models ---"
if [[ -d "$REPO_ROOT/models" ]]; then
    gguf_count=$(find "$REPO_ROOT/models" -maxdepth 2 -name "*.gguf" 2>/dev/null | wc -l)
    if [[ "$gguf_count" -gt 0 ]]; then
        ok "$gguf_count GGUF file(s) in models/"
    else
        warn "No .gguf files in models/ (download first or run discover)"
    fi
else
    fail "models/ directory missing (mkdir models)"
fi

# 9. WSL memory (Linux only)
echo ""
echo "--- WSL memory (Linux only) ---"
if [[ "$(uname -r)" == *microsoft* ]] || [[ -f /proc/version ]] && grep -qi "microsoft" /proc/version; then
    if [[ -f /etc/wsl.conf ]]; then
        if grep -q "memory=" /etc/wsl.conf 2>/dev/null; then
            mem=$(grep "memory=" /etc/wsl.conf | head -1)
            ok "WSL memory: $mem"
        else
            warn "WSL memory not in /etc/wsl.conf (recommended for inference)"
        fi
    fi
fi

# Summary
echo ""
echo "=== Summary ==="
if [[ $checks_failed -eq 0 ]]; then
    printf "${GREEN}All critical checks passed${NC}"
    if [[ $warnings -gt 0 ]]; then
        printf " (${YELLOW}%d warnings${NC})" "$warnings"
    fi
    echo ""
    exit 0
else
    printf "${RED}%d critical check(s) failed${NC}" "$checks_failed"
    if [[ $warnings -gt 0 ]]; then
        printf ", ${YELLOW}%d warnings${NC}" "$warnings"
    fi
    echo ""
    echo ""
    echo "Fix the FAIL items above before running autoloop."
    exit 1
fi
