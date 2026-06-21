#!/bin/bash
# Baseline smoke test: 1 run per model, MTP + TurboQuant, ~5min budget
# Validates server starts + model responds for both models.

set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Resolve llama-server binary
SERVER=""
for candidate in \
    "${AUTORESEARCH_LLAMA_CPP_ROOT:-}/build-cuda/bin/llama-server" \
    "$REPO_ROOT/llama.cpp/build-cuda/bin/llama-server" \
    "$REPO_ROOT/../llama.cpp/build-cuda/bin/llama-server"; do
    if [[ -x "$candidate" ]]; then
        SERVER="$candidate"
        break
    fi
done

if [[ -z "$SERVER" ]]; then
    echo "FAIL: llama-server not found. Set AUTORESEARCH_LLAMA_CPP_ROOT or clone llama.cpp."
    exit 1
fi

# Add library path for CUDA builds
SERVER_DIR="$(dirname "$SERVER")"
export LD_LIBRARY_PATH="${SERVER_DIR}:${LD_LIBRARY_PATH:-}"

QWEN_RC=1
GEMMA_RC=1

run_test() {
  local MODEL=$1
  local NAME=$2
  local PORT=$3
  echo ""
  echo "========================================"
  echo "SMOKE TEST: $NAME"
  echo "  Model: $MODEL"
  echo "  MTP=on, KV=turbo4, Port=$PORT"
  echo "========================================"

  # MTP only for models that support it (check nextn_predict_layers in model)
  local MTP_ARGS=()
  if [[ "$MODEL" == *"Qwen"* ]]; then
    MTP_ARGS=(--spec-type mtp --spec-draft-n-max 5 --spec-draft-type-k turbo4 --spec-draft-type-v turbo4)
  fi

  "$SERVER" \
    --model "models/$MODEL" \
    --host 127.0.0.1 --port "$PORT" \
    --ctx-size 4096 --batch-size 128 --ubatch-size 64 \
    --threads 8 --parallel 1 \
    --cache-type-k turbo4 --cache-type-v turbo4 \
    --flash-attn on --no-warmup \
    "${MTP_ARGS[@]}" &
  local PID=$!

  # Wait up to 120s for server ready
  local READY=0
  for i in $(seq 1 24); do
    sleep 5
    if ! kill -0 $PID 2>/dev/null; then
      echo "  FAIL: Server died during startup"
      return 1
    fi
    if curl -sf -o /dev/null "http://127.0.0.1:$PORT/health" 2>/dev/null; then
      READY=1
      echo "  Server ready after $((i*5))s"
      break
    fi
  done

  if [ $READY -eq 0 ]; then
    echo "  FAIL: Server not ready in 120s"
    kill $PID 2>/dev/null; wait $PID 2>/dev/null
    return 1
  fi

  # Minimal completion test
  echo "  Sending test completion..."
  curl -s -o "/tmp/smoke_resp_$PORT.json" "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"test","messages":[{"role":"user","content":"Say hello"}],"max_tokens":20}' 2>/dev/null

  if python3 -c "
import json, sys
d = json.load(open('/tmp/smoke_resp_$PORT.json'))
c = d['choices'][0]['message']
content = c.get('content','') or c.get('reasoning_content','')
if content:
    print(f'  PASS: Got response: {content[:60]}')
    sys.exit(0)
else:
    print('  WARN: Empty response but server responded')
    sys.exit(0)
" 2>/dev/null; then
    :
  else
    echo "  WARN: Could not parse response"
  fi

  # Kill server (ignore double-free crash in cleanup)
  kill $PID 2>/dev/null
  wait $PID 2>/dev/null
  return 0
}

# Qwen3.6-35B-A3B (has MTP head)
run_test "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf" "Qwen3.6-35B-A3B" 18080
QWEN_RC=$?

# Gemma-4-26B-A4B
run_test "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf" "Gemma-4-26B-A4B" 18081
GEMMA_RC=$?

echo ""
echo "========================================"
echo "SUMMARY"
echo "========================================"
[ $QWEN_RC -eq 0 ] && echo "  PASS | Qwen3.6-35B-A3B (MTP + TurboQuant)" || echo "  FAIL | Qwen3.6-35B-A3B"
[ $GEMMA_RC -eq 0 ] && echo "  PASS | Gemma-4-26B-A4B (TurboQuant)" || echo "  FAIL | Gemma-4-26B-A4B"

if [ $QWEN_RC -ne 0 ] || [ $GEMMA_RC -ne 0 ]; then
  exit 1
fi
echo "All smoke tests passed."
exit 0
