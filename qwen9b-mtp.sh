#!/usr/bin/env bash
# Quick Qwen3.5-9B-MTP server control
# Usage: ./qwen9b-mtp.sh up|down|status|logs
set -euo pipefail

MODEL="/mnt/d/LLM-Models/Qwen3.5-9B-MTP-Q4_K_M.gguf"
SERVER="/home/shark/workspace/Nexus-System/llama.cpp/build-cuda/bin/llama-server"
PORT="${PORT:-18080}"
PIDFILE="/tmp/qwen9b-mtp.pid"

CMD=(
  "$SERVER"
  --model "$MODEL"
  --host 127.0.0.1
  --port "$PORT"
  --ctx-size 32768
  --batch-size 512
  --ubatch-size 128
  --threads 8
  --threads-batch 8
  --parallel 1
  --n-gpu-layers 999
  --cache-type-k q4_0
  --cache-type-v q4_0
  --flash-attn on
  --spec-type draft-mtp
  --spec-draft-n-max 1
  --spec-draft-type-k q4_0
  --spec-draft-type-v q4_0
)

up() {
  if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "already running PID $(cat "$PIDFILE") on port $PORT"
    return
  fi
  nohup "${CMD[@]}" > "/tmp/qwen9b-mtp.log" 2>&1 &
  echo $! > "$PIDFILE"
  echo "started PID $! on port $PORT"
  sleep 2
  status
}

down() {
  if [ ! -f "$PIDFILE" ]; then
    echo "not running"
    return
  fi
  pid=$(cat "$PIDFILE")
  kill "$pid" 2>/dev/null || true
  rm -f "$PIDFILE"
  echo "stopped PID $pid"
}

status() {
  if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "running PID $(cat "$PIDFILE") on port $PORT"
    curl -sf "http://127.0.0.1:$PORT/health" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'  model: {d.get(\"model_path\",\"?\")}')" 2>/dev/null || echo "  health: unreachable"
  else
    echo "stopped"
  fi
}

logs() {
  tail -f /tmp/qwen9b-mtp.log
}

case "${1:-help}" in
  up|start)     up ;;
  down|stop)    down ;;
  status|st)    status ;;
  logs|log)     logs ;;
  *)            echo "usage: qwen9b-mtp.sh up|down|status|logs" ;;
esac
