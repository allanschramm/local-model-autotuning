#!/usr/bin/env bash
# Generic llama-server monitoring tmux session.
# Discovers any llama-server process, port, health, log, and surrounding hardware state.
# Layout (4 panes):
#   +----------------------+----------------+
#   |  server info (watch) |  GPU (top)     |
#   +----------------------+----------------+
#   |  log tail -F         |  HW (bottom)   |
#   |  (modelo pensando)   |                |
#   +----------------------+----------------+
# Usage: ./llama-monitor.sh up | down | attach | a
set -euo pipefail

SESSION="llama-monitor"

up() {
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "session '$SESSION' already exists (use 'attach')"
    return
  fi

  # Start with a clean detached session.
  tmux new-session -d -s "$SESSION" -x 240 -y 50

  # pane 0 = left (llama log live)
  # First split: vertical -> left | right
  tmux split-window -h -t "$SESSION.0"
  # Focus right and split horizontally.
  tmux select-pane -t "$SESSION.1"
  tmux split-window -v -t "$SESSION.1"
  # Now: 0 = left (llama log), 1 = top-right (GPU), 2 = bottom-right (HW)

  # --- Pane 0: live log of the active llama-server (LEFT, full height) ---
  # Strategy: find the actual llama-server PID, follow its real stdout/stderr
  # (whether redirected to a file or attached to a TTY). If no PID, pick the
  # most recently modified /tmp/*log matching llama/server.
  tmux send-keys -t "$SESSION.0" 'bash -c "
echo \"=== detecting active llama-server log ===\"
find_log() {
  # Try real fds of every llama-server process: stdout (1), stderr (2), stdin (0)
  for PID in \$(pgrep -f llama-server); do
    for FD in 1 2 0; do
      if [ -L \"/proc/\$PID/fd/\$FD\" ]; then
        TARGET=\$(readlink \"/proc/\$PID/fd/\$FD\" 2>/dev/null)
        # If it points to a regular file we can tail, use it
        case \"\$TARGET\" in
          /tmp/*|*/log/*) [ -f \"\$TARGET\" ] && { echo \"\$TARGET\"; return 0; } ;;
        esac
        # If it points to a TTY (pts/N), there is no log file — we have to
        # use the most recent /tmp/*llama* log instead.
      fi
    done
  done
  # Fallback: most recent /tmp file matching llama/server
  ls -t /tmp/*llama*.log /tmp/llama_server.log /tmp/llama-direct.log /tmp/qwen9b-mtp.log 2>/dev/null | head -1
}
while true; do
  LOG=\$(find_log)
  if [ -n \"\$LOG\" ] && [ -f \"\$LOG\" ]; then
    echo \"following \$LOG (size \$(stat -c%s \"\$LOG\" 2>/dev/null) bytes, mtime \$(stat -c%y \"\$LOG\" 2>/dev/null | cut -d. -f1))\"
    tail -F \"\$LOG\"
    exit 0
  fi
  # If PID exists but stdout is a TTY (e.g. you ran llama-server directly),
  # there is no log file to tail. Tell the user clearly.
  if pgrep -f llama-server > /dev/null; then
    echo \"llama-server is running but its stdout is a TTY (no log file).\"
    echo \"restart it with: qwen9b-mtp down  &&  qwen9b-mtp up\"
    sleep 5
  else
    echo \"(no llama-server running, no /tmp log found — waiting 3s)\"
    sleep 3
  fi
done
"' Enter

  # --- Pane 1: GPU (TOP-RIGHT) ---
  tmux send-keys -t "$SESSION.1" 'nvidia-smi --loop=1' Enter

  # --- Pane 2: HARDWARE (BOTTOM-RIGHT) ---
  tmux send-keys -t "$SESSION.2" 'watch -n 1 "echo === CPU ===; lscpu | grep -E \"Model name|CPU\\(s\\)\" | head -2; echo; echo === Load ===; uptime; cat /proc/loadavg; echo; echo === RAM ===; free -h; echo; echo === Disk ===; df -h / | head -2; echo; echo === Temps ===; (sensors 2>/dev/null) || (test -d /proc/thermal && find /proc/thermal -name temp -type f 2>/dev/null | head -3 | while read t; do printf \"  %s: %.1f C\n\" \"\$t\" \"\$(awk \"{print \\\$1/1000}\" < \"\$t\")\"; done) || echo \"(no sensor data)\""' Enter

  # Focus the live-log pane by default.
  tmux select-pane -t "$SESSION.0"
  echo "started session '$SESSION'"
  echo "  layout: [llama log live | GPU / HW ]"
  echo "  attach with: $0 attach"
}

down() {
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    tmux kill-session -t "$SESSION"
    echo "killed session '$SESSION'"
  else
    echo "no session '$SESSION'"
  fi
}

attach() {
  tmux a -t "$SESSION"
}

case "${1:-help}" in
  up|start)     up ;;
  down|stop)    down ;;
  attach|a)     attach ;;
  *)            echo "usage: $0 up|down|attach" ;;
esac
