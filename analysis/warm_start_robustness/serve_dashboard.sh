#!/usr/bin/env bash
# Keep the local dashboard HTTP server available if the Python process exits.
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
DASHBOARD_ROOT="$(cd "$HERE/.." && pwd)"
RESULTS="$HERE/results"
PORT="${NLA_DASHBOARD_PORT:-8766}"
LOG="$RESULTS/dashboard_server.log"

mkdir -p "$RESULTS"
while true; do
  if curl -fsS --max-time 2 \
      "http://127.0.0.1:${PORT}/warm_start_robustness/dashboard.html" \
      >/dev/null 2>&1; then
    sleep 5
    continue
  fi
  printf '%s starting dashboard server on 127.0.0.1:%s\n' \
    "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$PORT" >> "$LOG"
  python3 -m http.server "$PORT" --bind 127.0.0.1 --directory "$DASHBOARD_ROOT" >> "$LOG" 2>&1
  status=$?
  printf '%s dashboard server exited with status %s; restarting in 2s\n' \
    "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$status" >> "$LOG"
  sleep 2
done
