#!/usr/bin/env bash
set -euo pipefail

GPU=$1
shift
ROOT=/data/agastyas/nla_fve_ufw
LOG="$ROOT/logs/remaining_gpu${GPU}.log"

while nvidia-smi --id="$GPU" --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; do
  echo "$(date -Iseconds) waiting for gpu=$GPU" >> "$LOG"
  sleep 30
done

cd "$ROOT"
echo "$(date -Iseconds) launching gpu=$GPU keys=$*" >> "$LOG"
exec bash remaining_rescore_queue.sh "$GPU" "$@" >> "$LOG" 2>&1
