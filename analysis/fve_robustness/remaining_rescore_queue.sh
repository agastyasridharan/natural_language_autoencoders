#!/usr/bin/env bash
set -euo pipefail

GPU=$1
shift
ROOT=/data/agastyas/nla_fve_ufw
cd "$ROOT"
for KEY in "$@"; do
  bash remaining_rescore_driver.sh "$KEY" "$GPU"
done
echo "REMAINING_QUEUE_DONE gpu=$GPU $(date -Iseconds)"
