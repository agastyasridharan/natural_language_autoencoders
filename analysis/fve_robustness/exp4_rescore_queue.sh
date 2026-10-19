#!/usr/bin/env bash
set -euo pipefail

GPU=$1
shift
ROOT=/data/agastyas/nla_fve_ufw
cd "$ROOT"

for KEY in "$@"; do
  bash exp4_rescore_driver.sh "$KEY" "$GPU"
done

echo "EXP4_QUEUE_DONE gpu=$GPU $(date -Iseconds)"
