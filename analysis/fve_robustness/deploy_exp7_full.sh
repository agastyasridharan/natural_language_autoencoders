#!/usr/bin/env bash
set -euo pipefail
HOST=agastyas@athena01.ig32s.ruhr-uni-bochum.de
ROOT=/data/agastyas/nla_fve_ufw
HERE="$(cd "$(dirname "$0")" && pwd)"

ssh -o BatchMode=yes -o ConnectTimeout=15 "$HOST" \
  'nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader'
ssh "$HOST" "mkdir -p $ROOT/results $ROOT/logs"
scp -q "$HERE/fve_resample.py" "$HERE/exp7_full_driver.sh" "$HOST:$ROOT/"
for KEY in qwen7b gemma12b gemma27b; do
  scp -q "$HERE/results/exp7_subsample_${KEY}.json" "$HOST:$ROOT/results/"
done

KEYS=(qwen7b gemma12b gemma27b)
GPUS=(1 2 3)
for I in 0 1 2; do
  KEY=${KEYS[$I]}
  GPU=${GPUS[$I]}
  ssh "$HOST" "if pgrep -af '^bash exp7_full_driver.sh $KEY $GPU\$' >/dev/null; then \
      echo 'already running: $KEY'; \
    else \
      cd $ROOT; nohup bash exp7_full_driver.sh $KEY $GPU > logs/exp7_full_${KEY}.log 2>&1 < /dev/null & \
      echo launched:$KEY:gpu$GPU:pid=\$!; \
    fi"
done
