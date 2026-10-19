#!/usr/bin/env bash
set -euo pipefail
HOST=agastyas@athena01.ig32s.ruhr-uni-bochum.de
ROOT=/data/agastyas/nla_fve_ufw
HERE="$(cd "$(dirname "$0")" && pwd)"

ssh -o BatchMode=yes -o ConnectTimeout=15 "$HOST" \
  'nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader'
scp -q "$HERE/rescore.py" "$HOST:$ROOT/code/rescore.py"
scp -q "$HERE/exp4_rescore_driver.sh" "$HOST:$ROOT/exp4_rescore_driver.sh"
scp -q "$HERE/exp4_rescore_queue.sh" "$HOST:$ROOT/exp4_rescore_queue.sh"
for KEY in qwen7b gemma12b gemma27b; do
  for ARM in exp4a exp4b; do
    scp -q "$HERE/results/edited_reads_${ARM}_${KEY}.json" \
      "$HOST:$ROOT/out_${KEY}/edited_reads_${ARM}.json"
    scp -q "$HERE/results/records_${ARM}_${KEY}.json" \
      "$HOST:$ROOT/out_${KEY}/records_${ARM}_${KEY}.json"
  done
done

ssh "$HOST" "if pgrep -af '^bash exp4_rescore_queue.sh [01] ' >/dev/null; then \
    echo 'an Experiment 4 queue is already running'; exit 1; \
  fi; \
  cd $ROOT; \
  nohup bash exp4_rescore_queue.sh 0 qwen7b gemma27b \
    > logs/exp4_rescore_gpu0.log 2>&1 < /dev/null & echo launched:gpu0:qwen7b,gemma27b:pid=\$!; \
  nohup bash exp4_rescore_queue.sh 1 gemma12b \
    > logs/exp4_rescore_gpu1.log 2>&1 < /dev/null & echo launched:gpu1:gemma12b:pid=\$!"
