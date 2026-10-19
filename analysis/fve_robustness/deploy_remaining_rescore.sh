#!/usr/bin/env bash
set -euo pipefail

HOST=agastyas@athena01.ig32s.ruhr-uni-bochum.de
ROOT=/data/agastyas/nla_fve_ufw
HERE="$(cd "$(dirname "$0")" && pwd)"

ssh -o BatchMode=yes -o ConnectTimeout=15 "$HOST" \
  "mkdir -p $ROOT/logs; if pgrep -af '^bash (remaining_rescore_queue|wait_launch_remaining)\\.sh [01] ' >/dev/null; then echo 'remaining queue already active'; exit 1; fi"
scp -q "$HERE/edits.py" "$HERE/exp5c_rebuild_from_records.py" "$HERE/rescore.py" "$HOST:$ROOT/code/"
scp -q "$HERE/remaining_rescore_driver.sh" "$HERE/remaining_rescore_queue.sh" \
  "$HERE/wait_launch_remaining.sh" "$HOST:$ROOT/"

# Rebuild 5c with the corrected v2 parser: the read-token ECHO is frozen. Existing records already
# hold verified offsets, so this tokenizer-free rewrite avoids any model-cache dependency.
ssh "$HOST" "cd $ROOT/code; /data/agastyas/Miniconda3/bin/python \
  exp5c_rebuild_from_records.py qwen7b gemma12b gemma27b"

for KEY in qwen7b gemma12b gemma27b; do
  for ARM in exp6p exp6t; do
    scp -q "$HERE/results/edited_reads_${ARM}_${KEY}.json" \
      "$HOST:$ROOT/out_${KEY}/edited_reads_${ARM}.json"
    scp -q "$HERE/results/records_${ARM}_${KEY}.json" \
      "$HOST:$ROOT/out_${KEY}/records_${ARM}_${KEY}.json"
  done
done

# At most two queues/devices: GPU 0 handles Qwen then Gemma-27B; GPU 1 handles Gemma-12B.
ssh "$HOST" "cd $ROOT; \
  nohup bash wait_launch_remaining.sh 0 qwen7b gemma27b > logs/remaining_wait_gpu0.log 2>&1 < /dev/null & echo watcher:gpu0:pid=\$!; \
  nohup bash wait_launch_remaining.sh 1 gemma12b > logs/remaining_wait_gpu1.log 2>&1 < /dev/null & echo watcher:gpu1:pid=\$!"
