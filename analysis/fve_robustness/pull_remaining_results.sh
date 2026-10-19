#!/usr/bin/env bash
set -euo pipefail

HOST=agastyas@athena01.ig32s.ruhr-uni-bochum.de
ROOT=/data/agastyas/nla_fve_ufw
HERE="$(cd "$(dirname "$0")" && pwd)"

for KEY in qwen7b gemma12b gemma27b; do
  for ARM in exp5b exp5c exp6p exp6t; do
    scp -q "$HOST:$ROOT/out_${KEY}/fve_${ARM}_${KEY}.json" "$HERE/results/"
    scp -q "$HOST:$ROOT/out_${KEY}/records_${ARM}_${KEY}.json" "$HERE/results/"
    if [[ "$ARM" == exp5c ]]; then
      scp -q "$HOST:$ROOT/out_${KEY}/edited_reads_${ARM}.json" \
        "$HERE/results/edited_reads_${ARM}_${KEY}.json"
    fi
  done
done

echo "pulled Exp. 5b/5c and Exp. 6P/6T summaries + scored records"
