#!/usr/bin/env bash
set -euo pipefail
HOST=agastyas@athena01.ig32s.ruhr-uni-bochum.de
ROOT=/data/agastyas/nla_fve_ufw
HERE="$(cd "$(dirname "$0")" && pwd)"
for KEY in qwen7b gemma12b gemma27b; do
  for ARM in exp4a exp4b; do
    scp -q "$HOST:$ROOT/out_${KEY}/fve_${ARM}_${KEY}.json" "$HERE/results/"
    scp -q "$HOST:$ROOT/out_${KEY}/records_${ARM}_${KEY}.json" "$HERE/results/"
  done
done
