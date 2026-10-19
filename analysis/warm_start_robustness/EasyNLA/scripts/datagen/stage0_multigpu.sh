#!/bin/bash
# Crash-resumable, document-sharded stage-0 extraction on one process per GPU.
set -euo pipefail

PYTHON=${PYTHON:-python}
NGPU=${NGPU:-$(nvidia-smi -L | wc -l | tr -d ' ')}
GPUS=${GPUS:-$(seq -s, 0 $(( NGPU - 1 )))}
CPF=${STAGE0_CHUNKS_PER_FILE:-8}
IFS=',' read -r -a gpu_arr <<< "$GPUS"
NGPU=${#gpu_arr[@]}

start="" length="" output=""
pass=()
while [ $# -gt 0 ]; do
  case "$1" in
    --corpus-start) start=$2; shift 2 ;;
    --corpus-length) length=$2; shift 2 ;;
    --output) output=$2; shift 2 ;;
    --chunks-per-file) CPF=$2; shift 2 ;;
    *) pass+=("$1"); shift ;;
  esac
done
: "${start:?--corpus-start missing}" "${length:?--corpus-length missing}" "${output:?--output missing}"
mkdir -p "$(dirname "$output")"

echo "[stage0_multigpu] $length docs from $start over $NGPU GPUs ($GPUS) -> $output"
base_len=$(( length / NGPU )); rem=$(( length % NGPU ))
s=$start; pids=(); shards=()
for i in $(seq 0 $(( NGPU - 1 ))); do
  len=$base_len; [ "$i" -lt "$rem" ] && len=$(( base_len + 1 ))
  [ "$len" -gt 0 ] || continue
  shard="${output%.parquet}_shard${i}.parquet"
  shards+=("$shard")
  echo "[shard $i] GPU ${gpu_arr[$i]} docs [$s, $(( s + len ))) log=${shard}.log"
  CUDA_VISIBLE_DEVICES=${gpu_arr[$i]} "$PYTHON" -m nla.datagen.stage0_extract \
    --corpus-start "$s" --corpus-length "$len" --output "$shard" \
    --chunks-per-file "$CPF" ${pass[@]+"${pass[@]}"} > "${shard}.log" 2>&1 &
  pids+=($!)
  s=$(( s + len ))
done

fail=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "[shard $i] FAILED; tail of ${shards[$i]}.log:" >&2
    tail -30 "${shards[$i]}.log" >&2 || true
    fail=1
  fi
done
[ "$fail" -eq 0 ] || exit 1

"$PYTHON" -m nla.datagen.merge_base --inputs "${shards[@]}" --output "$output"
echo "[stage0_multigpu] DONE: $output"
