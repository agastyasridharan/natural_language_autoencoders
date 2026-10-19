#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/agastyas/nla_cross_model_swap
export OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
mkdir -p "$ROOT/logs" "$ROOT/shards"
test -f "$ROOT/ENV_READY"
test -f "$ROOT/teacher_text_v2.parquet.json"
echo extracting > "$ROOT/phase.txt"
pids=()
for rank in {0..7}; do
    CUDA_VISIBLE_DEVICES=$rank "$ROOT/venv/bin/python" "$ROOT/extract_qwen.py" \
      --input "$ROOT/teacher_text_v2.parquet" --output "$ROOT/shards/rank$rank.parquet" \
      --checkpoint /data/hf_cache/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28 \
      --rank "$rank" --world-size 8 --batch-size 32 > "$ROOT/logs/extract_$rank.log" 2>&1 &
    pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
if [[ $failed != 0 ]]; then echo extraction_failed > "$ROOT/phase.txt"; exit 1; fi
echo extraction_ready > "$ROOT/phase.txt"
touch "$ROOT/EXTRACTION_READY"
