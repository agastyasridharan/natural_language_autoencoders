#!/usr/bin/env bash
set -euo pipefail
KEY=$1
GPU=$2
ROOT=/data/agastyas/nla_fve_ufw
export CUDA_VISIBLE_DEVICES="$GPU"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME=/data/hf_cache HF_HUB_CACHE=/data/hf_cache
export HUGGINGFACE_HUB_CACHE=/data/hf_cache TRANSFORMERS_CACHE=/data/hf_cache
export HF_TOKEN_PATH="$HOME/.cache/huggingface/token"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$ROOT"
for ARM in floor shuffle; do
  echo "=== $(date -Iseconds) key=$KEY arm=$ARM gpu=$GPU ==="
  /data/agastyas/Miniconda3/bin/python fve_resample.py "$KEY" --arm "$ARM" --batch 48
done
echo "EXP7_CONTROLS_DONE key=$KEY gpu=$GPU $(date -Iseconds)"
