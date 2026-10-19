#!/usr/bin/env bash
# Exp 7 — Step 7.1 pilot: deploy the resample harness + subsample to the cluster, run the 3 arms on a
# small doc subset, pull the raw resamples back, then cluster + metric + render the dashboard locally.
# Pre-authorized by "go with defaults, start"; still CHECK IN after (per the 7.1 gate).
#
# Requires a STABLE ssh to the cluster (currently blocked on VPN/agent — see the check below).
#
# Usage:  bash deploy_exp7_pilot.sh [KEY=qwen7b] [PILOT=10] [GPU=0]
set -euo pipefail

KEY="${1:-qwen7b}"
PILOT="${2:-10}"
GPU="${3:-auto}"                  # any GPU is fine (all 4 booked by the user); 'auto' = least-used
HOST="agastyas@athena01.ig32s.ruhr-uni-bochum.de"
ROOT="/data/agastyas/nla_fve_ufw"
HERE="$(cd "$(dirname "$0")" && pwd)"

echo "[0/6] connectivity check ..."
if ! ssh -o BatchMode=yes -o ConnectTimeout=12 "$HOST" 'true' 2>/dev/null; then
  echo "  ✗ cannot reach $HOST (VPN down or ssh key not loaded)."
  echo "    Fix from your side, e.g.:  ! ssh-add ~/.ssh/id_ed25519  &&  ! ssh $HOST hostname"
  exit 2
fi
echo "  ✓ reachable"

echo "[1/6] GPU occupancy (all GPUs) ..."
ssh "$HOST" "nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader" || true
if [ "$GPU" = "auto" ]; then
  GPU=$(ssh "$HOST" "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | tr -d ' ' | cut -d, -f1")
  echo "  -> auto-selected least-used GPU: $GPU"
fi

echo "[2/6] deploy harness + subsample -> $ROOT ..."
ssh "$HOST" "mkdir -p $ROOT/results"
scp -q "$HERE/fve_resample.py" "$HOST:$ROOT/fve_resample.py"
scp -q "$HERE/results/exp7_subsample_${KEY}.json" "$HOST:$ROOT/results/exp7_subsample_${KEY}.json"

echo "[3/6] run the 3 arms (pilot=$PILOT docs, GPU $GPU) ..."
ssh "$HOST" "cd $ROOT && export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  HF_HOME=/data/hf_cache HF_HUB_CACHE=/data/hf_cache \
  HUGGINGFACE_HUB_CACHE=/data/hf_cache TRANSFORMERS_CACHE=/data/hf_cache \
  HF_TOKEN_PATH=\$HOME/.cache/huggingface/token && \
  for arm in main floor shuffle; do \
    echo \"=== \$arm ===\"; \
    CUDA_VISIBLE_DEVICES=$GPU /data/agastyas/Miniconda3/bin/python fve_resample.py \
      $KEY --arm \$arm --pilot $PILOT --batch 48; \
  done"

echo "[4/6] pull raw resamples back -> $HERE/out_${KEY}/ ..."
mkdir -p "$HERE/out_${KEY}"
for arm in main floor shuffle; do
  scp -q "$HOST:$ROOT/out_${KEY}/resample_${arm}_pilot${PILOT}.json" \
    "$HERE/out_${KEY}/resample_${arm}.json"
done

echo "[5/6] cluster (meaning+surface, API) + metrics locally ..."
cd "$HERE"
for arm in main floor shuffle; do python3 exp7_cluster.py "$KEY" --arm "$arm"; done
python3 exp7_metrics.py "$KEY"

echo "[6/6] render the Exp 7 dashboard ..."
python3 exp7_dashboard.py --cards 40
echo "DONE — open exp7_resample.html ; then CHECK IN before the full 200-doc run."
