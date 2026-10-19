#!/usr/bin/env bash
# Resume the 20k implausible run after SFT/merge completed and the original
# launcher stopped at the (now fixed) merged-pair verifier.
set -euo pipefail

ROOT=/data/agastyas/nla_warmstart_robustness
CODE="$ROOT/easynla"
RUN="$ROOT/qwen25_7b_seed42"
COND="$RUN/conditions/implausible"
CKPT="$RUN/checkpoints/implausible"
LOGS="$ROOT/logs"
BASE=/data/agastyas/hf_cache/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28
PY=/data/agastyas/envs/easynla/bin/python
VLLM_PY=/data/agastyas/envs/vllm-lens/bin/python

AV_ADAPTER="$CKPT/av_sft/iter_0000774"
AR_ADAPTER="$CKPT/ar_sft/iter_0000774"
AV_HF="$CKPT/merged/av_hf"
AR_HF="$CKPT/merged/ar_hf"

cd "$CODE"
export PYTHONPATH="$CODE${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/data/agastyas/hf_home
export HF_HUB_CACHE=/data/agastyas/hf_cache
export HF_TOKEN_PATH=/home/agastyas/.cache/huggingface/token
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export http_proxy=http://www-cache.rz.ruhr-uni-bochum.de:80
export https_proxy=http://www-cache.rz.ruhr-uni-bochum.de:80
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$https_proxy"
export no_proxy=.rub.de,.ruhr-uni-bochum.de,localhost,127.0.0.1,::1
export WANDB_API_KEY="$(< /home/agastyas/.config/nla/wandb_api_key)"

for path in "$AV_ADAPTER" "$AR_ADAPTER" "$AV_HF" "$AR_HF"; do
  test -d "$path" || { echo "[resume] missing required checkpoint: $path" >&2; exit 1; }
done
if test -d "$CKPT/rl_vllm" && find "$CKPT/rl_vllm" -mindepth 1 -print -quit | grep -q .; then
  echo "[resume] refusing to overwrite nonempty $CKPT/rl_vllm" >&2
  exit 1
fi

echo "[resume] running Karvonen-path merged-pair smoke"
CUDA_VISIBLE_DEVICES=2 "$PY" -u scripts/verify_hf_nla_pair.py \
  --av-ckpt "$AV_HF" \
  --ar-ckpt "$AR_HF" \
  --parquet "$COND/rl_shuf.parquet" \
  --row 0 --max-new-tokens 150 \
  --output "$CKPT/merged/end_to_end_smoke.json" \
  > "$LOGS/qwen25_merged_pair_smoke.log" 2>&1

sleep 2
mapfile -t GPU_PIDS < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sed '/^[[:space:]]*$/d' | sort -u)
if (( ${#GPU_PIDS[@]} != 0 )); then
  echo "[resume] refusing four-GPU RL; GPUs have compute PIDs: ${GPU_PIDS[*]}" >&2
  nvidia-smi >&2
  exit 1
fi

echo "[resume] merged-pair smoke passed; launching 780-step four-H100 GRPO"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
unset PYTORCH_CUDA_ALLOC_CONF
"$VLLM_PY" -m torch.distributed.run --standalone --nproc_per_node=4 \
  -m nla.train_rl_vllm \
  --config configs/rl_qwen25_7b_ufw20k_implausible.yaml \
  --base-ckpt "$BASE" \
  --av-adapter "$AV_ADAPTER" \
  --av-ckpt "$AV_HF" \
  --ar-ckpt "$AR_HF" \
  --rl-parquet "$COND/rl_shuf.parquet" \
  --sidecar "$COND/rl_shuf.parquet" \
  --save-dir "$CKPT/rl_vllm" \
  > "$LOGS/qwen25_rl_vllm_implausible.log" 2>&1

echo "[resume] complete"
