#!/usr/bin/env bash
# Continue the verified 20k implausible warm-start run from its atomic step-540
# actor/critic/optimizer checkpoint through the configured final step (780).
set -euo pipefail

ROOT=/data/agastyas/nla_warmstart_robustness
CODE="$ROOT/easynla"
RUN="$ROOT/qwen25_7b_seed42"
COND="$RUN/conditions/implausible"
CKPT="$RUN/checkpoints/implausible"
SAVE="$CKPT/rl_vllm"
RESUME="$SAVE/iter_000540"
LOG="$ROOT/logs/qwen25_rl_vllm_implausible_resume540.log"
BASE=/data/agastyas/hf_cache/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28
PY=/data/agastyas/envs/vllm-lens/bin/python

cd "$CODE"
export PYTHONPATH="$CODE${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/data/agastyas/hf_home
export HF_HUB_CACHE=/data/agastyas/hf_cache
export HF_TOKEN_PATH=/home/agastyas/.cache/huggingface/token
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export http_proxy=http://www-cache.rz.ruhr-uni-bochum.de:80
export https_proxy="$http_proxy"
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$https_proxy"
export no_proxy=.rub.de,.ruhr-uni-bochum.de,localhost,127.0.0.1,::1
export WANDB_API_KEY="$(< /home/agastyas/.config/nla/wandb_api_key)"
export WANDB_RUN_ID=7jcur2ou
export WANDB_RESUME=must
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
unset PYTORCH_CUDA_ALLOC_CONF

for path in \
  "$RESUME/adapter_model.safetensors" \
  "$RESUME/reference/adapter_model.safetensors" \
  "$SAVE/critic_latest/value_head.safetensors" \
  "$SAVE/optim_latest.pt" \
  "$SAVE/metrics.jsonl" \
  "$CKPT/av_sft/iter_0000774" \
  "$CKPT/merged/av_hf" \
  "$CKPT/merged/ar_hf"; do
  test -e "$path" || { echo "[continue540] missing required state: $path" >&2; exit 1; }
done

test "$(< "$SAVE/critic_latest/saved_at_step.txt")" = 540 || {
  echo "[continue540] critic_latest is not step 540" >&2
  exit 1
}

"$PY" - "$SAVE" <<'PY'
from pathlib import Path
import json
import sys
import torch

save = Path(sys.argv[1])
state = torch.load(save / "optim_latest.pt", map_location="cpu", weights_only=True)
assert int(state["step"]) == 540, state.get("step")
rows = [json.loads(line) for line in (save / "metrics.jsonl").read_text().splitlines() if line]
assert [row["step"] for row in rows] == list(range(540))
for suffix in ("critic_latest.tmp", "critic_latest.old", "optim_latest.pt.tmp"):
    assert not (save / suffix).exists(), suffix
print("[continue540] atomic checkpoint, optimizer, and metrics preflight passed")
PY

mapfile -t GPU_PIDS < <(
  nvidia-smi --query-compute-apps=pid --format=csv,noheader |
    sed '/^[[:space:]]*$/d' |
    sort -u
)
if (( ${#GPU_PIDS[@]} != 0 )); then
  echo "[continue540] refusing launch; GPUs have compute PIDs: ${GPU_PIDS[*]}" >&2
  nvidia-smi >&2
  exit 1
fi

{
  echo "[continue540] launch $(date -Is)"
  echo "[continue540] resume=$RESUME start_step=540 final_step=780"
} >> "$LOG"

exec "$PY" -m torch.distributed.run --standalone --nproc_per_node=4 \
  -m nla.train_rl_vllm \
  --config configs/rl_qwen25_7b_ufw20k_implausible.yaml \
  --base-ckpt "$BASE" \
  --av-adapter "$CKPT/av_sft/iter_0000774" \
  --av-ckpt "$CKPT/merged/av_hf" \
  --ar-ckpt "$CKPT/merged/ar_hf" \
  --rl-parquet "$COND/rl_shuf.parquet" \
  --sidecar "$COND/rl_shuf.parquet" \
  --save-dir "$SAVE" \
  --resume-from-lora "$RESUME" \
  --start-step 540 \
  >> "$LOG" 2>&1
