#!/usr/bin/env bash
# Durable end-to-end launcher after the two OpenAI teacher workers publish
# their AV/AR explained parquets. It intentionally trains only the 20k-document
# implausible condition; the public plausible NLA remains an external baseline.
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

AV_EXPL="$COND/splits/av_sft_explained.parquet"
AR_EXPL="$COND/splits/ar_sft_explained.parquet"

mkdir -p "$CKPT" "$LOGS"
cd "$CODE"

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

echo "[orchestrator] waiting for complete teacher parquets"
while ! { test -s "$AV_EXPL" && test -s "$AV_EXPL.nla_meta.yaml" &&
          test -s "$AR_EXPL" && test -s "$AR_EXPL.nla_meta.yaml"; }; do
  sleep 60
done

"$PY" - "$AV_EXPL" "$AR_EXPL" <<'PY'
import sys
import pyarrow.parquet as pq

expected = {sys.argv[1]: 49_983, sys.argv[2]: 49_964}
for path, requested in expected.items():
    retained = pq.ParquetFile(path).metadata.num_rows
    assert retained >= int(0.95 * requested), (path, retained, requested)
    print(f"[teacher] {path}: retained {retained}/{requested}")
PY

echo "[orchestrator] building and deterministically shuffling SFT/RL datasets"
"$PY" -u -m nla.datagen.run_pipeline \
  --config configs/datagen/qwen25_7b_ufw20k_implausible.yaml \
  --stages 3,shuffle \
  > "$LOGS/qwen25_stage3_shuffle_implausible.log" 2>&1

"$PY" - "$COND" <<'PY'
import sys
from pathlib import Path
import pyarrow.parquet as pq

root = Path(sys.argv[1])
for name in ("av_sft_shuf.parquet", "ar_sft_shuf.parquet", "rl_shuf.parquet"):
    path = root / name
    assert path.is_file() and path.with_suffix(path.suffix + ".nla_meta.yaml").is_file()
    print(f"[dataset] {name}: {pq.ParquetFile(path).metadata.num_rows} rows")
PY

echo "[orchestrator] launching one-epoch AV and AR SFT concurrently"
CUDA_VISIBLE_DEVICES=0 "$PY" -u -m nla.train_sft \
  --config configs/sft_qwen25_7b_ufw20k_implausible_av.yaml \
  > "$LOGS/qwen25_sft_implausible_av.log" 2>&1 &
AV_PID=$!
CUDA_VISIBLE_DEVICES=1 "$PY" -u -m nla.train_sft \
  --config configs/sft_qwen25_7b_ufw20k_implausible_ar.yaml \
  > "$LOGS/qwen25_sft_implausible_ar.log" 2>&1 &
AR_PID=$!

set +e
wait "$AV_PID"
AV_STATUS=$?
wait "$AR_PID"
AR_STATUS=$?
set -e
if [[ "$AV_STATUS" -ne 0 || "$AR_STATUS" -ne 0 ]]; then
  echo "[orchestrator] SFT failed: AV=$AV_STATUS AR=$AR_STATUS" >&2
  exit 1
fi

AV_ADAPTER="$(find "$CKPT/av_sft" -maxdepth 1 -type d -name 'iter_*' | sort | tail -n 1)"
AR_ADAPTER="$(find "$CKPT/ar_sft" -maxdepth 1 -type d -name 'iter_*' | sort | tail -n 1)"
test -n "$AV_ADAPTER"
test -n "$AR_ADAPTER"
echo "[orchestrator] final AV adapter: $AV_ADAPTER"
echo "[orchestrator] final AR adapter: $AR_ADAPTER"

echo "[orchestrator] merging LoRA warm starts into clean bf16 HF checkpoints"
CUDA_VISIBLE_DEVICES=2 "$PY" -u scripts/merge_lora_to_hf.py \
  --base-ckpt "$BASE" \
  --av-dir "$AV_ADAPTER" \
  --ar-dir "$AR_ADAPTER" \
  --av-out "$CKPT/merged/av_hf" \
  --ar-out "$CKPT/merged/ar_hf" \
  --mode both \
  > "$LOGS/qwen25_merge_implausible.log" 2>&1

CUDA_VISIBLE_DEVICES=2 "$PY" -u scripts/verify_hf_nla_pair.py \
  --av-ckpt "$CKPT/merged/av_hf" \
  --ar-ckpt "$CKPT/merged/ar_hf" \
  --parquet "$COND/rl_shuf.parquet" \
  --row 0 --max-new-tokens 150 \
  --output "$CKPT/merged/end_to_end_smoke.json" \
  > "$LOGS/qwen25_merged_pair_smoke.log" 2>&1

echo "[orchestrator] launching 780-step, four-H100 GRPO"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
unset PYTORCH_CUDA_ALLOC_CONF
"$VLLM_PY" -m torch.distributed.run --standalone --nproc_per_node=4 \
  -m nla.train_rl_vllm \
  --config configs/rl_qwen25_7b_ufw20k_implausible.yaml \
  --base-ckpt "$BASE" \
  --av-adapter "$AV_ADAPTER" \
  --av-ckpt "$CKPT/merged/av_hf" \
  --ar-ckpt "$CKPT/merged/ar_hf" \
  --rl-parquet "$COND/rl_shuf.parquet" \
  --sidecar "$COND/rl_shuf.parquet" \
  --save-dir "$CKPT/rl_vllm" \
  > "$LOGS/qwen25_rl_vllm_implausible.log" 2>&1

echo "[orchestrator] complete"
