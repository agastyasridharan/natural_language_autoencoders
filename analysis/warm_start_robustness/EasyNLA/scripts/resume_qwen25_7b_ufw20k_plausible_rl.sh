#!/usr/bin/env bash
# Resume only from the newest self-consistent immutable plausible checkpoint.
set -euo pipefail

if [[ "${1:-}" != "--execute" ]]; then
  echo "DRY RUN ONLY. Execute on Athena: bash scripts/resume_qwen25_7b_ufw20k_plausible_rl.sh --execute"
  exit 0
fi

ROOT=/data/agastyas/nla_warmstart_robustness
CODE="$ROOT/easynla"
RUN="$ROOT/qwen25_7b_seed42"
COND="$RUN/conditions/plausible"
IMPL="$RUN/conditions/implausible"
CKPT="$RUN/checkpoints/plausible"
SAVE="$CKPT/rl_vllm"
LOGS="$ROOT/logs"
BASE=/data/agastyas/hf_cache/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28
PY=/data/agastyas/envs/easynla/bin/python
VLLM_PY=/data/agastyas/envs/vllm-lens/bin/python
HF_MODEL_REPO=${PLAUSIBLE_HF_MODEL_REPO:-agastyasridharan/nla-qwen25-7b-plausible-warmstart-20k}
RL_DATA="$IMPL/rl_shuf.parquet"

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

mapfile -t GPU_PIDS < <(
  nvidia-smi --query-compute-apps=pid --format=csv,noheader |
    sed '/^[[:space:]]*$/d' |
    sort -u
)
if (( ${#GPU_PIDS[@]} != 0 )); then
  echo "[resume-plausible] refusing launch; GPU compute PIDs: ${GPU_PIDS[*]}" >&2
  exit 1
fi

START_STEP=$("$VLLM_PY" - "$SAVE" <<'PY'
import json
import sys
from pathlib import Path

import torch

save = Path(sys.argv[1])
snapshots = sorted(
    path for path in (save / "history").glob("step_*")
    if path.is_dir() and (path / "COMPLETE.json").is_file()
)
assert snapshots, "no immutable snapshots to resume"
latest = snapshots[-1]
step = int(latest.name.removeprefix("step_"))
assert step < 780, f"run already has final immutable step {step}"
assert (latest / "actor" / "adapter_model.safetensors").is_file()
assert (latest / "actor" / "reference" / "adapter_model.safetensors").is_file()
assert int((latest / "critic" / "saved_at_step.txt").read_text()) == step
assert int((save / "critic_latest" / "saved_at_step.txt").read_text()) == step, (
    "live critic_latest is not the newest immutable step; this indicates an "
    "interrupted checkpoint transaction and requires recovery review"
)
snapshot_optim = torch.load(latest / "optim.pt", map_location="cpu", weights_only=True)
live_optim = torch.load(save / "optim_latest.pt", map_location="cpu", weights_only=True)
assert int(snapshot_optim["step"]) == step
assert int(live_optim["step"]) == step, (
    "live optim_latest is not the newest immutable step; refusing mixed-state resume"
)
rows = [json.loads(line) for line in (save / "metrics.jsonl").read_text().splitlines() if line]
assert rows and int(rows[-1]["step"]) == step - 1
for suffix in ("critic_latest.tmp", "critic_latest.old", "optim_latest.pt.tmp"):
    assert not (save / suffix).exists(), suffix
print(step)
PY
)

RESUME="$SAVE/history/step_$(printf '%06d' "$START_STEP")/actor"
LOG="$LOGS/qwen25_rl_vllm_plausible_resume${START_STEP}.log"
if [[ -e "$LOG" ]]; then
  echo "[resume-plausible] refusing existing resume log: $LOG" >&2
  exit 1
fi

export WANDB_RUN_ID="$(< "$SAVE/wandb_run_id.txt")"
export WANDB_RESUME=must
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export NLA_RUN_LOG_PATH="$LOG"
unset PYTORCH_CUDA_ALLOC_CONF

STOP_FILE="$SAVE/ARCHIVER_STOP"
if [[ -e "$STOP_FILE" ]]; then
  mv "$STOP_FILE" "$SAVE/ARCHIVER_STOP.before_resume_${START_STEP}"
fi
HF_HUB_OFFLINE=0 "$PY" -u scripts/archive_immutable_rl_history.py \
  --save-dir "$SAVE" --repo-id "$HF_MODEL_REPO" --path-prefix rl_history \
  --stop-step 780 --poll-seconds 30 --stop-file "$STOP_FILE" \
  > "$LOGS/qwen25_rl_plausible_archiver_resume${START_STEP}.log" 2>&1 &
ARCHIVER_PID=$!

echo "[resume-plausible] resuming immutable step $START_STEP -> 780"
"$VLLM_PY" -m torch.distributed.run --standalone --nproc_per_node=4 \
  -m nla.train_rl_vllm \
  --config configs/rl_qwen25_7b_ufw20k_plausible.yaml \
  --base-ckpt "$BASE" \
  --av-adapter "$CKPT/av_sft/iter_0000774" \
  --av-ckpt "$CKPT/merged/av_hf" \
  --ar-ckpt "$CKPT/merged/ar_hf" \
  --rl-parquet "$RL_DATA" \
  --sidecar "$RL_DATA" \
  --save-dir "$SAVE" \
  --resume-from-lora "$RESUME" \
  --start-step "$START_STEP" \
  >> "$LOG" 2>&1 &
RL_PID=$!
STATUS=-1
ARCHIVER_STATUS=-1
while [[ "$STATUS" -eq -1 ]]; do
  if [[ "$ARCHIVER_STATUS" -eq -1 ]] && ! kill -0 "$ARCHIVER_PID" 2>/dev/null; then
    set +e; wait "$ARCHIVER_PID"; ARCHIVER_STATUS=$?; set -e
    if [[ "$ARCHIVER_STATUS" -ne 0 ]]; then
      echo "[resume-plausible] archive watcher failed; stopping unprotected training" >&2
      kill -TERM "$RL_PID" 2>/dev/null || true
    fi
  fi
  if ! kill -0 "$RL_PID" 2>/dev/null; then
    set +e; wait "$RL_PID"; STATUS=$?; set -e
  else
    sleep 30
  fi
done
if [[ "$STATUS" -ne 0 || "$ARCHIVER_STATUS" -gt 0 ]]; then
  touch "$STOP_FILE"
  if [[ "$ARCHIVER_STATUS" -eq -1 ]]; then
    wait "$ARCHIVER_PID" || true
  fi
  echo "[resume-plausible] training/archive exited $STATUS/$ARCHIVER_STATUS; durable state retained" >&2
  [[ "$STATUS" -ne 0 ]] && exit "$STATUS"
  exit 1
fi
if [[ "$ARCHIVER_STATUS" -eq -1 ]]; then
  set +e; wait "$ARCHIVER_PID"; ARCHIVER_STATUS=$?; set -e
fi
if [[ "$ARCHIVER_STATUS" -ne 0 ]]; then
  echo "[resume-plausible] final archive watcher exited $ARCHIVER_STATUS" >&2
  exit 1
fi

"$VLLM_PY" - "$SAVE" <<'PY'
import json
import sys
from pathlib import Path

import torch

save = Path(sys.argv[1])
expected = list(range(20, 781, 20))
history = [int(path.name.removeprefix("step_")) for path in sorted((save / "history").glob("step_*"))]
receipts = [int(path.stem.removeprefix("step_")) for path in sorted((save / "archive_receipts").glob("step_*.json"))]
assert history == expected, history
assert receipts == expected, receipts
state = torch.load(save / "history" / "step_000780" / "optim.pt", map_location="cpu", weights_only=True)
assert int(state["step"]) == 780
rows = [json.loads(line) for line in (save / "history" / "step_000780" / "metrics.jsonl").read_text().splitlines() if line]
assert int(rows[-1]["step"]) == 779
print("[resume-plausible] final step 780 and all 39 off-cluster receipts verified")
PY
