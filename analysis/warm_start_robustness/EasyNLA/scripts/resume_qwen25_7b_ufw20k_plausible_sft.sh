#!/usr/bin/env bash
# Resume matched plausible AV/AR SFT from their newest exact-state checkpoints.
set -euo pipefail

if [[ "${1:-}" != "--execute" ]]; then
  echo "DRY RUN ONLY. Execute on Athena: bash scripts/resume_qwen25_7b_ufw20k_plausible_sft.sh --execute"
  exit 0
fi

ROOT=/data/agastyas/nla_warmstart_robustness
CODE="$ROOT/easynla"
RUN="$ROOT/qwen25_7b_seed42"
CKPT="$RUN/checkpoints/plausible"
LOGS="$ROOT/logs"
PY=/data/agastyas/envs/easynla/bin/python
HF_MODEL_REPO=${PLAUSIBLE_HF_MODEL_REPO:-agastyasridharan/nla-qwen25-7b-plausible-warmstart-20k}

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
  echo "[resume-sft] refusing launch; GPU compute PIDs: ${GPU_PIDS[*]}" >&2
  exit 1
fi

latest_checkpoint() {
  local save=$1
  local mode=$2
  "$PY" - "$save" "$mode" <<'PY'
import json
import sys
from pathlib import Path

import torch

save = Path(sys.argv[1])
mode = sys.argv[2]
checkpoints = sorted(
    path for path in save.glob("iter_[0-9][0-9][0-9][0-9][0-9][0-9][0-9]")
    if path.is_dir() and (path / "COMPLETE.json").is_file()
)
assert checkpoints, f"no complete SFT checkpoint in {save}"
latest = checkpoints[-1]
step = int(latest.name.removeprefix("iter_"))
marker = json.loads((latest / "COMPLETE.json").read_text())
assert marker["mode"] == mode and int(marker["completed_steps"]) == step
state = torch.load(latest / "trainer_state.pt", map_location="cpu", weights_only=False)
assert int(state["completed_steps"]) == step
assert int(state["num_steps"]) == 774
rows = [json.loads(line) for line in (save / "metrics.jsonl").read_text().splitlines() if line]
assert rows and int(rows[-1]["step"]) == step - 1
assert (save / "wandb_run_id.txt").is_file()
print(latest)
PY
}

AV_SAVE="$CKPT/av_sft"
AR_SAVE="$CKPT/ar_sft"
AV_RESUME=$(latest_checkpoint "$AV_SAVE" av)
AR_RESUME=$(latest_checkpoint "$AR_SAVE" ar)
AV_STEP=${AV_RESUME##*iter_}
AR_STEP=${AR_RESUME##*iter_}
AV_STEP=$((10#$AV_STEP))
AR_STEP=$((10#$AR_STEP))
for step in "$AV_STEP" "$AR_STEP"; do
  if (( step <= 0 || step > 774 )); then
    echo "[resume-sft] invalid checkpoint step $step" >&2
    exit 1
  fi
done

AV_STOP="$AV_SAVE/ARCHIVER_STOP"
AR_STOP="$AR_SAVE/ARCHIVER_STOP"
for stop in "$AV_STOP" "$AR_STOP"; do
  if [[ -e "$stop" ]]; then
    mv "$stop" "${stop}.before_resume_$(date +%s)"
  fi
done

HF_HUB_OFFLINE=0 "$PY" -u scripts/archive_sft_checkpoints.py \
  --save-dir "$AV_SAVE" --mode av --repo-id "$HF_MODEL_REPO" \
  --path-prefix sft/av --receipt-dir "$AV_SAVE/archive_receipts" \
  --status-path "$AV_SAVE/archive_status.json" --stop-file "$AV_STOP" \
  --stop-step 774 --save-every 20 --poll-seconds 15 \
  > "$LOGS/qwen25_sft_plausible_av_archiver_resume${AV_STEP}.log" 2>&1 &
AV_ARCHIVER_PID=$!
HF_HUB_OFFLINE=0 "$PY" -u scripts/archive_sft_checkpoints.py \
  --save-dir "$AR_SAVE" --mode ar --repo-id "$HF_MODEL_REPO" \
  --path-prefix sft/ar --receipt-dir "$AR_SAVE/archive_receipts" \
  --status-path "$AR_SAVE/archive_status.json" --stop-file "$AR_STOP" \
  --stop-step 774 --save-every 20 --poll-seconds 15 \
  > "$LOGS/qwen25_sft_plausible_ar_archiver_resume${AR_STEP}.log" 2>&1 &
AR_ARCHIVER_PID=$!

AV_PID=0
AR_PID=0
AV_STATUS=0
AR_STATUS=0
if (( AV_STEP < 774 )); then
  AV_STATUS=-1
  NLA_RUN_LOG_PATH="$LOGS/qwen25_sft_plausible_av.log" \
    CUDA_VISIBLE_DEVICES=0 "$PY" -u -m nla.train_sft \
    --config configs/sft_qwen25_7b_ufw20k_plausible_av.yaml \
    --resume-from "$AV_RESUME" \
    >> "$LOGS/qwen25_sft_plausible_av.log" 2>&1 &
  AV_PID=$!
fi
if (( AR_STEP < 774 )); then
  AR_STATUS=-1
  NLA_RUN_LOG_PATH="$LOGS/qwen25_sft_plausible_ar.log" \
    CUDA_VISIBLE_DEVICES=1 "$PY" -u -m nla.train_sft \
    --config configs/sft_qwen25_7b_ufw20k_plausible_ar.yaml \
    --resume-from "$AR_RESUME" \
    >> "$LOGS/qwen25_sft_plausible_ar.log" 2>&1 &
  AR_PID=$!
fi

AV_ARCHIVER_STATUS=-1
AR_ARCHIVER_STATUS=-1
while [[ "$AV_STATUS" -eq -1 || "$AR_STATUS" -eq -1 ]]; do
  if [[ "$AV_ARCHIVER_STATUS" -eq -1 ]] && ! kill -0 "$AV_ARCHIVER_PID" 2>/dev/null; then
    set +e; wait "$AV_ARCHIVER_PID"; AV_ARCHIVER_STATUS=$?; set -e
  fi
  if [[ "$AR_ARCHIVER_STATUS" -eq -1 ]] && ! kill -0 "$AR_ARCHIVER_PID" 2>/dev/null; then
    set +e; wait "$AR_ARCHIVER_PID"; AR_ARCHIVER_STATUS=$?; set -e
  fi
  if [[ "$AV_ARCHIVER_STATUS" -gt 0 || "$AR_ARCHIVER_STATUS" -gt 0 ]]; then
    echo "[resume-sft] archive watcher failed; stopping unprotected trainers" >&2
    (( AV_PID > 0 )) && kill -TERM "$AV_PID" 2>/dev/null || true
    (( AR_PID > 0 )) && kill -TERM "$AR_PID" 2>/dev/null || true
  fi
  if [[ "$AV_STATUS" -eq -1 ]] && ! kill -0 "$AV_PID" 2>/dev/null; then
    set +e; wait "$AV_PID"; AV_STATUS=$?; set -e
  fi
  if [[ "$AR_STATUS" -eq -1 ]] && ! kill -0 "$AR_PID" 2>/dev/null; then
    set +e; wait "$AR_PID"; AR_STATUS=$?; set -e
  fi
  if [[ "$AV_STATUS" -eq -1 || "$AR_STATUS" -eq -1 ]]; then
    sleep 15
  fi
done

touch "$AV_STOP" "$AR_STOP"
if [[ "$AV_ARCHIVER_STATUS" -eq -1 ]]; then
  set +e; wait "$AV_ARCHIVER_PID"; AV_ARCHIVER_STATUS=$?; set -e
fi
if [[ "$AR_ARCHIVER_STATUS" -eq -1 ]]; then
  set +e; wait "$AR_ARCHIVER_PID"; AR_ARCHIVER_STATUS=$?; set -e
fi
if [[ "$AV_STATUS" -ne 0 || "$AR_STATUS" -ne 0 || "$AV_ARCHIVER_STATUS" -ne 0 || "$AR_ARCHIVER_STATUS" -ne 0 ]]; then
  echo "[resume-sft] failed: AV=$AV_STATUS AR=$AR_STATUS archives=$AV_ARCHIVER_STATUS/$AR_ARCHIVER_STATUS" >&2
  exit 1
fi

"$PY" - "$AV_SAVE/archive_status.json" "$AR_SAVE/archive_status.json" <<'PY'
import json
import sys
from pathlib import Path

expected = list(range(20, 774, 20)) + [774]
for path in map(Path, sys.argv[1:]):
    status = json.loads(path.read_text())
    assert status["state"] == "complete", status
    assert status["local_complete_steps"] == expected, status
    assert status["verified_off_cluster_steps"] == expected, status
print("[resume-sft] both exact 774-step runs and all 78 off-cluster checkpoints verified")
PY
