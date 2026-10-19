#!/usr/bin/env bash
# Train/merge/RL the matched 20k plausible warm start. No command runs unless
# --execute is present. Every phase is fail-closed on stale/nonempty outputs.
set -euo pipefail

PHASE=all
EXECUTE=0
for arg in "$@"; do
  case "$arg" in
    --execute) EXECUTE=1 ;;
    --phase=sft|--phase=merge|--phase=rl|--phase=all) PHASE=${arg#--phase=} ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

if [[ "$EXECUTE" -ne 1 ]]; then
  cat <<'EOF'
DRY RUN ONLY (no GPU, API, Hugging Face, or filesystem mutations).

Exact Athena launch sequence after plausible data preparation:
  bash scripts/run_qwen25_7b_ufw20k_plausible.sh --phase=sft --execute
  bash scripts/run_qwen25_7b_ufw20k_plausible.sh --phase=merge --execute
  bash scripts/run_qwen25_7b_ufw20k_plausible.sh --phase=rl --execute

Or run the three training phases continuously:
  bash scripts/run_qwen25_7b_ufw20k_plausible.sh --phase=all --execute

The launcher refuses nonempty fresh-phase destinations, snapshots code/config
and exact input digests, archives all 39 checkpoints for each SFT, externally
verifies merged artifacts, and archives all 39 immutable 20-step RL
transactions through step 780.
EOF
  exit 0
fi

ROOT=/data/agastyas/nla_warmstart_robustness
CODE="$ROOT/easynla"
RUN="$ROOT/qwen25_7b_seed42"
COND="$RUN/conditions/plausible"
IMPL="$RUN/conditions/implausible"
CKPT="$RUN/checkpoints/plausible"
LOGS="$ROOT/logs"
BASE=/data/agastyas/hf_cache/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28
PY=/data/agastyas/envs/easynla/bin/python
VLLM_PY=/data/agastyas/envs/vllm-lens/bin/python
HF_MODEL_REPO=${PLAUSIBLE_HF_MODEL_REPO:-agastyasridharan/nla-qwen25-7b-plausible-warmstart-20k}
RL_DATA="$IMPL/rl_shuf.parquet"

cd "$CODE"
mkdir -p "$CKPT" "$LOGS"
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

require_empty_or_absent() {
  local path=$1
  if [[ -e "$path" && ! -d "$path" ]]; then
    echo "[plausible] fresh destination exists and is not a directory: $path" >&2
    exit 1
  fi
  if [[ -e "$path" ]] && find "$path" -mindepth 1 -print -quit 2>/dev/null | grep -q .; then
    echo "[plausible] refusing nonempty fresh destination: $path" >&2
    exit 1
  fi
}

require_idle_gpus() {
  mapfile -t GPU_PIDS < <(
    nvidia-smi --query-compute-apps=pid --format=csv,noheader |
      sed '/^[[:space:]]*$/d' |
      sort -u
  )
  if (( ${#GPU_PIDS[@]} != 0 )); then
    echo "[plausible] refusing GPU phase; compute PIDs present: ${GPU_PIDS[*]}" >&2
    nvidia-smi >&2
    exit 1
  fi
}

snapshot_provenance() {
  local dst="$CKPT/provenance"
  if [[ -s "$dst/code_snapshot.tar.gz" && -s "$dst/input_sha256.txt" ]]; then
    return
  fi
  mkdir -p "$dst/configs"
  git rev-parse HEAD > "$dst/easynla_git_commit.txt"
  git status --short > "$dst/easynla_git_status.txt"
  git diff --binary > "$dst/easynla_worktree.patch"
  cp \
    configs/datagen/qwen25_7b_ufw20k_plausible.yaml \
    configs/sft_qwen25_7b_ufw20k_plausible_av.yaml \
    configs/sft_qwen25_7b_ufw20k_plausible_ar.yaml \
    configs/rl_qwen25_7b_ufw20k_plausible.yaml \
    configs/prompts/warmstart_plausible.txt \
    "$dst/configs/"
  printf '%q ' "$0" "$@" > "$dst/launch_command.txt"
  printf '\n' >> "$dst/launch_command.txt"
  tar --exclude=.git --exclude='__pycache__' --exclude='*.pyc' \
    -czf "$dst/code_snapshot.tar.gz" .
  sha256sum \
    "$COND/av_sft_shuf.parquet" \
    "$COND/av_sft_shuf.parquet.nla_meta.yaml" \
    "$COND/ar_sft_shuf.parquet" \
    "$COND/ar_sft_shuf.parquet.nla_meta.yaml" \
    "$RL_DATA" \
    "$RL_DATA.nla_meta.yaml" \
    > "$dst/input_sha256.txt"
  sync "$dst"
}

preflight() {
  "$PY" - "$COND" "$IMPL" "$RL_DATA" "$BASE" "$CKPT" <<'PY'
import shutil
import sys
from pathlib import Path

import pyarrow.parquet as pq
import yaml

condition, implausible, rl_data, base, ckpt = map(Path, sys.argv[1:])
for path in (base, rl_data, Path(str(rl_data) + ".nla_meta.yaml")):
    assert path.exists(), path
for split, expected in (("av_sft", 49_495), ("ar_sft", 49_516)):
    plausible_path = condition / f"{split}_shuf.parquet"
    reference_path = implausible / f"{split}_shuf.parquet"
    for path in (
        plausible_path,
        Path(str(plausible_path) + ".nla_meta.yaml"),
        reference_path,
        Path(str(reference_path) + ".nla_meta.yaml"),
        Path(str(plausible_path) + ".match_manifest.json"),
    ):
        assert path.is_file() and path.stat().st_size > 0, path
    exact_columns = [
        "doc_id",
        "n_raw_tokens",
        "activation_layer",
        "detokenized_text_truncated",
        "activation_vector",
    ]
    p = pq.read_table(plausible_path, columns=exact_columns)
    r = pq.read_table(reference_path, columns=exact_columns)
    assert p.equals(r), f"{split}: plausible source/activations/order differ from implausible"
    assert p.num_rows == expected

rl_meta = yaml.safe_load(Path(str(rl_data) + ".nla_meta.yaml").read_text())
assert rl_meta["stage"] == "rl" and rl_meta["extraction"]["norm"] == "none"
free = shutil.disk_usage(ckpt.parent).free
# Historical critics/optimizer are physically retained every 20 steps. Allow
# ample headroom beyond the projected ~0.6 TB run footprint.
assert free >= 800 * 2**30, f"only {free / 2**30:.1f} GiB free; require >=800 GiB"
print(f"[plausible] matched-data/sidecar/disk preflight passed; free={free / 2**30:.1f} GiB")
PY

  HF_HUB_OFFLINE=0 "$PY" scripts/archive_immutable_rl_history.py \
    --save-dir "$CKPT/rl_vllm" \
    --repo-id "$HF_MODEL_REPO" \
    --path-prefix rl_history \
    --preflight-only
}

verify_sft_final() {
  "$PY" - "$CKPT" <<'PY'
import sys
from pathlib import Path
from safetensors import safe_open

root = Path(sys.argv[1])
checks = {
    "av": root / "av_sft" / "iter_0000774" / "adapter_model.safetensors",
    "ar": root / "ar_sft" / "iter_0000774" / "ar_lora_value_head.safetensors",
}
for name, path in checks.items():
    assert path.is_file() and path.stat().st_size > 0, path
    tensors = elements = 0
    with safe_open(path, framework="pt", device="cpu") as sf:
        for key in sf.keys():
            tensor = sf.get_tensor(key)
            assert tensor.isfinite().all(), (name, key)
            tensors += 1
            elements += tensor.numel()
    print(f"[plausible] verified {name} final: {tensors} finite tensors, {elements} elements")
for side in ("av_sft", "ar_sft"):
    assert sorted(path.name for path in (root / side).glob("iter_*"))[-1] == "iter_0000774"
PY
}

run_sft() {
  require_idle_gpus
  require_empty_or_absent "$CKPT/av_sft"
  require_empty_or_absent "$CKPT/ar_sft"
  local av_log="$LOGS/qwen25_sft_plausible_av.log"
  local ar_log="$LOGS/qwen25_sft_plausible_ar.log"
  for path in "$av_log" "$ar_log"; do
    if [[ -e "$path" ]]; then
      echo "[plausible] refusing to overwrite existing fresh SFT log: $path" >&2
      exit 1
    fi
  done
  mkdir -p \
    "$CKPT/av_sft/provenance" "$CKPT/ar_sft/provenance" \
    "$CKPT/av_sft/archive_receipts" "$CKPT/ar_sft/archive_receipts"
  cp -a "$CKPT/provenance/." "$CKPT/av_sft/provenance/"
  cp -a "$CKPT/provenance/." "$CKPT/ar_sft/provenance/"

  local av_stop="$CKPT/av_sft/ARCHIVER_STOP"
  local ar_stop="$CKPT/ar_sft/ARCHIVER_STOP"
  HF_HUB_OFFLINE=0 "$PY" scripts/archive_sft_checkpoints.py \
    --save-dir "$CKPT/av_sft" --mode av --repo-id "$HF_MODEL_REPO" \
    --path-prefix sft/av --receipt-dir "$CKPT/av_sft/archive_receipts" \
    --status-path "$CKPT/av_sft/archive_status.json" --stop-file "$av_stop" \
    --stop-step 774 --save-every 20 --preflight-only
  HF_HUB_OFFLINE=0 "$PY" scripts/archive_sft_checkpoints.py \
    --save-dir "$CKPT/ar_sft" --mode ar --repo-id "$HF_MODEL_REPO" \
    --path-prefix sft/ar --receipt-dir "$CKPT/ar_sft/archive_receipts" \
    --status-path "$CKPT/ar_sft/archive_status.json" --stop-file "$ar_stop" \
    --stop-step 774 --save-every 20 --preflight-only
  HF_HUB_OFFLINE=0 "$PY" -u scripts/archive_sft_checkpoints.py \
    --save-dir "$CKPT/av_sft" --mode av --repo-id "$HF_MODEL_REPO" \
    --path-prefix sft/av --receipt-dir "$CKPT/av_sft/archive_receipts" \
    --status-path "$CKPT/av_sft/archive_status.json" --stop-file "$av_stop" \
    --stop-step 774 --save-every 20 --poll-seconds 15 \
    > "$LOGS/qwen25_sft_plausible_av_archiver.log" 2>&1 &
  AV_ARCHIVER_PID=$!
  HF_HUB_OFFLINE=0 "$PY" -u scripts/archive_sft_checkpoints.py \
    --save-dir "$CKPT/ar_sft" --mode ar --repo-id "$HF_MODEL_REPO" \
    --path-prefix sft/ar --receipt-dir "$CKPT/ar_sft/archive_receipts" \
    --status-path "$CKPT/ar_sft/archive_status.json" --stop-file "$ar_stop" \
    --stop-step 774 --save-every 20 --poll-seconds 15 \
    > "$LOGS/qwen25_sft_plausible_ar_archiver.log" 2>&1 &
  AR_ARCHIVER_PID=$!

  echo "[plausible] launching matched 774-step AV and AR SFT"
  NLA_RUN_LOG_PATH="$av_log" CUDA_VISIBLE_DEVICES=0 "$PY" -u -m nla.train_sft \
    --config configs/sft_qwen25_7b_ufw20k_plausible_av.yaml \
    > "$av_log" 2>&1 &
  AV_PID=$!
  NLA_RUN_LOG_PATH="$ar_log" CUDA_VISIBLE_DEVICES=1 "$PY" -u -m nla.train_sft \
    --config configs/sft_qwen25_7b_ufw20k_plausible_ar.yaml \
    > "$ar_log" 2>&1 &
  AR_PID=$!

  AV_STATUS=-1
  AR_STATUS=-1
  AV_ARCHIVER_STATUS=-1
  AR_ARCHIVER_STATUS=-1
  WATCHER_FAILED=0
  while [[ "$AV_STATUS" -eq -1 || "$AR_STATUS" -eq -1 ]]; do
    if [[ "$AV_ARCHIVER_STATUS" -eq -1 ]] && ! kill -0 "$AV_ARCHIVER_PID" 2>/dev/null; then
      set +e; wait "$AV_ARCHIVER_PID"; AV_ARCHIVER_STATUS=$?; set -e
    fi
    if [[ "$AR_ARCHIVER_STATUS" -eq -1 ]] && ! kill -0 "$AR_ARCHIVER_PID" 2>/dev/null; then
      set +e; wait "$AR_ARCHIVER_PID"; AR_ARCHIVER_STATUS=$?; set -e
    fi
    if [[ "$AV_ARCHIVER_STATUS" -gt 0 || "$AR_ARCHIVER_STATUS" -gt 0 ]]; then
      WATCHER_FAILED=1
    fi
    if [[ "$WATCHER_FAILED" -ne 0 ]]; then
      echo "[plausible] SFT archive watcher failed; stopping trainers before more unprotected checkpoints" >&2
      kill -TERM "$AV_PID" "$AR_PID" 2>/dev/null || true
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

  touch "$av_stop" "$ar_stop"
  if [[ "$AV_ARCHIVER_STATUS" -eq -1 ]]; then
    set +e; wait "$AV_ARCHIVER_PID"; AV_ARCHIVER_STATUS=$?; set -e
  fi
  if [[ "$AR_ARCHIVER_STATUS" -eq -1 ]]; then
    set +e; wait "$AR_ARCHIVER_PID"; AR_ARCHIVER_STATUS=$?; set -e
  fi
  if [[ "$AV_ARCHIVER_STATUS" -ne 0 || "$AR_ARCHIVER_STATUS" -ne 0 ]]; then
    echo "[plausible] SFT archive failed: AV=$AV_ARCHIVER_STATUS AR=$AR_ARCHIVER_STATUS" >&2
    exit 1
  fi
  if [[ "$AV_STATUS" -ne 0 || "$AR_STATUS" -ne 0 || "$WATCHER_FAILED" -ne 0 ]]; then
    echo "[plausible] SFT failed: AV=$AV_STATUS AR=$AR_STATUS; every complete checkpoint was archived" >&2
    exit 1
  fi
  "$PY" - "$CKPT/av_sft/archive_status.json" "$CKPT/ar_sft/archive_status.json" <<'PY'
import json
import sys
from pathlib import Path

expected = list(range(20, 774, 20)) + [774]
for path in map(Path, sys.argv[1:]):
    status = json.loads(path.read_text())
    assert status["state"] == "complete", status
    assert status["local_complete_steps"] == expected, status
    assert status["verified_off_cluster_steps"] == expected, status
    assert not status["pending_steps"], status
print("[plausible] all 39 AV and all 39 AR SFT checkpoints verified off-cluster")
PY
  verify_sft_final
  sync "$CKPT/av_sft" "$CKPT/ar_sft"
  mkdir -p "$CKPT/archive_receipts"
  HF_HUB_OFFLINE=0 "$PY" scripts/archive_folder_verified.py \
    --source "$CKPT/av_sft" --repo-id "$HF_MODEL_REPO" \
    --remote-dir sft/av --receipt "$CKPT/archive_receipts/sft_av.json"
  HF_HUB_OFFLINE=0 "$PY" scripts/archive_folder_verified.py \
    --source "$CKPT/ar_sft" --repo-id "$HF_MODEL_REPO" \
    --remote-dir sft/ar --receipt "$CKPT/archive_receipts/sft_ar.json"
}

run_merge() {
  require_idle_gpus
  verify_sft_final
  local merge_log="$LOGS/qwen25_merge_plausible.log"
  local smoke_log="$LOGS/qwen25_merged_pair_smoke_plausible.log"
  for path in "$merge_log" "$smoke_log"; do
    if [[ -e "$path" ]]; then
      echo "[plausible] refusing to overwrite existing fresh merge log: $path" >&2
      exit 1
    fi
  done
  if [[ -e "$CKPT/merged" ]]; then
    echo "[plausible] refusing existing merge destination: $CKPT/merged" >&2
    exit 1
  fi
  local staging="$CKPT/.merged.tmp.$$"
  if [[ -e "$staging" ]]; then
    echo "[plausible] merge staging collision: $staging" >&2
    exit 1
  fi
  mkdir "$staging"
  CUDA_VISIBLE_DEVICES=2 "$PY" -u scripts/merge_lora_to_hf.py \
    --base-ckpt "$BASE" \
    --av-dir "$CKPT/av_sft/iter_0000774" \
    --ar-dir "$CKPT/ar_sft/iter_0000774" \
    --av-out "$staging/av_hf" \
    --ar-out "$staging/ar_hf" \
    --mode both \
    > "$merge_log" 2>&1
  for path in \
    "$staging/av_hf/model.safetensors.index.json" \
    "$staging/ar_hf/value_head.safetensors" \
    "$staging/av_hf/nla_meta.yaml" \
    "$staging/ar_hf/nla_meta.yaml"; do
    test -s "$path" || { echo "[plausible] merged artifact missing: $path" >&2; exit 1; }
  done
  mv "$staging" "$CKPT/merged"
  CUDA_VISIBLE_DEVICES=2 "$PY" -u scripts/verify_hf_nla_pair.py \
    --av-ckpt "$CKPT/merged/av_hf" \
    --ar-ckpt "$CKPT/merged/ar_hf" \
    --parquet "$RL_DATA" \
    --row 0 --max-new-tokens 150 \
    --output "$CKPT/merged/end_to_end_smoke.json" \
    > "$smoke_log" 2>&1
  sync "$CKPT/merged"
  HF_HUB_OFFLINE=0 "$PY" scripts/archive_folder_verified.py \
    --source "$CKPT/merged" --repo-id "$HF_MODEL_REPO" \
    --remote-dir merged --receipt "$CKPT/archive_receipts/merged.json"
}

run_rl() {
  require_idle_gpus
  for path in \
    "$CKPT/av_sft/iter_0000774" \
    "$CKPT/merged/av_hf" \
    "$CKPT/merged/ar_hf" \
    "$CKPT/merged/end_to_end_smoke.json"; do
    test -e "$path" || { echo "[plausible] missing RL prerequisite: $path" >&2; exit 1; }
  done
  local save="$CKPT/rl_vllm"
  require_empty_or_absent "$save"
  mkdir -p "$save/provenance"
  cp -a "$CKPT/provenance/." "$save/provenance/"
  local log="$LOGS/qwen25_rl_vllm_plausible.log"
  if [[ -e "$log" ]]; then
    echo "[plausible] refusing to overwrite existing fresh RL log: $log" >&2
    exit 1
  fi
  local wandb_id
  wandb_id=$("$VLLM_PY" -c 'import wandb; print(wandb.util.generate_id())')
  printf '%s\n' "$wandb_id" > "$save/wandb_run_id.txt.tmp"
  mv "$save/wandb_run_id.txt.tmp" "$save/wandb_run_id.txt"

  export WANDB_RUN_ID="$wandb_id"
  export WANDB_RESUME=allow
  export CUDA_VISIBLE_DEVICES=0,1,2,3
  export VLLM_ALLOW_INSECURE_SERIALIZATION=1
  export NLA_RUN_LOG_PATH="$log"
  unset PYTORCH_CUDA_ALLOC_CONF

  local stop_file="$save/ARCHIVER_STOP"
  HF_HUB_OFFLINE=0 "$PY" -u scripts/archive_immutable_rl_history.py \
    --save-dir "$save" \
    --repo-id "$HF_MODEL_REPO" \
    --path-prefix rl_history \
    --stop-step 780 \
    --poll-seconds 30 \
    --stop-file "$stop_file" \
    > "$LOGS/qwen25_rl_plausible_archiver.log" 2>&1 &
  ARCHIVER_PID=$!

  echo "[plausible] launching 780-step four-H100 GRPO with 20-step immutable history"
  "$VLLM_PY" -m torch.distributed.run --standalone --nproc_per_node=4 \
    -m nla.train_rl_vllm \
    --config configs/rl_qwen25_7b_ufw20k_plausible.yaml \
    --base-ckpt "$BASE" \
    --av-adapter "$CKPT/av_sft/iter_0000774" \
    --av-ckpt "$CKPT/merged/av_hf" \
    --ar-ckpt "$CKPT/merged/ar_hf" \
    --rl-parquet "$RL_DATA" \
    --sidecar "$RL_DATA" \
    --save-dir "$save" \
    >> "$log" 2>&1 &
  RL_PID=$!
  RL_STATUS=-1
  ARCHIVER_STATUS=-1
  while [[ "$RL_STATUS" -eq -1 ]]; do
    if [[ "$ARCHIVER_STATUS" -eq -1 ]] && ! kill -0 "$ARCHIVER_PID" 2>/dev/null; then
      set +e; wait "$ARCHIVER_PID"; ARCHIVER_STATUS=$?; set -e
      if [[ "$ARCHIVER_STATUS" -ne 0 ]]; then
        echo "[plausible] RL archive watcher failed; stopping unprotected training" >&2
        kill -TERM "$RL_PID" 2>/dev/null || true
      fi
    fi
    if ! kill -0 "$RL_PID" 2>/dev/null; then
      set +e; wait "$RL_PID"; RL_STATUS=$?; set -e
    else
      sleep 30
    fi
  done
  if [[ "$RL_STATUS" -ne 0 || "$ARCHIVER_STATUS" -gt 0 ]]; then
    touch "$stop_file"
    if [[ "$ARCHIVER_STATUS" -eq -1 ]]; then
      wait "$ARCHIVER_PID" || true
    fi
    echo "[plausible] RL/train archive exited $RL_STATUS/$ARCHIVER_STATUS; immutable local history retained" >&2
    [[ "$RL_STATUS" -ne 0 ]] && exit "$RL_STATUS"
    exit 1
  fi
  if [[ "$ARCHIVER_STATUS" -eq -1 ]]; then
    set +e; wait "$ARCHIVER_PID"; ARCHIVER_STATUS=$?; set -e
  fi
  if [[ "$ARCHIVER_STATUS" -ne 0 ]]; then
    echo "[plausible] RL completed but archive watcher failed: $ARCHIVER_STATUS" >&2
    exit 1
  fi

  "$VLLM_PY" - "$save" <<'PY'
import json
import sys
from pathlib import Path
import torch

save = Path(sys.argv[1])
expected = list(range(20, 781, 20))
history = [int(path.name.removeprefix("step_")) for path in sorted((save / "history").glob("step_*"))]
receipts = [int(path.stem.removeprefix("step_")) for path in sorted((save / "archive_receipts").glob("step_*.json"))]
assert history == expected, (history[:3], history[-3:], len(history))
assert receipts == expected, (receipts[:3], receipts[-3:], len(receipts))
state = torch.load(save / "history" / "step_000780" / "optim.pt", map_location="cpu", weights_only=True)
assert int(state["step"]) == 780
rows = [json.loads(line) for line in (save / "history" / "step_000780" / "metrics.jsonl").read_text().splitlines() if line]
assert rows[-1]["step"] == 779
for path in save.rglob("*"):
    assert not path.name.endswith((".tmp", ".partial", ".old")), path
print("[plausible] all 39 local + off-cluster checkpoint transactions verified through step 780")
PY
}

preflight
snapshot_provenance "$@"

case "$PHASE" in
  sft) run_sft ;;
  merge) run_merge ;;
  rl) run_rl ;;
  all)
    run_sft
    run_merge
    run_rl
    ;;
esac

echo "[plausible] phase $PHASE complete"
