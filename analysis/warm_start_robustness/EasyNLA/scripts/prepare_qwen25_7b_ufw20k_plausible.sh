#!/usr/bin/env bash
# Prepare exact-key-matched plausible AV/AR SFT datasets. This script never
# extracts activations, reruns stage 1, copies/reshuffles RL data, or trains.
set -euo pipefail

if [[ "${1:-}" != "--execute" ]]; then
  cat <<'EOF'
DRY RUN ONLY (no API calls made).

This will:
  1. label the exact 49,495 AV + 49,516 AR keys retained by the implausible run;
  2. retry plausible formatting failures up to three times;
  3. build plausible stage-3 SFT data;
  4. put it in the exact implausible shuffled key order; and
  5. continuously off-cluster archive and SHA-verify every raw Batch record,
     ledger revision, final dataset, and match manifest.

Execute on Athena only after reviewing configs/datagen/qwen25_7b_ufw20k_plausible.yaml:
  bash scripts/prepare_qwen25_7b_ufw20k_plausible.sh --execute
EOF
  exit 0
fi

ROOT=/data/agastyas/nla_warmstart_robustness
CODE="$ROOT/easynla"
RUN="$ROOT/qwen25_7b_seed42"
BASE_DATA="$RUN/base_data"
IMPL="$RUN/conditions/implausible"
COND="$RUN/conditions/plausible"
CKPT="$RUN/checkpoints/plausible"
LOGS="$ROOT/logs"
PY=/data/agastyas/envs/easynla/bin/python
CONFIG=configs/datagen/qwen25_7b_ufw20k_plausible.yaml
HF_MODEL_REPO=${PLAUSIBLE_HF_MODEL_REPO:-agastyasridharan/nla-qwen25-7b-plausible-warmstart-20k}

cd "$CODE"
mkdir -p "$COND/splits" "$CKPT/archive_receipts" "$LOGS"

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

if [[ -z "${OPENAI_API_KEY_2:-}" ]]; then
  echo "[prepare-plausible] OPENAI_API_KEY_2 is unset; refusing API work" >&2
  exit 1
fi
if [[ -n "${OPENAI_API_KEY:-}" ]]; then
  echo "[prepare-plausible] note: OPENAI_API_KEY is ignored; provider is pinned to OPENAI_API_KEY_2"
fi

"$PY" - "$CONFIG" "$BASE_DATA" "$IMPL" "$COND" <<'PY'
import shutil
import sys
from pathlib import Path

import pyarrow.parquet as pq
import yaml

from nla.datagen.providers import OpenAIBatchProvider

config_path, base_data, impl, plausible = map(Path, sys.argv[1:])
cfg = yaml.safe_load(config_path.read_text())
p = cfg["stage2"]["provider_kwargs"]
expected = {
    "model": "gpt-5-mini",
    "max_tokens": 600,
    "reasoning_effort": "minimal",
    "verbosity": "low",
    "api_key_env": "OPENAI_API_KEY_2",
    "terminal_batch_retries": 0,
    "batch_input_per_million_usd": 0.25,
    "batch_cached_input_per_million_usd": 0.025,
    "batch_output_per_million_usd": 2.0,
    "spend_cap_usd": 87.23,
    "project_spend_cap_usd": 100.0,
    "reserved_downstream_spend_usd": 12.0,
}
for key, value in expected.items():
    assert p.get(key) == value, (key, p.get(key), value)
assert p["minimum_prior_cache_spend_usd"] >= 42.2395
assert p["spend_cap_usd"] + p["known_external_spend_usd"] + p["reserved_downstream_spend_usd"] < 100
assert p["spend_cap_usd"] - p["minimum_prior_cache_spend_usd"] < 45.0
assert cfg["stage2"]["chunk_size"] == 2000

paths = []
for split, count in (("av_sft", 49_495), ("ar_sft", 49_516)):
    raw = base_data / "splits" / f"{split}_raw.parquet"
    ref = impl / "splits" / f"{split}_explained.parquet"
    paths.extend((raw, Path(str(raw) + ".nla_meta.yaml"), ref, Path(str(ref) + ".nla_meta.yaml")))
    assert pq.ParquetFile(ref).metadata.num_rows == count, (ref, count)
rl = impl / "rl_shuf.parquet"
paths.extend((rl, Path(str(rl) + ".nla_meta.yaml")))
for path in paths:
    assert path.is_file() and path.stat().st_size > 0, path

free = shutil.disk_usage(plausible.parent).free
assert free >= 40 * 2**30, f"only {free / 2**30:.1f} GiB free; require >=40 GiB"
provider = OpenAIBatchProvider(**p)
try:
    actual = provider._actual_spend()
    assert actual + 1e-9 >= p["minimum_prior_cache_spend_usd"], (
        actual, p["minimum_prior_cache_spend_usd"]
    )
    ledger = provider._write_cumulative_ledger()
finally:
    provider.client.close()
assert ledger["maximum_project_spend_after_commitments_usd"] < 100, ledger
print(
    f"[prepare-plausible] static/input/prior-ledger preflight passed; "
    f"recorded cache spend=${actual:.6f}; free={free / 2**30:.1f} GiB"
)
PY

ARCHIVE_RUNTIME="$CKPT/archive_runtime"
API_ARCHIVE_RECEIPTS="$CKPT/archive_receipts/api_cache"
API_ARCHIVE_STATUS="$ARCHIVE_RUNTIME/api_cache_status.json"
API_ARCHIVE_STOP="$ARCHIVE_RUNTIME/API_CACHE_ARCHIVER_STOP"
mkdir -p "$ARCHIVE_RUNTIME" "$API_ARCHIVE_RECEIPTS"
if [[ -e "$API_ARCHIVE_STOP" ]]; then
  mv "$API_ARCHIVE_STOP" "$ARCHIVE_RUNTIME/API_CACHE_ARCHIVER_STOP.before_resume_$(date +%s)"
fi

echo "[prepare-plausible] preflighting private per-Batch off-cluster archive"
HF_HUB_OFFLINE=0 "$PY" scripts/archive_openai_batch_cache.py \
  --cache-dir "$RUN/api_cache" --repo-id "$HF_MODEL_REPO" \
  --path-prefix teacher/api_cache \
  --receipts-dir "$API_ARCHIVE_RECEIPTS" \
  --status-path "$API_ARCHIVE_STATUS" --stop-file "$API_ARCHIVE_STOP" \
  --preflight-only
HF_HUB_OFFLINE=0 "$PY" -u scripts/archive_openai_batch_cache.py \
  --cache-dir "$RUN/api_cache" --repo-id "$HF_MODEL_REPO" \
  --path-prefix teacher/api_cache \
  --receipts-dir "$API_ARCHIVE_RECEIPTS" \
  --status-path "$API_ARCHIVE_STATUS" --stop-file "$API_ARCHIVE_STOP" \
  --poll-seconds 15 \
  > "$LOGS/qwen25_teacher_plausible_api_archiver.log" 2>&1 &
ARCHIVER_PID=$!

echo "[prepare-plausible] generating exact-key teacher outputs (AV + AR concurrently)"
"$PY" -u scripts/datagen/generate_matched_teacher.py \
  --config "$CONFIG" --split av_sft \
  >> "$LOGS/qwen25_teacher_plausible_av.log" 2>&1 &
AV_PID=$!
"$PY" -u scripts/datagen/generate_matched_teacher.py \
  --config "$CONFIG" --split ar_sft \
  >> "$LOGS/qwen25_teacher_plausible_ar.log" 2>&1 &
AR_PID=$!

AV_STATUS=-1
AR_STATUS=-1
ARCHIVER_STATUS=0
while [[ "$AV_STATUS" -eq -1 || "$AR_STATUS" -eq -1 ]]; do
  if ! kill -0 "$ARCHIVER_PID" 2>/dev/null; then
    set +e
    wait "$ARCHIVER_PID"
    ARCHIVER_STATUS=$?
    set -e
    echo "[prepare-plausible] API archive watcher exited early ($ARCHIVER_STATUS); stopping local teacher workers" >&2
    kill -TERM "$AV_PID" "$AR_PID" 2>/dev/null || true
    set +e
    wait "$AV_PID" 2>/dev/null
    AV_STATUS=$?
    wait "$AR_PID" 2>/dev/null
    AR_STATUS=$?
    set -e
    break
  fi
  if [[ "$AV_STATUS" -eq -1 ]] && ! kill -0 "$AV_PID" 2>/dev/null; then
    set +e
    wait "$AV_PID"
    AV_STATUS=$?
    set -e
  fi
  if [[ "$AR_STATUS" -eq -1 ]] && ! kill -0 "$AR_PID" 2>/dev/null; then
    set +e
    wait "$AR_PID"
    AR_STATUS=$?
    set -e
  fi
  if [[ "$AV_STATUS" -eq -1 || "$AR_STATUS" -eq -1 ]]; then
    sleep 15
  fi
done

if kill -0 "$ARCHIVER_PID" 2>/dev/null; then
  touch "$API_ARCHIVE_STOP"
  set +e
  wait "$ARCHIVER_PID"
  ARCHIVER_STATUS=$?
  set -e
fi
if [[ "$ARCHIVER_STATUS" -ne 0 ]]; then
  echo "[prepare-plausible] API archive watcher failed: status=$ARCHIVER_STATUS; inspect $API_ARCHIVE_STATUS" >&2
  exit 1
fi
"$PY" - "$API_ARCHIVE_STATUS" <<'PY'
import json
import sys
from pathlib import Path

status = json.loads(Path(sys.argv[1]).read_text())
assert status["state"] == "complete", status
assert status["completed_local"] == status["verified_off_cluster"], status
assert not status["pending_content_keys"], status
assert not status["unarchived_incomplete"], status
assert status["latest_cumulative_ledger_sha256"], status
assert status["latest_cumulative_ledger_sha256"] == status["current_cumulative_ledger_sha256"], status
print(
    "[prepare-plausible] per-Batch archive healthy: "
    f"{status['verified_off_cluster']} raw Batch records verified off-cluster"
)
PY

if [[ "$AV_STATUS" -ne 0 || "$AR_STATUS" -ne 0 ]]; then
  echo "[prepare-plausible] teacher failed: AV=$AV_STATUS AR=$AR_STATUS" >&2
  echo "[prepare-plausible] all completed Batch records are independently archived; inspect logs and resume" >&2
  exit 1
fi

build_stage3() {
  local stage=$1
  local explained="$COND/splits/${stage}_explained.parquet"
  local final="$COND/${stage}.parquet"
  local final_meta="${final}.nla_meta.yaml"
  if [[ -s "$final" && -s "$final_meta" ]]; then
    echo "[prepare-plausible] reusing existing published stage3 $final"
    return
  fi
  if [[ -e "$final" ]]; then
    echo "[prepare-plausible] incomplete parquet publication at $final; refusing overwrite" >&2
    exit 1
  fi
  local staging
  staging=$(mktemp -d "$COND/.stage3_${stage}.XXXXXX")
  local staged="$staging/${stage}.parquet"
  "$PY" -u -m nla.datagen.stage3_build \
    --input "$explained" --stage "$stage" --output "$staged" \
    --keep-debug-metadata
  test -s "$staged"
  test -s "${staged}.nla_meta.yaml"
  # Sidecar first, parquet last. A crash can expose only a sidecar, never a
  # complete-looking parquet without its contract.
  mv -f "${staged}.nla_meta.yaml" "$final_meta"
  mv "$staged" "$final"
  rmdir "$staging"
}

build_stage3 av_sft
build_stage3 ar_sft

echo "[prepare-plausible] matching exact implausible SFT row order and vectors"
"$PY" -u scripts/datagen/reorder_to_reference.py \
  --input "$COND/av_sft.parquet" \
  --teacher "$COND/splits/av_sft_explained.parquet" \
  --stage av_sft \
  --reference "$IMPL/av_sft_shuf.parquet" \
  --output "$COND/av_sft_shuf.parquet" \
  --expected-rows 49495
"$PY" -u scripts/datagen/reorder_to_reference.py \
  --input "$COND/ar_sft.parquet" \
  --teacher "$COND/splits/ar_sft_explained.parquet" \
  --stage ar_sft \
  --reference "$IMPL/ar_sft_shuf.parquet" \
  --output "$COND/ar_sft_shuf.parquet" \
  --expected-rows 49516

LEDGER="$RUN/api_cache/cumulative_usage.json"
test -s "$LEDGER"
cp "$LEDGER" "$COND/teacher_cumulative_usage.json.tmp"
mv "$COND/teacher_cumulative_usage.json.tmp" "$COND/teacher_cumulative_usage.json"
sync "$COND"

"$PY" - "$COND" "$IMPL" "$LEDGER" <<'PY'
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

condition, implausible, ledger_path = map(Path, sys.argv[1:])
for split, expected in (("av_sft", 49_495), ("ar_sft", 49_516)):
    plausible = pq.read_table(condition / f"{split}_shuf.parquet", columns=["doc_id", "n_raw_tokens", "activation_layer"])
    reference = pq.read_table(implausible / f"{split}_shuf.parquet", columns=["doc_id", "n_raw_tokens", "activation_layer"])
    assert plausible.equals(reference), split
    assert plausible.num_rows == expected, (split, plausible.num_rows)
ledger = json.loads(ledger_path.read_text())
assert ledger["maximum_project_spend_after_commitments_usd"] < 100
assert ledger["reserved_downstream_spend_usd"] >= 12
print(json.dumps(ledger, indent=2, sort_keys=True))
print("[prepare-plausible] exact-key/order preparation complete; RL remains the original implausible rl_shuf.parquet")
PY

echo "[prepare-plausible] verifying API-cache completeness before private off-cluster archive"
"$PY" - "$RUN/api_cache" <<'PY'
import json
import sys
from pathlib import Path

cache = Path(sys.argv[1])
states = sorted(cache.glob("*/state.json"))
assert states, f"no Batch state files in {cache}"
incomplete = []
for state_path in states:
    state = json.loads(state_path.read_text())
    run = state_path.parent
    if (
        state.get("status") != "completed"
        or not (run / "input.jsonl").is_file()
        or not (run / "output.jsonl").is_file()
        or not (run / "usage.json").is_file()
    ):
        incomplete.append((run.name, state.get("status")))
assert not incomplete, (
    f"API cache contains incomplete/unreconciled Batch records: {incomplete[:10]}"
)
print(
    f"[prepare-plausible] {len(states)} Batch records have raw input/output "
    "and reconciled usage"
)
PY

sync "$COND" "$RUN/api_cache"
HF_HUB_OFFLINE=0 "$PY" scripts/archive_folder_verified.py \
  --source "$COND" --repo-id "$HF_MODEL_REPO" \
  --remote-dir teacher/plausible_condition \
  --receipt "$CKPT/archive_receipts/teacher_plausible_condition.json"
echo "[prepare-plausible] exact matched data and continuously archived raw API ledger independently SHA-256 verified"
