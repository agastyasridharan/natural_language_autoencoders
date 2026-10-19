#!/usr/bin/env bash
# Refresh the local warm-start dashboard from the active Athena run.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
RESULTS="$HERE/results"
REMOTE_HOST=agastyas@athena01.ig32s.ruhr-uni-bochum.de
REMOTE_METRICS=/data/agastyas/nla_warmstart_robustness/qwen25_7b_seed42/checkpoints/implausible/rl_vllm/metrics.jsonl
REMOTE_LOG=/data/agastyas/nla_warmstart_robustness/logs/qwen25_rl_vllm_implausible_resume540.log
REMOTE_VERIFIED_INDEX=/data/agastyas/nla_warmstart_robustness/qwen25_7b_seed42/checkpoints/implausible/continuation_snapshots/VERIFIED_INDEX.json
REMOTE_FINAL_RECEIPT=/data/agastyas/nla_warmstart_robustness/qwen25_7b_seed42/checkpoints/implausible/continuation_snapshots/step_000780/RECEIPT_PUBLISHED.json
REMOTE_HEALTH=/data/agastyas/nla_warmstart_robustness/ops/CONTINUATION_HEALTH.json
TRAIN_PATTERN='[n]la.train_rl_vllm.*--resume-from-lora.*/iter_000540'
TARGET_STEPS=780
INACTIVE_CONFIRMATIONS=3

training_state() {
  ssh -q -o BatchMode=yes -o ConnectTimeout=20 "$REMOTE_HOST" \
    "rows=\$(wc -l < '$REMOTE_METRICS' 2>/dev/null || printf 0); \
     if test \"\$rows\" -ge '$TARGET_STEPS' && test -s '$REMOTE_FINAL_RECEIPT'; then printf complete; \
     elif test \"\$rows\" -ge '$TARGET_STEPS'; then printf finalizing; \
     elif pgrep -f '$TRAIN_PATTERN' >/dev/null; then printf running; \
     else printf inactive; fi"
}

sync_once() {
  local metrics_tmp
  metrics_tmp="$(mktemp "$RESULTS/.rl_metrics.XXXXXX")"
  if ! scp -q "$REMOTE_HOST:$REMOTE_METRICS" "$metrics_tmp"; then
    rm -f "$metrics_tmp"
    return 1
  fi
  python - "$metrics_tmp" <<'PY'
import json
import sys
from pathlib import Path

p = Path(sys.argv[1])
rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
assert rows and [int(row["step"]) for row in rows] == list(range(len(rows)))
PY
  mv "$metrics_tmp" "$RESULTS/rl_metrics.jsonl"
  scp -q "$REMOTE_HOST:$REMOTE_LOG" "$RESULTS/rl_resume540.log.tmp" || true
  if test -f "$RESULTS/rl_resume540.log.tmp"; then
    mv "$RESULTS/rl_resume540.log.tmp" "$RESULTS/rl_resume540.log"
  fi
  scp -q "$REMOTE_HOST:$REMOTE_VERIFIED_INDEX" "$RESULTS/continuation_verified_index.json.tmp" || true
  if test -f "$RESULTS/continuation_verified_index.json.tmp"; then
    python - "$RESULTS/continuation_verified_index.json.tmp" <<'PY'
import json
import sys
from pathlib import Path

p = Path(sys.argv[1])
payload = json.loads(p.read_text())
records = payload.get("records", [])
steps = [int(row["checkpoint_step"]) for row in records]
assert steps == sorted(set(steps))
assert all(row.get("verification") == "size_and_SHA256_for_every_file" for row in records)
PY
    mv "$RESULTS/continuation_verified_index.json.tmp" "$RESULTS/continuation_verified_index.json"
  fi
  scp -q "$REMOTE_HOST:$REMOTE_HEALTH" "$RESULTS/continuation_health.json.tmp" || true
  if test -f "$RESULTS/continuation_health.json.tmp"; then
    python - "$RESULTS/continuation_health.json.tmp" <<'PY'
import json
import sys
from pathlib import Path

p = Path(sys.argv[1])
payload = json.loads(p.read_text())
assert payload.get("schema_version") == 1
assert isinstance(payload.get("alerts"), list)
PY
    mv "$RESULTS/continuation_health.json.tmp" "$RESULTS/continuation_health.json"
  fi
  python "$HERE/build_dashboard.py"
  python "$REPO/analysis/fve_robustness/build_experiments_dashboard.py"
  printf '%s synced %s rows\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$(wc -l < "$RESULTS/rl_metrics.jsonl")" \
    >> "$RESULTS/dashboard_sync.log"
}

mkdir -p "$RESULTS"
inactive_count=0
while true; do
  sync_once || true
  state="$(training_state 2>/dev/null || printf unreachable)"
  case "$state" in
    complete)
      break
      ;;
    inactive)
      inactive_count=$((inactive_count + 1))
      if test "$inactive_count" -ge "$INACTIVE_CONFIRMATIONS"; then
        printf '%s refresher stopped after %s confirmed inactive probes\n' \
          "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$inactive_count" >> "$RESULTS/dashboard_sync.log"
        break
      fi
      ;;
    running)
      inactive_count=0
      ;;
    finalizing)
      inactive_count=0
      printf '%s training complete; retaining refresher until final receipt is published\n' \
        "$(date '+%Y-%m-%dT%H:%M:%S%z')" >> "$RESULTS/dashboard_sync.log"
      ;;
    unreachable)
      printf '%s remote probe unreachable; retaining refresher\n' \
        "$(date '+%Y-%m-%dT%H:%M:%S%z')" >> "$RESULTS/dashboard_sync.log"
      ;;
    *)
      printf '%s unexpected remote state %q; retaining refresher\n' \
        "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$state" >> "$RESULTS/dashboard_sync.log"
      ;;
  esac
  sleep 60
done
sync_once || true
