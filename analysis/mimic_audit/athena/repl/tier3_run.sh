#!/bin/bash
# Tier-3 band-widening re-run for the cycle-consistency audit (2026-07-23).
#
# Re-runs cycle_run.py with the f3/f4 selection bands widened P90 -> P95
# (calibration_tier3.json), which admits more of the already-generated decoy
# candidates so validity-fail claims (< min_k clean placebos) can reach a valid
# family. This is the GPU-bound lever the no-GPU replay could not do: the newly
# admitted candidates have no stored dstar/x_m, so they must be measured.
#
# Output goes to cycle_runs_tier3/ (never clobbers the committed P90 runs).
# tau_val / kappa_0 / k are inherited from the Tier-0 calibration; the decoy-vs-
# decoy FP budget is RE-AUDITED after the run (report.py) and kappa_0 re-fit if
# the wider families breach 5%.
#
# GPU policy: waits for a FULLY-IDLE H100 among {2,3} ONLY (never 0,1), per the
# standing reservation + the cluster no-sharing rule. cycle_run.py is resumable
# (skips claims whose json already exists), so a killed job resumes on re-launch.
#
# Usage:  tier3_run.sh <key>     key in {qwen7b, gemma12b, gemma27b, llama70b}
set -u
KEY=${1:?usage: tier3_run.sh <key>}
NLA=/data/agastyas/nla_mimic
R=$NLA/code/repl
PY=/data/agastyas/Miniconda3/bin/python
export HF_HOME=/data/hf_cache HF_HUB_CACHE=/data/hf_cache
export HF_TOKEN_PATH=$HOME/.cache/huggingface/token   # avoid root-owned /data/hf_cache/token (perm-denied)
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
ALLOWED="2 3"          # GPUs 2,3 ONLY - never 0,1
NEED=92000             # MiB free => H100 essentially idle (no-sharing policy)

# RUNS = dir holding <run>/records.jsonl (the AV-claims+activation input). qwen7b and
# gemma12b live under claim_ablation_runs/; gemma27b and llama70b under code/runs/.
case "$KEY" in
  qwen7b)   RUN=qwen7b-cp6-fineweb-atoms;  GOLD=$NLA/out_corpus_qwen7b;          TAG=qwen7b-d4g24-full; SPLIT=0; RUNS=$NLA/claim_ablation_runs ;;
  gemma12b) RUN=gemma12b-cp7-fineweb-atoms; GOLD=$NLA/out_corpus_gemma12b_fineweb; TAG=gemma12b-d4g24;   SPLIT=0; RUNS=$NLA/claim_ablation_runs ;;
  gemma27b) RUN=gemma27b-g4-fineweb-atoms;  GOLD=$NLA/out_corpus_gemma27b_fineweb; TAG=gemma27b-d4g24;   SPLIT=0; RUNS=$NLA/code/runs ;;
  llama70b) RUN=llama70b-l4-fineweb-atoms;  GOLD=$NLA/out_corpus_llama70b_fineweb; TAG=llama70b-d4g24;   SPLIT=1; RUNS=$NLA/code/runs ;;
  *) echo "unknown key $KEY"; exit 2 ;;
esac
CALIB=$NLA/calib_d4g24/$RUN/calibration_tier3.json
OUT=$NLA/cycle_runs_tier3/$TAG
MARK=$NLA/cycle_runs_tier3/${KEY}.marker
mkdir -p "$NLA/cycle_runs_tier3"
[ -f "$CALIB" ] || { echo "MISSING $CALIB (create the P95-band calibration first)"; echo "FAIL_NO_CALIB" > "$MARK"; exit 2; }

need_n=$([ "$SPLIT" = "1" ] && echo 2 || echo 1)
echo "$(date '+%F %T') [tier3:$KEY] waiter start: need $need_n fully-idle GPU(s) among {2,3}"
while true; do
  freelist=""
  for g in $ALLOWED; do
    f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$g" 2>/dev/null | tr -d ' ')
    [ "${f:-0}" -ge "$NEED" ] && freelist="$freelist $g"
  done
  nfree=$(echo $freelist | wc -w)
  [ "$nfree" -ge "$need_n" ] && break
  echo "$(date '+%F %T') [tier3:$KEY] idle{2,3}=$nfree (need $need_n); sleep 120"
  sleep 120
done
if [ "$SPLIT" = "1" ]; then
  export CUDA_VISIBLE_DEVICES=$(echo $freelist | awk '{print $1","$2}')
else
  export CUDA_VISIBLE_DEVICES=$(echo $freelist | awk '{print $1}')
fi
echo "$(date '+%F %T') [tier3:$KEY] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES; launching"

cd "$R"
AR_FLAG=""
if [ "$SPLIT" = "1" ]; then
  # decoy_directions are cached over TEXT-passing candidates (f1/f2), computed
  # BEFORE the f3/f4 band selection that cycle_run.py applies downstream. The
  # cache is therefore band-INDEPENDENT: the existing ar_cache_d4g24 already holds
  # a direction for every P95 survivor (P95 survivors are a subset of the text-
  # passing set, a superset of P90 survivors). Reuse it instead of re-running the
  # 70B AR precompute; the per-claim alignment guard on load (cand-order / z_inner
  # / prompt-version) fails loudly on any mismatch, so reuse is safe.
  AR_CACHE=$NLA/ar_cache_d4g24
  [ -f "$AR_CACHE/$RUN/status.json" ] || { echo "FAIL_NO_ARCACHE $AR_CACHE/$RUN" > "$MARK"; exit 1; }
  AR_FLAG="--ar-cache $AR_CACHE"
fi

$PY cycle_run.py "$KEY" --run "$RUN" --runs-dir "$RUNS" \
    --gold-dir "$GOLD" --decoys-dir "$NLA/decoys_d4g24" \
    --calib "$CALIB" --out "$OUT" --prompt-version d4 $AR_FLAG \
  && $PY report.py --run-dir "$OUT/$RUN" --calib "$CALIB" \
      --calib-claims-dir "$NLA/calib_d4g24/$RUN/claims" \
  && echo "DONE $(date '+%F %T')" > "$MARK" \
  || echo "FAIL $(date '+%F %T')" > "$MARK"
echo "$(date '+%F %T') [tier3:$KEY] finished; marker=$(cat "$MARK")"
