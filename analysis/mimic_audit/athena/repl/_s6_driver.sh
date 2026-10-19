#!/bin/bash
# S6 cross-model cycle pipeline driver (Athena, GPUs 2,3 ONLY).
#   $1 KEY       repl_common.MODELS key (gemma12b|gemma27b|llama70b)
#   $2 RUN       atom run id
#   $3 RUNS_DIR  dir containing <RUN>/records.jsonl and <RUN>/vectors/
#   $4 MODE      "cores" (AV+AR co-reside) | "split" (70B: AR precompute first)
# Calibrates on the 25 ok[::16] activations, cycle-runs all 100 ok[::4]
# activations (25 in-sample + 75 out-of-sample). Gold is always the _fineweb
# corpus (the doc_id-matched one; out_corpus_<key> without the suffix is a
# different UltraFineWeb corpus and fails the l0 integrity check).
set -euo pipefail
KEY=$1; RUN=$2; RUNS_DIR=$3; MODE=${4:-cores}
NLA=/data/agastyas/nla_mimic
GOLD=$NLA/out_corpus_${KEY}_fineweb
ARC=$NLA/ar_cache_d4g24
PY=/data/agastyas/Miniconda3/bin/python
export CUDA_VISIBLE_DEVICES=2,3
export HF_HUB_CACHE=/data/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
CI="--idx 0 --idx 16 --idx 32 --idx 48 --idx 65 --idx 81 --idx 97 --idx 113 \
--idx 129 --idx 145 --idx 161 --idx 177 --idx 193 --idx 209 --idx 225 --idx 241 \
--idx 257 --idx 273 --idx 289 --idx 305 --idx 321 --idx 337 --idx 353 --idx 369 \
--idx 385"
cd "$NLA/code/repl"

AR_FLAG=""
if [ "$MODE" = "split" ]; then
  echo "### [$KEY] STEP 0/4 AR precompute (critic only, all acts) $(date +%H:%M:%S)"
  $PY ar_precompute.py --key "$KEY" --run "$RUN" --runs-dir "$RUNS_DIR" \
      --gold-dir "$GOLD" --decoys-dir "$NLA/decoys_d4g24" --out "$ARC" \
      --prompt-version d4
  AR_FLAG="--ar-cache $ARC"
fi

echo "### [$KEY] STEP 1/4 calibration measure (25 acts) $(date +%H:%M:%S)"
$PY calib_cycle.py "$KEY" --run "$RUN" --runs-dir "$RUNS_DIR" --gold-dir "$GOLD" \
    --decoys-dir "$NLA/decoys_d4g24" --out "$NLA/calib_d4g24" $CI \
    --prompt-version d4 $AR_FLAG

echo "### [$KEY] STEP 2/4 analyze_signed $(date +%H:%M:%S)"
$PY calib_cycle.py "$KEY" --run "$RUN" --analyze --out "$NLA/calib_d4g24"

echo "### [$KEY] STEP 3/4 cycle_run (all 100 acts) $(date +%H:%M:%S)"
$PY cycle_run.py "$KEY" --run "$RUN" --runs-dir "$RUNS_DIR" --gold-dir "$GOLD" \
    --decoys-dir "$NLA/decoys_d4g24" \
    --calib "$NLA/calib_d4g24/$RUN/calibration_signed.json" \
    --out "$NLA/cycle_runs/${KEY}-d4g24" --prompt-version d4 $AR_FLAG

echo "### [$KEY] STEP 4/4 report $(date +%H:%M:%S)"
$PY report.py --run-dir "$NLA/cycle_runs/${KEY}-d4g24/$RUN" \
    --calib "$NLA/calib_d4g24/$RUN/calibration_signed.json" \
    --calib-claims-dir "$NLA/calib_d4g24/$RUN/claims"

echo "### [$KEY] PIPELINE DONE $(date +%H:%M:%S)"
