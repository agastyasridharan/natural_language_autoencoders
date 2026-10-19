#!/bin/bash
# Run the full replication pipeline for one model on a single idle H100.
# extract (base -> gold) -> avgen (AV -> real reads) -> score (AR -> cos).
# Usage: run_one.sh <key>     key in {qwen7b, gemma12b, gemma27b}
set -u
KEY=${1:?usage: run_one.sh <key>}
export http_proxy=http://www-cache.rz.ruhr-uni-bochum.de:80
export https_proxy=$http_proxy
export no_proxy=.rub.de,.ruhr-uni-bochum.de,localhost,127.0.0.1,::1
export HF_HOME=/data/hf_cache HF_HUB_CACHE=/data/hf_cache
export HUGGINGFACE_HUB_CACHE=/data/hf_cache TRANSFORMERS_CACHE=/data/hf_cache
export HF_TOKEN_PATH=$HOME/.cache/huggingface/token
PY=/data/agastyas/Miniconda3/bin/python
R=/data/agastyas/nla_mimic/code/repl
NEED=88000     # one fully-idle H100 (gemma27b base/AV ~54GB; don't co-locate)

echo "$(date '+%F %T') [$KEY] waiting for one idle GPU"
while true; do
  one=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
        | awk -F, -v n=$NEED '$2+0>=n{gsub(/ /,"",$1);print $1}' | head -1)
  [ -n "$one" ] && break
  sleep 60
done
export CUDA_VISIBLE_DEVICES=$one
echo "$(date '+%F %T') [$KEY] GPU=$one"

if $PY "$R/extract_model.py" "$KEY"; then
  if $PY "$R/avgen_model.py" "$KEY"; then
    if $PY "$R/score_model.py" "$KEY"; then
      echo "$(date '+%F %T') [$KEY] ALL DONE"; exit 0
    else echo "$(date '+%F %T') [$KEY] SCORE FAILED"; fi
  else echo "$(date '+%F %T') [$KEY] AVGEN FAILED"; fi
else echo "$(date '+%F %T') [$KEY] EXTRACT FAILED"; fi
exit 1
