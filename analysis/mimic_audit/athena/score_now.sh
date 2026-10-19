#!/bin/bash
# Re-run Phase 2 (score) only. Phase 1 gold.npy is already saved and valid
# (alignment 128/128). Picks 2 genuinely-idle H100s at launch time.
set -u
export http_proxy=http://www-cache.rz.ruhr-uni-bochum.de:80
export https_proxy=$http_proxy
export no_proxy=.rub.de,.ruhr-uni-bochum.de,localhost,127.0.0.1,::1
export HF_HOME=/data/hf_cache HF_HUB_CACHE=/data/hf_cache
export HUGGINGFACE_HUB_CACHE=/data/hf_cache TRANSFORMERS_CACHE=/data/hf_cache
export HF_TOKEN_PATH=$HOME/.cache/huggingface/token
PY=/data/agastyas/Miniconda3/bin/python
CODE=/data/agastyas/nla_mimic/code
NEED=92000

free=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
       | awk -F, -v n=$NEED '$2+0>=n{gsub(/ /,"",$1);print $1}')
cnt=$(printf '%s\n' "$free" | grep -c .)
if [ "${cnt:-0}" -lt 2 ]; then
  echo "$(date '+%F %T') only ${cnt:-0} idle GPU(s) (need 2); aborting"; exit 1
fi
two=$(printf '%s\n' "$free" | head -2 | paste -sd, -)
export CUDA_VISIBLE_DEVICES=$two
echo "$(date '+%F %T') SCORE on CUDA_VISIBLE_DEVICES=$two"
if $PY "$CODE/score_llama.py"; then echo "$(date '+%F %T') ALL DONE"; else echo "$(date '+%F %T') SCORE FAILED"; fi
