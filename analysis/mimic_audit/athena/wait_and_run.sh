#!/bin/bash
# Unattended launcher for the llama70b activation-reconstruction run on Athena.
# Waits for (1) the model download to finish and (2) two genuinely-idle H100s
# (respecting the cluster's no-sharing policy), then runs extract -> score.
set -u
export http_proxy=http://www-cache.rz.ruhr-uni-bochum.de:80
export https_proxy=$http_proxy
export no_proxy=.rub.de,.ruhr-uni-bochum.de,localhost,127.0.0.1,::1
export HF_HOME=/data/hf_cache HF_HUB_CACHE=/data/hf_cache
export HUGGINGFACE_HUB_CACHE=/data/hf_cache TRANSFORMERS_CACHE=/data/hf_cache
export HF_TOKEN_PATH=$HOME/.cache/huggingface/token
PY=/data/agastyas/Miniconda3/bin/python
CODE=/data/agastyas/nla_mimic/code
NEED=92000   # MiB free => GPU is essentially idle (H100 NVL ~95830)

echo "$(date '+%F %T') waiter start"
while pgrep -f snapshot_download >/dev/null; do echo "$(date '+%F %T') download still running..."; sleep 60; done
echo "$(date '+%F %T') download finished"

while true; do
  free=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
         | awk -F, -v n=$NEED '$2+0>=n{gsub(/ /,"",$1);print $1}')
  cnt=$(printf '%s\n' "$free" | grep -c .)
  if [ "${cnt:-0}" -ge 2 ]; then
    two=$(printf '%s\n' "$free" | head -2 | paste -sd, -)
    export CUDA_VISIBLE_DEVICES=$two
    echo "$(date '+%F %T') idle GPUs -> CUDA_VISIBLE_DEVICES=$two"
    echo "===== EXTRACT $(date '+%F %T') ====="
    if $PY "$CODE/extract_llama.py"; then
      echo "===== SCORE $(date '+%F %T') ====="
      if $PY "$CODE/score_llama.py"; then
        echo "$(date '+%F %T') ALL DONE"
      else
        echo "$(date '+%F %T') SCORE FAILED"
      fi
    else
      echo "$(date '+%F %T') EXTRACT FAILED"
    fi
    break
  fi
  echo "$(date '+%F %T') only ${cnt:-0} idle GPU(s) (need 2); sleeping 60s"
  sleep 60
done
