#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/agastyas/nla_cross_model_swap
export http_proxy=http://www-cache.rz.ruhr-uni-bochum.de:80
export https_proxy="$http_proxy"
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$https_proxy"
export no_proxy=localhost,127.0.0.1,.ruhr-uni-bochum.de
export PATH="/home/agastyas/.local/bin:$PATH"
export UV_CACHE_DIR="$ROOT/uv_cache"
mkdir -p "$ROOT/logs"
cd "$ROOT/EasyNLA"
if [[ ! -x "$ROOT/venv/bin/python" ]]; then
    bash scripts/install_vllm_lens.sh "$ROOT/venv"
else
    "$ROOT/venv/bin/python" utils/patch_vllm_lens.py
fi
uv pip install --python "$ROOT/venv/bin/python" -e .
"$ROOT/venv/bin/python" -c 'import torch, transformers, vllm, peft; print(torch.__version__, transformers.__version__, vllm.__version__, peft.__version__)'
touch "$ROOT/ENV_READY"
