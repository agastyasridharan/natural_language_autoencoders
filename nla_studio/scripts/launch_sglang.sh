#!/usr/bin/env bash
# Launch the SGLang server that serves a family's AV (actor) for input_embeds
# injection. One server per family. The FastAPI app talks to it over HTTP.
#
#   scripts/launch_sglang.sh <family> [extra sglang args...]
#
# Env:
#   NLA_SGLANG_PORT     listen port (default 30000; must match NLA_SGLANG_URL)
#   NLA_MEM_FRACTION    --mem-fraction-static (default 0.85)
#   NLA_TP              tensor-parallel size (default: #visible GPUs for multi_gpu families, else 1)
#   NLA_CTX             --context-length (default 2048; NLA prompts are short, 512 is plenty)
#   HF_TOKEN            required for gated (Gemma/Llama) repos
#   CUDA_VISIBLE_DEVICES  pin which GPUs SGLang uses (leave the in-process base+AR a GPU)
set -euo pipefail

FAMILY="${1:-}"
shift || true
if [[ -z "$FAMILY" ]]; then
  echo "usage: $0 <qwen7b|gemma12b|gemma27b|llama70b> [extra sglang args...]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REG="$SCRIPT_DIR/../app/families.yaml"
PORT="${NLA_SGLANG_PORT:-30000}"

# Pull this family's fields out of families.yaml (single source of truth).
# Space-separated values read into shell vars (repo ids contain no spaces).
read -r AV_REPO GEMMA_PATCH GATED MULTI_GPU <<EOF
$(python - "$REG" "$FAMILY" <<'PY'
import sys, yaml
reg = yaml.safe_load(open(sys.argv[1])); fam = sys.argv[2]
if fam not in reg:
    sys.stderr.write(f"unknown family {fam!r}; have: {sorted(reg)}\n"); sys.exit(2)
f = reg[fam]
print(f["av_repo"], int(bool(f.get("gemma_patch"))),
      int(bool(f.get("gated"))), int(bool(f.get("multi_gpu"))))
PY
)
EOF
if [[ -z "${AV_REPO:-}" ]]; then
  echo "[launch_sglang] could not resolve family '$FAMILY' from $REG" >&2
  exit 2
fi

echo "[launch_sglang] family=$FAMILY  av=$AV_REPO  port=$PORT"

if [[ "$GATED" == "1" && -z "${HF_TOKEN:-}" ]]; then
  echo "[launch_sglang] ERROR: $FAMILY uses a gated base/checkpoint — export HF_TOKEN first." >&2
  exit 1
fi

# Gemma needs the gemma3_mm input_embeds bypass applied to the installed sglang.
if [[ "$GEMMA_PATCH" == "1" ]]; then
  echo "[launch_sglang] applying Gemma input_embeds patch to installed sglang…"
  python "$SCRIPT_DIR/apply_gemma_patch.py"
fi

# Tensor parallel: span visible GPUs for multi_gpu families (e.g. llama70b).
if [[ -n "${NLA_TP:-}" ]]; then
  TP="$NLA_TP"
elif [[ "$MULTI_GPU" == "1" ]]; then
  TP="$(python -c 'import torch; print(max(1, torch.cuda.device_count()))' 2>/dev/null || echo 1)"
else
  TP=1
fi
TP_ARG=()
[[ "$TP" -gt 1 ]] && TP_ARG=(--tp "$TP")

set -x
exec python -m sglang.launch_server \
  --model-path "$AV_REPO" \
  --port "$PORT" \
  --disable-radix-cache \
  --mem-fraction-static "${NLA_MEM_FRACTION:-0.85}" \
  --context-length "${NLA_CTX:-2048}" \
  --trust-remote-code \
  "${TP_ARG[@]}" "$@"
