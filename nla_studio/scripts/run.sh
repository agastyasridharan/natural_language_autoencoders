#!/usr/bin/env bash
# One command to bring up NLA Studio.
#
#   scripts/run.sh <family> [--sglang]
#
#   --sglang   also launch the AV SGLang server for <family> in the background
#              and wait until it is healthy before starting the web app.
#              Omit it if you already started SGLang yourself (separate GPU/host).
#
# Env (forwarded to the app — see app/server.py):
#   NLA_INPROC_DEVICE   base+AR device      (default cuda:0)
#   NLA_BASE_DEVICE     base device         (default NLA_INPROC_DEVICE)
#   NLA_AR_DEVICE       AR device           (default NLA_INPROC_DEVICE)
#   NLA_EXTRACT_CACHE   cached extractions  (default 8)
#   NLA_SGLANG_PORT     AV server port      (default 30000)
#   NLA_PORT            web app port        (default 8000)
#   NLA_HOST            web app host        (default 0.0.0.0)
#   HF_TOKEN            for gated families
#   HF_HUB_ENABLE_HF_TRANSFER=1  faster weight downloads (hf_transfer)
set -euo pipefail

FAMILY="${1:-}"
LAUNCH_SGLANG=0
for a in "$@"; do [[ "$a" == "--sglang" ]] && LAUNCH_SGLANG=1; done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SGLANG_PORT="${NLA_SGLANG_PORT:-30000}"
export NLA_SGLANG_URL="http://localhost:${SGLANG_PORT}"
export NLA_INPROC_DEVICE="${NLA_INPROC_DEVICE:-cuda:0}"

SGLANG_PID=""
cleanup() { [[ -n "$SGLANG_PID" ]] && kill "$SGLANG_PID" 2>/dev/null || true; }
trap cleanup EXIT

if [[ "$LAUNCH_SGLANG" == "1" ]]; then
  [[ -z "$FAMILY" ]] && { echo "usage: $0 <family> --sglang" >&2; exit 2; }
  LOG="$ROOT/sglang_${FAMILY}.log"
  echo "[run] launching SGLang AV for $FAMILY (logs -> $LOG)…"
  NLA_SGLANG_PORT="$SGLANG_PORT" "$SCRIPT_DIR/launch_sglang.sh" "$FAMILY" >"$LOG" 2>&1 &
  SGLANG_PID=$!
  echo "[run] waiting for SGLang to load (this can take several minutes for big models)…"
  for _ in $(seq 1 240); do  # up to ~20 min
    if curl -fsS "http://localhost:${SGLANG_PORT}/get_model_info" >/dev/null 2>&1; then
      echo "[run] SGLang is up."; break
    fi
    if ! kill -0 "$SGLANG_PID" 2>/dev/null; then
      echo "[run] SGLang process died — see $LOG"; tail -n 40 "$LOG"; exit 1
    fi
    sleep 5
  done
fi

echo "[run] starting web app on ${NLA_HOST:-0.0.0.0}:${NLA_PORT:-8000}  (open it in a browser)"
cd "$ROOT"
exec python -m uvicorn app.server:app \
  --host "${NLA_HOST:-0.0.0.0}" --port "${NLA_PORT:-8000}"
