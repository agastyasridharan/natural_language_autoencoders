#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/agastyas/nla_cross_model_swap
export OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
rm -f "$ROOT/PREFLIGHT_FAILED" "$ROOT/gemma_probe.json"
PY="$ROOT/venv/bin/python"
paused=""
cleanup() {
  code=$?
  if [[ -n "$paused" ]]; then kill -CONT "$paused" 2>/dev/null || true; fi
  if [[ $code != 0 ]]; then echo "$code" > "$ROOT/PREFLIGHT_FAILED"; fi
}
trap cleanup EXIT
cd "$ROOT/EasyNLA"
"$PY" "$ROOT/test_swap.py"
"$PY" "$ROOT/prepare_tiny_smoke.py"
for mode in av ar; do
  extra=()
  if [[ $mode == av ]]; then extra+=(--use-lora --lora-r 8); fi
  CUDA_VISIBLE_DEVICES=0,1 "$PY" -m torch.distributed.run --standalone --nproc-per-node=2 -m nla.train_sft \
    --mode "$mode" --base-ckpt "$ROOT/tiny_smoke/base" --parquet "$ROOT/tiny_smoke/$mode.parquet" \
    --save-dir "$ROOT/tiny_smoke/${mode}_out" --batch-size 1 --gradient-accumulation-steps 2 \
    --num-steps 3 --attn-implementation eager --lr-warmup-steps 1 --save-every 3 --no-wandb "${extra[@]}" \
    > "$ROOT/logs/tiny_$mode.log" 2>&1
done
touch "$ROOT/DISTRIBUTED_SFT_PASSED"
while [[ ! -f "$ROOT/checkpoints/gemma2-base/weights_verified.json" ]]; do sleep 15; done
# Hold this owned extraction process briefly so vLLM memory profiling sees a stable GPU.
paused=$(pgrep -f '^/data/agastyas/nla_cross_model_swap/venv/bin/python /data/agastyas/nla_cross_model_swap/extract_qwen.py .*--rank 6 ' || true)
if [[ -n "$paused" ]]; then kill -STOP "$paused"; fi
CUDA_VISIBLE_DEVICES=6 "$PY" "$ROOT/probe_gemma.py" --checkpoint "$ROOT/checkpoints/gemma2-base" \
  --teacher-receipt "$ROOT/teacher_text_v2.parquet.json" --output "$ROOT/gemma_probe.json" \
  > "$ROOT/logs/gemma_probe.log" 2>&1
echo Preflight passed
