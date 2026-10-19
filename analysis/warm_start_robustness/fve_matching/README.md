# Exact-checkpoint FVE evaluation and matching

This component computes FVE for the actor and historical critic from the same
immutable RL transaction, then applies the frozen warm-start matching rule. It
does not infer checkpoint FVE from the training curve.

The indexing offset is mandatory: the trainer's `eval@N` runs after zero-based
update `N`, while `iter_N` and transaction `step_N` were saved after update
`N-1`. Therefore `eval@N` describes the state corresponding to transaction
`N+1`, not `iter_N`. With evaluations every 10 steps and checkpoints every 20,
the logged curve has no exact observation for the ordinary 20-step checkpoint
transactions. Every comparison checkpoint is re-evaluated directly.

`protocol.yaml` freezes this mapping, the fixed evaluation, selector, and the
two downstream claim-analysis datasets in machine-readable form.

## Exact evaluator

`evaluate_checkpoint.py` performs these checks before reporting a number:

- validates `COMPLETE.json` or the older `READY.json`, actor plus reference
  adapters, contemporaneous critic marker and weights, contiguous point-in-time
  metrics, run config, sidecar, and optimizer presence;
- SHA-256 hashes every FVE-relevant actor, critic, config, sidecar, metric, and
  transaction-marker file;
- reconstructs the trainer's first 256 CRC32 document-hash-held-out rows from
  the exact RL parquet and hashes the full parquet, prompts, and raw activations;
- loads the raw Qwen base plus the transaction's continued PEFT adapter, loads
  the condition's merged AV-SFT model in vLLM, and reuses EasyNLA's
  `sync_actor_to_vllm` and `rollout_batch_vllm` functions;
- requires the patched vLLM-lens Karvonen layer-1 norm-matched residual path,
  per-request steering verification, and the global application counter;
- greedily generates the original 256-prompt evaluation batch at temperature
  0, top-p 1, top-k -1, and 150 tokens with TP1/eager/no-prefix-cache settings;
- scores with the transaction's historical critic through EasyNLA's batched,
  suffix-anchored `score_with_critic`, using the held-out set's own raw-variance
  FVE denominator and the trainer eval's `-2` failed-sample convention.

TP1 is intentional: the historical evaluations used one TP1 vLLM engine on
rank 0. To use four GPUs, launch four independent evaluator processes, each
with exactly one physical GPU visible. Do not evaluate one checkpoint with
TP4, which changes the numerical engine being compared.

Example for one transaction on Athena:

```bash
PY=/data/agastyas/Miniconda3/bin/python
ROOT=/data/agastyas/nla_warmstart_robustness
CODE=$ROOT/easynla
FVE_CODE=$ROOT/fve_matching
RUN=$ROOT/qwen25_7b_seed42
RL=$RUN/conditions/implausible/rl_shuf.parquet
TX=$RUN/checkpoints/plausible/rl_vllm/history/step_000780
OUT=$RUN/evaluation/exact_fve/plausible_step_000780.json

CUDA_VISIBLE_DEVICES=0 $PY "$FVE_CODE/evaluate_checkpoint.py" \
  --condition plausible \
  --transaction "$TX" \
  --easynla-root "$CODE" \
  --rl-parquet "$RL" \
  --expected-rl-parquet-sha256 REPLACE_WITH_VERIFIED_SHA256 \
  --output "$OUT"
```

The transaction's resolved `base_ckpt` and `av_ckpt` are used by default.
`--base-ckpt` and `--av-ckpt` are relocation overrides for restored archives;
all resolved paths and runtime contracts are recorded. The generation batch is
atomically saved to `<output-stem>.generations.json` before critic scoring, so
a scoring interruption can resume without regenerating. The final JSON keeps
raw text, tokens, sampled log probabilities, extraction status, steering/CJK/
marker diagnostics, per-prompt reward/FVE, all provenance, and payload hashes.

`--skip-transaction-hashes` is for development smoke only. The selector rejects
such a result.

## Frozen selector

First evaluate `step_000780` for both conditions. Hold the condition with lower
exact final FVE at step 780. For the higher-final-FVE condition, evaluate every
historical transaction at steps 20, 40, ..., 780. The selector minimizes the
absolute FVE difference from the held final and breaks an exact tie in favor of
the later checkpoint.

```bash
RESULT_ARGS=(
  --result "$RUN/evaluation/exact_fve/implausible_step_000780.json"
)
for STEP in $(seq 20 20 780); do
  RESULT_ARGS+=(
    --result "$RUN/evaluation/exact_fve/plausible_step_$(printf '%06d' "$STEP").json"
  )
done
$PY "$FVE_CODE/select_checkpoint.py" "${RESULT_ARGS[@]}" \
  --output "$RUN/evaluation/exact_fve/fve_matched_selection.json"
```

The selector determines which condition is higher rather than assuming it. It
requires the complete 20-step candidate grid for that condition, identical
evaluation protocol, identical RL parquet and held-out-set hashes, identical
FVE baseline, valid transaction hashes, and self-consistent result payload
hashes. Missing historical critics fail closed. In particular, actor-only
implausible checkpoints before the preserved historical transactions cannot be
assigned exact FVE using a later critic.

## Two separate claim-analysis datasets

The selector emits `claim_analysis_sets` with two fully identified datasets:

1. `final_vs_final`: plausible `iter_000780` versus implausible `iter_000780`.
2. `fve_matched`: the lower-final-FVE condition at `iter_000780` versus the
   higher condition's objectively selected 20-step transaction.

Each member includes a checkpoint ID, transaction fingerprint, actor-adapter
SHA-256, exact-FVE result hash, step, last update index, and raw-base and
merged-AV checkpoint tree hashes. Claim generation and scoring
must use separate output directories, explanation JSONLs, and
pipeline invocations for these two datasets. Current activation identities omit
`checkpoint_id`; combining the datasets would collide keys and could silently
pair explanations from different checkpoints. Even a checkpoint shared by both
comparisons should be namespaced under each analysis-set ID.

CPU-only validation:

```bash
python3 -m unittest discover \
  -s analysis/warm_start_robustness/fve_matching/tests -v
```
