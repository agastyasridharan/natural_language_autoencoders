# Claim segmentation, equivalence, scoring, and inference

The AV generator is documented in `README.md`. This document covers the independent downstream
claim-agreement pipeline in `run_pipeline.py`; its frozen decisions are in `protocol.yaml`.

The pipeline uses temperature-0.7 explanations from two samples per AV and activation. It evaluates
all four plausible–implausible pairs plus the one repeated-sample pair within each AV. Every
explanation is returned by the segmenter as an ordered tiling whose source fields must concatenate to
the input exactly. Substantive units are atomic claims labeled `broad_theme` or `entity_detail`.

Equivalence requires mutual entailment at the same specificity. Same-topic, broader/narrower,
different-entity, quantity, polarity, modality, or temporal claims do not match. Scoring follows the
paper's existential definition: a claim is covered if any claim on the other side is equivalent;
matching is not forced to be one-to-one. Directional overlaps and their harmonic mean are emitted for
all claims and both specificity strata.

The primary analysis includes complete instrument pairs with nonzero denominators. The sensitivity
analysis retains every planned pair and imputes zero to generation, segmentation, equivalence, and
zero-denominator failures. Pair variants are averaged within activation before inference. FineWeb
bootstraps documents; OOD bootstraps trajectories. The primary OOD macro gives each of the 16
benchmark families weight 1/16. Results also include each benchmark family, alignment and capability
group summaries, an OOD micro estimate, and a secondary 50/50 alignment–capability group macro.

## Input

Pass one or more generator `explanations.jsonl` files using repeated `--input`. The v1 input contract
requires:

```text
activation_id, document_id, trajectory_id, distribution, benchmark, family,
condition, sample_index, model_id, checkpoint_id, temperature, explanation,
generation_ok, failure_reason
```

Failed generations must remain present with `generation_ok: false`. FineWeb requires `document_id`;
OOD requires a trajectory identifier shared by all correlated activations from one trajectory.

## No-API dry run and resumable batches

The default `status` stage launches no API calls. It reports cache coverage and conservative cost and
token bounds for both the selected batch and all currently estimable pending calls.

```bash
python -m analysis.warm_start_robustness.claim_agreement.run_pipeline \
  --input /path/plausible/explanations.jsonl \
  --input /path/implausible/explanations.jsonl \
  --stage status
```

Uncached calls require `--execute` and `OPENAI_API_KEY_2`. `--max-calls` bounds new attempts across an
`all` invocation. Run stages repeatedly to resume from immutable cache entries:

```bash
python -m analysis.warm_start_robustness.claim_agreement.run_pipeline \
  --input /path/plausible/explanations.jsonl --input /path/implausible/explanations.jsonl \
  --stage segment --execute --max-calls 100 \
  --archive-status /data/agastyas/nla_warmstart_robustness/claim_agreement/archive_runtime/status.json

python -m analysis.warm_start_robustness.claim_agreement.run_pipeline \
  --input /path/plausible/explanations.jsonl --input /path/implausible/explanations.jsonl \
  --stage equivalence --execute --max-calls 100 \
  --archive-status /data/agastyas/nla_warmstart_robustness/claim_agreement/archive_runtime/status.json

python -m analysis.warm_start_robustness.claim_agreement.run_pipeline \
  --input /path/plausible/explanations.jsonl --input /path/implausible/explanations.jsonl \
  --stage score --finalize
```

The API layer reads only `OPENAI_API_KEY_2`, sends `store: false`, uses the pinned configured model,
has no model fallback, and performs no automatic retry. `--retry-failures` explicitly appends a
same-model attempt without deleting earlier failures. A conservative reservation precedes every
request. Reported usage is priced exactly; when usage is unavailable, the full reservation remains
charged. The claim pipeline's protocol maximum is $12 because teacher-generation spend shares the
global experiment budget.

Every logical call stores a redacted request, full response body and headers, parsed output, usage,
cost, and error state before a derived artifact is emitted. Final outputs are:

- `segmented_explanations.jsonl`
- `pair_plan.jsonl`
- `equivalence.jsonl`
- `pair_scores.jsonl`
- `summary.json`
- `bootstrap_replicates.npz` and `bootstrap_index.json`
- `run_metadata.json`, including input/protocol/artifact hashes and the full budget ledger

## Required off-cluster cache watcher

Paid execution fails closed unless `--archive-status` points to a healthy watcher heartbeat no more
than 150 seconds old. The watcher creates or opens only a private Hugging Face repository, scans for
both OpenAI key values before packaging, and never calls an OpenAI endpoint. Every changed cache and
results tree becomes a deterministic content-addressed tar snapshot. The bundle embeds an exact
per-file SHA-256/byte manifest; after upload, the watcher independently downloads and hashes the
remote bundle and metadata before writing an atomic local receipt. Lock and temporary files are
excluded. Verified receipts make restarts idempotent, and a restore command reconstructs both trees
into empty destinations with per-file verification.

On Athena, keep archive runtime state outside both source trees:

```bash
ROOT=/data/agastyas/nla_warmstart_robustness/claim_agreement
REPO_ROOT=$ROOT/repo
CODE=$REPO_ROOT/analysis/warm_start_robustness/claim_agreement
CACHE=$ROOT/cache
RESULTS=$ROOT/results
ARCHIVE=$ROOT/archive_runtime
REPO=agastyasridharan/nla-warmstart-claim-agreement

export PYTHONPATH="$REPO_ROOT"
export HF_HOME=/data/hf_cache
export HF_HUB_CACHE=/data/hf_cache
export HF_TOKEN_PATH=$HOME/.cache/huggingface/token
export http_proxy=http://www-cache.rz.ruhr-uni-bochum.de:80
export https_proxy=$http_proxy
export no_proxy=.rub.de,.ruhr-uni-bochum.de,localhost,127.0.0.1,::1

python -m analysis.warm_start_robustness.claim_agreement.archive_claim_cache watch \
  --cache-dir "$CACHE" \
  --results-dir "$RESULTS" \
  --protocol "$CODE/protocol.yaml" \
  --repo-id "$REPO" --repo-type dataset \
  --path-prefix claim_agreement/v1 \
  --receipts-dir "$ARCHIVE/receipts" \
  --staging-dir "$ARCHIVE/staging" \
  --hf-cache-dir /data/hf_cache \
  --status-path "$ARCHIVE/status.json" \
  --stop-file "$ARCHIVE/STOP" \
  --poll-seconds 60
```

Run the watcher in a detached supervisor before setting `--execute`; retain its stdout/stderr under
`$ARCHIVE`, not under `$RESULTS`. After the final `score --finalize`, create `$ARCHIVE/STOP`. Do not
declare completion until `status.json` says `state: complete`, `verified_off_cluster: true`, and its
snapshot ID has a verified receipt under `receipts/snapshots/`.

To prove recoverability without modifying the live cache, restore a receipt's snapshot ID into fresh
paths:

```bash
python -m analysis.warm_start_robustness.claim_agreement.archive_claim_cache restore \
  --repo-id "$REPO" --repo-type dataset \
  --path-prefix claim_agreement/v1 \
  --snapshot-id SNAPSHOT_ID_FROM_STATUS \
  --hf-cache-dir /data/hf_cache \
  --cache-destination "$ROOT/restore_test/cache" \
  --results-destination "$ROOT/restore_test/results"
```

Run the CPU tests with:

```bash
python -m unittest discover \
  -s analysis/warm_start_robustness/claim_agreement/tests -v
```
