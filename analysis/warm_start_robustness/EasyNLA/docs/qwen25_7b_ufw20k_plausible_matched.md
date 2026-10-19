# Matched 20k plausible warm-start runbook

This run is the control for the completed implausible-initialization run. It
changes only the teacher explanations used for AV/AR SFT. It must not re-extract
activations, rerun the document split, or create a plausible copy of RL data.

## Locked comparison

- Base: `Qwen/Qwen2.5-7B-Instruct`, layer 20, Ultra-FineWeb 20,000-document slice, seed 42.
- AV SFT: exactly the 49,495 activation keys retained by the implausible teacher.
- AR SFT: exactly the 49,516 activation keys retained by the implausible teacher.
- Key: `(doc_id, n_raw_tokens, activation_layer)`, unique within each split.
- SFT order: exactly the corresponding implausible `*_sft_shuf.parquet` key order.
- Activation vectors, source text, document partition, and provenance: byte/value checked against the implausible parquets.
- RL: directly use `conditions/implausible/rl_shuf.parquet` and its sidecar. There is no plausible RL parquet.

`generate_matched_teacher.py` uses the implausible explained parquet as the
authoritative key set, selects those rows from the immutable stage-1 raw split,
and atomically publishes only after every key has a parseable plausible label.
Malformed responses are retried with attempt-specific formatting reminders;
the reminders change formatting instructions only, not the semantic task. Raw
request, response, error, token-usage, cleaned attempt, and failure records are
all retained.

## Teacher and budget lock

- Model: `gpt-5-mini`
- Endpoint: OpenAI Batch Chat Completions
- Key environment variable: `OPENAI_API_KEY_2` only
- Maximum completion tokens: 600
- Reasoning effort: `minimal`
- Verbosity: `low`
- Current Batch prices checked 2026-08-28: $0.25/M input, $0.025/M cached input, $2.00/M output ([official model page](https://developers.openai.com/api/docs/models/gpt-5-mini)).

Recorded pre-run project spend is $42.31016656. The implausible Batch cache
accounts for approximately $42.2396 of that total; the remainder came from
preserved QC/smoke calls. The shared cache cap is now $87.23, with a prior-cache
floor of $42.2395. Therefore this teacher can spend at most approximately
$44.9905, total project spend cannot exceed approximately $87.30057, and at
least $12.69943 remains for downstream claim adjudication. A missing/incomplete
prior cache fails closed rather than undercounting.

Scaling the implausible run's $42.2396 / 99,947-request actual cost to 99,011
matched requests estimates $41.84 before retries. At the previous 0.94% format
drop rate, one retry round adds roughly $0.40, so a reasonable planning estimate
is $42-43. The sub-$45 generation allowance is a guard, not a target.

Every provider write uses temporary-file plus atomic rename for input JSONL,
state, raw output, errors, per-batch usage, and cumulative usage. The cumulative
ledger records actual cache spend, outstanding pessimistic reservations, known
external spend, downstream reserve, and maximum project spend after commitments.
Requests are chunked at 2,000 rows so the 600-token worst-case reservation is
smaller than the previous 8,000-row batches. Each reservation uses a true
byte-level upper bound for input tokens plus all 600 possible output tokens;
it does not rely on an average bytes-per-token estimate. If a late 2,000-row
reservation does not fit, only that never-submitted call is recursively split
until it fits at finer granularity or a single request fails closed.
Failed, expired, or cancelled Batch jobs remain journaled with their IDs,
counts, error-file IDs, terminal state, and original pessimistic reservation.
Automatic terminal resubmission is prohibited because such jobs can contain
billable partial successes. A terminal job therefore stops generation and
continues consuming its full reservation until its partial output and usage are
manually reconciled; this prevents partial work from evading the global cap.
After both exact-matched teacher parquets are complete, the preparation launcher
requires every Batch state to have raw input, raw output, and reconciled usage,
then uploads the full plausible condition to the private checkpoint repository.
During generation, a separate watcher uploads each completed content-addressed
Batch directory as soon as it appears and archives every cumulative-ledger
revision by content hash. Terminal or interrupted Batch evidence (input, state,
partial output/errors when available, and unreleased reservation) is archived
under an immutable attempt hash before a failure-path stop can complete. Watcher
health, lag, incomplete records, latest ledger hash, and receipts are written
atomically under `checkpoints/plausible/archive_runtime` and
`checkpoints/plausible/archive_receipts/api_cache`. Every remote file is
SHA-256 verified; an existing different remote manifest is never overwritten,
and the API credential itself is scanned for and excluded from all artifacts.

## SFT/RL recipe

Both SFT runs are exactly one epoch and exactly 774 optimizer steps:

- AV: 49,495 rows; micro-batch 32; accumulation 2; effective batch 64; LoRA r=128/alpha=16; LR 1e-4; minimum LR 1e-5; 50 warmup steps.
- AR: 49,516 rows; micro/effective batch 64; 21 layers; LoRA r=128/alpha=16; LR 2e-5; minimum LR 2e-6; 50 warmup steps; final norm stripped.

SFT saves every 20 steps plus the final step 774. This persistence-only change
does not alter batches, gradients, learning rates, sample order, or the 774-step
one-epoch optimization recipe. Each atomic checkpoint contains adapter/value
head weights, optimizer and scheduler state, the exact shuffled permutation and
cursor, CPU/CUDA/NumPy RNG state, point-in-time metrics and log, resolved config,
sidecar, and provenance. Exact resume is supported without repeating or skipping
training rows; W&B retains the same run ID and excludes the local `resume_from`
path from experiment config while rejecting any substantive recipe drift.

The RL config is parameter-identical to the implausible 20k recipe: 780 steps,
256 prompts, group size 8, 150-token cap, temperature 1, actor LR 1e-4,
critic LR 8e-5, KL k3 beta 0.01, deterministic 256-prompt held-out evaluation
every 10 steps, and a checkpoint every 20 steps.

## Exact launch sequence

The scripts are dry-run-only unless `--execute` is explicit. After synchronizing
this EasyNLA tree to `/data/agastyas/nla_warmstart_robustness/easynla` on Athena:

```bash
cd /data/agastyas/nla_warmstart_robustness/easynla
export OPENAI_API_KEY_2="..."
bash scripts/prepare_qwen25_7b_ufw20k_plausible.sh --execute

bash scripts/run_qwen25_7b_ufw20k_plausible.sh --phase=sft --execute
bash scripts/run_qwen25_7b_ufw20k_plausible.sh --phase=merge --execute
bash scripts/run_qwen25_7b_ufw20k_plausible.sh --phase=rl --execute
```

The three GPU phases can instead run continuously with:

```bash
bash scripts/run_qwen25_7b_ufw20k_plausible.sh --phase=all --execute
```

On an interruption after an internally consistent immutable RL transaction:

```bash
bash scripts/resume_qwen25_7b_ufw20k_plausible_rl.sh --execute
```

On an interruption after an atomic SFT checkpoint:

```bash
bash scripts/resume_qwen25_7b_ufw20k_plausible_sft.sh --execute
```

The resume launcher refuses mixed actor/critic/optimizer steps and transaction
remnants. With `ar_lora`, current trainer semantics restore critic weights and
the value head but intentionally start fresh critic optimizer moments after
reinjecting the critic LoRA. The actor optimizer moments are restored. The
historical critic optimizer state is still preserved in every transaction.

## Durability and independent verification

AV SFT and AR SFT each retain 39 atomic checkpoints at steps 20, 40, ..., 760,
774. Independent watchers upload and SHA-verify each checkpoint while training
continues; the launcher treats a fatal watcher exit as a reason to stop creating
unprotected checkpoints. Completed SFT directories and merged checkpoints also
receive final folder-level verified archives.

RL publishes 39 immutable transactions at steps 20, 40, ..., 780. Every
transaction contains:

- trainable actor and frozen reference adapters;
- that step's merged critic shards and value head;
- actor and critic optimizer state;
- exact point-in-time scalar metrics;
- resolved run config and NLA sidecar;
- code/config/provenance snapshot and point-in-time training log;
- a completion marker and local inventory.

RL local history uses hard links for immutable large files, but historical critic
and optimizer inodes still consume physical space after `critic_latest` and
`optim_latest` advance. The launcher therefore requires at least 800 GiB free.
An independent watcher uploads every transaction to the private Hugging Face
repo and verifies all large LFS SHA-256 digests plus downloaded Git-file hashes.
Training completion is not reported until all 39 receipts, including step 780,
exist locally.

Expected timing is uncertain. Prior measurements imply approximately 1.5-2
hours for concurrent SFT compute, under one hour for merge/smoke, and roughly
10-13 hours for RL compute before any off-cluster archive tail. The 20-step SFT
archive cadence adds I/O and may extend SFT wall time; it does not change
optimization. Uploading and verifying roughly 0.5-0.6 TB of RL historical
restart state can continue substantially longer depending on
Athena-to-Hugging-Face bandwidth. Batch API generation may complete
in several hours but has a documented 24-hour completion window per submitted
batch; AV and AR workers run concurrently and resume without duplicate billing.

## Residual fail-closed conditions

- More than three formatting retry rounds: no final teacher parquet is published; increase `--format-retry-rounds` and resume the durable attempts.
- Pessimistic next-batch reservation breaches $87.23 even if likely actual cost is smaller: generation stops before submission.
- The shared prior Batch ledger is below $42.2395: generation stops because spend history may be missing.
- A Batch expires, fails, or is cancelled: generation stops with its full reservation still charged; partial billable usage must be reconciled before any manual retry.
- An API/SFT archive watcher reports a conflicting immutable remote manifest: the associated local worker is stopped and the conflict must be audited; no remote history is overwritten.
- Any key, source text, activation vector, order, sidecar, or expected row count differs from the implausible reference: training does not start.
- Any GPU is occupied, a fresh destination is nonempty, disk headroom is below 800 GiB, or the archive repository is not private/reachable: the relevant GPU phase does not start.
- Hugging Face quota or sustained proxy/bandwidth failure can make the archive watcher lag training. Local immutable transactions remain available; the watcher retries and the launcher does not declare completion before final verification.
