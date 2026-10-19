# Qwen2.5 to Gemma-2 NLA experiment

This run trains a Gemma-2-9B-IT verbalizer and reconstructor on frozen
Qwen2.5-7B-Instruct block-20 activations. It uses the EasyNLA LoRA-verbalizer,
full-reconstructor recipe. Both backbones have width 3,584, so this implementation
adds no input projection. Equal width does not imply aligned coordinates: the
trainable models must learn the cross-model mapping. The reconstructor retains
32 Gemma blocks (approximately the same relative depth as 21/28 Qwen blocks).

The run lives on `agastyas@athena02.ig32s.ruhr-uni-bochum.de` under
`/data/agastyas/nla_cross_model_swap`. Eight H200 GPUs are assigned. Nothing in
the parent repository's existing training implementation or upstream Miles was
modified. The isolated EasyNLA checkout starts at
`4d728477960c18cdfa36dc04ec738d7f55af9f0b`.

## Repository snapshot

This directory is a source and provenance archive of the experiment. `EasyNLA/`
contains the actual modified training source as ordinary files, including its MIT
license; no submodule initialization is needed. `UPSTREAM.json` records the pinned
upstream revision and modified files. The original `run_manifest.json` is retained
as historical run provenance, rather than regenerated to describe publication.

Model weights, activation datasets, optimizer binaries, and full training logs
remain on Athena02 and are not included in Git. `pause_state/` contains lightweight
checkpoint metadata, checksums, and the prepared resume command, not the tensors.
The recorded pause status is from October 5, 2026; publication does not restart or
recheck the cluster job.

The launch scripts preserve the original Athena filesystem paths and proxy setup.
For another machine, adapt those deployment paths and provision the pinned models
and data first. Run the CPU checks with `python test_swap.py` from this directory
in the experiment's dependency environment. See the execution and resumption
sections before launching anything; `pipeline.py` starts a new full run and is
not the paused-run resume entry point.

## Data and interpretation

The existing EasyNLA public warm-start corpus supplies visible text prefixes and
explanation labels originally associated with Qwen3-8B. Its activation vectors
are discarded. Each prefix is retokenized with Qwen2.5, and its final-token
block-20 output is re-extracted. Positions below index 50 are excluded. Prefixes
are never silently truncated. Thus these are transferred teacher labels for
warm-starting, not new teacher annotations of Qwen2.5. RL subsequently scores
reconstruction against the newly extracted Qwen2.5 targets.

Before the position filter, the document-disjoint allocation contains 180,517 AV
rows, 182,593 AR rows, 364,839 RL rows, and 14,712 validation rows. Original
validation documents remain held out. Training documents are deterministically
assigned to AV, AR, or RL; no document crosses these boundaries. One repeated
prefix is retained within its assigned document partition and recorded in the
teacher receipt. Final counts appear in `data/manifest.json`.

Parquets contain raw FP32 storage of BF16 forward-pass outputs, with `norm:none`.
EasyNLA injects an activation additively at the output of block 1, matching the
norm of the receiving residual at the marker. Loss uses direction-normalized
MSE. This differs from the original repository's embedding-replacement recipe;
any matched baseline must use the same EasyNLA injection, data, and optimizer
schedule. This first run does not by itself establish a cross-family effect.

## Execution

`run_extract.sh` launches eight independent extraction workers. `run_preflight.sh`
checks source-layer equivalence, injection behavior, patch installation, actual
two-rank SFT execution, and full-Gemma HF/vLLM injected continuation probabilities.
`pipeline.py` waits for completed shards and a passing Gemma preflight, builds
the datasets, and runs the following stages without another launch request:

1. Concurrent four-GPU AV LoRA SFT and four-GPU full AR SFT; approximately one
   epoch each, global effective batch 128. AV uses rank-128 rsLoRA, learning rate
   1e-4; AR uses FP32 master parameters with BF16 compute, learning rate 2e-5.
2. Merge the AV adapter into a BF16 rollout checkpoint. RL continues the same
   SFT adapter and retains a frozen SFT adapter as its KL reference.
3. Eight-rank data-parallel RL: 2,100 updates, 256 activations per update, eight
   explanations per activation, up to 256 generated tokens per explanation.
   The planned sampling budget is 4,300,800 explanations. AV LR is 1e-4 and AR LR
   is 8e-5; KL uses upstream's k3 estimator and coefficient 0.01. AR is fully
   co-trained. No paid text-judge calls are enabled.

Gemma training explicitly uses eager attention because the pinned Transformers
4.57.1 SDPA path omits its attention soft-capping. Rollouts retain Gemma's
soft-capping in vLLM. Prefix caching is disabled because identical prompt tokens
can carry different injected activations. Per-request injection verification
and the sampler-probability mismatch filter remain enabled.

The runtime target is about one day, not a measured guarantee. The original
EasyNLA timing is for another model and a smaller RL budget. `metrics_rank*.jsonl`
records per-stage timing so a Gemma-specific estimate can be made after RL begins.
The sampling budget is not silently shortened to claim a one-day completion.

## Status and artifacts

The authoritative controller status is `status.json`. Logs are in `logs/`, exact
launched argument lists in `commands/`, SFT checkpoints in `sft_av/` and `sft_ar/`,
and RL adapters plus the latest co-trained critic in `rl/`. Evaluation examples
and metrics are written to `rl/evaluation.jsonl`. `rl/best_pair/` retains both
models at the best monitored FVE, including extraction failures at reward -2.
This is checkpoint selection on 128 monitoring examples, not an independent
final test result. Checkpoints are saved every 100 RL updates and at the final
update. Training losses in rank-zero logs are
rank-local; held-out evaluation is on the shared document-disjoint validation
set. Full validation files remain available beyond the 128 monitoring examples.

Inspect remotely with:

```bash
ssh agastyas@athena02.ig32s.ruhr-uni-bochum.de \
  'cat /data/agastyas/nla_cross_model_swap/status.json; tmux list-sessions'
```

The controller refuses a second launch if `PIPELINE_STARTED` exists. A failed
stage records `phase: failed` and prevents later stages from starting; inspect
the corresponding log before resuming. Completion requires the `COMPLETE` marker.

On 2026-10-05 at 01:44 UTC, the first RL launch failed before any update: other
processes occupied approximately 129 GiB on GPUs 4–7 during actor loading. Their
origin has not been established. Both SFT stages and AV merging had completed.
At 01:52 UTC, after verifying that all eight GPUs had no compute processes,
`restart_rl.py` relaunched the original RL command using the final SFT checkpoints.
It neither reruns SFT nor changes the training budget. This controller uses an
exclusive lock and refuses to start over if RL checkpoints already exist.
Read the active attempt's log path from `status.json`; the original `logs/rl.log`
preserves the first failed attempt. Restart records are in `restart_history/`.
The restart controller runs via nohup, so inspect its process as well as tmux.

## Requested pause and resumption

**Verified paused:** the monitor completed at 2026-10-05 04:52:10 UTC
(00:52 EDT), at update **200/2,100**, with all eight ranks at the same boundary
and zero completed updates discarded. Both optimizers have populated state for
every trainable parameter (588 AV tensors and 354 AR tensors). The preserved
snapshot contains 51 hashed files. At 05:09 UTC, all eight H200s reported zero
memory use and zero utilization, with no GPU compute processes or old training
controller remaining. The hourly ETA automation is paused. Latest monitored
held-out FVE is 29.50%, evaluated after update 151 on 128 examples; update 200
has not had a fresh held-out evaluation. Local copies of the pause manifest,
hash list and prepared resume command are in `pause_state/`.

The user requested a pause on October 5 to free Athena02 for other experiments.
`pause_at_checkpoint.py --step 200 --launcher-pid 2936615` waits for the complete
scheduled save, checks that AV, AR, and both optimizer states agree on update 200,
and preserves a snapshot under `pause_snapshots/step_000200`. It then terminates
only the identified training process tree. `PAUSED.json` and `status.json` are
written after verification and shutdown; `PAUSE_REQUESTED.json` alone means the
pause is still pending. The snapshot has SHA-256 hashes, an optimizer summary,
the original command, and a prepared `resume_command.json`. It also records any
completed updates beyond the saved boundary, rather than hiding a rollback.

Do not resume until the user requests it. Before resuming, verify the snapshot
hashes and matching update numbers, retain the current logs and provenance,
and require successful restoration of **both** optimizers. The existing trainer
can warn and fall back to fresh Adam moments on an incompatibility; that fallback
is not acceptable for this paused run. Keep the same SFT reference adapter,
data split, model code, scientific settings, and total 2,100-update budget. The
prepared command resumes at update 200 and fast-forwards the deterministic data
cursor. Initial vLLM weight synchronization is mandatory and already enabled by
the trainer's resume path. Also restore the best-pair selection threshold from
the saved selection record: the current trainer otherwise initializes it to
negative infinity, which could replace a better pre-pause pair with a worse one.

This trainer saves model weights, Adam moments and the completed-update counter,
but not exact generation RNG state. Resumption therefore preserves the learned
state and training schedule, not a bitwise-identical stochastic trajectory.
The pause boundary must remain part of experimental provenance.
