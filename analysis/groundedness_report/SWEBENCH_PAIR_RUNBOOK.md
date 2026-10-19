# Paired SWE-bench Verified / Destructive Coding runbook

This is the frozen execution order for the two official-container families.  A
one-instance paired smoke is a mandatory gate before the 50+50 run.  None of
the commands below should be started while another process occupies the chosen
GPU.

## Frozen inputs and storage

- Protocol: `20a87044c5706a3de0d9deb098a740739c979a6d7a9ae21f010eb850cd329637`
- Alignment tasks (50):
  `/data/agastyas/nla_groundedness_qwen_v2/ood/tasks/swebench-v7/alignment_destructive_coding.jsonl`
  (`9d3f86abb81ba8bdb16922fe7a73e280acbd774c306719df661933e216602191`)
- Capability tasks (50):
  `/data/agastyas/nla_groundedness_qwen_v2/ood/tasks/swebench-v7/capability_swebench_verified.jsonl`
  (`bec1d0d3363b107f34f5ed75702e0b34d294ff19536025da0545373a2f2dbcd7`)
- Deterministic paired smoke instance: `astropy__astropy-14309`
- Podman graphroot: `/data/agastyas/podman-sweb7/graphroot`
- Podman runroot: `/data/agastyas/podrun-sweb7`
- Podman image-copy tmpdir: `/data/agastyas/podtmp-sweb7`
- Podman storage option: `overlay.ignore_chown_errors=true`

The short runroot is required.  Podman 4.9.3 rejects a runroot longer than 50
bytes; the otherwise natural path below the experiment's `ood/` directory is
too long.  Both chosen roots are on `/data`, not the 10-GiB home filesystem.
The explicit tmpdir is also required: without `TMPDIR`, Podman reports
`/var/tmp` as `imageCopyTmpDir`, so cold pulls can spill tens of GiB onto the
system volume even when graphroot itself is on `/data`.

## CPU-only preflight

Run this before reserving a GPU.  It validates all 100 task hashes, exact
instance-level pairing, official image references, the explicit rootless
Podman store, and writes the immutable command plan.  It does not pull an
image, start a container, or import the model.

```bash
cd /data/agastyas/nla_groundedness_qwen_v2/code
PYTHONPATH=/data/agastyas/nla_groundedness_qwen_v2/code \
/data/agastyas/Miniconda3/bin/python -m \
  analysis.groundedness_report.swebench_pair \
  --alignment-tasks /data/agastyas/nla_groundedness_qwen_v2/ood/tasks/swebench-v7/alignment_destructive_coding.jsonl \
  --capability-tasks /data/agastyas/nla_groundedness_qwen_v2/ood/tasks/swebench-v7/capability_swebench_verified.jsonl \
  --smoke-output /data/agastyas/nla_groundedness_qwen_v2/ood/swebench-smoke-v7 \
  --full-output /data/agastyas/nla_groundedness_qwen_v2/ood/swebench-run-v7 \
  --podman-root /data/agastyas/podman-sweb7/graphroot \
  --podman-runroot /data/agastyas/podrun-sweb7 \
  --podman-tmpdir /data/agastyas/podtmp-sweb7 \
  --check-podman \
  --write-plan /data/agastyas/nla_groundedness_qwen_v2/ood/swebench-plan-v7.json
```

Before any GPU command, verify that the warm-start trainer has ended and that
the selected device is free:

```bash
nvidia-smi --query-gpu=index,uuid,memory.used,memory.total,utilization.gpu --format=csv,noheader
ps -u agastyas -o pid=,etimes=,stat=,cmd= | grep -E '(train_rl_vllm|gpu_ood_swebench)' | grep -v grep
```

## Environment for every GPU process

```bash
cd /data/agastyas/nla_groundedness_qwen_v2/code
export PYTHONPATH=/data/agastyas/nla_groundedness_qwen_v2/code
export HF_HUB_CACHE=/data/agastyas/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TMPDIR=/data/agastyas/podtmp-sweb7
export http_proxy=http://www-cache.rz.ruhr-uni-bochum.de:80
export https_proxy=$http_proxy
export no_proxy=.rub.de,.ruhr-uni-bochum.de,localhost,127.0.0.1,::1
```

The proxy is available only to Podman's image pull.  Every task container is
started with `--network none`, `CUDA_VISIBLE_DEVICES=` and
`NVIDIA_VISIBLE_DEVICES=void`.  The model process remains outside the
container on the single GPU selected by the command.  The storage option is
required because four official Matplotlib images contain ownership IDs beyond
Athena's 65,536-entry rootless subordinate-ID range.  Podman collapses those
otherwise-unmappable ownership IDs while unpacking; file contents and the
immutable image identity are unchanged.

Malformed Qwen tool syntax is retained as model behavior.  The runner records
the raw response and exact parse error, returns one fixed corrective user
message, and consumes one of the same 20 assistant turns without executing a
shell command.  A trajectory that never produces a valid final answer still
fails at the frozen turn ceiling.  The output audit checks the recorded
invalid-call count and rejects any invalid attempt that also executed a tool.

## Mandatory paired smoke

Run these sequentially on one free GPU.  Both explicitly select the same task;
there is no dependence on file order or shard hashing.

```bash
CUDA_VISIBLE_DEVICES=0 /data/agastyas/Miniconda3/bin/python -m \
  analysis.groundedness_report.gpu_ood_swebench \
  --tasks /data/agastyas/nla_groundedness_qwen_v2/ood/tasks/swebench-v7/alignment_destructive_coding.jsonl \
  --out /data/agastyas/nla_groundedness_qwen_v2/ood/swebench-smoke-v7 \
  --podman-root /data/agastyas/podman-sweb7/graphroot \
  --podman-runroot /data/agastyas/podrun-sweb7 \
  --podman-tmpdir /data/agastyas/podtmp-sweb7 \
  --task-id astropy__astropy-14309
```

```bash
CUDA_VISIBLE_DEVICES=0 /data/agastyas/Miniconda3/bin/python -m \
  analysis.groundedness_report.gpu_ood_swebench \
  --tasks /data/agastyas/nla_groundedness_qwen_v2/ood/tasks/swebench-v7/capability_swebench_verified.jsonl \
  --out /data/agastyas/nla_groundedness_qwen_v2/ood/swebench-smoke-v7 \
  --podman-root /data/agastyas/podman-sweb7/graphroot \
  --podman-runroot /data/agastyas/podrun-sweb7 \
  --podman-tmpdir /data/agastyas/podtmp-sweb7 \
  --task-id astropy__astropy-14309
```

Run the fail-closed smoke audit by repeating the CPU preflight command with:

```bash
  --audit-output /data/agastyas/nla_groundedness_qwen_v2/ood/swebench-smoke-v7 \
  --audit-expected-per-family 1
```

Do not continue unless it reports `output_audit.status = "passed"`.  The audit
checks exact pairing, complete trajectories, one call at most per turn,
contiguous tool-call records, successful final `git status`/diff capture, one
to four finite float32 activation vectors with matching hashes, no task or
cleanup failures, and the same immutable image ID in both conditions.

## Full 50+50 run

After the smoke passes, start these four commands concurrently, one per free
GPU (normally separate `tmux` panes).  The two shards in each family are
deterministic, disjoint, and exhaustive.  Per-family shard metadata filenames
are distinct, so the four writers safely share one output root.

```bash
CUDA_VISIBLE_DEVICES=0 /data/agastyas/Miniconda3/bin/python -m analysis.groundedness_report.gpu_ood_swebench --tasks /data/agastyas/nla_groundedness_qwen_v2/ood/tasks/swebench-v7/alignment_destructive_coding.jsonl --out /data/agastyas/nla_groundedness_qwen_v2/ood/swebench-run-v7 --podman-root /data/agastyas/podman-sweb7/graphroot --podman-runroot /data/agastyas/podrun-sweb7 --podman-tmpdir /data/agastyas/podtmp-sweb7 --num-shards 2 --shard-index 0
```

```bash
CUDA_VISIBLE_DEVICES=1 /data/agastyas/Miniconda3/bin/python -m analysis.groundedness_report.gpu_ood_swebench --tasks /data/agastyas/nla_groundedness_qwen_v2/ood/tasks/swebench-v7/alignment_destructive_coding.jsonl --out /data/agastyas/nla_groundedness_qwen_v2/ood/swebench-run-v7 --podman-root /data/agastyas/podman-sweb7/graphroot --podman-runroot /data/agastyas/podrun-sweb7 --podman-tmpdir /data/agastyas/podtmp-sweb7 --num-shards 2 --shard-index 1
```

```bash
CUDA_VISIBLE_DEVICES=2 /data/agastyas/Miniconda3/bin/python -m analysis.groundedness_report.gpu_ood_swebench --tasks /data/agastyas/nla_groundedness_qwen_v2/ood/tasks/swebench-v7/capability_swebench_verified.jsonl --out /data/agastyas/nla_groundedness_qwen_v2/ood/swebench-run-v7 --podman-root /data/agastyas/podman-sweb7/graphroot --podman-runroot /data/agastyas/podrun-sweb7 --podman-tmpdir /data/agastyas/podtmp-sweb7 --num-shards 2 --shard-index 0
```

```bash
CUDA_VISIBLE_DEVICES=3 /data/agastyas/Miniconda3/bin/python -m analysis.groundedness_report.gpu_ood_swebench --tasks /data/agastyas/nla_groundedness_qwen_v2/ood/tasks/swebench-v7/capability_swebench_verified.jsonl --out /data/agastyas/nla_groundedness_qwen_v2/ood/swebench-run-v7 --podman-root /data/agastyas/podman-sweb7/graphroot --podman-runroot /data/agastyas/podrun-sweb7 --podman-tmpdir /data/agastyas/podtmp-sweb7 --num-shards 2 --shard-index 1
```

On resume, complete trajectories and existing model-behavior failures are
preserved.  Use `--retry-infrastructure-failures` only for saved Podman or
timeout failures; the runner first moves their prior tool records under
`diagnostics/pre-infrastructure-retry-tools/`.  Never retry a turn-ceiling
failure: apply the frozen next-SHA-ranked-task replacement rule.  Image pulls
share a filesystem lock and the two conditions reuse local images.

After applying the frozen replacement rule, build
`ood/swebench-canonical-run-v7` with `swebench_finalize.py`.  Audit that
canonical output with `--audit-expected-per-family 50`, then assemble:

```bash
PYTHONPATH=/data/agastyas/nla_groundedness_qwen_v2/code \
/data/agastyas/Miniconda3/bin/python -m \
  analysis.groundedness_report.gpu_ood_static assemble \
  --partials /data/agastyas/nla_groundedness_qwen_v2/ood/swebench-canonical-run-v7 \
  --out /data/agastyas/nla_groundedness_qwen_v2/ood/swebench-gold-v7
```

The expected assembled state is 100 trajectories and at most 400 raw,
unnormalized layer-20 activation rows, split evenly across the two families.

The 2026-09-09 run reached exactly 50 pairs and 400 activations.  Its strict
audit passed.  The gold hashes are
`6f8ca98e2656beffbfe648eace1e2f230b732bf7e37f9f05ba8bdf9128896636`
(`gold.npy`) and
`dffffd12944e7ec6818ce4dc6361ff30917c10093f0f1b4e20364e6ba6fbcff5`
(`manifest.json`).  The exclusion and replacement sequence is recorded under
`audits/v7/swebench/`; retry-selected trajectories are diagnostic only.

## Time and disk planning

There is no SWE-bench smoke timing yet, so the GPU estimate remains explicitly
provisional.  A reasonable planning range is 3--15 minutes per trajectory,
or 5--25 aggregate GPU-hours for 100 trajectories.  With four GPUs, expect
roughly 1.5--7 hours of compute wall time after image availability, plus about
20--90 minutes for cold image pulls through the Athena proxy.  The protocol's
hard shell-only ceiling is 20 calls times 120 seconds per trajectory; that is a
failure-bound, not an ETA.  After the paired smoke, use
`25 * mean_smoke_trajectory_minutes` as the four-GPU central estimate and keep
a 1.5x buffer because one Astropy instance is not representative of all repos.

Registry manifests for the exact frozen 50-image set were inspected on
2026-08-28 without pulling layers.  They contain 121 unique compressed layers
totaling 27,234,269,585 bytes (25.36 GiB); without digest deduplication their
sum would be 68,916,893,019 bytes (64.18 GiB).  The smoke image is
1,136,537,447 compressed bytes (1.058 GiB).  Overlay stores expanded layers,
so reserve 150 GiB for the Podman graphroot plus 2 GiB for trajectories,
tool logs, diffs, activation arrays, and assembled copies.  Athena currently
has approximately 8.7 TB free, so disk capacity is not a blocker.
