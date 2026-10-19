# Exact resume state after GPU pause

Paused at: 2026-08-27 (America/New_York)

Remote live root:
`/data/agastyas/nla_groundedness_qwen_v2`

Frozen protocol SHA-256:
`20a87044c5706a3de0d9deb098a740739c979a6d7a9ae21f010eb850cd329637`

## Atomic state at pause

- No groundedness-report GPU process remains running.
- Unrelated user-owned NLA training processes were not stopped or modified.
- No `.tmp` file exists under the remote live root.
- FineWeb is complete through raw activation extraction and AV generation:
  2,000 raw activations and 2,000 AV reads.
- The seven static OOD families are complete through AV generation:
  350 trajectories, 1,400 raw activations, and 1,400 AV reads.
- The authoritative BrowseComp run is
  `ood/browsecomp-run-v7-search4`. It contains 48 complete trajectories and
  192 raw activations. There are no current failure artifacts.
- BrowseComp is missing exactly `browsecomp-1161` and `browsecomp-1222`.
  Both belong to deterministic shard 1. Rerunning shard 1 skips its five
  complete items and computes only these two missing tasks.
- BrowseComp `search1`, `search2`, `search3`, and the first portion of
  `search4` affected by throttling/OOM are diagnostics. Only complete per-task
  artifacts in `search4` are authoritative. The current implementation archives
  DuckDuckGo throttling responses and routes both HTTP 202 and 403 to the frozen
  Bing fallback.
- The paired SWE-bench Verified / Destructive Coding task artifacts exist, but
  the official-container smoke test and all trajectories remain unrun.  The
  CPU-only Athena preflight established a working rootless overlay store at
  `/data/agastyas/podman-sweb7/graphroot` with the required short runroot
  `/data/agastyas/podrun-sweb7` and image-copy tmpdir
  `/data/agastyas/podtmp-sweb7`; see `SWEBENCH_PAIR_RUNBOOK.md`.
- Numerical AR/AV claim measurement, calibration thresholds, and category
  percentages have not been generated.

All writers use atomic per-item replacement. The interrupted BrowseComp item
left no temporary or partial result, so no repair or deletion is needed.

## Resume BrowseComp

The public web tools require a local HTTP CONNECT proxy and SSH reverse tunnel.
From the local repository, start the proxy in one terminal:

```bash
python -m analysis.groundedness_report.http_connect_proxy --port 18080
```

In a second terminal:

```bash
ssh -N -R 127.0.0.1:18080:127.0.0.1:18080 \
  agastyas@athena01.ig32s.ruhr-uni-bochum.de
```

On Athena, choose one free GPU and run:

```bash
cd /data/agastyas/nla_groundedness_qwen_v2/code
CUDA_VISIBLE_DEVICES=GPU_INDEX \
HF_HUB_CACHE=/data/agastyas/hf_cache \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
PYTHONPATH=/data/agastyas/nla_groundedness_qwen_v2/code \
HTTPS_PROXY=http://127.0.0.1:18080 \
HTTP_PROXY=http://127.0.0.1:18080 \
/data/agastyas/Miniconda3/bin/python -m \
  analysis.groundedness_report.gpu_ood_browser \
  --tasks /data/agastyas/nla_groundedness_qwen_v2/ood/tasks/browsecomp-v7/capability_browsecomp.jsonl \
  --out /data/agastyas/nla_groundedness_qwen_v2/ood/browsecomp-run-v7-search4 \
  --num-shards 4 --shard-index 1
```

The expected terminal state is 50 complete trajectories, 200 activations, zero
failure files, and no exact marker failures. Then assemble:

```bash
cd /data/agastyas/nla_groundedness_qwen_v2/code
PYTHONPATH=/data/agastyas/nla_groundedness_qwen_v2/code \
/data/agastyas/Miniconda3/bin/python -m \
  analysis.groundedness_report.gpu_ood_static assemble \
  --partials /data/agastyas/nla_groundedness_qwen_v2/ood/browsecomp-run-v7-search4 \
  --out /data/agastyas/nla_groundedness_qwen_v2/ood/browsecomp-gold-v7
```

Run AV generation over the assembled 200 activations. With one free GPU:

```bash
cd /data/agastyas/nla_groundedness_qwen_v2/code
CUDA_VISIBLE_DEVICES=GPU_INDEX \
HF_HUB_CACHE=/data/agastyas/hf_cache \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
PYTHONPATH=/data/agastyas/nla_groundedness_qwen_v2/code \
/data/agastyas/Miniconda3/bin/python -m \
  analysis.groundedness_report.gpu_verbalize \
  --gold-dir /data/agastyas/nla_groundedness_qwen_v2/ood/browsecomp-gold-v7 \
  --out /data/agastyas/nla_groundedness_qwen_v2/ood/browsecomp-av-v7
```

## Next work after BrowseComp

1. Follow `SWEBENCH_PAIR_RUNBOOK.md`: run the explicit paired
   `astropy__astropy-14309` smoke and its fail-closed artifact audit.  Do not
   launch all 100 paired trajectories until it passes.
2. Continue only faithful public environments. Do not substitute direct chat
   for ToolEmu, tau2-bench, Terminal-Bench 2, FinanceAgent, or CyberGym.
3. Keep Humanity's Last Exam paused until the `cais/hle` terms are accepted for
   the Athena Hugging Face account.
4. Do not freeze thresholds or generate category outcomes until the AR
   reconstruction-metric issue in `RUN_STATUS.md` is resolved.
5. Complete both blinded human-audit halves before full Terra expansion.

## Integrity verification

The pause backup directory contains a SHA-256 manifest relative to the live
root. Verify the live tree from its parent directory with:

```bash
cd /data/agastyas/nla_groundedness_qwen_v2
sha256sum --check /data/agastyas/nla_groundedness_backups/20260827T172652Z/LIVE_TREE.sha256
```

Do not restore over the live tree without first preserving any newer artifacts.
The remote copied tree and archive are independent restore sources; the local
archive is an off-host source.
