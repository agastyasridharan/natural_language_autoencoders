# Qwen NLA groundedness replication

This directory implements the cycle-consistency groundedness procedure in the
attached report. The frozen machine-readable authority is `preregistration.yaml`.
The approved directional-reconstruction correction is recorded separately in
`audits/v7/reconstruction_metric_amendment.json`; every artifact containing a
reconstruction error carries both hashes. `PREREGISTRATION.md` gives the readable
rationale, `evaluation_registry.yaml` pins the seven alignment and nine
capability replications, and `RUN_STATUS.md` records completed artifacts and
current stopping conditions.

The base V7 file mistakenly substituted Luna for the user's explicitly frozen
Terra model. The pre-outcome correction is recorded separately in
`audits/v7/text_model_correction_amendment.json`. Operational code permits only
`gpt-5.6-terra`; all Luna-generated text and all artifacts derived from it are
archived but inadmissible.

Only one NLA is in scope:

- target: `Qwen/Qwen2.5-7B-Instruct`, layer 20;
- AV: `kitft/nla-qwen2.5-7b-L20-av`;
- AR: `kitft/nla-qwen2.5-7b-L20-ar`.

The exact revisions are in the preregistration. No other checkpoint is selected
after outcomes. The text pipeline uses `gpt-5.6-terra` and reads only
`OPENAI_API_KEY_2`; generic-key and model fallbacks are rejected.

The implementation does not run the report's separate robustness experiments.

## Frozen arithmetic

- Raw activations are stored without normalization. A source activation is unit
  normalized once at measurement time.
- A claim direction is
  `unit(unit(AR(z)) - unit(AR(z without claim)))`.
- Projection is `h' = h - (h.T @ delta_hat) * delta_hat`; `h'` is not normalized
  again.
- Injection is `injection_scale * h` or `injection_scale * h'`.
- Claim scores are per-token mean log probabilities under teacher forcing.
- Candidate overlap is
  `abs(abs(h @ delta_decoy) - abs(h @ delta_real))`.
- Under the approved pre-outcome amendment, reconstruction matching uses
  directional squared L2 error
  `||unit(h) - unit(AR(z))||^2 = 2(1 - cosine(h, AR(z)))`.
  A non-finite or norm-at-most-`1e-12` input fails closed.
- Decoy activation support is one-sided. Large negative support is not rejected.
- Ownership uses `1.5 * max(0, max(off_diagonal_effects))`.
- Exactly 20 controls are selected from 64 generated candidates using the frozen
  overlap ladder `[.10, .20, .35, .50, .75, 1.00]` and type-7 linear quantiles.
- The specificity gate precedes attribution. Specificity failure proceeds to the
  seeded/confabulated gates regardless of attribution.
- The report leaves the weak-theme/high-support residual branch undefined; it is
  `unresolved` here rather than forced into a substantive category.

The measurement injector scans each live token sequence for the sidecar marker
and validates both neighbors inside the embedding hook. No precomputed injection
position is used.

## Study scale and inference

FineWeb supplies 2,000 unique documents: 500 calibration and 1,500 evaluation.
The staged sample consists of 80 deterministic 100-row blocks spread across the
14,868,862-row `sample-10BT` split. The fetch is permitted only while hub `main`
equals the frozen revision, and every response is archived and hashed.

Each of the 16 OOD families contributes 50 analyzable Qwen trajectories and at
most four assistant-language activation positions per trajectory. These are
public Qwen replications, not the original proprietary Claude transcripts.

CyberGym is excluded by explicit user instruction. The same instruction removes
three named Cyber-related cases from the ToolEmu ranking and replaces them with
the next three ranks; see `audits/v7/toolemu_noncyber_scope_amendment.json`.
Because this departs from the original 16-family scope, the preregistered
16-family macro estimator remains disabled unless the scope is revised again.

The AV produces one keyed sample per activation at temperature 1.0 with a
500-new-token limit. An emitted sidecar injection marker is a hard failure.
Non-marker CJK language mixing is retained and reported; protocol v1's broader
rule was retired before segmentation or numerical measurement, with all v1 AV
artifacts archived and excluded.

At most three claims per activation are sampled uniformly. Their inverse
inclusion probabilities recover the all-claim estimand, while all other real
claims remain in ownership comparisons.

## Text audit gate

Terra performs exact claim/connective segmentation, claim taxonomy, generation of
64 candidate decoys, source-absence judgments, and splice-naturalness judgments.
The absence source is the exact causal target-model context through the
activation token, never later text or later tool results.

Before expansion, manually review:

- 200 segmentations: 100 FineWeb and 100 OOD;
- 400 candidate decoys: 200 FineWeb and 200 OOD.

The OOD half is allocated across all 16 families. Exact-tiling failures must be
zero, atomic-claim acceptability at least 90%, and human/automated agreement at
least 90% for both decoy gates. Failure requires a new protocol version and
relabeling; groundedness outcomes cannot be used to tune prompts.

The FineWeb half is prepared with:

```bash
python -m analysis.groundedness_report.prepare_text segment-fineweb-audit \
  --av-dir /data/nla/qwen-v2/fineweb/av \
  --manifest /data/nla/qwen-v2/fineweb/gold/manifest.json \
  --out /data/nla/qwen-v2/audit/fineweb

python -m analysis.groundedness_report.prepare_text decoys-fineweb-audit \
  --records-dir /data/nla/qwen-v2/audit/fineweb/records \
  --manifest /data/nla/qwen-v2/fineweb/gold/manifest.json \
  --source-documents /data/nla/qwen-v2/fineweb/source/documents.jsonl \
  --out /data/nla/qwen-v2/audit/fineweb-decoys
```

The second command writes blinded CSVs and a separate answer key.

After all 16 OOD trajectory families have been assembled and verbalized, make
the family-balanced OOD audit half with:

```bash
python -m analysis.groundedness_report.prepare_study_text segment-ood-audit \
  --av-dir OOD_AV --manifest OOD_GOLD/manifest.json --out OOD_AUDIT

python -m analysis.groundedness_report.prepare_study_text decoys-ood-audit \
  --records-dir OOD_AUDIT/records --manifest OOD_GOLD/manifest.json \
  --out OOD_AUDIT
```

The 100 segmentation rows are allocated 6 or 7 per family and use distinct
transcript clusters. The 25 audited claims are allocated 1 or 2 per family;
eight blinded candidates per claim give 200 OOD decoy rows. Full text expansion
requires passing result files for both the FineWeb and OOD audit halves.

## Full-run artifacts

The numerical workflow expects:

- `runs/RUN_ID/records.jsonl`, with exact claim/connective character tiling;
- `decoys/RUN_ID/IDX_CLAIM.terra.json`, for every sampled claim;
- `gold/gold.npy` and `gold/manifest.json`, with raw target activations;
- `split.json`, `claim-sample.json`, and a unit `h_ref.npy`;
- a calibration artifact frozen only from FineWeb and its deranged-activation
  controls.

Create the document split, claim sample, and reference:

```bash
python -m analysis.groundedness_report.cli make-split \
  --records runs/RUN_ID/records.jsonl --out split.json \
  --calibration-fraction 0.25 --seed 20260827

python -m analysis.groundedness_report.cli make-claim-sample \
  --records runs/RUN_ID/records.jsonl --split split.json \
  --out claim-sample.json --seed 20260827

python -m analysis.groundedness_report.cli make-reference \
  --gold-dir gold --split split.json --out h_ref.npy

python -m analysis.groundedness_report.cli make-negative-map \
  --records runs/RUN_ID/records.jsonl --split split.json \
  --out negative-map.json --seed 20260827
```

An OOD run uses an explicit evaluation-only split; it does not relabel any OOD
activation as calibration:

```bash
python -m analysis.groundedness_report.cli make-evaluation-split \
  --records OOD_RUN/records.jsonl --out OOD_RUN/split.json
```

The frozen calibration's model, protocol, reconstruction-metric amendment,
reference vector, and artifact fingerprint must match. Its FineWeb run ID and
split hash are retained as calibration provenance and are not required to equal
the OOD run's identifiers.

The 0.25 split is exact for 500/2,000. `preflight` checks document leakage,
tiling, the claim sample, decoy completeness, the reference vector, and optional
CUDA availability.

AR and AV measurement are separate processes:

```bash
python -m analysis.groundedness_report.gpu_measure ar qwen7b \
  --run RUN_ID --runs-dir runs --decoys-dir decoys --gold-dir gold \
  --split split.json --claim-sample claim-sample.json \
  --partition calibration --out measurements

python -m analysis.groundedness_report.gpu_measure av qwen7b \
  --run RUN_ID --runs-dir runs --decoys-dir decoys --gold-dir gold \
  --split split.json --claim-sample claim-sample.json \
  --partition calibration --reference-vector h_ref.npy \
  --negative-map negative-map.json --out measurements
```

The amended run must use a fresh output root (for example,
`measurements-directional-v1`). Do not force-overwrite pre-amendment AR or AV
measurement caches; those remain preserved as incompatible diagnostics.

Calibration AV measures every text-valid control. Freeze the preregistered
thresholds from these readings plus the deranged FineWeb control artifact:

```bash
python -m analysis.groundedness_report.cli freeze-calibration \
  --measurements calibration-measurements.jsonl \
  --model-key qwen7b --run-id RUN_ID --split split.json \
  --reference h_ref.npy --out calibration.json
```

Evaluation AV requires that frozen calibration. Classification and intervals:

```bash
python -m analysis.groundedness_report.cli classify \
  --measurements evaluation-measurements \
  --calibration calibration.json --out verdicts.jsonl

python -m analysis.groundedness_report.cli clustered-ci \
  --verdicts verdicts.jsonl --kind fineweb --replicates 10000 \
  --seed 20260827 --out fineweb-ci.json
```

OOD intervals use `--kind ood-macro`, resample whole transcripts, and compute an
equal-weight macro-average across the 16 families. Claim-i.i.d. intervals are
forbidden.

## Verification

```bash
python -m unittest discover -s analysis/groundedness_report/tests -v
python -m compileall -q analysis/groundedness_report
```

Each phase writes atomic per-item artifacts and refuses stale protocol or
reconstruction-metric-amendment hashes.
