# Code for Evaluating the Groundedness and Robustness of Natural Language Autoencoders

Read the [current paper draft](groundedness-and-robustness-draft.pdf). **The paper
is incomplete and is being prepared for submission to a conference.** The PDF is
the supplied draft, uploaded without changes.

This directory maps the draft to the code available in this branch: groundedness,
warm-start robustness, explanation-edit robustness, shared inference, statistical
analysis, and tests. The earlier Qwen-to-Gemma swap is retained as a related experiment;
it is not the warm-start experiment described in the manuscript.

**Status matters:** this is a code archive, not a claim that every manuscript
result has been completed. The groundedness population, controlled-pair AUROC,
and SAE tables in the supplied manuscript match the explicitly invented numbers
in [the illustrative-results notes](../docs/research/groundedness_results_illustrative/README.md).
Those numbers are not empirical results. The controlled-pair benchmark and SAE
audit are described there as proposed studies; a completed implementation/results
chain for those tables was not located. Do not cite those tables as reproduced
findings from this repository.

## Paper-to-code map

| Manuscript component | Implementation and supporting files | Scope/status |
| --- | --- | --- |
| Claim segmentation and exact token/character spans | [groundedness text pipeline](../analysis/groundedness_report/text_pipeline.py), [text preparation](../analysis/groundedness_report/prepare_text.py), [legacy segmenter](../analysis/mimic_audit/athena/repl/segment_run.py) | Current frozen protocol and earlier implementation are both preserved. |
| Claim directions, ablation and teacher-forced rescoring | [GPU measurement](../analysis/groundedness_report/gpu_measure.py), [legacy cycle consistency](../analysis/mimic_audit/athena/repl/cycle_consistency.py), [inference wrapper](../analysis/mimic_audit/athena/nla_inference.py) | Includes normalized AR differences, projection ablation, injection and scoring. |
| Matched decoys, calibration, classification and uncertainty | [groundedness package and commands](../analysis/groundedness_report/README.md), [preregistration](../analysis/groundedness_report/preregistration.yaml), [protocol amendments](../analysis/groundedness_report/audits/v7), [saved run status](../analysis/groundedness_report/RUN_STATUS.md) | Includes data preparation and audit gates. Saved status is historical, not a live cluster check. |
| OOD evaluation families | [evaluation registry](../analysis/groundedness_report/evaluation_registry.yaml), `build_*`, `gpu_ood_*` and `*_finalize.py` in the groundedness package | Public Qwen replications with documented exclusions; not proprietary original-model trajectories. |
| Groundedness population / controlled-pair / SAE tables | [illustrative-results provenance](../docs/research/groundedness_results_illustrative/README.md) | Placeholder numbers; no completed empirical reproduction is claimed. |
| Standard versus implausible warm starts | [modified training source](../analysis/warm_start_robustness/EasyNLA), [matched recipe](../analysis/warm_start_robustness/EasyNLA/docs/qwen25_7b_ufw20k_plausible_matched.md), [run manifest](../analysis/warm_start_robustness/RUN_MANIFEST.yaml) | Includes teacher generation, SFT/RL, configs, continuation and checkpoint archiving. Distinguish the original unmatched public reference from the later matched recipe. |
| Cross-initialization and within-model agreement | [claim agreement](../analysis/warm_start_robustness/claim_agreement/README.md), [scoring protocol](../analysis/warm_start_robustness/claim_agreement/SCORING_README.md) | Keyed AV sampling, segmentation, semantic matching and statistics. |
| Matching checkpoints by reconstruction quality | [exact-checkpoint FVE evaluation](../analysis/warm_start_robustness/fve_matching/README.md) | Historical paired AV/AR checkpoints; preserves the evaluation/update indexing distinction. |
| Quote removal, paraphrase and reordering | [robustness code](../analysis/fve_robustness), [arm definitions](../analysis/fve_robustness/build_experiments_dashboard.py) | Main manuscript arms: `exp1`, `exp3a`, `exp3b`, `exp4b`, `exp5a`, `exp5b`, `exp5c`. Includes additional controls. |
| Source-quote replacement appendix | [parser and transformations](../analysis/fve_robustness/exp6_edits.py), [arm construction](../analysis/fve_robustness/exp6_build_arms.py), [controls](../analysis/fve_robustness/exp6_build_controls.py) | `exp6a` includes the read token; `exp6b` excludes it while keeping the same activation target. |
| AR rescoring, FVE, confidence intervals and tests | [baseline](../analysis/fve_robustness/fve_ufw.py), [rescoring](../analysis/fve_robustness/rescore.py), [paired inference](../analysis/fve_robustness/dashboard_inference.py), [statistics](../analysis/fve_robustness/inference_stats.py) | Includes document bootstrap and multiple-testing correction. |
| Robustness table and figures | [measured summary](../analysis/fve_robustness/results/dashboard_inference.json), [rendering helper](render_robustness.py) | Summary matches the manuscript robustness table. Renderer added for this archive; original PDF-generating source was not located. It reuses estimates and intervals, not fabricated or resampled values. |
| Related cross-model swap | [Qwen–Gemma experiment](../experiments/cross_model_swap/README.md) | Separate experiment, recorded as paused at update 200 on October 5. |

## What is included

`SOURCE_MANIFEST.json` records the original SHA-256 and size of each selected
source/config/document/summary file, and the workspace's base commit. The live
workspace had uncommitted research code, so the base commit alone is insufficient
provenance. The two EasyNLA copies are distinct, vendored snapshots with their
upstream MIT licenses. `WARMSTART_UPSTREAM.json` and the swap's `UPSTREAM.json`
identify their upstream origins. All actual Python/shell/JavaScript source in the
selected experiment trees is included, apart from runtime cache directories.

Shared original NLA code and deployment tools are included. Earlier mimic-audit
and cycle-test tools are retained for dependencies and methodological history;
the groundedness V7 protocol remains the authority for its own study.

Raw corpora, activation arrays, model/optimizer weights, API responses, credentials,
and large per-example generated dashboards are excluded. Aggregate robustness
summaries needed by the offline artifact tests are included. Full GPU/API reruns
require the documented external inputs, model access, dependencies and hardware.
Private checkpoint/archive references in historical status files do not grant
public access to those artifacts.

## Local validation and use

Use a separate environment for the parent `nla` package and for each EasyNLA
training checkout: they expose the same Python package name and are not
interchangeable. Existing launch scripts retain their historical Athena paths;
adapt deployment paths deliberately before running them. Model names, prompt
versions and protocol differences are preserved, not silently substituted.

CPU validation from the repository root (with the project/research dependencies):

```bash
python -m pytest -q analysis/groundedness_report/tests
python -m pytest -q analysis/warm_start_robustness/claim_agreement/tests analysis/warm_start_robustness/fve_matching/tests
PYTHONPATH=analysis python -m pytest -q analysis/tests
```

These passed at publication: **105**, **44 (+16 subtests)**, and **42** tests,
respectively. All packaged Python files were syntax checked. No GPU training,
paid API generation, or cluster experiments were run for publication.

Generate the two robustness figures and the manuscript table values locally:

```bash
python paper/render_robustness.py --output-dir paper/generated
```

The helper needs NumPy and Matplotlib and writes the two PDF names referenced by
the manuscript, PNG previews, a table CSV, and a source-hash receipt. It uses the
saved measured summary and does not generate the illustrative groundedness tables.

Two archive-only portability fixes delay loading `ufw_docs.json` in
`exp6_complete.py` and `exp7_cluster.py` until document text is actually needed.
Missing corpora still raise an error during an experiment; importing pure helpers
for unit tests does not require a dataset. `PUBLICATION_CHANGES.json` records these
changes relative to the original source hashes.
