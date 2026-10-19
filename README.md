# Evaluating the groundedness and robustness of natural language autoencoders

This repo contains the code for our work on the groundedness and robustness of natural language autoencoders (NLAs). You can read the [current paper draft here](paper/groundedness-and-robustness-draft.pdf). **The paper is incomplete and is being prepared for submission to a conference.** Experiments, analyses, and the write-up are still in progress.

An NLA consists of an activation verbalizer (AV), which turns a model's activation into text, and an activation reconstructor (AR), which maps that text back into activation space. They are trained to make the reconstruction accurate. Our question is whether this also makes the explanation faithful: does it describe information in the activation, and does that interpretation survive changes to the training setup or the explanation itself?

We approach this with three methods:

1. **A claim-level cycle-consistency test.** We split an explanation into atomic claims and compare the AR's reconstruction of the full explanation with its reconstruction after deleting one claim. The difference between the normalized reconstructions gives a candidate direction for that claim. We project this direction out of the original activation, then measure the decrease in the AV's average token log-probability for the claim while holding the preceding explanation tokens fixed. A drop alone is not enough: removing a broad thematic signal could also suppress a specific claim that the activation never supported. We therefore compare the effect with self-ablation effects for plausible, theme-matched decoys, and test whether other claims' directions produce comparable effects. These controls distinguish evidence for a specific claim from shared thematic sensitivity and overlap between claim directions.

2. **A comparison across warm starts.** We train NLAs from plausible and deliberately implausible teacher explanations while holding the target model, activation data, and subsequent training recipe fixed. We compare checkpoints at similar fraction of variance explained (FVE), so differences in reconstruction quality do not dominate the comparison. We then segment their explanations into claims and use a separate judge to match claims by semantic equivalence at the same level of specificity. Repeated samples from each AV provide a baseline: the question is how much additional disagreement comes from initialization beyond ordinary sampling variation. The evaluation covers both held-out pretraining-like text and out-of-distribution tasks.

3. **Controlled edits to the explanation.** We keep the target activation and AR fixed, edit the AV's explanation, and measure the change in FVE. Removing quoted spans tests how much reconstruction information they carry; paraphrasing and reordering test sensitivity to wording and order. Replacing the AV's source-like quotes with the actual source text tests whether its deviations from the input help or hurt reconstruction. We evaluate these interventions on Qwen2.5-7B, Gemma-3-12B, and Gemma-3-27B, using paired comparisons and document-level bootstrap intervals. Together, the edits help separate reliance on an explanation's semantic claims from reliance on text that resembles the original input.

The [paper-to-code guide](paper/README.md) maps these methods to their implementations, protocols, and saved results.

## Code layout

| Folder | Question or purpose |
| --- | --- |
| [analysis/groundedness_report](analysis/groundedness_report) | Are individual claims grounded? Claim segmentation, ablations, decoy controls, calibration, and evaluation. |
| [analysis/warm_start_robustness](analysis/warm_start_robustness) | Do different warm starts converge? Training code, checkpoint matching, and semantic claim agreement. |
| [analysis/fve_robustness](analysis/fve_robustness) | Which changes to an explanation affect reconstruction? Text edits, AR rescoring, and paired statistical analysis. |
| [analysis/mimic_audit](analysis/mimic_audit) | Earlier groundedness experiments and shared inference helpers. |
| [paper](paper) | Current draft, detailed code map, provenance, and robustness figure/table rendering. |
| [nla](nla), [configs](configs), [tools](tools) | Shared NLA training, configuration, and checkpoint conversion. |
| [experiments/cross_model_swap](experiments/cross_model_swap) | A related Qwen–Gemma swap experiment, separate from the paper's warm-start comparison. |

The experiment folders contain their protocols, commands, and run notes. Some notes describe earlier versions of an experiment; use the paper-to-code guide to find the relevant implementation. This branch collects work developed in separate experiment directories, so its publication commits do not capture every intermediate research step.

## Setup and reproduction

This repo builds on [the original NLA implementation](https://github.com/kitft/natural_language_autoencoders) and [EasyNLA](https://github.com/asherps/EasyNLA). The original library README, including checkpoint links and attribution, is preserved in [README.upstream.md](README.upstream.md). See [the setup guide](docs/setup.md) for the parent training package and [the inference guide](docs/inference.md) for loading and using released NLAs.

Use the dependencies and commands in each experiment folder. The parent package and the vendored EasyNLA checkouts need separate environments: they expose the same Python package name but are not interchangeable. Launch scripts retain historical cluster paths, which need to be adapted to your machine.

Model weights, activation dumps, raw corpora, and API caches are not committed. Full reruns require the external inputs and model access documented by each experiment. The repo includes aggregate robustness statistics, so its table and figures can be rendered without GPU inference or API calls:

```bash
# Requires NumPy and Matplotlib.
python paper/render_robustness.py --output-dir paper/generated
```

The figure renderer was added when assembling this archive and uses the saved estimates and confidence intervals. It is not the original plotting script.

The CPU test suites can be run from the repo root with the research dependencies installed:

```bash
python -m pytest -q analysis/groundedness_report/tests
python -m pytest -q analysis/warm_start_robustness/claim_agreement/tests analysis/warm_start_robustness/fve_matching/tests
PYTHONPATH=analysis python -m pytest -q analysis/tests
```

At the time this archive was assembled, these passed **191 tests and 16 subtests**. This checks the tested code paths; it does not establish that every experiment described in the draft has been completed.

The parent code retains its [Apache-2.0 license](LICENSE). The vendored EasyNLA snapshots retain their own MIT licenses.
