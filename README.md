# Evaluating the groundedness and robustness of natural language autoencoders

This repo contains the code for our work on the groundedness and robustness of natural language autoencoders (NLAs). You can read the [current paper draft here](paper/groundedness-and-robustness-draft.pdf). **The paper is incomplete and is being prepared for submission to a conference.** Experiments, analyses, and the write-up are still in progress.

An NLA translates a language model's activation into a text explanation, then reconstructs the activation from that explanation. The training objective rewards accurate reconstruction. But an explanation can help reconstruct an activation without being a faithful account of what the activation represents. We study this gap in three ways:

1. **Groundedness.** When an explanation makes a specific claim, is that claim supported by the activation? We recover a candidate claim direction through the reconstructor, ablate it, and measure the change in the verbalizer's support for the claim. Theme-matched decoys help distinguish support for a specific claim from sensitivity to its broader topic.
2. **Warm-start robustness.** Do NLAs trained from plausible and deliberately implausible initial explanations converge on the same claims? We compare their explanations at matched reconstruction quality, using repeated samples from the same model as a baseline for sampling variation.
3. **Explanation-edit robustness.** Does reconstruction depend on what an explanation means, or on its particular wording? We remove, paraphrase, reorder, and replace quoted spans, then measure the change in reconstruction quality.

The draft is ahead of the completed experiments in some places. In particular, **Tables 1–3 contain illustrative placeholder numbers, not measured results**; their [provenance is recorded here](docs/research/groundedness_results_illustrative/README.md). The explanation-edit robustness table is backed by [saved experimental statistics](analysis/fve_robustness/results/dashboard_inference.json). The [paper-to-code guide](paper/README.md) distinguishes implemented analyses, historical run records, and work that remains unfinished.

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
