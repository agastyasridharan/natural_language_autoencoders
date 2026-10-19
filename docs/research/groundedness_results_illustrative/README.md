# Revised groundedness results

The main section is now about 1,000 words including table content and LaTeX commands, down from about 2,900. It retains three findings and three tables. Optional target-model interventions, filter ablations, and implementation discussion have been removed from the main text. The detailed validation appendix remains separate and is not included in the shortened PDF.

All outcome numbers and confidence intervals are invented placeholders. The controlled-pair experiment and SAE audit are proposed studies, not completed experiments. Replace the interpretations as well as the numbers when results arrive.

## Files

- `results.tex`: insert before the manuscript's Robustness section.
- `preview.tex`: standalone main-section preview.
- `validation_appendix.tex`: detailed protocol notes retained separately; optional extensions are not claims of completed work.
- Rendered preview: `output/pdf/groundedness_results_illustrative.pdf` at the repository root.

## Writing references

The revision follows the finding--comparison--interpretation structure in:

- [Li et al., Do Activation Verbalization Methods Convey Privileged Information?, §5.2](https://arxiv.org/html/2509.13316v4#S5.SS2).
- [Atanasova et al., Faithfulness Tests for Natural Language Explanations, §3.1, ACL 2023](https://aclanthology.org/2023.acl-short.25/).
- [Hewitt and Liang, Designing and Interpreting Probes with Control Tasks, §§3.5 and 4.2, EMNLP 2019](https://aclanthology.org/D19-1275/).

These are writing references, not evidence for the synthetic findings. No wording or numerical results was copied from them.

## Interpretation of the synthetic results

- Specificity passes for 50% of FineWeb claims: 31% grounded plus 19% entangled. Seeded claims are more common than confabulated claims, suggesting unsupported detail within a supported topic is the larger problem detected by this test.
- Correcting for decoys raises AUROC from .73 to .82 and lowers false positives from 24% to 11%. Claim deletion reaches .72 AUROC. The useful addition is the comparison against plausible alternatives.
- SAE evidence distinguishes exact from thematic support: seeded claims have 83% thematic support but only 25% exact support. SAEs perform slightly better on the controlled cases they cover; the proposed method is not presented as superior there.

The comparison has limits: explicit source attributes are not ground truth for latent beliefs, and SAE disagreement cannot establish absence of information.

## Selected public evaluation settings

These are four settings from the original NLA paper's evaluation-awareness study. Evaluation awareness is the phenomenon measured across settings, not a separate dataset. The population table reports each setting separately, using 50 tasks per setting; the percentages describe all sampled claims, not only awareness claims.

| Setting | Public reconstruction material | Sampling and scope |
| --- | --- | --- |
| [GPQA Diamond](https://github.com/idavidrein/gpqa) | Released questions and reference prompts | 50 scientific multiple-choice questions; no tools. |
| [MMMLU](https://huggingface.co/datasets/openai/MMMLU) | Released question files for 14 languages | 50 questions distributed across the 14 languages. |
| [SWE-bench Verified](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified) | Repository commits, issues, tests, and a [public harness](https://www.swebench.com/SWE-bench/) | 50 coding-agent tasks; retain tool interactions and causal prefixes. Issue text alone would not reproduce the agentic setting. |
| [Blackmail](https://github.com/anthropic-experimental/agentic-misalignment) | Released fictional email scenarios, templates, and configurations | 50 task conditions from the saved Inspect-based implementation; conditions can share templates and are not independent scenario designs. |

The saved evaluation registry and run record list these families, with AV generation recorded as complete. This is a saved status, not a fresh verification of outputs. Public Qwen runs reproduce evaluation settings, not the original Claude transcripts. Their selection does not prove absence from NLA training data. No broader experiment registry has been changed by this presentation choice.

## Before integrating

The primary setup follows the saved Qwen protocol, not the separate three-model robustness study. Preserve claim-sampling weights, unresolved cases, and document/trajectory clustering. Keep the four evaluation families separate. For template-based tasks, assess dependence across shared templates when computing uncertainty.

Use normalized AR predictions for reconstruction error, as specified by the saved metric amendment. Reconcile the manuscript's segmentation-model description with the model actually used. The specificity-failure branch should proceed to seeded/confabulated tests regardless of ownership; weak-theme/high-support cases remain unresolved.

For SAE validation, confirm compatibility with the target checkpoint and activation site. Feature coverage must be determined independently of activity on the evaluated example. Every covered-subset comparison uses the same 660 cases; all non-SAE full comparisons use the same 1,056 cases. The 1,000-claim natural SAE audit is a separate category-balanced sample.
