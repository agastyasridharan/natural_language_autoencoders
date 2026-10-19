# Experiment 7 capstone — verbalization stability

Completed 2026-08-27 on 200 frozen activations per model. Each stochastic temperature used 24 resamples;
T=0 used one greedy decode. Meaning clusters were produced by `gpt-5.6-luna` after a labelled gate
(97.8% pairwise agreement, zero false merges) and an exact-cardinality schema check on the largest real
pool (142/142 claims). All final clustering and H5 runs had zero judge failures and zero singleton
fallbacks.

## Verdicts

- **H1 — mixed, not clean bimodality.** At T=0.4, low-consensus clusters dominate by cluster count
  (86–95% for LCR/PC), while high-consensus clusters still carry 16–33% of claim mass. The middle carries
  16–31% of claim mass, so the distribution is not a clean core/periphery split.
- **H2 — modest support.** From T=0.4 to 1.5, semantic entropy increased less than surface entropy for
  every model/tag, but the separation was sometimes small (especially Qwen PC). Core mass declined with
  temperature; the remaining stability is partly paraphrase-invariant rather than surface-form reuse.
- **H3 — partial for LCR, rejected for genre.** LCR was more stable than PC on both Gemma models
  (T=0.4 core mass 0.400 vs 0.266 for Gemma-12B; 0.362 vs 0.302 for Gemma-27B), but nearly tied on Qwen
  (0.220 vs 0.207). Genre core mass was only 0.000, 0.011, and 0.020, contrary to the prediction that
  genre/register would be stable.
- **H4 — specified floor is not estimable; activation-specific alternative fails.** The no-injection
  path produced no valid explanations for Gemma-12B/27B and only 127/4,800 for Qwen, the expected
  injection-failure smell. Qwen's selected survivors are therefore not a comparable floor. The valid
  shuffled-activation control is decisive: paired main-minus-shuffle confidence intervals include zero
  for every model/tag. Stable recurrence does not identify the particular injected activation.
- **H5 — rejected.** Only 26.5%, 46.0%, and 41.2% of greedy LCR claims, and 23.8%, 30.7%, and 32.7% of
  greedy PC claims, matched a T=0.4 core cluster (Qwen, Gemma-12B, Gemma-27B respectively). Greedy claims
  are not a subset of the stochastic core.

## Quantitative checks

At T=0.4, main core mass exceeded its deterministic permutation null for LCR and PC in all three
models. Paired deltas (95% bootstrap CI) were Qwen LCR +0.033 [0.022, 0.044], PC +0.053 [0.040, 0.066];
Gemma-12B LCR +0.050 [0.035, 0.067], PC +0.066 [0.055, 0.077]; Gemma-27B LCR +0.068 [0.048, 0.089], PC
+0.061 [0.051, 0.072]. Thus there is above-chance recurrence, but the shuffled control attributes it to
the verbalizer/scaffold distribution rather than activation identity.

Main-arm validity stayed above 98.7% through T=1.0. At T=1.5 it was 94.9% for Qwen, 99.4% for
Gemma-12B, and 99.6% for Gemma-27B. The sharper Qwen degradation is a temperature-sensitivity result,
not a clustering artifact.

The OpenAI usage ledger records 25,530,094 input tokens, 3,870,878 output tokens, and an estimated
cost of $9.7511 under the internal $90 hard stop.
