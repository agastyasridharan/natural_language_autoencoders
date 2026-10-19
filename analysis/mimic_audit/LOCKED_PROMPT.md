# Locked single mimic prompt

**Artifact:** `prompt_final.txt` (= `prompt_v1_1.txt`). This is the one frozen
prompt. It is used unchanged on every input: the model is given the full text
prefix and produces a text-only "mimic" of the AV read at that token.

## How it was tuned (the single-prompt method)

1. **Fingerprint** the real style from all 3,048 reads (`STYLE_FINGERPRINT.md`):
   measurable signatures with realistic p10-p90 bands. This is the objective
   target, replacing the gameable "beat the judge" signal that drove the [OLD]
   run to caricature (6%).
2. **One prompt**, style carried by 6 real train-split examples plus a short
   description, not an 11k-char rule pile.
3. **Calibrate, two-sided.** Generate on 128 held-out eval tasks; check the
   mimic's fingerprint against the real bands; check a cross-model (Sonnet 4.6)
   style discriminator against 50% from both sides.
4. **One safe nudge** (v1 -> v1.1: probabilistic slash/ellipsis, stronger
   conditional recycling; no forced incoherence), re-validate, lock the better.

## Calibration evidence (identical 128 held-out tasks)

| | v1 | v1.1 (locked) | real target |
|---|---|---|---|
| discriminator acc | 43.8% | 47.2% | 50% |
| two-sided \|acc-0.5\| | 0.062 | 0.028 | 0 |
| numeric signatures OVER band | none | none | none |

All 11 numeric style signatures are in-band for both. The discriminator is at
chance: the mimic is indistinguishable from real AV output by a different model
judging style alone. (The v1 -> v1.1 discriminator delta is within sampling
noise; v1.1 is locked as the deliberate, no-worse-on-any-axis version.)

## Known, accepted residual

The mimic is slightly too tidy on categorical markers: it nearly always uses the
"X/ Y" slash classification (100% vs real 68%) and ends on "..." (98% vs 67%),
and rarely leaves a quote unclosed or recycles (vs 33% / 39% real). Soft
frequency instructions did not move these; LLMs do not execute "two-thirds of the
time" across independent samples.

This residual is **immaterial to the audit and partly desirable**:
- The discriminator is already at chance despite it, because real AV output also
  uses these markers very often, so frequency is a weak tell.
- The deeper "too clean" residual is the mimic, holding the full clean text, not
  reproducing the real AV's low-cosine reconstruction garble. That garble is an
  activation/reconstruction artifact, NOT present in the text prior. A text-prior
  baseline SHOULD stay coherent; faking garble would overstate what text alone
  recovers and corrupt the content audit.

## Status

Locked. Next: the content audit (per-claim derivable-by-mimic vs
grounded-in-activation, plus reconstruction-ablation and scramble grounding
controls) using this prompt as the fixed instrument.
