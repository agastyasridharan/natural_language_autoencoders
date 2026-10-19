# Tier-3: band-widening unresolved recovery (cycle-consistency audit)

**Date:** 2026-07-23. **Status:** validated on all 4 models, shipped as a documented
variant (the committed P90 runs remain primary; see "Framing" below).

## What Tier-3 is

Tier-3 is the one lever in the unresolved-reduction program that needs GPU. The
cycle-consistency test selects, per claim B, a set of decoy placebos that pass four
filters; filters f3 (matched-mass) and f4 (naturalness/FVE) admit a decoy only if its
`mean_lbar_b` and `fve` fall inside calibrated tolerance bands (`eps_pr`, `fve_tol`).
Those bands were calibrated at the **P90** of the calibration distribution. A claim
whose decoys mostly fall outside the P90 bands ends up with fewer than `min_k=5` clean
placebos and is ruled **unresolved** on a validity pre-check (never-guess: too few
placebos to form a trustworthy null).

Tier-3 re-runs the pipeline with the bands widened to **P95**, admitting more of the
already-generated decoy candidates. The newly-admitted candidates have no stored
`dstar`/`x_m` readings, so they must be measured on GPU (this is why the no-GPU replays
in Tier-0/1/2 could not do it). Everything else is inherited from the Tier-0
calibration: `tau_val`, `kappa_0`, `k=5`. The decoy-vs-decoy false-positive budget is
re-audited after each run.

P95 band values (from each model's own percentile ladder):

| model | eps_pr P90→P95 | fve_tol P90→P95 |
|---|---|---|
| qwen7b | 5.2251 → 7.0312 | 0.0221 → 0.0314 |
| gemma12b | 5.6635 → 7.1510 | 0.0304 → 0.0455 |
| gemma27b | 9.2089 → 12.7902 | 0.0222 → 0.0333 |
| llama70b | 3.9133 → 5.8776 | 0.0378 → 0.0715 |

## Results

Unresolved before (P90) and after (P95), with recovered/lost counts and the
decoy-vs-decoy false-positive rate (full split / OOS split):

| model | N | unresolved P90→P95 | recovered / lost | FP full / OOS | κ₀ (applied / refit) |
|---|---|---|---|---|---|
| Qwen2.5-7B L20 | 1196 | 31.9 → 28.8% (−3.1) | 44 / 7 | 4.74 / 4.75% | 0.9 / 0.9 |
| Gemma-3-12B L32 | 298 | 31.2 → 30.5% (−0.7) | 10 / 8 | 4.49 / 4.51% | 1.2 / 1.1 |
| Gemma-3-27B L41 | 299 | 31.1 → 27.1% (−4.0) | 12 / 0 | 4.46 / 4.56% | 1.2 / 1.0 |
| Llama-3.3-70B L53 | 298 | 28.5 → 25.5% (−3.0) | 11 / 2 | 4.41 / 4.53% | 1.1 / 1.1 |

The effect is real but modest and model-dependent (−0.7 to −4.0 pp). Recovered claims
land mostly in the not-grounded buckets (seeded + confabulated dominate every model's
recovered distribution), consistent with the reading that the unresolved population is
dominated by measurement-limited claims (too few clean placebos) rather than genuinely
ambiguous causal signal. `lost` counts are claims demoted resolved→unresolved by the
richer null (a larger surviving-decoy set raises the power/Gate-1 bar and reveals the
claim was underpowered): this is the never-guess principle reasserting under better
data, i.e. the conservative direction.

## Guardrails

All four models pass the substantive, N-independent guardrails.

1. **Integrity (M_BB band-independent).** M_BB is B's own-effect and does not depend on
   which decoys survive, so the tier3 re-measure must reproduce the committed M_BB
   exactly. `max |M_BB_tier3 − M_BB_base| = 0.00` on every model. The re-measure is
   faithful.

2. **False-positive budget.** The decoy-vs-decoy leave-one-decoy-out FP holds at
   4.41–4.74% (full) and 4.53–4.75% (OOS), transfers=True on all 4, and the refit κ₀ is
   ≤ the applied κ₀. Widening the bands admits *less*-well-matched decoys, which can only
   raise the Q95(D*) grounding bar, so it cannot cheaply manufacture grounding; the FP
   audit confirms the budget was not breached. No κ₀ refit was required.

3. **No manufactured grounding.** Of the 16 grounded/entangled recoveries across the 4
   models, 15 were base `power=True` (well-powered claims blocked only by the validity
   pre-check) and clear the full P95 null; the 1 exception (llama 153_2) had base
   `power=False` only because it had `n_surv=0` clean placebos (a no-data fail, not
   genuine underpower). The 2 genuine power-fail recoveries (qwen 108_2 base
   `validity=True, power=False`; 377_7 likewise) landed in **seeded**/**confabulated**
   (M_BB=0.380, 0.014), i.e. not-grounded. So band-widening never converts a genuine
   power-fail into a grounding verdict.

## The smell test, and why its flags are small-N artifacts

The pre-registered smell test asks whether recovered claims spike toward grounding
relative to the baseline resolved population (recovered grounded+entangled fraction vs
baseline resolved grounded+entangled fraction, red-flag at +5 pp). qwen (N=1196, 44
recoveries) passed cleanly (11.4% vs 9.7%). The three N≈298 models flagged:

| model | recovered g+e | baseline resolved g+e | screen |
|---|---|---|---|
| gemma12b | 30.0% (3/10) | 12.7% | RED FLAG |
| gemma27b | 25.0% (3/12) | 8.3% | RED FLAG |
| llama70b | 45.5% (5/11) | 14.1% | RED FLAG |

These are small-N artifacts of a fraction-based threshold: at 10–12 recoveries a single
grounded claim moves the fraction 8–10 pp. Two independent lines of evidence clear the
flag without any post-hoc filtering:

- **Every grounded/entangled recovery is a validity-fail clearing the full null** (see
  guardrail 3). The recoveries are the *intended* mechanism, not contamination.
- **The negative-Q95(D*) pattern that drives the flag is normal, not a Tier-3
  artifact.** Some recovered groundings clear a *negative* decoy null (Q95(D*)<0, a null
  sitting at or below zero). This initially looked suspicious. But negative-null
  groundings occur in **20% / 38% / 35% / 10%** of the *base, published* grounded+
  entangled claims (qwen/gemma12b/gemma27b/llama70b). The framework already accepts them
  (Gate-2, `M_BB > 1.5·max M_iB`, guards cross-effect confusion; the FP audit is computed
  over these same nulls and holds). Recovered claims draw from the same distribution; the
  higher recovered fraction (2/3, 0/3, 3/5) is noise over 3–5 claims.

No negative-null gate or power-flip gate was added: doing so would judge Tier-3
recoveries by a stricter standard than the base results already meet, which would be
p-hacking. The honest resolution is the baseline data above.

The single most edge-case recovery, llama **153_2** (base doubly-blocked with
`n_surv=0`, recovered to grounded with M_BB=3.554 clearing a negative null), is
documented rather than suppressed. It is 1 claim in 2091.

## Framing (decision deferred to user)

The committed **P90 runs remain the primary published result.** Tier-3 is shipped here
as a validated variant that quantifies how much of the unresolved rate is recoverable by
supplying more clean placebos. Whether to **promote P95 to primary** (which would change
the published headline unresolved rates and grounded counts, and warrants wiring the
tier3 runs into the studio + republish) is left as a user decision; P90 is the
principled calibration point, and the recovery is modest, so the conservative default is
to keep P90 primary.

## Reproduce

On the Athena box (GPUs 2,3 only): `code/repl/tier3_run.sh <key>` for key in
{qwen7b, gemma12b, gemma27b, llama70b}. It waits for a fully-idle allowed GPU, runs
`cycle_run.py` with `calibration_tier3.json` (P95 bands) into `cycle_runs_tier3/`, then
`report.py`. llama70b uses the phase-split path and reuses the band-independent
`ar_cache_d4g24` (directions are cached over text-passing candidates before band
selection). Analysis: `code/repl/tier3_compare.py`, `tier3_soundness.py`,
`tier3_charter.py`, `tier3_spotcheck.py`. Full per-claim tier3 trees live on the box
under `cycle_runs_tier3/`; the FP-audit reports and the consolidated
`tier3_summary.json` are committed here.
