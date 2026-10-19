# Cycle-Consistency Causal Grounding Audit: Capstone

All four NLAs complete: Qwen2.5-7B (L20), Gemma-3-12B (L32), Gemma-3-27B (L41), Llama-3.3-70B (L53). Three model families, 7B to 70B.

## Where we are

Governing spec: `cycle-consistency v2` (Steps 1-7, filters 1-4, Gates 1-3, five verdicts).

- Steps 1-4 (segment, type, decoy generation, four text/mass/prior/FVE filters): done.
- Step 5b (same-subtrahend directions, unit overlap): done.
- Step 6 (three teacher-forced readings M_BB, X_m, D*): done.
- Step 7 (power + validity pre-checks, Gate 1 with the M_BB>0 floor, Gate 2, Gate 3, five verdicts): done.

Stages: S0-S5 done (qwen7b end-to-end, full N=400). **S6 cross-model**: all four NLAs complete. **S7**: cycle-consistency studio tab built and jsdom-verified; this capstone complete. **Post-hoc refinements**: the Gate-3 M_BB<=0 correction and Tier-1 unresolved recovery (2026-07-22, validity decouple + power-fail floor), and the Tier-0 κ₀ recalibration (2026-07-23, restore the ≤5% decoy FP budget OOS). All are judge-free arithmetic replays over the stored readings, rescored across all four models and reconciled on the Athena box. A verification panel (2026-07-23) additionally confirmed the no-GPU well is dry: after Tier-0+Tier-1 every remaining unresolved claim is unresolved for a principled reason (underpowered with no MDE floor, too few genuine placebos, or missing measurements), and every further no-GPU lever tested either inflates the FP budget or causes two-way churn. The one GPU lever, **Tier-3 band-widening** (2026-07-23), was then run on all four models: it recovers a modest, model-dependent share of unresolved (unresolved 31.9/31.2/31.1/28.5% → 28.8/30.5/27.1/25.5%) with every guardrail intact, and is shipped as a documented variant (committed P90 runs stay primary).

## TL;DR

The cycle test is a judge-free, decoy-controlled causal grounding audit for NLA activation-verbalizer (AV) claims. No LLM ever sees a reading or a verdict; LLMs only segment text, type claims, and write/judge decoy text. The mechanical verdict for a claim B is: is B's own reconstruction direction the thing that causally moves B's teacher-forced log-prob, above a calibrated decoy null and above every other claim's direction?

Across four NLAs spanning three model families (Qwen2.5, Gemma-3, Llama-3.3) and 7B-70B, the result replicates:

1. **Most "grounded" claims from the simpler atom test do not survive.** Only 13-18% of atom-grounded claims remain grounded under the decoy-controlled cycle test. The cycle test is strictly more conservative.
2. **The verdict mix is stable across models.** Roughly 4-6% grounded, 27-33% seeded, 28-34% confabulated, 28-32% unresolved.
3. **Confabulation is essentially never rescued.** Of claims the atom test called confabulated, the cycle test promotes 0 (qwen7b, gemma27b, llama70b) or 1 (gemma12b) to grounded.

**Gate-3 correction (2026-07-22).** A non-positive own-effect (M_BB <= 0, erasing the claim's own direction does not hurt its log-prob) is classified confabulated outright; "seeded" is reserved for a positive own-effect that merely fails the decoy null. This moved 72 (qwen7b), 26 (gemma12b), 24 (gemma27b), 19 (llama70b) claims from seeded to confabulated. It is scoped to resolved claims: applied after the power/validity pre-checks, so a broken decoy control still yields unresolved. Grounded, entangled and unresolved counts are unchanged by this correction.

**Tier-1 unresolved recovery (2026-07-22).** Two judge-free, GPU-free refinements reduce the unresolved rate without touching the ~5% decoy false-positive budget. (a) *Validity decouple*: a flagged decoy (its own direction encoded in the text, or a D* outlier) is trimmed from the clean-placebo count but kept in the Gate-1 null, so a decoy family is valid whenever at least `min_k=5` clean placebos remain. Because the Gate-1 `Q95(D*)` bar is always computed over the full survivor set, trimming for the validity count can never lower the grounding bar (monotone-null lock), so no false-grounded is manufactured (verified: naively dropping flagged decoys from the null too would manufacture 17/2/3/2 false-grounded; the lock prevents them). (b) *Power-fail floor*: a claim that fails the power pre-check but has a valid decoy family and a non-positive own-effect (`M_BB <= 0`) resolves to confabulated rather than unresolved, because the valid family is the minimum-detectable-effect floor and the own-effect is non-positive. Tier-1 is churn-locked: it only recovers unresolved claims, never re-judges a resolved one, so the decoy-vs-decoy false-positive rate is byte-identical before and after (5.27/5.47/5.20/4.35%). Unresolved drops from 45.2/42.6/42.1/36.6% to **31.9/31.2/31.1/28.5%** (qwen7b/gemma12b/gemma27b/llama70b), recovering 159/34/33/24 claims. Most recovered claims are confabulated or seeded; a small number (11/2/0/2 grounded, 8/0/0/1 entangled) clear the full-null Gate-1 bar, each verified to own its effect against the full decoy null.

**Tier-0 κ₀ recalibration (2026-07-23).** The applied Gate-1 margin κ₀ was fit in-sample; on the run's own survivor D* families and the disjoint out-of-sample split, the leave-one-decoy-out decoy-vs-decoy false-positive rate exceeded the 5% budget in 3 of 4 models (5.32/5.65/5.39% OOS for qwen7b/gemma12b/gemma27b; llama70b already held at 4.26%). Tier-0 raises κ₀ to the smallest value holding the LODO FP ≤ 5% on both the full and OOS families, `κ₀* = max(calibration κ₀, full-run refit, OOS refit)`: 0.8→0.9 (qwen7b), 0.9→1.2 (gemma12b), 1.0→1.2 (gemma27b), 1.1 held (llama70b). κ₀ is a one-sided safety margin: raising it only raises the Gate-1 bar, so this recalibration can only demote grounded/entangled to seeded/confabulated (never manufacture grounding) and never touches a pre-check (so the unresolved set is invariant). This is a judge-free arithmetic replay; it demoted 4/4/1/0 claims and restored the budget on all four models (OOS FP now 4.90/4.50/4.79/4.26%, transfers=True everywhere). Because raising κ₀ has no headline-inflation direction, using the OOS split to pick a more-conservative κ₀ carries no peeking risk; the residual caveat is that this OOS split is not classic held-out (it reuses the run's own claims), so the guarantee rests on the max()+monotonicity argument, not on OOS being independent. All tables below are post-correction, post-Tier-1 and post-Tier-0.

**Tier-3 band-widening (2026-07-23, GPU, documented variant).** The one unresolved-reduction lever that needs GPU. Filters f3/f4 admit a decoy placebo only if its matched-mass and FVE fall inside calibrated tolerance bands, calibrated at P90; a claim whose decoys mostly fall outside those bands ends up with fewer than `min_k=5` clean placebos and is ruled unresolved on the validity pre-check. Tier-3 re-runs the pipeline with the bands widened to P95, admitting more already-generated candidates so more validity-fail claims reach a valid family; the newly-admitted candidates have no stored readings, so they are measured on GPU (this is why the no-GPU replays could not do it). Unresolved drops from 31.9/31.2/31.1/28.5% to **28.8/30.5/27.1/25.5%** (qwen7b/gemma12b/gemma27b/llama70b), recovering 44/10/12/11 claims (mostly seeded+confabulated). The effect is modest and model-dependent (−0.7 to −4.0 pp). All guardrails hold: M_BB reproduces the committed value exactly (band-independent own-effect, drift 0.00); the decoy-vs-decoy FP stays within budget (4.41–4.74% full, transfers OOS, no κ₀ refit needed); and no grounding is manufactured (15 of 16 grounded/entangled recoveries are base-`power=True` validity-fails clearing the full null, the 1 exception is a no-data `n_surv=0` fail, and the 2 genuine power-fail recoveries land in seeded/confabulated). The smell test flagged the three N≈298 models, but this is a small-N artifact of a fraction-based threshold: every grounded recovery is a validity-fail (the intended mechanism), and the negative-Q95(D*) groundings that drive the flag occur in 20/38/35/10% of the *base* published grounded+entangled claims, so recovered claims draw from the same distribution (no post-hoc gate was added; that would be p-hacking). The committed P90 runs remain primary; Tier-3 is shipped as a validated variant that quantifies the recoverable share of unresolved. Details and per-model numbers: `tier3/TIER3_REPORT.md` + `tier3/tier3_summary.json`.

## Method (judge-free by construction)

The AV maps a hidden activation `h` to an English explanation. The concern the test addresses: the AV may narrate a plausible prior rather than read `h`. For each atomic claim B in the explanation:

- The activation reconstructor (AR) builds B's direction with a fixed subtrahend, `Δ_B = norm(AR(z)) - norm(AR(z\B))`, where `z` is the explanation and `z\B` deletes B's span. A decoy claim B*_m spliced into B's slot gives `Δ*_m = norm(AR(z*_m)) - norm(AR(z\B))`. The unit overlap `ov = h·Δ̂` measures how much of B (or a decoy) the reconstruction places along `h`.
- Decoys are same-type sibling claims the source text does not support, written and absence/naturalness-judged by an LLM that never sees an activation or a number. A decoy survives four mechanical filters: (1) judged absent, (2) matched removed mass `|ov*-ov_b|`, (3) matched residual prior `ℓ̄`, (4) natural and within the FVE band. Thresholds are calibrated on a held-out activation split.
- Three teacher-forced log-prob readings:
  - `M_BB`: ablate `Δ̂_B` in `z`, score B in `z`. B's own causal weight.
  - `X_m`: ablate `Δ̂_{B*_m}` in `z`, score B in `z`. Does a decoy's direction move B (seeding)?
  - `D*_m`: ablate `Δ̂_{B*_m}` in `z*_m`, score B*_m in `z*_m`. The decoy's own weight, the matched null for `M_BB`.
- Gating (Step 7): a claim is **unresolved** unless powered (supp_b above the decoys' Q95) and valid. Validity (Tier-1 decouple) trims flagged decoys (own direction encoded, or D* outlier) from the clean-placebo count while keeping them in the Gate-1 null, and passes when at least `min_k=5` clean placebos remain. A power-fail with a valid family and `M_BB <= 0` is resolved to **confabulated** (the family is the minimum-detectable-effect floor). Gate 1: `M_BB > Q95(D*) + κ₀` and `M_BB > 0`, with `Q95(D*)` over the full survivor set. Passing goes to Gate 2 (`M_BB > 1.5·max_{i≠B} M_iB` -> **grounded**, else **entangled**). Failing goes to Gate 3: a non-positive own-effect (`M_BB ≤ 0`) is **confabulated** outright; otherwise median `X_m > 0` -> **seeded**, else **confabulated**.

`M_BB` is numerically the atom test's teacher-forced Δlog-prob under the same deletion string and normalization, which is what makes the cycle verdict directly comparable to the published atom verdict, and gives an exact integrity link (below).

## Study design

Each model has a published atom run of 400 activations (FineWeb, one read position each). For the cross-model replication:

- Decoys were generated for the 100 activations at `ok[::4]` (300 claims/model, g=24 candidates each).
- Calibration (filters + κ₀) was fit on the 25-activation `ok[::16]` subset (a strict subset of the 100).
- The cycle test ran on all 100 activations, giving **25 in-sample + 75 out-of-sample** activations for an honest κ₀ transfer check.

qwen7b used the full N=400 run (S5), calibrated on 25, so 375 activations are out-of-sample.

Gold activations are the doc_id-matched `out_corpus_<model>_fineweb` corpora. (The unsuffixed `out_corpus_gemma12b` is a different UltraFineWeb corpus; the l0 integrity check caught the mismatch, 314-nat drift, before any full run.)

## Cross-model results

| Model | L | claims | grounded | entangled | seeded | confab | unresolved |
|---|---|---|---|---|---|---|---|
| Qwen2.5-7B | 20 | 1196 | 4.3% | 2.3% | 32.9% | 28.5% | 31.9% |
| Gemma-3-12B | 32 | 298 | 4.4% | 4.4% | 28.2% | 31.9% | 31.2% |
| Gemma-3-27B | 41 | 299 | 4.0% | 1.7% | 29.1% | 34.1% | 31.1% |
| Llama-3.3-70B | 53 | 298 | 6.0% | 4.0% | 31.2% | 30.2% | 28.5% |

The seeded, confabulated and unresolved fractions are close across models and families. Grounded sits at 4.0-6.0% across all four, rising gently with scale. Post-correction, post-Tier-1 and post-Tier-0, seeded (28-33%), confabulated (28-34%) and unresolved (28-32%) are all comparable in every model.

### Finding 1: the cycle test demotes atom-grounded claims

Atom-grounded claims that remain grounded under the cycle test:

| Model | survive | of atom-grounded | rate |
|---|---|---|---|
| Qwen2.5-7B | 51 | 384 | 13.3% |
| Gemma-3-12B | 12 | 80 | 15.0% |
| Gemma-3-27B | 12 | 81 | 14.8% |
| Llama-3.3-70B | 18 | 99 | 18.2% |

The rest go mostly to seeded or unresolved. The decoy control removes the atom test's main blind spot: an atom-grounded claim can lose its own direction while a matched decoy loses just as much, in which case the effect is not specific to B.

### Finding 2: confabulation is essentially never rescued

Atom-confabulated claims the cycle test promotes to grounded:

| Model | promoted | of atom-confab |
|---|---|---|
| Qwen2.5-7B | 0 | 452 |
| Gemma-3-12B | 1 | 180 |
| Gemma-3-27B | 0 | 151 |
| Llama-3.3-70B | 0 | 96 |

The single gemma12b exception (1/180) is flagged honestly; it is a claim the atom test's stricter null called confabulated but the decoy-matched cycle null clears. It is within sampling noise and does not change the pattern, but it is a real difference from the clean zero in qwen7b, gemma27b and llama70b.

### Finding 3: grounding magnitude

For grounded and entangled claims, the grounding estimate `M_BB - median D*` (the causal weight above the decoy null) has median 4.7 nats (qwen7b), 5.5 (gemma12b), 6.4 (gemma27b), 5.2 (llama70b). The claims a model does read carry several nats of separable causal weight above the decoy null in every case. (Tier-0 raised these medians slightly, since the demoted claims were the ones nearest the Gate-1 boundary.)

## κ₀ out-of-sample validation

κ₀ is the Gate-1 margin above the decoy Q95, fit by leave-one-decoy-out so the decoy-vs-decoy false-positive rate is about 5% on the calibration split. The original in-sample fit exceeded the 5% budget out-of-sample in three of four models; **Tier-0 (2026-07-23) recalibrated κ₀ to hold the budget on both the full and OOS families**. Post-Tier-0:

| Model | κ₀ (pre → post) | OOS decoy FP (pre → post) | transfers (≤5%) |
|---|---|---|---|
| Qwen2.5-7B | 0.8 → 0.9 | 5.32% → 4.90% | **yes** |
| Gemma-3-12B | 0.9 → 1.2 | 5.65% → 4.50% | **yes** |
| Gemma-3-27B | 1.0 → 1.2 | 5.39% → 4.79% | **yes** |
| Llama-3.3-70B | 1.1 (held) | 4.26% → 4.26% | **yes** |

All four now hold the decoy-vs-decoy FP budget out-of-sample. κ₀ is a one-sided safety margin (higher only tightens Gate 1), so the recalibration can only demote grounded/entangled and never manufactures grounding; the demotions were 4/4/1/0 claims. The headline is insensitive to κ₀: on qwen7b, sweeping κ₀ over {0.8, 0.9, 1.0} moves grounded only by a couple of claims, and the κ-grid is smooth and monotone in all models. Caveat: the OOS split reuses the run's own claims (calib activations excluded) rather than a fresh corpus, so the guarantee rests on the max()+monotonicity argument; the LODO budget is also fit on families with more than k survivors, i.e. a >k-family budget rather than a per-claim guarantee for the rare 1..k-survivor families that still reach Gate 1.

## Integrity

- `M_BB` reproduces the published atom Δlog-prob to ~1e-6 nats (qwen7b full run), and every AR direction reproduces the stored atom npz delta to <1e-3 (`npz_delta_check`), so the cycle test is measuring the same reconstruction the published runs used.
- The gold-activation l0 check (`l0_check_max`) is 0.0 on the correct FineWeb corpora, confirming the injected `h` matches the atom run exactly.
- For llama70b, which used the phase-split, integrity holds across all 298 claims: max l0 drift 8e-6 nats, max npz-delta drift 0.0. The two-phase AR-cache path is byte-identical to the co-resident path.

## Engineering: the llama70b phase-split

The 70B AV (132 GB) and AR (89 GB) cannot co-reside on the 2x H100 NVL available (192 GB). They never run concurrently within a claim, so the pipeline splits into two GPU phases: `ar_precompute.py` loads only the critic and caches every claim's `decoy_directions` output (`ar_cache.py`, arrays in npz + an alignment manifest), then `calib_cycle.py`/`cycle_run.py` with `--ar-cache` load only the AV and consume the cache. The cache carries a candidate-order + z_inner + prompt-version manifest and is asserted on load, so a stale or misaligned cache raises rather than silently scoring the wrong directions. The 7B/12B/27B models co-reside and use the live critic path unchanged.

## Caveats

- Cross-model N is 100 activations (300 claims) vs qwen7b's 400. Enough to establish the replication and an OOS split; not the full-catalog scale of qwen7b.
- After Tier-0, κ₀ holds the decoy-vs-decoy FP budget out-of-sample in all four models (≤4.90%). The residual caveat is that the OOS split is not a fresh corpus (it reuses the run's own claims), so the budget guarantee rests on max()+monotonicity, and it is a >k-family budget by the LODO convention. A run on genuinely disjoint activations remains the strongest future check.
- gemma12b has one atom-confabulated claim promoted to grounded (see Finding 2).
- "Unresolved" is 28-32% of claims post-Tier-1 (down from 37-45%). This is the test refusing to rule when a claim lacks a powered, valid decoy panel, by design, not a measurement gap. Tier-1's validity decouple and power-fail floor recover the claims where the panel is in fact adequate; the remaining unresolved genuinely lack a powered, valid null. The GPU Tier-3 band-widening variant recovers a further modest slice (to 25.5-30.5%) by admitting more clean placebos, and the residual after that is the hardest core of measurement-limited claims.

## Artifacts

- Reports: `cycle_runs/<model>-d4g24/<run>/{cycle_report.json, CYCLE_REPORT.md}`; per-claim `claims/<idx>_<ci>.json`.
- Calibration: `calib_d4g24/<run>/calibration_signed.json` + `DECOY_AUDIT.md`.
- Decoys: `decoys_d4g24/<run>/`; AR cache (llama70b): `ar_cache_d4g24/<run>/`.
- Studio: `studio/cycle.html` + `studio/cycle_studio.py` (cross-model overview, how-it-works, per-model Tier-1 before/after, shift/gate/OOS, claim drill-down); jsdom test `studio/test_cycle_ui.js`.
- Code: `repl/{cycle_consistency,calib_cycle,cycle_run,report,decoy_gen,ar_cache,ar_precompute}.py`; offline tests `repl/test_cycle_offline.py`.
- Post-hoc rescore + provenance: `repl/rescore_cycle.py` (Gate-3 correction), `repl/rescore_tier1.py` (Tier-1 apply, churn-locked + flag-faithful), `repl/rescore_tier0.py` (Tier-0 κ₀ refit, churn = grounded/entangled→seeded/confab only), `repl/tier1_provenance.py` + `repl/tier0_provenance.py` (before/after blocks for the studio), `repl/replay_unresolved.py` (judge-free variant replay that motivated Tier-1).
- Tier-3 (GPU band-widening variant): driver `repl/tier3_run.sh` (GPU-2,3-only waiter, phase-split + AR-cache reuse for llama70b); analysis `repl/tier3_compare.py` (P95-vs-P90 diff + guardrails), `repl/tier3_soundness.py` (negative-null baseline), `repl/tier3_charter.py` (validity-fail charter check), `repl/tier3_spotcheck.py` (per-claim grounding-recovery audit); results in `tier3/TIER3_REPORT.md`, `tier3/tier3_summary.json`, `tier3/reports/<model>.json` (FP audits). Full per-claim tier3 trees live on the Athena box under `cycle_runs_tier3/`.
