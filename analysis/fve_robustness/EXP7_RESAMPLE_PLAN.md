# Exp 7 — "resample the verbalizer": how stable is a claim across resamples & temperature?

Status: **7.0 GATE PASSED (2026-07-24).** Meaning-judge sanity done (0 false-merges, 95.1%); decisions
locked (§9); harness/subsample/clustering/dashboard being built CPU-side; **7.1 GPU pilot blocked only on
cluster connectivity (VPN).** Synthesized from a 4-angle design panel
(framing / semantic-similarity / protocol / controls-skeptic), with the meaning-similarity method set
to **an LLM judge** per user decision (dropping the embedding-threshold-calibration track). Unlike
Exp 1-6 (edit one fixed explanation, re-run only the cheap AR), **Exp 7 re-runs the AV (generation)** —
a heavier compute footprint that the protocol scopes with a doc subsample.

## 1. Problem & idea

The AV is a **stochastic** decoder: the same frozen activation `h` yields a different explanation on
every sample (temp 0.4, seed 0 is just one draw). Exp 7 treats the explanation as a **random variable**
and asks: how much of the verbalization is **signal** (determined by `h` → stable across resamples and
temperature) vs **noise** (produced by the sampler → grows with temperature)? Resample the AV N times
per activation at several temperatures, decompose each into claims, cluster claims **by meaning** across
resamples, and read off a **stable core** (claims asserted in most resamples) vs a **temperature-sensitive
periphery** (one-off claims).

This is a **judge-free, input-free grounding signal** complementary to Exp 3/6: those ask whether a claim
is in the INPUT; consensus asks whether a claim is reliably **read from the activation at all**. Crossed
with Exp-6 input-derivability it gives a 2×2:

| | in input | not in input |
|---|---|---|
| **stable core** | faithful read | **candidate privileged content OR systematic AV prior** |
| **unstable** | noisy but grounded | pure confabulation |

The **not-in-input × stable** cell is the interesting one; consensus alone cannot tell "privileged
content" from "AV prior" — the **no-injection** and **shuffled-activation** controls (below) do.

## 2. Hypotheses (falsifiable)

- **H1 (bimodality):** the consensus-frequency histogram is bimodal — a core (freq→1) and a periphery
  (freq→0), not a flat middle. A large flat middle = the verbalization is mostly sampler noise.
- **H2 (temperature scaling):** surface diversity rises with T; **meaning diversity of the core rises
  much slower** — a determinate reading survives temperature. Null: meaning diversity tracks surface.
- **H3 (per-tag):** genre/register and LCR (input-restatement) are the stable core; **PC (predicted
  continuations) are the temperature-sensitive periphery** (the AV guesses, and guesses vary). Plausible
  alternative: the core is content-free scaffold and everything specific is periphery.
- **H4 (grounding floor):** the stable core exceeds the **no-injection** floor — i.e. it is driven by
  the activation, not just the prompt scaffold + AV prior. Null (the experiment-killer): core ≈ floor →
  "stability" is the AV always saying similar things regardless of `h`.
- **H5 (greedy subset):** the T=0 greedy claims ⊆ the T=0.4 core (sanity).

## 3. Resample protocol

**Held FIXED across every resample** (this is what makes the RV well-defined): the frozen gold
activation `gold[idx]` (never re-extracted), read position/token, `injection_scale`, marker position,
actor prompt, `top_p=0.95`, `max_new_tokens=200`. **Only the sampler RNG and temperature vary.** Reuse
`fve_ufw.py` phase-avgen injection path verbatim (`fve_resample.py` imports `BatchInjector`,
`build_actor_prompt`, `find_layers`, `read_sidecar`, `load_av` unchanged), adding an outer (arm,
temperature, resample) loop. Each T>0 draws **24 fresh independent samples** (batched, one seed per
resume-stable batch ordinal over the full doc×k grid); the existing single `av_reads.json` draw is NOT
reused (24 clean independent draws are cleaner than mixing one legacy draw in).

- **N = 24** stochastic resamples per (datapoint, temperature) **for T > 0 only**.
- **T = 0 is NOT resampled — it is drawn ONCE (N=1), greedy/deterministic** (`do_sample=False`). Sampling
  a deterministic decoder N times would return N identical strings and manufacture a fake "core", so T=0
  is the greedy anchor only (feeds H5: greedy claims ⊆ T=0.4 core) and is excluded from the
  signal-vs-noise stability comparison.
- **Temperature grid T ∈ {0.0(×1), 0.4, 0.7, 1.0, 1.5}** — 0.0 greedy anchor; 0.4 harness default; 0.7 &
  1.0 rising stress; **1.5 high-T stress** (added). `top_p=0.95` is held across all T (bounds the worst
  degeneracy), but at T=1.5 expect a non-trivial fraction of samples to hit the CJK/injection-failure/
  degenerate filter — that **filtered fraction is itself a signal** ("noise grows with T"), so the harness
  reports **effective-N per (datapoint, temperature)** and computes stability on the surviving samples.
- **Doc subsample: 200 datapoints/model**, stratified by Exp-6 LCR divergence (equal thirds
  low/mid/high; per-datapoint divergence = mean over its LCR quotes, from `records_exp6b`) so the
  confabulation axis is covered. Held fixed across all T and both meaning/surface clustering.
- **Filter** CJK/injection-failure samples (CLAUDE.md smell) and empty/no-`<explanation>` generations
  before clustering; log the filtered count per temperature.

## 4. Claim decomposition

Reuse the **Exp-6 parser** (`exp6_edits.classify_expl_spans`) to split each sampled explanation into
typed claim units: **genre** (B0 register line), **LCR** (left-context restatements), **PC** (predicted
continuations), with ECHO/DUP/JUNK frozen out. Track stability **per tag**.

## 5. Meaning-similarity method — an LLM judge (per user decision)

For each (datapoint, temperature, tag) pool all claims across the N resamples and have **Claude (Haiku)
partition them into meaning equivalence-classes** in one call: "group claims that assert the same thing;
keep genuinely distinct claims separate." The prompt is explicit that **same-topic ≠ same-claim** — e.g.
for the read token "by", *"his owner's husband"*, *"a robotic vacuum"*, and *"the family's cleaning
staff"* are **three distinct** PC guesses that must NOT be merged just because they share the vacuum
topic; whereas *"a robotic vacuum"* and *"an automated cleaning device"* ARE one class. The judge returns
, per claim, a cluster id + a short canonical label (used on the dashboard). No embeddings, no threshold
calibration, no separate metric-validation gate.

**Paired surface-clustering control (free):** compute every metric a second time with **string/ROUGE-L≥0.9
clustering** instead of the LLM. The **gap (meaning-stability − surface-stability)** is itself a headline
readout: meaning ≫ surface ⇒ the apparent diversity is mostly paraphrase of a stable idea; meaning ≈
surface ⇒ the diversity is genuinely different claims.

## 6. Stability metrics + consensus-FVE tie-in

Per (model, datapoint, temperature, tag), on the meaning-clusters:
- **Consensus frequency** `c(cluster) = (#resamples asserting ≥1 claim in the cluster)/N`, binomial CI.
- **Core/periphery:** core = `c ≥ 0.5` (also report 0.33 / 0.8); `core_fraction`, periphery count,
  claim-mass in core. **Consensus histogram** (test H1 bimodality: mass in [0.7,1] vs [0,0.2] vs middle).
- **Pairwise semantic agreement** = expected fraction of one resample's claims that have a meaning-match
  in another (soft Jaccard over meaning-clusters), averaged over resample pairs.
- **Semantic entropy** of the cluster distribution, as a function of T (H2 curve).
- **Per-tag stability curves** (genre/LCR/PC × temperature) (H3).
- Aggregated with **doc-clustered bootstrap 95% CIs** (resample the 200 docs, matching Exp-6).
- **Optional consensus-FVE tie-in (GPU):** build a "consensus explanation" (the core claims) and AR-score
  it vs a single sample — does reconstructing from the consensus give higher/steadier FVE?

## 7. Controls (what makes a stability number interpretable)

- **No-injection scaffold FLOOR (the key control):** run the full N-resample protocol with the injection
  hook DISABLED (marker token present, activation absent). This bounds how much "stability" is reachable
  with **zero information from the target activation** — the AV's prompt scaffold + generic prior. H4:
  the real core must **exceed this floor**. Without it, a "stable core" is uninterpretable.
- **Shuffled-activation control:** inject a *different* datapoint's activation. If the "core" persists,
  it is a prior, not activation-specific. Disambiguates the stable×not-in-input 2×2 cell.
- **Degenerate-greedy (T=0):** decoder must be deterministic; greedy claims ⊆ T=0.4 core (H5). Excluded
  from the signal-vs-noise comparison (it is a ceiling artifact).
- **Permutation null:** shuffle the claim→resample assignment, recompute; report **observed − null**. A
  core is only claimed above this null.
- **Surface-vs-meaning** (Section 5): the paired surface run.

## 8. Compute budget + gated run order

Per doc: T=0 → 1 greedy + T∈{0.4,0.7,1.0,1.5} → 4×24 = **97 generations/doc**. × 200 docs × 3 models =
**58.2k** AV generations (200 tok each) + the no-injection floor at T=0.4 (N=24: ~14.4k) +
shuffled-activation at T=0.4 (N=24: ~14.4k) ≈ **~87k generations** on GPUs 0/1 — the heaviest arm so far,
but bounded; batched avgen (batch 64) makes it a few GPU-hours/model. LLM clustering ≈ (200 docs × 4
stochastic-T × 3 tags × 3 models) ≈ **7.2k Haiku calls** (+ floor/shuffle) — cheap (~$5-15), CPU/API on
the Mac. Surface clustering + all metrics are free CPU.

**Gated run order (cheapest-first):**
- **7.0 — meaning-judge sanity (API, tiny):** hand the LLM a labelled handful of same/distinct claim
  sets (incl. the vacuum-topic trap) and eyeball its clustering. **Check in.**
- **7.1 — tiny generation pilot (GPU, ~10 docs × full T grid, qwen7b):** confirm the resample harness +
  injection path + CJK filter + decomposition + clustering end-to-end; look at real resample spreads.
  **Wire the pilot's real resample-spread examples into the Exp 7 dashboard tab** (per-datapoint: the N
  samples per temperature, the meaning-clusters with consensus frequency, per-tag core/periphery) so the
  spread is inspectable before the full run. **Check in.**
- **7.2 — floor + full run (GPU):** the no-injection floor, then the full N=24 grid over 200 docs × 3
  models, then shuffled-activation.
- **7.3 — cluster + metrics + dashboard (CPU/API):** LLM + surface clustering, all stability metrics,
  the per-datapoint resample-spread dashboard tab.
- **7.4 — optional consensus-FVE (GPU).**

## 9. Decisions (RESOLVED at the 7.0 gate — user "go with defaults" + 3 amendments)

1. **Temperature grid {0.0(×1 greedy), 0.4, 0.7, 1.0, 1.5}** — added 1.5 as a stronger stress point (user);
   T=0 drawn once, not resampled (user).
2. **N = 24** stochastic resamples at each T>0.
3. **Doc subsample 200/model**, stratified into low/mid/high thirds by Exp-6 LCR divergence.
4. **First pass runs the no-injection floor AND shuffled-activation** (both needed to read the
   stable×not-in-input 2×2 cell); **consensus-FVE (7.4) deferred** as a follow-on.
5. **Cluster per-tag** (genre / LCR / PC), tracking H3.
6. **Meaning-judge validated** at Step 7.0: 0 false-merges, 95.1% pairwise agreement on the labelled set
   (incl. the vacuum-topic trap); conservative (splits rather than merges) → any reported core is a lower
   bound.
7. **Dashboard preliminary examples** wired from the 7.1 pilot before the full run (user).
