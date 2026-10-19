# Independent vetting of the activation-space mimic audit (Δcos / FVE)

Date: 2026-06-18. Vetting done from a local checkout (no GPU, no cluster `gold.npy`),
plus live Neuronpedia API calls and web literature checks. Method: I re-derived the
math from first principles, recomputed every headline number from the stored per-row
data, reproduced numbers against the live hosted NLAs, confirmed the literature the
explanation rests on, and ran a six-dimension adversarial review that red-teamed the
four load-bearing conclusions.

---

## 1. Bottom line in one paragraph

The experiment is sound. Its central result holds: the injected activation carries
real information beyond what the surrounding text alone implies, in all four models.
The "odd cosine values on Gemma" that prompted this review are not a bug. They are a
well-documented geometry artifact (Gemma's residual stream is dominated by one or two
enormous, near-constant dimensions, so a plain cosine reads about 0.99 for almost any
text, even a deliberately wrong one). The fix already applied, switching the headline
metric from raw cosine to FVE (the paper's own metric, which subtracts that shared
bias), is the correct and literature-standard fix, and it is not a thumb on the scale:
the activation's advantage is already statistically visible in the un-processed cosine
before any metric change. I found no calculation error. I found one documentation
mismatch worth fixing (the scored mimic is a different calibration than the one the
write-up's fairness story describes, though the mismatch is conservative), two prose
overstatements, and a set of honest verification ceilings (some links can only be fully
closed on the cluster with a GPU).

### Verdict table

| Conclusion | Verdict | Confidence |
|---|---|---|
| The FVE metric is faithful to the paper; the cosine→FVE switch is a real fix, not a fudge | Sound | High |
| The activation carries genuine payload beyond the text prior (ΔFVE>0, mse-ratio>1, all 4 models) | Sound | High |
| No calculation bug; the numbers reproduce on Neuronpedia | Sound, with scope caveats | High |
| The cross-model comparison is honest | Sound in substance, one sentence overstated | High |

---

## 2. What the experiment is, in plain English

An NLA (Natural Language Autoencoder) is a pair of models. The **AV** reads a single
internal activation vector from a target model and writes a short text description of
it. The **AR** reads that text and tries to rebuild the original activation vector. If
the rebuilt vector points the same way as the original (high cosine), the description
captured the activation well.

The audit asks a sharp question: **how much of that description actually came from the
activation, versus what any smart reader could have guessed from the text on the page?**

To answer it, for each read position the audit compares three rebuilds:

- **real**: rebuild from the AV's actual read of the activation.
- **mimic**: rebuild from a "text-only" description. A strong model (Opus 4.8) is shown
  the same passage but NOT the activation, and writes a same-style description. This is
  the "what the text alone implies" baseline.
- **control**: rebuild from a mismatched description (another position's read). This is
  the "wrong answer" floor.

The payload is `real minus mimic`. If it is near zero, the activation added nothing the
text did not already give away. If it is large, the activation carried specific content
a text-only reader could not have known.

**Running example (qwen, idx 115 / token position 70, a tool-call transcript).** The real read names
the exact on-screen content: a food-processing barcode error, "Product code: CITRUS
Juice." The text-only mimic guesses a plausible but wrong domain (a warehouse SAP/ERP
stock lookup). The real rebuild scores cosine 0.88; the mimic 0.29. The activation
carried the specific entity the page had not yet spelled out. That gap is the payload.

---

## 3. The Gemma "odd cosine" worry, explained intuitively

This is the thing that triggered the review, so it gets its own section.

**The symptom.** On Gemma, cos_real, cos_mimic, and even cos_control all sit around
0.94 to 0.99. The real read barely beats a deliberately wrong read. Taken at face
value, that says "the activation adds almost nothing," which contradicts what we see on
Llama and Qwen.

**Why it happens (the picture).** Imagine every activation vector as an arrow. On most
models the arrows point in many directions, so cosine is informative. On Gemma, almost
every arrow is dragged toward one giant shared direction, because one or two coordinates
of the vector are enormous (tens of thousands in magnitude) and nearly constant across
all positions. Picture every arrow forced to start by pointing 93 to 95 percent of the
way toward the same wall. Any two arrows then look almost parallel (cosine about 0.92),
no matter what they actually encode. So a wrong rebuild still scores about 0.94, purely
by reproducing that shared wall-pointing component. The cosine is saturated: it is
measuring the giant shared bias, not the content.

**This is a known, named phenomenon, and I verified the citations:**

- Rogue dimensions: Timkey and van Schijndel, EMNLP 2021, "All Bark and No Bite: Rogue
  Dimensions in Transformer Language Models Obscure Representational Quality." A handful
  of dimensions dominate cosine and make it misleading; the recommended fix is
  per-dimension standardization. Confirmed.
- Massive activations: Sun et al., 2024, arXiv 2402.17762, "Massive Activations in Large
  Language Models." A few activations sit thousands of times above the median and act as
  near-constant biases, concentrated in fixed dimensions (the paper gives examples like
  about 13000x for a 7B model). Confirmed.
- Gemma's amplifier: Gemma multiplies its token embeddings by the square root of the
  hidden size, which inflates residual-stream norms and makes the effect extreme.
  Confirmed in the HuggingFace Gemma-3 modeling code.

**The fix, and why it is not cheating.** FVE (Fraction of Variance Explained) is the
paper's own metric. It divides the reconstruction error not by the raw vector size but
by the variance *around* the shared mean. In the arrow picture, it ignores the common
wall-pointing part and asks only how well you matched the part that actually varies from
position to position. Under FVE, the Gemma payload is large (about +0.33), not zero.

I checked four times over that this is a real fix and not a metric picked to get a
desired answer:

1. The raw cosine advantage was already positive for Gemma before any metric change,
   and a paired test on the raw cosines (no FVE, no mean-centering at all) is already
   overwhelmingly significant (Wilcoxon p about 3e-22 for Gemma-27B). FVE only
   un-compressed a signal that was already there.
2. A completely separate de-biasing method, full per-dimension z-standardization, gives
   the same answer: the control collapses from cosine 0.94 to 0.045, the real read
   survives, and the per-genre payload from standardization correlates 0.98 with the
   per-genre payload from FVE. Two unrelated methods agree.
3. Neuronpedia, which hosts this exact Gemma NLA, returns the same saturated cosines
   (median about 0.99). So the saturation is real and external, not a local bug.
4. Neuronpedia's own per-NLA `norm` field (Gemma 0.0579, Llama 0.9169) is the same
   variance normalizer FVE uses, and it matches our independently computed denominator
   (Gemma 0.0745, Llama 0.886) in ordering and rough magnitude.

Conclusion: the Gemma cosine is correct but uninformative, exactly as the write-up
already says. FVE is the honest metric and it is applied correctly.

---

## 4. What I verified, and how

### 4a. The math is faithful to the paper (re-derived and recomputed)

- `NLACritic.score` normalizes both the predicted and the gold vector to length
  sqrt(d), then reports mse = mean squared difference. With both at that length, this is
  exactly `2(1 - cos)`. Verified algebraically and numerically (error about 1e-15), and
  confirmed live on Neuronpedia (`mse == 2(1-cos)` to within 7e-7 over 44 positions).
- `fve.py` computes `FVE = 1 - sum 2(1-cos) / sum D`, where D is each unit gold vector's
  squared distance from the mean unit gold. This is the paper's `1 - E||p-h||^2 /
  E||h-hbar||^2` on unit-normalized activations; the sqrt(d) scaling cancels in the
  ratio, and "predict the mean" gives exactly FVE = 0. Faithful.
- It uses the correct sum-weighted aggregate. The alternative (averaging per-position
  FVE) would have inflated Gemma by about 0.08 and is unstable for tiny-variance rows;
  the code does not use it.
- Every published headline number reproduces bit-for-bit from the local per-row data,
  for all four models. All 128 rows join cleanly with no row-to-gold scrambling.

### 4b. The conclusion is statistically solid (bootstrap + paired tests)

A 20,000-sample bootstrap over the 128 positions:

| model | ΔFVE 95% CI | mse-ratio 95% CI | paired Wilcoxon p |
|---|---|---|---|
| llama70b | [0.10, 0.21] | [1.30, 1.60] | 3e-10 |
| qwen7b | [0.37, 0.48] | [2.08, 2.49] | 1e-22 |
| gemma12b | [0.21, 0.40] | [1.73, 2.04] | 1e-18 |
| gemma27b | [0.25, 0.41] | [1.92, 2.28] | 3e-22 |

Every ΔFVE interval is strictly above 0; every mse-ratio interval is strictly above 1.
The conclusion does not depend on a few lucky rows (drop-one jackknife moves ΔFVE by at
most 0.07) and is not dominated by the tiny-variance Gemma rows (the five smallest hold
under 3.3 percent of the total weight).

### 4c. The injection and extraction are faithful (the Gemma √d worry resolved)

This was the most important code-level concern, since the "real" reads for Qwen and
Gemma are generated locally by injecting the activation.

- **The Gemma square-root-d scaling does not cause a scale error.** The local generator
  (`avgen_model.py`) overwrites the marker token's hidden state at the input to layer 0
  with the activation rescaled to `injection_scale`. The canonical reference path
  (`NLAClient._build_embeds`) does the same: it applies the sqrt(d) scaling to the other
  tokens, then overwrites the marker to `injection_scale` afterward. In both paths the
  marker ends up at exactly `injection_scale`, and the sqrt(d) factor lands only on the
  non-marker tokens. A numerical simulation of both paths gave an identical marker norm.
  The feared 62x error does not occur.
- The extraction layer `hidden_states[K+1]` is the output of block K (the layer the NLA
  reads), matching the training extractor. The layers (Llama 53/80, Qwen 20/28,
  Gemma-12B 32/48, Gemma-27B 41/62) all sit at about two-thirds depth and match the
  released NLA names and the Neuronpedia `layerNum`.
- `injection_scale` is read from each sidecar, not hardcoded: Qwen 150, Gemma-12B 80000,
  Gemma-27B 60000, Llama 30. `mse_scale` is sqrt(d) for all four, which is what makes
  the `mse = 2(1-cos)` identity exact.
- The injection-failure smell test (Chinese-character output when injection misfires)
  is clean: 1 occurrence in 384 generated reads. A 62x scale error would have produced
  mass garbage.

### 4d. No calculation bug, checked against the live hosted NLAs

I queried the live Neuronpedia API (44 verbalized positions across the two hosted
models):

- `mse == 2(1-cos)` to 7e-7. The core identity holds on the real server.
- Gemma: NP cosine 0.991 vs our 0.992 (per row within 0.003); NP `l2_norm` 40,000 to
  72,000 (the massive-activation signature, matching our extracted gold scale); and the
  NP-hosted read TEXT describes the same content as our locally generated read, several
  near-verbatim. This validates the local injection path end to end for Gemma-27B.
- Llama: an independent live re-verbalization correlates with our stored numbers at
  pearson 0.98. (The write-up's headline 0.9995 is a tighter self-consistency check; see
  the scope caveat in section 6.)

---

## 5. Issues found

None of these change the central conclusion. They are ranked by how much they matter.

### 5a. Moderate: the scored mimic is v2_3, not the near-chance v1_1 the fairness story describes

The audit scores the **v2_3** mimic (`build_run_input.py` reads `calib_sample_v2_3.json`
and `calib_out_v2_3/`). Its calibration report says its style discriminator sits at
**29.7 percent**, labeled an "overshoot (judge picks the mimic as genuine; caricature)."
The "carefully calibrated to near-chance, no caricature" story (discriminator 47.2
percent, all style signatures in band) is the **v1_1** set (`prompt_final.txt`), which
was **not** the one scored.

Why this does not break the conclusion, and is in fact conservative: a discriminator at
29.7 percent means the mimic reads *more* like a prototypical NLA description than the
real reads do. That makes it a *stronger* text-only baseline, so "the activation beats
the mimic" is harder to clear, not easier. And the property the audit actually depends
on is reconstruction quality, measured directly: the v2_3 mimic recovers 69 to 90
percent of the recoverable signal (the real-minus-control span) and sits far above the
control in 125 to 127 of 128 rows. It is a genuinely strong baseline, not a strawman.

Why it still matters: the fairness evidence cited in the documentation describes a
different mimic than the one in the headline numbers. Two of the six review agents
anchored on the wrong report before the adversarial pass caught it.

Recommendation: either re-document the mimic's fairness using v2_3's actual
reconstruction evidence (strong, conservative), or re-run the audit with the v1_1 locked
near-chance mimic so the numbers and the calibration story match. The first is cheaper
and sufficient.

### 5b. Minor: ΔFVE is mislabeled as "D-independent"

`RESULTS_cross_model.md` (lines 110, 161-163) groups ΔFVE with the mse-ratio as the
"trustworthy comparatives" because "systematic offsets cancel." That conflates two
different properties. ΔFVE cancels a systematic *cosine* offset, but it equals
`2 * sum(cos_real - cos_mimic) / sum(D)`, so its *magnitude* scales as 1/D and inherits
the very same D-sensitivity the write-up correctly flags for FVE_real (a 20 percent D
error moves both by about 20 percent). The genuinely D-independent quantities are: the
**sign** of ΔFVE, the **win count** (real beats mimic per row), and the **mse-ratio**
`(1-cos_mimic)/(1-cos_real)`. The substantive conclusion is unaffected, because those
three D-free statistics already carry it, but the robustness wording overstates the case.

Recommendation: reserve the "D-independent" label for the sign, the win count, and the
mse-ratio; state that ΔFVE's magnitude is no more D-robust than FVE_real.

**Resolved 2026-06-18.** `RESULTS_cross_model.md` (the gemma-sensitivity caveat) now
states that ΔFVE = 2·Σ(cos_real − cos_mimic)/Σ D scales as 1/D and is no more D-robust
than FVE_real, and reserves the D-independent label for the sign, win count, and
mse-ratio. The viewer prose never made this claim (it already restricts cross-model
statements to the mse-ratio ordering).

### 5c. Minor: one cross-model magnitude sentence is not defensible

The write-up mostly restricts cross-model claims to the sign and the D-free mse-ratio,
which is correct. But one sentence (line 110, Gemma "+0.33, comparable to qwen, larger
than llama") compares absolute ΔFVE magnitudes across models. The "comparable to qwen"
half is not supported: the Qwen and Gemma-27B bootstrap intervals overlap, and a D
perturbation within sampling noise flips their order. Only "larger than llama" survives
(non-overlapping intervals). This is one sentence stepping outside the document's own
stated scope.

Recommendation: keep cross-model comparison to the mse-ratio ordering (Qwen 2.27 >
Gemma-27B 2.07 > Gemma-12B 1.89 > Llama 1.44) and the sign; drop or soften "comparable
to qwen."

**Resolved 2026-06-18.** `RESULTS_cross_model.md` now states that gemma's absolute ΔFVE
is not reliably separable from qwen's (overlapping bootstrap intervals, order flips
under a within-noise D perturbation), keeps only "larger than llama" (non-overlapping),
and routes the cross-model ranking through the D-free mse-ratio. The viewer already
phrased it this way (it compares only the error-ratio ordering and states that absolute
magnitude is not comparable across models).

### 5d. Nits

- The control is a single deterministic neighbor (75 percent same-genre), which makes it
  a hard, slightly noisy floor. It is not used in ΔFVE or the mse-ratio, so it is not
  load-bearing; averaging over several random mismatches would be a robustness nicety.
- The Sun et al. citation uses the paper's body ratios (10^3 to 10^4 x median); the
  abstract headlines 10^5. No correction needed, optionally note it.
- FVE here is a direction-only (unit-normalized) metric by construction, matching the
  paper's normalized-MSE training. Worth one explicit sentence so a reader does not
  expect magnitude error to be captured.

---

## 6. What I could NOT verify (honest ceilings)

These are not failures of the experiment. They are limits of vetting it from a laptop
without the cluster data or a GPU. They define what a fully airtight follow-up would
close.

1. **The raw gold vectors (`gold.npy`) are cluster-only.** Every local recompute is
   conditional on the stored per-row cosines and variances being correct. I verified the
   arithmetic *given* those, and verified them externally where Neuronpedia hosts the
   model, but the upstream gold extraction itself was not re-run here. A symmetric error
   would cancel (it would hit real and mimic equally), but an asymmetric error (for
   example a row-to-gold misalignment in extraction) is not excluded by any local check.
2. **No GPU, so the AR scoring was not re-executed.** The `mse = 2(1-cos)` identity is
   verified by algebra and on Neuronpedia, but the actual on-GPU scoring of the locally
   generated reads was not re-run.
3. **Qwen-7B and Gemma-12B are not hosted on Neuronpedia**, so their generated reads
   have no external ground truth. Their faithfulness rests on indirect signals (the clean
   injection smell test, the difficulty correlation, FVE landing in the paper's 0.6 to
   0.8 band). Note the asymmetry: Qwen carries the largest payload (+0.43) yet has the
   least external validation.
4. **The Llama Neuronpedia check is partly circular.** Llama's reads were reused *from*
   Neuronpedia, so the headline pearson 0.9995 is our scorer agreeing with NP's scorer on
   the same text, which validates the scoring code but not independent read generation.
   The genuinely independent live re-verbalization is noisier (pearson 0.98) due to an
   off-by-one position artifact on some chat documents.
5. **`std_full_llama70b.json` is missing**, so the cleanest de-biasing cross-check exists
   for three of four models, not Llama (which is least at risk, being unsaturated).

**Highest-value next check.** A single run would close gaps 1, 2, and 3 at once: feed one
identical activation through the local `avgen` path and through the canonical SGLang
`NLAClient` path on the two hosted models, and confirm the reads and cosines match within
sampling noise. That validates the local injection substitution end to end and transfers
that confidence to the two unhosted models, which share the code path. Second: log raw
gold-norm percentiles during extraction and assert the 75th percentile is near
`injection_scale`, turning the scale assumption from plausible into checked.

---

## 7. Reproduced numbers (all independently recomputed unless noted)

Headline FVE (sum-based aggregate, matches stored to 4 decimals for all four models):

| model | FVE_real | FVE_mimic | FVE_control | ΔFVE | mse-ratio | real>mimic | D |
|---|---|---|---|---|---|---|---|
| llama70b | 0.647 | 0.493 | -0.884 | +0.154 | 1.44 | 92/128 | 0.886 |
| qwen7b | 0.666 | 0.241 | -0.723 | +0.425 | 2.27 | 125/128 | 0.725 |
| gemma12b | 0.642 | 0.323 | -0.485 | +0.320 | 1.89 | 108/128 | 0.046 |
| gemma27b | 0.689 | 0.354 | -0.551 | +0.335 | 2.07 | 122/128 | 0.074 |

Mimic strength (not a strawman): mimic beats control in 125 to 127 of 128 rows; mimic
recovers 69 to 90 percent of the real-minus-control cosine span.

Standardization de-bias (decisive saturation test): Gemma-27B control cosine 0.94 →
0.045 under z-standardization; real survives (0.85, R^2 0.68); per-genre FVE-payload vs
standardized-payload correlation 0.98.

Live Neuronpedia: `mse == 2(1-cos)` mean error 7e-7 over 44 positions; Gemma cosine
0.991 vs our 0.992; Gemma `l2_norm` 40k to 72k; sources `norm` field Gemma 0.0579 /
Llama 0.9169 vs our D 0.0745 / 0.886.

Sidecars (fetched live from HuggingFace): injection_scale Qwen 150 / Gemma-12B 80000 /
Gemma-27B 60000 / Llama 30; mse_scale = sqrt(d) for all four; extraction layer 20 / 32 /
41 / 53.

Injection smell test: Chinese-character output 1 in 384 generated reads.
