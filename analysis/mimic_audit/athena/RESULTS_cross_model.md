# Activation-space audit — cross-model replication (2026-06-17)

Does the injected activation carry payload beyond the text prior? Measured on the
**four open NLAs** by holding the experiment fixed (same 128 documents, same read
LOCATIONS re-anchored per tokenizer, same text-only mimics) and varying only the
model: each model's own base (gold), AV (real read, generated on-GPU via layer-0
injection), and AR (cosine score). llama70b reused Neuronpedia reads; qwen/gemma
reads were generated locally.

> **2026-06-17 CORRECTION + 2026-06-18 HARDENING (metric).** The first pass reported
> raw cosine and Δcos. For gemma that was **saturated and misleading**: gemma-3's
> residual stream is so anisotropic that ~93–95% of the unit-activation energy is a
> single shared mean direction, so cosine reads ≈0.99 for *any* text (even a
> mismatched control) and Δcos collapses to ≈0. This is NOT "no payload" — it is the
> wrong metric. The fix is **per-dimension standardization** (z-score with the gold
> μ/σ), the literature-standard remedy for rogue dimensions (Timkey & van Schijndel
> 2021); it is now the **primary metric** across the audit and the GUI. Under it a
> mismatched control collapses to ≈0.045 (chance) for *all four* models while the
> real read survives and beats the mimic everywhere. Mean-centered **FVE** is
> retained as the paper-comparison cross-check (it agrees), and raw cosine only as a
> reference / for the Neuronpedia validation. **All four standardized runs are
> complete (incl. llama70b);** the concentrated-payload finding (code/math/
> specific-answer ≫ poetry/news/philosophy) survives de-biasing on every model.

## Headline — standardized cosine (primary, de-biased)

Per-dim z-score with the gold μ/σ removes the rogue/massive-activation dims, so
cosine measures content rather than 1–2 huge dimensions. The decisive control:

| model | std real | std mimic | **std control** | **Δstd = real − mimic** | act-specific % | real>mimic | std R²(real) |
|---|---|---|---|---|---|---|---|
| llama70b | 0.812 | 0.737 | **0.046** | **+0.076** | 9% | 90/128 | 0.623 |
| qwen7b   | 0.742 | 0.527 | **0.043** | **+0.215** | 29% | 124/128 | 0.182 |
| gemma12b | 0.837 | 0.683 | **0.044** | **+0.153** | 18% | 109/128 | 0.696 |
| gemma27b | 0.849 | 0.668 | **0.045** | **+0.181** | 21% | 121/128 | 0.677 |

Two facts settle the gemma skepticism for good:

1. **The control collapses to ≈chance (0.043–0.046) on every model.** The
   "random vector reconstructs at cos 0.94" effect was *entirely* the rogue
   dimensions; once they are standardized away, a mismatched read carries no
   signal — exactly as it must. (For gemma this is the whole story: raw control
   0.94–0.97 → standardized 0.045.)
2. **The real read survives and beats the mimic everywhere** (Δstd +0.076 to
   +0.215, real-wins 90–124/128). The injected activation carries genuine,
   text-prior-independent payload on all four NLAs. qwen's std R²(real) is low
   (0.18) because its standardized residual has large magnitude variance that
   cosine ignores but R² penalizes; the cosine and Δ are still the largest of
   the four, so the *direction* is well recovered.

(Cross-model magnitude is still not directly comparable — different AR
sensitivity and a shared llama-tuned mimic — but the sign + control-collapse +
concentration replicate identically.)

## Why raw cosine fails for gemma (and the fix)

The paper's metric is **FVE = 1 − 𝔼‖p − h‖² / 𝔼‖h − h̄‖²** on unit-normalized
activations: FVE = 0 means "no better than predicting the mean activation h̄",
FVE = 1 is perfect. Cosine (and MSE = 2(1−cos)) is *not* mean-centered. When the
activations cluster tightly around a dominant mean, cosine saturates near 1 while
FVE — which divides by the variance *around* the mean — stays honest.

Direct measurement of the gold geometry (no AR involved):

| model | gold ‖·‖ (mean) | energy in shared mean ‖h̄‖² | variance D = 1−‖h̄‖² | gold pairwise cos |
|---|---|---|---|---|
| llama70b | ~90 | 0.114 | 0.886 | ~0.17 |
| qwen7b | ~227 | 0.275 | 0.725 | ~0.27 |
| gemma12b | ~62k | **0.954** | **0.046** | **~0.92** |
| gemma27b | ~56k | **0.926** | **0.074** | **~0.92** |

gemma's gold vectors are near-parallel (any two sit at cos ~0.92), so a control
(mismatched) reconstruction still scores cos 0.94–0.97. That is the saturation.

## Mechanism: 1–2 rogue dimensions (why a "random" read scores 0.94)

This is the **rogue-dimensions / massive-activations** phenomenon, well documented:
Timkey & van Schijndel (EMNLP 2021), *"All Bark and No Bite: Rogue Dimensions in
Transformer Language Models Obscure Representational Quality"* — a handful of
dimensions (often 1–3) dominate cosine/Euclidean similarity and **make cosine
misleading**; the fix is per-dimension **standardization**. Sun et al. (2024),
*"Massive Activations in LLMs"* — a few activations sit 10³–10⁴× the median and act
as near-constant biases. Gemma-3 is an extreme case (√d embedding scaling + massive
activations), measured directly here:

| model | rogue dim(s) (share of mean energy) | top-dim value (σ) | gold pairwise cos: raw → centered → **standardized** |
|---|---|---|---|
| llama70b | dim 5114 = 44% | −6 (5) | 0.107 → −0.007 → −0.007 |
| qwen7b | dims 458,2570 = 91% | −112 (961) | 0.268 → 0.526 → −0.007 |
| gemma12b | **dim 2339 = 99.7%** | **69,528** (13,389) | 0.953 → −0.007 → −0.008 |
| gemma27b | **dims 104,2733 = 99.7%** | **43,394 / 33,036** | 0.925 → −0.008 → −0.008 |

Two facts pin the mechanism:

1. **`cos_control` ≈ energy-in-shared-mean.** llama 0.165 vs 0.114; qwen 0.376 vs
   0.275; gemma12b 0.966 vs 0.954; gemma27b 0.942 vs 0.926. A mismatched read scores
   high *only* because it reproduces the constant rogue dimensions — it carries no
   specific signal.
2. **Standardize, and the control collapses to chance.** Running the AR end-to-end
   and scoring under raw / mean-centered / standardized (z-scored) representations:

   | gemma27b | raw cos | centered cos | **standardized cos** | std R²(FVE) |
   |---|---|---|---|---|
   | real | 0.988 | 0.584 | **0.849** | 0.677 |
   | mimic | 0.976 | 0.498 | 0.668 | 0.355 |
   | control | 0.942 | 0.076 | **0.045** | −1.11 |

   | gemma12b | raw cos | centered cos | **standardized cos** | std R²(FVE) |
   |---|---|---|---|---|
   | real | 0.992 | 0.745 | **0.837** | 0.696 |
   | mimic | 0.984 | 0.654 | 0.683 | 0.469 |
   | control | 0.966 | 0.105 | **0.044** | −0.87 |

   The "random vector reconstructs at 0.94" effect **vanishes** under standardization
   (control → 0.045, R² → −1.1: worse than predicting the mean, exactly as a
   mismatched read should be). The real read still reconstructs the de-biased signal
   (standardized cos 0.85, R² 0.68 — back in the paper's 0.6–0.8 band), and the
   activation payload survives (real − mimic = +0.18, real − control = +0.80). So
   there is **no fundamental problem**: cosine was reporting 1–2 massive dimensions,
   not reconstruction quality. (Note qwen: mean-centering alone is *insufficient*
   — its rogue dims have huge variance, so centered pairwise cos rises to 0.53;
   only standardization fully removes them. For gemma, centering already suffices,
   so the FVE numbers below are sound; standardization confirms them. qwen control
   end-to-end: raw 0.376 → centered 0.382 → **standardized 0.043** — same collapse,
   but only standardization gets there.)

## Headline in FVE (the honest metric)

| model | FVE_real | FVE_mimic | FVE_control | **ΔFVE** | activation-specific % (ΔFVE/FVE_real) | real>mimic | (raw Δcos, old) |
|---|---|---|---|---|---|---|---|
| llama70b | 0.647 | 0.493 | −0.88 | **+0.154** | 24% | 92/128 | +0.068 |
| qwen7b   | 0.666 | 0.241 | −0.72 | **+0.425** | 64% | 125/128 | +0.154 |
| gemma12b | 0.642 | 0.323 | −0.48 | **+0.320** | 50% | 108/128 | +0.007 |
| gemma27b | 0.689 | 0.354 | −0.55 | **+0.335** | 49% | 122/128 | +0.012 |

Three things fall out:

1. **FVE_real ≈ 0.64–0.69 for all four models** — squarely in the paper's stated
   "0.6–0.8 FVE" band. The four NLAs reconstruct to the *same* medium fidelity;
   raw cosine only made them look different (0.84 vs 0.99) because of activation
   geometry. This also validates the whole pipeline (gemma included) against the
   paper's reported range.
2. **ΔFVE reverses the gemma conclusion.** Raw Δcos said gemma's activation added
   almost nothing (+0.01, the *smallest* of the four). FVE shows it adds a *large*
   payload (+0.33), larger than llama (non-overlapping bootstrap intervals). Its
   absolute ΔFVE is *not* reliably separable from qwen's: the two bootstrap intervals
   overlap and a D perturbation within sampling noise flips their order, so the only
   defensible cross-model ranking is by the D-free mse-ratio (qwen 2.27 > gemma27b
   2.07 > gemma12b 1.89 > llama 1.44). The earlier "gemma activation adds nothing"
   was a pure metric artifact.
3. **FVE_control is negative everywhere** (−0.48 to −0.88): a mismatched read
   reconstructs *worse* than predicting the mean — the correct behaviour, hidden
   entirely by cosine (where gemma control looked like 0.94).

The most robust, D-independent cross-model statement (a ratio, so the saturation
cancels): reconstructing from the mimic costs **mse_mimic/mse_real = 1.4–2.3×**
the direction-MSE of the real read (llama 1.44, qwen 2.27, gemma12b 1.89,
gemma27b 2.07).

## Neuronpedia cross-check (gemma27b is hosted)

The cosines are *correct* — they just measure the wrong thing. Verified against
Neuronpedia's hosted gemma-3-27b NLA (modelId `gemma-3-27b-it`, source `kitft-l41`)
on the same documents:

- **286 verbalized positions across 24 genres: cosine median 0.99, mean 0.977,
  95% > 0.95** — the same saturation we measure locally.
- Per-position **mse == 2(1−cos)** exactly (confirmed on the llama corpus too), and
  the API returns **l2_norm ≈ 6×10⁴** for gemma — matching our extracted gold norms
  (25k–76k). So our gold extraction and AR scoring reproduce Neuronpedia exactly.
- At the read tokens specifically, NP cosine (≈0.985) ≈ our cos_real (≈0.992).

So there is **no calculation bug**: gemma genuinely has saturated cosines, on
Neuronpedia and locally. What looks "medium" (and what the paper headlines) is the
**FVE ≈ 0.6–0.7**, which both we and the geometry confirm.

## Structure replicates under standardization (where the activation is load-bearing)

The concentration is the same finding under the *primary* (standardized) metric —
not an FVE-only artifact. Top vs bottom Δstd genres per model:

| model | top-Δstd genres | bottom-Δstd genres |
|---|---|---|
| llama70b | math_cot +0.38, code_python +0.23, interview +0.17, recipe +0.14 | poetry −0.03, finance −0.02, philosophy −0.02, news_tech ≈0 |
| qwen7b | tool_misreport +0.48, instructions +0.39, debate +0.37 | finance +0.02, history +0.08, email +0.09 |
| gemma12b | math_cot +0.34, tool_misreport +0.31, code_web +0.28 | finance −0.01, news_sports +0.03, news_tech +0.03 |
| gemma27b | code_web +0.39, math_cot +0.38, trivia_short +0.32 | news_tech +0.05, philosophy +0.06, poetry +0.07 |

- **By genre:** code / math / tool-misreport / specific-answer carry the largest
  activation-specific payload; poetry / news / finance / philosophy / email the
  smallest (the text prior already wins there). Identical split across all four
  models, and identical to the raw-cosine and FVE orderings — de-biasing changes
  the *magnitudes*, not *which genres* are load-bearing.
- **By difficulty** (stored-cosine band): U-shaped — Δstd largest in the hardest
  (<0.60) band, dipping mid-range, rising again at ≥0.95. Replicates across models.
- (FVE agrees: gemma27b ΔFVE biggest in code_web +0.62, math_cot +0.54, trivia
  +0.50; smallest poetry/philosophy/news_tech +0.08–0.13.)

## Gates / validity

| model | reads source | injection-clean (no CJK) | anchor ok | difficulty corr | cosine validation |
|---|---|---|---|---|---|
| llama70b | Neuronpedia | n/a (reused) | 128/128 token-match | — | pearson 0.9995 vs stored |
| qwen7b | generated | 128/128 | 122/128 | +0.40 | n/a |
| gemma12b | generated | 127/128 | 120/128 | +0.24 | n/a |
| gemma27b | generated | 128/128 | 120/128 | +0.24 | NP cosine matches (above) |

## Caveats

- **Standardized cosine is the robust primary; FVE is the paper-comparison.** The
  μ/σ for both are estimated from the 128-position gold sample, so absolute
  standardized values carry sampling noise, but the control-collapse and Δstd are
  differences on matched gold (offsets cancel) and are stable.
- **gemma FVE absolute values are sensitive.** Because D ≈ 0.05–0.07 is tiny, FVE
  = 1 − 2(1−cos)/D amplifies small cosine differences (∂FVE/∂cos = 2/D ≈ 27–44).
  A 0.01 cosine error shifts FVE by ~0.3. This is *why* standardized cosine (not
  FVE) is the headline. ΔFVE shares this sensitivity, not immunity to it: ΔFVE =
  2·Σ(cos_real − cos_mimic)/Σ D also scales as 1/D, so its *magnitude* is no more
  D-robust than FVE_real. The genuinely D-independent comparatives are the **sign**
  of ΔFVE, the per-row **win count**, and the **mse-ratio** (1−cos_mimic)/(1−cos_real);
  those three carry the cross-model conclusion. Treat FVE_real ≈ 0.65 as "medium",
  not a precise point.
- The mean activation h̄ (hence D) is estimated from the 128-position sample.
- n = 4 per genre; genre-level numbers are indicative.
- Single mimic set, llama-style-calibrated and reused across models.
- One layer per model.

## Artifacts
- Harness: `athena/repl/` (repl_common, build_anchors, extract_model, avgen_model,
  score_model, **std_recompute.py** (de-biased primary), **fve.py**, **rogue.py**,
  **diag.py**). Per-model outputs:
  `athena/out_<key>/{gold.npy,manifest.json,av_reads.json,results.json}`;
  standardized in `athena/fve/std_full_<key>.json`, FVE in `athena/fve/fve_<key>.json`
  (all four models complete).
- Neuronpedia cross-check: `analysis/mimic_audit/np_xcheck.py` (paired, read-token)
  and `np_xcheck2.py` (distribution over genres) → `np_xcheck*_result.json`.
- Viewer: `corpus/viewer.html` "Activation audit" tab — **standardized-cosine-led**
  and now a full **14-section dossier** per model (what-it-measures, design+config,
  geometry table across all 4 models, all-metrics fidelity, bootstrap/Wilcoxon
  statistics, Neuronpedia external validation, injection faithfulness, gates,
  by-genre/band/position, caveats+ceilings, literature). The per-model dossier
  (`DOSSIER`) + shared suite prose (`SUITE`) live in `build_act_audit_view.py` and
  emit as `entry.dossier`/`entry.suite`; `build_act_audit_view.py <key>` merges
  `std_full_<key>.json` + `fve_<key>.json` → `corpus/act_audit_<key>.js` (all four
  rebuilt + render-tested via a DOM-stub harness).
