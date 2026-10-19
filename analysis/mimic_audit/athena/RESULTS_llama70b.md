# Activation-space audit — llama70b (2026-06-17)

**Question.** How much of an NLA explanation is genuinely carried by the injected
activation, versus recoverable from the text prior alone? Measured in *activation
space* on the live `Llama-3.3-70B-NLA-L53` model, over the 128 v2_3 calibration
positions that already have text-only mimics.

**Method.** For each read token: extract the gold residual `hidden_states[54]`
from the base model (raw, no-norm), then have the AR (`NLACritic`) reconstruct it
from three texts:
- `cos_real`    = AR.score(real AV read, gold)
- `cos_mimic`   = AR.score(v2_3 mimic, gold)   ← the experiment
- `cos_control` = AR.score(another item's real read, gold)  ← chance floor

Headline metric **Δcos = cos_real − cos_mimic** (≈0 → activation adds nothing
beyond the prior; ≫0 → genuine activation payload). MSE is redundant with cos:
`mse = 2(1 − cos)`, both vectors L2-normalised to `mse_scale = √8192 = 90.51`.

## Gates (both passed)
- **Alignment** (extract): token_match **128/128 (100%)**, in_range 128/128. Corpus
  tokenisation replicated exactly (raw text / `format_chat` llama3 markers,
  `add_special_tokens=True`).
- **Validation** (score): recomputed `cos_real` vs stored Neuronpedia cosine —
  mean|err| **0.0019**, max 0.0373, **pearson 0.9995**. Pipeline proven; mimic
  numbers trustworthy.

## Headline (N = 128)
| metric | cos | mse = 2(1−cos) |
|---|---|---|
| real  | **0.844** | 0.313 |
| mimic | **0.776** | 0.449 |
| control | 0.165 | 1.669 |

- mean **Δcos = +0.068**, median +0.037
- real beats mimic **91/128 (71%)**; real beats control **128/128**
- Δ distribution is heavy-tailed: min −0.128, p10 −0.029, median +0.037, p90 +0.233, **max +0.892**

## Reading
Both real and mimic crush the control floor (0.84 / 0.78 vs 0.17), so most of the
reconstructable signal is recoverable from context + genre + position: the
text-only mimic captures **~92%** of the real read's cosine (0.776 / 0.844). But
the real read carries a genuine, activation-specific increment that is
**concentrated, not uniform**.

**Where the activation is load-bearing** (Δcos by genre, top):
- math_cot **+0.378** (mimic 0.432 — the prior fails badly), code_python +0.210,
  interview_dialogue +0.152, translation +0.133, recipe_howto +0.131,
  code_web +0.113, arithmetic_short +0.108, alignment_eval +0.107, history +0.103.
- These are positions where the local token is a *specific non-inferable value*
  (a CoT number, a code token, a translated word) — the activation supplies it,
  the paraphrase loses it. Biggest single win: math_cot pos125 real 0.985 vs
  mimic 0.093 (Δ +0.892).

**Where the prior already wins** (Δcos ≈ 0 or negative):
- email +0.010, forum_personal +0.011, news_politics +0.012, medical +0.017,
  news_tech −0.003, philosophy −0.010, finance −0.020, poetry −0.020.
- Highly genre-predictable local content; the mimic ties or beats the real read.
- 35/128 (27%) have Δ<0, but all small-magnitude (min −0.128).

**By stored-cosine band** (Δcos is U-shaped — largest where reconstruction is hard):
| band | n | real | mimic | Δ |
|---|---|---|---|---|
| <0.60 | 11 | 0.522 | 0.371 | **+0.151** |
| 0.60–0.80 | 20 | 0.753 | 0.650 | +0.102 |
| 0.80–0.90 | 43 | 0.856 | 0.808 | +0.048 |
| 0.90–0.95 | 39 | 0.923 | 0.885 | +0.038 |
| ≥0.95 | 15 | 0.961 | 0.862 | +0.099 |

## Interpretation
Activation-space support for **evidence-gated prior substitution**
([[nla-failure-research-program]]): the AV read defaults to the text prior, and
the injected activation only becomes load-bearing when the prior is insufficient —
specific-value tokens and hard-to-reconstruct (low-cosine) positions. When local
content is predictable from context, real ≈ mimic (the activation adds nothing
measurable); when it is not, real ≫ mimic (Δcos up to +0.89). This is the
activation-space counterpart to the claim-space Axis-1 result.

## Caveats
- n = 4 per genre (32 genres); genre-level Δ is indicative, not significant.
- Single mimic set (v2_3), temperature > 0 at generation; not averaged over
  resamples.
- One layer (K=53), one model. cos/mse are the same measurement (mse = 2(1−cos)).

## Artifacts
- Code: `extract_llama.py`, `score_llama.py`, `nla_inference.py` (patched:
  2-GPU `device_map` + `ignore_mismatched_sizes=True` for the AR's stray
  `lm_head` shard tensor), `build_run_input.py`, `wait_and_run.sh`, `score_now.sh`.
- Data: `out/results.json` (per-row), `out/gold.npy` [128, 8192], `out/manifest.json`.
- Run: Athena, base then AR sequential on 2×H100 (`device_map="auto"`), ~1 min
  extract + ~1 min score once GPUs were free.
