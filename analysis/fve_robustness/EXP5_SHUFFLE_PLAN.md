# Exp 5 — shuffle claims at random (complete plan)

**Status: COMPLETE (2026-08-27).** All three arms were scored on 2,000 rows × 3 models; all null
gates passed with `max|Δcos|=0`. Before the final 5c run, the implementation was corrected to freeze
the first post-`Final token` quote (the read-token ECHO). Final ΔFVE (95% document-bootstrap CI):

| Arm | Qwen2.5-7B | Gemma-3-12B | Gemma-3-27B |
|---|---:|---:|---:|
| 5a block | −0.0511 [−0.0547, −0.0479] | −0.0182 [−0.0199, −0.0167] | −0.0226 [−0.0243, −0.0210] |
| 5b sentence | −0.0396 [−0.0419, −0.0374] | −0.0181 [−0.0195, −0.0168] | −0.0228 [−0.0247, −0.0211] |
| 5c fine claim-slot | −0.1121 [−0.1197, −0.1046] | −0.0312 [−0.0335, −0.0290] | −0.0215 [−0.0232, −0.0198] |

Thus order is not invariant. Fine local claim-slot order matters beyond coarse block order for Qwen
and Gemma-12B; the two grains are statistically indistinguishable for Gemma-27B.

**Question.** Is the AR's reconstruction *order-invariant* — a bag of claims it re-embeds regardless of
sequence — or does it depend on the explanation's narrative order? If FVE is preserved under shuffling,
the round-trip is content-driven, not sequence-driven. We test this at two grains: coarse (block order)
and fine (the claim order *inside* the final-token block).

## What the explanations actually look like (measured, n=6000)
| | Qwen | Gemma-12B | Gemma-27B |
|---|---|---|---|
| blocks / expl | 3 (2000/2000) | 3 (1998) | 3 (1997) |
| sentences / expl (quote-masked) | mostly 3 | mostly 3 | mostly 3 |
| **sentences in the last block** | **1 (1999/2000)** | 1 (1874) | 1 (1992) |
| quoted continuations in last block | mean 5.7 (med 6) | 7.1 (med 7) | 8.5 (med 8) |

The three blocks are: (0) genre/register line, (1) the "sentence structure/phrase" claim + its quote,
(2) the "Final token X" block = **one sentence** enumerating ~6–8 quoted predicted-continuations.

**Implication.** Because each block is ~1 sentence, a plain sentence-shuffle collapses to the
block-shuffle for ~90% of datapoints — it is nearly redundant with 5a. The real fine-grained "claims"
are the enumerated quotes in the last block. So we define the fine shuffle there.

## Claim taxonomy (the shufflable units)
`block` (3)  ⊃  `sentence` (≈1 per block)  ⊃  `quoted claim` (~6–8, only in the last block)

## The three arms
### Exp 5a — block shuffle (coarse) — *as you specified*
Split the explanation on blank-line boundaries (`\n\s*\n+`) into its blocks (≈3), shuffle their order
with a seeded non-identity permutation, rejoin with `\n\n`. Each block stays internally intact. Tests
whether coarse claim order matters.

### Exp 5b — sentence shuffle (medium)
Flatten to quote-aware sentences (segmenter below), shuffle, rejoin with spaces. **Honest caveat:**
by the measurements above this equals 5a for ~90% of datapoints and only differs where block 0 or 1
splits into 2 sentences. Included for completeness / a clean gradient, but expected ΔFVE ≈ 5a.

### Exp 5c — quoted-claim shuffle (fine) — *resolves the "quotes block" vagueness*
Within the final-token block, find its K quoted claim spans (the ~6–8 predicted continuations + the
lead-in clause quote), freeze the first quote after `Final token` (the read-token ECHO), and **permute
the claim CONTENTS among their slots**, keeping all connective scaffolding
("… expecting ⟨A⟩ or ⟨B⟩ or ⟨C⟩ …") fixed. No sentence-splitting needed inside the quotes block — a
"claim" is a quoted span, and we shuffle which claim sits in which slot. This is the finest, most
informative shuffle and the one the quotes block actually enables.

**Recommendation:** run **5a + 5c** (coarse block order vs fine within-list claim order — the two
genuinely different questions); treat **5b as optional** since it is near-redundant with 5a. Open to
running all three if you want the full gradient.

## Shuffle mechanics (deterministic, edit-only)
- **Quote-aware sentence segmenter** (5b): mask every double-quoted span (straight + curly) to a
  punctuation-free placeholder so no split occurs inside a quote; split on `[.!?]` + whitespace +
  capital/quote; suppress splits after a small abbreviation set (`e.g. i.e. U.S. Dr. Mr. Mrs. Ms. etc.
  vs. No. Fig.`) and after single-letter / bare-number list markers (`"4."`, `"G."`); unmask.
- **Quoted-claim parser** (5c): reuse `_straight_spans`/`_curly_spans`/`_coalesce` on the last block;
  freeze the first quote after `Final token` as ECHO, then permute the remaining claims' inner texts
  across slots.
- **Seeded permutation:** `random.Random(md5(expl_orig))` — stable per explanation, reproducible, no
  cluster/doc-id plumbing needed. Reject the identity permutation for n≥2 (reshuffle); n≤1 → no-op.
- **No-op accounting:** datapoints with <2 units are logged and reported as a share so the aggregate
  ΔFVE isn't misread (5a: ~0 no-ops since blocks=3; 5c: no-op when the last block has <2 quotes).
- **Edit-only invariant:** gold activation, read position, doc, and AR are frozen; only the explanation
  text is reordered, so ΔFVE is attributable per datapoint exactly as in Exp 1–3. Shuffling preserves
  characters, so any FVE move is purely a function of order.
- **edit_meta:** `{kind: block|sentence|quoted-claim, n_units, permutation, noop: bool}` (no
  removed/kept — this is a reorder).

## Dashboard
New arms `exp5a_blocks`, `exp5c_claims` (and `exp5b_sentences` if run). The per-datapoint tab renders
the original with unit boundaries tinted and the edited (shuffled) text with the same units in their
new order + the explicit permutation (e.g. `[0,1,2] → [2,0,1]`); FVE per datapoint once re-scored. A
shuffle-aware summary line replaces the chars-changed line (shows n_units + no-op share).

## Interpretation
- 5a preserved but 5c drops ⇒ the AR is insensitive to coarse order but sensitive to the fine claim
  order within the predicted-continuation list.
- 5a already collapsing ⇒ sequence-sensitive even at block grain.
- All preserved ⇒ order-invariant bag-of-claims inversion (the strongest "content, not narrative" read),
  which would dovetail with Exp 2 (quotes carry the signal) and Exp 3a/b.

## Gates & sequencing
1. **Gate 1 (this plan):** confirm arms to run (rec: 5a + 5c; 5b optional) and the 5c "permute quote
   contents in the last block" definition.
2. Build `edits.py` shuffle fns + `exp5{a,c}_edit.py` + records (CPU, **no API, no GPU**); show worked
   shuffle examples in the dashboard with FVE blank.
3. **Gate 2 (pre-GPU):** confirm, then AR re-score on GPU 0/1 → fill FVE → report.
