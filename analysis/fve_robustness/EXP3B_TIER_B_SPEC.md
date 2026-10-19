# Exp 3b — Tier B: paraphrase/synonym-aware input-derivability (API judge)

Sibling of **Exp 3a** (Tier A, deterministic verbatim ≥2-content-word run). Same experiment
question and same edit mechanic; the only change is *how a quote is judged input-derivable*.

## Why Tier B exists
Tier A only removes quotes that reproduce a **verbatim** multi-word span of the input. But the AV
almost never quotes verbatim (Exp 3a: ~1 verbatim-derivable quote per 7–9, ~half of explanations
untouched at min=2). Its restatements are **paraphrases / synonym swaps / correct continuation
predictions**, which a string matcher cannot see. If reconstruction leans on the AV echoing the
input, that echo lives in the paraphrased content — exactly what Tier A misses. Tier B is the
version that actually tests the input-inversion hypothesis.

## Definition (what the judge decides, per quote)
A quoted span is **input-derivable (Tier B)** iff its propositional content is **stated in, or
directly entailed by, the input document** — as verbatim text, a synonym/paraphrase restatement, or
a *correct* continuation prediction of the read position. It is **divergent** iff it introduces
content **not present** in the input: a genre/format guess, a wrong continuation, or a confabulation.
The bar is "restates something the input actually says," **not** "is topically related" or "is a
generic phrase for this domain."

Key invariant: **Tier A ⊆ Tier B.** Every verbatim restatement is trivially a semantic one, so any
quote Tier A flags must also be flagged by Tier B. Violations measure judge miscalibration.

## Method
- **Unit:** one API call per explanation (batched over its quoted spans), not per quote — cheaper and
  gives the judge cross-quote context.
- **Inputs to the judge:** (a) the input document (windowed to the read region + enough right-context
  to score continuations; full doc if it fits), (b) the read token + position, (c) the numbered list
  of the explanation's quoted spans (verbatim, with their marks).
- **Output (structured / JSON, one row per quote):** `{id, label: "derivable"|"divergent",
  basis: "verbatim"|"paraphrase"|"continuation"|"none", reason: <≤12 words>, confidence: 0–1}`.
  Forced via a StructuredOutput schema so nothing needs parsing and the model retries on mismatch.
- **Model:** Claude Haiku, temperature 0, fixed. (Judge is not bit-reproducible across model
  versions — record the model id in the sidecar.)
- **Edit:** identical to 3a — strike the Tier-B-derivable spans, keep divergent quotes + all prose,
  freeze gold/read/AR, re-score AR → ΔFVE. Records carry `removed`, and `edit_meta.quotes` gains the
  judge's `basis`/`reason` per quote (shown on the dashboard tab).

## Prompt sketch (judge)
> You are auditing an activation-verbalizer's explanation. Below is an INPUT DOCUMENT and a list of
> QUOTED SPANS taken from a model's explanation of one token in it (the read token is marked). For
> each quote decide whether its content is **actually stated or directly entailed by the INPUT**
> (label `derivable`) or introduces content **not in the input** (label `divergent`). Count verbatim
> text, synonym/paraphrase restatements, and *correct* next-token continuations as derivable. Do NOT
> count generic domain phrases, plausible-but-absent detail, or wrong guesses. Return JSON …

## Cost & gating (Gate 2 before any API spend)
~2000 explanations × 3 models = ~6000 calls, ~2k tokens each ≈ 13M input / 2–3M output tokens.
On Haiku this is a modest one-time spend (order tens of dollars); exact estimate confirmed at Gate 2.

## Validation
1. **A ⊆ B**: fraction of Tier-A-derivable quotes the judge marks divergent — target < ~2%; higher ⇒
   tighten the prompt.
2. **Human calibration**: hand-label ~50 quotes, compare to judge (report agreement).
3. **Confidence floor**: quotes with confidence < 0.6 flagged for review, not silently removed.
4. **NULL/repro**: temperature 0, same tuple ⇒ same verdict within a model version; log the model id.

## Hand-worked examples (no API) — how Tier B would classify vs Tier A

### From the real datapoint (qwen7b, doc `05f758ebaab32aed`, read token `"of"` @1671)
| Quote | Tier A (verbatim ≥2 cw) | Tier B (semantic) | Note |
|---|---|---|---|
| `"4. Hydrogeologic Evaluation - … the injection zone is hydrocarbon-bearing and the areal extent of the injection formation …"` | **derivable** (3-cw verbatim run) | **derivable** (restates a regulatory section in the doc) | agree |
| `"the area of investigation and the area of pressure containment within the area of"` | divergent (only 1-cw run) | **derivable** (paraphrase of the doc's area-of-review / containment definitions) | **B catches, A misses** |
| `"potential,"` | divergent (1 cw) | divergent (lone word, no specific input proposition) | agree — the bug you caught stays fixed |
| `" injection"` / `"containment"` / `"of"` | divergent | divergent (generic domain words) | agree |
| continuation guesses `"containment"` / `"injection"` / `"potential"` | divergent (single words) | divergent **iff** they are not the actual next token (doc continues `… area of influence`) | B could flag a *correct* guess; here all three are wrong ⇒ divergent |

### Clean synthetic illustration
Input: *"The reactor core reached 900 degrees before the coolant pumps failed."*
| Quote | Tier A | Tier B | Kind |
|---|---|---|---|
| `"the coolant pumps failed"` | derivable | derivable | verbatim |
| `"the cooling system stopped working"` | **divergent** | **derivable** | synonym/paraphrase → **B only** |
| read token `"pumps"`, predicted `"failed"` | divergent (1 word) | **derivable** | correct continuation → **B only** |
| `"the technicians evacuated to the shelter"` | divergent | divergent | confabulation → both keep |

**Expected effect vs 3a:** Tier B's derivable set strictly contains Tier A's and adds the paraphrase
+ correct-continuation quotes, so 3b removes strictly more per explanation and should show a larger
(more negative) ΔFVE than 3a — the gap between them *is* the paraphrased/echoed-input contribution
that verbatim matching cannot see.
