# Exp 6 — "realquote": replace the AV's quoted spans with the TRUE input spans

Status: **COMPLETE (2026-08-27).** A1/A2/B/C_otherdoc/F/P/C_cont were scored on 2,000 rows × 3
models; all null gates passed with `max|Δcos|=0`. Design produced by a
4-angle design panel (crux-minimalist / two-axis / control-maximalist / skeptic-refuter),
judged and synthesized; the panel read 9 real datapoints and repaired two flaws in the
first framing (below). This is the dashboard's `exp6_realquote` arm = the original
`EXPERIMENT_PLAN.md` "Exp 5".

**Execution record.** P used GPT-5.6 Luna via `OPENAI_API_KEY_2`, with a separate semantic audit for
every cached paraphrase. Exact word matching was abandoned before the accepted run because it forced
filler or invented completions on truncated true-context fragments; the final registered deviation
targets the source length within a logged 40–160% envelope and fails closed to the true slice after
three rejected audits. Mean absolute Δwords was 0.58–1.04; 120/12,609 paraphrases failed closed after
semantic audit and 7 after mechanical validation. Tracked new API spend was $3.14. Including prior
Exp-7 spend and a conservative $10 reserve for interrupted-process counter ambiguity, the upper-bound
ledger total is $22.89 (<$100).

Key final contrasts (document-bootstrap 95% CI):

| Contrast | Qwen2.5-7B | Gemma-3-12B | Gemma-3-27B |
|---|---:|---:|---:|
| P ΔFVE, high-div changed | −0.0195 [−0.0225, −0.0165] | +0.0018 [−0.0026, +0.0066] | −0.0011 [−0.0040, +0.0017] |
| P − A2, high-div changed | +0.1841 [+0.1735, +0.1951] | +0.1193 [+0.1101, +0.1286] | +0.1631 [+0.1524, +0.1737] |
| C_cont ΔFVE, AV-wrong | −0.0445 [−0.0503, −0.0390] | −0.0994 [−0.1110, −0.0868] | −0.1825 [−0.1984, −0.1676] |

Raw A2 true-context injection collapses while semantically paraphrased true content is flat for both
Gemmas and only mildly harmful for Qwen. This rejects the clean exact-input-inversion pattern and
attributes most A2 damage to surface/form mismatch. Correct future text also fails to improve a
causally pre-future activation, including where all original AV continuation candidates were wrong.

## 1. Why this is the crux of the whole sweep

The frozen activation `h = hidden_states[K+1][pos]` causally summarizes only the
left-context `[0:pos+1]`. The round-trip decodes `h → AV explanation → AR reconstruction`.
The AV's quoted spans are its *guesses* at that summarized input (and its predictions of
what comes next). If we surgically replace the AV's **guessed left-context** with the
**true input tokens** and the AR reconstructs `h` **better** (FVE rises, net of controls),
that is direct evidence the AV/AR channel prefers the real input — **input inversion**, the
Li-et-al. skeptic reading. If FVE is **flat/drops** and controls rule out mere
off-distribution surface damage, the AV's confabulated-but-gist-right quotes already carry
what the AR needs — consistent with conveying **activation-genuine gist**, not verbatim
input.

Gold is frozen ⇒ the FVE denominator `D` is identical across arms ⇒ `ΔFVE` is a clean,
per-datapoint-attributable function of `Δcos`. All inference lives in **target−control
contrasts**, never raw `ΔFVE`, and only on the **stratum where the AV actually confabulated**
(elsewhere there is nothing to substitute and the arm has no power).

## 2. Two flaws the panel caught by reading the data (and the repairs)

1. **Boundary-token leak.** The true left-context that `h` summarizes *ends at the read
   token*, so `doc[:char_end]` literally ends in the gold token's surface string
   (qwen idx0 → "...vacuumed **by**"; gemma idx7 → "...includes **apples**"). Injecting it
   hands the AR the gold token itself → a rise could be a trivial final-token echo, not
   inversion. **Repair:** a twin pair — **A1** (slice ends at the read token) vs **A2**
   (slice ends one token *before*). The inversion claim requires **A2** (not A1) to rise.
2. **"Recon quotes end at char_end" is false.** LCR quotes often end elsewhere (qwen idx7
   Block-1 ends "...plant families"; gemma12b idx7 ends "roses"). **Repair:** anchor each
   LCR quote's true slice to *its own* true right-edge (lexical anchor of its last ~6
   content words), not blindly to `char_end`; drop rows with `no_faithful_anchor`.

Also confirmed: read positions/tokens differ per model for the same `idx` ⇒ **never
aggregate "same idx" across models**; the `echo == token` anchor fails ~4/9 ⇒ tag the echo
by structural position, not string equality; the AV is often already gist-right ⇒
**divergence stratification is mandatory** for a "flat" result to be interpretable.

## 3. Arm set

Manipuland = **left-context-reconstruction (LCR) quotes only.** The read-token **echo**
quote and the **predicted-continuation (PC)** quotes are **frozen in every headline arm**,
so every headline `ΔFVE` is attributable to left-context inversion alone. Continuation is a
separate, non-headline arm (`C_cont`) because the activation does not causally contain the
future, the AV emits 6–9 alternative slots vs the doc's one continuation, and AV
continuations are often already correct (qwen idx0 "a robotic vacuum" ≈ real "Roomba").

Length/format matching (all substitution arms, uniform): replacement matched to the AV
quote's **word count** (model-agnostic; char count logged as covariate), snapped to word
boundaries minimizing `|Δwords|`; `Δsubtok_len` logged; identical matching in every arm so
length cancels in contrasts.

| # | Arm | Role | Replace → with | True-span source |
|---|-----|------|----------------|------------------|
| 0 | **A0_null** | null-gate | nothing (re-score `expl_orig`); must reproduce baseline `max\|Δcos\|<1e-3`; emits parsed quote inventory | none |
| 1 | **A1_realctx_endT** | core | LCR quote contents → true left-context tail ending at the quote's own right-edge (**includes** the read token where the quote ends there) | `doc[:right_edge]` |
| 2 | **A2_realctx_preT** | **core (load-bearing)** | as A1 but slice ends one token **before** the read token (gold token absent) | `doc[:char_start]` |
| 3 | **B_randctx** | control | as A1 spans/length; seeded **random same-doc** span from outside `[char_start−W:char_end+W]`, never from the continuation | random same-doc |
| 4 | **C_otherdoc** | control / load-bearing gate | as A1; true left-context of a **different** held-out doc (fixed derangement) — real, grammatical, wrong content | different-doc left-context |
| 5 | **F_shuffle** | control (form-OOD floor) | as A1; **seeded word-shuffle of the AV's own LCR quote** (same words, destroyed order) | none |
| 6 | **P_paraphrase** | core (content-vs-form; small API) | LCR quote → one-sentence Haiku paraphrase of A1's true slice, fluent register, quotes kept | left-context (paraphrased) |
| 7 | **C_cont** | secondary (non-headline) | LCR/echo frozen; **PC** slots → ordered word-boundary tiles of true `doc[char_end:]`; reported only where AV continuation was wrong | continuation |

Headline: **A1, A2, P**. Controls: **B, C_otherdoc, F**. Gate: **A0**. Secondary: **C_cont**.

## 4. Span identification + matching (deterministic, tokenizer-free)

Parse: split on blank lines (assert 3 blocks; quarantine failures symmetrically). Extract
all double-quoted spans (straight/curly/guillemet/backtick; existing helpers; unbalanced =
AV 200-tok truncation rule). **Classify by structural position:** Block-0 = genre (dropped);
the Block-1/2 structure quote = **LCR**; Block-3 first quote at `Final token` = **ECHO**
(frozen); Block-3 parenthetical clause after `ends/opens/...` = **LCR**; Block-3 quotes at a
continuation trigger (`expecting/like/likely/such as/...`) = **PC**; else **UNKNOWN**
(frozen everywhere). **Directional dual-gate:** an LCR quote must have higher ROUGE-L vs
`doc[:char_end]` than vs `doc[char_end:]` (PC the reverse); disagreement → demote to UNKNOWN
(prevents a mislabeled continuation quote from leaking future tokens into the left-context
arm).

Map each LCR quote to its true analog: locate its **own right-edge** by sliding a window
over `doc[:char_end]` maximizing ROUGE-L of its final ~6 content words (matching only
*locates* the edge; the injected content is then the contiguous true tail — no
matcher-quality-into-content leak). No faithful edge (ROUGE-L below threshold) →
`no_faithful_anchor`, drop from A1/A2/P.

**Divergence stratifier (pre-registered):** `div = 1 − ROUGE-L(AV quote, true analog)`.
Primary test on the **high-div** stratum (AV confabulated → substitution has power);
low-div reported separately as a null-power sanity stratum, never as evidence.

## 5. Interpretation

**Evidence FOR input inversion** = the conjunction, on ≥2/3 models, high-div stratum:
`A2 rises` ∧ `(A2−B) CI>0` ∧ `(A2−F) CI>0` ∧ `C_otherdoc drops`.

| A1 | A2 | B | F | C_otherdoc | P | Conclusion |
|----|----|---|---|-----------|---|------------|
| rise | **rise** | flat | flat | drop | rise | **Clean input inversion** (Li-et-al.-aligned) |
| rise | **flat** | flat | flat | drop | – | **Final-token echo artifact**, not inversion (why A2 exists) |
| rise | rise | **rise≈A2** | flat | drop | – | **OOD-topicality** (any same-doc text helps), not targeted inversion |
| rise | rise | flat | **rise≈A2** | – | – | **Form/edit-path artifact** — audit harness, don't interpret |
| flat | flat | flat | flat | drop | **rise** | **Content present but surface-OOD** — AR wants the content, not raw input form |
| flat | flat | flat | flat | drop | **flat** | **Genuine-content / AV already sufficient** (P + C_otherdoc license this) |
| drop | drop | drop | drop | drop | drop | **Pure form-OOD dominates** — no content claim |
| – | – | – | – | **flat** | – | **Spans inert / gate fails** — uninterpretable |

Controls kill each alternative: length (all arms word-matched → cancels); lexical-overlap
(`A2−F`, same words shuffled); topicality (`A2−B`, same doc wrong position); generic real
prose (`C_otherdoc` must hurt); surface-vs-content (`P`); boundary echo (`A1−A2`).

## 6. Statistics + dashboard

Per model (never cross-model on same idx): per arm `FVE_base/arm`, `ΔFVE` with
**doc-clustered** bootstrap 95% CI (resample docs, not points), Wilcoxon on `Δcos`,
`n_paired`, `n_dropped` by cause. Headline **contrasts** on the *same* resampled doc indices:
`A2−B`, `A2−F`, `A2−C_otherdoc`, `A1−A2`, `P−A2`; Holm within model. Stratified by div.
Winsorize/flag low-`D_i` (sink-state) rows; set-level FVE stays headline.

Dashboard `exp6_realquote.html` (per-model tab; A2/B/C_otherdoc three-way compare + 9-point
spotlight): add `quote_tag`, `echo_matches_token`, `div`, `stratum`, `right_edge_at_char_end`,
`no_faithful_anchor`, `Δwords`, `Δsubtok_len`, `low_D_flag`, `donor_doc_id`, and per-arm
`cos_edit, Δcos, FVE_i_edit, edit_ops`. Sortable by `Δcos` and `div`.

## 7. Harness changes + feasibility

- `records_common.build_records`: hoist offset computation before `edit_fn`; support
  `edit_fn(expl, doc, ctx)` with `ctx={char_start,char_end,position,doc_id,token,other_doc_text}`;
  keep 1-/2-arg edit_fns working (~15 lines).
- New `exp6_edits.py` (parse + tag + right-edge locate + 8 edit_fns on existing `edits.py`
  helpers); thin per-arm drivers; `exp6_contrasts.py` bootstrap helper.
- **AR-max-length / suffix-anchor guard** after each edit; injection-marker/CJK smoke test.
- `rescore.py` unchanged. Dashboard tab added to `build_experiments_dashboard.py`.

Cost: all edits are deterministic CPU on frozen `av_reads.json`+`ufw_docs.json`, **$0 API
except P** (~few-thousand cached Haiku calls, ~$2–4). GPU: one AR re-score per (arm,model),
7 arms × 3 models ≈ **~2–2.5 GPU-hrs** on GPUs 0/1; +C_cont ≈ 25 min. **llama70b excluded
from the headline** (matches the spec; parser unvalidated on llama; outlier FVE geometry).

## 8. Gated run order

- **6.0 PARSER-AUDIT (CPU, $0, no GPU):** run the parser over 2000×3, emit tag-coverage /
  `n_blocks` / echo / unbalanced / `no_faithful_anchor` / `Δwords` histograms, hand-inspect
  ~50 flagged rows. **CHECK IN before any GPU/API.**
- **6.1 A0 null-gate** (+ length-inflated NULL): must pass `max|Δcos|<1e-3`.
- **6.2 deterministic core+controls:** A1, A2, B, C_otherdoc, F → headline table.
- **6.3 P (paraphrase, API):** only if 6.2 lands in a flat/form-OOD cell needing
  content-vs-form disambiguation.
- **6.4 C_cont (secondary):** optional, AV-continuation-wrong subset only.

## 9. Open decisions

1. Injection-length basis: AV-quote word-count (default) vs fixed-N-token window.
2. P arm: conditional Step 6.3 (default) vs first-pass alongside deterministic arms.
3. C_cont: labeled secondary (default off headline) vs defer entirely.
4. `no_faithful_anchor` rows: drop-and-log (default) vs blind-`char_end` flagged variant.
5. Divergence-stratum threshold: pre-register the high/low cut (median or fixed 0.5) vs
   report `div` as a continuous covariate.
