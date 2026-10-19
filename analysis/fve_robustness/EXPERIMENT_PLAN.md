# FVE robustness — manipulation experiments plan

Models: **Qwen2.5-7B-L20, Gemma-3-12B-L32, Gemma-3-27B-L41** (not Llama-70B for the
manipulation series). Held-out set: the same n=2000 Ultra-FineWeb docs as the baseline.
GPUs 2,3 on Athena only.

## 0. The shared mechanic (why every arm is cheap and comparable)

An NLA round-trip is `activation --AV--> explanation --AR--> reconstruction`. A manipulation
edits the **explanation text only**; everything else is held fixed:

- **Fixed across all arms:** the sampled read position, the gold activation
  `gold[idx] = hidden_states[K+1][pos]`, the doc, and the AR model. Already computed and
  frozen in `out_<key>/{gold.npy, manifest.json, av_reads.json, cos.json, fve_<key>.json}`.
- **Changes per arm:** the text handed to the AR. We take the *existing* baseline
  explanation (temp 0.4, seed 0 — not regenerated), apply an edit, and re-run only the
  AR scoring phase.

Because gold is fixed, the FVE denominator `D = Σ_i ||ĝ_i − ḡ||²` is **identical across
arms**, so `FVE_arm = 1 − Σ_i 2(1−cos_i^arm) / D` and `ΔFVE = FVE_arm − FVE_base` is a
clean, monotone function of the change in per-point cosine. This makes the arms directly
comparable and lets us attribute FVE movement to individual data points.

Per data point we report:
- `cos_orig`, `cos_edit`, `Δcos` (the granular reconstruction signal),
- `D_i = ||ĝ_i − ḡ||²` (that point's predict-mean variance),
- `FVE_i = 1 − 2(1−cos_i)/D_i` — a per-row, variance-normalized analog of FVE (can be
  negative; this is the "corresponding FVE" for a single datapoint). The **headline FVE is
  the set-level aggregate**; `FVE_i` is the per-row decomposition of it.

Reused code: `phase_score` / `phase_fve` from `fve_ufw.py`. New generic module
`rescore.py <key> <arm>` reads `edited_reads_<arm>.json`, runs the AR, writes
`cos_<arm>.json`, `fve_<arm>_<key>.json`, and a per-datapoint join
`records_<arm>_<key>.json` = `[{idx, doc_id, position, token, token_ctx, input_window,
input_full, expl_orig, expl_edit, edit_ops, cos_orig, cos_edit, D_i, fve_i_orig,
fve_i_edit}]`.

## 1. The critical caveat and the control philosophy

The AR was trained on the AV's *natural* outputs. Every edit pushes it **off-distribution**,
so a raw FVE drop conflates two things: (a) load-bearing content was removed, vs (b) the
text was made malformed/OOD and the AR chokes on form. Each experiment therefore ships a
**length/format-matched control** that perturbs a comparable amount of *non-target* text.
The contrast (target-edit vs matched-control) is what carries the interpretation, not the
raw drop. Every arm also runs a **NULL check**: re-score the *unedited* explanation through
the new harness and confirm it reproduces the baseline cos exactly (instrument gate).

Statistics: paired design (same gold/doc across arms) → Wilcoxon signed-rank on `Δcos` +
bootstrap 95% CI on `ΔFVE` (resample docs), reported per model.

## 2. The five experiments

### Exp 1 — remove ALL quoted content  [deterministic, no API]
- **Edit:** strip every quoted span (curly/straight double, single, guillemet, backtick),
  including the marks, then normalize orphaned whitespace/punctuation. Unbalanced quotes
  (AV truncated at 200 tokens) → strip from the last unmatched opener to end; log count.
- **Control (length-matched prose deletion):** delete the same number of characters from
  *non-quoted* prose (random spans, seeded), preserving quotes. If FVE drops as much as
  quote-removal, the effect is generic text loss, not quote-specific.
- **Tests:** does the genre line + connective meta-description alone reconstruct, or is the
  quoted specific content load-bearing?

### Exp 2 — remove only INPUT-DERIVABLE quotes; keep divergent  [verbatim = deterministic; synonym = API]
- **Reference context:** left-context tokens `[0:pos+1]` (what the activation causally
  summarizes); the full doc is a looser secondary check.
- **Per quote, classify:**
  - *verbatim* — normalized quote is a substring of the reference (deterministic), or
    high ROUGE-L against a sliding window (fuzzy-verbatim).
  - *synonym-restatement* — for quotes not caught above, one Claude (Haiku) call:
    "does this quote restate info present in the passage via paraphrase/synonyms, adding no
    new information (RESTATE) or introduce content not in it (DIVERGE)?"
  - Remove `verbatim ∪ synonym-restatement`; **keep DIVERGE**.
- **Control / companion arm:** the complement — remove only DIVERGE quotes, keep derivable.
  If removing divergent quotes hurts FVE *more* than removing derivable ones, the AR leans
  on genuine/confabulated content rather than input inversion (and vice-versa).
- **Tests:** is reconstruction carried by the input-derivable quotes or the divergent ones?

### Exp 3 — rephrase the trailing quoted block into natural language  [API]
- **Locate the block:** from the `Final token` marker to the end (falls back to the trailing
  quote-dense paragraph). This is the Part-2/Part-3 region identified above.
- **Edit:** one Claude (Sonnet) call — "rephrase into natural-language prose, no quotation
  marks and no verbatim phrases; preserve meaning." Content preserved, quotation *form*
  removed.
- **Control (form-preserving null + prose paraphrase):** (i) rephrase but *keep* quotes
  (checks the rewriter isn't destroying content); (ii) paraphrase a non-quoted prose span
  (the genre line) of similar length — if FVE drops similarly, the effect is "any rewrite is
  OOD," not "quotation form removed."
- **Tests (vs Exp 1):** Exp 1 deletes quoted content; Exp 3 keeps the content but drops the
  verbatim/quotation form. The pair separates "content in quotes" from "surface form."

### Exp 4 — shuffle claims aggressively at random  [deterministic]
- **Segment:** into claims — sentence split with quoted spans kept intact as atomic units
  (optionally the two-tier LLM segmenter as a variant). Aggressive = full seeded random
  permutation.
- **Controls:** (i) identity re-join (segment then rejoin in order → must reproduce
  baseline; validates segmentation); (ii) word-level shuffle within sentences (destroys
  syntax, keeps the claim set) vs sentence-level shuffle (keeps syntax, destroys claim
  order) — separates syntax-sensitivity from claim-order-sensitivity.
- **Tests:** is the AR order-sensitive/compositional, or a bag-of-claims reader?

### Exp 5 — replace AV quotes with matched REAL input quotes  [matching: lexical default / embedding variant]
- **Edit:** for each AV quoted span, find the best-matching real span in the left-context
  (window maximizing token-overlap / ROUGE-L; embedding-similarity variant optional), and
  substitute the real span (same punctuation). 
- **Controls:** (i) substitute with a *random* real span from the same doc (if FVE rises even
  for random real spans, real tokens help broadly, not via targeted inversion); (ii)
  substitute with a real span from a *different* doc (negative control, should hurt).
- **Tests:** if reconstruction error drops (FVE rises) when we feed the AR the *true* input
  tokens, that is direct evidence the AV/AR pair prefers **input inversion** over conveying
  activation-genuine content.

## 3. The dashboard / per-experiment tabs

Belief-geometry style (EB Garamond, parchment `#faf6f1`, panels `#fffdf9/#e8e0d4`, green
`#4a7c59`, KaTeX). Multi-file, shared top-nav (works over `file://`, no server):

- **`index.html` — Overview:** the baseline table + the master matrix (rows = 3 models;
  columns = baseline FVE and each arm's FVE, ΔFVE, 95% CI, Wilcoxon p). Links to each arm.
- **`exp<k>_<name>.html` — one per experiment:** in-page model toggle (Qwen / Gemma-12B /
  Gemma-27B). Header = definition, method, arm results (FVE_base, FVE_edit, ΔFVE, mean Δcos,
  CI, p, n). Then a **paginated, searchable, sortable per-datapoint table** (every point),
  each row expandable to:
  - `idx`, `doc_id`, `position`, and the **exact read token** highlighted with its neighbors;
  - **input text** — windowed ±~800 chars around the read token (token highlighted),
    expandable to the full document;
  - **original explanation** with quotes highlighted, `cos_orig`, `FVE_i^orig`;
  - **edited explanation** diff-highlighted (removed = strikethrough, added = underline),
    `cos_edit`, `FVE_i^edit`;
  - `Δcos`, `Δ FVE_i`, `D_i`. Sortable by `Δcos` to surface the biggest movers.

Builder `build_experiments_dashboard.py` regenerates everything from the
`records_*/fve_*` artifacts (fully reproducible).

## 4. Compute, API, cost

- **Re-scoring:** one single-GPU AR job per (arm, model), ~5–8 min each; run with `gpurun
  GPU=2/3` once Llama-70B frees the GPUs (~30 min). Edit steps (string ops + API) run on
  CPU/API independently, anytime.
- **API (anthropic SDK, key in `.env`):** Exp 2 ≈ few-thousand Haiku classifications (~$2);
  Exp 3 ≈ up to 6000 Sonnet rephrases (~$20–60); Exp 5 lexical = free (embedding variant
  cheap). Concurrency + on-disk cache. Full n=2000×3 is affordable; a subsample is offered
  for fast first looks.

## 5. Proposed step-by-step order (each gated on your go-ahead)

- **Step 0 — harness + gate:** `rescore.py` + NULL check (re-score original == baseline) +
  the tab template. Shared infra, built once.
- **Step 1 — Exp 1** (all quotes; deterministic): first real arm, validates pipeline + UX.
- **Step 2 — Exp 4** (shuffle; deterministic).
- **Step 3 — Exp 3** (rephrase block → NL; first API arm).
- **Step 4 — Exp 2** (derivable vs divergent; verbatim + API classify).
- **Step 5 — Exp 5** (real-quote substitution; matching).

Deterministic arms first (fastest to a trustworthy result and they build the harness), then
the API arms. Each experiment includes its core arm + its control(s); controls can be
deferred to a second pass per experiment if you want each step even more minimal.

## Decisions for you
1. **Order:** the recommended build order (1, 4, 3, 2, 5) or your listed order (1..5)?
2. **Controls:** run each experiment's control(s) alongside its core arm, or core arms first
   and controls as a second pass?
3. **n for the API arms (2, 3, 5):** full 2000/model, or a pre-registered subsample (e.g.
   750) for faster/cheaper first looks? (Deterministic arms 1, 4 stay at 2000.)
4. **Input display in tabs:** windowed excerpt by default with full-doc-on-expand
   (lighter files) — confirm, or always show full text (heavier)?
