# Part A — NLA Failure Modes: A Unified Taxonomy and the Search for a Governing Mechanism

**Status:** working research memo (Part A of the failure-mode program). Conceptual + empirically grounded in the released NLAs, *inference-only*, *data-first*.
**Sources:** the NLA paper (`papers/nla paper.pdf`), "Building Better Activation Oracles" (`papers/building better AOs.pdf`), and direct mining of the four per-token decode dumps in `examples/` (Qwen-2.5-7B L20, Gemma-3-12B L32, Gemma-3-27B L41, Llama-3.3-70B L53).
**Caveat up front:** the `examples/` dumps are **one prompt per model**. Everything drawn from them below is *qualitative, hypothesis-generating* evidence — illustrative, not statistical. Quantifying these patterns is exactly what the Part B harness (and the Neuronpedia probes) are for. Where a number appears, it is the value reported in that single dump.

---

## 0. What this memo is trying to do

The NLA paper and the AO paper each ship a *list* of failure modes — confabulation, vagueness, text-inversion, activation-insensitivity, layer/position blindness, miscalibrated awareness, steganography, probe-ceiling, writing-quality decay. The premise of this program is that **these are not independent defects but one mechanism wearing many masks.** Part A does three things:

1. **Catalog** every failure into a single comparable schema (§2).
2. **Ground** the catalog in what the released NLAs *actually* do wrong, read off the real decodes (§1) — this is the "novel findings" layer that goes beyond restating the papers.
3. **Adjudicate, data-first**, between candidate unifying mechanisms (§3–4), and name the **novel paradigmatic pattern** worth chasing (§4).

The headline claim I'll build toward: **NLA failures are the verbalizer faithfully reporting a posterior that is prior-dominated wherever the activation is uninformative about a claim.** Faithfulness is then not a property of the explanation but a *scalar field over activation space* — "verbalization evidence" — and every named failure is a low-evidence region of that field. The research move is to make that field **measurable without ground truth** and then **expose it** (diagnosis) and **make the objective sensitive to it** (fix).

---

## 1. What the released NLAs actually do wrong (read off the decodes)

Eight observations, each tied to concrete evidence in `examples/`. These are the empirical spine of the taxonomy.

### O1 — Reconstruction fidelity and semantic faithfulness are decoupled (universal, and the single most important finding)
High `cos` routinely coexists with **false specifics**:
- Qwen, position 57: `cos=0.942` while the explanation invents "**ChatGPT**"; positions 7/9/37/42 confabulate "**Claude**", "**Anthropic**" on a *Qwen* model.
- Gemma-12B, position ~25: `cos=0.996` while naming "**LaMDA … Google AI**" — a *different lab's* model.
- Gemma-27B: `cos≈0.99` while a Paris answer slips to "**Rome**", and `<bos>` (`cos=0.998`) emits a "diabetes / August 26, 2021" confabulation.

Meanwhile the **theme/genre is recovered accurately** at those same positions ("AI self-introduction", "factual answer about Paris"). So the reconstruction metric (MSE/`cos`) certifies the *coarse* content and is near-blind to the *specific* content. **Fidelity ≠ faithfulness, and the gap is structured: it lives in the specifics.**

### O2 — A theme → entity → detail accuracy gradient (universal)
This is the NLA paper's confabulation finding (theme 64% > entity 28% > detail 24%), and it is visible directly: across all four dumps the *theme* is right, while invented **entities** (Anthropic, LaMDA, DAMO Academy, DANTE, Quora) and **details** (Fibonacci sequence, a song titled "What are you hiding?", specific dates) are wrong. The gradient is the fingerprint of the mechanism, not a separate bug.

### O3 — Confabulated content is drawn from the verbalizer's own knowledge prior (universal)
The invented specifics are not random — they are exactly what the *base model's parametric prior* would emit to fill the slot: AI-lab/identity entities ("Anthropic", "ChatGPT", "LaMDA"), canonical landmarks, plausible dates. Qwen even leaks **Chinese script** into several explanations (positions 1, 22, 23, 32) — its bilingual prior surfacing — while Gemma and Llama (English-dominant priors) show no such leakage. **What fills the gap is the model's prior, conditioned on the inferred theme.**

### O4 — Failures concentrate sharply on out-of-distribution activations (universal; sharpest signal)
Special/structural tokens (BOS, turn-markers, newlines) are *under-sampled in datagen* (`min_position=50`) and they fail hardest:
- Llama `<|begin_of_text|>`: `cos=-0.059`, `mse=2.119` — **orthogonal**, with degenerate `</</</` tag-soup output.
- Qwen `<|im_start|>`: `cos=0.019`; another at `cos=0.144`, output garbled.
- Gemma special tokens: Java-tutorial / diabetes confabulations; stray `<explanation>` tags, truncation.

Llama is the cleanest demonstration of a **faithfulness cliff**: positions 0–30 (system region, OOD) are catastrophic; from the in-distribution user/reply region onward it reconstructs faithfully. **Distance-from-training-distribution gates the failure**, and the boundary is sharp.

### O5 — Sampling variance isolates a leverage spectrum (the mechanism, caught on camera)
Resampling *one* activation at temp=1 (each dump's §4):
- The **theme is invariant** across samples; the **specifics drift freely** ("ChatAI" vs "confidential" vs "Hiya!"; "learned patterns" vs "emergent behavior").
- The **reconstruction MSE is ~constant despite the drift** (Qwen 0.132–0.157; Gemma 0.006–0.007). The AR *cannot tell the varying specifics apart* — they have ≈0 reconstruction leverage.

This is the coarse-basin mechanism made visible: low-leverage specifics are under-constrained by the objective, so they wander under sampling while the high-leverage theme stays locked. It is also the empirical basis for **consensus/recurrence as a reliability signal** (stable-across-samples ⇒ high-leverage ⇒ reliable).

### O6 — `injection_scale` is load-bearing and interacts with an OOD raw-norm regime
The per-model scales span ~2500× (Gemma-12B **80000**, Gemma-27B **60000**, Qwen **150**, Llama **30**) — Gemma's are huge because Gemma multiplies embeddings by √d. Raw-norm **outliers** (Gemma `<bos>` `||v||≈7.1e5`; Qwen `\n` `||v||≈1.4e4`) mark OOD positions. Subtle but important: rescaling to a fixed norm fixes magnitude but **not the direction-regime** — Gemma `<bos>` reconstructs at `cos=1.000` yet emits incoherent text. Even at the geometric level, fidelity and faithfulness come apart.

### O7 — Apparent faithfulness conflates two things (the evaluation trap)
Llama looks *most* faithful in its reply region — but partly because the reply is generic safety boilerplate ("I'm not hiding anything, I'm a large language model…"), which is **trivially recoverable from the surrounding tokens**. This is the AO paper's **text-inversion** confound, observed live: when content is high-prior/predictable, you cannot tell whether the NLA *read the activation* or merely *inferred the obvious*. Any honest failure characterization must subtract off the token-predictability term.

### O8 — Writing-quality degradation (universal)
Boilerplate repetition (Llama repeats one clause ~70 positions), theme-hammering ("hiding… hiding… hiding"), and the parenthetical-with-embedded-quotes style that makes explanations hard to parse. Consistent with the paper's "writing quality declines over training."

**Cross-model caveat (do not over-read rankings):** the four dumps use *different prompts* — Gemma-27B answered "capital of France" (factual, high-prior, easy to verbalize) while Qwen/Gemma-12B/Llama answered "what are you hiding?" Differences in apparent FVE/faithfulness across models are confounded by prompt difficulty (see O7). The dumps are for *mechanism*, not leaderboard.

---

## 2. The unified failure taxonomy

Every known failure, normalized so the candidate mechanisms in §3 can be tested against one schema. The right-hand columns connect each to the §1 evidence and to a *ground-truth-free* signal (the currency of the whole program).

| # | Failure mode | One-line definition | Evidence (§1) | Ground-truth-free signal it should correlate with |
|---|---|---|---|---|
| F1 | **Confabulation** | False claims about the context | O1,O2,O3 | low AR-leverage; low recurrence |
| F2 | **Specificity gradient** | Details far less accurate than themes | O2 | leverage falls from theme→entity→detail |
| F3 | **Vagueness** | Generic, unfalsifiable, dodges the question | O8 (and paper) | low activation-sensitivity (would say it anywhere) |
| F4 | **Text-inversion** | Answer recoverable from inferred tokens, not the activation | O7 | high token-predictability; low activation-conditional info |
| F5 | **Activation-insensitivity** | Same tokens / different context → no change when it should | (probe needed) | Δexplanation ≈ 0 under fixed-token, varied-context |
| F6 | **Excessive expressivity** | Verbalizer infers beyond the activation | O3 | claim prob. under base-model prior ≫ its leverage |
| F7 | **OOD / distribution cliff** | Catastrophic failure off the training manifold | O4,O6 | distance-to-training-distribution; raw-norm outlier |
| F8 | **Steganography** | AV→AR channel carries non-overt info | (paper: small) | FVE drop under meaning-preserving transforms |
| F9 | **Layer/position blindness** | One (layer, token) misses distributed info | O4 (position); paper (layer) | single- vs multi-position/layer recovery gap |
| F10 | **Calibration failure** | Measured quantity directionally right, not calibrated | (paper: eval-awareness) | measured value vs ground-truth credence |
| F11 | **Unverbalizable content** | Activation info not expressible in language | (paper) | probe-decodable yet NLA-missed |
| F12 | **Probe ceiling** | NLA fails to beat a linear probe | (AO paper) | NLA-acc − probe-acc ≤ 0 |
| F13 | **Writing-quality decay** | Later checkpoints harder to parse | O8 | judge writing-quality score over training |

The two "trust heuristics" the NLA paper already validated — **cross-position recurrence** and **AR reconstruction-leverage** — appear here not as heuristics but as the first two *measurements of evidence*. That reframing is the bridge to §3.

---

## 3. Candidate governing mechanisms (lenses), scored against the evidence

Five candidate unifying mechanisms. For each: the claim, its discriminating prediction, and what the four dumps already say. **This is the data-first adjudication** the scope called for — the dumps can't settle it, but they tilt it.

### Lens 1 — Evidence-starved prior substitution *(lead)*
**Claim:** failures localize wherever the activation carries weak evidence about a claim; the prior fills the gap, conditioned on the recoverable theme.
**Prediction:** a single "evidence" latent (AR-leverage + recurrence + probe-decodability) predicts F1–F6 jointly; failures concentrate in low-variance activation directions.
**Evidence:** **strong.** O1 (theme recovered, specifics not), O2 (gradient), O5 (specifics drift at constant MSE ⇒ ≈0 leverage). This lens directly explains the most universal observations.

### Lens 2 — Reconstruction–faithfulness objective gap (the deep geometric "why")
**Claim:** MSE is a proxy for faithfulness; they decouple where thematically-close-but-false explanations reconstruct as well as true ones (many-to-one text↔activation), *and* because MSE is dominated by high-variance (thematic) directions.
**Prediction:** failure tracks activation-space **degeneracy / local density**; a *whitened* (Mahalanobis) metric would shrink the fidelity–faithfulness gap.
**Evidence:** **indirect but compelling.** O1/O5 show MSE is blind to specifics — consistent with the metric being dominated by coarse directions. **Not yet directly tested** (needs the whitening experiment). I read Lens 2 as the *mechanistic explanation of why leverage is low for specifics*, i.e. the substrate under Lens 1 — not a competitor.

### Lens 3 — Distribution-distance gating (the OOD cliff)
**Claim:** verbalization quality is gated by how in-distribution the activation is for the NLA; off-manifold, the verbalizer falls back entirely to its prior.
**Prediction:** failure rises monotonically with distance-to-training-distribution; the worst failures are at the most OOD positions.
**Evidence:** **strong and sharp.** O4 (Llama 0–30 cliff; orthogonal special tokens everywhere), O6 (raw-norm outliers). This is the *extreme* of Lens 1 — when evidence → 0 (meaningless/OOD activation), substitution → total.

### Lens 4 — Shared-weights introspector (what fills the gap)
**Claim:** because the AV *is* the target model, it conflates *reading* the activation with *re-deriving* what it would think given the inferred context; the gap-filler is the model's own knowledge.
**Prediction:** confabulation content matches the base model's prior; confabulation rate scales with the prior probability of the content.
**Evidence:** **strong for the "what," not yet the "how-much."** O3 (lab entities, Chinese leakage in Qwen only) shows the filler is the prior. Whether *rate* scales with prior-probability is untested. I read Lens 4 as the *substitution operator* in Lens 1 — it names what gets substituted, not a rival cause.

### Lens 5 — Coarse-basin attractor (geometric special case of 1+2)
**Claim:** RL pulls explanations into the correct thematic basin; within-basin specifics are under-constrained.
**Prediction:** faithfulness is high *between* basins (thematic) and collapses *within* (specific); the transition tracks the local eigenspectrum.
**Evidence:** **strong, illustrative.** O5 is almost a definition of this: within-basin specifics drift at constant reconstruction. Largely a restatement of 1+2 in geometric terms.

**Provisional verdict (data-first):** the dumps already let me **collapse Lenses 1, 3, 4, 5 into one account** — *evidence-gated prior substitution*: where activation evidence about a claim is low (Lens 1), up to and including OOD where it's ~zero (Lens 3), the model substitutes its own prior (Lens 4), and because the reconstruction objective only constrains coarse/thematic directions (Lens 5/2) the substitution is rewarded as long as it lands in the right basin. **Lens 2** is the one genuinely distinct, deeper hypothesis still to be *tested* (the whitening/degeneracy experiment) — it explains *why* leverage is low for specifics in the first place.

---

## 4. The novel paradigmatic pattern to chase

Stop treating failures as a checklist of bugs in the verbalizer. Instead:

> **Treat NLA faithfulness as a scalar field — "verbalization evidence" — defined per claim over activation space, and treat every failure mode as a low-evidence region of that field.** The verbalizer is not malfunctioning; it is faithfully reporting a *posterior* that is prior-dominated wherever the activation is uninformative.

This reframing carries three concrete, novel research handles:

1. **A ground-truth-free evidence estimator.** Combine (i) **AR reconstruction-leverage** (ΔMSE when a claim is ablated), (ii) **cross-sample / cross-position recurrence** (O5), and (iii) **distribution-distance / activation-sensitivity** (O4, O7) into a single per-claim score. If one latent absorbs the variance across F1–F6 (the §3 test), you have a **trust meter that needs no labels** — usable at inference to flag, gate, or hedge claims. This is the thing to build first.

2. **The metric is complicit — and that's the fix.** MSE/`cos` is dominated by high-variance thematic directions (O1, O5), so it (a) lets the AV nail the theme while ignoring specifics and (b) makes the AR a *structurally weak per-claim verifier*. The diagnosis (low leverage for specifics) and the disease (a coarse-biased objective) share a cause. **A whitened/Mahalanobis reconstruction metric** that re-weights low-variance directions is therefore both the cleanest *test* of Lens 2 and the most direct route to an *objective-level fix* — it would raise the leverage of exactly the specifics that currently confabulate. (Training-side, so Part B/Phase 5, but it is the keystone.)

3. **Activation-conditional faithfulness.** Apparent faithfulness = genuine-activation-readout + token-predictability (O7). The quantity that actually matters is faithfulness **with token-predictability conditioned out** — "is the explanation right *beyond what the surrounding tokens already imply*." Elevating the AO paper's text-inversion control into a first-class, per-claim NLA metric turns F4/F5 from vague worries into a measured baseline that everything else is scored against.

If this pattern holds, it unifies the taxonomy under one measurable variable, converts "vagueness" from a failure into *calibrated abstention* (honestly reporting low evidence), and hands Part B a single objective: **measure the evidence field, expose it, and make the loss sensitive to it.**

---

## 5. What Part A still needs to resolve (light experiments + the Neuronpedia probes)

Part A is conceptual, but three cheap, inference-only probes would convert the §3 verdict from "tilted" to "supported." Two run on the repo today; the targeted ones want Neuronpedia (the endpoint contract — prompt+position→explanation vs raw-vector-in — decides how cheap they are).

- **P1 — Leverage × veracity (repo today).** On a few hundred claims mined from new decodes: does AR-leverage rank true above false (reproducing the paper's signal) *and* does it fall monotonically theme→entity→detail? Confirms Lens 1/2 and the evidence estimator's first component.
- **P2 — Activation-sensitivity / text-inversion (Neuronpedia).** Fixed final token, varied upstream context → does the explanation track the *activation* or the *tokens*? Quantifies F4/F5 and grounds "activation-conditional faithfulness." This is the single most valuable probe and the main reason to wire Neuronpedia.
- **P3 — Recurrence/consensus (Neuronpedia or repo).** Resample one activation k× → does theme-stability vs specific-drift (O5) hold at scale, and does consensus isolate the true claims? Confirms the estimator's second component.

**Neuronpedia status:** confirmed at least Llama-3.3-70B is hosted (`/{model}/nla`); the public api-doc is JS-rendered, so I still need (a) an API key, (b) one example request/response for the NLA route to fix the input contract, (c) which of the four NLAs are exposed and whether AR mse/cos is returned. If the endpoint is prompt+position→explanation, P2/P3 are essentially free; if raw-vector-in, they pull in the Part B extraction path.

---

## 6. Open questions / what would falsify the lead

- **Falsifier for the unified account:** find a failure that is *not* low-evidence — e.g. a confidently-wrong **specific** claim that has *high* AR-leverage *and* recurs across samples/positions. If high-evidence claims confabulate at the same rate as low-evidence ones, the prior-substitution story is wrong.
- **Falsifier for Lens 2/the metric-complicity fix:** if a whitened metric does *not* shrink the fidelity–faithfulness gap, the coarse-direction-dominance explanation fails and the gap is something else (e.g. genuine many-to-one degeneracy that no reweighting fixes).
- **Steganography (F8) and unverbalizable content (F11)** sit *outside* the evidence-field story as currently framed — they are about information that is present but hidden/inexpressible, not absent. Worth holding separately until measured; they may need a second axis.
- **Calibration (F10):** the evidence score being *predictive* (ranks true above false) is necessary but not sufficient for it being *calibrated* (an evidence of 0.2 means 20% true). Calibration is a distinct, later target.

---

### One-paragraph summary
Across all four released NLAs, the same picture: the theme is recovered, the specifics are confabulated from the model's own prior, reconstruction fidelity certifies the former and is blind to the latter, failures spike on OOD activations, and resampling shows the unreliable specifics are exactly the ones with ~zero reconstruction leverage. This collapses confabulation, specificity-gradient, excessive-expressivity, OOD-cliff, vagueness, and text-inversion into one mechanism — **prior substitution gated by activation evidence** — whose deepest cause is a reconstruction objective dominated by coarse thematic directions. The program is to make "verbalization evidence" a measurable, ground-truth-free scalar field, expose it at inference, and (Part B) make the loss sensitive to it.
