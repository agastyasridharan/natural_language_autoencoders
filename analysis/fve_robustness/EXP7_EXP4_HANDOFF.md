# Handoff — complete Exp 7 (resample stability) then Exp 4 (de-quote), from a cold start

**Audience:** a fresh agent with no prior context. Read this top-to-bottom; it is self-contained.
**Two jobs:** (1) run **Exp 7** — code is built, unit-tested, and adversarially verified; it only needs a
working GPU link. (2) build + run **Exp 4** — not built yet; follow the Exp 3b precedent.

---

## 0. Paths, access, constraints (read first)

- **LOCAL** (Mac, this repo): `/Users/agastyasridharan/ruhr research/natural_language_autoencoders_original/analysis/fve_robustness/`
- **CLUSTER**: host `athena01.ig32s.ruhr-uni-bochum.de`, project **ROOT** `/data/agastyas/nla_fve_ufw/`,
  importable modules in `ROOT/code/` (`repl_common.py` has `MODELS`/`FAMILIES`; `nla_inference.py` has
  `NLACritic`, `normalize_activation`). Frozen per-model artifacts live in `ROOT/out_<key>/`
  (`gold.npy`, `manifest.json`, `av_reads.json`, `cos.json`, baseline `fve_<key>.json`).
- **Sync**: push code/data to the cluster with `scp` (the Exp 7 deploy script does this); results are
  `scp`'d back. (Historically some arms synced via `git pull` on the cluster; scp is simpler and is what
  `deploy_exp7_pilot.sh` uses.)
- **GPU policy (2026-07-24):** **all 4 GPUs (0–3) are the user's and ANY is usable.** `nvidia-smi` first;
  don't stomp a running job. (Older 0,1-reserved / 2,3-only rules are retired.)
- **`.env`** (LOCAL, repo root `../../.env`) holds API keys. Key **names only**, never print values:
  `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `HUGGINGFACE_API_KEY`, `NEURONPEDIA_API_KEY`, `WANDB_API_KEY`.
  Exp 7 clustering and Exp 4 rewriting both use `ANTHROPIC_API_KEY` (Claude Haiku) from here.
- **Gating discipline (user rule):** every experiment checks in at two gates — (a) present the plan and
  wait, (b) status + confirm before any GPU run or API spend. Orchestrate in the main loop, not a
  fire-and-forget background job. Exp 7's plan + pilot are already user-approved ("go with defaults,
  start"); still report at each milestone ("update me regularly").
- **CLAUDE.md invariants that matter here:** open-source deps only; **data-gen never normalizes**
  (`norm="none"`; normalization happens at injection via `injection_scale` and at loss via `mse_scale`,
  both from the sidecar `nla_meta.yaml`); Stage-0 `_MIN_POSITION=50`; **the injection hook scans for the
  marker token ID inside the hook, positions are not precomputed**; `cp_size==1`. **Injection-failure
  smell:** if injection silently fails the AV sees the literal CJK marker char and free-associates
  Chinese — grep generated text for CJK as the loudest smoke test.

---

## 1. What all these experiments share

The object of study is a released open-weight **NLA** (Natural Language Autoencoder). Round-trip:
frozen activation `h = hidden_states[K+1][pos]` → **AV** (a full LLM that reads `h` via a marker-token
injection hook) verbalizes an explanation `z` → **AR** (`NLACritic`, a truncated copy of the target LLM
+ learned affine head) reconstructs `ĥ` from `z`. Quality = **FVE** (fraction of variance explained):

```
FVE = 1 - Σ_i 2(1 - cos_i) / Σ_i D_i        # on unit-normalized golds; D_i = ||ĝ_i - mean(ĝ)||^2
```

**Edit-only mechanic (Exp 1–6, and Exp 4):** edit *only* the AV's explanation text, keep gold / read
position / AV output frozen, and re-run *only* the AR (`rescore.py`). Because gold is frozen the FVE
denominator is identical to baseline, so **ΔFVE is cleanly attributable to the text edit, per datapoint.**
(Exp 7 is different — it re-runs the AV generation itself; see §4.)

**Held-out eval:** n=2000 Ultra-FineWeb docs (`data/ufw_docs.json`, id→text), one keyed-random read
position ≥50 per doc.

**Models** (`key`, layer, `d_model`): `qwen7b` (Qwen2.5-7B-Instruct, L20, 3584), `gemma12b`
(gemma-3-12b-it, L32, 3840), `gemma27b` (gemma-3-27b-it, L41), `llama70b` (Llama-3.3-70B, L53) — **llama
is baseline-only**; the manipulation suite runs on the other three.

**Baseline FVE:** qwen7b 0.748, gemma12b 0.750, gemma27b 0.768, llama70b 0.788. (Gemma mean-cos ≈0.99 but
FVE ≈0.75: Gemma activations are anisotropic, so FVE is geometry-amplified vs cosine — keep this in mind
reading Gemma deltas.)

---

## 2. The experiment arc so far (why Exp 7 exists)

All ΔFVE vs baseline, per model, from `results/fve_<arm>_<key>.json`:

| arm | edit | qwen7b | gemma12b | gemma27b | takeaway |
|---|---|---|---|---|---|
| 1 | remove **all** quotes | −0.99 | −0.73 | −0.82 | quoted spans carry ~all reconstruction |
| 2 | keep **only** quotes | −0.48 | −0.19 | −0.26 | quotes mostly sufficient; prose a real increment |
| 3a | remove **verbatim** input-derivable quotes | −0.04 | −0.03 | −0.07 | input-restating quotes barely matter |
| 3b | remove **paraphrase** input-derivable quotes (LLM judge) | −0.03 | −0.03 | −0.08 | ≈3a → load-bearing quotes are the **divergent/confabulated** ones |
| 5a | shuffle whole blocks | −0.05 | −0.02 | −0.02 | AR is largely **order-insensitive** |
| 6a | LCR→true pre-token ctx, token **included** | +0.004 | +0.019 | +0.012 | (boundary-token echo) |
| 6b | LCR→true pre-token ctx, token **excluded** | −0.204 | −0.128 | −0.177 | true input **collapses** FVE → **not input inversion** |
| 6f | LCR→shuffle AV's own words | −0.082 | −0.025 | −0.037 | |
| 6r | LCR→random same-doc span | −0.288 | −0.090 | −0.093 | |
| 6o | LCR→**other doc's** true ctx | −0.515 | −0.159 | −0.171 | coherent wrong-doc content hurts most |

**Story:** the load-bearing quotes are the *divergent/confabulated* left-context-reconstruction (LCR)
quotes, not the input-derivable ones (Exp 3). Substituting the **literal true input** for them *collapses*
FVE instead of raising it (Exp 6b) → the AR is **not** inverting the input; it uses the LCR quote as a
**position-anchored content slot** where AV confabulation serves as well as (or better than) truth, and
damage scales with how *coherently* a substitute mis-describes the activation's location.

**Exp 7's question:** are those load-bearing confabulated claims **activation-determined** (recur across
resamples → a genuine, if abstract, read — or a systematic AV prior) or just **sampler noise**?

---

## 3. EXP 7 — resample the verbalizer (BUILT; run it)

### 3.1 Idea, hypotheses, controls
The AV is stochastic: same `h`, different `z` each draw. Treat `z` as a random variable; resample **N=24×**
on the *frozen* `h` across a temperature grid; split each `z` into typed claims; cluster claims **by
meaning**; read a **stable core** (recurs in most resamples) vs a **temperature-sensitive periphery**.
Crossed with Exp-6 input-derivability → 2×2; the interesting cell is **stable × not-in-input** (candidate
privileged read OR AV prior).

Hypotheses: **H1** consensus histogram is bimodal (core vs periphery, not a flat middle); **H2** meaning
diversity of the core rises with T much slower than surface diversity; **H3** genre+LCR are the stable
core, **PC (predicted continuations) are the periphery**; **H4 (the killer)** the real core must exceed
the **no-injection floor**; **H5** greedy(T=0) claims ⊆ T=0.4 core.

**Grid** `{0.0×1 greedy, 0.4, 0.7, 1.0, 1.5}` — T=0 drawn **once** (deterministic; resampling it would
fake a core); T>0 draws N=24. At T=1.5 expect a real filtered (CJK/empty) fraction — that is the
"noise-grows-with-T" signal, so the harness reports **effective-N per (datapoint, temperature)**.

**Three arms** (all reuse the identical injection path; only the injected vector differs):
- `main` — inject `gold[idx]` (the real read), full grid.
- `floor` — inject **nothing** (hook pass-through: marker token present, activation absent = the
  CLAUDE.md injection-failure path). T=0.4, N=24. **H4 control, required.**
- `shuffle` — inject a **different** subsample doc's `gold` (deterministic derangement, different
  doc_id). T=0.4, N=24. Disambiguates read vs prior.

### 3.2 Pipeline & files (data flow)
```
exp7_subsample.py ─> results/exp7_subsample_<key>.json   (200 docs/model, stratified by Exp-6 divergence)
        │  [DONE for all 3 models: ~67/67/66 low/mid/high thirds]
        ▼
fve_resample.py (GPU, on cluster) ─> ROOT/out_<key>/resample_{main,floor,shuffle}.json
        │  samples keyed "idx|T|k" -> {idx,T,k,text,cjk,noexp,empty,stratum,divergence}
        │  + top-level {grid, per_T effective-N, provenance{inj_scale,marker_pos,prompt_len,...}}
        ▼   [pull back to LOCAL out_<key>/]
exp7_cluster.py (API, Mac) ─> out_<key>/exp7_clusters_{main,floor,shuffle}.json
        │  per (idx,T): effective_N, tags{genre,LCR,PC:{claims,meaning-clusters,surface-clusters}}
        │  meaning clustering = exp7_judge.cluster_claims (Haiku); cache results/exp7_haiku_cache.json
        ▼
exp7_metrics.py (Mac) ─> results/exp7_metrics.json
        │  per (T,tag): core-fraction, consensus histogram, entropy, pair-agreement, surface gap,
        │  + T=0.4 contrasts: main vs floor vs shuffle vs permutation-null; doc-bootstrap 95% CIs
        ▼
exp7_dashboard.py (Mac) ─> exp7_resample.html   (resample-spread cards + summary; currently placeholder)
```

File-by-file:
- **`EXP7_RESAMPLE_PLAN.md`** — the full spec (idea, H1–H5, protocol, controls, budget, gated run order,
  resolved decisions in §9). Read it.
- **`exp7_subsample.py`** — DONE, already produced the 3 subsample files. Deterministic (seed 7).
- **`exp7_judge.py`** — the meaning-judge + its **Step-7.0 sanity gate** (already PASSED: 0 false-merges,
  95.1% pairwise agreement on a labelled set incl. the "robotic vacuum / owner's husband / cleaning
  staff" same-topic trap). `cluster_claims(list[str]) -> list[list[int]]` is the reusable partitioner.
- **`fve_resample.py`** — the GPU harness. **Imports the injection path VERBATIM from `fve_ufw.py`**
  (`BatchInjector`, `build_actor_prompt`, `find_layers`, `read_sidecar`, `load_av`). Args:
  `python fve_resample.py <key> --arm {main,floor,shuffle} [--pilot K] [--batch 48]`. Resume-safe
  (checkpoints per arm; `assert batch >= N_RESAMPLE=24`).
- **`exp7_cluster.py`** — claim extraction (reuses `exp6_edits.analyze_datapoint` for authoritative
  genre/LCR/PC tagging) + meaning + ROUGE-L surface clustering. `--no-api` runs extraction+surface only.
- **`exp7_metrics.py`** — stability aggregates + contrasts + doc-bootstrap CIs.
- **`exp7_dashboard.py`** — renders `exp7_resample.html` (server-side; degrades to "pilot pending").
- **`deploy_exp7_pilot.sh`** — one-command runbook: connectivity check → all-GPU probe + auto-pick
  least-used → scp harness+subsample → run 3 arms → pull resamples → cluster+metrics+dashboard locally.

### 3.3 Verification status (already done)
A 5-lens adversarial review of `fve_resample.py` returned **GO-with-fixes**; the critical property —
*no row can be scored against the wrong activation* — is verified (injection byte-identical to the
headline `fve_ufw.phase_avgen`; floor/shuffle semantics correct; frozen gold reused not re-extracted).
**All fixes are already landed:** resume-stable seeding (batch over full doc×k grid, seed on stable
ordinal), `empty`-flag + per-T effective-N, `{tag}.json` existence guard, donor `n≥2`/no-self-map
asserts, provenance persisted, `assert batch≥N`. No open harness issues.

### 3.4 EXACT run steps (once ssh works)
Gated order: **pilot → report → full → report → dashboard**. The user pre-authorized the run; still
report at each milestone.

1. **Pilot (smoke, ~10 docs, one model):**
   ```
   bash deploy_exp7_pilot.sh qwen7b 10
   ```
   This runs all 3 arms on 10 docs across the full grid, pulls resamples, clusters (API, ~$0.15),
   metrics, and renders `exp7_resample.html`. **Inspect** the resample spreads + effective-N per T +
   that CJK is near-zero at low T. Report, then proceed.
2. **Full run (200 docs × 3 models):** on the cluster, for each `key` in `qwen7b gemma12b gemma27b` and
   each `arm` in `main floor shuffle`:
   ```
   CUDA_VISIBLE_DEVICES=<free-gpu> HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
     python fve_resample.py <key> --arm <arm> --batch 48
   ```
   (~87k generations total; a few GPU-hours/model, batched. gemma27b is largest — batch=48 is fine, the
   headline run cleared batch=64 on the same model.) Pull `out_<key>/resample_*.json` back to LOCAL
   `out_<key>/`.
3. **Cluster + metrics + dashboard (Mac):**
   ```
   for key in qwen7b gemma12b gemma27b; do
     for arm in main floor shuffle; do python3 exp7_cluster.py $key --arm $arm; done
   done
   python3 exp7_metrics.py qwen7b gemma12b gemma27b
   python3 exp7_dashboard.py --cards 40
   python3 build_experiments_dashboard.py     # refresh nav so Exp 7 shows as populated
   ```
4. **Update the dashboard "clearly accordingly"** (user's explicit ask): the Exp 7 tab auto-fills from
   the cluster JSONs; make sure the intro states the result plainly (core beats floor? PC = periphery?
   stable×not-in-input populated?), and add the headline to `index.html` overview. Write a short
   `EXP7_CAPSTONE.md` with the H1–H5 verdicts.

### 3.5 Exp 7 completion checklist
- [ ] pilot inspected, spreads sane, CJK low at low-T
- [ ] full `resample_{main,floor,shuffle}.json` for all 3 models pulled to LOCAL `out_<key>/`
- [ ] `exp7_clusters_*.json` (9 files) + `exp7_metrics.json` built
- [ ] H4 checked: `main` core > `floor` core (else the "stability" is a prior — report honestly)
- [ ] H3 checked: PC periphery vs genre/LCR core
- [ ] shuffle vs main contrast reported (read vs prior)
- [ ] `exp7_resample.html` populated; `index.html` overview updated; capstone written
- [ ] check in with the user

---

## 4. EXP 4 — rephrase the quoted block into natural language (BUILD, then run)

### 4.1 Idea
Exp 1–3 removed quotes; Exp 4 **keeps the meaning but destroys the quotation FORM**: rewrite each quoted
span as fluent natural-language prose asserting the same content, with **no verbatim quotes**. This
separates *quoted content* from *quotation form / verbatim surface string*.
- If FVE **holds** after de-quoting → the AR reads the *content*, the quote marks/verbatim strings are
  incidental.
- If FVE **drops** → the AR leans on the verbatim surface form (the literal token strings), not just
  the meaning. Given Exp 6 (the AR loves the boundary-token echo and the exact LCR surface), a drop is
  plausible and would sharpen the Exp 6 story.

Status: only the **dashboard scaffold exists** — `exp4_dequote.html` (greyed "planned") and the ARMS
entry `armkey="exp4", slug="exp4_dequote", kind="API"` in `build_experiments_dashboard.py`. No edit
script yet.

### 4.2 Build path — mirror Exp 3b exactly (it is the closest precedent)
Exp 3b (`tierb_judge.py`) is an **API arm**: a Haiku call per datapoint edits the explanation, writes
records + edited_reads, then a GPU `rescore.py` pass fills FVE. Exp 4 is the same shape, but the API call
**rewrites** quoted spans into prose instead of **removing** derivable ones.

Create **`exp4_rewrite.py`** (model it on `tierb_judge.py`), doing, per in-range datapoint:
1. Load `results/records_exp3a_<key>.json` (reuse its display fields + `expl_orig` + char offsets), and
   `data/ufw_docs.json`.
2. Find quoted spans with the shared helpers in **`edits.py`** (`_curly_spans`/`_straight_spans`/
   `split_blocks`; see how `exp6_edits.classify_expl_spans` locates quotes).
3. One **Haiku** call (`ANTHROPIC_API_KEY` from `../../.env`, `anthropic.Anthropic()`, model
   `claude-haiku-4-5-20251001`, `temperature=0`, a **tool** for structured output) that rewrites the
   explanation so every quoted span becomes fluent prose of the **same meaning with the quote marks and
   verbatim strings gone**, leaving the rest of the explanation intact. Hardening: forbid re-introducing
   quotes; forbid changing claims; keep genre/register wording. Thread-pool for throughput (see
   `tierb_judge.py`'s `ThreadPoolExecutor` + a lock + a JSONL cache like `results/tierb_cache_<key>.jsonl`
   so reruns are cheap).
4. Write, per model, mirroring `records_common` schema and the Exp 3b/6 outputs:
   - `results/records_exp4_<key>.json` (rich records; `cos/FVE` null until re-score)
   - `results/edited_reads_exp4_<key>.json` = `{"reads": {idx: edited}, "arm": "exp4"}`
   - `results/exp4_summary_<key>.json` (quotes rewritten, any refusals)
5. **Sanity before GPU:** grep the edited reads for residual quote marks (should be ~0) and for CJK
   (injection-failure smell). Show the user ~50 examples in the Exp 4 tab format first (the user asked
   for this on Exp 6 too — do it here).

### 4.3 Run + score
1. Deploy the edited reads to the cluster and **re-score on GPU** (same as every arm). The rescore reads
   `ROOT/out_<key>/edited_reads_<arm>.json`, so land the file there (the deploy step renames
   `edited_reads_exp4_<key>.json` → `out_<key>/edited_reads_exp4.json`). Then:
   ```
   CUDA_VISIBLE_DEVICES=<free-gpu> python rescore.py <key> exp4
   ```
   which writes `out_<key>/cos_exp4.json`, `out_<key>/fve_exp4_<key>.json` (ΔFVE + bootstrap CI +
   Wilcoxon), and fills `records_exp4_<key>.json`.
   **Run the NULL gate first once per model:** `python rescore.py <key> exp4 --null` must reproduce
   baseline cos (max|Δ|<1e-3) — proves the AR harness is wired to the frozen gold correctly.
2. Pull `fve_exp4_<key>.json` + `records_exp4_<key>.json` back to LOCAL `results/`.
3. `python3 build_experiments_dashboard.py` — the Exp 4 tab auto-populates (arm already registered) and
   flips from "planned" to done.

### 4.4 Exp 4 completion checklist
- [ ] `exp4_rewrite.py` written; ~50 examples shown to the user; residual-quote grep ~0; CJK ~0
- [ ] `records_exp4_<key>.json` + `edited_reads_exp4_<key>.json` for all 3 models
- [ ] NULL gate passes per model
- [ ] `rescore.py <key> exp4` run; `fve_exp4_<key>.json` pulled back
- [ ] Exp 4 tab populated; overview updated; result interpreted vs Exp 6 (content vs form)
- [ ] check in with the user

---

## 5. Full file index (LOCAL `analysis/fve_robustness/`)

**Shared harness / helpers**
- `fve_ufw.py` — the headline FVE pipeline (extract gold → AV avgen → AR score → FVE). **Source of the
  injection path** Exp 7 reuses. Runs on the cluster.
- `edits.py` — shared quote/span helpers (`_curly_spans`, `_straight_spans`, `split_blocks`, `_cleanup`,
  `_seed`, …) used by every edit arm.
- `records_common.py` / `records_common_cluster.py` — `build_records(key, arm, edit_fn, docs)`: re-tokenize
  each doc like extraction, recover token + char offsets, emit the uniform per-datapoint record schema.
- `rescore.py` — generic AR re-score (+ `--null` gate). The GPU step for every edit arm.
- `build_experiments_dashboard.py` — builds `index.html` + one tab per ARM (+ the Exp 7 nav link).
  `cleanup_stale()` keeps `exp7_resample.html`.

**Exp 6 (context + the parser Exp 7 reuses)**
- `exp6_edits.py` — the block/quote parser + directional tagger (`classify_expl_spans`,
  `analyze_datapoint`, `rouge_l`, `split_blocks`). **Exp 7 claim extraction depends on this.**
- `exp6_build_arms.py`, `exp6_build_controls.py`, `exp6_audit.py` — build the 6a/6b/6f/6r/6o arms.

**Exp 7 (this handoff)** — `EXP7_RESAMPLE_PLAN.md`, `exp7_subsample.py`, `exp7_judge.py`,
`fve_resample.py`, `exp7_cluster.py`, `exp7_metrics.py`, `exp7_dashboard.py`, `deploy_exp7_pilot.sh`.

**Exp 3b (Exp 4 template)** — `build_tierb_input.py`, `tierb_judge.py`.

**Data / results** — `data/ufw_docs.json` (the 2000 docs); `results/` (per-arm `records_*`,
`edited_reads_*`, `fve_*`, the 3 `exp7_subsample_*`); cluster `out_<key>/` (frozen `gold.npy`,
`manifest.json`, `av_reads.json`, `cos.json`, baseline `fve_<key>.json`).

---

## 6. TL;DR next actions
1. **Exp 7:** `bash deploy_exp7_pilot.sh qwen7b 10` → inspect → full 3×3 run → cluster/metrics/dashboard →
   update dashboard + capstone → check in.
2. **Exp 4:** write `exp4_rewrite.py` (Haiku, mirror `tierb_judge.py`) → show 50 examples → NULL gate →
   `rescore.py <key> exp4` → dashboard → interpret vs Exp 6 → check in.
Keep the user updated at every milestone.
