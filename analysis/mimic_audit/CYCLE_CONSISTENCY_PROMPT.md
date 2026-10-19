# Task: implement the "cycle-consistency causal grounding test" end to end

You are working in the NLA repo at:
  /Users/agastyasridharan/ruhr research/natural_language_autoencoders_original

The full method spec is a LaTeX writeup here (READ IT COMPLETELY AND CAREFULLY FIRST,
it is the source of truth for the math and the verdict logic):
  /Users/agastyasridharan/Downloads/cycle-consistency v2 (1).tex

This test extends an existing, working "claim-ablation" harness. Roughly 70% of the
mechanical pipeline the spec describes is ALREADY BUILT and validated on four NLAs
(Qwen2.5-7B L20, Gemma-3-12B L32, Gemma-3-27B L41, Llama-3.3-70B L53). Your job is to
reuse that core and build the new decoy-null + 5-way-verdict machinery on top of it,
ending in a live explanatory GUI and a written report. Do NOT reimplement what exists.

## Step 0 — read before writing any code

1. The .tex spec above (every step, the confound derivation in 5a, and the Step-7 gates).
2. ./CLAUDE.md  (hard invariants; obey all of them).
3. The harness core, under analysis/mimic_audit/athena/repl/ :
   - claim_ablation.py   — READ IN FULL. Key functions you will reuse:
       project_out()            (Step-3 projection ablation: h - alpha*(h.dhat)*dhat)
       atom_claim_directions()  (Step-2 direction: norm(AR(z)) - norm(AR(z\C_i)), unit-normed)
       atom_spans()             (char span -> token range on the teacher-forced sequence)
       build_tf_sequence(), _inject_span_logp()  (Step-4 teacher-forced span log p)
       audit_activation_atoms() (the per-activation atom driver; spec_rows[i][j] IS the
                                 cross-claim effect matrix M_{i->j} you need for Gate 2)
       _atom_specificity(), _classify()  (the CURRENT 3-way verdict; you are ADDING a
                                 separate 5-way verdict, not editing these)
   - claim_taxonomy.py   — the Step-5b TYPE classifier already exists here. THEME_TAXO =
       genre/topic/entity/structure/era/other, DETAIL_TAXO = quote/date/number/value/other.
       Reuse it verbatim for decoy type-matching. Its judge NEVER sees verdicts/Δlogp.
   - segment_run.py + ../decomp_demo/segment.py — Step-1 segmentation (two-layer atoms:
       theme/detail/connective + assertion + hedged + quotes-own-claim + exact tiling).
       This already matches the spec's Step-1 guidelines. Cache: athena/segments/<key>.jsonl.
   - fve.py               — FVE (uses hbar = mean activation). Reuse for filter 4.
   - run_claim_ablation.py — orchestrator (has --claim-mode block|atoms, --reuse-run,
       --segments, summarize()). You will add a new mode without touching the old ones.
   - repl_common.py       — MODELS config (base/AV/AR/layer/d_model/injection_scale/family).
4. The studio GUI: analysis/mimic_audit/athena/studio/server.py + studio/web/index.html
   (stdlib-only HTTP server + single-page app; read-only over runs/, with a taxonomy tab).
5. Existing writeup: docs/research/claim_ablation_atoms.md (method + the current 3-way results).

Then WRITE DOWN a reuse-vs-build map (Steps 1-7 of the spec, each marked EXISTS(reuse X) or
NEW(build Y)) and a staged plan, and show it to me before implementing. Verify my thesis
below against the actual code; correct it where I am wrong.

## Reuse-vs-build thesis (verify, then refine)

EXISTS, reuse directly:
- Step 1 segmentation, Step 2 AR claim directions, Step 3 projection ablation (with
  rescale to the sidecar injection_scale S), Step 4 teacher-forced Δlogp AND the cross-claim
  matrix M_{i->j}, the claim TYPE classifier, and FVE.

NEW, must build (this is the actual work — the matched-decoy null + gated verdict):
- Corpus mean h̄ = mean of gold.npy over the corpus, and a scoring path that injects ι(h̄).
  (The current harness uses a RANDOM-DIRECTION null, not h̄; the spec's power/validity
  pre-checks and filter 3 need h̄.)
- Decoy generation: per claim B, over-generate candidate sibling claims of the SAME type
  that the AV never wrote (Claude Sonnet 4.6, temp 0), then filter to k survivors.
- Splice: z*_m = (z\B) ∪ B*_m (replace B's char span with the decoy text, in B's exact slot).
- Same-subtrahend decoy direction: Δ_{B*_m} = norm(AR(z*_m)) - norm(AR(z\B)), sharing the
  subtrahend AR(z\B) with B's own direction. (Reuse atom_claim_directions' machinery.)
- The four decoy filters:
    (1) Absence from input   — LLM judge: reject decoys that name/paraphrase/are supported
        by any span of the SOURCE DOCUMENT (use the full doc text from
        out_corpus_<key>/manifest.json, not just the 800-char window).
    (2) Matched removed mass — keep decoy iff |h·Δ̂_{B*_m} - h·Δ̂_B| <= eps_ov·|h·Δ̂_B|.
    (3) Matched prior propensity — inject ι(h̄), keep iff the per-token mean log p of the
        decoy in its slot is within eps_pr of B's in its slot.
    (4) Naturalness — LLM judge fluency/coherence of z*_m vs z, AND FVE(z*_m) within a fixed
        tolerance of FVE(z).
  Over-generate and widen bands until >= k decoys survive per claim; LOG claims that can't
  reach k (they become unresolved), never silently drop.
- The three readings per claim: M_BB (=existing dlogp: ablate Δ̂_B, score B in z),
  X_m (ablate Δ̂_{B*_m}, score B in z), D*_m (ablate Δ̂_{B*_m}, score B*_m in the SPLICED
  text z*_m — this needs a fresh teacher-forced sequence per splice).
- The Step-7 pre-checks + gates → FIVE verdicts:
    pre-check Power:    supp(B|h) = score(B|h) - score(B|h̄) clearly positive, else unresolved
    pre-check Validity: supp(B*_m|h) ≈ 0 across the set + D* outlier-free, else unresolved
    Gate 1 (specificity beyond decoys): M_BB > Q95(D*_{1..k}) + kappa_0
    Gate 2 (ownership, if Gate 1 passes): M_BB > 1.5·max_{i≠B} M_{iB} over real claims
            → grounded, else entangled
    Gate 3 (shared signal, if Gate 1 fails): med_m X_m > 0 → seeded, else confabulated
  NOTE a real tension to flag to me: the project's CP4 decision replaced the "max" ownership
  denominator with p90+1-nat-floor because max is scale-biased with claim count. The spec's
  Gate 2 uses max. Implement per spec but surface this and let me choose.
  AMENDMENT (2026-07-22, applied): a non-positive own-effect (M_BB ≤ 0, i.e. erasing B's own
  direction did not hurt B's log-prob) is confabulated outright at Gate 3, and Gate 1
  additionally requires M_BB > 0. "seeded" therefore now requires a POSITIVE own-effect that
  merely fails the decoy null (med_m X_m > 0). Scoped to resolved claims: applied after the
  Power/Validity pre-checks, so a broken/underpowered control still yields unresolved. The gate
  text above is the original ask, preserved; see athena/CYCLE_CAPSTONE.md for the rationale and
  the effect on the counts (seeded → confabulated moves only; grounded/entangled/unresolved
  unchanged).
- Calibration: set eps_ov, eps_pr, kappa_0, the FVE tolerance, and the "clearly positive"/
  "near zero" thresholds on a held-out calibration split; write them to a small JSON +
  a short calibration doc so a run is reproducible.
- GUI + report (see below).

## Hard constraints (non-negotiable)

- JUDGE-FREE MEASUREMENT. LLMs do ONLY: segmentation, type classification, decoy generation,
  the absence judge, and the naturalness judge. The verdict is purely mechanical (Δlogp +
  gates). No LLM ever sees Δlogp, the readings, or verdicts. This is the load-bearing
  epistemics of the whole project — preserve it.
- DO NOT CLOBBER EXISTING RESULTS. Build this as a NEW module
  (repl/cycle_consistency.py) + a new orchestrator mode (e.g. --claim-mode cycle), exactly
  mirroring how atom-mode was added alongside block-mode without touching it. Namespace new
  record fields (cc_verdict, readings, decoys, precheck_*), keep the existing 3-way atom
  verdict fields intact, and write to new run dirs. The published runs
  (qwen7b-cp6, gemma12b-cp7, gemma27b-g4, llama70b-l4 -atoms) must remain byte-reproducible.
- Sidecar is the contract: injection_scale, d_model, token IDs, layer, chat template are
  read from the model sidecar/repl_common and asserted against the live tokenizer. Never
  hardcode. Data-gen never normalizes; normalization happens at injection/loss time.
- Injection health: the marker is scanned inside the hook; cp_size==1; on SGLang paths use
  --disable-radix-cache (input-embeds injection is unsafe with the radix cache); grep every
  generated explanation for CJK as the injection-failure smell test (must be ~0).
- Open-source repo, standard libs only (pathlib, pyarrow, transformers, datasets, httpx,
  pyyaml, numpy, orjson, safetensors, the public anthropic SDK, and whatever Miles/SGLang
  pull in). Do NOT edit anything under miles/. ANTHROPIC_API_KEY is in repo-root .env.
- Compute: ALL Athena GPU runs on GPUs 2,3 ONLY (never 0,1). Cluster python is
  /data/agastyas/Miniconda3/bin/python; staged code under /data/agastyas/nla_mimic/code;
  gold.npy + manifest.json live under /data/agastyas/nla_mimic/out_corpus_<key>*/; set
  HF_HUB_CACHE=/data/hf_cache + HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1; add
  ServerAliveInterval to long ssh sessions and nohup the actual run.
- Cost awareness: decoys multiply forwards (per claim: k decoys × (1 AR reconstruct + a few
  teacher-forced passes for X_m and the spliced D*_m)). Budget accordingly: calibration
  split and tiny-N GPU smoke BEFORE any N=400 run. Cap k and log any cap.

## What uses GPU vs not (so you stage cluster access correctly)

- GPU (Athena, GPUs 2,3): every AV/AR forward — all AR reconstructions (z, z\B, z*_m),
  every injection + teacher-forced Δlogp (M_BB/X_m/D*_m), the h̄-baseline scoring for the
  power/validity pre-checks and the prior-propensity filter, and FVE. The models are 7B-70B,
  so these will not run on a laptop; CPU tests must use a tiny fake AR/injector stub.
- CPU / API only: the LLM judges (segmentation, type, decoy generation, absence, naturalness),
  the gate/verdict arithmetic, h̄ = gold.mean(0), filter thresholding, calibration analysis,
  and the studio GUI server.
- Consequence: the calibration split (S2) and the smoke (S3) already need GPU for the AR/AV
  passes, so plan an Athena session from S2 onward, not only at the full-run stage.

## Staged plan — HARD STOP for my sign-off at each checkpoint

S0. Reuse-vs-build map + plan + proposed module layout + cost estimate. (no code)
S1. h̄ baseline + supp(B|h) power/validity scoring, reusing _inject_span_logp. CPU-unit-
    testable with a tiny fake AR/injector. Exit: unit tests green; supp() sign/scale sane.
S2. Decoy generation + the 4 filters, run on a small CALIBRATION SPLIT of qwen7b. Set and
    record eps_ov, eps_pr, kappa_0, FVE tol. Exit: >=k survivors for most claims; a decoy
    audit table (survivors/claim, rejection reasons) reviewed by me.
S3. Three readings (M_BB/X_m/D*_m) + the 5-way gated verdict in repl/cycle_consistency.py.
    CPU unit tests for the gate logic on synthetic readings covering all five verdicts.
    Then a 1-ACTIVATION GPU smoke that (a) reproduces the existing atom-run M_BB/dlogp for
    the same claims (proves the new path shares the measurement), (b) CJK=0, marker@p, and
    (c) emits sensible readings. Exit: reproduction matches + smoke clean.
S4. Small-N run (e.g. N=25 qwen7b) end to end. Exit: verdict distribution is sane, "seeded"
    is nonzero, unresolved is explained, artifacts + a report.py reviewed by me.
S5. Full qwen7b N=400. Exit: headline table (5-way rates, and cross-tabs vs the type/axis
    labels from claim_taxonomy) + reproducibility.
S6. Cross-model replication on gemma12b / gemma27b / llama70b (reuse the staged gold + AV/AR).
S7. Live GUI + capstone report.

## Live GUI (required, in the studio style)

Extend analysis/mimic_audit/athena/studio (stdlib-only server + SPA) with a "cycle-
consistency" tab: per-claim the three readings, the decoy family with per-filter pass/fail,
the five-verdict decision, and the ownership matrix. Render the spec's diagrams (the pipeline
funnel, the two-channel subtraction, the ablation-as-rotation figure, the channel-coverage
table, the effect matrix, and the taxonomy flowchart) as inline SVG. Include a plain-language
"how it works" section (AV/AR, why projection not CAA, the two confounds, what each of
M_BB/X_m/D*_m controls for, and what "seeded" means). Live-poll during a run. jsdom/DOM-stub
render test. No em-dashes in the chrome (model-generated text is data, leave it untouched).

## Style + reporting

Precise, no sycophancy, complete sentences, no em-dashes, flag uncertainty explicitly.
Mirror the surrounding code's idiom. Commit NOTHING until I approve; stage work and show me
diffs/artifacts at each hard stop. When you report a stage, state what you verified and how,
show the real numbers (including failures), and never claim done-and-verified without the check.

Start with S0: read the spec and the core files, then give me the reuse-vs-build map, the
module layout, and the cost estimate.
