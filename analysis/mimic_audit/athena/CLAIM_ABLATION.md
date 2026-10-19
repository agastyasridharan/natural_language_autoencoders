# Claim-Ablation Test — a judge-free, ground-truth-free faithfulness metric

For each claim an NLA verbalizer makes about an activation `h`, this asks a causal
question: **was the claim read out of `h`, or would the verbalizer have said it
anyway?** It needs no human labels and no knowledge of what the target model was
actually computing — the NLA audits itself.

```
h ──AV──▶ explanation z = {claim C₁ … Cₙ}        (verbalizer: vector → claims)
C  ──AR──▶ Δ = AR(z) − AR(z\C)                    (reconstructor: claim → direction)
           h' = h − (h·Δ̂)Δ̂                       (project the claim's direction out of h)
           Δlogp = logP_AV(C|h) − logP_AV(C|h')   (teacher-force: does the AV still emit C?)
```

Each claim gets one of four verdicts:

- **confabulated** — its `Δlogp` stays within the random-direction null: not read from
  `h` at all (produced from the verbalizer's prior, true or not). The headline finding.
- **entangled** — it beats the null (so it does depend on `h`), but ablating a *different*
  claim's direction suppresses it nearly as much, so the drop cannot be attributed to its
  own direction (two claims share a direction).
- **grounded** — beats the null *and* its own ablation dominates its column of the effect
  matrix (suppresses it more, by `--spec-margin`, than any other claim's ablation does).
- **unlocated** — the claim's tokens were not found in the teacher-forced sequence.

Specificity is **column dominance of the effect matrix**, not "off-diagonal ≈ 0". Random
directions in d=3584 are near-orthogonal to `h`, so the random-direction null sits near
zero and only certifies that a real effect occurred; it cannot be the yardstick for
collateral, because AR claim directions are `h`-aligned and so always move the read a
little. (This was the correction made after the first GPU smoke, where all three claims
beat the null by 11-32 nats but differed sharply in column dominance.)

This measures *faithfulness-to-activation*, the gap the paper's three confabulation
heuristics (factual accuracy, AR ΔMSE, recurrence) cannot close.

The "how do you get a rate from a claim that appears once?" problem is solved by
teacher forcing: we score the log-probability of the claim's exact tokens, which is
defined for a one-off claim. No LLM judge, so the rejected soft-scorer never enters.

## Files

```
repl/extract_corpus.py      fresh FineWeb/UltraFineWeb activations -> gold.npy + manifest.json
repl/claim_ablation.py      core library: injector (gen + teacher-forcing), span log-prob,
                            AR claim directions, projection, null + specificity, audit_activation
repl/run_claim_ablation.py  orchestrator/CLI; writes rich incremental run artifacts (+ a --mock path)
studio/server.py            stdlib http.server read-only viewer of the run artifacts
studio/web/index.html       single-page studio: experimental map, live run, activation detail
```

It reuses the existing harness and honors the repo invariants:
`avgen_model.{load_av, read_sidecar}` (AV load + sidecar), `nla_inference.NLACritic`
(the AR), `repl_common.{MODELS, FAMILIES}`. Activations are `hidden_states[K+1]`
(output of block K), raw/unnormalized, sampled at positions ≥ 50 with a per-doc keyed
RNG. All scales (`injection_scale`, `mse_scale`, marker ids) come from the sidecar.

The **one** mechanism addition is `claim_ablation.LayerZeroInjector`, which injects
at the marker whenever `seq_len > marker_pos`. That single condition serves both the
generation prefill *and* the one full `[prompt + z]` teacher-forced forward, where the
existing `avgen_model.Injector` (gated on `seq_len != prompt_len`) would skip. The
original `Injector` is untouched, so `avgen_model.py` generation is unchanged.

## Run it (Athena, 2–4 × H100)

Models: **Qwen-7B first, then Gemma-3-12B** (each fits on one H100; AV and AR load
separately via `device_map="auto"`). Point the runs dir at box storage so the studio
and the harness agree:

```bash
export CLAIM_ABLATION_RUNS=/data/agastyas/nla_mimic/runs
cd /data/agastyas/nla_mimic/code        # the synced repo (has repl/, nla_inference.py)

# 0. fresh activation corpus  ->  out_corpus_<key>/{gold.npy, manifest.json}
python repl/extract_corpus.py qwen7b   --n 400
python repl/extract_corpus.py gemma12b --n 400        # uses UltraFineWeb by default

# 1. smoke (1 activation, verbose): confirm marker neighbors, no CJK, a plausible split
python repl/run_claim_ablation.py qwen7b --smoke

# 2. corpus run  ->  $CLAIM_ABLATION_RUNS/qwen7b-<stamp>/
python repl/run_claim_ablation.py qwen7b   --n 400
python repl/run_claim_ablation.py gemma12b --n 400
```

Per-activation cost: 1 generation + (1 + n + R) teacher-forced forwards (≈ 14 with
n = 3, R = 10) — all cheap local HF forwards, no SGLang.

Useful flags: `--R` null samples (default 10), `--alpha` ablation strength (default
1), `--null-pct` threshold percentile (95), `--temperature` (0.4), `--gold-dir` to
point at a non-default corpus, `--n` to cap, `--tag` to label a run.

### View it

```bash
CLAIM_ABLATION_RUNS=/data/agastyas/nla_mimic/runs python studio/server.py --port 8011
# then port-forward 8011 and open http://127.0.0.1:8011
```

The page polls the run artifacts, so a run in progress updates live (experimental map
every 4 s, a running run every 3 s).

## Run artifacts (the contract)

```
<runs_dir>/<run_id>/
  run.json        config + LIVE status (atomic-rewritten after every activation)
  records.jsonl   one record per activation: text, snippets, and per-snippet
                  dlogp / null samples / specificity row / ΔMSE / ‖Δ‖ / proj / verdict
  summary.json    corpus aggregates: grounded-rate, verdict counts, Δlogp histogram,
                  Δlogp-vs-ΔMSE scatter + Pearson r, mean specificity
  vectors/<i>.npz full vectors per activation — nothing discarded:
                  h, Δ_i, AR(z), AR(z\Cᵢ), null matrix, per-token logp under h and
                  each ablation, token ids, snippet token spans
```

**Causally-grounded-claim rate** = grounded / located claims, over the whole corpus;
reported alongside the entangled and confabulated rates and the Δlogp↔ΔMSE correlation.

## Develop / preview without a GPU

`--mock` fabricates a schema-identical run (no torch) so the studio can be built and
viewed on a laptop:

```bash
export CLAIM_ABLATION_RUNS=$PWD/analysis/mimic_audit/athena/runs
python analysis/mimic_audit/athena/repl/run_claim_ablation.py qwen7b --mock --slow 0.8
python analysis/mimic_audit/athena/studio/server.py --port 8011   # open http://127.0.0.1:8011
```

Mock runs are tagged "synthetic" in the UI. They exercise every panel (the built-in
Δlogp↔ΔMSE correlation and a realistic grounded/confabulated split) but contain no
real measurements.

## Validate before trusting the metric

- **Positive control** — an activation with a known-grounded claim (e.g. a
  language-switching read, causally validated in the paper) should show large Δlogp
  and high specificity.
- **Negative control** — warm-start boilerplate ("formal academic tone") should sit
  inside the null (correctly flagged confabulated).
- **Labeled separation** — hand-label ~50–100 snippets True/False. Show the test (a)
  separates grounded from ungrounded and (b) flags grounded-but-false claims: the
  headline **grounded ≠ true**.
- **Null sanity** — the random-direction ablation must not suppress claims; the
  specificity matrix must be near-diagonal (shown per activation in the studio).

## Notes / caveats

- MVP scope: snippet-level claims (split on blank lines), single Δ per claim, rank-1
  ablation at α = 1. Sub-snippet claim ablation and big-model scaling are deferred.
- Gemma's AV/AR load through the multimodal wrapper exactly as in `avgen_model.py` /
  `extract_model.py` (the proven cross-model path); `injection_scale` differs wildly
  per model and is always read from the sidecar.
- The studio is intentionally stdlib-only (no FastAPI/uvicorn) so it runs identically
  on the box and a laptop; it is read-only and never loads a model.
