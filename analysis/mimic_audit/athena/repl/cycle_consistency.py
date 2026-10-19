"""Cycle-consistency causal grounding test - the decoy-null extension of the
claim-ablation harness. S1 scope: baseline + support scores + Step-7 pre-checks.

THE IDEA (plain English)
------------------------
The existing claim-ablation test (claim_ablation.py) compares each claim's ablation
effect against a RANDOM-direction null. The cycle-consistency spec replaces that null
with a family of DECOY claims - fabricated claims of the same type that are absent
from the input - and splits the ablation effect dlogp into three channels:

  theme      "any plausible claim of this kind loses probability when you rotate h"
  artifact   "subtracting a claim's own AR direction suppresses that claim's tokens
              even if h never encoded it"  (the anti-self -cab*shat term)
  signal     the part that is actually read out of h  ->  grounding

Three readings per claim B (S3) separate the channels; five verdicts follow from
pure arithmetic gates (grounded / entangled / seeded / confabulated / unresolved).

WHAT THIS MODULE PROVIDES (S1)
------------------------------
The pieces underneath the decoy machinery, all judge-free:

  corpus_hbar        h_bar = raw mean of the gold corpus activations. Injecting it
                     (spec iota(h_bar)) is "a generic activation": the AV's prior.
                     NOTE the spec's single "h_bar" is TWO objects in this codebase:
                     the raw mean here (injection; _set_v rescales to the sidecar
                     injection_scale) and the unit-mean below (FVE denominator).
  unit_hbar_di       (unit-mean, per-row D_i) exactly as fve.py computes them; S2's
                     filter 4 scores FVE(z*) = 1 - 2(1-cos)/D_i from these.
  span_supports      supp(C|h) = score(C|h) - score(C|h_bar) for every span in two
                     teacher-forced forwards (spec Step 7: the pre-check quantity;
                     "Power" for the real claim, "Validity" for decoys).
  span_mean_logp     per-token mean log p of a span under iota(h_bar) - spec Step 5b
                     filter 3's "matched prior propensity" quantity.
  precheck_power     Step 7 entry gates as pure functions over stored numbers, so
  precheck_validity  re-gating never needs a GPU. Thresholds (tau_pow, tau_val) come
                     from the S2 calibration JSON - never hardcoded here.

It reuses the SAME measurement path as the published atom runs (_inject_span_logp ->
_token_logp -> _span_sums, batch-1 LayerZeroInjector), so cycle numbers are directly
comparable to qwen7b-cp6 / gemma12b-cp7 / gemma27b-g4 / llama70b-l4 atom records.

JUDGE-FREE MEASUREMENT: nothing in this module calls an LLM. Every quantity is a
teacher-forced log-prob or numpy arithmetic; no LLM ever sees these numbers.

S2 adds the decoy MECHANICS (judge-free side): the slot splice, the
same-subtrahend directions, filter 2 (matched removed mass, with the spec's
band-widening), filter 3 (matched prior propensity) and filter 4b (FVE band).
Filters 1 and 4a are LLM text judges and live in decoy_gen.py; they run BEFORE
any measurement exists and never see one.

Later stages extend this module: the three readings M_BB / X_m / D*_m and
Gates 1-3 (S3).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

# Make the athena harness importable both on the GPU box and from a checkout.
_REPL = Path(__file__).resolve().parent          # .../athena/repl
_ATHENA = _REPL.parent                            # .../athena
for _p in (str(_REPL), str(_ATHENA), "/data/agastyas/nla_mimic/code"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from claim_ablation import _inject_span_logp      # noqa: E402
from nla_inference import normalize_activation    # noqa: E402

_EPS = 1e-8


# ── the corpus-mean baseline h_bar ───────────────────────────────────────────

def corpus_hbar(gold) -> np.ndarray:
    """Raw-mean baseline for INJECTION: h_bar = mean over gold.npy rows, float32.

    Do NOT pre-normalize - _set_v rescales whatever it is given to the sidecar
    injection_scale, so iota(h_bar) = S * h_bar / ||h_bar|| falls out of the same
    path every real h takes. Distinct from the FVE unit-mean (unit_hbar_di).
    """
    g = np.asarray(gold, dtype=np.float32)
    assert g.ndim == 2 and g.shape[0] > 0, f"gold must be [N,d], got {g.shape}"
    hbar = g.mean(axis=0).astype(np.float32)
    assert float(np.linalg.norm(hbar)) > _EPS, "degenerate corpus mean (~zero norm)"
    return hbar


def unit_hbar_di(gold) -> tuple[np.ndarray, np.ndarray]:
    """FVE pair (unit-mean, per-row D_i), the exact fve.py computation:
    Ghat = row-unit-normalized gold, hbar_u = Ghat.mean(0), D_i = ||Ghat_i - hbar_u||^2.

    Filter 4 (S2) scores a decoy explanation's reconstruction via
    fve_from_cos(cos(AR(z*), h_i), D_i) - no extra GPU forward needed.
    """
    g = np.asarray(gold, dtype=np.float64)
    assert g.ndim == 2 and g.shape[0] > 1, f"gold must be [N>1,d], got {g.shape}"
    norms = np.linalg.norm(g, axis=1, keepdims=True)
    assert float(norms.min()) > _EPS, "zero-norm gold row"
    ghat = g / norms
    hbar_u = ghat.mean(axis=0)
    di = ((ghat - hbar_u) ** 2).sum(axis=1)
    return hbar_u, di


def fve_from_cos(cos: float, di: float) -> float:
    """Per-position FVE from a cosine: 1 - 2(1-cos)/D_i (fve.py's row formula)."""
    if not (np.isfinite(cos) and np.isfinite(di)) or di <= 0:
        return float("nan")
    return float(1.0 - 2.0 * (1.0 - cos) / di)


# ── support scores: supp(C|h) = score(C|h) - score(C|h_bar) ──────────────────

def span_supports(model, injector, input_ids, h, hbar, inj_scale, ranges) -> dict:
    """Spec Step 7 pre-check quantity for every span, in exactly two teacher-forced
    forwards over the SAME [prompt+z] token sequence:

        supp[i] = score(span_i | h) - score(span_i | iota(h_bar))

    Positive supp = the span's exact tokens are likelier under the real activation
    than under the generic corpus-mean injection; that is what "encoded in h" means
    operationally. The real claim must clearly clear this bar (Power); every decoy
    must NOT (Validity).

    Returns dict (all JSON-safe except the two per-token float arrays):
      l0        [n]  score(span_i | h)            nats, nan for unlocated spans
      lbar      [n]  score(span_i | h_bar)
      supp      [n]  l0 - lbar                    (nan propagates)
      mean_lbar [n]  per-token mean of lbar       filter 3's prior-propensity value
                     for spans scored in their own slot (B's side of the match)
      tlp_h, tlp_hbar   per-token logp arrays [T-1] for vectors/<idx>.npz
    """
    l0, tlp_h = _inject_span_logp(model, injector, input_ids, h, inj_scale, ranges)
    lbar, tlp_hbar = _inject_span_logp(model, injector, input_ids, hbar, inj_scale,
                                       ranges)
    supp = [a - b for a, b in zip(l0, lbar)]
    return dict(l0=l0, lbar=lbar, supp=supp,
                mean_lbar=span_mean_logp(lbar, ranges),
                tlp_h=tlp_h, tlp_hbar=tlp_hbar)


def span_mean_logp(sums, ranges) -> list[float]:
    """Per-token mean log p per span: sum / token-count. Filter 3 (matched prior
    propensity) compares this value for a decoy in its own slot against the real
    claim's - lengths differ, so the comparison must be per-token, not summed."""
    out = []
    for s, r in zip(sums, ranges):
        if r is None or not np.isfinite(s):
            out.append(float("nan"))
            continue
        lo, hi = r
        out.append(float(s) / (hi - lo) if hi > lo else float("nan"))
    return out


# ── Step 7 pre-checks (pure arithmetic; thresholds from the calibration JSON) ─

def precheck_power(supp_b: float, tau_pow: float) -> dict:
    """Power: the real claim B must be CLEARLY encoded - supp(B|h) > tau_pow.

    The spec (Step 7) requires only that supp(B|h) be "clearly positive"; it
    prescribes no threshold or procedure. tau_pow is therefore a HARNESS
    OPERATIONALIZATION fixed at S2 sign-off. Caveat found in S2 calibration
    (qwen7b): the decoy supp population is NOT near zero (median +26.5 nats,
    87% positive - theme-matched decoys inherit topical support from h) and
    span-sum supp is length-confounded (corr with n_span_tokens ~0.74), so a
    pooled Q95 of decoy supp is a very aggressive threshold; per-family and
    per-token constructions are recorded in calibration_proposed.json as
    alternatives. Fail -> unresolved. NaN (unlocated span) fails."""
    ok = bool(np.isfinite(supp_b) and supp_b > tau_pow)
    return dict(passed=ok,
                supp_b=float(supp_b),
                tau_pow=float(tau_pow),
                margin=float(supp_b - tau_pow) if np.isfinite(supp_b) else None)


def precheck_power_family(supp_b: float, supp_family, q: float = 95.0) -> dict:
    """Power, per-family construction (signed package, S2 sign-off 2026-07-21):
    supp(B|h) must exceed the Q95 of the claim's OWN surviving decoys' supp*.

    Rationale from S2 calibration (qwen7b, d3g12): the pooled decoy supp* null
    is fat and topic-inherited - corr(supp_b, own-family median supp*) = 0.88,
    and span-sum supp is length-confounded (corr with n_span_tokens = 0.74).
    Comparing B against ITS OWN family's decoys matches topic and span length
    by construction, so the threshold is a matched null rather than a pooled
    mixture over unrelated claims.

    supp_family must be the SURVIVING decoys' supp* (post filters 1-4), so the
    null is the same family Gate 1 will use. Empty family or any non-finite
    entry fails CLOSED (-> unresolved): a family we cannot score is not a null."""
    fam = np.asarray(list(supp_family), dtype=np.float64)
    if fam.size == 0 or not np.all(np.isfinite(fam)) or not np.isfinite(supp_b):
        return dict(passed=False,
                    supp_b=float(supp_b) if np.isfinite(supp_b) else None,
                    q95_family=None, margin=None, n_family=int(fam.size))
    q95 = float(np.percentile(fam, q))
    return dict(passed=bool(supp_b > q95),
                supp_b=float(supp_b), q95_family=q95,
                margin=float(supp_b - q95), n_family=int(fam.size))


def precheck_validity(supp_decoys, tau_val: float, dstar=None,
                      fence_k: float = 3.0, min_k: int = 5) -> dict:
    """Validity: the decoy family must contain a working placebo null.

    Flags (per-index, computed exactly as before):
      - encoded:      |supp(B*_m | h)| > tau_val. A decoy the activation actually
                      supports is not a placebo (supp_viol_idx).
      - D* outlier:   high-side MAD-fence outlier, fence = median + fence_k *
                      1.4826 * MAD (dstar_outlier_idx). Median/MAD rather than
                      Q3+IQR because at k ~ 5 a single contaminant escapes an IQR
                      fence; the MAD fence holds with up to half the family broken.
                      Low-side outliers only make Gate 1 harder, so not flagged.
      Non-finite entries (unlocated decoy spans / supp) always flag.

    Tier-1 decoupling (2026-07-22). Previously a SINGLE flagged decoy failed the
    whole claim closed (-> unresolved). Audit of the 4-model runs showed this is
    the dominant unresolved cause and that most of it is one high-side D* outlier
    in an otherwise clean family. So a flag no longer condemns the claim: drop the
    flagged decoys from the clean-placebo COUNT and PASS iff >= min_k CLEAN
    placebos remain. The flagged decoys are NOT removed from the Gate-1 null -
    classify_claim receives the FULL survivor D* set, so trimming here can never
    lower the grounding bar (Q95(D*) still includes the high outliers). This is
    the 'monotone-null lock': it recovers the 'one bad decoy killed a good family'
    slice with provably no false-grounding path (verified by the replay A/B, where
    the naive drop-from-null variant manufactured grounded verdicts and this one
    does not). n < min_k still fails closed (too few decoys to form any null).

    Returns per-index flags + n_clean/dropped_idx so callers can report exactly
    which decoys were trimmed and why. CONTRACT: classify_claim's D* null MUST be
    the full survivor set (cycle_run passes the same `dstars`), or the lock breaks."""
    supp = np.asarray(supp_decoys, dtype=np.float64)
    if supp.size < min_k:
        return dict(passed=False,
                    reason=f"insufficient decoys (n={int(supp.size)} < min_k={min_k})",
                    n=int(supp.size), n_clean=int(supp.size), dropped_idx=[],
                    supp_viol_idx=[], dstar_outlier_idx=[],
                    supp_max_abs=None, fence=None)
    supp_viol = [int(i) for i in range(supp.size)
                 if not np.isfinite(supp[i]) or abs(supp[i]) > tau_val]

    out_idx, fence = [], None
    if dstar is not None:
        d = np.asarray(dstar, dtype=np.float64)
        assert d.size == supp.size, f"dstar/supp length mismatch {d.size}/{supp.size}"
        bad = [int(i) for i in range(d.size) if not np.isfinite(d[i])]
        fin = d[np.isfinite(d)]
        if fin.size:
            med = float(np.median(fin))
            mad = float(np.median(np.abs(fin - med)))
            fence = med + fence_k * 1.4826 * mad + 1e-9   # slack: ties never flag
            out_idx = sorted(set(bad) | {int(i) for i in range(d.size)
                                         if np.isfinite(d[i]) and d[i] > fence})
        else:
            out_idx = bad

    # decouple: trim flagged from the CLEAN COUNT, keep them in the Gate-1 null.
    dropped = sorted(set(supp_viol) | set(out_idx))
    n_clean = int(supp.size) - len(dropped)
    passed = n_clean >= min_k
    fin_supp = supp[np.isfinite(supp)]
    return dict(passed=passed,
                reason="" if passed else
                       (f"insufficient clean placebos (clean={n_clean} < "
                        f"min_k={min_k}; {len(supp_viol)} encoded, "
                        f"{len(out_idx)} D* outlier)"),
                n=int(supp.size),
                n_clean=n_clean,
                dropped_idx=dropped,
                supp_viol_idx=supp_viol,
                dstar_outlier_idx=out_idx,
                supp_max_abs=float(np.abs(fin_supp).max()) if fin_supp.size else None,
                fence=fence)


# ── Step 7: five-way gated verdict (pure arithmetic on the three readings) ───

# spec Step 7 gate constants (fixed here, not calibrated): Gate 2 ownership
# factor and the Gate-1 kappa are the only free knobs; kappa_0 comes from the
# calibration JSON, GATE2_FACTOR is the spec's literal 1.5.
GATE2_FACTOR = 1.5


def classify_claim(power: dict, validity: dict, m_bb: float, dstars,
                   x_ms, m_ib_max, kappa_0: float,
                   gate2_factor: float = GATE2_FACTOR) -> dict:
    """Turn the three readings (M_BB, D*_1..k, X_1..k) + the effect-matrix column
    max_{i!=B} M_iB into one of five verdicts (spec Step 7). PURE arithmetic - no
    GPU, no LLM, fully unit-testable. No LLM ever sees any argument here.

    Pre-checks:
      power    : precheck_power_family result - supp(B|h) clearly positive.
      validity : precheck_validity result - >= min_k CLEAN placebos remain after
                 trimming encoded / D* outliers (Tier-1 decoupling).
      A validity fail -> unresolved. A power fail -> unresolved, EXCEPT the Tier-1
      confabulated shortcut: a power-fail with a VALID family and M_BB <= 0 is
      confabulated (only-confabulated, so no false-positive path); M_BB > 0 with a
      failed power bar (the ambiguous middle) stays unresolved.

    Gates (only for claims passing both pre-checks):
      1 specificity : M_BB > Q95(D*_1..k) + kappa_0 ?  pass -> Gate 2, fail -> Gate 3.
      2 ownership   : M_BB > gate2_factor * max_{i!=B} M_iB ?  grounded : entangled.
      3 shared sig  : M_BB <= 0 -> confabulated outright; else med_m X_m > 0 ?
                      seeded : confabulated. (M_BB <= 0 means erasing B does not
                      hurt B, so B is not read - 'seeded' needs a positive own-effect.)

    grounding_estimate = M_BB - med_m D*_m (spec: subtract the diagonal null,
    NOT med X_m, so the artifact term stays subtracted out).

    Fails closed: a non-finite M_BB, an empty/all-nan D* null, or (at Gate 3) an
    all-nan X set that the pre-checks somehow let through -> unresolved with a
    reason, never a silent grounded/confabulated."""
    d = np.asarray([x for x in dstars if x is not None], dtype=np.float64)
    d = d[np.isfinite(d)]
    med_dstar = float(np.median(d)) if d.size else float("nan")
    q95_dstar = float(np.percentile(d, 95)) if d.size else float("nan")
    grounding = float(m_bb - med_dstar) if (np.isfinite(m_bb) and d.size) else None

    base = dict(verdict=None, gate=None, reason="",
                m_bb=float(m_bb) if np.isfinite(m_bb) else None,
                q95_dstar=q95_dstar if d.size else None,
                med_dstar=med_dstar if d.size else None,
                kappa_0=float(kappa_0),
                gate1_bar=(q95_dstar + kappa_0) if d.size else None,
                m_ib_max=float(m_ib_max) if (m_ib_max is not None
                                             and np.isfinite(m_ib_max)) else None,
                grounding_estimate=grounding)

    if not power.get("passed"):
        # Tier-1 (2026-07-22): a power-fail whose decoy family is nonetheless
        # VALID (>= min_k clean placebos) and whose own-effect is non-positive
        # (M_BB <= 0) is confabulated, not unresolved. The valid family is the
        # MDE sensitivity floor - it gave the measurement the power to have
        # detected a real effect - and M_BB <= 0 means erasing B did not hurt B.
        # This emits ONLY confabulated (never grounded/entangled/seeded), so it
        # cannot manufacture a false positive. A power-fail with no valid family,
        # or with M_BB > 0 (the ambiguous middle: h raises B but not above its
        # own topical decoys), stays unresolved - never-guess.
        if validity.get("passed") and np.isfinite(m_bb) and m_bb <= 0.0 and d.size:
            return {**base, "gate": 3, "verdict": "confabulated",
                    "reason": "power-fail non-positive own-effect (valid decoy family)"}
        return {**base, "verdict": "unresolved", "reason": "power"}
    if not validity.get("passed"):
        return {**base, "verdict": "unresolved", "reason": "validity"}
    if not np.isfinite(m_bb) or d.size == 0:
        return {**base, "verdict": "unresolved",
                "reason": "degenerate readings (M_BB or D* non-finite)"}

    # Gate 1: specificity beyond the decoy null. TWO conditions:
    #   (a) M_BB > 0 - erasing B must actually HURT B. The spec prose (Step 7
    #       Gate 1) frames this as "does erasing B hurt B by more than erasing
    #       siblings hurts them"; "hurt" is directional. The bare relative test
    #       m_bb > Q95(D*)+kappa can be met by a NEGATIVE m_bb when the decoy
    #       null is even more negative (a column inversion, e.g. the anti-self
    #       artifact dominating a degenerate direction), which would license a
    #       grounded/entangled verdict off an effect that RAISED B's logp. The
    #       absolute floor resolves that spec ambiguity in favour of the prose.
    #   (b) M_BB > Q95(D*) + kappa_0 - specific beyond equally-plausible siblings.
    # Failing EITHER routes to Gate 3 (the shared-signal branch), where a
    # non-positive own-effect (M_BB <= 0) is confabulated OUTRIGHT and only a
    # positive own-effect that fails the null can still be seeded (see Gate 3).
    if m_bb > 0.0 and m_bb > q95_dstar + kappa_0:
        # Gate 2: ownership. max over real neighbors i != B; no neighbors
        # (single-claim activation) -> nothing to share with -> grounded.
        if m_ib_max is None or not np.isfinite(m_ib_max):
            return {**base, "gate": 2, "verdict": "grounded",
                    "reason": "no real neighbors to contest ownership"}
        if m_bb > gate2_factor * m_ib_max:
            return {**base, "gate": 2, "verdict": "grounded"}
        return {**base, "gate": 2, "verdict": "entangled"}

    # Gate 3: shared-signal branch. A non-positive own-effect (M_BB <= 0) means
    # erasing B did NOT hurt B, so B is not read from h - confabulated outright,
    # regardless of whether decoy directions move B. Reached only after the
    # power/validity pre-checks pass, so a broken decoy control stays unresolved
    # above; this rule never forces an unresolved claim to confabulated.
    # (M_BB is finite here: the degenerate-readings guard returned already.)
    if m_bb <= 0.0:
        return {**base, "gate": 3, "verdict": "confabulated",
                "reason": "non-positive own-effect (M_BB <= 0)"}
    # M_BB > 0 but failed the null: med X_m decides seeded vs confabulated.
    xf = np.asarray([x for x in x_ms if x is not None], dtype=np.float64)
    xf = xf[np.isfinite(xf)]
    if xf.size == 0:
        return {**base, "gate": 3, "verdict": "unresolved",
                "reason": "no finite X_m at Gate 3"}
    med_x = float(np.median(xf))
    base["med_x"] = med_x
    return {**base, "gate": 3,
            "verdict": "seeded" if med_x > 0 else "confabulated"}


# ── Step 4/6: disjoint token spans (cycle-mode scoring convention) ───────────

def locate_z_inner(full_text: str, z_inner: str, cont_char_start: int) -> int:
    """Char offset of z_inner inside the teacher-forced text (atom_spans' exact
    location logic: search from the continuation start, then anywhere). -1 when
    unlocatable - callers log the claim as unresolved, never guess."""
    z0 = full_text.find(z_inner, cont_char_start)
    if z0 < 0:
        z0 = full_text.find(z_inner)
    return z0


def first_char_ranges(offsets, char_spans):
    """Token ranges by FIRST-CHARACTER assignment: token ti belongs to span j
    iff its first char offset cs falls in [cstart_j, cend_j).

    atom_spans (claim_ablation) assigns by OVERLAP, so a token straddling the
    boundary between two character-adjacent claims lands in BOTH ranges - on
    the qwen7b-cp6 corpus 674 boundaries straddle a real token, touching
    254/400 records (audit 2026-07-21). Spec Step 4 sums "its own tokens only";
    first-char assignment restores that: each token has one first character, so
    character-disjoint spans give token-disjoint ranges. Published atom-run
    artifacts keep the overlap convention untouched - this is cycle-mode only,
    and the S3 smoke reproduces the stored atom M_BB with the OLD convention
    before switching to this one for verdicts.

    char_spans: absolute full-text [cstart, cend) pairs (already translated by
    locate_z_inner), or None entries to skip. Returns [(lo, hi) | None] aligned
    to char_spans; None when no token STARTS inside the span (e.g. a short
    claim swallowed by a neighbor-starting token - logged, not guessed).
    Zero-width/special tokens (ce <= cs) are never assigned."""
    out = []
    for span in char_spans:
        if span is None:
            out.append(None)
            continue
        cstart, cend = span
        lo = hi = None
        for ti, (cs, ce) in enumerate(offsets):
            if ce <= cs:                        # special / zero-width token
                continue
            if cstart <= cs < cend:             # token STARTS inside the span
                lo = ti if lo is None else lo
                hi = ti + 1
        out.append((lo, hi) if lo is not None else None)
    return out


# ── Step 5b: splice + same-subtrahend decoy directions ───────────────────────

def splice(z_inner: str, char_span, replacement: str):
    """z*_m = (z\\B) union B*_m in B's EXACT slot, PRESERVING the slot's own
    leading/trailing whitespace: z[:s] + lead_ws + B*_m + trail_ws + z[e:].

    38/75 qwen7b calibration slots carry edge whitespace inside char_span (atom
    segmentation convention) while candidate hygiene collapses/strips decoy
    text, so a bare z[:s]+rep+z[e:] would glue words ('...\"STEF\"carries...')
    in half the corpus - a seam artifact that either starves filter 4a or, if
    it slips through, one-sidedly depresses the decoy's filter-3 prior (audit
    2026-07-21). Re-attaching the ORIGINAL slot's whitespace makes z*_m differ
    from z only in claim content, which is the whole point of Step 5b.

    Returns (z_star, star_span). star_span is the FULL slot [s, s+len(lead)+
    len(rep)+len(trail)) INCLUDING the whitespace - the exact analogue of B's
    char_span in z (which also includes its edge whitespace), so filter 3 and
    the D* readings build both token spans by one convention.

    The deletion side (z\\B) is NOT built here - decoy_directions builds it with
    the atom-mode convention (z_inner[:s] + z_inner[e:]).strip() so the shared
    subtrahend AR(z\\B) is byte-identical to the published atom runs'."""
    s0, e0 = char_span
    assert 0 <= s0 < e0 <= len(z_inner), f"bad span {char_span} in len {len(z_inner)}"
    orig = z_inner[s0:e0]
    assert orig.strip(), "all-whitespace claim slot"
    lead = orig[:len(orig) - len(orig.lstrip())]
    trail = orig[len(orig.rstrip()):]
    rep = replacement.strip()
    assert rep, "empty decoy text"
    slot = lead + rep + trail
    return z_inner[:s0] + slot + z_inner[e0:], (s0, s0 + len(slot))


def unit_overlap(h, delta) -> float:
    """h . delta_hat - filter 2's 'amount removed' by ablating this direction.
    Zero/degenerate direction -> nan (an unusable candidate, never a silent 0)."""
    d = np.asarray(delta, np.float32)
    dn = float(np.linalg.norm(d))
    if not np.isfinite(dn) or dn < _EPS:
        return float("nan")
    return float(np.asarray(h, np.float32) @ (d / dn))


def _mse_cos(pred: torch.Tensor, gold: torch.Tensor, mse_scale: float) -> tuple[float, float]:
    """NLACritic.score's exact math on an ALREADY-COMPUTED reconstruction.
    score() re-runs the AR forward on the same text internally, which doubled
    the AR cost of decoy_directions for zero information (audit 2026-07-21):
    both scalars are pure arithmetic on the pred vector in hand."""
    pred_n = pred / pred.norm().clamp_min(1e-12) * mse_scale
    gold_n = gold / gold.norm().clamp_min(1e-12) * mse_scale
    mse = ((pred_n - gold_n) ** 2).mean().item()
    cos = (pred_n @ gold_n / (pred_n.norm() * gold_n.norm())).item()
    return float(mse), float(cos)


def decoy_directions(critic, z_inner: str, char_span, candidates, h) -> dict:
    """Spec Step 5b: every direction is computed against the SAME SUBTRAHEND,

        Delta_B     = norm(AR(z))     - norm(AR(z\\B))
        Delta_B*_m  = norm(AR(z*_m))  - norm(AR(z\\B))

    so B's direction and every decoy's differ only in which claim was added, and
    whatever non-local disturbance the deletion causes cancels in Gate 1.
    Delta_B here is numerically the atom run's deltas[B] (same deletion string,
    same normalization), which is what makes cycle M_BB comparable to the
    published atom dlogp.

    Returns dict:
      delta_b [d]      raw direction of the real claim
      ov_b             h . Delta_hat_B          (filter 2 reference)
      cos_full, mse_full                       AR(z) vs h (filter 4b reference)
      z_stars [g]      spliced explanations    star_spans [g] their decoy spans
      deltas  [g,d]    raw decoy directions
      ov      [g]      h . Delta_hat_B*_m       (filter 2 candidate values)
      cos_star[g], mse_star[g]                 AR(z*_m) vs h (filter 4b values)
      ar_full [d], ar_minus [d]                raw AR outputs, for the npz
    """
    s = critic.mse_scale
    s0, e0 = char_span
    z_minus = (z_inner[:s0] + z_inner[e0:]).strip()   # EXACT atom-mode deletion
    gold_t = torch.as_tensor(np.asarray(h, dtype=np.float32))
    ar_full = critic.reconstruct(z_inner).float()
    mse_full, cos_full = _mse_cos(ar_full, gold_t, s)
    if z_minus:
        ar_minus = critic.reconstruct(z_minus).float()
    else:                                             # atom-mode no-op fallback
        ar_minus = ar_full.clone()
    nf = normalize_activation(ar_full.view(1, -1), s)[0]
    nm = normalize_activation(ar_minus.view(1, -1), s)[0]
    delta_b = (nf - nm).numpy().astype(np.float32)

    g, d = len(candidates), ar_full.numel()
    deltas = np.zeros((g, d), np.float32)
    ov = np.full(g, np.nan, np.float32)
    cos_star = np.full(g, np.nan, np.float32)
    mse_star = np.full(g, np.nan, np.float32)
    z_stars, star_spans = [], []
    for m, cand in enumerate(candidates):
        z_star, sp = splice(z_inner, char_span, cand)
        z_stars.append(z_star)
        star_spans.append(sp)
        ar_star = critic.reconstruct(z_star).float()
        mse_m, cos_m = _mse_cos(ar_star, gold_t, s)
        ns = normalize_activation(ar_star.view(1, -1), s)[0]
        deltas[m] = (ns - nm).numpy()
        ov[m] = unit_overlap(h, deltas[m])
        cos_star[m], mse_star[m] = float(cos_m), float(mse_m)
    return dict(delta_b=delta_b, ov_b=unit_overlap(h, delta_b),
                cos_full=float(cos_full), mse_full=float(mse_full),
                z_stars=z_stars, star_spans=star_spans,
                deltas=deltas, ov=ov, cos_star=cos_star, mse_star=mse_star,
                ar_full=ar_full.numpy().astype(np.float32),
                ar_minus=ar_minus.numpy().astype(np.float32))


# ── Step 5b filters 2/3/4b (mechanical; 1 and 4a are decoy_gen.py LLM judges) ─

def filter2_matched_mass(ov_b: float, ov_stars, eps_ov: float) -> list[bool]:
    """Filter 2: |h.Dhat_B* - h.Dhat_B| <= eps_ov * |h.Dhat_B|. The decoy must
    remove about as much of h as B's own ablation does. NaN overlap fails."""
    ref = abs(float(ov_b))
    return [bool(np.isfinite(o) and abs(float(o) - float(ov_b)) <= eps_ov * ref)
            for o in ov_stars]


# signed-package ladder (S2 sign-off 2026-07-21): spec-literal per-claim
# widening. Extends the original calibration ladder past 0.5 so genuinely
# matchable claims resolve at a finite, logged eps; the unbounded_final
# fallthrough below handles the rest. float("inf") is deliberately NOT a rung
# (not JSON-serializable in strict mode; inf * 0 = nan when ov_b == 0).
EPS_OV_LADDER_SIGNED = (0.10, 0.15, 0.20, 0.30, 0.50, 0.75,
                        1.0, 1.5, 2.0, 3.0, 5.0, 10.0)


def widen_eps_until_k(ov_b: float, ov_stars, k: int, ladder,
                      unbounded_final: bool = False) -> dict:
    """Spec: 'widen the band until at least k decoys survive per claim'.
    Walk the ascending eps ladder; stop at the first eps admitting >= k.

    unbounded_final=False (legacy / d3g12 calibration): if even the last rung
    admits fewer than k, return that rung with capped=True - the caller logs
    the shortfall and the claim becomes unresolved downstream.

    unbounded_final=True (signed package, S2 sign-off 2026-07-21): the spec
    puts NO cap on the widening, so after the finite ladder fails on a pool
    that HAS finite-overlap candidates, the band is DISABLED entirely - every
    finite-overlap candidate passes filter 2, with eps_used=None and
    band_disabled=True so downstream artifacts can report exactly which claims
    needed the spec-literal widening taken to its limit. capped=True then means
    even the disabled band admits fewer than k (the pool has 1..k-1 usable
    candidates). A pool with ZERO finite-overlap candidates is NOT band_disabled
    (there is no band to disable): it returns band_disabled=False, capped=True,
    so it buckets with the no-usable-pool claims rather than inflating the
    band-disabled headline.

    A non-finite ov_b has no defined band at any eps (including disabled - the
    matched-mass quantity does not exist), so it always returns all-False."""
    ladder = [float(x) for x in ladder]
    assert ladder == sorted(ladder) and ladder, "eps ladder must be ascending"
    ov_stars = list(ov_stars)
    if not np.isfinite(ov_b):
        return dict(eps_used=None if unbounded_final else ladder[-1],
                    mask=[False] * len(ov_stars),
                    n_pass=0, capped=True, band_disabled=False)
    for eps in ladder:
        mask = filter2_matched_mass(ov_b, ov_stars, eps)
        if sum(mask) >= k:
            return dict(eps_used=eps, mask=mask, n_pass=int(sum(mask)),
                        capped=False, band_disabled=False)
    if unbounded_final:
        mask = [bool(np.isfinite(o)) for o in ov_stars]
        n = int(sum(mask))
        # only a pool with real candidates counts as "band disabled"; an empty
        # (or all-nan) pool has nothing to widen over -> no-pool, not disabled
        return dict(eps_used=None, mask=mask, n_pass=n,
                    capped=bool(n < k), band_disabled=bool(n >= 1))
    mask = filter2_matched_mass(ov_b, ov_stars, ladder[-1])
    return dict(eps_used=ladder[-1], mask=mask, n_pass=int(sum(mask)),
                capped=True, band_disabled=False)


def filter3_matched_prior(mean_b: float, mean_stars, eps_pr: float) -> list[bool]:
    """Filter 3: |meanlogp(B*|hbar) - meanlogp(B|hbar)| <= eps_pr, per-token means
    from span_mean_logp, each span in its OWN slot (B in z, B* in z*). NaN fails."""
    if not np.isfinite(mean_b):
        return [False] * len(list(mean_stars))
    return [bool(np.isfinite(m) and abs(float(m) - float(mean_b)) <= eps_pr)
            for m in mean_stars]


def filter4_fve_band(fve_z: float, fve_stars, tol: float) -> list[bool]:
    """Filter 4b: |FVE(z*) - FVE(z)| <= tol. FVEs come from fve_from_cos over the
    filter-2 AR reconstructions - no extra GPU forward. NaN fails."""
    if not np.isfinite(fve_z):
        return [False] * len(list(fve_stars))
    return [bool(np.isfinite(f) and abs(float(f) - float(fve_z)) <= tol)
            for f in fve_stars]
