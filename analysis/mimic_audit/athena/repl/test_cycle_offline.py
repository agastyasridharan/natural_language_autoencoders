"""Offline (CPU-only, no GPU, no API, no model download) unit tests for the
cycle-consistency core.

S1: corpus_hbar / unit_hbar_di / span_supports / span_mean_logp /
precheck_power / precheck_validity.
S2: splice / unit_overlap / decoy_directions (same-subtrahend identity, atom
deletion convention, via a FakeCritic) / filters 2-3-4b / widen_eps_until_k /
decoy_gen pure helpers (calib_claims keyed sampling, clean_candidates hygiene,
taxo_label join, select_work). The LLM calls themselves are NOT tested here -
their prompts are reviewed via decoy_gen.py --dry-run before any API spend.

The point of the harness here: drive the REAL measurement path
(_set_v -> _token_logp -> _span_sums via _inject_span_logp, plus the real
LayerZeroInjector object) with a FakeModel whose logits depend deterministically
on the injected vector. That validates the token-index convention
(tok_logp[a-1] = logp of the token at absolute index a) and the supp() sign and
scale end to end, without loading a transformer.

FakeModel: logits[t] = base + W @ vhat at EVERY position t, where vhat is the
unit-normalized injected vector. Same distribution at all positions, so the
per-token logp of a vocab id is position-independent -> supp of a span of n
identical tokens is exactly n times the single-token supp (linearity check),
and every expected value can be computed by hand with log_softmax.

Run:  python3 test_cycle_offline.py     (exits nonzero on any failure)
"""
from __future__ import annotations

import json
import pathlib
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

_REPL = Path(__file__).resolve().parent
if str(_REPL) not in sys.path:
    sys.path.insert(0, str(_REPL))

from claim_ablation import LayerZeroInjector, _set_v          # noqa: E402
import cycle_consistency as cc                                  # noqa: E402


def hr(title):
    print(f"\n── {title} " + "─" * max(0, 60 - len(title)))


# ── fake model: deterministic logits from the injected vector ────────────────

V, D, INJ_SCALE = 11, 8, 150.0


class FakeModel:
    """logits[0, t] = base + W @ vhat for all t; reads injector.v directly (the
    hook is never registered - _set_v only assigns injector.v, which is the one
    contract this fake needs)."""

    def __init__(self, injector, W, base):
        self.injector, self.W, self.base = injector, W, base
        self._emb = torch.nn.Embedding(1, 1)      # device probe for _token_logp

    def get_input_embeddings(self):
        return self._emb

    def __call__(self, input_ids=None, use_cache=False):
        assert use_cache is False
        v = self.injector.v.float().view(-1)
        vhat = v / v.norm().clamp_min(1e-12)
        bias = self.W @ vhat                                   # [V]
        T = input_ids.shape[1]
        logits = (self.base.unsqueeze(0) + bias.unsqueeze(0)).expand(T, -1)
        return SimpleNamespace(logits=logits.unsqueeze(0).clone())


def manual_span_sum(model, ids_t, lo, hi):
    """Independent reimplementation of the teacher-forced span score: the logp of
    the token at absolute index a is log_softmax(logits[a-1])[ids[a]]."""
    logits = model(input_ids=ids_t, use_cache=False).logits[0].float()
    lsm = torch.log_softmax(logits[:-1], dim=-1)
    return sum(float(lsm[a - 1, int(ids_t[0, a])]) for a in range(lo, hi))


def build():
    inj = LayerZeroInjector(marker_pos=3)          # real class; hook not registered
    W = torch.zeros(V, D)
    W[5, 0] = 1.0                                  # token 5 is boosted along e0
    base = torch.zeros(V)
    model = FakeModel(inj, W, base)
    # h points along e0 (token-5 direction), hbar orthogonal along e1. Arbitrary
    # raw norms - _set_v rescales both to INJ_SCALE, mirroring the real path.
    h = np.zeros(D, np.float32); h[0] = 42.0
    hbar = np.zeros(D, np.float32); hbar[1] = 7.0
    ids = torch.tensor([[0, 5, 5, 5, 7, 7, 7, 5, 2, 9]], dtype=torch.long)
    ranges = [(1, 4), (4, 7), (7, 8), None]        # A=three 5s, B=three 7s, D=one 5
    return model, inj, h, hbar, ids, ranges


def main():
    fails = 0

    hr("_set_v rescales to injection_scale")
    model, inj, h, hbar, ids, ranges = build()
    _set_v(inj, h, INJ_SCALE)
    n = float(inj.v.float().norm())
    print(f"||iota(h)|| = {n:.6f}  (target {INJ_SCALE})")
    assert abs(n - INJ_SCALE) < 1e-3
    assert float(inj.v[0]) > 0 and abs(float(inj.v[1])) < 1e-6, "direction preserved"

    hr("span_supports: exactness through the real path")
    out = cc.span_supports(model, inj, ids, h, hbar, INJ_SCALE, ranges)
    _set_v(inj, h, INJ_SCALE)
    exp_h = [manual_span_sum(model, ids, lo, hi) for lo, hi in ranges[:3]]
    _set_v(inj, hbar, INJ_SCALE)
    exp_b = [manual_span_sum(model, ids, lo, hi) for lo, hi in ranges[:3]]
    for i in range(3):
        print(f"span {i}: l0 {out['l0'][i]:+.6f} vs manual {exp_h[i]:+.6f} | "
              f"lbar {out['lbar'][i]:+.6f} vs manual {exp_b[i]:+.6f}")
        assert abs(out["l0"][i] - exp_h[i]) < 1e-5
        assert abs(out["lbar"][i] - exp_b[i]) < 1e-5
    assert np.isnan(out["l0"][3]) and np.isnan(out["supp"][3]), "None range -> nan"
    assert out["tlp_h"].shape == (ids.shape[1] - 1,)

    hr("supp sign and scale")
    sA, sB, sD = out["supp"][0], out["supp"][1], out["supp"][2]
    print(f"supp A(3x tok5, encoded)  = {sA:+.6f}   expect > 0")
    print(f"supp B(3x tok7, neutral)  = {sB:+.6f}   expect <= 0, small (softmax)")
    print(f"supp D(1x tok5)           = {sD:+.6f}   expect exactly A/3")
    assert sA > 0.1, "encoded span must gain clearly under h"
    assert sB <= 0 and abs(sB) < sA, "neutral span moves only via renormalization"
    assert abs(sA - 3 * sD) < 1e-5, "identical-token spans -> supp linear in length"
    # under hbar the distribution is uniform: mean logp = log(1/V)
    assert abs(out["mean_lbar"][0] - float(np.log(1 / V))) < 1e-5

    hr("span_supports identity: h == hbar -> supp exactly 0")
    out0 = cc.span_supports(model, inj, ids, h, h, INJ_SCALE, ranges)
    print("supp:", [f"{s:+.1f}" if np.isfinite(s) else "nan" for s in out0["supp"]])
    assert all(s == 0.0 for s in out0["supp"][:3])

    hr("corpus_hbar")
    gold = np.array([[1, 2], [3, 4], [5, 0]], np.float64)
    hb = cc.corpus_hbar(gold)
    print("hbar:", hb, hb.dtype)
    assert hb.dtype == np.float32 and np.allclose(hb, [3.0, 2.0])
    try:
        cc.corpus_hbar(np.zeros((0, 4)))
        fails += 1; print("FAIL: empty gold accepted")
    except AssertionError:
        print("empty gold rejected: OK")

    hr("unit_hbar_di matches the fve.py formula")
    rng = np.random.default_rng(0)
    G = rng.normal(size=(7, 5)) + 3.0              # offset -> nontrivial mean share
    hu, di = cc.unit_hbar_di(G)
    Ghat = G / np.linalg.norm(G, axis=1, keepdims=True)
    hu_ref = Ghat.mean(0)
    di_ref = ((Ghat - hu_ref) ** 2).sum(1)
    print(f"max |hbar_u - ref| = {np.abs(hu - hu_ref).max():.2e}, "
          f"max |D_i - ref| = {np.abs(di - di_ref).max():.2e}")
    assert np.allclose(hu, hu_ref) and np.allclose(di, di_ref)
    assert cc.fve_from_cos(1.0, float(di[0])) == 1.0
    assert np.isnan(cc.fve_from_cos(float("nan"), 1.0))
    assert np.isnan(cc.fve_from_cos(0.5, 0.0))

    hr("span_mean_logp")
    m = cc.span_mean_logp([-6.0, float("nan"), -2.0], [(1, 4), (4, 7), None])
    print("means:", m)
    assert abs(m[0] - (-2.0)) < 1e-12 and np.isnan(m[1]) and np.isnan(m[2])

    hr("precheck_power")
    assert cc.precheck_power(2.0, 0.5)["passed"]
    assert cc.precheck_power(2.0, 0.5)["margin"] == 1.5
    assert not cc.precheck_power(0.3, 0.5)["passed"]
    assert not cc.precheck_power(float("nan"), 0.5)["passed"]
    print("pass / fail / nan branches: OK")

    hr("precheck_validity")
    ok = cc.precheck_validity([0.1, -0.2, 0.05], tau_val=0.5, min_k=3)
    print("clean set:", ok["passed"], ok["supp_max_abs"])
    assert ok["passed"] and ok["supp_viol_idx"] == []
    enc = cc.precheck_validity([0.1, 3.0, -0.05], tau_val=0.5, min_k=3)
    print("encoded decoy:", enc["passed"], enc["supp_viol_idx"], enc["reason"])
    assert not enc["passed"] and enc["supp_viol_idx"] == [1]  # 2 clean < min_k=3
    # Tier-1 (2026-07-22): an encoded decoy in a big-enough family is trimmed
    # from the clean COUNT and the family passes on its >= min_k clean placebos.
    enc_ok = cc.precheck_validity([0.1, -0.2, 0.05, 3.0, 0.0, -0.1],
                                  tau_val=0.5, min_k=5)
    assert enc_ok["passed"] and enc_ok["supp_viol_idx"] == [3] and enc_ok["n_clean"] == 5
    nanv = cc.precheck_validity([0.1, float("nan")], tau_val=0.5, min_k=2)
    assert not nanv["passed"] and nanv["supp_viol_idx"] == [1]
    # D* outliers: high side flagged, low side NOT (only makes Gate 1 harder).
    # Tier-1 decouple: a high-side outlier is FLAGGED but the family PASSES on its
    # >= min_k clean placebos (the outlier stays in the Gate-1 null via classify).
    hi_out = cc.precheck_validity([0.0] * 6, tau_val=0.5,
                                  dstar=[1.0, 1.1, 0.9, 1.05, 0.95, 9.0])
    print("dstar high outlier (decoupled):", hi_out["passed"],
          hi_out["dstar_outlier_idx"], "n_clean", hi_out["n_clean"])
    assert hi_out["passed"] and hi_out["dstar_outlier_idx"] == [5] \
        and hi_out["n_clean"] == 5, "1 outlier of 6 -> 5 clean >= min_k -> PASS"
    # but if trimming drops the clean count below min_k, it fails closed
    hi5 = cc.precheck_validity([0.0] * 5, tau_val=0.5,
                               dstar=[1.0, 1.1, 0.9, 0.95, 9.0])
    assert not hi5["passed"] and hi5["dstar_outlier_idx"] == [4] and hi5["n_clean"] == 4
    lo_out = cc.precheck_validity([0.0] * 6, tau_val=0.5,
                                  dstar=[1.0, 1.1, 0.9, 1.05, 0.95, -9.0])
    assert lo_out["passed"], "low-side dstar outlier must not flag"
    nan_d = cc.precheck_validity([0.0, 0.0], tau_val=0.5, min_k=2,
                                 dstar=[1.0, float("nan")])
    assert not nan_d["passed"] and nan_d["dstar_outlier_idx"] == [1]
    empty = cc.precheck_validity([], tau_val=0.5)
    print("empty set:", empty["passed"], repr(empty["reason"]))
    assert not empty["passed"] and empty["reason"].startswith("insufficient decoys")
    # fail-closed below min_k (audit 2026-07-21: old fence was vacuous at n<=2
    # and self-masked at n=3-4 - a 50x contaminant escaped its own Q3+IQR)
    for n in (1, 2, 3, 4):
        small = cc.precheck_validity([0.0] * n, tau_val=0.5,
                                     dstar=[1.0] * (n - 1) + [50.0])
        assert not small["passed"] and "insufficient decoys" in small["reason"], n
    print("n=1..4 fail closed: OK")
    # MAD fence catches a single contaminant at exactly n=5 (Q3+IQR did not)
    cont5 = cc.precheck_validity([0.0] * 5, tau_val=0.5,
                                 dstar=[1.0, 0.9, 1.1, 0.95, 50.0])
    print(f"n=5 contaminant: fence={cont5['fence']:.3f}",
          cont5["dstar_outlier_idx"])
    assert not cont5["passed"] and cont5["dstar_outlier_idx"] == [4]
    ties = cc.precheck_validity([0.0] * 5, tau_val=0.5, dstar=[1.0] * 5)
    assert ties["passed"], "identical dstar values must not flag (MAD=0 + slack)"
    print("MAD fence + ties: OK")

    hr("classify_claim: five-way gated verdict (spec Step 7)")
    PASS = dict(passed=True)
    FAIL = dict(passed=False)
    dnull = [1.0, 0.9, 1.1, 0.95, 1.05]      # Q95 ~ 1.08, med 1.0
    # pre-check gates come FIRST: a power-fail with M_BB>0 (ambiguous middle) or a
    # validity-fail -> unresolved. (Tier-1 power-fail+valid+M_BB<=0 is tested below.)
    up = cc.classify_claim(FAIL, PASS, m_bb=9.0, dstars=dnull, x_ms=[1.0],
                           m_ib_max=0.0, kappa_0=0.5)
    assert up["verdict"] == "unresolved" and up["reason"] == "power"
    uv = cc.classify_claim(PASS, FAIL, m_bb=9.0, dstars=dnull, x_ms=[1.0],
                           m_ib_max=0.0, kappa_0=0.5)
    assert uv["verdict"] == "unresolved" and uv["reason"] == "validity"
    # Gate 1 pass + Gate 2 pass (M_BB dominates its column) -> grounded
    g = cc.classify_claim(PASS, PASS, m_bb=9.0, dstars=dnull, x_ms=[-1.0],
                          m_ib_max=2.0, kappa_0=0.5)
    print("grounded:", g["verdict"], "grounding_est", g["grounding_estimate"])
    assert g["verdict"] == "grounded" and g["gate"] == 2
    assert abs(g["grounding_estimate"] - (9.0 - 1.0)) < 1e-9  # M_BB - med D*
    assert abs(g["gate1_bar"] - (float(np.percentile(dnull, 95)) + 0.5)) < 1e-9
    # Gate 1 pass + Gate 2 fail (a neighbor shares the mass) -> entangled
    e = cc.classify_claim(PASS, PASS, m_bb=9.0, dstars=dnull, x_ms=[-1.0],
                          m_ib_max=8.0, kappa_0=0.5)   # 9 > 1.5*8=12 is False
    assert e["verdict"] == "entangled" and e["gate"] == 2
    # Gate 1 pass + no real neighbors -> grounded (nothing to contest ownership)
    gn = cc.classify_claim(PASS, PASS, m_bb=9.0, dstars=dnull, x_ms=[-1.0],
                           m_ib_max=None, kappa_0=0.5)
    assert gn["verdict"] == "grounded" and "no real neighbors" in gn["reason"]
    # Gate 1 FAIL (M_BB inside the decoy null) + med X_m > 0 -> seeded
    s = cc.classify_claim(PASS, PASS, m_bb=1.0, dstars=dnull, x_ms=[0.5, 0.7, 0.6],
                          m_ib_max=0.0, kappa_0=0.5)
    print("seeded:", s["verdict"], "med_x", s.get("med_x"))
    assert s["verdict"] == "seeded" and s["gate"] == 3 and s["med_x"] > 0
    # Gate 1 FAIL + med X_m <= 0 -> confabulated (theme itself did not support B)
    cf = cc.classify_claim(PASS, PASS, m_bb=1.0, dstars=dnull,
                           x_ms=[-0.5, 0.1, -0.3], m_ib_max=0.0, kappa_0=0.5)
    assert cf["verdict"] == "confabulated" and cf["gate"] == 3
    # med X exactly 0 is NOT > 0 -> confabulated (boundary)
    cf0 = cc.classify_claim(PASS, PASS, m_bb=1.0, dstars=dnull, x_ms=[0.0],
                            m_ib_max=0.0, kappa_0=0.5)
    assert cf0["verdict"] == "confabulated"
    # fail-closed: non-finite M_BB, empty D* null, all-nan X at Gate 3
    assert cc.classify_claim(PASS, PASS, float("nan"), dnull, [1.0], 0.0,
                             0.5)["verdict"] == "unresolved"
    assert cc.classify_claim(PASS, PASS, 9.0, [], [1.0], 0.0,
                             0.5)["verdict"] == "unresolved"
    xnan = cc.classify_claim(PASS, PASS, m_bb=1.0, dstars=dnull,
                             x_ms=[float("nan")], m_ib_max=0.0, kappa_0=0.5)
    assert xnan["verdict"] == "unresolved" and "no finite X_m" in xnan["reason"]
    # kappa_0 raises the Gate-1 bar: same M_BB, bigger kappa can flip pass->fail
    near = 1.10                              # just above Q95(dnull)=~1.08
    assert cc.classify_claim(PASS, PASS, near, dnull, [-1.0], 0.0,
                             kappa_0=0.0)["gate"] == 2      # passes Gate 1
    assert cc.classify_claim(PASS, PASS, near, dnull, [0.5], 0.0,
                             kappa_0=5.0)["gate"] == 3      # kappa pushes to G3
    # Tier-0 (2026-07-23): raising kappa_0 only DEMOTES grounded/entangled to
    # Gate 3 (seeded/confabulated); it is downstream of the pre-checks (unresolved
    # is invariant) and never crosses Gate 2's kappa-independent threshold.
    g_lo = cc.classify_claim(PASS, PASS, 1.20, dnull, [-1.0], 0.0, kappa_0=0.0)
    assert g_lo["verdict"] == "grounded"                   # low kappa -> grounded
    g_hi = cc.classify_claim(PASS, PASS, 1.20, dnull, [-1.0], 0.0, kappa_0=5.0)
    assert g_hi["verdict"] in ("seeded", "confabulated") and g_hi["gate"] == 3
    # a pre-check fail is unresolved at ANY kappa (kappa cannot resolve/undo it)
    assert cc.classify_claim(FAIL, PASS, 9.0, dnull, [1.0], 0.0, 0.0)["verdict"] == "unresolved"
    assert cc.classify_claim(FAIL, PASS, 9.0, dnull, [1.0], 0.0, 9.0)["verdict"] == "unresolved"
    # Gate-1 ABSOLUTE FLOOR (audit 2026-07-21): M_BB <= 0 can NEVER be
    # grounded/entangled, even when the decoy null is more negative and the
    # relative test m_bb > Q95(D*)+kappa is met. It routes to Gate 3 instead.
    negd = [-4.6, -5.0, -6.1, -9.9, -4.7]    # 352.8-like: null all-negative
    fl = cc.classify_claim(PASS, PASS, m_bb=-3.19, dstars=negd, x_ms=[-0.49],
                           m_ib_max=22.5, kappa_0=0.8)
    print("neg-M_BB floor:", fl["verdict"], "gate", fl["gate"])
    assert fl["verdict"] == "confabulated" and fl["gate"] == 3, \
        "M_BB<0 must fail Gate 1 (absolute floor) -> Gate 3, not entangled"
    # CORRECTION (2026-07-22): the SAME negative M_BB with med X_m > 0 is now
    # CONFABULATED, not seeded. A non-positive own-effect means erasing B did
    # not hurt B, so B is not read; 'seeded' requires a POSITIVE own-effect that
    # merely fails the null.
    fl2 = cc.classify_claim(PASS, PASS, m_bb=-3.19, dstars=negd, x_ms=[0.4],
                            m_ib_max=22.5, kappa_0=0.8)
    assert fl2["verdict"] == "confabulated" and fl2["gate"] == 3 \
        and "non-positive own-effect" in fl2["reason"], \
        "M_BB<=0 with med X_m>0 must be confabulated, not seeded"
    # M_BB exactly 0 with med X_m > 0 -> confabulated too (0 is not > 0)
    z0 = cc.classify_claim(PASS, PASS, 0.0, negd, [0.5, 0.6], m_ib_max=0.0, kappa_0=0.0)
    assert z0["verdict"] == "confabulated" and z0["gate"] == 3
    # SCOPE: the Gate-3 M_BB<=0 rule (claims that PASS both pre-checks) is applied
    # after the pre-checks. A validity-FAIL control stays unresolved (no MDE floor).
    assert cc.classify_claim(PASS, FAIL, -3.19, negd, [0.4], 0.0, 0.8)["verdict"] == "unresolved"
    # Tier-1 (2026-07-22): a POWER-fail with a VALID family and M_BB<=0 is now
    # CONFABULATED (the valid decoy family is the MDE floor: it CAN detect an
    # effect this size, and the own-effect is non-positive). Only a power-fail
    # WITHOUT a valid family (or with M_BB>0, the ambiguous middle) stays unresolved.
    pf = cc.classify_claim(FAIL, PASS, -3.19, negd, [0.4], 0.0, 0.8)
    assert pf["verdict"] == "confabulated" and pf["gate"] == 3 \
        and "power-fail non-positive" in pf["reason"], \
        "power-fail + valid family + M_BB<=0 must resolve to confabulated (Tier-1)"
    # M_BB exactly 0 with a valid family + power-fail -> confabulated too
    assert cc.classify_claim(FAIL, PASS, 0.0, negd, [0.4], 0.0, 0.8)["verdict"] == "confabulated"
    # power-fail + INVALID family + M_BB<=0 -> unresolved (no MDE floor)
    assert cc.classify_claim(FAIL, FAIL, -3.19, negd, [0.4], 0.0, 0.8)["verdict"] == "unresolved"
    # power-fail + valid family + M_BB>0 (ambiguous middle) -> unresolved (never-guess)
    assert cc.classify_claim(FAIL, PASS, 9.0, negd, [0.4], 0.0, 0.8)["verdict"] == "unresolved"
    # power-fail + valid family + M_BB<=0 but EMPTY null -> unresolved (nothing to gate)
    assert cc.classify_claim(FAIL, PASS, -3.19, [], [0.4], 0.0, 0.8)["verdict"] == "unresolved"
    # a genuinely positive M_BB that fails the null but has med X_m>0 is STILL seeded
    assert cc.classify_claim(PASS, PASS, 1.0, dnull, [0.5, 0.7], 0.0, 0.5)["verdict"] == "seeded"
    # a genuinely positive M_BB clearing the null still grounds normally
    assert cc.classify_claim(PASS, PASS, 9.0, dnull, [-1.0], 2.0,
                             kappa_0=0.5)["verdict"] == "grounded"
    print("power/validity/grounded/entangled/seeded/confabulated/fail-closed/"
          "kappa-sensitivity/absolute-floor: OK")

    hr("first_char_ranges (disjoint token spans)")
    # two character-adjacent claims A=[0,10) B=[10,20); token (8,12) straddles.
    offs = [(0, 4), (4, 8), (8, 12), (12, 16), (16, 20), (20, 20)]
    fc = cc.first_char_ranges(offs, [(0, 10), (10, 20)])
    print("first-char:", fc)
    assert fc == [(0, 3), (3, 5)], fc
    assert fc[0][1] <= fc[1][0], "ranges must be disjoint"
    # the overlap convention (atom_spans rule) shares token 2 between both:
    ov_a = [ti for ti, (cs, ce) in enumerate(offs) if ce > cs and ce > 0 and cs < 10]
    ov_b = [ti for ti, (cs, ce) in enumerate(offs) if ce > cs and ce > 10 and cs < 20]
    assert 2 in ov_a and 2 in ov_b, "sanity: overlap rule double-counts token 2"
    # None span skipped; span with no starting token -> None; zero-width skipped
    fc2 = cc.first_char_ranges(offs, [None, (9, 10), (20, 24)])
    assert fc2 == [None, None, None], fc2
    print("straddler assigned to earlier claim only; degenerate spans -> None: OK")
    assert cc.locate_z_inner("chat header\n\nBODY text", "BODY", 10) == 13
    assert cc.locate_z_inner("BODY early\n\nother", "BODY", 12) == 0
    assert cc.locate_z_inner("no match", "BODY", 0) == -1

    # ── S2 ───────────────────────────────────────────────────────────────────

    hr("splice: exact slot")
    z = "First claim here.\n\nSecond one about dates.\n\nThird claim."
    span = (19, 41)                              # "Second one about dates"
    assert z[span[0]:span[1]] == "Second one about dates"
    z_star, sp = cc.splice(z, span, "  A replacement claim ")
    print(repr(z_star))
    assert z_star == "First claim here.\n\nA replacement claim.\n\nThird claim."
    assert z_star[sp[0]:sp[1]] == "A replacement claim"
    try:
        cc.splice(z, (5, 5), "x"); fails += 1; print("FAIL: empty span accepted")
    except AssertionError:
        print("empty span rejected: OK")
    try:
        cc.splice(z, span, "   "); fails += 1; print("FAIL: blank decoy accepted")
    except AssertionError:
        print("blank decoy rejected: OK")
    # slot-edge whitespace preserved (audit 2026-07-21: 38/75 calib slots carry
    # it; without this, z* glues words and every decoy shares the seam artifact)
    zw = "alpha beta gamma"
    spw = (5, 11)
    assert zw[spw[0]:spw[1]] == " beta "
    z_ws, sp_ws = cc.splice(zw, spw, "  delta ")
    print(repr(z_ws), sp_ws)
    assert z_ws == "alpha delta gamma", repr(z_ws)
    assert z_ws[sp_ws[0]:sp_ws[1]] == " delta ", "star_span must be the FULL slot"
    # leading-space slot glued to a closing quote (the real failure shape)
    zq = 'token "STEF" carries weight.'
    spq = (12, 20)                               # ' carries'
    assert zq[spq[0]:spq[1]] == " carries"
    z_q, sp_q = cc.splice(zq, spq, "resolves")
    assert z_q == 'token "STEF" resolves weight.', repr(z_q)
    assert z_q[sp_q[0]:sp_q[1]] == " resolves"

    hr("unit_overlap")
    hv = np.array([3.0, 4.0, 0.0], np.float32)
    assert abs(cc.unit_overlap(hv, [1, 0, 0]) - 3.0) < 1e-6
    assert abs(cc.unit_overlap(hv, [0, 2, 0]) - 4.0) < 1e-6   # scale-invariant
    assert np.isnan(cc.unit_overlap(hv, [0.0, 0.0, 0.0]))
    print("projection + scale-invariance + zero-direction nan: OK")

    hr("decoy_directions: same subtrahend, atom deletion convention")
    from nla_inference import normalize_activation

    class FakeCritic:
        """Deterministic text->vector. NO score() method: decoy_directions must
        compute mse/cos from the reconstruction in hand (_mse_cos) instead of
        re-running the AR forward per text (audit 2026-07-21)."""
        mse_scale = 8.0

        def __init__(self):
            self.calls = []

        def _vec(self, text):
            import hashlib as _h
            key = _h.sha256(text.encode()).digest()[:8]
            rng = np.random.default_rng(int.from_bytes(key, "big"))
            return torch.tensor(rng.normal(size=16), dtype=torch.float32)

        def reconstruct(self, text):
            self.calls.append(text)
            return self._vec(text)

    crit = FakeCritic()
    hh = np.random.default_rng(1).normal(size=16).astype(np.float32)
    cands = ["A replacement claim.", "Another different claim."]
    dd = cc.decoy_directions(crit, z, span, cands, hh)
    z_minus_expect = (z[:span[0]] + z[span[1]:]).strip()
    assert z_minus_expect in crit.calls, "z\\B must use the atom-mode .strip() deletion"
    assert crit.calls.count(z_minus_expect) == 1, "subtrahend reconstructed once"
    assert len(crit.calls) == 2 + len(cands), \
        f"exactly one AR forward per text (z, z\\B, each z*): got {len(crit.calls)}"
    # _mse_cos must replicate NLACritic.score's normalize-to-mse_scale math
    v = crit._vec(z); hg = torch.as_tensor(hh)
    cos_ref = float(v @ hg / (v.norm() * hg.norm()))
    mse_ref = float(2 * crit.mse_scale ** 2 * (1 - cos_ref) / v.numel())
    assert abs(dd["cos_full"] - cos_ref) < 1e-5, "cos must match score()'s formula"
    assert abs(dd["mse_full"] - mse_ref) < 1e-5, "mse must match score()'s formula"
    s = crit.mse_scale
    nf = normalize_activation(crit._vec(z).view(1, -1), s)[0].numpy()
    nm = normalize_activation(crit._vec(z_minus_expect).view(1, -1), s)[0].numpy()
    assert np.allclose(dd["delta_b"], nf - nm, atol=1e-6)
    for m, cand in enumerate(cands):
        zs, _ = cc.splice(z, span, cand)
        ns = normalize_activation(crit._vec(zs).view(1, -1), s)[0].numpy()
        assert np.allclose(dd["deltas"][m], ns - nm, atol=1e-6)
        # same-subtrahend identity: Delta_B - Delta_B* = nf - ns (nm cancels)
        assert np.allclose(dd["delta_b"] - dd["deltas"][m], nf - ns, atol=1e-6)
        assert dd["z_stars"][m] == zs
    assert np.isfinite(dd["ov_b"]) and np.isfinite(dd["ov"]).all()
    assert np.isfinite(dd["cos_star"]).all() and np.isfinite(dd["cos_full"])
    print(f"delta_b/deltas match manual; ov_b={dd['ov_b']:+.3f} "
          f"ov={np.round(dd['ov'], 3)}")

    hr("filter 2 + widen_eps_until_k")
    ovb = 2.0
    ovs = [2.1, 1.7, 2.9, -2.0, float("nan")]
    m01 = cc.filter2_matched_mass(ovb, ovs, 0.10)
    assert m01 == [True, False, False, False, False]
    m05 = cc.filter2_matched_mass(ovb, ovs, 0.50)
    assert m05 == [True, True, True, False, False]   # -2.0 is 2*ref away; nan fails
    w = cc.widen_eps_until_k(ovb, ovs, k=2, ladder=[0.10, 0.25, 0.50])
    print("widen k=2:", w)
    assert w["eps_used"] == 0.25 and w["n_pass"] == 2 and not w["capped"]
    w4 = cc.widen_eps_until_k(ovb, ovs, k=4, ladder=[0.10, 0.25, 0.50])
    print("widen k=4 (capped):", w4)
    assert w4["capped"] and w4["eps_used"] == 0.50 and w4["n_pass"] == 3
    wn = cc.widen_eps_until_k(float("nan"), ovs, k=2, ladder=[0.10])
    assert wn["capped"] and wn["n_pass"] == 0
    assert w["band_disabled"] is False, "legacy path never disables the band"

    hr("widen_eps_until_k: signed package (unbounded_final)")
    # a finite rung that reaches k still wins - fallthrough only after the
    # ladder. (0.20, not 0.15: |1.7-2.0| carries fp error 3e-17 ABOVE
    # 0.15*2.0, so the exact tie falls to the next rung - fine, the band is a
    # calibration knob, not a sharp boundary)
    ws = cc.widen_eps_until_k(ovb, ovs, k=2, ladder=cc.EPS_OV_LADDER_SIGNED,
                              unbounded_final=True)
    assert ws["eps_used"] == 0.20 and not ws["band_disabled"] and not ws["capped"]
    # k unreachable at any finite rung (nan candidate) -> band disabled: every
    # FINITE ov passes, eps_used=None, capped because the pool itself is < k
    wd = cc.widen_eps_until_k(ovb, ovs, k=5, ladder=cc.EPS_OV_LADDER_SIGNED,
                              unbounded_final=True)
    print("band-disabled (pool<k):", wd)
    assert wd["band_disabled"] and wd["eps_used"] is None
    assert wd["mask"] == [True, True, True, True, False] and wd["n_pass"] == 4
    assert wd["capped"], "4 finite ovs < k=5 -> no widening can help"
    # ov_b == 0 edge (why inf was rejected as a rung: eps*|0| = 0 admits
    # nothing at ANY finite eps; the disabled band must still resolve it)
    wz = cc.widen_eps_until_k(0.0, [5.0, -3.0, 7.0], k=2, ladder=[0.1],
                              unbounded_final=True)
    assert wz["band_disabled"] and wz["n_pass"] == 3 and not wz["capped"]
    assert wz["eps_used"] is None
    # non-finite ov_b: the matched-mass quantity does not exist - all-False,
    # and NOT reported as band-disabled (nothing was widened)
    wf = cc.widen_eps_until_k(float("nan"), ovs, k=2, ladder=[0.1],
                              unbounded_final=True)
    assert wf["n_pass"] == 0 and wf["capped"] and not wf["band_disabled"]
    assert wf["eps_used"] is None
    # finite ov_b but the pool is EMPTY (or all-nan): nothing to widen over, so
    # NOT band_disabled - it must bucket with no-pool, capped, 0 survivors
    we = cc.widen_eps_until_k(2.0, [], k=2, ladder=[0.1], unbounded_final=True)
    assert we["n_pass"] == 0 and we["capped"] and not we["band_disabled"]
    wan = cc.widen_eps_until_k(2.0, [float("nan"), float("nan")], k=2,
                               ladder=[0.1], unbounded_final=True)
    assert wan["n_pass"] == 0 and wan["capped"] and not wan["band_disabled"]
    print("finite-rung win, band-disable fallthrough, ov_b=0, nan ov_b, "
          "empty/all-nan pool: OK")

    hr("precheck_power_family (signed power rule)")
    fam = [1.0, 2.0, 3.0, 4.0, 100.0]
    q95 = float(np.percentile(np.asarray(fam), 95))
    pf = cc.precheck_power_family(150.0, fam)
    assert pf["passed"] and pf["n_family"] == 5
    assert abs(pf["q95_family"] - q95) < 1e-12
    assert abs(pf["margin"] - (150.0 - q95)) < 1e-12
    assert not cc.precheck_power_family(q95, fam)["passed"], "tie fails (strict >)"
    assert not cc.precheck_power_family(50.0, fam)["passed"]
    # fail-closed: empty family / nan member / nan supp_b all fail
    e1 = cc.precheck_power_family(150.0, [])
    assert not e1["passed"] and e1["n_family"] == 0 and e1["q95_family"] is None
    assert not cc.precheck_power_family(150.0, [1.0, float("nan")])["passed"]
    assert not cc.precheck_power_family(float("nan"), fam)["passed"]
    print("q95 matched null, strict inequality, fail-closed: OK")

    hr("filters 3 + 4b")
    f3 = cc.filter3_matched_prior(-2.0, [-2.3, -1.2, float("nan")], eps_pr=0.5)
    assert f3 == [True, False, False]
    assert cc.filter3_matched_prior(float("nan"), [-2.0], 0.5) == [False]
    f4 = cc.filter4_fve_band(0.70, [0.65, 0.40, float("nan")], tol=0.10)
    assert f4 == [True, False, False]
    assert cc.filter4_fve_band(float("nan"), [0.7], 0.1) == [False]
    print("band logic + nan handling: OK")

    hr("decoy_gen: calib_claims keyed sampling")
    import decoy_gen as dg
    mk = lambda i, t, loc=True: dict(located=loc, scored=True, type=t,
                                     text=f"claim {i}", char_span=[0, 5])
    rec_fake = dict(idx=7, doc_id="fineweb:7", snippet_results=[
        mk(0, "theme"), mk(1, "detail"), mk(2, "theme"), mk(3, "detail"),
        mk(4, "theme", loc=False), dict(located=True, scored=False, type="theme"),
        mk(6, "connective")])
    p1 = dg.calib_claims(rec_fake, per_act=3, seed=0)
    p2 = dg.calib_claims(rec_fake, per_act=3, seed=0)
    print("picked:", p1)
    assert p1 == p2, "keyed RNG must be deterministic"
    assert len(p1) == 3 and all(i in (0, 1, 2, 3) for i in p1), \
        "only located+scored content claims"
    types = [rec_fake["snippet_results"][i]["type"] for i in p1]
    assert "theme" in types and "detail" in types, "type-stratified"
    other_doc = dg.calib_claims(dict(rec_fake, doc_id="fineweb:8"), 3, 0)
    print("other doc picks:", other_doc, "(may differ)")

    hr("decoy_gen: clean_candidates hygiene")
    cl = dg.clean_candidates(
        ["  A fine   decoy. ", "a fine decoy.", "", "CJK 文字 here",
         "The Claim Itself", "Another good one."], "the claim itself")
    for c in cl:
        print(f"  kept={c['kept']!s:5} reason={c['reason']:13} {c['text'][:30]!r}")
    assert [c["kept"] for c in cl] == [True, False, False, False, False, True]
    assert cl[1]["reason"] == "duplicate" and cl[2]["reason"] == "empty"
    assert cl[3]["reason"] == "non-ascii" and cl[4]["reason"] == "equals claim"
    assert cl[0]["text"] == "A fine decoy."          # whitespace collapsed

    hr("decoy_gen: d4 hygiene - claim-mirrored non-ASCII allowed")
    assert dg.PROMPT_VERSION == "d4"
    # idx 32.15's killer under d3: the claim itself contains an em-dash, so a
    # blanket ASCII rule rejected every format-matching candidate
    emdash_claim = "— a celebrity gossip item"
    cl3 = dg.clean_candidates(
        ["— a political scandal story",      # mirrors the claim's em-dash
         "an ASCII-only alternative",             # plain ASCII always fine
         "CJK 文字 decoy here",           # foreign chars still rejected
         "curly ‘quote’ decoy"],        # chars NOT in the claim
        emdash_claim)
    for c in cl3:
        print(f"  kept={c['kept']!s:5} reason={c['reason']:13} {c['text'][:30]!r}")
    assert [c["kept"] for c in cl3] == [True, True, False, False]
    assert cl3[2]["reason"] == "non-ascii" and cl3[3]["reason"] == "non-ascii"
    # regression: an all-ASCII claim still rejects every non-ASCII candidate
    cl4 = dg.clean_candidates(["— has an em-dash", "plain fine one"],
                              "plain ascii claim")
    assert [c["kept"] for c in cl4] == [False, True]
    assert cl4[0]["reason"] == "non-ascii"
    print("claim's own chars mirrored; foreign chars still rejected: OK")

    hr("decoy_gen: taxo_label 1-based join + select_work")
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        tdir = pathlib.Path(td)
        (tdir / "runX").mkdir()
        (tdir / "runX" / "5.sonnet.json").write_text(json.dumps(
            {"claims": [{"i": 1, "taxo": "genre"}, {"i": 2, "taxo": "quote"}]}))
        assert dg.taxo_label(tdir, "runX", 5, 0) == "genre"
        assert dg.taxo_label(tdir, "runX", 5, 1) == "quote"
        assert dg.taxo_label(tdir, "runX", 5, 9) is None
        assert dg.taxo_label(tdir, "runX", 6, 0) is None
    recs_fake = [dict(rec_fake, idx=i, doc_id=f"fineweb:{i}",
                      status="ok" if i != 2 else "skip") for i in range(5)]
    wk = dg.select_work(recs_fake, every=2, per_act=2, seed=0, only_idx=None)
    idxs = sorted({r["idx"] for r, _ in wk})
    print("selected idxs (every=2, skip idx2):", idxs)
    assert idxs == [0, 3], "every 2nd OK record: ok list is [0,1,3,4] -> [0,3]"
    wk2 = dg.select_work(recs_fake, every=2, per_act=2, seed=0, only_idx={1})
    assert sorted({r["idx"] for r, _ in wk2}) == [1], "--idx overrides the split"
    print("taxo join + selection: OK")

    hr("decoy_gen: _join_by_m fails closed on judge misnumbering")
    ok = dg._join_by_m([{"m": 1, "label": "absent"}, {"m": 2, "label": "present"}],
                       2, "absence")
    assert ok[1]["label"] == "absent" and ok[2]["label"] == "present"
    for bad, why in ([[{"m": 0}, {"m": 1}], "0-based numbering"],
                     [[{"m": 1}, {"m": 1}], "duplicate m"],
                     [[{"m": 1}], "one candidate unlabeled"],
                     [[], "empty payload"],
                     [[{"m": 1}, {"m": 2}, {"m": 3}], "extra item"]):
        try:
            dg._join_by_m(bad, 2, "absence")
            raise AssertionError(f"must raise on {why}")
        except RuntimeError as e:
            assert "misnumbered" in str(e)
    print("exact-set {1..K} enforced; shifts/dups/omissions raise: OK")

    hr("decoy_gen: _payload_list double-encoding recovery + garbage guards")
    items = [{"m": 1, "text": "a"}, {"m": 2, "text": "b"}]
    assert dg._payload_list({"decoys": items}, "decoys", "gen") == items
    # the live idx 352 claim 8 failure: array double-encoded as a JSON string
    assert dg._payload_list({"decoys": json.dumps(items)}, "decoys", "gen") == items
    for bad in ({"decoys": "not json ["}, {"decoys": {"m": 1}}, {}):
        try:
            dg._payload_list(bad, "decoys", "gen")
            raise AssertionError(f"must raise on {bad}")
        except RuntimeError as e:
            assert "not a list" in str(e)
    # single-char soup is now killed mechanically before any judge sees it
    cl2 = dg.clean_candidates(["[", "{", '"', "a plausible decoy claim here",
                               "x" * 400], "a claim of about this length ok")
    assert [c["kept"] for c in cl2] == [False, False, False, True, False]
    assert cl2[0]["reason"] == "length-ratio" and cl2[4]["reason"] == "length-ratio"
    print("double-encoded arrays decode; non-lists raise; char soup rejected: OK")

    hr("decoy_gen: cache_is_current stale-prompt/g guard")
    with tempfile.TemporaryDirectory() as td:
        p = pathlib.Path(td) / "0_0.sonnet.json"
        assert not dg.cache_is_current(p, 12), "missing file"
        p.write_text(json.dumps({"prompt_version": dg.PROMPT_VERSION,
                                 "g_requested": 12}))
        assert dg.cache_is_current(p, 12), "current file must count"
        assert not dg.cache_is_current(p, 8), "different --g regenerates"
        p.write_text(json.dumps({"prompt_version": "d1", "g_requested": 12}))
        assert not dg.cache_is_current(p, 12), "stale prompt regenerates"
        p.write_text("not json{")
        assert not dg.cache_is_current(p, 12), "garbage regenerates"
    print("prompt_version + g checked on resume; stale files regenerate: OK")

    hr("decoy_gen: _source_block truncation + hidden-window disclosure")
    # truncated doc, read window beyond the visible text (the 8-record case):
    # source_text at exactly the cap, context_left from the uncapped text
    src = ("lorem ipsum " * 600)[:dg.TEXT_CAP_CHARS]
    hidden_left = "completely different hidden prefix text ending in tok"
    rec_tr = dict(source_text=src, context_left=hidden_left, read_token="tok",
                  context_right="the continuation")
    blk = dg._source_block(rec_tr)
    assert f"TRUNCATED at {dg.TEXT_CAP_CHARS}" in blk, \
        "truncation must be inferred from the cap (records drop the flag)"
    assert "NOTE: this window lies beyond" in blk, "hidden read window disclosed"
    assert "<<<tok>>>" in blk
    # short doc, window visible: no banner, no note
    src2 = "A short document whose read window sits right here at the end tok"
    rec_ok = dict(source_text=src2, context_left=src2[-40:], read_token="tok",
                  context_right="after")
    blk2 = dg._source_block(rec_ok)
    assert "TRUNCATED" not in blk2 and "NOTE:" not in blk2
    # truncated doc but window visible (33/41 case): banner yes, note no
    rec_mid = dict(source_text=src, context_left=src[-40:], read_token=src[-3:],
                   context_right="after")
    blk3 = dg._source_block(rec_mid)
    assert "TRUNCATED" in blk3 and "NOTE:" not in blk3, \
        "trailing-only truncation must not trigger the hidden-window note"
    print("banner inferred from cap; NOTE only when the read window is hidden: OK")

    hr("decoy_gen: _call transient-vs-refusal routing")
    try:
        import anthropic  # noqa: F401
        have_sdk = True
    except ImportError:
        have_sdk = False
        print("anthropic SDK not installed - skipping (GPU-box environment)")
    if have_sdk:
        from types import SimpleNamespace as NS
        dg.time.sleep = lambda s: None          # no real backoff in tests

        def fake_msg(stop, payload=None):
            blocks = [NS(type="tool_use", input=payload)] if payload else []
            return NS(stop_reason=stop, content=blocks,
                      usage=NS(input_tokens=1, output_tokens=1))

        def client_with(side_effect):
            return NS(messages=NS(create=side_effect))

        calls = []
        def refusing_then_ok(**kw):
            calls.append(kw["model"])
            if kw["model"] == dg.MODELS["sonnet"]:
                return fake_msg("refusal")
            return fake_msg("tool_use", {"x": 1})
        payload, mk2, _ = dg._call(client_with(refusing_then_ok), "sonnet",
                                   "sys", {"name": "t"}, "content")
        assert mk2 != "sonnet" and payload == {"x": 1}, \
            "refusal (and ONLY refusal) falls back cross-model"

        calls = []
        def always_429(**kw):
            calls.append(kw["model"])
            e = RuntimeError("rate limited")
            e.status_code = 429
            raise e
        try:
            dg._call(client_with(always_429), "sonnet", "sys",
                     {"name": "t"}, "content")
            raise AssertionError("must raise after transient retries")
        except RuntimeError as e:
            assert "transient failures exhausted" in str(e)
        assert set(calls) == {dg.MODELS["sonnet"]} and \
            len(calls) == dg._TRANSIENT_TRIES, \
            f"429 must retry the PINNED model only, never haiku: {calls}"

        calls = []
        def bad_request(**kw):
            calls.append(kw["model"])
            e = RuntimeError("invalid request")
            e.status_code = 400
            raise e
        try:
            dg._call(client_with(bad_request), "sonnet", "sys",
                     {"name": "t"}, "content")
            raise AssertionError("400 must propagate immediately")
        except RuntimeError as e:
            assert "invalid request" in str(e) and len(calls) == 1
        print("refusal->fallback, 429->pinned retries then loud fail, "
              "400->immediate: OK")

        calls = []
        def truncating(**kw):
            calls.append(kw["model"])
            return fake_msg("max_tokens", {"x": 1})
        try:
            dg._call(client_with(truncating), "sonnet", "sys",
                     {"name": "t"}, "content", max_tokens=123)
            raise AssertionError("max_tokens must raise loudly")
        except RuntimeError as e:
            assert "max_tokens=123" in str(e) and len(calls) == 1, \
                "truncation is deterministic - no retry, no cross-model fallback"
        print("stop_reason=max_tokens -> immediate loud failure: OK")

    hr("decoy_gen: g-scaled token budgets (d4/g24)")
    # monkeypatch _call to capture the max_tokens each of the three call sites
    # passes; the fakes return well-formed payloads so the joins succeed
    G = 24
    claim_text = "sales rose sharply in March"
    zi = f"The report says that {claim_text} across all regions."
    s0 = zi.find(claim_text)
    rec_g = dict(idx=3, z_inner=zi, source_text="doc text",
                 context_left="prefix tok", read_token="tok",
                 context_right="after",
                 snippet_results=[dict(text=claim_text, type="detail",
                                       subtype=None, role=None, assertion=None,
                                       char_span=[s0, s0 + len(claim_text)])])
    budgets = {}
    real_call = dg._call
    def spy_call(client, mk2, system, tool, content, max_tokens=4000):
        budgets[tool["name"]] = max_tokens
        usage = {"input_tokens": 1, "output_tokens": 1}
        if tool is dg._GEN_TOOL:
            return ({"decoys": [f"figure {i:02d} moved differently there"
                                for i in range(G)]}, mk2, usage)
        if tool is dg._ABSENCE_TOOL:
            return ({"candidates": [{"m": m, "label": "absent", "evidence": ""}
                                    for m in range(1, G + 1)]}, mk2, usage)
        return ({"variants": [{"m": m, "score": 5, "issue": ""}
                              for m in range(1, G + 1)]}, mk2, usage)
    dg._call = spy_call
    try:
        cache_g = dg.gen_decoys_for_claim(None, rec_g, 0, None, G, "sonnet", "runG")
    finally:
        dg._call = real_call
    assert cache_g["n_text_pass"] == G, "all fakes must survive the filters"
    assert budgets[dg._GEN_TOOL["name"]] == max(4000, 300 * G) == 7200
    assert budgets[dg._ABSENCE_TOOL["name"]] == max(4000, 200 * G) == 4800
    assert budgets[dg._NATURAL_TOOL["name"]] == max(2000, 100 * G) == 2400
    print(f"budgets at g={G}:", budgets, "OK")

    if fails:
        print(f"\n{fails} FAILURE(S)"); sys.exit(1)
    print("\nALL S1+S2 OFFLINE TESTS PASSED")


if __name__ == "__main__":
    main()
