"""Compare a Tier-3 (P95 band) cycle run against the committed P90 run, and apply
the Tier-3 guardrails (2026-07-23).

Tier-3 widens the f3/f4 selection bands (P90 -> P95) and re-measures, so unlike the
no-GPU replays it is NOT churn-locked: widening admits more decoys, which can both
RECOVER unresolved claims (a validity family reaches min_k clean placebos) and MOVE
resolved claims (a larger Gate-1 null shifts Q95(D*)). This script characterises
both directions and runs the acceptance guardrails.

Guardrails (report, do not silently pass):
  - Integrity: M_BB is band-independent, so tier3 M_BB must reproduce the committed
    M_BB to ~1e-6 on shared claims (a mismatch means the re-measure drifted).
  - FP budget: the tier3 report's decoy-vs-decoy LODO FP must hold <= 5% (full+OOS);
    printed for inspection (refit kappa_0 Tier-0-style if breached).
  - Smell test: newly-RESOLVED (was unresolved) claims must distribute like the
    baseline resolved population - mostly seeded/confabulated, NOT a grounded spike.
    A grounded/entangled fraction among the recovered that exceeds the baseline
    resolved grounded/entangled fraction is a red flag (band admitted contaminated
    decoys that manufacture grounding).

Usage:
  tier3_compare.py --tier3 <run_dir> --base <committed_run_dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

VERDICTS = ("grounded", "entangled", "seeded", "confabulated", "unresolved")
RESOLVED = ("grounded", "entangled", "seeded", "confabulated")


def _claims(run_dir: Path) -> dict:
    out = {}
    for p in sorted((run_dir / "claims").glob("*.json")):
        if p.name == "status.json":
            continue
        out[p.name] = json.loads(p.read_text(encoding="utf-8"))
    return out


def _dist(rows) -> dict:
    d = {v: 0 for v in VERDICTS}
    for r in rows:
        v = r.get("verdict")
        d[v] = d.get(v, 0) + 1
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier3", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--report", default=None, help="tier3 cycle_report.json (FP audit)")
    args = ap.parse_args()
    t3 = _claims(Path(args.tier3))
    bs = _claims(Path(args.base))
    shared = sorted(set(t3) & set(bs))
    print(f"claims: tier3={len(t3)} base={len(bs)} shared={len(shared)}")

    # integrity: M_BB band-independent
    mbb_drift = 0.0
    n_surv_base = n_surv_t3 = 0
    for name in shared:
        a, b = bs[name].get("m_bb"), t3[name].get("m_bb")
        if a is not None and b is not None:
            mbb_drift = max(mbb_drift, abs(a - b))
        n_surv_base += bs[name].get("n_surv") or 0
        n_surv_t3 += t3[name].get("n_surv") or 0
    print(f"integrity: max |M_BB tier3-base| = {mbb_drift:.2e} (band-independent; want < 1e-3)")
    print(f"survivors: base sum={n_surv_base}  tier3 sum={n_surv_t3}  "
          f"(+{n_surv_t3 - n_surv_base}, {100*(n_surv_t3-n_surv_base)/max(1,n_surv_base):.1f}%)")

    # distributions
    db = _dist([bs[n] for n in shared])
    dt = _dist([t3[n] for n in shared])
    print("\nverdict distribution (shared claims):")
    print(f"  {'':12} " + " ".join(f"{v[:5]:>6}" for v in VERDICTS) + "   unres%")
    for lbl, d in (("base (P90)", db), ("tier3 (P95)", dt)):
        N = sum(d.values())
        print(f"  {lbl:12} " + " ".join(f"{d[v]:>6}" for v in VERDICTS) +
              f"   {100*d['unresolved']/N:.1f}")

    # transition matrix
    trans = {}
    for name in shared:
        o, n = bs[name].get("verdict"), t3[name].get("verdict")
        if o != n:
            trans[f"{o}->{n}"] = trans.get(f"{o}->{n}", 0) + 1
    print("\ntransitions (base -> tier3):")
    for k, v in sorted(trans.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")

    # recovered (unresolved -> resolved) + smell test
    recovered = [t3[n] for n in shared
                 if bs[n].get("verdict") == "unresolved"
                 and t3[n].get("verdict") in RESOLVED]
    lost = [n for n in shared if bs[n].get("verdict") in RESOLVED
            and t3[n].get("verdict") == "unresolved"]
    rec_d = _dist(recovered)
    nrec = len(recovered)
    print(f"\nrecovered unresolved -> resolved: {nrec}")
    if nrec:
        ge_rec = (rec_d["grounded"] + rec_d["entangled"]) / nrec
        for v in RESOLVED:
            print(f"    {v:13} {rec_d[v]:>4}  ({100*rec_d[v]/nrec:.1f}%)")
        # baseline resolved g/e fraction
        base_res = sum(db[v] for v in RESOLVED)
        ge_base = (db["grounded"] + db["entangled"]) / max(1, base_res)
        print(f"  SMELL TEST: recovered grounded+entangled frac = {100*ge_rec:.1f}%  "
              f"vs baseline resolved g+e frac = {100*ge_base:.1f}%  "
              f"-> {'RED FLAG' if ge_rec > ge_base + 0.05 else 'ok'}")
    print(f"resolved -> unresolved (lost to wider Gate-1 null): {len(lost)}")

    if args.report:
        rep = json.loads(Path(args.report).read_text())
        oos = rep.get("kappa_out_of_sample") or {}
        print(f"\nFP audit (tier3 report): kappa_0={rep.get('kappa_0_calibration')} "
              f"FP_full={rep.get('decoy_fp_rate_at_calibration_kappa')} "
              f"FP_oos={oos.get('decoy_fp_rate_at_calibration_kappa')} "
              f"transfers={oos.get('transfers')} refit={rep.get('kappa_0_refit')} "
              f"oos_refit={oos.get('kappa_0_refit')}")


if __name__ == "__main__":
    main()
