"""Rescore a cycle_run under Tier-0: refit kappa_0 to restore the <=5% decoy-vs-
decoy false-positive budget (2026-07-23).

The applied calibration kappa_0 was fit in-sample; on the run's own survivor D*
families (and the disjoint out-of-sample split) the leave-one-decoy-out FP exceeds
5% in 3/4 models. Tier-0 raises kappa_0 to the smallest value that holds the LODO
FP <= 5% on BOTH the full survivor set and the OOS split:

    kappa_0*  =  max( calibration kappa_0,
                      full-run LODO refit,
                      OOS-only LODO refit )

kappa_0 is a one-sided safety margin: raising it only RAISES the Gate-1 bar
(M_BB > Q95(D*) + kappa_0), so this recalibration can only make the test stricter
(grounded/entangled -> seeded/confabulated). It can never manufacture grounding and
never changes a pre-check, so the unresolved set is invariant. This makes the OOS
refit a legitimate input to the choice: using it to pick a MORE conservative
kappa_0 has no headline-inflation risk by construction.

Verdicts are a pure function of the stored readings, so this is a judge-free,
GPU-free arithmetic replay of cc.classify_claim with the new kappa_0.

Churn guard (abort on violation): the ONLY admissible change is
grounded/entangled -> seeded/confabulated. A raised bar cannot newly PASS Gate 1,
cannot touch a pre-check, and cannot cross Gate 2's (kappa-independent) threshold,
so seeded/confabulated/unresolved verdicts and grounded<->entangled must be stable.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np

import cycle_consistency as cc
import report as rp

_MERGE = ("verdict", "gate", "reason", "grounding_estimate",
          "q95_dstar", "med_dstar", "gate1_bar", "med_x")
_DEMOTE_FROM = ("grounded", "entangled")
_DEMOTE_TO = ("seeded", "confabulated")


def _scored(row: dict) -> list[dict]:
    return [c for c in row.get("candidates", [])
            if c.get("supp_star") is not None and c.get("dstar") is not None]


def _rows(claims_dir: Path) -> list[dict]:
    out = []
    for p in sorted(claims_dir.glob("*.json")):
        if p.name == "status.json":
            continue
        out.append(json.loads(p.read_text(encoding="utf-8")))
    return out


def _calib_idxs(calib_claims_dir: Path) -> set:
    idxs = set()
    for p in calib_claims_dir.glob("*.json"):
        if p.name == "status.json":
            continue
        r = json.loads(p.read_text(encoding="utf-8"))
        if r.get("idx") is not None:
            idxs.add(r["idx"])
    return idxs


def choose_kappa(rows, k, calib_k, calib_idxs) -> dict:
    """kappa_0* = max(calib, full-run LODO refit, OOS-only LODO refit)."""
    full = rp.kappa_lodo(rows, k)
    oos_rows = [r for r in rows if r.get("idx") not in calib_idxs] if calib_idxs else rows
    oos = rp.kappa_lodo(oos_rows, k)
    cands = [v for v in (calib_k, full["kappa_0_refit"], oos["kappa_0_refit"])
             if v is not None]
    kstar = max(cands)
    return dict(kappa_0_star=float(kstar), calib_kappa_0=calib_k,
                full_refit=full["kappa_0_refit"], oos_refit=oos["kappa_0_refit"],
                fp_full_at_star=rp.rate_at(full["grid"], kstar),
                fp_oos_at_star=rp.rate_at(oos["grid"], kstar),
                fp_full_at_calib=rp.rate_at(full["grid"], calib_k),
                fp_oos_at_calib=rp.rate_at(oos["grid"], calib_k),
                n_families_full=full["n_families"], n_families_oos=oos["n_families"])


def _reclassify(row: dict, kappa_0: float) -> dict:
    scored = _scored(row)
    dstars = [c["dstar"] for c in scored]
    x_ms = [c["x_m"] for c in scored if c.get("x_m") is not None]
    m_bb = row.get("m_bb")
    return cc.classify_claim(row.get("power") or {"passed": False},
                             row.get("validity") or {"passed": False},
                             m_bb if m_bb is not None else float("nan"),
                             dstars, x_ms, row.get("m_ib_max"), float(kappa_0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--calib", required=True, help="calibration_signed.json")
    ap.add_argument("--calib-claims-dir", required=True,
                    help=".../calib_d4g24/<run>/claims (defines the OOS split)")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    claims_dir = run_dir / "claims"
    calib = json.loads(Path(args.calib).read_text(encoding="utf-8"))
    calib_k = calib.get("kappa_0")
    k = int(calib.get("k", 5))
    calib_idxs = _calib_idxs(Path(args.calib_claims_dir))

    rows = _rows(claims_dir)
    ch = choose_kappa(rows, k, calib_k, calib_idxs)
    kstar = ch["kappa_0_star"]

    files = sorted(p for p in claims_dir.glob("*.json") if p.name != "status.json")
    trans: dict[str, int] = {}
    changed = 0
    for p in files:
        row = json.loads(p.read_text(encoding="utf-8"))
        if "power" not in row or "validity" not in row:
            continue
        old_v = row.get("verdict")
        res = _reclassify(row, kstar)
        new_v = res["verdict"]
        if new_v != old_v:
            # churn guard: raising kappa can ONLY demote grounded/entangled -> seeded/confab
            assert old_v in _DEMOTE_FROM and new_v in _DEMOTE_TO, (
                f"ILLEGAL Tier-0 transition {p.name}: {old_v} -> {new_v} "
                f"(kappa raised {calib_k}->{kstar})")
            trans[f"{old_v}->{new_v}"] = trans.get(f"{old_v}->{new_v}", 0) + 1
            changed += 1
        if args.apply:
            for kk in _MERGE:
                if kk in res:
                    row[kk] = res[kk]
                elif kk in row and kk not in res:
                    row.pop(kk, None)
            row["kappa_0"] = kstar
            tmp = p.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(row, indent=1), encoding="utf-8")
            tmp.replace(p)

    if args.apply and kstar != calib_k:
        calib["kappa_0"] = kstar
        calib.setdefault("tier0", {}).update(dict(
            kappa_0_original=calib_k, kappa_0_star=kstar,
            method="max(calib, full-run LODO refit, OOS LODO refit) to hold "
                   "decoy-vs-decoy FP <= 5% on full and OOS families",
            fp_full_at_original=ch["fp_full_at_calib"],
            fp_oos_at_original=ch["fp_oos_at_calib"],
            fp_full_at_star=ch["fp_full_at_star"],
            fp_oos_at_star=ch["fp_oos_at_star"], date="2026-07-23"))
        Path(args.calib).write_text(json.dumps(calib, indent=1), encoding="utf-8")

    def _fp(x):
        return "n/a" if x is None else f"{x:.4f}"
    mode = "APPLIED" if args.apply else "DRY-RUN"
    print(f"[{mode}] {run_dir}  kappa_0 {calib_k} -> {kstar} "
          f"(full-refit {ch['full_refit']}, oos-refit {ch['oos_refit']})")
    print(f"  FP full {_fp(ch['fp_full_at_calib'])}->{_fp(ch['fp_full_at_star'])}  "
          f"OOS {_fp(ch['fp_oos_at_calib'])}->{_fp(ch['fp_oos_at_star'])}  "
          f"(families full={ch['n_families_full']} oos={ch['n_families_oos']})")
    print(f"  changed={changed}")
    for kk, vv in sorted(trans.items(), key=lambda x: -x[1]):
        print(f"    {kk}: {vv}")


if __name__ == "__main__":
    main()
