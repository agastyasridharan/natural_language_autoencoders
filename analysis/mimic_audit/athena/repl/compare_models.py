#!/usr/bin/env python3
"""Fair cross-model comparison of two claim-ablation runs -- NO GPU.

The raw confabulation rate is NOT comparable across models when their injected
activations differ in fragility: a model whose read collapses under ANY
perturbation has a high random-direction null, so its test has less power and
more claims fall below the (high) bar -- inflating "confabulated" without the
verbalizer inventing more. (qwen null ~0; gemma null ~50x larger.)

This script makes the comparison fair three ways:
  1. NULL SCALE        per-activation random-ablation floor, per model (the confound)
  2. FRAGILITY-STRATIFIED CONFAB   confab rate within matched fragility bins
                                   (the overlap/low-fragility bin is the fair test)
  3. NULL-NORMALIZED SIGNAL (zeta)  Delta_logp in robust-SD units of each
                                    activation's own null -- comparable across models

Usage:
  python repl/compare_models.py runs/qwen7b-... runs/gemma12b-...
"""
import json
import sys
from pathlib import Path

import numpy as np

EPS = 1e-6
BINS = [(0.0, 0.5), (0.5, 2.0), (2.0, 10.0), (10.0, float("inf"))]
BIN_LABELS = ["<0.5", "0.5-2", "2-10", ">=10"]


def load(run: Path):
    """Return per-snippet rows and per-activation null scale."""
    recs = [json.loads(l) for l in (run / "records.jsonl").read_text().splitlines() if l.strip()]
    rows = []          # one per located snippet
    act_null95 = {}    # idx -> 95th pct of all null samples in that activation
    for r in recs:
        if r.get("status") != "ok":
            continue
        pool = [x for s in r["snippet_results"] if s.get("located") for x in s["null_dlogps"]]
        a95 = float(np.percentile(pool, 95)) if pool else float("nan")
        act_null95[r["idx"]] = a95
        for s in r["snippet_results"]:
            if not s.get("located"):
                continue
            nd = np.array(s["null_dlogps"], float)
            med = float(np.median(nd))
            mad = float(np.median(np.abs(nd - med)))
            zeta = (s["dlogp"] - med) / (1.4826 * mad + EPS)
            rows.append(dict(idx=r["idx"], dlogp=s["dlogp"], verdict=s["verdict"],
                             zeta=zeta, act95=a95))
    return rows, act_null95


def q(a, p):
    return float(np.percentile(a, p)) if len(a) else float("nan")


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    runs = [Path(sys.argv[1]), Path(sys.argv[2])]
    names = [r.name.split("-")[0] + ":" + ("fineweb" if "fineweb" in r.name else
             "ultrafineweb" if "cross-offline" in r.name else r.name.split("-")[-1]) for r in runs]
    data = [load(r) for r in runs]

    print("cross-model comparison")
    for nm, r in zip(names, runs):
        print(f"  {nm:24} {r.name}")

    # ── 1. null scale (the confound) ─────────────────────────────────────────
    print("\n" + "=" * 74 + "\n 1. ACTIVATION NULL SCALE  (random-ablation floor; the fragility confound)\n" + "=" * 74)
    print(f"  {'model':24} {'median':>9} {'mean':>9} {'p90':>9} {'max':>9}")
    for nm, (rows, a95) in zip(names, data):
        v = np.array([x for x in a95.values() if not np.isnan(x)])
        print(f"  {nm:24} {np.median(v):>9.2f} {v.mean():>9.2f} {q(v,90):>9.2f} {v.max():>9.2f}")
    print("  (per-activation 95th-pct of random-direction Delta_logp; higher = more fragile = less test power)")

    # ── 2. fragility-stratified confab rate (the fair comparison) ────────────
    print("\n" + "=" * 74 + "\n 2. CONFAB RATE STRATIFIED BY FRAGILITY  (compare within matched bins)\n" + "=" * 74)
    print(f"  {'null95 bin':>10} | " + " | ".join(f"{nm:>22}" for nm in names))
    for (lo, hi), lab in zip(BINS, BIN_LABELS):
        cells = []
        for rows, a95 in data:
            sub = [x for x in rows if lo <= x["act95"] < hi]
            n = len(sub)
            c = sum(x["verdict"] == "confabulated" for x in sub)
            cells.append(f"{(c/n*100 if n else 0):>5.1f}%  (n={n:>4})" if n else f"{'--':>13}")
        print(f"  {lab:>10} | " + " | ".join(f"{c:>22}" for c in cells))
    print("  -> the LOW-fragility bin (<0.5) is where both models have test power:")
    print("     if gemma confabulates more THERE too, it is real; if only in high bins, it is low-power.")

    # ── 3. null-normalized signal (comparable magnitude) ─────────────────────
    print("\n" + "=" * 74 + "\n 3. NULL-NORMALIZED SIGNAL zeta = (Delta_logp - med_null) / (1.4826*MAD_null)\n" + "=" * 74)
    print(f"  {'model':24} {'grounded zeta med':>18} {'confab zeta med':>16} {'all zeta med':>13}")
    for nm, (rows, _) in zip(names, data):
        g = [x["zeta"] for x in rows if x["verdict"] == "grounded"]
        c = [x["zeta"] for x in rows if x["verdict"] == "confabulated"]
        allz = [x["zeta"] for x in rows]
        print(f"  {nm:24} {np.median(g) if g else float('nan'):>18.1f} "
              f"{np.median(c) if c else float('nan'):>16.2f} {np.median(allz):>13.1f}")
    print("  (grounded claims should sit well above their own noise in BOTH models;")
    print("   if gemma's grounded zeta ~ qwen's, the test works -- the confab gap is power/fragility.)")

    # ── headline ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 74)
    for nm, (rows, _) in zip(names, data):
        n = len(rows)
        c = sum(x["verdict"] == "confabulated" for x in rows)
        g = sum(x["verdict"] == "grounded" for x in rows)
        print(f"  {nm:24} raw: grounded {g/n*100:.1f}%  confab {c/n*100:.1f}%  (n={n})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
