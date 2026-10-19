#!/usr/bin/env python3
"""CP4 specificity-denominator methodology check (reproducible, no GPU).

Reads the 8-activation probe output (cp4_records.json + cp4_deltas.npz, produced by
repl/cp4_specificity_probe.py on Athena) and reproduces the CP4 evidence that led to the
chosen rule: the specificity denominator for atom mode is the 90th percentile of the
collateral column (not its max), with a 1-nat absolute floor.

Three findings:
  1) atomic AR directions are moderately coupled (median max-offdiag |cos| ~0.42); a third of
     claims have a >0.5 sibling. So many atomic claims share substrate.
  2) the token-boundary-overlap artifact and Δ-duplicate exclusion change ZERO verdicts: the
     sibling that dominates `max` is a different, partially-overlapping claim, not a neighbor
     token nor a geometric duplicate.
  3) `max` is scale-biased: corr(grounded-fraction, claim-count) = -0.74 (finer decomposition
     mechanically lowers grounded-rate). p90 removes the bias; a 1-nat floor stops p90 from
     grounding near-null claims. p75 is too loose (entangled nearly vanishes).

The verdict's confabulated/unlocated split is denominator-INDEPENDENT (it is the random-null
test); the denominator only moves the grounded<->entangled boundary. So this choice does not
affect the confabulation-by-type headline, only the grounded/entangled distinction.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RECS = json.load(open(HERE / "cp4_records.json"))
DEL = np.load(HERE / "cp4_deltas.npz")
SPEC_MARGIN = 1.5


def spec_matrix(r):
    n = r["n_content"]
    return np.array([[(s["spec_row"][j] if s["spec_row"][j] is not None else np.nan)
                      for j in range(n)] for s in r["snippet_results"]], float)


def classify(r, mode, pct=90, floor=0.0):
    """Verdict per content claim under a denominator rule. mode in {max, p}."""
    M, sr, n, out = spec_matrix(r), r["snippet_results"], r["n_content"], []
    for i in range(n):
        s = sr[i]
        if not s["located"]:
            out.append("unlocated"); continue
        if not s["depends_on_h"]:                       # random-null test: fixed across rules
            out.append("confabulated"); continue
        col = [M[k, i] for k in range(n) if k != i]
        denom = (max(col) if col else 0.0) if mode == "max" \
            else (float(np.nanpercentile(col, pct)) if col else 0.0)
        denom = max(denom, 0.0)
        grounded = (s["dlogp"] > SPEC_MARGIN * denom) and (s["dlogp"] >= floor)
        out.append("grounded" if grounded else "entangled")
    return out


def corpus(mode, **kw):
    T = dict(grounded=0, entangled=0, confabulated=0, unlocated=0)
    for r in RECS:
        for v in classify(r, mode, **kw):
            T[v] += 1
    return T


def scale_corr(mode, **kw):
    xs, ys = [], []
    for r in RECS:
        v = classify(r, mode, **kw); g, e = v.count("grounded"), v.count("entangled")
        if g + e:
            xs.append(r["n_content"]); ys.append(g / (g + e))
    return float(np.corrcoef(xs, ys)[0, 1])


def main():
    print("=== 1) AR-direction coupling (|cos(Δi,Δk)| off-diagonal) ===")
    allmx = []
    for r in RECS:
        D = DEL[str(r["idx"])]; Dn = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-9)
        C = np.abs(Dn @ Dn.T); np.fill_diagonal(C, np.nan); mx = np.nanmax(C, axis=1); allmx.append(mx)
    allmx = np.concatenate(allmx)
    print(f"  median max-offdiag |cos| = {np.median(allmx):.2f};  "
          f"{np.mean(allmx > 0.5):.0%} of claims have a >0.5 sibling")

    print("\n=== 2) corpus tallies (8 activations, 103 located claims) ===")
    for name, mode, kw in [("max (current)", "max", {}), ("p90", "p", dict(pct=90)),
                           ("p90 + floor 1.0", "p", dict(pct=90, floor=1.0)),
                           ("p75", "p", dict(pct=75))]:
        T = corpus(mode, **kw); loc = T["grounded"] + T["entangled"]
        print(f"  {name:16s} grounded={T['grounded']:3d} entangled={T['entangled']:3d} "
              f"confab={T['confabulated']} unloc={T['unlocated']}  grounded-rate={T['grounded']/loc:.0%}")

    print("\n=== 3) scale bias  corr(grounded-fraction, n_claims) ===")
    print(f"  max={scale_corr('max'):+.2f}   p90={scale_corr('p', pct=90):+.2f}   "
          f"p90+floor1.0={scale_corr('p', pct=90, floor=1.0):+.2f}")
    print("  (negative = finer decomposition mechanically lowers grounded-rate)")

    T = corpus("p", pct=90, floor=1.0)
    assert (T["grounded"], T["entangled"], T["confabulated"]) == (24, 35, 44), T
    print("\nCHOSEN RULE: p90 denominator + 1-nat floor  ->  24 grounded / 35 entangled / 44 confab")
    print("Implemented in repl/claim_ablation.py:_atom_specificity (spec_pct=90, spec_floor=1.0).")


if __name__ == "__main__":
    main()
