#!/usr/bin/env python3
"""CP5 by-type / by-role / by-subtype / hedged report over the 25-activation atom run.

Reads records.jsonl (atom mode, p90+floor) and prints the verdict contingency sliced by the
segmenter's claim metadata. Denominator for every rate is LOCATED claims of that slice; 95%
CIs are Wilson. The confab/grounded split is the judge-free teacher-forced Δlogp verdict.
"""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RECS = [json.loads(l) for l in (HERE / "records.jsonl").read_text().splitlines() if l.strip()]
SNIPS = [s for r in RECS if r.get("status") == "ok" for s in r.get("snippet_results", [])]


def wilson(k, n, z=1.96):
    if not n:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def row(items, label):
    L = [s for s in items if s["located"]]
    g = sum(s["verdict"] == "grounded" for s in L)
    e = sum(s["verdict"] == "entangled" for s in L)
    c = sum(s["verdict"] == "confabulated" for s in L)
    dl = [s["dlogp"] for s in L if s["dlogp"] is not None]
    gci, cci = wilson(g, len(L)), wilson(c, len(L))
    med = np.median(dl) if dl else 0.0
    print(f"  {label:26s} n={len(items):3d} loc={len(L):3d} | grounded {g:3d} "
          f"({g/len(L) if L else 0:.0%}) CI[{gci[0]:.0%},{gci[1]:.0%}]  entangled {e:3d} "
          f"({e/len(L) if L else 0:.0%})  confab {c:3d} ({c/len(L) if L else 0:.0%}) "
          f"CI[{cci[0]:.0%},{cci[1]:.0%}]  dlogp~{med:.1f}")


def main():
    loc = [s for s in SNIPS if s["located"]]
    print(f"CP5: {len({id(r) for r in RECS})} records, {len(SNIPS)} content claims, "
          f"{len(loc)} located, grounded {sum(s['verdict']=='grounded' for s in loc)/len(loc):.1%}")
    print("\n=== BY TYPE ===")
    for t in ("theme", "detail"):
        row([s for s in SNIPS if s.get("type") == t], t)
    print("\n=== BY ROLE (the headline axis) ===")
    for r in ("describes", "quotes", "predicts"):
        row([s for s in SNIPS if s.get("role") == r], r)
    print("\n=== DETAIL SUBTYPES (>=5) ===")
    for su, _ in Counter(s.get("subtype", "") for s in SNIPS if s.get("type") == "detail").most_common():
        items = [s for s in SNIPS if s.get("type") == "detail" and s.get("subtype") == su]
        if len(items) >= 5:
            row(items, f"detail/{su[:20]}")
    print("\n=== HEDGED vs CONFIDENT ===")
    row([s for s in SNIPS if s.get("hedged")], "hedged")
    row([s for s in SNIPS if not s.get("hedged")], "confident")


if __name__ == "__main__":
    main()
