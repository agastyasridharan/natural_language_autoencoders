#!/usr/bin/env python3
"""CP6 full-corpus (N=400) atom report: verdict contingency by type / role / subtype /
hedged, with Wilson 95% CIs and a hand-rolled chi-square independence test (scipy is not an
allowed dep). Denominator for every rate is LOCATED content claims of that slice.

The grounded/entangled/confabulated verdict is the judge-free teacher-forced Δlogp test with
the CP4 rule (p90 collateral denominator + 1-nat floor). 'confabulated' = within the
random-direction null: the claim's tokens are no less likely once h's projection onto the
claim's AR direction is removed, so it was not read from the activation.
"""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RECS = [json.loads(l) for l in (HERE / "records.jsonl").read_text().splitlines() if l.strip()]
OK = [r for r in RECS if r.get("status") == "ok"]
SNIPS = [s for r in OK for s in r.get("snippet_results", [])]
VERD = ("grounded", "entangled", "confabulated")


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
    n = len(L)
    g, e, c = (sum(s["verdict"] == v for s in L) for v in VERD)
    dl = [s["dlogp"] for s in L if s["dlogp"] is not None]
    gci, cci = wilson(g, n), wilson(c, n)
    print(f"  {label:24s} n={len(items):4d} loc={n:4d} | grounded {g:4d} "
          f"({g/n if n else 0:.0%}) CI[{gci[0]:.0%},{gci[1]:.0%}]  entangled {e:4d} "
          f"({e/n if n else 0:.0%})  confab {c:4d} ({c/n if n else 0:.0%}) "
          f"CI[{cci[0]:.0%},{cci[1]:.0%}]  Δlogp~{np.median(dl) if dl else 0:.1f}")
    return [g, e, c]


def chisq(rows, labels):
    """rows: list of [g,e,c] counts. Hand-rolled Pearson chi-square + Cramer's V."""
    M = np.array(rows, float)
    rt, ct, N = M.sum(1, keepdims=True), M.sum(0, keepdims=True), M.sum()
    exp = rt @ ct / N
    chi2 = float(((M - exp) ** 2 / exp).sum())
    r, c = M.shape
    dof = (r - 1) * (c - 1)
    V = float(np.sqrt(chi2 / (N * min(r - 1, c - 1))))
    # chi-square critical values (no scipy): {dof: (p.05, p.01, p.001)}
    crit = {2: (5.99, 9.21, 13.82), 4: (9.49, 13.28, 18.47), 6: (12.59, 16.81, 22.46)}
    c05, c01, c001 = crit.get(dof, (None, None, None))
    sig = "p<.001" if (c001 and chi2 > c001) else ("p<.01" if (c01 and chi2 > c01)
          else ("p<.05" if (c05 and chi2 > c05) else "ns"))
    print(f"  chi2={chi2:.1f} dof={dof} ({sig}; crit.05={c05})  Cramer's V={V:.3f}  [{'/'.join(labels)}]")


def _has(su, *ks):
    return any(k in su for k in ks)


def bucket(s):
    """Group a content claim by the SEMANTIC CATEGORY of its subtype (the axis that actually
    predicts groundedness, far more than type or role)."""
    su = (s.get("subtype") or "").lower()
    if s.get("role") == "quotes":
        return "QUOTE (verbatim span)"
    if s.get("role") == "predicts" or _has(su, "expect", "predict", "next-token", "next token",
                                            "candidate", "continuation", "completion", "upcoming"):
        return "PREDICTION (what comes next)"
    if _has(su, "position", "syntact", "structural", "structure", "token", "clause",
            "parenthetic", "mid-", "boundary"):
        return "POSITION / STRUCTURE"
    if _has(su, "genre", "topic", "tone", "style", "register", "rhetor", "subject", "purpose",
            "narrative", "mood", "voice", "format", "content", "implication", "theme"):
        return "GIST (genre/topic/tone/style)"
    return "other"


def semantic_buckets():
    print("\n=== SEMANTIC BUCKETS (the headline axis) ===")
    order = ["GIST (genre/topic/tone/style)", "QUOTE (verbatim span)",
             "POSITION / STRUCTURE", "PREDICTION (what comes next)", "other"]
    rows = []
    for b in order:
        items = [s for s in SNIPS if bucket(s) == b]
        r = row(items, b)
        if b != "other":
            rows.append(r)
    chisq(rows, ("gist", "quote", "position", "prediction"))


def main():
    loc = [s for s in SNIPS if s["located"]]
    S = json.load(open(HERE / "summary.json"))
    gc = sum(s["verdict"] == "grounded" for s in loc)
    print(f"CP6 N={len(OK)} activations, {len(SNIPS)} content claims, {len(loc)} located, "
          f"unlocated {len(SNIPS)-len(loc)}")
    print(f"grounded {gc} ({gc/len(loc):.1%})  entangled "
          f"{sum(s['verdict']=='entangled' for s in loc)}  confab "
          f"{sum(s['verdict']=='confabulated' for s in loc)}  | "
          f"Δlogp-ΔMSE pearson={S.get('dlogp_dmse_pearson'):.3f}")

    print("\n=== BY TYPE ===")
    rt = [row([s for s in SNIPS if s.get("type") == t], t) for t in ("theme", "detail")]
    chisq(rt, ("theme", "detail"))

    print("\n=== BY ROLE (headline axis) ===")
    rr = [row([s for s in SNIPS if s.get("role") == r], r)
          for r in ("describes", "quotes", "predicts")]
    chisq(rr, ("describes", "quotes", "predicts"))

    print("\n=== DETAIL SUBTYPES (>=10) ===")
    for su, _ in Counter(s.get("subtype", "") for s in SNIPS if s.get("type") == "detail").most_common():
        items = [s for s in SNIPS if s.get("type") == "detail" and s.get("subtype") == su]
        if len(items) >= 10:
            row(items, f"detail/{su[:18]}")

    print("\n=== THEME SUBTYPES (>=15) ===")
    for su, _ in Counter(s.get("subtype", "") for s in SNIPS if s.get("type") == "theme").most_common():
        items = [s for s in SNIPS if s.get("type") == "theme" and s.get("subtype") == su]
        if len(items) >= 15:
            row(items, f"theme/{su[:18]}")

    print("\n=== HEDGED vs CONFIDENT ===")
    rh = [row([s for s in SNIPS if s.get("hedged")], "hedged"),
          row([s for s in SNIPS if not s.get("hedged")], "confident")]
    chisq(rh, ("hedged", "confident"))

    semantic_buckets()


if __name__ == "__main__":
    main()
