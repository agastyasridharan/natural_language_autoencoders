#!/usr/bin/env python3
"""CP7 cross-model comparison: qwen-7B (cp6) vs gemma-12B (cp7) atom runs.

Tests whether the CP6 headline (the NLA reads semantic GIST and confabulates POSITION /
STRUCTURE) replicates on a second model. Reads both runs' records.jsonl, buckets every
content claim by semantic category, and reports grounded/confab rates side by side plus the
gist-minus-position gap and the rank/level correlation of the bucket grounded-rates.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ATH = HERE.parent
BUCKETS = ("GIST", "QUOTE", "POSITION", "PREDICTION")


def load(p):
    return [s for l in Path(p).read_text().splitlines() if l.strip()
            for r in [json.loads(l)] if r.get("status") == "ok"
            for s in r.get("snippet_results", [])]


def _has(su, *ks):
    return any(k in su for k in ks)


def bucket(s):
    su = (s.get("subtype") or "").lower()
    if s.get("role") == "quotes":
        return "QUOTE"
    if s.get("role") == "predicts" or _has(su, "expect", "predict", "next-token", "next token",
                                           "candidate", "continuation", "completion", "upcoming"):
        return "PREDICTION"
    if _has(su, "position", "syntact", "structural", "structure", "token", "clause",
            "parenthetic", "mid-", "boundary"):
        return "POSITION"
    if _has(su, "genre", "topic", "tone", "style", "register", "rhetor", "subject", "purpose",
            "narrative", "mood", "voice", "format", "content", "implication", "theme"):
        return "GIST"
    return "other"


def rates(SN):
    out = {}
    for b in BUCKETS:
        L = [s for s in SN if bucket(s) == b and s["located"]]
        g = sum(s["verdict"] == "grounded" for s in L)
        c = sum(s["verdict"] == "confabulated" for s in L)
        out[b] = (len(L), g / len(L) if L else 0, c / len(L) if L else 0)
    return out


def overall(SN):
    L = [s for s in SN if s["located"]]
    n = len(L)
    return tuple(sum(s["verdict"] == v for s in L) / n for v in ("grounded", "entangled", "confabulated"))


def rank(a):
    order = a.argsort()
    r = np.empty_like(order, float)
    r[order] = np.arange(len(a))
    return r


def main():
    Q = load(ATH / "cp6_analysis" / "records.jsonl")
    G = load(HERE / "records.jsonl")
    rq, rg = rates(Q), rates(G)
    print("SEMANTIC BUCKET   grounded% (confab%)     QWEN-7B        GEMMA-12B")
    for b in BUCKETS:
        print(f"  {b:11s}  qwen {rq[b][1]:4.0%} ({rq[b][2]:.0%}c)   gemma {rg[b][1]:4.0%} ({rg[b][2]:.0%}c)")
    oq, og = overall(Q), overall(G)
    print(f"\noverall grounded/entangled/confab:  qwen {oq[0]:.0%}/{oq[1]:.0%}/{oq[2]:.0%}   "
          f"gemma {og[0]:.0%}/{og[1]:.0%}/{og[2]:.0%}")
    print(f"GIST - POSITION grounded gap:  qwen +{(rq['GIST'][1]-rq['POSITION'][1])*100:.0f}pts   "
          f"gemma +{(rg['GIST'][1]-rg['POSITION'][1])*100:.0f}pts   (both > 0 = gist read > position, cross-model)")
    qv = np.array([rq[b][1] for b in BUCKETS])
    gv = np.array([rg[b][1] for b in BUCKETS])
    print(f"bucket grounded-rate: Spearman rho={np.corrcoef(rank(qv), rank(gv))[0,1]:+.2f}  "
          f"Pearson={np.corrcoef(qv, gv)[0,1]:+.2f}")
    print(f"top-grounded bucket: qwen={BUCKETS[int(qv.argmax())]} gemma={BUCKETS[int(gv.argmax())]}   "
          f"(REPLICATES if both GIST)")


if __name__ == "__main__":
    main()
