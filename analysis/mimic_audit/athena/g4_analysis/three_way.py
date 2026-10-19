#!/usr/bin/env python3
"""Three-way cross-model comparison: qwen-7B (cp6) vs gemma-12B (cp7) vs gemma-27B (g4).

Tests whether the headline (the NLA reads semantic GIST and confabulates POSITION /
STRUCTURE) holds across a 7B, a 12B, and a 27B model. Reads each run's records.jsonl,
buckets every content claim by semantic category, and reports grounded/confab rates side
by side plus the gist-minus-position gap and the level correlation across the three models.

Pure stdlib + numpy (repo dep constraint). Bucketing logic is copied verbatim from
cp7_analysis/cross_model.py so the 27B numbers are computed the same way as the others.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ATH = HERE.parent
BUCKETS = ("GIST", "QUOTE", "POSITION", "PREDICTION")

MODELS = [
    ("qwen-7B",   ATH / "cp6_analysis" / "records.jsonl"),
    ("gemma-12B", ATH / "cp7_analysis" / "records.jsonl"),
    ("gemma-27B", HERE / "records.jsonl"),
]


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
    n = len(L) or 1
    return tuple(sum(s["verdict"] == v for s in L) / n for v in ("grounded", "entangled", "confabulated"))


def main():
    data = [(name, load(p)) for name, p in MODELS if p.exists()]
    R = {name: rates(sn) for name, sn in data}
    names = [n for n, _ in data]

    print("SEMANTIC BUCKET   grounded% (confab%) per model")
    hdr = "  {:12s}".format("bucket") + "".join(f"{n:>16s}" for n in names)
    print(hdr)
    for b in BUCKETS:
        row = "  {:12s}".format(b) + "".join(
            f"{R[n][b][1]*100:6.0f}% ({R[n][b][2]*100:3.0f}%c)" for n in names)
        print(row)

    print("\noverall grounded/entangled/confab:")
    for name, sn in data:
        o = overall(sn)
        print(f"  {name:12s} {o[0]:.0%} / {o[1]:.0%} / {o[2]:.0%}   (n_located={sum(s['located'] for s in sn)})")

    print("\nGIST - POSITION grounded gap (positive = reads gist more than position):")
    for name in names:
        print(f"  {name:12s} +{(R[name]['GIST'][1]-R[name]['POSITION'][1])*100:.0f} pts")

    print("\ntop-grounded bucket per model (REPLICATES if all GIST):")
    for name in names:
        gv = {b: R[name][b][1] for b in BUCKETS}
        print(f"  {name:12s} {max(gv, key=gv.get)}")

    if len(names) >= 2:
        print("\nbucket grounded-rate level correlation (Pearson) vs gemma-27B:")
        g27 = np.array([R["gemma-27B"][b][1] for b in BUCKETS]) if "gemma-27B" in R else None
        if g27 is not None:
            for name in names:
                if name == "gemma-27B":
                    continue
                v = np.array([R[name][b][1] for b in BUCKETS])
                print(f"  {name:12s} vs gemma-27B: {np.corrcoef(v, g27)[0,1]:+.2f}")


if __name__ == "__main__":
    main()
