#!/usr/bin/env python3
"""Four-way cross-model comparison: qwen-7B, gemma-12B, gemma-27B, llama-70B.

Adds Llama-3.3-70B (L53) to the three-way. This is also a cross-FAMILY test: qwen and gemma
share nothing with llama3 in tokenizer, chat template, or injection scale (llama inj_scale 30
vs gemma 60000 vs qwen 150), so a replication here is the strongest evidence that "reads gist,
confabulates position/structure" is a property of single-position NLA readout rather than of
one model family. Buckets every content claim identically to the other runs.

Pure stdlib + numpy. Bucketing copied verbatim from cross_model.py / three_way.py.
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
    ("gemma-27B", ATH / "g4_analysis" / "records.jsonl"),
    ("llama-70B", HERE / "records.jsonl"),
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
    print("  {:12s}".format("bucket") + "".join(f"{n:>16s}" for n in names))
    for b in BUCKETS:
        print("  {:12s}".format(b) + "".join(
            f"{R[n][b][1]*100:6.0f}% ({R[n][b][2]*100:3.0f}%c)" for n in names))

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

    if "llama-70B" in R:
        print("\nbucket grounded-rate level correlation (Pearson) vs llama-70B:")
        l70 = np.array([R["llama-70B"][b][1] for b in BUCKETS])
        for name in names:
            if name == "llama-70B":
                continue
            v = np.array([R[name][b][1] for b in BUCKETS])
            print(f"  {name:12s} vs llama-70B: {np.corrcoef(v, l70)[0,1]:+.2f}")


if __name__ == "__main__":
    main()
