"""Exp 7 — pick the 200-doc/model resample subsample, stratified by Exp-6 LCR divergence (CPU, local).

Per-datapoint divergence = mean over the datapoint's LCR quote divergences (from records_exp6b, the A2
arm; divergence = 1 - restate_fidelity of each LCR quote against the true pre-token tail). Datapoints are
split into low/mid/high divergence terciles (over LCR-bearing docs) and ~equal counts are drawn from each
tercile by a fixed-seed rng, so the confabulation axis (faithful restatement -> pure confabulation) is
covered. LCR-less datapoints (genre/PC only) form a 4th small bucket sampled proportionally so pure-scaffold
reads are represented too. The selection is held fixed across every temperature and both clustering runs.

Output (pushed to the cluster, consumed by fve_resample.py):
  results/exp7_subsample_<key>.json  {"key", "n", "seed", "bounds":[t1,t2], "docs":[{idx,doc_id,divergence,
                                       stratum,n_lcr}...]}

Usage:  python3 exp7_subsample.py [qwen7b gemma12b gemma27b] [--n 200] [--seed 7]
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
KEYS = ["qwen7b", "gemma12b", "gemma27b"]


def datapoint_divergence(rec: dict) -> float | None:
    """Mean LCR-quote divergence for one datapoint, or None if it has no LCR quotes."""
    qs = (rec.get("edit_meta") or {}).get("quotes") or []
    divs = [q["divergence"] for q in qs if q.get("divergence") is not None]
    return mean(divs) if divs else None


def terciles(vals: list[float]) -> tuple[float, float]:
    s = sorted(vals)
    n = len(s)
    return s[n // 3], s[2 * n // 3]


def stratify(key: str, n_target: int, seed: int) -> dict:
    recs = json.loads((RES / f"records_exp6b_{key}.json").read_text())
    inr = [r for r in recs if r.get("in_range") and r.get("char_start") is not None]

    with_lcr, no_lcr = [], []
    for r in inr:
        d = datapoint_divergence(r)
        item = dict(idx=r["idx"], doc_id=r["doc_id"], divergence=d,
                    n_lcr=(r.get("edit_meta") or {}).get("n_lcr", 0))
        (with_lcr if d is not None else no_lcr).append(item)

    t1, t2 = terciles([x["divergence"] for x in with_lcr])
    buckets = {"low": [], "mid": [], "high": []}
    for x in with_lcr:
        x["stratum"] = "low" if x["divergence"] <= t1 else ("high" if x["divergence"] > t2 else "mid")
        buckets[x["stratum"]].append(x)

    # allocate: reserve a proportional slice for the no-LCR bucket, split the rest into equal thirds
    frac_nolcr = len(no_lcr) / max(1, len(inr))
    n_nolcr = min(len(no_lcr), round(n_target * frac_nolcr))
    n_lcr = n_target - n_nolcr
    per = [n_lcr // 3 + (1 if i < n_lcr % 3 else 0) for i in range(3)]

    rng = random.Random(f"exp7-subsample|{key}|{seed}")
    chosen = []
    for name, k in zip(("low", "mid", "high"), per):
        pool = sorted(buckets[name], key=lambda z: z["idx"])
        chosen += rng.sample(pool, min(k, len(pool)))
    if n_nolcr:
        for x in no_lcr:
            x["stratum"] = "no_lcr"
        chosen += rng.sample(sorted(no_lcr, key=lambda z: z["idx"]), min(n_nolcr, len(no_lcr)))

    chosen = sorted(chosen, key=lambda z: z["idx"])
    out = dict(key=key, n=len(chosen), seed=seed, bounds=[round(t1, 4), round(t2, 4)],
               counts={s: sum(1 for c in chosen if c["stratum"] == s)
                       for s in ("low", "mid", "high", "no_lcr")},
               docs=[dict(idx=c["idx"], doc_id=c["doc_id"],
                          divergence=(round(c["divergence"], 4) if c["divergence"] is not None else None),
                          stratum=c["stratum"], n_lcr=c["n_lcr"]) for c in chosen])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("keys", nargs="*", default=KEYS)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    for key in (args.keys or KEYS):
        out = stratify(key, args.n, args.seed)
        (RES / f"exp7_subsample_{key}.json").write_text(json.dumps(out, ensure_ascii=False))
        print(f"[{key}] n={out['n']} bounds={out['bounds']} counts={out['counts']}")


if __name__ == "__main__":
    main()
