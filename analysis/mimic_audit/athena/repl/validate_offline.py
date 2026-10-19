#!/usr/bin/env python3
"""Offline validation of a claim-ablation run -- NO GPU, NO model.

Re-analyzes the saved artifacts (records.jsonl + vectors/*.npz) of a finished run
to check the metric's free parameters and controls before we trust the headline:

  1. null sanity        random-direction ablation must not suppress claims
  2. spec-margin sweep  how the grounded/entangled split moves with --spec-margin
  3. null-pct stability  is the 95th-pct-of-R=10 threshold stable / verdict-robust?
  4. entanglement       cos(delta_i, delta_j): is the 29.9% entangled geometric?
  5. dlogp vs dMSE      Pearson + the disagreements (where causal != AR opinion)

Everything here comes from fields the run already saved; nothing is recomputed
through a model. Usage:

  python repl/validate_offline.py runs/qwen7b-<stamp>-fineweb
"""
import json
import sys
from pathlib import Path

import numpy as np


def load(run: Path):
    recs = [json.loads(l) for l in (run / "records.jsonl").read_text().splitlines() if l.strip()]
    snips = []  # (idx, i, snippet_result dict) for LOCATED snippets only
    for r in recs:
        if r.get("status") != "ok":
            continue
        for s in r["snippet_results"]:
            if s.get("located"):
                snips.append((r["idx"], s["i"], s))
    return recs, snips


def pctl(a, p):
    return float(np.percentile(a, p)) if len(a) else float("nan")


def hr(title):
    print("\n" + "=" * 74 + f"\n {title}\n" + "=" * 74)


# ── 1. null sanity ───────────────────────────────────────────────────────────
def null_sanity(snips):
    hr("1. NULL SANITY  (random-direction ablation must not suppress claims)")
    null_all = np.array([x for _, _, s in snips for x in s["null_dlogps"]], float)
    dlogp = np.array([s["dlogp"] for _, _, s in snips], float)
    gnd = np.array([s["verdict"] == "grounded" for _, _, s in snips])
    print(f"  null samples: {len(null_all)}  ({len(snips)} snippets x ~10 random dirs)")
    print(f"  null dlogp   mean={null_all.mean():+.3f}  median={np.median(null_all):+.3f}  "
          f"sd={null_all.std():.3f}")
    print(f"  null dlogp   95th={pctl(null_all,95):+.3f}  99th={pctl(null_all,99):+.3f}  "
          f"max={null_all.max():+.3f}")
    # a random direction almost never produces a "grounded-sized" effect
    g_med = float(np.median(dlogp[gnd])) if gnd.any() else float("nan")
    big = float((null_all > g_med).mean()) if not np.isnan(g_med) else float("nan")
    print(f"  grounded dlogp median = {g_med:+.2f} nats;  "
          f"fraction of NULL samples that big = {big*100:.2f}%")
    pos1 = float((null_all > 1.0).mean())
    print(f"  fraction of null samples > +1.0 nat = {pos1*100:.2f}%  "
          f"(should be tiny: random dirs are ~orthogonal to h)")
    # per-snippet: nulls above that snippet's own 95th threshold ~ 5% by construction
    above = []
    for _, _, s in snips:
        nd = np.array(s["null_dlogps"], float)
        above.append(float((nd > s["null_thr"]).mean()))
    print(f"  per-snippet mean(null > its own null_thr) = {np.mean(above)*100:.1f}%  "
          f"(~5% expected; sanity on the threshold)")
    verdict = "PASS" if (pos1 < 0.02 and (np.isnan(big) or big < 0.01)) else "CHECK"
    print(f"  -> {verdict}: random ablation does not fabricate grounding.")
    return dict(null_mean=float(null_all.mean()), null_p95=pctl(null_all, 95),
                null_p99=pctl(null_all, 99), null_max=float(null_all.max()),
                grounded_median=g_med, frac_null_gt_grounded_median=big,
                frac_null_gt_1nat=pos1)


# ── 2. spec-margin sweep ─────────────────────────────────────────────────────
def spec_sweep(snips, margins=(1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 5.0)):
    hr("2. SPEC-MARGIN SWEEP  (grounded/entangled boundary; confab is margin-free)")
    n = len(snips)
    depends = np.array([s["dlogp"] > s["null_thr"] for _, _, s in snips])
    dlogp = np.array([s["dlogp"] for _, _, s in snips], float)
    coff = np.array([s["col_off_max"] for _, _, s in snips], float)
    n_confab = int((~depends).sum())
    print(f"  located snippets: {n};  confabulated (margin-independent): "
          f"{n_confab} ({n_confab/n*100:.1f}%)")
    print(f"  {'margin':>7} | {'grounded':>9} {'entangled':>10} {'confab':>8}")
    rows = {}
    for m in margins:
        # specific if its own ablation dominates the column by factor m
        specific = (coff <= 0) | (dlogp > m * coff)
        gnd = depends & specific
        ent = depends & ~specific
        rows[m] = dict(grounded=int(gnd.sum()), entangled=int(ent.sum()), confab=n_confab,
                       grounded_rate=float(gnd.mean()), entangled_rate=float(ent.mean()))
        star = "  <- current" if abs(m - 1.5) < 1e-9 else ""
        print(f"  {m:>7.2f} | {gnd.sum():>4} {gnd.mean()*100:>5.1f}%  "
              f"{ent.sum():>4} {ent.mean()*100:>5.1f}%  {n_confab:>4} "
              f"{n_confab/n*100:>4.1f}%{star}")
    print("  -> confab rate is fixed; the margin only trades grounded <-> entangled.")
    return rows


# ── 3. null-percentile / R stability ─────────────────────────────────────────
def null_stability(snips):
    hr("3. NULL-PERCENTILE & R STABILITY  (does 'depends' flip with the threshold?)")
    base = np.array([s["dlogp"] > s["null_thr"] for _, _, s in snips])  # 95th of R=10
    for p in (90, 95, 99):
        dep = []
        for _, _, s in snips:
            nd = np.array(s["null_dlogps"], float)
            dep.append(s["dlogp"] > pctl(nd, p))
        dep = np.array(dep)
        flips = int((dep != base).sum())
        print(f"  null pct={p:>2}: depends={dep.sum():>4}/{len(dep)}  "
              f"flips vs 95th={flips} ({flips/len(dep)*100:.1f}%)")
    # use max of R (most conservative) and min, to bound R sensitivity
    dep_max = np.array([s["dlogp"] > max(s["null_dlogps"]) for _, _, s in snips])
    print(f"  threshold=max(null) (most conservative): depends="
          f"{dep_max.sum()}/{len(dep_max)}  flips vs 95th={int((dep_max!=base).sum())}")
    print(f"  -> the 'depends' decision (confab boundary) is {'robust' if (dep_max!=base).mean()<0.05 else 'sensitive'} to the null threshold.")
    return dict(base_depends=int(base.sum()))


# ── 4. entanglement geometry ─────────────────────────────────────────────────
def entanglement(run, recs, snips):
    hr("4. ENTANGLEMENT GEOMETRY  (is the entangled class shared AR directions?)")
    vd = run / "vectors"
    # per (idx,i) -> max cosine of delta_i with any sibling delta_j (j!=i)
    by_idx = {}
    for r in recs:
        if r.get("status") == "ok":
            by_idx[r["idx"]] = r
    rows = []  # (verdict, max_sib_cos, spec_ratio)
    n_missing = 0
    for idx, r in by_idx.items():
        f = vd / f"{idx}.npz"
        if not f.is_file():
            n_missing += 1
            continue
        z = np.load(f, allow_pickle=True)
        D = z["deltas"].astype(np.float64)          # (n, d)
        nrm = np.linalg.norm(D, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        U = D / nrm
        C = U @ U.T                                  # cosine matrix
        np.fill_diagonal(C, -1.0)
        for s in r["snippet_results"]:
            if not s.get("located"):
                continue
            i = s["i"]
            sib = float(C[i].max()) if C.shape[0] > 1 else float("nan")
            rows.append((s["verdict"], sib, s.get("spec_ratio")))
    if n_missing:
        print(f"  (note: {n_missing} activations missing vectors/*.npz)")
    vts = ("grounded", "entangled", "confabulated")
    print(f"  {'verdict':>13} | {'n':>4} {'mean max-sibling cos(Δi,Δj)':>30} {'median':>8}")
    out = {}
    for v in vts:
        cs = np.array([c for (vv, c, _) in rows if vv == v and not np.isnan(c)])
        if len(cs):
            print(f"  {v:>13} | {len(cs):>4} {cs.mean():>30.3f} {np.median(cs):>8.3f}")
            out[v] = dict(n=len(cs), mean_max_sib_cos=float(cs.mean()),
                          median_max_sib_cos=float(np.median(cs)))
    ge = [c for (vv, c, _) in rows if vv == "entangled" and not np.isnan(c)]
    gg = [c for (vv, c, _) in rows if vv == "grounded" and not np.isnan(c)]
    if ge and gg:
        print(f"  -> entangled max-sibling-cos ({np.mean(ge):.3f}) vs grounded "
              f"({np.mean(gg):.3f}): "
              f"{'HIGHER (geometric overlap = granularity artifact)' if np.mean(ge) > np.mean(gg)+0.02 else 'similar (not purely geometric)'}")
    return out


# ── 5. dlogp vs AR dMSE ──────────────────────────────────────────────────────
def dlogp_vs_dmse(snips):
    hr("5. Δlogp vs AR ΔMSE  (independent cross-check + the disagreements)")
    dl = np.array([s["dlogp"] for _, _, s in snips], float)
    dm = np.array([s["dmse"] for _, _, s in snips], float)
    ok = ~(np.isnan(dl) | np.isnan(dm))
    r = float(np.corrcoef(dl[ok], dm[ok])[0, 1])
    print(f"  Pearson r(Δlogp, ΔMSE) = {r:+.3f}  over {ok.sum()} snippets")
    # disagreements: causal test says grounded but AR says removing claim didn't hurt
    gnd = np.array([s["verdict"] == "grounded" for _, _, s in snips])
    conf = np.array([s["verdict"] == "confabulated" for _, _, s in snips])
    ar_blind = gnd & (dm <= 0)                      # grounded by Δlogp, ΔMSE indifferent
    ar_flag = conf & (dm > np.median(dm[gnd]) if gnd.any() else dm > 0)
    print(f"  grounded by Δlogp but ΔMSE<=0 (AR-blind):   {int(ar_blind.sum())} "
          f"({ar_blind.mean()*100:.1f}% of located)")
    print(f"  confab by Δlogp but large ΔMSE (AR-only):   {int(ar_flag.sum())}")
    print("  these are exactly the cases the paper's ΔMSE heuristic alone would mislabel.")
    # a few AR-blind examples
    ex = [(idx, s) for (idx, _, s), b in zip(snips, ar_blind) if b][:4]
    for idx, s in ex:
        print(f"    #{idx} Δlogp={s['dlogp']:+.1f} ΔMSE={s['dmse']:+.3f} :: "
              f"{s['text'][:84]}")
    return dict(pearson=r, n_ar_blind=int(ar_blind.sum()), n_ar_flag=int(ar_flag.sum()))


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    run = Path(sys.argv[1])
    if not (run / "records.jsonl").is_file():
        print(f"no records.jsonl in {run}")
        return 2
    recs, snips = load(run)
    print(f"run: {run.name}\nlocated snippets: {len(snips)} from {sum(r.get('status')=='ok' for r in recs)} activations")
    summary = dict(run=run.name, n_located=len(snips))
    summary["null_sanity"] = null_sanity(snips)
    summary["spec_sweep"] = spec_sweep(snips)
    summary["null_stability"] = null_stability(snips)
    summary["entanglement"] = entanglement(run, recs, snips)
    summary["dlogp_vs_dmse"] = dlogp_vs_dmse(snips)
    out = run / "validation_offline.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
