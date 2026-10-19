"""Recompute the audit in FVE (Fraction of Variance Explained) — the paper's
metric: FVE = 1 - E||p - h||^2 / E||h - hbar||^2 on unit-normalized activations.
hbar = mean activation; FVE=0 == "predict the mean", FVE=1 == perfect.

Raw cosine is SATURATED for gemma: activations are near-parallel (l2_norm ~60k,
93-95% of unit energy lies in the shared mean), so cos ~0.99 even for a mismatched
read. MSE=2(1-cos) is just a monotone re-skin of that. FVE removes the shared mean
so it measures the part that carries content. This matches the paper headline
("0.6-0.8 FVE") and what looks "medium" on Neuronpedia.

Per position i: L_i = 2(1-cos_i)  (matches NP mse == 2(1-cos), verified on llama),
D_i = ||hhat_i - hbar||^2.  Aggregate FVE = 1 - sum(L_i)/sum(D_i)  (sum-based, exact).
Per-position fve_i = 1 - L_i/D_i (for the distribution/row view).
Emits fve_<key>.json with summary + by-genre + by-band(stored cosine) + per-row.
"""
import sys, json, numpy as np

KEY = sys.argv[1]
BASE = "/data/agastyas/nla_mimic"
OUT = f"{BASE}/out" if KEY == "llama70b" else f"{BASE}/out_{KEY}"

gold = np.load(f"{OUT}/gold.npy").astype(np.float64)
mani = json.load(open(f"{OUT}/manifest.json"))
res = json.load(open(f"{OUT}/results.json"))
rows = res["rows"]

def is_valid(m):
    ok = m.get("in_range", True)
    if "anchor_ok" in m:
        ok = ok and m["anchor_ok"]
    return ok
valid = [i for i, m in enumerate(mani) if is_valid(m)]
if len(valid) != len(rows):
    valid = list(range(len(rows))) if len(gold) >= len(rows) else valid
assert len(valid) == len(rows), f"{KEY}: {len(valid)} valid vs {len(rows)} rows"

G = gold[valid]
Ghat = G / np.linalg.norm(G, axis=1, keepdims=True)
hbar = Ghat.mean(0)
Di = ((Ghat - hbar) ** 2).sum(1)
D = float(Di.mean())
norm_share = float(np.linalg.norm(hbar) ** 2)

def agg_fve(coskey, idxs):
    num = sum(2.0 * (1.0 - rows[j][coskey]) for j in idxs)
    den = sum(float(Di[j]) for j in idxs)
    return (1.0 - num / den) if den > 0 else float("nan")

ALL = list(range(len(rows)))
fr, fm, fc = agg_fve("cos_real", ALL), agg_fve("cos_mimic", ALL), agg_fve("cos_control", ALL)
mean_cos = lambda k: sum(r[k] for r in rows) / len(rows)
mcr, mcm = mean_cos("cos_real"), mean_cos("cos_mimic")

# per-position fve (global per-position D_i)
for j, r in enumerate(rows):
    r["fve_real"] = round(1 - 2 * (1 - r["cos_real"]) / Di[j], 4)
    r["fve_mimic"] = round(1 - 2 * (1 - r["cos_mimic"]) / Di[j], 4)
    r["fve_control"] = round(1 - 2 * (1 - r["cos_control"]) / Di[j], 4)
    r["Di"] = round(float(Di[j]), 5)
win = sum(1 for r in rows if r["fve_real"] > r["fve_mimic"])

# by genre
themes = sorted({r["theme"] for r in rows})
genres = []
for th in themes:
    sub = [j for j, r in enumerate(rows) if r["theme"] == th]
    genres.append(dict(theme=th, n=len(sub),
                       fve_real=round(agg_fve("cos_real", sub), 3),
                       fve_mimic=round(agg_fve("cos_mimic", sub), 3),
                       dfve=round(agg_fve("cos_real", sub) - agg_fve("cos_mimic", sub), 3)))
genres.sort(key=lambda g: -g["dfve"])

# by stored-cosine band (difficulty)
BANDS = [("<0.60", -1, 0.60), ("0.60-0.80", 0.60, 0.80), ("0.80-0.90", 0.80, 0.90),
         ("0.90-0.95", 0.90, 0.95), (">=0.95", 0.95, 2.0)]
bands = []
for name, lo, hi in BANDS:
    sub = [j for j, r in enumerate(rows) if lo <= r["cosine_stored"] < hi]
    if sub:
        bands.append(dict(name=name, n=len(sub),
                          fve_real=round(agg_fve("cos_real", sub), 3),
                          fve_mimic=round(agg_fve("cos_mimic", sub), 3),
                          dfve=round(agg_fve("cos_real", sub) - agg_fve("cos_mimic", sub), 3)))

out = dict(
    key=KEY, n=len(rows), D_variance=round(D, 5), norm_share_in_mean=round(norm_share, 4),
    summary=dict(
        FVE_real=round(fr, 4), FVE_mimic=round(fm, 4), FVE_control=round(fc, 4),
        dFVE=round(fr - fm, 4),
        activation_specific_frac=round((fr - fm) / fr, 4) if fr > 0 else None,
        mse_ratio_mimic_over_real=round((1 - mcm) / (1 - mcr), 3),  # D-independent
        mean_cos_real=round(mcr, 4), mean_cos_mimic=round(mcm, 4),
        mean_cos_control=round(mean_cos("cos_control"), 4),
        real_beats_mimic_FVE=f"{win}/{len(rows)}"),
    genres=genres, bands=bands,
    rows=[dict(idx=r["idx"], fve_real=r["fve_real"], fve_mimic=r["fve_mimic"],
               fve_control=r["fve_control"], Di=r["Di"]) for r in rows])
json.dump(out, open(f"{OUT}/fve_{KEY}.json", "w"), indent=2)
print(f"[{KEY}] n={len(rows)} norm_share={norm_share:.3f} D={D:.4f} | "
      f"FVE real={fr:.3f} mimic={fm:.3f} ctl={fc:.3f} dFVE={fr-fm:+.3f} "
      f"act_frac={(fr-fm)/fr:.2f} mse_ratio={(1-mcm)/(1-mcr):.2f} win={win}/{len(rows)} "
      f"(raw dcos={mcr-mcm:+.3f})")
