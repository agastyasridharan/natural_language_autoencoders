"""De-biased (standardized) recompute of the full audit — the robust metric
(Timkey & van Schijndel 2021): per-dim z-score with gold mu/sd removes the rogue
/ massive-activation dims, so cosine measures content, not 1-2 huge dims.

Emits std_full_<key>.json: per-position + by-genre + by-band standardized cosine
for real / mimic / control, plus raw cosine (reference) and dataset standardized
R2 (FVE). Control should collapse to ~0 everywhere (no saturation); the question
is whether real > mimic and the code/math concentration SURVIVE de-biasing.
"""
import sys, json, glob, numpy as np, torch
sys.path.insert(0, "/data/agastyas/nla_mimic/code")
sys.path.insert(0, "/data/agastyas/nla_mimic/code/repl")
from nla_inference import NLACritic
from repl_common import MODELS

key = sys.argv[1]
AR_REPO = {"llama70b": "kitft/Llama-3.3-70B-NLA-L53-ar"}.get(key) or MODELS[key]["ar"]
BASE = "/data/agastyas/nla_mimic"
OUT = f"{BASE}/out" if key == "llama70b" else f"{BASE}/out_{key}"

def local_snapshot(repo):
    base = "/data/hf_cache/models--" + repo.replace("/", "--") + "/snapshots/*"
    snaps = sorted(glob.glob(base))
    assert snaps, f"no cached snapshot for {repo}"
    return snaps[-1]

gold = np.load(f"{OUT}/gold.npy").astype(np.float64)
mani = json.load(open(f"{OUT}/manifest.json"))
rows = json.load(open(f"{OUT}/results.json"))["rows"]
inr = [i for i, m in enumerate(mani) if m.get("in_range", True)]
G = gold[inr]
mu = G.mean(0); sd = G.std(0) + 1e-9
gold_for = (lambda r, j: gold[r["idx"]]) if len(gold) > max(r["idx"] for r in rows) \
    else (lambda r, j: gold[j])

# texts: replication models embed them in results rows; llama keeps them in
# data/run_input.json (reads were reused from the corpus).
if "real_text" in rows[0]:
    real_txt = [r["real_text"] for r in rows]
    mim_txt = [r["mimic_text"] for r in rows]
else:
    ri = {it["idx"]: it for it in json.load(open(f"{BASE}/data/run_input.json"))}
    real_txt = [ri[r["idx"]]["real_text"] for r in rows]
    mim_txt = [ri[r["idx"]]["mimic_text"] for r in rows]

critic = NLACritic(local_snapshot(AR_REPO), dtype=torch.bfloat16, device_map="auto")
rec = lambda t: critic.reconstruct(t).numpy().astype(np.float64)
print(f"[{key}] reconstructing {len(rows)} real + {len(rows)} mimic ...", flush=True)
p_real = [rec(t) for t in real_txt]
p_mim = [rec(t) for t in mim_txt]
p_ctl = [p_real[(j + 1) % len(rows)] for j in range(len(rows))]
golds = [gold_for(r, j) for j, r in enumerate(rows)]

z = lambda x: (x - mu) / sd
def cos(a, b): return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

rows_out = []
for j, r in enumerate(rows):
    g = golds[j]
    rows_out.append(dict(
        idx=r["idx"], theme=r["theme"], cosine_stored=r["cosine_stored"],
        raw_real=r["cos_real"], raw_mimic=r["cos_mimic"], raw_control=r["cos_control"],
        std_real=round(cos(z(p_real[j]), z(g)), 4),
        std_mimic=round(cos(z(p_mim[j]), z(g)), 4),
        std_control=round(cos(z(p_ctl[j]), z(g)), 4)))

def magg(field, idxs): return round(sum(rows_out[j][field] for j in idxs) / len(idxs), 4)
ALL = list(range(len(rows_out)))

themes = sorted({r["theme"] for r in rows_out})
genres = []
for th in themes:
    sub = [j for j, r in enumerate(rows_out) if r["theme"] == th]
    genres.append(dict(theme=th, n=len(sub), std_real=magg("std_real", sub),
                       std_mimic=magg("std_mimic", sub),
                       std_delta=round(magg("std_real", sub) - magg("std_mimic", sub), 4)))
genres.sort(key=lambda g: -g["std_delta"])

BANDS = [("<0.60", -1, 0.60), ("0.60-0.80", 0.60, 0.80), ("0.80-0.90", 0.80, 0.90),
         ("0.90-0.95", 0.90, 0.95), (">=0.95", 0.95, 2.0)]
bands = []
for name, lo, hi in BANDS:
    sub = [j for j, r in enumerate(rows_out) if lo <= r["cosine_stored"] < hi]
    if sub:
        bands.append(dict(name=name, n=len(sub), std_real=magg("std_real", sub),
                          std_mimic=magg("std_mimic", sub),
                          std_delta=round(magg("std_real", sub) - magg("std_mimic", sub), 4)))

def r2(preds):
    num = sum(float(((z(p) - z(g)) ** 2).sum()) for p, g in zip(preds, golds))
    den = sum(float((z(g) ** 2).sum()) for g in golds)
    return round(1 - num / den, 4)

summary = dict(
    raw=dict(real=magg("raw_real", ALL), mimic=magg("raw_mimic", ALL), control=magg("raw_control", ALL)),
    std_cos=dict(real=magg("std_real", ALL), mimic=magg("std_mimic", ALL), control=magg("std_control", ALL)),
    std_R2=dict(real=r2(p_real), mimic=r2(p_mim), control=r2(p_ctl)),
    std_delta=round(magg("std_real", ALL) - magg("std_mimic", ALL), 4),
    std_win=f"{sum(1 for r in rows_out if r['std_real']>r['std_mimic'])}/{len(rows_out)}")
out = dict(key=key, n=len(rows_out), summary=summary, genres=genres, bands=bands, rows=rows_out)
json.dump(out, open(f"{OUT}/std_full_{key}.json", "w"), indent=2)

print(f"\n==== {key} (standardized / de-biased) ====")
print(f"  std_cos   real={summary['std_cos']['real']:.3f} mimic={summary['std_cos']['mimic']:.3f} "
      f"control={summary['std_cos']['control']:.3f}  Δ={summary['std_delta']:+.3f} win={summary['std_win']}")
print(f"  std_R2    real={summary['std_R2']['real']:.3f} mimic={summary['std_R2']['mimic']:.3f} "
      f"control={summary['std_R2']['control']:.3f}")
print(f"  (raw cos  real={summary['raw']['real']:.3f} mimic={summary['raw']['mimic']:.3f} "
      f"control={summary['raw']['control']:.3f})")
print(f"  top-Δ genres:   {[(g['theme'], g['std_delta']) for g in genres[:4]]}")
print(f"  bottom-Δ genres:{[(g['theme'], g['std_delta']) for g in genres[-4:]]}")
print("STD_FULL_DONE", flush=True)
