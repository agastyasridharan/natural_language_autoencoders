"""Rogue-dimension analysis (Timkey & van Schijndel 2021, EMNLP): is gemma's
saturated cosine driven by a few outlier dimensions, and does standardization
(their recommended fix) remove it?

For each model's gold activations: how concentrated is the mean / variance in a
few dims, and what is the mean pairwise cosine under raw / mean-centered /
standardized (z-scored) representations. If standardized pairwise cosine ~ 0,
the residual stream is isotropic once the rogue dims are removed and the high
raw cosine was an artifact of those dims.
"""
import sys, json, numpy as np

key = sys.argv[1]
BASE = "/data/agastyas/nla_mimic"
OUT = f"{BASE}/out" if key == "llama70b" else f"{BASE}/out_{key}"
gold = np.load(f"{OUT}/gold.npy").astype(np.float64)
mani = json.load(open(f"{OUT}/manifest.json"))
valid = [i for i, m in enumerate(mani)
         if m.get("in_range", True) and m.get("anchor_ok", True)]
G = gold[valid]
N, d = G.shape
mu = G.mean(0); sd = G.std(0)
om = np.argsort(-np.abs(mu)); ov = np.argsort(-(sd ** 2))

def pw_cos(X):
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True).clip(1e-12)
    S = Xn @ Xn.T
    return float((S.sum() - N) / (N * (N - 1)))

raw, cen, std = pw_cos(G), pw_cos(G - mu), pw_cos((G - mu) / (sd + 1e-9))
mu2 = mu ** 2; var = sd ** 2

print(f"\n==== {key}  (N={N}, d={d}) ====")
print(f"  gold norm mean={np.linalg.norm(G,axis=1).mean():.0f}")
print(f"  MEAN-vector energy concentration:")
for k in (1, 2, 3, 10):
    print(f"    top-{k:>2} dims hold {mu2[om[:k]].sum()/mu2.sum():.3f} of ||mean||^2"
          f"   (dims {om[:k].tolist()})")
print(f"  CENTERED-variance concentration:")
for k in (1, 2, 3, 10, 50):
    print(f"    top-{k:>2} dims hold {var[ov[:k]].sum()/var.sum():.3f} of total variance")
print(f"  largest-|mean| dims  (idx, mean, std):  "
      f"{[(int(i), round(mu[i],0), round(sd[i],0)) for i in om[:3]]}")
print(f"  MEAN PAIRWISE COSINE:  raw={raw:.3f}   mean-centered={cen:.3f}   "
      f"standardized(z)={std:.3f}")
print(f"    -> raw cosine floor a mismatched read can't go below ~ {raw:.2f}; "
      f"after de-biasing it drops to {std:.2f}")
