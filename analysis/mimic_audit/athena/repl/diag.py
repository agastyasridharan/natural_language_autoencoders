"""Diagnostic: is the high gemma cosine real (anisotropic gold) or a bug (AR
collapse / wrong gold)? Run per model; compare gemma vs qwen as a normal control.

Tests:
  1. GOLD geometry: norms, shared-component ratio ||centroid||/mean||g||, cosine to
     centroid, pairwise cosine. Anisotropic gold -> high cosine floor is REAL.
  2. AR final-norm actually stripped? (must be Identity, else reconstruct is normed)
  3. AR collapse test: reconstruct 8 DELIBERATELY DIFFERENT texts; their pairwise
     cosine should be LOW if the AR is discriminative. ~1.0 => AR maps everything to
     one vector (would inflate cos_real/mimic/control alike).
"""
import sys, json, numpy as np, torch
sys.path.insert(0, "/data/agastyas/nla_mimic/code")
sys.path.insert(0, "/data/agastyas/nla_mimic/code/repl")
from nla_inference import NLACritic
from huggingface_hub import snapshot_download
from repl_common import MODELS

DIVERSE = [
    "A Python code snippet about sorting a list of integers.",
    "The weather in Paris is sunny and warm today.",
    "Quantum entanglement violates Bell inequalities.",
    "I love my dog; she is a golden retriever.",
    "This legal clause defines the limitation of liability.",
    "The stock market fell three percent on Tuesday.",
    "Recipe: whisk two eggs with flour and a pinch of salt.",
    "Once upon a time there was a dragon in a tall mountain.",
]


def gstats(G, tag):
    norms = np.linalg.norm(G, axis=1)
    cen = G.mean(0); cn = cen / (np.linalg.norm(cen) + 1e-9)
    Gn = G / norms[:, None]
    rng = np.random.default_rng(0)
    a = rng.integers(0, len(G), 3000); b = rng.integers(0, len(G), 3000)
    pc = np.array([float(Gn[i] @ Gn[j]) for i, j in zip(a, b) if i != j])
    s = np.linalg.svd(G[:min(128, len(G))], compute_uv=False)
    print(f"  [{tag}] N={len(G)} norm mean={norms.mean():.1f} (min {norms.min():.1f} max {norms.max():.1f})")
    print(f"  [{tag}] shared-component ||centroid||/mean||g|| = {np.linalg.norm(cen)/norms.mean():.3f}")
    print(f"  [{tag}] cos(gold,centroid) mean={float((Gn@cn).mean()):.3f}")
    print(f"  [{tag}] gold pairwise cosine mean={pc.mean():.3f} p10={np.percentile(pc,10):.3f} p90={np.percentile(pc,90):.3f}")
    print(f"  [{tag}] raw top-3 singular-value energy frac = {(s[:3]**2/(s**2).sum()).round(3)}")


def main():
    key = sys.argv[1]
    cfg = MODELS[key]
    outdir = f"/data/agastyas/nla_mimic/out_{key}"
    gold = np.load(outdir + "/gold.npy")
    mani = json.load(open(outdir + "/manifest.json"))
    valid = [i for i, m in enumerate(mani) if m["in_range"]]
    G = gold[valid].astype(np.float32)
    print(f"==== {key} ====", flush=True)
    gstats(G, "gold")

    critic = NLACritic(snapshot_download(cfg["ar"]), dtype=torch.bfloat16, device_map="auto")
    inner = critic.backbone.model
    nrm = getattr(inner, "norm", None)
    print(f"  [AR] backbone.model.norm = {type(nrm).__name__} (Identity == stripped OK)")
    print(f"  [AR] mse_scale={critic.mse_scale:.2f}", flush=True)

    R = np.array([critic.reconstruct(t).numpy() for t in DIVERSE], dtype=np.float32)
    Rn = R / np.linalg.norm(R, axis=1, keepdims=True)
    pc = [float(Rn[i] @ Rn[j]) for i in range(len(R)) for j in range(len(R)) if i < j]
    print(f"  [AR] reconstruction pairwise cosine (8 diverse texts): "
          f"mean={np.mean(pc):.3f} min={np.min(pc):.3f} max={np.max(pc):.3f}")
    # determinism + cos of each recon to gold[0]
    g0 = G[0] / np.linalg.norm(G[0])
    print(f"  [AR] cos(recon_i, gold[0]) = {[round(float(rn@g0),3) for rn in Rn]}")
    # real read vs a clearly-wrong read against the same gold (sharper control)
    rec_diverse_vs_g0 = [round(float((R[i]/np.linalg.norm(R[i])) @ g0), 3) for i in range(len(R))]
    print(f"  [AR] spread of cos(diverse_text, gold[0]) = min {min(rec_diverse_vs_g0)} max {max(rec_diverse_vs_g0)}")


if __name__ == "__main__":
    main()
