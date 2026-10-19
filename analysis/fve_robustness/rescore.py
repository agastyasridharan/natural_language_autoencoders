"""Generic AR re-score harness for the FVE-robustness arms (Step 0).

Re-runs ONLY the AR reconstruction on an edited explanation set, against the SAME
frozen gold activations, and recomputes FVE with the paper metric. Because gold is
fixed, the FVE denominator is identical to baseline, so dFVE is a clean function of
the per-point cosine change and is attributable per datapoint.

  python rescore.py <key> <arm>          # score out_<key>/edited_reads_<arm>.json
  python rescore.py <key> <arm> --null   # gate: score the ORIGINAL reads, assert it
                                         #   reproduces baseline cos.json (max|Δ|<1e-3)

Writes (non-null mode):
  out_<key>/cos_<arm>.json               {idx: cos_edit}
  out_<key>/fve_<arm>_<key>.json         arm vs baseline summary (+ bootstrap CI, Wilcoxon)
  out_<key>/records_<arm>_<key>.json     fills cos_orig/cos_edit/D_i/fve_i_* in the records

GPU: single AR model; run with gpurun GPU=2/3 (one visible GPU) once free.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/data/agastyas/nla_fve_ufw/code")
from repl_common import MODELS                               # noqa: E402
from nla_inference import NLACritic                          # noqa: E402
from huggingface_hub import snapshot_download                # noqa: E402

ROOT = Path("/data/agastyas/nla_fve_ufw")


def _score_reads(cfg, reads: dict, idxs: list[int], gold) -> dict:
    ar_dir = snapshot_download(cfg["ar"])
    critic = NLACritic(ar_dir, dtype=torch.bfloat16, device_map="auto")
    print(f"[rescore] AR loaded; mse_scale={critic.mse_scale:.3f}; scoring {len(idxs)}", flush=True)
    cos = {}
    for n, i in enumerate(idxs):
        _, c = critic.score(reads[str(i)], gold[i])
        cos[str(i)] = round(float(c), 5)
        if (n + 1) % 128 == 0 or n == len(idxs) - 1:
            print(f"  [{n+1}/{len(idxs)}] cos={c:+.3f}", flush=True)
    return cos


def _fve(idxs, gold, cosmap) -> tuple[float, np.ndarray, np.ndarray]:
    """Paper FVE on unit-normalized golds; returns (FVE, Di, Li) over idxs order."""
    G = gold[idxs].astype(np.float64)
    Ghat = G / np.linalg.norm(G, axis=1, keepdims=True)
    hbar = Ghat.mean(0)
    Di = ((Ghat - hbar) ** 2).sum(1)
    Li = np.array([2.0 * (1.0 - cosmap[str(i)]) for i in idxs])
    return float(1.0 - Li.sum() / Di.sum()), Di, Li


def _boot_ci(dcos_contrib_base, dcos_contrib_arm, Di, n=2000, seed=0):
    """Bootstrap 95% CI on dFVE by resampling docs (paired)."""
    rng = np.random.default_rng(seed)
    m = len(Di)
    out = []
    Dsum = Di.sum()
    for _ in range(n):
        s = rng.integers(0, m, m)
        fb = 1.0 - dcos_contrib_base[s].sum() / Di[s].sum()
        fa = 1.0 - dcos_contrib_arm[s].sum() / Di[s].sum()
        out.append(fa - fb)
    lo, hi = np.percentile(out, [2.5, 97.5])
    return float(lo), float(hi)


def run(key: str, arm: str, null: bool) -> None:
    cfg = MODELS[key]
    od = ROOT / f"out_{key}"
    manifest = json.loads((od / "manifest.json").read_text())
    gold = np.load(od / "gold.npy")
    base_cos = json.loads((od / "cos.json").read_text())

    if null:
        reads = json.loads((od / "av_reads.json").read_text())["reads"]
        idxs = [int(k) for k in base_cos]
        cos = _score_reads(cfg, reads, idxs, gold)
        d = np.array([abs(cos[str(i)] - base_cos[str(i)]) for i in idxs])
        passed = bool(d.max() < 1e-3)
        print(f"\n[null:{key}] max|Δcos|={d.max():.5f} mean|Δcos|={d.mean():.5f} "
              f"over {len(idxs)} rows -> {'PASS' if passed else 'FAIL'}", flush=True)
        if not passed:
            raise RuntimeError(f"null gate failed for {key}: max|Δcos|={d.max():.5f} >= 0.001")
        return

    reads_path = od / f"edited_reads_{arm}.json"
    reads = json.loads(reads_path.read_text())["reads"]
    # paired set: in_range rows non-empty in BOTH original and edited
    inr = {m["idx"] for m in manifest if m["in_range"]}
    idxs = sorted(i for i in (int(k) for k in base_cos)
                  if i in inr and reads.get(str(i), "").strip())
    dropped = len(base_cos) - len(idxs)
    cos = _score_reads(cfg, reads, idxs, gold)

    fve_arm, Di, Li_arm = _fve(idxs, gold, cos)
    fve_base, _, Li_base = _fve(idxs, gold, base_cos)
    dcos = np.array([cos[str(i)] - base_cos[str(i)] for i in idxs])
    lo, hi = _boot_ci(Li_base, Li_arm, Di)
    try:
        from scipy.stats import wilcoxon
        wp = float(wilcoxon(dcos).pvalue) if np.any(dcos != 0) else 1.0
    except Exception:
        wp = None

    (od / f"cos_{arm}.json").write_text(json.dumps(cos), encoding="utf-8")
    summary = dict(
        key=key, arm=arm, model=cfg["label"], layer=cfg["layer"],
        edited_reads_sha256=hashlib.sha256(reads_path.read_bytes()).hexdigest(),
        n_input_reads=len(reads),
        n_paired=len(idxs), n_dropped_empty=dropped,
        FVE_base=round(fve_base, 4), FVE_arm=round(fve_arm, 4),
        dFVE=round(fve_arm - fve_base, 4), dFVE_ci95=[round(lo, 4), round(hi, 4)],
        mean_cos_base=round(float(np.mean([base_cos[str(i)] for i in idxs])), 4),
        mean_cos_arm=round(float(np.mean([cos[str(i)] for i in idxs])), 4),
        mean_dcos=round(float(dcos.mean()), 4), wilcoxon_p=wp,
    )
    (od / f"fve_{arm}_{key}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # fill per-datapoint records
    rec_path = od / f"records_{arm}_{key}.json"
    if rec_path.exists():
        recs = json.loads(rec_path.read_text())
        Dmap = {i: float(Di[n]) for n, i in enumerate(idxs)}
        for r in recs:
            i = r["idx"]
            if i in Dmap:
                co, ce, dd = base_cos.get(str(i)), cos.get(str(i)), Dmap[i]
                r["cos_orig"], r["cos_edit"], r["D_i"] = co, ce, round(dd, 5)
                r["fve_i_orig"] = round(1 - 2 * (1 - co) / dd, 4) if co is not None else None
                r["fve_i_edit"] = round(1 - 2 * (1 - ce) / dd, 4) if ce is not None else None
        rec_path.write_text(json.dumps(recs, ensure_ascii=False), encoding="utf-8")

    print(f"\n[rescore:{key}:{arm}] === FVE_base={fve_base:.4f} FVE_arm={fve_arm:.4f} "
          f"dFVE={fve_arm-fve_base:+.4f} (95% [{lo:+.4f},{hi:+.4f}]) "
          f"mean_dcos={dcos.mean():+.4f} wilcoxon_p={wp} n={len(idxs)} dropped={dropped} ===",
          flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("key", choices=list(MODELS))
    ap.add_argument("arm")
    ap.add_argument("--null", action="store_true")
    a = ap.parse_args()
    run(a.key, a.arm, a.null)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
