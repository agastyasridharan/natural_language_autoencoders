"""Phase 2 (llama70b): reconstruct each gold activation from real vs mimic text.

Loads the AR (kitft/Llama-3.3-70B-NLA-L53-ar) across the visible GPUs
(device_map="auto"; needs >=2 H100 — the ~47B AR does not fit one). For every
aligned position it scores three texts against the gold extracted in phase 1:
  cos_real    = AR.score(real AV read,  gold)
  cos_mimic   = AR.score(v2_3 mimic,    gold)     <-- the experiment
  cos_control = AR.score(another item's real read, gold)   <-- chance floor

VALIDATION GATE: cos_real must reproduce the stored Neuronpedia cosine (same
gold, same AR, same text -> deterministic). If mean|cos_real - cosine_stored| is
small and correlation high, the whole pipeline is proven; only then are the
cos_mimic numbers trustworthy.

Reads /data/agastyas/nla_mimic/out/{gold.npy,manifest.json} + data/run_input.json.
Writes results.json there. Run via direct SSH with the HF proxy/cache env.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/data/agastyas/nla_mimic/code")
from nla_inference import NLACritic  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402

OUTDIR = Path("/data/agastyas/nla_mimic/out")
INPUT = "/data/agastyas/nla_mimic/data/run_input.json"
AR_REPO = "kitft/Llama-3.3-70B-NLA-L53-ar"


def pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return num / den if den else float("nan")


def main() -> int:
    gold = np.load(OUTDIR / "gold.npy")
    manifest = json.loads((OUTDIR / "manifest.json").read_text())
    texts = {it["idx"]: it for it in json.loads(Path(INPUT).read_text(encoding="utf-8"))}
    print(f"[score] {len(manifest)} rows; torch {torch.__version__} "
          f"gpus={torch.cuda.device_count()}", flush=True)

    ar_dir = snapshot_download(AR_REPO)
    critic = NLACritic(ar_dir, dtype=torch.bfloat16, device_map="auto")
    print(f"[score] AR loaded; mse_scale={critic.mse_scale:.3f} "
          f"device={critic.device}", flush=True)

    valid = [i for i, m in enumerate(manifest) if m["in_range"] and m["token_match"]]
    print(f"[score] {len(valid)}/{len(manifest)} rows aligned (in_range & token_match)", flush=True)

    rows = []
    for j, i in enumerate(valid):
        m = manifest[i]
        g = gold[i]
        real = texts[m["idx"]]["real_text"]
        mimic = texts[m["idx"]]["mimic_text"]
        ctl = texts[manifest[valid[(j + 1) % len(valid)]]["idx"]]["real_text"]
        _, c_real = critic.score(real, g)
        _, c_mim = critic.score(mimic, g)
        _, c_ctl = critic.score(ctl, g)
        rows.append({"idx": m["idx"], "theme": m["theme"], "position": m["position"],
                     "cosine_stored": m["cosine_stored"],
                     "cos_real": round(c_real, 4), "cos_mimic": round(c_mim, 4),
                     "cos_control": round(c_ctl, 4),
                     "delta": round(c_real - c_mim, 4)})
        if (j + 1) % 16 == 0 or j == len(valid) - 1:
            print(f"  [{j+1}/{len(valid)}] {m['theme']:15s} "
                  f"real={c_real:+.3f} mimic={c_mim:+.3f} ctl={c_ctl:+.3f} "
                  f"(stored {m['cosine_stored']:+.3f})", flush=True)

    # validation gate: recomputed cos_real vs stored Neuronpedia cosine
    sr = [r["cos_real"] for r in rows]
    st = [r["cosine_stored"] for r in rows]
    abserr = [abs(a - b) for a, b in zip(sr, st)]
    val = {"n": len(rows), "mean_abs_err": round(sum(abserr) / len(abserr), 4),
           "max_abs_err": round(max(abserr), 4), "pearson": round(pearson(sr, st), 4)}

    deltas = [r["delta"] for r in rows]
    summary = {
        "mean_cos_real": round(sum(sr) / len(sr), 4),
        "mean_cos_mimic": round(sum(r["cos_mimic"] for r in rows) / len(rows), 4),
        "mean_cos_control": round(sum(r["cos_control"] for r in rows) / len(rows), 4),
        "mean_delta": round(sum(deltas) / len(deltas), 4),
        "median_delta": round(sorted(deltas)[len(deltas) // 2], 4),
        "real_beats_mimic": f"{sum(1 for d in deltas if d > 0)}/{len(deltas)}",
        "real_beats_control": f"{sum(1 for r in rows if r['cos_real'] > r['cos_control'])}/{len(rows)}",
    }
    out = {"validation": val, "summary": summary, "rows": rows}
    (OUTDIR / "results.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n[score] VALIDATION cos_real vs stored: mean|err|={val['mean_abs_err']} "
          f"max={val['max_abs_err']} pearson={val['pearson']}", flush=True)
    print(f"[score] RESULT mean cos_real={summary['mean_cos_real']} "
          f"cos_mimic={summary['mean_cos_mimic']} control={summary['mean_cos_control']} "
          f"| mean Δ={summary['mean_delta']} real>mimic {summary['real_beats_mimic']}", flush=True)
    print(f"[score] wrote {OUTDIR/'results.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
