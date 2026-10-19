"""Phase 2 (any model): reconstruct each gold from the generated real AV read vs
the reused text-only mimic vs a control, via the model's AR.

cos_real    = AR.score(generated AV read, gold)
cos_mimic   = AR.score(reused v2_3 mimic, gold)
cos_control = AR.score(another item's AV read, gold)

No independent stored cosine exists for the non-llama models (reads are generated
here, not from Neuronpedia), so the "validation" gate is null; instead we report
injection health (CJK / missing-<explanation> counts) and the anchor success rate,
plus the correlation of cos_real with llama's content difficulty as a sanity check.
Writes results.json to out_<key>/ (rows embed real/mimic text for the viewer).

Usage: python score_model.py <key>
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/data/agastyas/nla_mimic/code")
from repl_common import MODELS  # noqa: E402
from nla_inference import NLACritic  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402

ROOT = Path("/data/agastyas/nla_mimic")


def pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return num / den if den else float("nan")


def main() -> int:
    key = sys.argv[1]
    cfg = MODELS[key]
    outdir = ROOT / f"out_{key}"
    gold = np.load(outdir / "gold.npy")
    manifest = json.loads((outdir / "manifest.json").read_text())
    av = json.loads((outdir / "av_reads.json").read_text())
    reads = av["reads"]
    print(f"[score:{key}] {len(manifest)} rows; ar={cfg['ar']}", flush=True)

    ar_dir = snapshot_download(cfg["ar"])
    critic = NLACritic(ar_dir, dtype=torch.bfloat16, device_map="auto")
    print(f"[score:{key}] AR loaded; mse_scale={critic.mse_scale:.3f}", flush=True)

    valid = [i for i, m in enumerate(manifest)
             if m["in_range"] and reads.get(str(m["idx"]), "").strip()]
    print(f"[score:{key}] {len(valid)}/{len(manifest)} rows valid (in_range & read non-empty)", flush=True)

    rows = []
    for j, i in enumerate(valid):
        m = manifest[i]
        g = gold[i]
        real = reads[str(m["idx"])]
        mimic = m["mimic_text"]
        ctl = reads[str(manifest[valid[(j + 1) % len(valid)]]["idx"])]
        _, c_real = critic.score(real, g)
        _, c_mim = critic.score(mimic, g)
        _, c_ctl = critic.score(ctl, g)
        rows.append(dict(idx=m["idx"], theme=m["theme"], position=m["position"],
                         cosine_stored=m["cosine_stored"], token=m["decoded_token"],
                         cos_real=round(c_real, 4), cos_mimic=round(c_mim, 4),
                         cos_control=round(c_ctl, 4), delta=round(c_real - c_mim, 4),
                         real_text=real, mimic_text=mimic))
        if (j + 1) % 16 == 0 or j == len(valid) - 1:
            print(f"  [{j+1}/{len(valid)}] {m['theme']:15s} real={c_real:+.3f} "
                  f"mimic={c_mim:+.3f} ctl={c_ctl:+.3f}", flush=True)

    sr = [r["cos_real"] for r in rows]
    deltas = [r["delta"] for r in rows]
    summary = dict(
        mean_cos_real=round(sum(sr) / len(sr), 4),
        mean_cos_mimic=round(sum(r["cos_mimic"] for r in rows) / len(rows), 4),
        mean_cos_control=round(sum(r["cos_control"] for r in rows) / len(rows), 4),
        mean_delta=round(sum(deltas) / len(deltas), 4),
        median_delta=round(sorted(deltas)[len(deltas) // 2], 4),
        real_beats_mimic=f"{sum(1 for d in deltas if d > 0)}/{len(deltas)}",
        real_beats_control=f"{sum(1 for r in rows if r['cos_real'] > r['cos_control'])}/{len(rows)}",
    )
    out = dict(
        validation=None,
        alignment=dict(match=sum(1 for m in manifest if m["anchor_ok"] and m["in_range"]),
                       n=len(manifest)),
        injection=dict(cjk=av["n_cjk"], noexp=av["n_noexp"], n=av["n_done"],
                       temperature=av.get("temperature")),
        difficulty_corr=round(pearson(sr, [r["cosine_stored"] for r in rows]), 4),
        summary=summary, rows=rows,
    )
    (outdir / "results.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n[score:{key}] RESULT real={summary['mean_cos_real']} mimic={summary['mean_cos_mimic']} "
          f"control={summary['mean_cos_control']} | Δ={summary['mean_delta']} "
          f"real>mimic {summary['real_beats_mimic']}", flush=True)
    print(f"[score:{key}] injection: CJK {av['n_cjk']}/{av['n_done']}, "
          f"no-explan {av['n_noexp']}; anchor_ok {out['alignment']['match']}/{out['alignment']['n']}; "
          f"difficulty_corr {out['difficulty_corr']}", flush=True)
    print(f"[score:{key}] wrote {outdir/'results.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
