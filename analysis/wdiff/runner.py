"""Per-pair orchestration: stream tensor pairs, compute statistics, persist.

Outputs under --out:
  manifest.json          repos, pinned shas, plan summary, options, totals
  stats.jsonl            one record per tensor, appended as computed (resume key)
  stats.csv              final tidy table (pandas)
  unmatched.json         ft-only / base-only tensors with shapes
  spectra.safetensors    top-16 (sigma, u, v) per 2D tensor, merged at finalize
  spectra_tmp/           per-tensor sidecars during the run (deleted on merge)

Resumability: a tensor is done iff its stats.jsonl row exists AND (it is 1D or
its spectra_tmp file exists). Sidecar is written before the jsonl row, so the
jsonl append is the commit point.
"""
from __future__ import annotations

import gc
import json
import math
import re
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from . import naming, stats
from .sources import open_repo

EXPECTED_FT_ONLY = {"value_head.weight"}  # AR affine head — no counterpart in M


def _safe_fname(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


def _spec_path(out: Path, canon: str) -> Path:
    return out / "spectra_tmp" / (_safe_fname(canon) + ".safetensors")


def pick_device(arg: str) -> str:
    if arg != "auto":
        return arg
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"  # MPS lacks reliable linalg.svd; CPU is the fallback


def run_pair(args) -> Path:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "spectra_tmp").mkdir(exist_ok=True)
    device = pick_device(args.device)

    base = open_repo(args.base, token=args.token)
    ft = open_repo(args.ft, token=args.token)
    plan = naming.build_plan(base.names(), ft.names())

    base_layers = [l for n in base.names() if (l := naming.layer_of(n)) is not None]
    n_base_layers = max(base_layers) + 1 if base_layers else 0

    matched = plan.matched
    if args.only:
        rx = re.compile(args.only)
        matched = [(f, b) for f, b in matched if rx.search(f)]
    # Stream in (shard, offset) order of the ft repo: sequential range reads.
    matched.sort(key=lambda p: (ft.meta(p[0]).file, ft.meta(p[0]).data_offsets[0]))
    if args.limit:
        matched = matched[: args.limit]

    planned_gb = sum(ft.meta(f).nbytes + base.meta(b).nbytes for f, b in matched) / 1e9
    surprise_ft_only = sorted(set(plan.ft_only) - EXPECTED_FT_ONLY)

    manifest = {
        "pair": args.pair_name,
        "base": {"spec": args.base, "sha": base.sha},
        "ft": {"spec": args.ft, "sha": ft.sha},
        "l_trunc": plan.l_trunc,
        "n_base_layers": n_base_layers,
        "n_matched": len(matched),
        "planned_gb": round(planned_gb, 2),
        "ft_only": [
            {"name": n, "shape": list(ft.meta(n).shape), "dtype": ft.meta(n).dtype}
            for n in plan.ft_only
        ],
        "ft_only_unexpected": surprise_ft_only,
        "base_truncated_count": len(plan.base_truncated),
        "base_vision_count": len(plan.base_vision),
        "base_other_unmatched": plan.base_other_unmatched,
        "options": {
            "full_svd_threshold": args.full_svd_threshold,
            "topk": args.topk,
            "niter": args.niter,
            "oversample": args.oversample,
            "keep_vectors": args.keep_vectors,
            "device": device,
            "max_svd_dim": args.max_svd_dim or None,
        },
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (out / "unmatched.json").write_text(json.dumps({
        "ft_only": manifest["ft_only"],
        "base_truncated": plan.base_truncated,
        "base_vision": plan.base_vision,
        "base_other_unmatched": plan.base_other_unmatched,
    }, indent=2))

    print(f"[{args.pair_name}] matched={len(matched)} planned={planned_gb:.1f} GB "
          f"l_trunc={plan.l_trunc} n_base_layers={n_base_layers} device={device}",
          flush=True)
    if surprise_ft_only:
        print(f"[{args.pair_name}] WARNING unexpected ft-only tensors: {surprise_ft_only}",
              flush=True)
    if args.abort_over_gb and planned_gb > args.abort_over_gb:
        raise SystemExit(
            f"planned transfer {planned_gb:.1f} GB exceeds --abort-over-gb "
            f"{args.abort_over_gb}; needs explicit approval"
        )

    jsonl = out / "stats.jsonl"
    done: set[str] = set()
    merged_keys: set[str] = set()
    mpath = out / "spectra.safetensors"
    if mpath.exists():  # a previous run already finalized
        from safetensors import safe_open

        with safe_open(str(mpath), framework="np") as f:
            merged_keys = {k.split("::")[0] for k in f.keys()}
    if jsonl.exists():
        for line in jsonl.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            canon = rec["canonical"]
            if (rec["ndim"] != 2 or rec.get("skipped")
                    or _spec_path(out, canon).exists() or canon in merged_keys):
                done.add(rec["name"])
    if done:
        print(f"[{args.pair_name}] resuming: {len(done)}/{len(matched)} already done",
              flush=True)

    t_run = time.time()
    with jsonl.open("a") as jf:
        for i, (fn, bn) in enumerate(matched):
            if fn in done:
                continue
            rec = _one_tensor(base, ft, fn, bn, plan, n_base_layers, out, device, args)
            jf.write(json.dumps(rec) + "\n")
            jf.flush()
            rf = rec.get("rel_fro")
            print(f"[{args.pair_name}] {i + 1}/{len(matched)} {rec['canonical']} "
                  f"shape={rec['shape']} "
                  f"rel_fro={'skip:' + rec['skipped'] if rec.get('skipped') else format(rf, '.3e')} "
                  f"({rec['seconds']:.1f}s)", flush=True)
            gc.collect()

    _finalize(out, args.pair_name)
    manifest["bytes_fetched_gb"] = round((base.bytes_fetched + ft.bytes_fetched) / 1e9, 2)
    manifest["wall_s"] = round(time.time() - t_run, 1)
    manifest["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[{args.pair_name}] DONE in {manifest['wall_s']}s, "
          f"fetched {manifest['bytes_fetched_gb']} GB this run", flush=True)
    return out


def _one_tensor(base, ft, fn, bn, plan, n_base_layers, out, device, args) -> dict:
    t0 = time.time()
    fm, bm = ft.meta(fn), base.meta(bn)
    canon = naming.canonical(fn)
    cls = naming.classify(fn)
    layer = naming.layer_of(fn)
    rec = {
        "name": fn,
        "base_name": bn,
        "canonical": canon,
        "cls": cls,
        "layer": layer,
        "depth_frac": (layer / (n_base_layers - 1))
        if layer is not None and n_base_layers > 1 else None,
        "ndim": len(fm.shape),
        "shape": list(fm.shape),
        "numel": int(np.prod(fm.shape)) if fm.shape else 1,
        "base_dtype": bm.dtype,
        "ft_dtype": fm.dtype,
        "vocab_overlap_only": False,
    }

    w_ft = ft.tensor(fn)
    w_base = base.tensor(bn)

    if tuple(w_ft.shape) != tuple(w_base.shape):
        same_cols = (
            w_ft.ndim == 2 and w_base.ndim == 2 and w_ft.shape[1] == w_base.shape[1]
        )
        if same_cols and cls in ("embed", "lm_head"):
            rows = min(w_ft.shape[0], w_base.shape[0])
            w_ft, w_base = w_ft[:rows], w_base[:rows]
            rec["vocab_overlap_only"] = True
            rec["overlap_rows"] = rows
        else:
            rec.update(skipped="shape_mismatch",
                       base_shape=list(w_base.shape), seconds=time.time() - t0)
            return rec
    else:
        # decode integrity: tensors must match their own header metadata
        assert tuple(w_ft.shape) == tuple(fm.shape), (fn, w_ft.shape, fm.shape)
        assert tuple(w_base.shape) == tuple(bm.shape), (bn, w_base.shape, bm.shape)

    fro_sq_base = stats.fro_sq(w_base)
    fro_sq_ft = stats.fro_sq(w_ft)
    rec["cos_flat"] = stats.cos_flat(w_ft, w_base)
    rec["fro_base"] = math.sqrt(fro_sq_base)
    rec["fro_ft"] = math.sqrt(fro_sq_ft)

    dW = w_ft.sub_(w_base)  # in-place: w_ft's storage becomes dW
    del w_ft, w_base
    fro_sq_dw = stats.fro_sq(dW)
    rec["fro_dw"] = math.sqrt(fro_sq_dw)
    rec["rel_fro"] = (
        math.sqrt(fro_sq_dw / fro_sq_base) if fro_sq_base > 0 else float("nan")
    )

    if dW.ndim == 2 and min(dW.shape) > 1:
        sstats, spec = stats.two_d_stats(
            dW, fro_sq_dw,
            full_threshold=args.full_svd_threshold, topk=args.topk,
            niter=args.niter, oversample=args.oversample,
            keep_vectors=args.keep_vectors, device=device,
            max_svd_dim=args.max_svd_dim or None,
        )
        rec.update({k: (v if not isinstance(v, np.floating) else float(v))
                    for k, v in sstats.items()})
        from safetensors.numpy import save_file

        k = min(args.keep_vectors, spec.sigmas.size)
        save_file(
            {
                "sigma": spec.sigmas[:k].astype(np.float64),
                "u": spec.u, "v": spec.v,
                "fro_sq_dw": np.array([fro_sq_dw], dtype=np.float64),
            },
            str(_spec_path(out, canon)),
        )
    del dW
    rec["seconds"] = round(time.time() - t0, 2)
    return rec


def _finalize(out: Path, pair_name: str) -> None:
    import pandas as pd

    rows = [json.loads(l) for l in (out / "stats.jsonl").read_text().splitlines() if l.strip()]
    # keep the last record per tensor (re-runs append)
    by_name = {r["name"]: r for r in rows}
    df = pd.DataFrame(list(by_name.values()))
    df.insert(0, "pair", pair_name)
    df = df.sort_values(["layer", "cls", "canonical"], na_position="first")
    df.to_csv(out / "stats.csv", index=False)

    # merge per-tensor sidecars into one spectra.safetensors (preserving any
    # previously merged spectra from earlier finalized runs)
    tmp = out / "spectra_tmp"
    files = sorted(tmp.glob("*.safetensors")) if tmp.exists() else []
    if files:
        from safetensors.numpy import load_file, save_file

        mpath = out / "spectra.safetensors"
        merged = load_file(str(mpath)) if mpath.exists() else {}
        for r in by_name.values():
            if r["ndim"] != 2 or r.get("skipped"):
                continue
            p = _spec_path(out, r["canonical"])
            if not p.exists():
                continue
            d = load_file(str(p))
            for k, v in d.items():
                merged[f"{r['canonical']}::{k}"] = v
        save_file(merged, str(mpath))
        shutil.rmtree(tmp)
    print(f"[{pair_name}] finalized: {len(by_name)} tensors -> stats.csv "
          f"+ spectra.safetensors", flush=True)
