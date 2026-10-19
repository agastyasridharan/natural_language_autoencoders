"""CLI for wdiff. argparse, matching repo convention.

Subcommands:
  run        diff one (base, ft) pair end to end
  selfdiff   sanity 1: M vs M must give rel_fro == 0 everywhere
  synth      sanity 2: known rank-r perturbation must be recovered exactly
  spotcheck  sanity 3: independent numpy recomputation of rel_fro on 3 tensors
  figures    heatmap + depth lines + spectra for a finished run
  cross      principal-angle overlap between dW_AV and dW_AR subspaces
"""
from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
import tempfile
from pathlib import Path


def _token(args) -> str | None:
    if getattr(args, "token", None):
        return args.token
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    p = Path.home() / ".cache/huggingface/token"
    return p.read_text().strip() if p.exists() else None


def _add_run_opts(p: argparse.ArgumentParser) -> None:
    p.add_argument("--full-svd-threshold", type=int, default=8192,
                   help="full SVD when min(shape) <= this (else randomized top-K)")
    p.add_argument("--topk", type=int, default=256)
    p.add_argument("--niter", type=int, default=2)
    p.add_argument("--oversample", type=int, default=16)
    p.add_argument("--keep-vectors", type=int, default=16,
                   help="singular vectors persisted per tensor (>= 8 per spec; "
                        "16 so --cross-module has top-16 subspaces)")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--max-svd-dim", type=int, default=0,
                   help="CPU safeguard: randomized SVD when max(shape) exceeds this (0=off)")
    p.add_argument("--limit", type=int, default=0, help="process only first N tensors")
    p.add_argument("--only", default=None, help="regex filter on ft tensor names")
    p.add_argument("--abort-over-gb", type=float, default=0,
                   help="abort if planned transfer exceeds this many GB (0=off)")
    p.add_argument("--token", default=None, help="HF token (default: env/CLI cache)")


def cmd_run(args) -> int:
    from .runner import run_pair

    args.token = _token(args)
    run_pair(args)
    return 0


def cmd_selfdiff(args) -> int:
    import pandas as pd

    from .runner import run_pair

    args.base = args.repo
    args.ft = args.repo
    args.pair_name = f"selfdiff-{args.repo.split('/')[-1]}"
    args.token = _token(args)
    out = run_pair(args)
    df = pd.read_csv(out / "stats.csv")
    bad = df[(df["rel_fro"] != 0.0) | (df["fro_dw"] != 0.0)]
    er_cols = [c for c in df.columns if c.startswith("energy_rank")]
    print(f"selfdiff: {len(df)} tensors; rel_fro range "
          f"[{df['rel_fro'].min()}, {df['rel_fro'].max()}]; "
          f"max sigma1={df['sigma1'].max() if 'sigma1' in df else 'n/a'}")
    if not bad.empty:
        print(bad[["name", "rel_fro", "fro_dw"]].to_string())
        print("SELF-DIFF FAIL: nonzero diffs above")
        return 1
    for c in er_cols:
        nz = df[pd.to_numeric(df[c], errors="coerce") != 0][c].dropna()
        if not nz.empty:
            print(f"SELF-DIFF FAIL: {c} nonzero for {len(nz)} tensors")
            return 1
    print("SELF-DIFF PASS: rel_fro exactly 0 everywhere")
    return 0


def cmd_synth(args) -> int:
    """W_base random; W_ft = W_base + rank-12 equal-singular-value perturbation
    at relative Frobenius norm 0.05. Pipeline must recover r and rel_fro within
    1%. Runs both the full-SVD path and (via --full-svd-threshold 8) the
    randomized path."""
    import pandas as pd
    import torch
    from safetensors.torch import save_file

    from .runner import run_pair

    g = torch.Generator().manual_seed(0)
    m, n, r, rel = 1536, 1024, 12, 0.05
    W = torch.randn(m, n, generator=g)
    U, _ = torch.linalg.qr(torch.randn(m, r, generator=g))
    V, _ = torch.linalg.qr(torch.randn(n, r, generator=g))
    E = U @ V.T  # all singular values exactly 1
    E *= rel * math.sqrt(float((W * W).sum()) / float((E * E).sum()))
    bias = torch.randn(n, generator=g)

    with tempfile.TemporaryDirectory() as td:
        bdir, fdir = Path(td, "base"), Path(td, "ft")
        bdir.mkdir(), fdir.mkdir()
        save_file({"model.layers.0.mlp.up_proj.weight": W.contiguous(),
                   "model.layers.0.mlp.up_proj.bias": bias.clone(),
                   "model.layers.0.input_layernorm.weight": torch.ones(n)},
                  str(bdir / "model.safetensors"))
        save_file({"model.layers.0.mlp.up_proj.weight": (W + E).contiguous(),
                   "model.layers.0.mlp.up_proj.bias": bias.clone(),
                   "model.layers.0.input_layernorm.weight": torch.ones(n)},
                  str(fdir / "model.safetensors"))

        ok = True
        for label, thresh in [("full", 8192), ("randomized", 8)]:
            ns = argparse.Namespace(
                base=str(bdir), ft=str(fdir), out=str(Path(td, f"out_{label}")),
                pair_name=f"synth-{label}", full_svd_threshold=thresh,
                topk=64, niter=2, oversample=16, keep_vectors=16,
                device="cpu", max_svd_dim=0, limit=0, only=None,
                abort_over_gb=0, token=None,
            )
            out = run_pair(ns)
            df = pd.read_csv(out / "stats.csv").set_index("canonical")
            row = df.loc["model.layers.0.mlp.up_proj.weight"]
            checks = {
                "rel_fro within 1%": abs(row["rel_fro"] - rel) / rel < 0.01,
                "energy_rank_0.99 == 12": str(row["energy_rank_0.99"]) == "12",
                "energy_rank_0.9 == 11": str(row["energy_rank_0.9"]) == "11",
                "energy_rank_0.5 == 6": str(row["energy_rank_0.5"]) == "6",
                "stable_rank ~ 12": abs(row["stable_rank"] - 12) < 0.05,
                "bias rel_fro == 0":
                    df.loc["model.layers.0.mlp.up_proj.bias", "rel_fro"] == 0.0,
                "norm rel_fro == 0":
                    df.loc["model.layers.0.input_layernorm.weight", "rel_fro"] == 0.0,
                "cos_flat < 1": row["cos_flat"] < 1.0,
            }
            if label == "full":
                checks["erank_entropy ~ 12"] = abs(row["erank_entropy"] - 12) < 0.2
                checks["spectrum == full"] = row["spectrum"] == "full"
            else:
                checks["spectrum == randomized"] = row["spectrum"] == "randomized"
            for name, passed in checks.items():
                print(f"  [{label}] {'PASS' if passed else 'FAIL'}: {name}")
                ok &= passed
        print("SYNTH " + ("PASS" if ok else "FAIL"))
        return 0 if ok else 1


def cmd_spotcheck(args) -> int:
    """Independent rel_fro recomputation: own range-fetch, own bf16 decode,
    numpy float64 arithmetic. No torch, no wdiff.sources."""
    import httpx
    import numpy as np
    import pandas as pd

    token = _token(args)
    client = httpx.Client(
        headers={"Authorization": f"Bearer {token}"} if token else {},
        timeout=httpx.Timeout(30.0, read=300.0), follow_redirects=True)

    def fetch(repo: str, name: str) -> np.ndarray:
        # find the shard via the index if present, else scan shards
        def header(fname):
            r = client.get(f"https://huggingface.co/{repo}/resolve/main/{fname}",
                           headers={"Range": "bytes=0-7"})
            (k,) = struct.unpack("<Q", r.content)
            r = client.get(f"https://huggingface.co/{repo}/resolve/main/{fname}",
                           headers={"Range": f"bytes=8-{8 + k - 1}"})
            return json.loads(r.content), 8 + k

        r = client.get(f"https://huggingface.co/api/models/{repo}")
        files = [s["rfilename"] for s in r.json()["siblings"]
                 if s["rfilename"].endswith(".safetensors")]
        idx_url = f"https://huggingface.co/{repo}/resolve/main/model.safetensors.index.json"
        ri = client.get(idx_url)
        if ri.status_code == 200:
            shard = ri.json()["weight_map"].get(name)
            files = [shard] if shard else files
        for f in files:
            hdr, ds = header(f)
            if name in hdr:
                ent = hdr[name]
                a = ds + ent["data_offsets"][0]
                b = ds + ent["data_offsets"][1] - 1
                raw = client.get(
                    f"https://huggingface.co/{repo}/resolve/main/{f}",
                    headers={"Range": f"bytes={a}-{b}"}).content
                if ent["dtype"] == "BF16":
                    u16 = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16
                    arr = u16.view(np.float32).astype(np.float64)
                elif ent["dtype"] == "F32":
                    arr = np.frombuffer(raw, dtype="<f4").astype(np.float64)
                elif ent["dtype"] == "F16":
                    arr = np.frombuffer(raw, dtype="<f2").astype(np.float64)
                else:
                    raise ValueError(ent["dtype"])
                return arr.reshape(ent["shape"])
        raise KeyError(f"{name} not found in {repo}")

    df = pd.read_csv(Path(args.run) / "stats.csv").set_index("name")
    ok = True
    for name in args.names.split(","):
        row = df.loc[name]
        wb = fetch(args.base, row["base_name"])
        wf = fetch(args.ft, name)
        rel = float(np.linalg.norm(wf - wb) / np.linalg.norm(wb))
        pipe = float(row["rel_fro"])
        agree = (abs(rel - pipe) <= 1e-4 * max(rel, 1e-30)) or (rel == pipe == 0.0)
        print(f"  {'PASS' if agree else 'FAIL'} {name}: numpy={rel:.10e} "
              f"pipeline={pipe:.10e}")
        ok &= agree
    print("SPOTCHECK " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def cmd_figures(args) -> int:
    from .figures import make_all

    for p in make_all(args.run):
        print(p)
    return 0


def cmd_cross(args) -> int:
    from .figures import cross_module

    print(cross_module(args.av_run, args.ar_run, args.out, k=args.k))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="wdiff", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run", help="diff a (base, ft) pair")
    p.add_argument("--base", required=True, help="HF repo id or local dir of M")
    p.add_argument("--ft", required=True, help="HF repo id or local dir of AV/AR")
    p.add_argument("--out", required=True)
    p.add_argument("--pair-name", required=True)
    _add_run_opts(p)
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("selfdiff", help="sanity: M vs M must be exactly zero")
    p.add_argument("--repo", required=True)
    p.add_argument("--out", required=True)
    _add_run_opts(p)
    p.set_defaults(fn=cmd_selfdiff)

    p = sub.add_parser("synth", help="sanity: recover a known rank-12 perturbation")
    p.set_defaults(fn=cmd_synth)

    p = sub.add_parser("spotcheck", help="sanity: independent numpy rel_fro check")
    p.add_argument("--run", required=True, help="finished run dir (stats.csv)")
    p.add_argument("--base", required=True)
    p.add_argument("--ft", required=True)
    p.add_argument("--names", required=True, help="comma-separated ft tensor names")
    p.add_argument("--token", default=None)
    p.set_defaults(fn=cmd_spotcheck)

    p = sub.add_parser("figures", help="render figures for a finished run")
    p.add_argument("--run", required=True)
    p.set_defaults(fn=cmd_figures)

    p = sub.add_parser("cross", help="dW_AV vs dW_AR subspace overlap")
    p.add_argument("--av-run", required=True)
    p.add_argument("--ar-run", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--k", type=int, default=16)
    p.set_defaults(fn=cmd_cross)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
