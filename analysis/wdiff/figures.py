"""Aggregation figures from a finished run dir (stats.csv + spectra.safetensors).

All figures restrict to 2D weight matrices except the heatmap's `norm` row,
which is the mean rel_fro of that layer's 1D norm weights. Biases (Qwen q/k/v)
are in stats.csv but not in figures.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

from .naming import CLASSES  # noqa: E402
from .stats import principal_angle_cos2  # noqa: E402

_PLOT_CLASSES = [c for c in CLASSES if c not in ("other",)]
_LINE_CLASSES = [c for c in _PLOT_CLASSES if c not in ("embed", "lm_head", "norm")]


def load_run(run_dir: str | Path) -> tuple[pd.DataFrame, dict]:
    run_dir = Path(run_dir)
    df = pd.read_csv(run_dir / "stats.csv")
    manifest = json.loads((run_dir / "manifest.json").read_text())
    return df, manifest


def _is_weight(df: pd.DataFrame) -> pd.Series:
    return (df["ndim"] == 2) & df["skipped"].isna() if "skipped" in df else (df["ndim"] == 2)


def fig_heatmap(df: pd.DataFrame, manifest: dict, out_png: Path, metric: str = "rel_fro"):
    n_layers = int(manifest["n_base_layers"])
    l_trunc = manifest.get("l_trunc")
    n_show = l_trunc if l_trunc else n_layers
    cols = ["emb"] + list(range(n_show)) + ["head"]
    acc = np.zeros((len(_PLOT_CLASSES), len(cols)))
    cnt = np.zeros_like(acc)

    w = df[_is_weight(df)]
    for ci, cls in enumerate(_PLOT_CLASSES):
        sub = df[(df["cls"] == "norm") & (df["ndim"] == 1)] if cls == "norm" \
            else w[w["cls"] == cls]
        for _, r in sub.iterrows():
            if cls == "embed":
                col = 0
            elif cls == "lm_head":
                col = len(cols) - 1
            elif not np.isnan(r["layer"]) and int(r["layer"]) < n_show:
                col = 1 + int(r["layer"])
            else:
                continue
            acc[ci, col] += r[metric]
            cnt[ci, col] += 1
    with np.errstate(invalid="ignore"):
        mat = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)

    fig, ax = plt.subplots(figsize=(max(8, n_show * 0.28), 4.2))
    pos = mat[np.isfinite(mat) & (mat > 0)]
    norm = LogNorm(vmin=pos.min(), vmax=pos.max()) if pos.size else None
    im = ax.imshow(mat, aspect="auto", cmap="viridis", norm=norm)
    ax.set_yticks(range(len(_PLOT_CLASSES)), _PLOT_CLASSES)
    xt = [0] + list(range(1, n_show + 1, max(1, n_show // 16))) + [len(cols) - 1]
    ax.set_xticks(xt, ["E"] + [str(c - 1) for c in xt[1:-1]] + ["LM"])
    ax.set_xlabel("layer")
    ax.set_title(f"{manifest['pair']}: {metric} per (layer x component)")
    fig.colorbar(im, ax=ax, label=metric + (" (log scale)" if norm else ""))
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def fig_depth_lines(df: pd.DataFrame, manifest: dict, out_png: Path):
    metrics = ["rel_fro", "stable_rank", "energy_rank_0.9"]
    w = df[_is_weight(df) & df["layer"].notna()].copy()
    w["energy_rank_0.9"] = pd.to_numeric(w["energy_rank_0.9"], errors="coerce")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, metric in zip(axes, metrics):
        for cls in _LINE_CLASSES:
            sub = w[w["cls"] == cls].sort_values("depth_frac")
            if sub.empty:
                continue
            ax.plot(sub["depth_frac"], sub[metric], marker=".", ms=4, lw=1, label=cls)
        ax.set_xlabel("depth_frac = layer/(n_layers-1)")
        ax.set_ylabel(metric)
        if metric != "rel_fro":
            ax.set_yscale("log")
        else:
            ax.set_yscale("log")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7, ncol=2)
    if manifest.get("l_trunc"):
        for ax in axes:
            ax.axvline(manifest["l_trunc"] / (manifest["n_base_layers"] - 1),
                       color="gray", ls=":", lw=1)
    fig.suptitle(f"{manifest['pair']}: per-component depth profiles "
                 f"(energy_rank '>K' shown as gaps)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def fig_spectra(df: pd.DataFrame, manifest: dict, run_dir: Path, out_png: Path):
    from safetensors.numpy import load_file

    spectra = load_file(str(run_dir / "spectra.safetensors"))
    w = df[_is_weight(df)].copy()

    picks: list[tuple[str, str]] = []  # (label, canonical)
    for cls in _LINE_CLASSES + ["embed", "lm_head"]:
        sub = w[w["cls"] == cls]
        if sub.empty:
            continue
        r = sub.loc[sub["rel_fro"].idxmax()]
        picks.append((f"{cls} (top)", r["canonical"]))
    layered = w[w["layer"].notna()]
    if not layered.empty:
        n_show = layered["layer"].max() + 1
        for ti, (lo, hi) in enumerate([(0, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 1.001)]):
            sub = layered[(layered["layer"] / n_show >= lo) & (layered["layer"] / n_show < hi)]
            if sub.empty:
                continue
            r = sub.loc[sub["rel_fro"].idxmax()]
            picks.append((f"tercile {ti + 1} (top)", r["canonical"]))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    seen = set()
    for label, canon in picks:
        if canon in seen:
            continue
        seen.add(canon)
        key = f"{canon}::sigma"
        if key not in spectra:
            continue
        sig = spectra[key]
        ax = axes[0] if "tercile" not in label else axes[1]
        ax.plot(np.arange(1, len(sig) + 1), sig, lw=1,
                label=f"{label}: {canon.replace('model.layers.', 'L')}")
    for ax, title in zip(axes, ["highest rel_fro per component class",
                                "highest rel_fro per depth tercile"]):
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_xlabel("singular value index i")
        ax.set_ylabel(r"$\sigma_i$")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6)
        ax.set_title(title)
    fig.suptitle(f"{manifest['pair']}: dW singular spectra")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def make_all(run_dir: str | Path) -> list[Path]:
    run_dir = Path(run_dir)
    df, manifest = load_run(run_dir)
    figdir = run_dir / "figures"
    figdir.mkdir(exist_ok=True)
    outs = []
    for fn, f in [("heatmap_rel_fro.png", lambda: fig_heatmap(df, manifest, figdir / "heatmap_rel_fro.png")),
                  ("depth_lines.png", lambda: fig_depth_lines(df, manifest, figdir / "depth_lines.png")),
                  ("spectra.png", lambda: fig_spectra(df, manifest, run_dir, figdir / "spectra.png"))]:
        f()
        outs.append(figdir / fn)
    return outs


def cross_module(av_dir: str | Path, ar_dir: str | Path, out_dir: str | Path, k: int = 16):
    """Principal-angle overlap between top-k singular subspaces of dW_AV and
    dW_AR for matched tensors at layers < L_TRUNC, vs. the random baseline."""
    from safetensors.numpy import load_file

    av_dir, ar_dir, out_dir = Path(av_dir), Path(ar_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df_av, man_av = load_run(av_dir)
    df_ar, man_ar = load_run(ar_dir)
    l_trunc = man_ar.get("l_trunc")
    sp_av = load_file(str(av_dir / "spectra.safetensors"))
    sp_ar = load_file(str(ar_dir / "spectra.safetensors"))

    w_ar = df_ar[(df_ar["ndim"] == 2) & df_ar["layer"].notna()]
    if l_trunc is not None:
        w_ar = w_ar[w_ar["layer"] < l_trunc]
    rows = []
    for _, r in w_ar.iterrows():
        canon = r["canonical"]
        if f"{canon}::u" not in sp_av or f"{canon}::u" not in sp_ar:
            continue
        for side in ("u", "v"):
            a, b = sp_av[f"{canon}::{side}"], sp_ar[f"{canon}::{side}"]
            kk = min(k, a.shape[1], b.shape[1])
            cos2 = principal_angle_cos2(a[:, :kk], b[:, :kk])
            rows.append({
                "canonical": canon, "cls": r["cls"], "layer": int(r["layer"]),
                "side": side, "k": kk, "d": a.shape[0],
                "mean_cos2": float(cos2.mean()),
                "baseline_mean_cos2": kk / a.shape[0],
            })
    cm = pd.DataFrame(rows)
    cm.to_csv(out_dir / "cross_module.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    for ax, side in zip(axes, ("u", "v")):
        sub = cm[cm["side"] == side]
        for cls in sorted(sub["cls"].unique()):
            s = sub[sub["cls"] == cls].sort_values("layer")
            ax.plot(s["layer"], s["mean_cos2"], marker=".", lw=1, label=cls)
        base = sub.groupby("layer")["baseline_mean_cos2"].mean()
        ax.plot(base.index, base.values, "k--", lw=1.2, label="random baseline k/d")
        ax.set_xlabel("layer")
        ax.set_title(f"{'left (output)' if side == 'u' else 'right (input)'} subspaces")
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(f"mean cos^2 principal angles (top-{k})")
    axes[0].legend(fontsize=7)
    fig.suptitle("dW_AV vs dW_AR singular-subspace overlap (layers < L_TRUNC)")
    fig.tight_layout()
    fig.savefig(out_dir / "cross_module.png", dpi=150)
    plt.close(fig)
    return out_dir / "cross_module.png"
