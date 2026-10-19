"""Publication-style figures for the claim-ablation results.

Reads the four atom-mode run summaries (the same numbers the studio renders) and
writes matplotlib bar charts to ./figures/. Regenerable and stdlib+matplotlib only,
so it runs on a laptop with no GPU and no run artifacts beyond summary.json.

  python make_figures.py                 # writes figures/*.png from runs/*-atoms/

Figures:
  1. verdicts_across_models          grounded/entangled/confabulated rate per model
  2. grounding_by_claimtype          grounded rate by claim type, grouped by model
  3. grounding_<model>               per-model grounded rate by claim type (x4)
  4. verdict_composition             2x2, verdict composition by claim type per model
  5. hedging_calibration             grounded rate for hedged vs confident claims

Style: classic boxed matplotlib look (all four spines, gridlines both directions,
framed legend inside the axes, centered title, DejaVu Sans, saturated palette,
integer "(%)" ticks). Wilson 95% CIs as error bars. No em-dashes in the chrome.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
from matplotlib.patches import Patch
import numpy as np

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
OUT = HERE / "figures"

# ---- models (size-ascending), each -> its atom run dir --------------------------
MODELS = [
    ("qwen7b-cp6-fineweb-atoms",   "Qwen2.5-7B",    "L20", "FineWeb"),
    ("gemma12b-cp7-fineweb-atoms", "Gemma-3-12B",   "L32", "FineWeb"),
    ("gemma27b-g4-fineweb-atoms",  "Gemma-3-27B",   "L41", "UltraFineWeb"),
    ("llama70b-l4-fineweb-atoms",  "Llama-3.3-70B", "L53", "FineWeb"),
]
BUCKETS = [("gist", "GIST"), ("quote", "QUOTE"),
           ("position", "POSITION"), ("prediction", "PREDICTION")]

# ---- saturated palette (echoes the reference red/blue energy) --------------------
VERDICT = {"grounded": "#2E8B57", "entangled": "#E0A81E", "confabulated": "#C0392B"}
VERDICT_ORDER = ["grounded", "entangled", "confabulated"]
MODEL_COLOR = {
    "qwen7b-cp6-fineweb-atoms":   "#3B6FB0",   # blue
    "gemma12b-cp7-fineweb-atoms": "#4FA45B",   # green  (Gemma family, light)
    "gemma27b-g4-fineweb-atoms":  "#22703A",   # green  (Gemma family, dark)
    "llama70b-l4-fineweb-atoms":  "#7E3F9E",   # purple
}
HEDGE_COLOR = {"hedged": "#C0392B", "confident": "#3B6FB0"}   # red / blue, as in ref
MUT, GRID = "#5a5a5a", "#d3d3d3"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 11, "axes.titlesize": 14, "axes.labelsize": 12,
    "axes.edgecolor": "#333333", "axes.linewidth": 1.0,
    "xtick.color": "#222", "ytick.color": "#222", "axes.labelcolor": "#111", "text.color": "#111",
    "legend.frameon": True, "legend.framealpha": 0.95, "legend.edgecolor": "#bfbfbf",
    "legend.fontsize": 9.5, "figure.dpi": 120, "savefig.dpi": 200,
})
PCT = FuncFormatter(lambda x, _: f"{x * 100:.0f}")   # 0.30 -> "30", axis label carries "(%)"


def wilson(k: int, n: int, z: float = 1.96):
    """Return (p, lo, hi): point rate and Wilson 95% CI. scipy-free."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def yerr(rates):
    p = np.array([r[0] for r in rates])
    lo = np.array([r[1] for r in rates])
    hi = np.array([r[2] for r in rates])
    return np.vstack([np.maximum(0, p - lo), np.maximum(0, hi - p)])


def _rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))


def darken(h, f=0.62):
    return tuple(c * f for c in _rgb(h))


def load(key):
    return json.loads((RUNS / key / "summary.json").read_text())


def style_axis(ax, ymax=None, step=0.1):
    """Boxed look: keep all spines, gridlines both directions behind the bars,
    integer percent ticks."""
    ax.set_axisbelow(True)
    ax.grid(True, color=GRID, linewidth=0.7)
    ax.tick_params(length=3.5, color="#555")
    ax.yaxis.set_major_formatter(PCT)
    ax.yaxis.set_major_locator(MultipleLocator(step))
    if ymax is not None:
        ax.set_ylim(0, ymax)


def titled(ax, title, subtitle=None):
    ax.set_title(title, pad=(24 if subtitle else 12))
    if subtitle:
        ax.text(0.5, 1.015, subtitle, transform=ax.transAxes, ha="center",
                va="bottom", fontsize=9.5, color=MUT)


def bar_edge(color):
    return dict(edgecolor=darken(color), linewidth=0.7)


ECFG = dict(ecolor="#2b2b2b", elinewidth=1.0, capsize=2.5, capthick=1.0)
FOOT = "Atomic claim-ablation · judge-free · FineWeb · N≈400 activations/model"


# ---- figure 1: verdict rates across models --------------------------------------
def fig_verdicts():
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    x = np.arange(len(MODELS))
    w = 0.26
    for j, v in enumerate(VERDICT_ORDER):
        rates = []
        for key, *_ in MODELS:
            d = load(key)
            rates.append(wilson(d["verdict_counts"][v], d["n_snippets"]))
        ax.bar(x + (j - 1) * w, [r[0] for r in rates], w, yerr=yerr(rates),
               color=VERDICT[v], error_kw=ECFG, label=v.capitalize(), **bar_edge(VERDICT[v]))
    ax.set_xticks(x)
    ax.set_xticklabels([f"{lab} ({layer})" for _, lab, layer, _ in MODELS])
    ax.set_xlabel("Model")
    ax.set_ylabel("Share of claims (%)")
    style_axis(ax, ymax=0.72)
    ax.legend(loc="upper right", ncol=1)
    titled(ax, "Claim verdicts across models",
           "How each model's activation-verbalizer splits its claims into grounded / entangled / confabulated")
    fig.text(0.008, 0.006, FOOT, fontsize=8, color=MUT)
    save(fig, "verdicts_across_models")


# ---- figure 2: grounded rate by claim type, grouped by model --------------------
def fig_grounding_by_type():
    fig, ax = plt.subplots(figsize=(9.8, 5.4))
    x = np.arange(len(BUCKETS))
    w = 0.20
    for j, (key, lab, layer, _) in enumerate(MODELS):
        d = load(key)
        rates = [wilson(d["by_bucket"][b]["grounded"], d["by_bucket"][b]["n"])
                 for b, _ in BUCKETS]
        ax.bar(x + (j - 1.5) * w, [r[0] for r in rates], w, yerr=yerr(rates),
               color=MODEL_COLOR[key], error_kw=dict(ECFG, capsize=2.0),
               label=f"{lab} ({layer})", **bar_edge(MODEL_COLOR[key]))
    ax.set_xticks(x)
    ax.set_xticklabels([disp for _, disp in BUCKETS])
    ax.set_xlabel("Claim type")
    ax.set_ylabel("Grounded rate (%)")
    style_axis(ax, ymax=0.62)
    ax.legend(loc="upper right", ncol=1)
    titled(ax, "Which claim types get grounded",
           "Fraction of claims read from the activation, by semantic type. Bars = Wilson 95% CI.")
    fig.text(0.008, 0.006,
             "GIST = theme/topic/genre    QUOTE = verbatim spans    POSITION = where-in-text    "
             "PREDICTION = expected continuation", fontsize=8, color=MUT)
    save(fig, "grounding_by_claimtype")


# ---- figure 3: per-model grounded rate by claim type ----------------------------
def fig_grounding_per_model():
    for key, lab, layer, corpus in MODELS:
        d = load(key)
        overall = d["grounded_rate"]
        fig, ax = plt.subplots(figsize=(7.2, 4.9))
        rates = [wilson(d["by_bucket"][b]["grounded"], d["by_bucket"][b]["n"]) for b, _ in BUCKETS]
        ns = [d["by_bucket"][b]["n"] for b, _ in BUCKETS]
        x = np.arange(len(BUCKETS))
        color = MODEL_COLOR[key]
        rects = ax.bar(x, [r[0] for r in rates], 0.6, yerr=yerr(rates),
                       color=color, error_kw=ECFG, **bar_edge(color))
        for r, top in zip(rects, [rr[2] for rr in rates]):
            ax.annotate(f"{r.get_height() * 100:.0f}%",
                        (r.get_x() + r.get_width() / 2, top), xytext=(0, 3),
                        textcoords="offset points", ha="center", va="bottom", fontsize=9)
        ax.axhline(overall, ls="--", lw=1.2, color="#444", zorder=3)
        ax.text(len(BUCKETS) - 0.46, overall, f" overall {overall * 100:.0f}%",
                va="center", ha="left", fontsize=8.5, color="#444")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{disp}\n$n={n}$" for (_, disp), n in zip(BUCKETS, ns)])
        ax.set_xlabel("Claim type")
        ax.set_ylabel("Grounded rate (%)")
        style_axis(ax, ymax=min(0.62, max(r[2] for r in rates) + 0.12))
        titled(ax, f"{lab}: grounding by claim type",
               f"Layer {layer[1:]} · {corpus} · grounded rate per semantic type, Wilson 95% CI")
        save(fig, f"grounding_{key.split('-')[0]}")


# ---- figure 4: verdict composition by claim type (2x2) --------------------------
def fig_composition():
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 8.4))
    fig.subplots_adjust(top=0.85, bottom=0.13, left=0.075, right=0.985,
                        hspace=0.36, wspace=0.17)
    for r, (ax, (key, lab, layer, _)) in enumerate(zip(axes.ravel(), MODELS)):
        d = load(key)
        x = np.arange(len(BUCKETS))
        bottoms = np.zeros(len(BUCKETS))
        for v in VERDICT_ORDER:
            vals = np.array([d["by_bucket"][b][v] / d["by_bucket"][b]["n"] for b, _ in BUCKETS])
            ax.bar(x, vals, 0.62, bottom=bottoms, color=VERDICT[v],
                   edgecolor="white", linewidth=0.8)
            if v == "grounded":
                for xi, val in zip(x, vals):
                    ax.text(xi, val / 2, f"{val:.0%}", ha="center", va="center",
                            fontsize=8.5, color="white", fontweight="bold")
            bottoms += vals
        ax.set_xticks(x)
        ax.set_xticklabels([disp for _, disp in BUCKETS], fontsize=9)
        ax.set_axisbelow(True)
        ax.grid(True, axis="y", color=GRID, linewidth=0.7)
        ax.tick_params(length=3.5, color="#555")
        ax.yaxis.set_major_formatter(PCT)
        ax.yaxis.set_major_locator(MultipleLocator(0.2))
        ax.set_ylim(0, 1)
        ax.set_title(f"{lab}  ({layer})", fontsize=12)
        if r % 2 == 0:
            ax.set_ylabel("Share (%)")
        if r >= 2:
            ax.set_xlabel("Claim type")
    handles = [Patch(facecolor=VERDICT[v], label=v.capitalize()) for v in VERDICT_ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.02),
               handlelength=1.1, columnspacing=1.8, fontsize=10.5)
    fig.suptitle("Verdict composition by claim type", fontsize=15, y=0.955)
    fig.text(0.5, 0.905, "Each bar sums to 100%; grounded (read from h) at base. One panel per model.",
             ha="center", fontsize=9.5, color=MUT)
    save(fig, "verdict_composition", tight=False)


# ---- figure 5: hedging calibration ----------------------------------------------
def fig_hedging():
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    x = np.arange(len(MODELS))
    w = 0.34
    for j, split in enumerate(["hedged", "confident"]):
        rates = []
        for key, *_ in MODELS:
            h = load(key)["hedged"][split]
            rates.append(wilson(h["grounded"], h["n"]))
        ax.bar(x + (j - 0.5) * w, [r[0] for r in rates], w, yerr=yerr(rates),
               color=HEDGE_COLOR[split], error_kw=ECFG,
               label=split.capitalize() + " claims", **bar_edge(HEDGE_COLOR[split]))
    ax.set_xticks(x)
    ax.set_xticklabels([f"{lab} ({layer})" for _, lab, layer, _ in MODELS])
    ax.set_xlabel("Model")
    ax.set_ylabel("Grounded rate (%)")
    style_axis(ax, ymax=0.48)
    ax.legend(loc="upper right", ncol=1)
    titled(ax, "Is the verbalizer's hedging calibrated to grounding?",
           "Grounded rate for claims the AV hedged vs stated confidently. Calibrated hedging would sit lower.")
    fig.text(0.008, 0.006, FOOT, fontsize=8, color=MUT)
    save(fig, "hedging_calibration")


def save(fig, name, tight=True):
    OUT.mkdir(exist_ok=True)
    kw = dict(dpi=200, facecolor="white")
    if tight:
        kw["bbox_inches"] = "tight"
    fig.savefig(OUT / f"{name}.png", **kw)
    plt.close(fig)
    print("wrote", (OUT / f"{name}.png").relative_to(HERE))


def main():
    fig_verdicts()
    fig_grounding_by_type()
    fig_grounding_per_model()
    fig_composition()
    fig_hedging()
    print("done ->", OUT)


if __name__ == "__main__":
    main()
