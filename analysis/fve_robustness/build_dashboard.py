"""Build the NLA FVE-robustness dashboard (belief-geometry visual style) from the
per-model fve_<key>.json results in ./results/. Rerun to repopulate as more
models / manipulation arms complete.

  python build_dashboard.py   # writes ./dashboard.html
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
OUT = HERE / "dashboard.html"
BUILD_DATE = "2026-07-23"

MODELS = [
    dict(key="qwen7b",   label="Qwen2.5-7B",    layer=20, d_model=3584, base="Qwen/Qwen2.5-7B-Instruct"),
    dict(key="gemma12b", label="Gemma-3-12B",   layer=32, d_model=3840, base="google/gemma-3-12b-it"),
    dict(key="gemma27b", label="Gemma-3-27B",   layer=41, d_model=5376, base="google/gemma-3-27b-it"),
    dict(key="llama70b", label="Llama-3.3-70B", layer=53, d_model=8192, base="meta-llama/Llama-3.3-70B-Instruct"),
]

# The planned robustness series. Baseline is measured; the rest are scaffolded
# and populated as each experiment runs.
ARMS = [
    dict(label="− all quotes",
         desc="Remove all quoted spans from the AV explanation, then reconstruct."),
    dict(label="− input-derivable quotes",
         desc="Remove only quotes that verbatim- or synonym-restate the input context; keep quotes that diverge from it."),
    dict(label="quote block → natural language",
         desc="Rephrase the trailing quoted block into unquoted prose."),
    dict(label="shuffle claims",
         desc="Randomly permute the claims / sentences within the explanation."),
    dict(label="real-quote substitution",
         desc="Replace AV-generated quotes with matched real quotes from the input. FVE↑ would indicate a preference for input inversion."),
]


def load(key):
    p = RES / f"fve_{key}.json"
    return json.loads(p.read_text()) if p.exists() else None


CSS = """
:root{
  --bg:#faf6f1; --panel:#fffdf9; --ink:#2c2416; --ink-strong:#1a1408;
  --ink-soft:#3d3020; --muted:#6b5d4d; --line:#e8e0d4; --line-soft:#efe8dc;
  --green:#4a7c59; --green-d:#3d6648; --terra:#a65d57; --pending:#bcae9b;
}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'EB Garamond',Georgia,'Times New Roman',serif;background:var(--bg);
  color:var(--ink);font-size:15px;line-height:1.55;padding:30px 34px}
.container{max-width:1080px;margin:0 auto}
h1{font-size:1.75rem;font-weight:600;color:var(--ink-strong);letter-spacing:-0.02em;margin-bottom:4px}
.subtitle{color:var(--muted);font-style:italic;font-size:0.98rem;margin-bottom:20px}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:6px;
  padding:18px 22px;margin-bottom:18px;box-shadow:0 2px 6px rgba(44,36,22,0.05)}
h2{font-size:1.08rem;font-weight:600;color:var(--ink-soft);margin-bottom:10px}
p{margin-bottom:8px}
.lead{color:var(--ink);margin-bottom:6px}
.formula{margin:12px 0 6px;text-align:center;color:var(--ink-soft)}
table{width:100%;border-collapse:collapse;font-size:0.95rem;font-variant-numeric:tabular-nums}
th{text-align:left;font-weight:600;color:var(--ink-soft);border-bottom:2px solid #e0d6c8;
  padding:8px 10px;font-size:0.82rem;letter-spacing:0.02em;text-transform:uppercase}
td{padding:9px 10px;border-bottom:1px solid var(--line-soft);vertical-align:baseline}
tr:last-child td{border-bottom:none}
.model{font-weight:600;color:var(--ink)}
.sub{color:var(--muted);font-size:0.82rem}
.fve{color:var(--green);font-weight:700;font-size:1.12rem}
.pending{color:var(--pending);font-style:italic}
.running{color:var(--terra);font-style:italic}
.arm th{font-size:0.8rem}
.armname{font-weight:600;color:var(--ink-soft)}
.legend li{margin:6px 0;color:var(--ink);list-style:none}
.legend .armname{display:inline-block;min-width:220px}
.note{color:var(--muted);font-size:0.88rem}
.accent{color:var(--green);font-weight:600}
footer{color:var(--muted);font-size:0.82rem;margin-top:6px;padding-top:12px;border-top:1px solid var(--line)}
code{font-family:'SFMono-Regular',Menlo,Consolas,monospace;font-size:0.8rem;color:var(--ink-soft);
  background:#f3ece1;padding:1px 5px;border-radius:3px}
.badge{display:inline-block;font-size:0.72rem;letter-spacing:0.03em;text-transform:uppercase;
  color:var(--green-d);border:1px solid #cfe0d3;background:#eef5ef;border-radius:10px;padding:1px 9px;margin-left:8px}
"""


def cell_fve(res):
    if res is None:
        return '<span class="running">running…</span>'
    return f'<span class="fve">{res["FVE"]:.3f}</span>'


def main() -> None:
    rows = []
    for m in MODELS:
        res = load(m["key"])
        rows.append((m, res))

    # ---- main table: baseline FVE + scaffolded manipulation columns ----
    arm_headers = "".join(f"<th>{a['label']}</th>" for a in ARMS)
    main_rows = ""
    for m, res in rows:
        n = f'{res["n_scored"]:,}' if res else "—"
        arm_cells = "".join('<td><span class="pending">—</span></td>' for _ in ARMS)
        main_rows += (
            f'<tr><td><span class="model">{m["label"]}</span><br>'
            f'<span class="sub">layer {m["layer"]} &middot; d={m["d_model"]}</span></td>'
            f'<td>{n}</td><td>{cell_fve(res)}</td>{arm_cells}</tr>'
        )

    # ---- diagnostics table ----
    diag_rows = ""
    for m, res in rows:
        if res:
            diag_rows += (
                f'<tr><td><span class="model">{m["label"]}</span></td>'
                f'<td>{res["mean_cos"]:.3f}</td><td>{res["D_variance"]:.3f}</td>'
                f'<td>{res["norm_share_in_mean"]:.2f}</td>'
                f'<td>{res["n_cjk"]}/{res["n_scored"]:,}</td>'
                f'<td>{res["n_scored"]-res["n_noexp"]:,}/{res["n_scored"]:,}</td></tr>'
            )
        else:
            diag_rows += (
                f'<tr><td><span class="model">{m["label"]}</span></td>'
                f'<td colspan="5" class="running">running…</td></tr>'
            )

    legend = "".join(
        f'<li><span class="armname">{a["label"]}</span> '
        f'<span class="note">{a["desc"]}</span></li>' for a in ARMS
    )

    done = sum(1 for _, r in rows if r)
    html = f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NLA Reconstruction Robustness &mdash; FVE</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=EB+Garamond:ital,wght@0,400;0,500;0,600;0,700;1,400&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"
 onload="renderMathInElement(document.body,{{delimiters:[{{left:'$$',right:'$$',display:true}},{{left:'$',right:'$',display:false}}],throwOnError:false}});"></script>
<style>{CSS}</style></head>
<body><div class="container">

<h1>NLA Reconstruction Robustness</h1>
<div class="subtitle">Fraction of variance explained (FVE) of the four open Natural-Language-Autoencoders,
under manipulations of the verbalizer&rsquo;s explanation &mdash; held-out UltraFineWeb.</div>

<div class="panel">
  <h2>What this measures</h2>
  <p class="lead">Each NLA verbalizes a residual-stream activation into text (AV), then reconstructs the
  activation from that text alone (AR). FVE is the fraction of activation variance the round-trip recovers,
  computed on unit-normalized activations so the shared mean does not inflate it:</p>
  <div class="formula">$$\\mathrm{{FVE}} \\;=\\; 1 \\;-\\; \\frac{{\\sum_i \\lVert \\hat h_i - h_i \\rVert^2}}{{\\sum_i \\lVert h_i - \\bar h \\rVert^2}}
  \\;=\\; 1 - \\frac{{\\sum_i 2(1-\\cos_i)}}{{\\sum_i \\lVert \\hat h_i - \\bar h \\rVert^2}}$$</div>
  <p class="note">Baseline uses the AV&rsquo;s own explanation. The manipulation columns re-run the AR on an
  <em>edited</em> explanation and report the resulting change &mdash; isolating how much of the reconstruction
  rides on quoted / input-derivable content versus genuine activation reading.</p>
</div>

<div class="panel">
  <h2>Baseline FVE <span class="badge">{done}/4 models</span></h2>
  <table><thead><tr>
    <th>Model</th><th>n</th><th>FVE (baseline)</th>{arm_headers}
  </tr></thead><tbody>
  {main_rows}
  </tbody></table>
  <p class="note" style="margin-top:10px">Held-out UltraFineWeb (<code>openbmb/Ultra-FineWeb</code>, en), n&nbsp;=&nbsp;2000 docs,
  one random read position &ge;&nbsp;50 tokens per doc (training sampler); AV temperature&nbsp;0.4.
  Manipulation columns (&Delta; vs. baseline) are pending.</p>
</div>

<div class="panel">
  <h2>Reconstruction diagnostics</h2>
  <table><thead><tr>
    <th>Model</th><th>mean cos</th><th>D (variance)</th><th>norm-in-mean</th><th>CJK</th><th>with &lt;explanation&gt;</th>
  </tr></thead><tbody>
  {diag_rows}
  </tbody></table>
  <p class="note" style="margin-top:10px"><span class="accent">mean cos</span> is the raw AR&ndash;gold cosine; it saturates
  for Gemma (near-parallel activations, most energy in the shared mean &mdash; see <span class="accent">norm-in-mean</span>),
  which is exactly why FVE removes the mean. <span class="accent">CJK</span> counts injection-failure smells (should be ~0).</p>
</div>

<div class="panel">
  <h2>Planned manipulation arms</h2>
  <ul class="legend">{legend}</ul>
</div>

<footer>
  Held-out set: first 2000 valid docs (&ge;600 chars, deduped) streamed from a late <code>en</code> shard of
  <code>openbmb/Ultra-FineWeb</code> (CC-MAIN-2021-43), disjoint from the training recipe&rsquo;s front slice.
  Metric matches the paper / <code>athena/repl/fve.py</code>. Built {BUILD_DATE}.
</footer>

</div></body></html>"""

    OUT.write_text(html, encoding="utf-8")
    print(f"wrote {OUT}  ({done}/4 models populated)")


if __name__ == "__main__":
    main()
