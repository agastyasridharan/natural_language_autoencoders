"""Build the multi-tab FVE-robustness dashboard (belief-geometry style).

Emits, under analysis/fve_robustness/:
  index.html                    Overview: FVE matrix (models x baseline+experiments) + nav
  exp{1..6}_*.html              one tab per experiment; a full per-datapoint tab once its
                                records exist, else a "planned" placeholder
  expdata/data_<armkey>_<key>.js  per-model records, loaded on demand (script tag -> file://-safe)

Reads results/fve_*.json (baseline), results/records_<armkey>_<key>.json and
results/fve_<armkey>_<key>.json (per experiment). Rerun to repopulate as arms complete.
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
DATA = HERE / "expdata"
WARM = HERE.parent / "warm_start_robustness"
BUILD_DATE = "2026-08-27"

MODELS = [
    dict(key="qwen7b",   label="Qwen2.5-7B",    layer=20, d_model=3584),
    dict(key="gemma12b", label="Gemma-3-12B",   layer=32, d_model=3840),
    dict(key="gemma27b", label="Gemma-3-27B",   layer=41, d_model=5376),
]
LLAMA = dict(key="llama70b", label="Llama-3.3-70B", layer=53, d_model=8192)  # baseline-only

# armkey drives artifact filenames (records_<armkey>_<key>.json, fve_<armkey>_<key>.json,
# edited_reads_<armkey>.json). slug drives the html filename. Exp 6 (real-quote) is
# scaffolded but queued (not part of the current workflow run).
ARMS = [
    dict(armkey="exp1", slug="exp1_quotes",     tab="Exp 1", kind="deterministic",
         title="remove all quoted content",
         q="Are the quoted specifics load-bearing, or does the AV's abstract prose reconstruct on its own?"),
    dict(armkey="exp2", slug="exp2_quotesonly", tab="Exp 2", kind="deterministic",
         title="keep only quoted content",
         q="Complement of Exp 1: are the quotes SUFFICIENT? Keep only quoted spans, drop the connective prose."),
    dict(armkey="exp3a", slug="exp3a_derivable", tab="Exp 3a", kind="deterministic",
         title="remove input-derivable quotes (Tier A: verbatim ≥2-content-word run, keep divergent)",
         q="Is reconstruction carried by verbatim input-restatement quotes? (Tier A, deterministic)"),
    dict(armkey="exp3b", slug="exp3b_semantic",  tab="Exp 3b", kind="API",
         title="remove input-derivable quotes (Tier B: paraphrase/synonym-aware, keep divergent)",
         q="Once paraphrase/synonym restatements count, does reconstruction lean on input-derived quotes? (Tier B, API)"),
    dict(armkey="exp4a", slug="exp4a_delimiters", tab="Exp 4a", kind="deterministic",
         title="remove quotation delimiters only",
         q="Does quotation punctuation matter when every literal content string is preserved?",
         result="Result: quote punctuation has a small but nonzero reconstruction effect; see the "
                "inferential summary above for model-specific estimates."),
    dict(armkey="exp4b", slug="exp4b_paraphrase", tab="Exp 4b", kind="API",
         title="strictly paraphrase every quoted span",
         q="Does reconstruction survive when meaning is retained but exact quoted surface strings are removed?",
         result="Result: semantic paraphrasing lowered FVE far more than delimiter removal. Together "
                "with Exp 6, this shows that AR reconstruction is sensitive "
                "to both claim content/location and exact lexical realization; the paraphrase arm also "
                "changes tokenization and style, so it is evidence of form sensitivity, not by itself a "
                "proof of hidden coding."),
    dict(armkey="exp5a", slug="exp5a_blocks",   tab="Exp 5a", kind="deterministic",
         title="shuffle whole blocks at random (~3 blocks)",
         q="Does coarse claim (block) order matter, or is the AR a bag-of-claims reader?"),
    dict(armkey="exp5b", slug="exp5b_sentences",tab="Exp 5b", kind="deterministic",
         title="shuffle sentences at random (quote-aware)",
         q="Does sentence order matter? (~1 sentence/block, so near-identical to 5a)",
         result="Result: sentence shuffling lowered FVE. It matched block "
                "shuffling almost exactly for both Gemmas; Qwen was somewhat less affected."),
    dict(armkey="exp5c", slug="exp5c_claims",   tab="Exp 5c", kind="deterministic",
         title="shuffle the quoted claims inside the final block",
         q="Does the ORDER of the enumerated predicted-continuation claims matter, or just their set?",
         result="Result: with the read-token ECHO frozen, fine claim-slot shuffling lowered FVE. "
                "Fine order matters beyond "
                "coarse block order for Qwen and Gemma-12B; the two grains are indistinguishable for "
                "Gemma-27B."),
    dict(armkey="exp6a", slug="exp6a_realquote_incl", tab="Exp 6a", kind="deterministic",
         title="real-quote substitution A1 — true pre-token context, read token INCLUDED",
         q="Replace each left-context-reconstruction (LCR) quote with the TRUE pre-token context tail "
           "(word-length-matched), the injected text ENDING AT the read token (so it contains the gold "
           "token's own surface string); read-token ECHO and predicted-continuation (PC) quotes frozen. "
           "Companion to 6b: a rise here that VANISHES in 6b is a boundary-token echo, not input inversion."),
    dict(armkey="exp6b", slug="exp6b_realquote_excl", tab="Exp 6b", kind="deterministic",
         title="real-quote substitution A2 (load-bearing) — true pre-token context, read token EXCLUDED",
         q="Same as 6a but the injected true context ENDS ONE TOKEN BEFORE the read token (the gold token "
           "is absent). This is the clean input-inversion test: FVE RISING here (net of controls) is "
           "evidence the AR prefers the real input; flat/dropping is consistent with the AV's "
           "confabulated-but-gist-right quotes already sufficing. A1&minus;A2 isolates the boundary echo."),
    dict(armkey="exp6f", slug="exp6f_shuffle",    tab="Exp 6·F", kind="deterministic",
         title="control F — shuffle the AV's OWN LCR quote (form-OOD floor)",
         q="Replace each LCR quote with a seeded word-shuffle of its own words (same bag of words, "
           "destroyed order, length identical). If F collapses like 6b (A2), the A2 drop is structural / "
           "off-distribution (breaking the quote), not specific to true content. A2&minus;F isolates "
           "ordered-true-content from mere disturbance."),
    dict(armkey="exp6r", slug="exp6r_randctx",    tab="Exp 6·B", kind="deterministic",
         title="control B — random same-doc span (wrong position)",
         q="Replace each LCR quote with a seeded random real span (same word-length) from the same "
           "document's far past. Rules out 'any real same-doc text helps' / topicality: if 6b beats B, "
           "the effect is locality-specific, not generic doc vocabulary."),
    dict(armkey="exp6o", slug="exp6o_otherdoc",   tab="Exp 6·C", kind="deterministic",
         title="control C_otherdoc — a DIFFERENT doc's pre-token context",
         q="Replace each LCR quote with the true pre-token context tail (token excluded) of a different "
           "held-out document (real, grammatical, wrong content). This must HURT; if it doesn't, the LCR "
           "slots are inert and the whole arm is uninterpretable (the load-bearing gate)."),
    dict(armkey="exp6p", slug="exp6p_paraphrase", tab="Exp 6·P", kind="API + deterministic",
         title="core P — paraphrase the TRUE A1 context slice",
         q="Replace each LCR quote with a semantically audited paraphrase of A1's true context slice; "
           "ECHO and PC frozen. Length targets are logged (40–160% envelope) because exact matching "
           "forced filler/completions on truncated fragments; semantic failures fail closed to source.",
         result="Result: on high-divergence changed rows the effect was flat for both Gemmas and mildly "
                "negative for Qwen. P beat raw A2 decisively: A2's collapse "
                "is predominantly surface/form mismatch, not evidence for exact-input inversion."),
    dict(armkey="exp6t", slug="exp6t_true_cont", tab="Exp 6·T", kind="API-labelled deterministic",
         title="secondary C_cont — inject ordered TRUE continuation tiles",
         q="Freeze LCR/ECHO and replace PC slots with ordered word-boundary tiles from the true future. "
           "The preregistered result subset contains only rows where a semantic judge found that none "
           "of the AV's original continuation candidates was correct.",
         result="Result (AV-continuation-wrong subset): true-future tiles lowered FVE for every model. "
                "Correct future text does not improve reconstruction of a causally pre-future activation."),
]

CSS = """
:root{
  --bg:#faf6f1; --panel:#fffdf9; --ink:#2c2416; --ink-strong:#1a1408;
  --ink-soft:#3d3020; --muted:#6b5d4d; --line:#e8e0d4; --line-soft:#efe8dc;
  --green:#4a7c59; --green-d:#3d6648; --terra:#a65d57; --pending:#bcae9b;
  --hl:#f5ecd6; --mark:#e7f0e7;
}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'EB Garamond',Georgia,serif;background:var(--bg);color:var(--ink);
  font-size:15px;line-height:1.55;padding:24px 30px}
.container{max-width:1180px;margin:0 auto}
h1{font-size:1.7rem;font-weight:600;color:var(--ink-strong);letter-spacing:-0.02em;margin-bottom:4px}
.subtitle{color:var(--muted);font-style:italic;font-size:0.98rem;margin-bottom:16px}
nav{display:flex;gap:2px;flex-wrap:wrap;margin-bottom:20px;border-bottom:2px solid var(--line);padding-bottom:0}
nav a{padding:7px 15px;color:var(--muted);text-decoration:none;font-size:0.92rem;border:1px solid transparent;
  border-bottom:none;border-radius:6px 6px 0 0;position:relative;top:2px}
nav a:hover{color:var(--ink)}
nav a.on{color:var(--green-d);font-weight:600;background:var(--panel);border-color:var(--line);border-bottom:2px solid var(--panel)}
nav a.off{color:var(--pending)}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:16px 20px;
  margin-bottom:16px;box-shadow:0 2px 6px rgba(44,36,22,0.05)}
h2{font-size:1.05rem;font-weight:600;color:var(--ink-soft);margin-bottom:9px}
h4{font-size:0.9rem;font-weight:600;color:var(--ink-soft);margin-bottom:6px}
p{margin-bottom:8px}
.note{color:var(--muted);font-size:0.88rem}
.accent{color:var(--green);font-weight:600}
table{width:100%;border-collapse:collapse;font-size:0.95rem;font-variant-numeric:tabular-nums}
th{text-align:left;font-weight:600;color:var(--ink-soft);border-bottom:2px solid #e0d6c8;padding:8px 10px;
  font-size:0.8rem;letter-spacing:0.02em;text-transform:uppercase}
td{padding:8px 10px;border-bottom:1px solid var(--line-soft);vertical-align:baseline}
.matrix td,.matrix th{text-align:center}
.matrix td:first-child,.matrix th:first-child{text-align:left}
.fve{color:var(--green);font-weight:700}
.fvepos{color:var(--green-d);font-weight:700}
.fveneg{color:var(--terra);font-weight:700}
.fvemid{color:var(--muted);font-weight:600}
.blank{color:var(--pending)}
.pending{color:var(--pending);font-style:italic}
.badge{display:inline-block;font-size:0.72rem;letter-spacing:0.03em;text-transform:uppercase;color:var(--green-d);
  border:1px solid #cfe0d3;background:#eef5ef;border-radius:10px;padding:1px 9px;margin-left:6px}
.badge.grey{color:var(--muted);border-color:var(--line);background:#f3ece1}
.tag{font-size:0.7rem;text-transform:uppercase;letter-spacing:0.03em;color:var(--muted);border:1px solid var(--line);
  border-radius:8px;padding:1px 7px;margin-left:6px}
code{font-family:'SFMono-Regular',Menlo,Consolas,monospace;font-size:0.8rem;color:var(--ink-soft);
  background:#f3ece1;padding:1px 5px;border-radius:3px}
footer{color:var(--muted);font-size:0.82rem;margin-top:8px;padding-top:12px;border-top:1px solid var(--line)}
.mtoggle{display:flex;gap:6px;margin-bottom:12px}
.mbtn{font-family:inherit;font-size:0.9rem;padding:5px 14px;border:1px solid var(--line);background:#fff;
  color:var(--muted);border-radius:16px;cursor:pointer}
.mbtn.on{background:var(--green);color:#fff;border-color:var(--green)}
.summary{display:flex;gap:22px;flex-wrap:wrap;margin-bottom:12px;font-size:0.9rem}
.summary b{color:var(--green-d)}
.statline{display:block;color:var(--muted);font-size:0.74rem;line-height:1.25;white-space:nowrap}
.statcell{vertical-align:top!important;position:relative}
.cellshade{margin:-6px -4px;padding:6px 4px;min-height:54px;border-radius:4px}
.sig-up{background:#e5f1e7;box-shadow:inset 0 0 0 1px #c9dfce}
.sig-down{background:#f6e3df;box-shadow:inset 0 0 0 1px #e8c8c2}
.not-sig{background:#f3eee4;box-shadow:inset 0 0 0 1px #e4dac8}
.fve-overview{table-layout:fixed;width:100%;font-size:0.82rem}
.fve-overview th,.fve-overview td{padding:6px 3px;overflow:visible}
.fve-overview th:first-child{width:96px}
.fve-overview .statcell{min-width:0}
.cellstats{position:relative;margin-top:2px}
.cellstats summary{cursor:pointer;color:var(--muted);font-size:0.68rem;line-height:1.1;list-style:none}
.cellstats summary::-webkit-details-marker{display:none}
.cellstats summary::before{content:'+';margin-right:2px;color:var(--green-d)}
.cellstats[open] summary::before{content:'−'}
.statpopup{position:absolute;z-index:20;left:50%;transform:translateX(-50%);top:18px;width:230px;
  padding:7px 9px;border:1px solid var(--line);border-radius:5px;background:var(--panel);
  box-shadow:0 5px 14px rgba(44,36,22,.18);text-align:left}
.fve-overview td:nth-last-child(-n+3) .statpopup{left:auto;right:0;transform:none}
.statpopup .statline{white-space:normal;font-size:0.76rem;line-height:1.35}
.inference-compact{table-layout:fixed;width:100%;font-size:0.82rem}
.inference-compact th,.inference-compact td{padding:6px 4px;overflow-wrap:anywhere}
.inference-compact .statline{white-space:normal}
.controls{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-bottom:10px;font-size:0.88rem}
.controls input,.controls select{font-family:inherit;font-size:0.88rem;padding:5px 9px;border:1px solid var(--line);
  border-radius:5px;background:#fff;color:var(--ink)}
.controls input{width:230px}
.pageinfo{color:var(--muted)}
.pager{display:flex;gap:8px;align-items:center;justify-content:center;margin-top:12px}
.pager button{font-family:inherit;font-size:0.88rem;padding:5px 13px;border:1px solid var(--line);background:#fff;
  color:var(--ink-soft);border-radius:5px;cursor:pointer}
.pager button:disabled{color:var(--pending);cursor:default}
.row{border:1px solid var(--line);border-radius:6px;margin-bottom:7px;background:#fff;overflow:hidden}
.rowhead{display:flex;align-items:center;gap:12px;padding:9px 12px;cursor:pointer}
.rowhead:hover{background:#fbf7f0}
.rid{color:var(--muted);font-variant-numeric:tabular-nums;min-width:52px;font-size:0.85rem}
.chip{font-family:'SFMono-Regular',Menlo,Consolas,monospace;font-size:0.82rem;background:var(--hl);
  border:1px solid #e6d9b8;border-radius:4px;padding:1px 7px;color:#6b4e18}
.rmeta{color:var(--muted);font-size:0.83rem;flex:1}
.rbadge{font-size:0.78rem;color:var(--terra)}
.caret{color:var(--pending);font-size:0.8rem}
.rowbody{padding:4px 14px 14px;border-top:1px solid var(--line-soft)}
.block{margin:12px 0}
.twocol{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media(max-width:820px){.twocol{grid-template-columns:1fr}}
.tokenstrip{font-family:'SFMono-Regular',Menlo,Consolas,monospace;font-size:0.82rem;background:#fbfaf7;
  border:1px solid var(--line-soft);border-radius:5px;padding:8px 10px;line-height:1.7;white-space:pre-wrap;word-break:break-word}
.inputwin,.expl{font-size:0.9rem;white-space:pre-wrap;word-break:break-word;background:#fbfaf7;
  border:1px solid var(--line-soft);border-radius:5px;padding:9px 11px;max-height:230px;overflow:auto}
.expl.full,.inputwin.full{max-height:none}
mark.tok{background:#ffe08a;color:#5a3d00;padding:0 2px;border-radius:2px;font-weight:600}
.rm{background:#f7ddd9;color:#7a2820;text-decoration:line-through;text-decoration-color:#c06;border-radius:2px}
.kp{background:#e2efe2;border-radius:2px}
.add{background:#e2efe2;color:#26492f;text-decoration:underline;text-decoration-color:#3d6648;text-decoration-thickness:2px;border-radius:2px}
.fvechip{font-size:0.74rem;color:var(--pending);font-style:italic;border:1px dashed var(--pending);
  border-radius:10px;padding:1px 8px;margin-left:6px}
.linkbtn{background:none;border:none;color:var(--green-d);font-family:inherit;font-size:0.82rem;
  cursor:pointer;padding:2px 0;text-decoration:underline}
.loading{color:var(--muted);font-style:italic;padding:20px}
"""

HEAD = """<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>__TITLE__</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=EB+Garamond:ital,wght@0,400;0,500;0,600;0,700;1,400&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"
 onload="renderMathInElement(document.body,{delimiters:[{left:'$$',right:'$$',display:true},{left:'$',right:'$',display:false}],throwOnError:false});"></script>
<style>__CSS__</style></head><body><div class="container">
"""


def load(name):
    p = RES / name
    return json.loads(p.read_text()) if p.exists() else None


def inference():
    return load("dashboard_inference.json") or {"baseline": {}, "arms": {}, "contrasts": {}}


def p_label(test: dict | None, include_q: bool = True) -> str:
    if not test or test.get("log10_p") is None:
        return "p=&mdash;"
    def one(prefix: str, ordinary, log10):
        if log10 == float("-inf"):
            return f"{prefix}&lt;10<sup>&minus;300</sup>"
        if log10 < -300:
            return f"{prefix}&asymp;10<sup>{log10:.0f}</sup>"
        if ordinary is not None and ordinary >= 0.001:
            return f"{prefix}={ordinary:.3f}"
        return f"{prefix}={10 ** log10:.2e}"
    parts = [one("p", test.get("p"), test["log10_p"])]
    if include_q and test.get("log10_q") is not None:
        parts.append(one("BH q", test.get("q"), test["log10_q"]))
    return " &middot; ".join(parts)


def ci_label(ci, digits: int = 3, signed: bool = False) -> str:
    if not ci or ci[0] is None:
        return "[&mdash;]"
    spec = f"{'+' if signed else ''}.{digits}f"
    return f"[{format(ci[0], spec)}, {format(ci[1], spec)}]"


def significance_class(test: dict | None, effect: float) -> str:
    """Use BH q when present, otherwise the raw p-value (baseline family)."""
    if not test:
        return "not-sig"
    probability = test.get("q") if test.get("log10_q") is not None else test.get("p")
    if probability is None or probability >= 0.05:
        return "not-sig"
    return "sig-up" if effect >= 0 else "sig-down"


def head(title: str) -> str:
    return HEAD.replace("__TITLE__", title).replace("__CSS__", CSS)


def nav(active_slug: str) -> str:
    items = ['<a class="%s" href="index.html">Overview</a>' %
             ("on" if active_slug == "index" else "")]
    warm_ready = (WARM / "dashboard.html").exists()
    items.append('<a class="%s" href="../warm_start_robustness/dashboard.html">Warm start</a>' %
                 ("" if warm_ready else "off"))
    for a in ARMS:
        ready = records_ready(a["armkey"])
        cls = "on" if a["slug"] == active_slug else ("" if ready else "off")
        items.append('<a class="%s" href="%s.html">%s</a>' % (cls, a["slug"], a["tab"]))
    # Exp 7 (resample stability) is a standalone page (exp7_dashboard.py), not an edit-arm; link it too.
    exp7_ready = any((Path("/data/agastyas/nla_fve_ufw") / f"out_{m['key']}" / "exp7_clusters_main.json").exists()
                     or (RES / f"out_{m['key']}" / "exp7_clusters_main.json").exists()
                     or (HERE / f"out_{m['key']}" / "exp7_clusters_main.json").exists() for m in MODELS)
    items.append('<a class="%s" href="exp7_resample.html">Exp 7</a>' %
                 ("on" if active_slug == "exp7_resample" else ("" if exp7_ready else "off")))
    return "<nav>" + "".join(items) + "</nav>"


def records_ready(armkey: str) -> bool:
    return any(records_usable(armkey, m["key"]) for m in MODELS)


def records_usable(armkey: str, key: str) -> bool:
    """Reject stale Exp-4 prototypes that lack the registered policy metadata."""
    if not (RES / f"records_{armkey}_{key}.json").exists():
        return False
    if armkey not in {"exp4a", "exp4b"}:
        return True
    summary = load(f"{armkey}_summary_{key}.json") or {}
    common = summary.get("residual_quote_datapoints") == 0 and summary.get("new_cjk_datapoints") == 0
    if armkey == "exp4a":
        return common and summary.get("policy_version") == 1 and summary.get("rewrite_policy") == "delimiter_only_v1"
    return (common and summary.get("policy_version") == 5
            and summary.get("rewrite_policy") == "semantic_paraphrase_with_short_literal_wrappers_v1"
            and summary.get("verbatim_phrase_violations") == 0)


def fve_cell(armkey: str, key: str) -> str:
    s = inference().get("arms", {}).get(armkey, {}).get(key)
    if not s:
        return '<span class="blank">&middot;</span>'
    v = s["FVE_arm"]
    cls = "fveneg" if v < 0 else ("fvepos" if v >= 0.5 else "fvemid")
    shade = significance_class(s.get("p_test"), s["dFVE"])
    return (f'<div class="statcell cellshade {shade}"><span class="{cls}">{v:.3f}</span>'
            f'<details class="cellstats"><summary>CI / p</summary><div class="statpopup">'
            f'<span class="statline">FVE 95% CI {ci_label(s.get("FVE_arm_ci95"))}</span>'
            f'<span class="statline">&Delta;FVE={s["dFVE"]:+.3f}</span>'
            f'<span class="statline">{p_label(s.get("p_test"))}</span>'
            f'</div></details></div>')


def remaining_contrasts_panel() -> str:
    contrasts = load("remaining_contrasts.json")
    if not contrasts:
        return ""

    def cell(result: dict) -> str:
        value = result["dFVE"]
        lo, hi = result["dFVE_ci95"]
        css = "fvepos" if value > 0 else "fveneg"
        return (f'<span class="{css}">{value:+.3f}</span> '
                f'<span class="statline">95% CI [{lo:+.3f}, {hi:+.3f}]</span>'
                f'<span class="statline">{p_label(result.get("p_test"))}</span>')

    rows = ""
    for model in MODELS:
        values = contrasts[model["key"]]
        rows += (f'<tr><td><span class="accent">{model["label"]}</span></td>'
                 f'<td>{cell(values["exp5b_full"])}</td>'
                 f'<td>{cell(values["exp5c_full"])}</td>'
                 f'<td>{cell(values["exp5c_minus_exp5a"])}</td>'
                 f'<td>{cell(values["exp6p_high_div_changed"])}</td>'
                 f'<td>{cell(values["exp6p_minus_a2_high_div_changed"])}</td>'
                 f'<td>{cell(values["exp6t_wrong"])}</td></tr>')
    return f"""
<div class="panel">
  <h2>Completed robustness contrasts</h2>
  <table class="inference-compact"><thead><tr><th>Model</th><th>5b &Delta;FVE</th><th>5c &Delta;FVE</th>
  <th>5c&minus;5a</th><th>P high-div changed</th><th>P&minus;A2 high-div</th>
  <th>true future, AV-wrong</th></tr></thead><tbody>{rows}</tbody></table>
  <p class="note" style="margin-top:9px">20,000-replicate document-bootstrap 95% CIs; two-sided paired
  tests of &Delta;FVE=0; BH q-values adjust across the displayed contrast family. P columns use rows where the
  AV LCR quote was high-divergence and the paraphrase changed the explanation. The true-future column
  uses only rows where no original AV continuation candidate was semantically correct.</p>
</div>"""


def warm_start_summary() -> dict | None:
    metrics_path = WARM / "results" / "rl_metrics.jsonl"
    manifest_path = WARM / "RUN_MANIFEST.yaml"
    if not metrics_path.exists() or not manifest_path.exists():
        return None
    rows = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
    if not rows:
        return None
    manifest = yaml.safe_load(manifest_path.read_text())
    eval_rows = [row for row in rows if "eval/fve_pct" in row]
    return {
        "completed": int(rows[-1]["step"]) + 1,
        "total": int(manifest.get("planned_training", {}).get("rl", {}).get("iterations", 780)),
        "eval": eval_rows[-1] if eval_rows else None,
        "condition": "implausible-initialized",
    }


def warm_start_overview_panel() -> str:
    summary = warm_start_summary()
    if not summary:
        return ""
    ev = summary["eval"]
    eval_text = (f'<span class="fve">{ev["eval/fve_pct"]:.1f}%</span> '
                 f'<span class="statline">held-out FVE at step {ev["step"]}; '
                 f'{100 * ev["eval/extraction_rate"]:.1f}% extraction</span>'
                 if ev else '<span class="pending">first evaluation pending</span>')
    state = "complete" if summary["completed"] >= summary["total"] else "running"
    return f"""
<div class="panel">
  <h2>Warm-start robustness <span class="badge">{state}</span></h2>
  <table><tbody><tr><td><span class="accent">Qwen2.5-7B L20</span>
  <span class="statline">{summary['condition']} &middot; new 20k-document run</span></td>
  <td>{eval_text}</td><td><b>{summary['completed']} / {summary['total']}</b>
  <span class="statline">RL steps recorded</span></td></tr></tbody></table>
  <p class="note" style="margin-top:9px">The released plausible NLA is an external, non-matched
  reference because it used a larger training recipe. Reconstruction does not by itself establish
  plausible or convergent explanations. <a href="../warm_start_robustness/dashboard.html" class="accent">
  Open the live warm-start curve and provenance.</a></p>
</div>"""


def exp7_complete() -> bool:
    metrics = load("exp7_metrics.json")
    if not metrics or any(m["key"] not in metrics for m in MODELS):
        return False
    for model in MODELS:
        od = HERE / f'out_{model["key"]}'
        if not (od / "exp7_h5.json").exists():
            return False
        for arm in ("main", "floor", "shuffle"):
            raw = od / f"resample_{arm}.json"
            clusters = od / f"exp7_clusters_{arm}.json"
            if not raw.exists() or not clusters.exists():
                return False
            try:
                if json.loads(raw.read_text()).get("n_docs") != 200:
                    return False
            except (OSError, json.JSONDecodeError):
                return False
    return True


def exp7_overview_panel() -> str:
    if not exp7_complete():
        return ""
    metrics = load("exp7_metrics.json")
    rows = []
    for model in MODELS:
        m = metrics[model["key"]]
        for tag in ("genre", "LCR", "PC"):
            c = m["contrasts"][tag]
            delta = c.get("delta_vs_shuffle", {}).get("ci")
            h5 = m.get("greedy_in_core", {}).get(tag, {})
            qfloor = m.get("quality", {}).get("floor", {}).get("0.4", {})
            fmt = lambda ci: ("&mdash;" if not ci or ci[0] is None
                              else f'{ci[0]:.2f} <span class="statline">95% CI [{ci[1]:.2f},{ci[2]:.2f}]</span>')
            dfmt = lambda row: (fmt(row.get("ci")) + f'<span class="statline">{p_label(row.get("p_test"))}</span>')
            rows.append(f'<tr><td>{model["label"]}</td><td>{tag}</td>'
                        f'<td>{fmt(c.get("main_core"))}</td><td>{fmt(c.get("shuffle_core"))}</td>'
                        f'<td>{dfmt(c.get("delta_vs_shuffle", {}))}</td>'
                        f'<td>{fmt(h5.get("doc_mean"))}<span class="statline">{p_label(h5.get("p_test_vs_one"))} vs 1</span></td>'
                        f'<td>{100*qfloor.get("valid_rate", 0):.1f}% '
                        f'<span class="statline">Wilson 95% CI {ci_label(qfloor.get("valid_rate_ci95"), 3)}</span></td></tr>')
    return (f'<div class="panel"><h2>Exp 7 &mdash; activation-specific resample stability</h2>'
            f'<table class="matrix inference-compact"><thead><tr><th>Model</th><th>claim type</th><th>main core</th>'
            f'<th>shuffled-act core</th><th>main&minus;shuffle</th><th>greedy in T=.4 core</th>'
            f'<th>no-inj valid</th></tr></thead><tbody>{"".join(rows)}</tbody></table>'
            f'<p class="note" style="margin-top:9px">Semantic core threshold = 0.5. The inferential p/q '
            f'values test paired main&minus;shuffle=0; H5 tests mean inclusion=1. The no-injection validity '
            f'rate is a quality diagnostic with a Wilson interval, not a hypothesis test, because no '
            f'validity threshold was preregistered. Malformed/CJK '
            f'outputs make its survivor-only core estimate low-powered. '
            f'<a href="exp7_resample.html" class="accent">Open Exp 7 for temperature curves, controls, '
            f'entropy, H5, and resample-level claims.</a></p></div>')


# ── Overview ─────────────────────────────────────────────────────────────────
def build_index() -> None:
    cols = "".join(f"<th>{a['tab']}</th>" for a in ARMS)
    rows = ""
    for m in MODELS:
        b = load(f"fve_{m['key']}.json")
        bi = inference().get("baseline", {}).get(m["key"])
        bshade = significance_class(bi.get("FVE_test"), bi["FVE"]) if bi else "not-sig"
        bcell = (f'<div class="statcell cellshade {bshade}"><span class="fve">{bi["FVE"]:.3f}</span>'
                 f'<details class="cellstats"><summary>CI / p</summary><div class="statpopup">'
                 f'<span class="statline">FVE 95% CI {ci_label(bi.get("FVE_ci95"))}</span>'
                 f'<span class="statline">{p_label(bi.get("FVE_test"), False)} vs FVE=0</span>'
                 f'</div></details></div>'
                 if bi else '<span class="blank">&middot;</span>')
        cells = "".join(f"<td>{fve_cell(a['armkey'], m['key'])}</td>" for a in ARMS)
        rows += (f'<tr><td><span class="accent">{m["label"]}</span> '
                 f'<span class="note">L{m["layer"]}</span></td><td>{bcell}</td>{cells}</tr>')
    lb = load(f"fve_{LLAMA['key']}.json")
    li = inference().get("baseline", {}).get(LLAMA["key"])
    lshade = significance_class(li.get("FVE_test"), li["FVE"]) if li else "not-sig"
    lcell = (f'<div class="statcell cellshade {lshade}"><span class="fve">{li["FVE"]:.3f}</span>'
             f'<details class="cellstats"><summary>CI / p</summary><div class="statpopup">'
             f'<span class="statline">FVE 95% CI {ci_label(li.get("FVE_ci95"))}</span>'
             f'<span class="statline">{p_label(li.get("FVE_test"), False)} vs FVE=0</span>'
             f'</div></details></div>'
             if li else '<span class="pending">running…</span>')
    rows += (f'<tr><td><span class="accent">{LLAMA["label"]}</span> <span class="note">L{LLAMA["layer"]} '
             f'&middot; baseline only</span></td><td>{lcell}</td>'
             + "".join('<td><span class="blank">n/a</span></td>' for _ in ARMS) + "</tr>")

    warm = warm_start_summary()
    if warm:
        warm_status = ('<span class="badge">done</span>' if warm["completed"] >= warm["total"]
                       else '<span class="badge">running</span>')
    else:
        warm_status = '<span class="badge grey">planned</span>'
    legend = ('<tr><td><a href="../warm_start_robustness/dashboard.html" class="accent" '
              'style="text-decoration:none">Warm start &mdash; standard vs implausible initialization</a> '
              '<span class="tag">GPU + API</span></td><td>' + warm_status + '</td>'
              '<td class="note">Does reconstruction training erase the semantics imposed by a radically '
              'different supervised warm start?</td></tr>')
    for a in ARMS:
        ready = records_ready(a["armkey"])
        done = all(load(f"fve_{a['armkey']}_{m['key']}.json") for m in MODELS)
        if done:
            st = '<span class="badge">done</span>'
        elif ready:
            st = '<span class="badge">records ready &middot; FVE pending</span>'
        elif "queued" in a["title"]:
            st = '<span class="badge grey">queued</span>'
        else:
            st = '<span class="badge grey">planned</span>'
        legend += (f'<tr><td><a href="{a["slug"]}.html" class="accent" style="text-decoration:none">'
                   f'{a["tab"]} &mdash; {a["title"]}</a> <span class="tag">{a["kind"]}</span></td>'
                   f'<td>{st}</td><td class="note">{a["q"]}</td></tr>')
    if exp7_complete():
        exp7_status = '<span class="badge">done</span>'
    elif any((HERE / f'out_{m["key"]}' / "resample_main.json").exists() for m in MODELS):
        exp7_status = '<span class="badge">running</span>'
    else:
        exp7_status = '<span class="badge grey">planned</span>'
    legend += ('<tr><td><a href="exp7_resample.html" class="accent" style="text-decoration:none">'
               'Exp 7 &mdash; resample stability: activation signal vs sampler noise</a> '
               '<span class="tag">GPU + API</span></td><td>' + exp7_status + '</td>'
               '<td class="note">Does a meaning-level core recur across resamples, exceed the no-injection '
               'floor, and change when a different document activation is injected?</td></tr>')

    html = head("NLA FVE Robustness &mdash; Overview") + nav("index") + f"""
<h1>NLA Reconstruction Robustness</h1>
<div class="subtitle">Fraction of variance explained (FVE) of the open NLAs, and how it moves under
manipulations of the verbalizer&rsquo;s explanation &mdash; held-out UltraFineWeb, n=2000.</div>

<div class="panel">
  <h2>FVE &mdash; baseline and every experiment</h2>
  <table class="matrix fve-overview"><thead><tr><th>Model</th><th>Baseline</th>{cols}</tr></thead><tbody>
  {rows}
  </tbody></table>
  <p class="note" style="margin-top:9px">Cells show the post-manipulation FVE (blank = not yet run).
  Each arm edits only the explanation text and re-runs the AR against the same frozen activation, so
  the value is comparable to Baseline. Arm p/q values test the paired &Delta;FVE against zero; baseline
  p-values test FVE=0. Open <b>CI / p</b> in any cell for its interval and tests.
  <span class="sig-up" style="padding:1px 5px">green = significant increase</span>,
  <span class="sig-down" style="padding:1px 5px">red = significant decrease</span>, and
  <span class="not-sig" style="padding:1px 5px">sand = not significant</span>, using BH q&lt;.05 for
  manipulation arms. $$\\mathrm{{FVE}} = 1 - \\tfrac{{\\sum_i 2(1-\\cos_i)}}{{\\sum_i \\lVert \\hat h_i - \\bar h\\rVert^2}}$$</p>
</div>

<div class="panel"><h2>Statistical convention</h2>
<p class="note">All aggregate uncertainty uses held-out documents as the independent unit. Intervals are
seeded 20,000-replicate percentile document bootstraps. Manipulation p-values are two-sided tests of the
paired document-level estimating equation for &Delta;FVE=0; BH q-values control false discovery rate
within the 48 model&times;manipulation family or the explicitly displayed secondary-contrast family.
Sample sizes, model layers, temperatures, edit counts, and single-document FVE/cosine contributions are
exact design or diagnostic quantities, not population estimands, so no artificial p-value is assigned.</p></div>

{warm_start_overview_panel()}

{exp7_overview_panel()}

{remaining_contrasts_panel()}

<div class="panel">
  <h2>Experiments</h2>
  <table><thead><tr><th>Experiment</th><th>Status</th><th>Question</th></tr></thead><tbody>{legend}</tbody></table>
  <p class="note" style="margin-top:9px">Open a tab to inspect every datapoint: the input, the exact read
  token, and the original vs. edited explanation with per-point cosine and FVE.</p>
</div>

<footer>Models: Qwen2.5-7B-L20, Gemma-3-12B-L32, Gemma-3-27B-L41 (manipulation series);
Llama-3.3-70B-L53 baseline-only. Built {BUILD_DATE}.</footer>
</div></body></html>"""
    (HERE / "index.html").write_text(html, encoding="utf-8")


# ── Generalized per-datapoint experiment tab ─────────────────────────────────
JS_APP = r"""
const esc=s=>(s==null?'':String(s)).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
let DATA=[],VIEW=[],KEY='',PAGE=0,PER=25;
const loaded={};
function q(id){return document.getElementById(id);}
function prob(t,k){
  if(!t||t['log10_'+k]==null)return k+'=&mdash;';
  const lp=t['log10_'+k],v=t[k];
  if(lp===-Infinity)return k+'&lt;10<sup>&minus;300</sup>';
  if(lp < -300)return k+'&asymp;10<sup>'+lp.toFixed(0)+'</sup>';
  if(v!=null&&v>=.001)return k+'='+v.toFixed(3);
  return k+'='+Math.pow(10,lp).toExponential(2);
}
function ci(a,d=3){return (!a||a[0]==null)?'[&mdash;]':'['+a[0].toFixed(d)+', '+a[1].toFixed(d)+']';}
function gname(key){return 'REC_'+ARMK+'_'+key;}
function loadModel(key){
  KEY=key;
  document.querySelectorAll('.mbtn').forEach(b=>b.classList.toggle('on',b.dataset.k===key));
  if(window[gname(key)]){DATA=window[gname(key)];apply();return;}
  q('rows').innerHTML='<div class="loading">loading '+key+' data …</div>';
  const s=document.createElement('script');
  s.src='expdata/data_'+ARMK+'_'+key+'.js';
  s.onload=()=>{DATA=window[gname(key)];apply();};
  s.onerror=()=>{q('rows').innerHTML='<div class="loading">could not load '+s.src+'</div>';};
  document.body.appendChild(s);
}
function sv(r,k){
  const m=r.edit_meta||{};
  if(k==='chars')return (m.chars_removed!=null?m.chars_removed:Math.abs((r.expl_orig||'').length-(r.expl_edit||'').length));
  if(k==='spans')return m.n_removed||0;
  if(k==='len')return (r.input_display||'').length;
  if(k==='dcos')return (r.cos_edit!=null&&r.cos_orig!=null)?(r.cos_edit-r.cos_orig):1;
  return r.idx;
}
function apply(){
  const Q=q('q').value.trim().toLowerCase(),S=q('sort').value;
  VIEW=DATA.filter(r=>!Q||r.doc_id.includes(Q)||(r.token||'').toLowerCase().includes(Q)||(r.expl_orig||'').toLowerCase().includes(Q));
  const key=S.replace('-',''),desc=S.startsWith('-');
  VIEW.sort((a,b)=>{const x=sv(a,key),y=sv(b,key);return desc?y-x:x-y;});
  PAGE=0;renderSummary();render();
}
function renderSummary(){
  const n=DATA.length;let editStat;const em0=(DATA[0]||{}).edit_meta||{};
  if(em0.arm==='exp4a'){
    let spans=0,delims=0;DATA.forEach(r=>{const m=r.edit_meta||{};spans+=(m.n_dequoted_spans||0);delims+=(m.n_delimiters_removed||0);});
    editStat='<div>avg <b>'+(spans/n).toFixed(2)+'</b> quoted spans &middot; <b>'+(delims/n).toFixed(2)
      +'</b> delimiters removed &middot; literal content preserved byte-for-byte</div>';
  }else if(em0.arm==='exp4b'){
    let spans=0,ch=0,wrap=0;DATA.forEach(r=>{const m=r.edit_meta||{};spans+=(m.n_rewritten||0);ch+=(m.chars_replaced||0);
      (m.quotes||[]).forEach(x=>{wrap+=(x.literal_wrapper?1:0);});});
    editStat='<div>avg <b>'+(spans/n).toFixed(2)+'</b> quoted spans paraphrased &middot; <b>'+(ch/n).toFixed(0)
      +'</b> source chars replaced &middot; <b>'+wrap+'</b> explicit short-literal wrappers</div>';
  }else if(em0.arm==='exp6p'){
    let q=0,ch=0,unch=0,fb=0,dw=0;DATA.forEach(r=>{const m=r.edit_meta||{};q+=(m.n_lcr||0);ch+=(m.chars_removed||0);
      (m.quotes||[]).forEach(x=>{unch+=(Number(x.surface_change||0)===0?1:0);fb+=(x.audit_fallback||x.mechanical_fallback)?1:0;dw+=Math.abs(x.delta_words||0);});});
    editStat='<div>avg <b>'+(q/n).toFixed(2)+'</b> true-context paraphrases &middot; mean |&Delta;words| <b>'+(q?dw/q:0).toFixed(2)
      +'</b> &middot; <b>'+unch+'</b> irreducible/no-op &middot; <b>'+fb+'</b> fail-closed</div>';
  }else if(em0.arm==='exp6t'){
    let pc=0,wrong=0,right=0,un=0;DATA.forEach(r=>{const m=r.edit_meta||{};pc+=(m.n_pc||0);
      if(m.continuation_wrong===true)wrong++;else if(m.continuation_wrong===false)right++;else un++;});
    editStat='<div>avg <b>'+(pc/n).toFixed(2)+'</b> PC slots &rarr; ordered true-future tiles &middot; semantic label: '
      +'<b>'+wrong+'</b> wrong / <b>'+right+'</b> at least one correct / <b>'+un+'</b> unlabelled</div>';
  }else if((em0.arm||'').indexOf('exp6')===0){
    let q=0,ch=0,hi=0,hn=0;DATA.forEach(r=>{const m=r.edit_meta||{};q+=(m.n_lcr||0);ch+=(m.chars_removed||0);
      (m.quotes||[]).forEach(x=>{if(x.divergence!=null){hn++;if(x.divergence>=0.5)hi++;}});});
    editStat='<div>avg <b>'+(q/n).toFixed(2)+'</b> LCR quote(s) &rarr; true pre-token context, <b>'+(ch/n).toFixed(0)+'</b> chars'
      +' &middot; <b>'+(hn?(100*hi/hn).toFixed(0):0)+'%</b> high-divergence (confabulated)</div>';
  }else if(em0.n_kept!=null){
    let k=0,ch=0;DATA.forEach(r=>{k+=(r.edit_meta.n_kept||0);ch+=(r.edit_meta.chars_kept||0);});
    editStat='<div>avg <b>'+(k/n).toFixed(2)+'</b> quotes kept, <b>'+(ch/n).toFixed(0)+'</b> chars</div>';
  }else if(em0.n_removed!=null){
    let sp=0,ch=0;DATA.forEach(r=>{sp+=(r.edit_meta.n_removed||0);ch+=(r.edit_meta.chars_removed||0);});
    editStat='<div>avg <b>'+(sp/n).toFixed(2)+'</b> spans, <b>'+(ch/n).toFixed(0)+'</b> chars edited</div>';
  }else if(em0.kind){
    let u=0,noop=0;DATA.forEach(r=>{u+=(r.edit_meta.n_units||0);noop+=(r.edit_meta.noop?1:0);});
    editStat='<div>avg <b>'+(u/n).toFixed(2)+'</b> '+em0.kind+' units shuffled &middot; <b>'+noop+'</b> no-ops ('+(100*noop/n).toFixed(0)+'%)</div>';
  }else{
    let ch=0;DATA.forEach(r=>{ch+=Math.abs((r.expl_orig||'').length-(r.expl_edit||'').length);});
    editStat='<div>avg <b>'+(ch/n).toFixed(0)+'</b> chars changed</div>';
  }
  const S=(window['SUM_'+ARMK]||{})[KEY];let fve;
  if(S){
    const d=S.dFVE,sg=d>=0?'+':'',col=d<0?'#a65d57':'#3d6648';
    const pq=prob(S.p_test,'p')+' &middot; '+prob(S.p_test,'q');
    fve='<div style="border-left:3px solid '+col+';padding-left:9px">FVE <b>'+S.FVE_base.toFixed(3)+'</b> '
      +'<span class="note">95% CI '+ci(S.FVE_base_ci95)+'</span> &rarr; <b>'+S.FVE_arm.toFixed(3)+'</b> '
      +'<span class="note">95% CI '+ci(S.FVE_arm_ci95)+'</span><br>'
      +'<span style="color:'+col+';font-weight:700">&Delta;'+sg+d.toFixed(3)+'</span> '
      +'<span class="note">95% CI '+ci(S.dFVE_ci95)+' &middot; '+pq+'</span></div>'
      +'<div>mean &Delta;cos <b>'+S.mean_dcos.toFixed(3)+'</b> '
      +'<span class="note">95% CI '+ci(S.mean_dcos_ci95)+' &middot; '+pq+'<br>'
      +'base '+S.mean_cos_base.toFixed(3)+' '+ci(S.mean_cos_base_ci95)+' &rarr; arm '
      +S.mean_cos_arm.toFixed(3)+' '+ci(S.mean_cos_arm_ci95)+'</span></div>';
  }else{fve='<div class="pending">FVE / cos &mdash; pending GPU re-score</div>';}
  q('summary').innerHTML='<div><b>'+n.toLocaleString()+'</b> datapoints <span class="note">(design count)</span></div>'+editStat+fve;
}
function hlToken(t,cs,ce){if(cs==null)return esc(t);
  return esc(t.slice(0,cs))+'<mark class="tok">'+esc(t.slice(cs,ce))+'</mark>'+esc(t.slice(ce));}
function hlSpans(t,spans,cls){
  if(!spans||!spans.length)return esc(t);
  const s=spans.slice().sort((a,b)=>a[0]-b[0]);let out='',prev=0;
  for(const[a,b]of s){if(a<prev)continue;out+=esc(t.slice(prev,a))+'<span class="'+cls+'">'+esc(t.slice(a,b))+'</span>';prev=b;}
  return out+esc(t.slice(prev));}
function tokenStrip(r){if(!r.ctx)return'';
  return r.ctx.map(c=>c.r?'<mark class="tok">'+esc(c.t)+'</mark>':esc(c.t)).join('');}
function inputWin(r){const t=r.input_display||'',cs=r.char_start,ce=r.char_end,W=520;
  if(cs==null)return esc(t.slice(0,1200));const a=Math.max(0,cs-W),b=Math.min(t.length,ce+W);
  return(a>0?'… ':'')+esc(t.slice(a,cs))+'<mark class="tok">'+esc(t.slice(cs,ce))+'</mark>'+esc(t.slice(ce,b))+(b<t.length?' …':'');}
function fchip(v){return v==null?'&mdash; pending':(v>=0?'+':'')+v.toFixed(3);}
function origHtml(r){
  // Exp 2 keeps quotes: highlight kept spans green, strike the removed prose gaps.
  if(r.kept&&r.kept.length){
    const s=r.kept.slice().sort((a,b)=>a[0]-b[0]);let out='',prev=0;const t=r.expl_orig||'';
    for(const[a,b]of s){if(a<prev)continue;
      if(a>prev)out+='<span class="rm">'+esc(t.slice(prev,a))+'</span>';
      out+='<span class="kp">'+esc(t.slice(a,b))+'</span>';prev=b;}
    if(prev<t.length)out+='<span class="rm">'+esc(t.slice(prev))+'</span>';
    return out;
  }
  if(r.removed&&r.removed.length)return hlSpans(r.expl_orig,r.removed,'rm');
  return esc(r.expl_orig);
}
function verdictsHtml(r){
  const qs=(r.edit_meta&&r.edit_meta.quotes)||[];
  if(!qs.length)return '';
  if((r.edit_meta||{}).arm==='exp4b'){
    const rows=qs.map(q=>{
      const f=q.fidelity||{};
      return '<tr style="border-top:1px solid #e7ddc9">'
        +'<td style="padding:3px 7px;max-width:260px;color:#7a2820">'+esc(q.original||'')+'</td>'
        +'<td style="padding:3px 7px;max-width:300px;color:#26492f">'+esc(q.replacement||'')+'</td>'
        +'<td style="padding:3px 7px;white-space:nowrap">'+Number(q.surface_change||0).toFixed(2)
        +(q.literal_wrapper?' &middot; wrapper':'')+'</td>'
        +'<td style="padding:3px 7px;color:#6b5d4d;font-style:italic">'+esc(f.reason||'')+'</td></tr>';
    }).join('');
    return '<div class="block"><h4>Semantic paraphrase audit <span class="note">'+esc(r.edit_meta.model||'')+'</span></h4>'
      +'<table style="width:100%;border-collapse:collapse;font-size:0.82rem;font-variant-numeric:tabular-nums">'
      +'<thead><tr style="text-align:left;color:#6b5d4d"><th style="padding:3px 7px">quoted source</th>'
      +'<th style="padding:3px 7px">replacement</th><th style="padding:3px 7px">surface change</th>'
      +'<th style="padding:3px 7px">fidelity rationale</th></tr></thead><tbody>'+rows+'</tbody></table></div>';
  }
  if(qs[0].reason==null)return '';                    // Exp 3b judge records carry top-level reasons
  const rows=qs.map(q=>{
    const t=(r.expl_orig||'').slice(q.span[0],q.span[1]);
    const der=(q.derivable===true)||q.label==='derivable';
    const col=der?'#a65d57':'#3d6648';
    const lab=der?'derivable &rarr; removed':'divergent &rarr; kept';
    const conf=(q.confidence!=null)?Number(q.confidence).toFixed(2):'';
    return '<tr style="border-top:1px solid #e7ddc9;color:'+col+'">'
      +'<td style="padding:3px 7px;max-width:300px">'+esc(t.slice(0,110))+'</td>'
      +'<td style="padding:3px 7px;white-space:nowrap">'+lab+'</td>'
      +'<td style="padding:3px 7px">'+esc(q.basis||'')+'</td>'
      +'<td style="padding:3px 7px;color:#6b5d4d;font-style:italic">'+esc(q.reason||'')+'</td>'
      +'<td style="padding:3px 7px">'+conf+'</td></tr>';
  }).join('');
  return '<div class="block"><h4>Tier-B judge verdicts <span class="note">'+esc(r.edit_meta.model||'')+'</span></h4>'
    +'<table style="width:100%;border-collapse:collapse;font-size:0.82rem;font-variant-numeric:tabular-nums">'
    +'<thead><tr style="text-align:left;color:#6b5d4d"><th style="padding:3px 7px">quote</th><th style="padding:3px 7px">label</th>'
    +'<th style="padding:3px 7px">basis</th><th style="padding:3px 7px">reason</th><th style="padding:3px 7px">conf</th></tr></thead>'
    +'<tbody>'+rows+'</tbody></table></div>';
}
function isExp6(r){return(((r.edit_meta||{}).arm||'').indexOf('exp6')===0);}
function exp6Html(r){
  const qs=(r.edit_meta&&r.edit_meta.quotes)||[];
  if(!isExp6(r)||!qs.length)return '';
  const arm=(r.edit_meta||{}).arm||'';
  if(arm==='exp6p'){
    const rows=qs.map(q=>'<tr style="border-top:1px solid #e7ddc9">'
      +'<td style="padding:3px 7px;max-width:240px;color:#7a2820">'+esc(q.orig||'')+'</td>'
      +'<td style="padding:3px 7px;max-width:260px">'+esc(q.source_true||'')+'</td>'
      +'<td style="padding:3px 7px;max-width:260px;color:#26492f">'+esc(q.injected||'')+'</td>'
      +'<td style="padding:3px 7px;white-space:nowrap">'+((q.delta_words||0)>=0?'+':'')+(q.delta_words||0)
      +(q.audit_fallback||q.mechanical_fallback?' &middot; fallback':'')+'</td></tr>').join('');
    return '<div class="block"><h4>True context &rarr; semantic paraphrase <span class="note">all outputs audited; ECHO/PC frozen</span></h4>'
      +'<table style="width:100%;border-collapse:collapse;font-size:0.82rem"><thead><tr><th>AV quote</th><th>true A1 slice</th>'
      +'<th>injected paraphrase</th><th>&Delta;words</th></tr></thead><tbody>'+rows+'</tbody></table></div>';
  }
  if(arm==='exp6t'){
    const rows=qs.map(q=>'<tr style="border-top:1px solid #e7ddc9">'
      +'<td style="padding:3px 7px;max-width:300px;color:#7a2820">'+esc(q.orig||'')+'</td>'
      +'<td style="padding:3px 7px;max-width:300px;color:#26492f">'+esc(q.injected||'')+'</td>'
      +'<td style="padding:3px 7px">'+Number(q.lexical_similarity||0).toFixed(2)+'</td></tr>').join('');
    const m=r.edit_meta||{},lab=m.continuation_wrong===true?'AV continuation wrong':(m.continuation_wrong===false?'at least one AV candidate correct':'unlabelled');
    return '<div class="block"><h4>PC &rarr; ordered true-continuation tiles <span class="note">'+esc(lab)+' &middot; '+esc(m.continuation_reason||'')+'</span></h4>'
      +'<table style="width:100%;border-collapse:collapse;font-size:0.82rem"><thead><tr><th>AV PC candidate</th><th>true-future tile</th>'
      +'<th>lexical sim.</th></tr></thead><tbody>'+rows+'</tbody></table></div>';
  }
  const rows=qs.map(q=>{
    const dv=(q.divergence!=null)?Number(q.divergence).toFixed(2):'';
    const col=(q.divergence!=null&&q.divergence>=0.5)?'#a65d57':'#6b5d4d';
    return '<tr style="border-top:1px solid #e7ddc9">'
      +'<td style="padding:3px 7px;max-width:320px;color:#7a2820"><s>'+esc(q.orig||'')+'</s></td>'
      +'<td style="padding:3px 7px;max-width:320px;color:#26492f">'+esc(q.injected||'')+'</td>'
      +'<td style="padding:3px 7px;color:'+col+';white-space:nowrap">'+dv+'</td></tr>';
  }).join('');
  return '<div class="block"><h4>LCR &rarr; true pre-token context <span class="note">'
    +qs.length+' left-context quote(s) replaced &middot; read token frozen</span></h4>'
    +'<table style="width:100%;border-collapse:collapse;font-size:0.82rem;font-variant-numeric:tabular-nums">'
    +'<thead><tr style="text-align:left;color:#6b5d4d"><th style="padding:3px 7px">AV quote (removed)</th>'
    +'<th style="padding:3px 7px">true input (injected)</th><th style="padding:3px 7px">divergence</th></tr></thead>'
    +'<tbody>'+rows+'</tbody></table></div>';
}
const UNIT_TINT=['#e9f0e4','#f5ecd9','#e6ecf5','#f5e2ec','#eef0dd','#dff0ec','#f0e8dd','#e8e2f2'];
function tintFor(i){return UNIT_TINT[((i%UNIT_TINT.length)+UNIT_TINT.length)%UNIT_TINT.length];}
function isShuffle(r){const k=(r.edit_meta||{}).kind;return k==='block'||k==='sentence'||k==='quoted-claim';}
function shufSpans(text,spans){
  if(!spans||!spans.length)return esc(text);
  const s=spans.slice().sort((a,b)=>a[0]-b[0]);let out='',prev=0;
  s.forEach((sp,k)=>{const a=sp[0],b=sp[1],oi=(sp.length>2?sp[2]:k);
    if(a<prev)return; if(a>prev)out+=esc(text.slice(prev,a));
    out+='<span style="background:'+tintFor(oi)+';border-radius:3px;padding:0 3px;box-shadow:inset 0 0 0 1px rgba(0,0,0,.06)">'
      +'<sup style="color:#8a7a63;font-weight:700">'+oi+'</sup>'+esc(text.slice(a,b))+'</span>';
    prev=b;});
  out+=esc(text.slice(prev));return out;
}
function permLine(r){const m=r.edit_meta||{};if(!m.perm)return'';
  const ident='['+(m.units_orig||[]).map((_,i)=>i).join(',')+']';
  return '<div class="note" style="margin:2px 0 6px">'+esc(m.kind)+' shuffle &middot; '+m.n_units+' units &middot; '
    +ident+' &rarr; ['+m.perm.join(',')+']'+(m.noop?' &middot; <b style="color:#a65d57">no-op</b>':'')+'</div>';
}
function rowHtml(r){
  const m=r.edit_meta||{},tok=(r.token||'').trim();
  const sh=isShuffle(r),e6=isExp6(r);
  const oh=sh?shufSpans(r.expl_orig,(m.units_orig||[])):origHtml(r);
  const eh=sh?shufSpans(r.expl_edit,(m.units_edit||[])):
    ((r.added&&r.added.length)?hlSpans(r.expl_edit,r.added,'add'):esc(r.expl_edit));
  const dcos=(r.cos_edit!=null&&r.cos_orig!=null)?(r.cos_edit-r.cos_orig):null;
  const dcell=dcos!=null?' &middot; <span style="color:'+(dcos<-0.02?'#a65d57':'#6b5d4d')+'">&Delta;cos '+dcos.toFixed(3)+'</span>':'';
  const nrm=m.arm==='exp4a'?('&minus;'+m.n_delimiters_removed+' quote delimiters'):
    (m.arm==='exp4b'?(m.n_rewritten+' spans paraphrased'):
    (m.arm==='exp6p'?(m.n_lcr+' true-context paraphrase'+(m.n_lcr===1?'':'s')):
    (m.arm==='exp6t'?(m.n_pc+' PC slot'+(m.n_pc===1?'':'s')+' &rarr; true future'):
    (e6?(m.n_lcr+' quote'+(m.n_lcr===1?'':'s')+' &rarr; true input'):(m.n_kept!=null)?('keep '+m.n_kept+' quotes'):(m.n_removed!=null)?('&minus;'+m.n_removed+' spans'):(m.kind?('shuffle '+m.n_units+(m.noop?' &middot; no-op':'')):(Math.abs((r.expl_orig||'').length-(r.expl_edit||'').length)+' ch Δ'))))));
  const head='<div class="rowhead" onclick="tog(this)">'
    +'<span class="rid">#'+r.idx+'</span>'
    +'<span class="chip">'+esc(tok||'?')+'</span>'
    +'<span class="rmeta">'+esc(r.doc_id)+' &middot; token '+r.position+' / '+r.n_tokens+'</span>'
    +'<span class="rbadge">'+nrm+dcell+'</span>'
    +'<span class="caret">&#9656;</span></div>';
  const body='<div class="rowbody" hidden>'
    +'<div class="block"><h4>Input &mdash; activation taken at token #'+r.position+' = "'+esc(tok)+'"</h4>'
    +'<div class="tokenstrip">'+tokenStrip(r)+'</div>'
    +'<div class="inputwin" style="margin-top:7px">'+inputWin(r)+'</div>'
    +'<button class="linkbtn" onclick="full(this,\''+r.idx+'\')">show full input</button></div>'
    +(sh?permLine(r):'')
    +'<div class="twocol">'
    +'<div class="block"><h4>Original explanation <span class="fvechip">FVE '+fchip(r.fve_i_orig)+'</span></h4>'
    +'<div class="expl">'+oh+'</div></div>'
    +'<div class="block"><h4>Edited explanation <span class="fvechip">FVE '+fchip(r.fve_i_edit)+'</span></h4>'
    +'<div class="expl">'+eh+'</div></div>'
    +'</div>'+exp6Html(r)+verdictsHtml(r)+'</div>';
  return '<div class="row" data-idx="'+r.idx+'">'+head+body+'</div>';
}
function tog(h){const b=h.nextElementSibling;b.hidden=!b.hidden;h.querySelector('.caret').innerHTML=b.hidden?'&#9656;':'&#9662;';}
function full(btn,idx){const r=DATA.find(x=>String(x.idx)===String(idx));const win=btn.previousElementSibling;
  if(btn.dataset.on){win.innerHTML=inputWin(r);btn.textContent='show full input';btn.dataset.on='';}
  else{win.innerHTML=hlToken(r.input_display,r.char_start,r.char_end);win.classList.add('full');btn.textContent='show window';btn.dataset.on='1';}}
function render(){
  const tot=VIEW.length,pages=Math.max(1,Math.ceil(tot/PER));if(PAGE>=pages)PAGE=pages-1;
  const a=PAGE*PER,b=Math.min(tot,a+PER);
  q('rows').innerHTML=VIEW.slice(a,b).map(rowHtml).join('')||'<div class="loading">no matches</div>';
  q('pageinfo').textContent=tot?('showing '+(a+1)+'–'+b+' of '+tot.toLocaleString()):'0';
  q('prev').disabled=PAGE<=0;q('next').disabled=PAGE>=pages-1;
}
function pg(d){PAGE+=d;render();window.scrollTo(0,0);}
window.addEventListener('DOMContentLoaded',()=>{
  q('q').addEventListener('input',apply);q('sort').addEventListener('change',apply);
  q('prev').addEventListener('click',()=>pg(-1));q('next').addEventListener('click',()=>pg(1));
  loadModel(FIRST_KEY);
});
"""


def build_exp_tab(arm) -> None:
    armkey = arm["armkey"]
    DATA.mkdir(exist_ok=True)
    present = [m for m in MODELS if records_usable(armkey, m["key"])]

    if not present:                                   # placeholder tab
        queued = "queued" in arm["title"]
        html = head(f'{arm["tab"]} &mdash; {arm["title"]}') + nav(arm["slug"]) + f"""
<h1>{arm["tab"]} &mdash; {arm["title"]}</h1>
<div class="subtitle">{'Queued &mdash; scheduled after Exp 5.' if queued else 'Planned &mdash; not yet run.'}
<span class="tag">{arm["kind"]}</span></div>
<div class="panel"><p class="note">{arm["q"]}</p>
<p class="note">This tab will populate with every datapoint (input, exact read token, original vs.
edited explanation, per-point cosine and FVE) once the edit step and AR re-score run &mdash; same
format as Exp 1. See <code>EXPERIMENT_PLAN.md</code>.</p></div>
</div></body></html>"""
        (HERE / f'{arm["slug"]}.html').write_text(html, encoding="utf-8")
        return

    for m in present:                                 # per-model data files
        recs = load(f"records_{armkey}_{m['key']}.json")
        (DATA / f"data_{armkey}_{m['key']}.js").write_text(
            "window.REC_%s_%s=%s;" % (armkey, m["key"], json.dumps(recs, ensure_ascii=False)),
            encoding="utf-8")
    inf = inference().get("arms", {}).get(armkey, {})
    summaries = {m["key"]: inf.get(m["key"]) for m in present}
    summaries = {k: v for k, v in summaries.items() if v}

    mbtns = "".join('<button class="mbtn" data-k="%s" onclick="loadModel(\'%s\')">%s</button>'
                    % (m["key"], m["key"], m["label"]) for m in present)
    inject = (f'<script>const ARMK="{armkey}";const FIRST_KEY="{present[0]["key"]}";'
              f'window["SUM_{armkey}"]={json.dumps(summaries)};</script>')
    result_note = (f'<p class="note" style="margin-top:9px"><b>{arm["result"]}</b></p>'
                   if arm.get("result") else "")

    html = head(f'{arm["tab"]} &mdash; {arm["title"]}') + nav(arm["slug"]) + f"""
<h1>{arm["tab"]} &mdash; {arm["title"]}</h1>
<div class="subtitle">{arm["q"]} <span class="tag">{arm["kind"]}</span></div>

<div class="panel">
  <div class="mtoggle">{mbtns}</div>
  <div id="summary" class="summary"></div>
  <p class="note">Aggregate intervals and tests use documents as the sampling unit. Edit counts and the
  single-datapoint values below are exact pipeline diagnostics, not separate hypothesis tests.</p>
  {result_note}
  <div class="controls">
    <input id="q" placeholder="filter: doc id, read token, or explanation text">
    <label>sort
      <select id="sort">
        <option value="idx">index</option>
        <option value="dcos">&Delta;cos (biggest drop)</option>
        <option value="-chars">chars edited &darr;</option>
        <option value="-spans">spans edited &darr;</option>
        <option value="-len">input length &darr;</option>
      </select></label>
    <span id="pageinfo" class="pageinfo"></span>
  </div>
  <div id="rows"></div>
  <div class="pager"><button id="prev">&larr; prev</button><button id="next">next &rarr;</button></div>
</div>

<footer>n=2000 held-out UltraFineWeb docs per model &middot; read position &ge;50 tokens &middot;
AV temp 0.4. Built {BUILD_DATE}.</footer>
{inject}
<script>{JS_APP}</script>
</div></body></html>"""
    (HERE / f'{arm["slug"]}.html').write_text(html, encoding="utf-8")


def cleanup_stale() -> None:
    # exp7_resample.html is a standalone page (built by exp7_dashboard.py), not an ARMS tab; keep it.
    keep = {"index.html", "exp7_resample.html"} | {f'{a["slug"]}.html' for a in ARMS}
    for p in HERE.glob("exp*_*.html"):
        if p.name not in keep:
            p.unlink()


def main() -> None:
    cleanup_stale()
    build_index()
    for a in ARMS:
        build_exp_tab(a)
    ready = [a["tab"] for a in ARMS if records_ready(a["armkey"])]
    print(f"wrote index.html + {len(ARMS)} experiment tabs; populated: {ready or '(none beyond scaffold)'}")


if __name__ == "__main__":
    main()
