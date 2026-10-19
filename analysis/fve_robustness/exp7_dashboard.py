"""Exp 7 — resample-stability dashboard (standalone HTML, matches the FVE-robustness style).

Reads whatever Exp 7 artifacts exist and renders:
  - an intro (the signal-vs-noise idea + the stable x not-in-input 2x2 + the floor/shuffle controls),
  - a summary per model: core-fraction by tag x temperature (with doc-bootstrap CIs), the consensus
    histogram (H1 bimodality), and the T=0.4 contrasts main vs floor vs shuffle vs permutation-null,
  - per-datapoint resample-spread CARDS: for each temperature, the meaning-clusters as chips sized by
    consensus frequency (core = green, periphery = grey), split by tag (genre / LCR / PC), with the raw
    N resamples on expand.

Degrades to a "pilot pending" placeholder when no data is present yet, so the tab + nav work immediately.

Usage:  python3 exp7_dashboard.py [--cards 40]
"""
from __future__ import annotations

import argparse
import html
import json
import math
import re
from pathlib import Path

from inference_stats import exact_binomial_p_half, wilson_interval

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
KEYS = ["qwen7b", "gemma12b", "gemma27b"]
LABELS = {"qwen7b": "Qwen2.5-7B-L20", "gemma12b": "Gemma-3-12B-L32", "gemma27b": "Gemma-3-27B-L41"}
TAGS = ["genre", "LCR", "PC"]
TAGDESC = {"genre": "genre / register line", "LCR": "left-context restatement",
           "PC": "predicted continuation"}


def _outdir(key: str) -> Path:
    for cand in (Path("/data/agastyas/nla_fve_ufw") / f"out_{key}", RES / f"out_{key}", HERE / f"out_{key}"):
        if (cand / "exp7_clusters_main.json").exists():
            return cand
    return RES / f"out_{key}"


def _rd(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def esc(s: str) -> str:
    return html.escape(s or "")


def shared_style_nav() -> tuple[str, str, str]:
    h = (HERE / "index.html").read_text()
    style = re.search(r"<style>.*?</style>", h, re.S).group(0)
    # the font/katex <link>+<script> block lives in <head> before <style>
    headlinks = "".join(re.findall(r'(<link[^>]+>|<script[^>]*>.*?</script>)', h[:h.index("<style>")], re.S))
    from build_experiments_dashboard import nav
    return style, nav("exp7_resample"), headlinks


def freq_bar(freq: float, core: bool) -> str:
    w = max(4, round(freq * 100))
    col = "var(--green)" if core else "var(--pending)"
    return (f'<span style="display:inline-block;width:46px;height:7px;background:var(--line);'
            f'border-radius:4px;overflow:hidden;vertical-align:middle;margin-right:5px">'
            f'<span style="display:block;height:100%;width:{w}%;background:{col}"></span></span>')


def cluster_label(cluster: dict, claims: list[dict]) -> str:
    texts = [claims[i]["text"] for i in cluster["members"] if i < len(claims)]
    return max(texts, key=len) if texts else "(empty)"


def p_label(test: dict | None, q: bool = True) -> str:
    if not test or test.get("log10_p") is None:
        return "p=&mdash;"

    def one(label, value, log10):
        if log10 == -math.inf:
            return f"{label}&lt;10<sup>&minus;300</sup>"
        if log10 < -300:
            return f"{label}&asymp;10<sup>{log10:.0f}</sup>"
        if value is not None and value >= 0.001:
            return f"{label}={value:.3f}"
        return f"{label}={10 ** log10:.2e}"

    parts = [one("p", test.get("p"), test["log10_p"])]
    if q and test.get("log10_q") is not None:
        parts.append(one("BH q", test.get("q"), test["log10_q"]))
    return " &middot; ".join(parts)


def render_clusters(tagblock: dict, effective_n: int, thr: float = 0.5) -> str:
    claims = tagblock.get("claims", [])
    meaning = tagblock.get("meaning", [])
    if not meaning:
        return '<span class="pending">&mdash;</span>'
    chips = []
    for c in meaning:
        core = c["freq"] >= thr
        lab = esc(cluster_label(c, claims))
        if len(lab) > 90:
            lab = lab[:88] + "&hellip;"
        successes = int(c["n_resamples"])
        ci = wilson_interval(successes, effective_n)
        bp = exact_binomial_p_half(successes, effective_n)
        stat = (f'Wilson 95% CI [{ci[0]:.2f},{ci[1]:.2f}] &middot; exact p<sub>0.5</sub>={bp:.3g}'
                if ci is not None and bp is not None else "uncertainty unavailable")
        chips.append(
            f'<div style="margin-bottom:4px;font-size:0.86rem">{freq_bar(c["freq"], core)}'
            f'<span style="color:{"var(--green-d)" if core else "var(--muted)"};font-weight:{600 if core else 400}">'
            f'{c["freq"]:.2f}</span> &middot; {lab} '
            f'<span class="note">({successes}/{effective_n}; {stat})</span></div>')
    return "".join(chips)


def render_card(block_by_T: dict, grid: list, idx: int, meta: dict, raw: dict) -> str:
    head = (f'<div style="font-weight:600;color:var(--ink-strong)">#{idx} &middot; '
            f'read token <code>{esc(meta.get("token",""))}</code>'
            f'<span class="tag">{meta.get("stratum","?")} divergence</span>'
            f'<span class="note" style="margin-left:8px">div={meta.get("divergence")}</span></div>')
    cols = []
    for T in grid:
        b = block_by_T.get(T)
        if not b:
            continue
        eff = f'{b["effective_N"]}/{b["n_total"]}' + (f' &minus;{b["n_filtered"]}' if b["n_filtered"] else "")
        tag_html = []
        for tag in TAGS:
            tb = b["tags"].get(tag, {})
            tag_html.append(
                f'<div style="margin-top:6px"><span class="tag">{tag}</span>'
                f'<div style="margin-top:3px">{render_clusters(tb, b["effective_N"])}</div></div>')
        cols.append(
            f'<td style="vertical-align:top;min-width:230px">'
            f'<div style="font-weight:600;color:var(--ink-soft)">T={T} '
            f'<span class="note">(eff {eff})</span></div>{"".join(tag_html)}</td>')
    # raw resamples on expand
    raws = raw.get(idx, [])
    rlist = "".join(f'<div style="border-bottom:1px solid var(--line-soft);padding:3px 0">'
                    f'<span class="note">T={r["T"]} #{r["k"]}{" ·CJK" if r.get("cjk") else ""}'
                    f'{" ·no-expl" if r.get("noexp") else ""}</span> {esc(r["text"][:220])}</div>'
                    for r in raws[:60])
    details = (f'<details style="margin-top:8px"><summary class="note">raw resamples ({len(raws)})</summary>'
               f'<div style="max-height:280px;overflow:auto;font-size:0.82rem;margin-top:6px">{rlist}</div></details>'
               if raws else "")
    return (f'<div class="panel">{head}'
            f'<div style="overflow-x:auto;margin-top:8px"><table style="width:auto"><tr>{"".join(cols)}</tr></table></div>'
            f'{details}</div>')


def ci_cell(ci, test: dict | None = None, null_label: str | None = None) -> str:
    if not ci or ci[0] is None:
        return '<span class="pending">&mdash;</span>'
    m, lo, hi = ci
    test_line = (f'<span class="statline">{p_label(test)}'
                 f'{" " + null_label if null_label else ""}</span>' if test else "")
    return f'{m:.2f} <span class="statline">95% CI [{lo:.2f},{hi:.2f}]</span>{test_line}'


def delta_cell(row) -> str:
    if not row:
        return '<span class="pending">&mdash;</span>'
    return (f'{ci_cell(row.get("ci"), row.get("p_test"))} '
            f'<span class="statline">n={row.get("n", 0)} docs</span>')


def hist_svg(buckets: list) -> str:
    lo, mid, hi = buckets
    seg = lambda w, col: (f'<span style="display:inline-block;height:9px;width:{round(w*54)}px;'
                          f'background:{col}"></span>')
    return ('<span title="low / mid / high consensus" style="display:inline-block;border:1px solid var(--line)">'
            f'{seg(lo,"var(--pending)")}{seg(mid,"#d9c9a8")}{seg(hi,"var(--green)")}</span>')


def render_summary(key: str, metrics: dict) -> str:
    m = metrics.get(key)
    if not m:
        return ""
    grid = m["grid"]
    Ts = [T for T in grid if T > 0]
    rows = []
    for tag in TAGS:
        cur = m["curves"][tag]
        core_cells = "".join(
            f"<td>{ci_cell(cur[str(T)]['core'], (cur[str(T)].get('delta_vs_null') or {}).get('p_test'), 'vs permutation null')}</td>"
            for T in Ts)
        hist_cells = "".join(f"<td>{hist_svg(cur[str(T)]['hist'])}</td>" for T in Ts)
        mass_cells = "".join(f"<td>{hist_svg(cur[str(T)].get('hist_mass', cur[str(T)]['hist']))}</td>" for T in Ts)
        rows.append(f'<tr><td><b>{tag}</b> <span class="note">core</span></td>{core_cells}</tr>')
        rows.append(f'<tr><td class="note">cluster-count hist</td>{hist_cells}</tr>')
        rows.append(f'<tr><td class="note">claim-mass hist</td>{mass_cells}</tr>')
    head = "".join(f"<th>T={T}</th>" for T in Ts)
    core_tbl = (f'<table class="matrix"><thead><tr><th>{esc(LABELS[key])}</th>{head}</tr></thead>'
                f'<tbody>{"".join(rows)}</tbody></table>')
    # contrasts at T=0.4
    crows = []
    for tag in TAGS:
        c = m["contrasts"][tag]
        crows.append(f'<tr><td>{tag}</td><td>{ci_cell(c["main_core"])}</td>'
                     f'<td>{ci_cell(c.get("floor_core"))}</td><td>{ci_cell(c.get("shuffle_core"))}</td>'
                     f'<td>{ci_cell(c.get("null_core"))}</td><td>{delta_cell(c.get("delta_vs_floor"))}</td>'
                     f'<td>{delta_cell(c.get("delta_vs_shuffle"))}</td>'
                     f'<td>{delta_cell(c.get("delta_vs_null"))}</td></tr>')
    contr = (f'<table class="matrix" style="margin-top:10px"><thead><tr><th>tag @ T=0.4</th>'
             f'<th>main core</th><th>no-inj floor</th><th>shuffled-act</th><th>permutation null</th>'
             f'<th>main&minus;floor</th><th>main&minus;shuffle</th><th>main&minus;null</th></tr></thead>'
             f'<tbody>{"".join(crows)}</tbody></table>')

    # H2: paired document-level entropy changes between the lowest and highest sampled T.
    erows = []
    if len(Ts) >= 2:
        for tag in TAGS:
            change = m.get("entropy_changes", {}).get(tag, {})
            erows.append(f'<tr><td>{tag}</td><td>{delta_cell(change.get("meaning"))}</td>'
                         f'<td>{delta_cell(change.get("surface"))}</td>'
                         f'<td>{delta_cell(change.get("meaning_minus_surface"))}</td></tr>')
    entropy_tbl = (f'<table class="matrix" style="margin-top:10px"><thead><tr><th>entropy change '
                   f'T={Ts[0]}&rarr;{Ts[-1]}</th><th>meaning</th><th>surface</th>'
                   f'<th>meaning&minus;surface</th></tr></thead>'
                   f'<tbody>{"".join(erows)}</tbody></table>') if erows else ""

    h5rows = []
    for tag in TAGS:
        h5 = m.get("greedy_in_core", {}).get(tag)
        if h5:
            claim_rate = h5.get("claim_weighted_match_rate")
            h5rows.append(f'<tr><td>{tag}</td>'
                          f'<td>{ci_cell(h5.get("doc_mean"), h5.get("p_test_vs_one"), "vs inclusion=1")}</td>'
                          f'<td>{"&mdash;" if claim_rate is None else f"{claim_rate:.2f}"} '
                          f'<span class="statline">descriptive weighting</span></td>'
                          f'<td>{h5.get("n_greedy_claims", 0)} <span class="statline">design count</span></td></tr>')
    h5tbl = (f'<table class="matrix" style="margin-top:10px"><thead><tr><th>H5: greedy claim in T=0.4 core</th>'
             f'<th>mean per doc</th><th>claim-weighted</th><th>greedy claims</th></tr></thead>'
             f'<tbody>{"".join(h5rows)}</tbody></table>') if h5rows else ""

    qrows = []
    quality = m.get("quality", {})
    for arm in ("main", "floor", "shuffle"):
        qr = quality.get(arm, {}).get("0.4")
        if qr:
            vci = qr.get("valid_rate_ci95")
            dci = qr.get("docs_effective_rate_ci95")
            qrows.append(f'<tr><td>{arm}</td><td>{qr["effective"]}/{qr["n"]} '
                         f'<span class="statline">{100*qr["valid_rate"]:.1f}%; Wilson 95% CI '
                         f'[{100*vci[0]:.1f}%,{100*vci[1]:.1f}%]</span></td>'
                         f'<td>{qr["docs_with_effective"]}/{qr["n_docs"]} '
                         f'<span class="statline">Wilson 95% CI [{100*dci[0]:.1f}%,{100*dci[1]:.1f}%]</span></td></tr>')
    quality_tbl = (f'<table class="matrix" style="margin-top:10px"><thead><tr><th>T=0.4 output validity</th>'
                   f'<th>valid samples</th><th>docs with &ge;1 valid sample</th></tr></thead>'
                   f'<tbody>{"".join(qrows)}</tbody></table>') if qrows else ""
    floor_q = quality.get("floor", {}).get("0.4", {})
    floor_warning = (f'<p class="note" style="color:var(--terra);margin-top:7px">'
                     f'No-injection floor validity is only {100*floor_q.get("valid_rate", 0):.1f}%. '
                     f'The marker-without-activation path therefore fails to produce an in-distribution '
                     f'explanation; any floor core above is a sparse survivor estimate, and H4 is not a '
                     f'clean quantitative core-vs-floor comparison for this model.</p>'
                     if floor_q and floor_q.get("valid_rate", 1) < 0.1 else "")
    method = ('<p class="note">CIs use 20,000 document-bootstrap replicates. Two-sided p-values test the '
              'null named in each cell; BH q-values adjust across the Exp 7 inferential family. Counts and '
              'claim-weighted diagnostics are descriptive.</p>')
    return (f'<div class="panel"><h4>{esc(LABELS[key])} &mdash; stability</h4>{method}{core_tbl}{contr}'
            f'{entropy_tbl}{h5tbl}{quality_tbl}{floor_warning}</div>')


INTRO = """<div class="panel"><h2>Exp 7 &mdash; resample the verbalizer: signal vs sampler noise</h2>
<p>The AV is a <b>stochastic</b> decoder: the same frozen activation <code>h</code> yields a different
explanation on every sample. Exp 7 treats the explanation as a <b>random variable</b> and asks how much
of it is <span class="accent">signal</span> (determined by <code>h</code> &rarr; recurs across resamples
&amp; temperatures) vs <span style="color:var(--terra)">noise</span> (produced by the sampler &rarr; grows
with temperature). We resample the AV on the same frozen <code>h</code> across a temperature grid, split
each explanation into typed claims (genre / LCR / PC), cluster claims <b>by meaning</b> (an LLM judge,
validated at Step&nbsp;7.0: 0 false-merges), and read off a <b>stable core</b> (claims asserted in most
resamples) vs a <b>temperature-sensitive periphery</b>.</p>
<p class="note">Crossed with Exp&nbsp;6 input-derivability this is a 2&times;2. The interesting cell is
<b>stable &times; not-in-input</b> &mdash; candidate privileged content OR a systematic AV prior. Two
controls disambiguate it: the <b>no-injection floor</b> (marker present, activation absent &mdash; how
stable is the AV with zero info from <code>h</code>? the real core must exceed this) and the
<b>shuffled-activation</b> arm (inject a different doc's <code>h</code> &mdash; if the core persists it is
a prior, not activation-specific). T=0 is a single greedy anchor (not resampled).</p></div>"""

OUTCOME = """<div class="panel"><h2>Outcome</h2>
<p><b>The activation-specificity hypothesis fails.</b> At T=0.4, main and shuffled-activation core
fractions are statistically indistinguishable for every tag and model. LCR and PC recurrence still
exceeds the permutation null, so the recurrence is real, but it is attributable to the verbalizer /
prompt distribution rather than the identity of the injected activation.</p>
<p class="note"><b>H1:</b> mixed, not clean bimodality. <b>H2:</b> modest support; meaning entropy grows
less than surface entropy, sometimes only slightly. <b>H3:</b> partial for LCR and rejected for genre.
<b>H4:</b> the no-injection floor is invalid (see Wilson intervals below), while the valid shuffled control rejects
activation specificity. <b>H5:</b> rejected; only a minority of greedy LCR/PC claims enter the T=0.4
core. Model-specific estimates, intervals, and tests are below.</p></div>"""


def build(n_cards: int) -> None:
    style, nav, headlinks = shared_style_nav()
    metrics = _rd(RES / "exp7_metrics.json")
    have_any = False
    body = [INTRO, OUTCOME] if metrics else [INTRO]

    for key in KEYS:
        od = _outdir(key)
        main = _rd(od / "exp7_clusters_main.json")
        if not main:
            continue
        have_any = True
        # raw resamples for the expand panels
        rs = _rd(od / "resample_main.json")
        raw = {}
        if rs:
            for s in rs["samples"].values():
                raw.setdefault(s["idx"], []).append(s)
            for k in raw:
                raw[k].sort(key=lambda r: (r["T"], r["k"]))
        # subsample meta (token/stratum/divergence)
        sub = _rd(RES / f"exp7_subsample_{key}.json")
        meta = {d["idx"]: d for d in (sub["docs"] if sub else [])}
        toks = {r["idx"]: r.get("token", "") for r in (_rd(RES / f"records_exp6b_{key}.json") or [])}

        body.append(f'<h2 style="margin-top:18px">{esc(LABELS[key])}</h2>')
        if metrics:
            body.append(render_summary(key, metrics))

        # group blocks by idx
        by_idx = {}
        for kk, b in main["blocks"].items():
            by_idx.setdefault(b["idx"], {})[b["T"]] = b
        order = sorted(by_idx, key=lambda i: (meta.get(i, {}).get("stratum", "z"), i))[:n_cards]
        for idx in order:
            mrow = dict(meta.get(idx, {}), token=toks.get(idx, ""))
            body.append(render_card(by_idx[idx], main["grid"], idx, mrow, raw))
        if len(by_idx) > n_cards:
            body.append(f'<p class="note">Showing {n_cards} of {len(by_idx)} datapoints (capped for page weight).</p>')

    if not have_any:
        body.append('<div class="panel"><h4>Pilot pending</h4><p class="pending">The resample harness, '
                     'stratified 200-doc subsample, meaning/surface clustering and stability metrics are '
                     'built and unit-tested; this tab populates with real resample spreads once the 7.1 GPU '
                     'pilot runs (blocked only on cluster connectivity). Pipeline: '
                     '<code>fve_resample.py</code> &rarr; <code>exp7_cluster.py</code> &rarr; '
                     '<code>exp7_metrics.py</code> &rarr; this page.</p></div>')

    page = (f'<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1.0">'
            f'<title>NLA FVE Robustness &mdash; Exp 7 (resample stability)</title>{headlinks}{style}</head>'
            f'<body><div class="container"><h1>Exp 7 &mdash; verbalization stability</h1>'
            f'<div class="subtitle">how much of the AV read is determined by the activation vs the sampler</div>'
            f'{nav}{"".join(body)}'
            f'<footer>NLA FVE-robustness &middot; Exp 7 resample-stability. '
            f'Meaning-judge validated (Step 7.0). Controls: no-injection floor + shuffled-activation.</footer>'
            f'</div></body></html>')
    (HERE / "exp7_resample.html").write_text(page, encoding="utf-8")
    print(f"wrote {HERE / 'exp7_resample.html'} ({len(page)} bytes, data={'yes' if have_any else 'placeholder'})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cards", type=int, default=40)
    args = ap.parse_args()
    build(args.cards)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
