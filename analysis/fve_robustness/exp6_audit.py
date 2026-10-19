"""Exp 6 — Step 6.0 PARSER AUDIT (CPU, $0, no GPU/API; hard check-in gate).

Runs the Exp-6 parser (exp6_edits.analyze_datapoint) over every in-range datapoint of all three
models on the frozen artifacts and emits, per model + pooled:
  - n_blocks distribution, unbalanced-quote rate
  - quote-tag coverage (GENRE / LCR / ECHO / PC / UNKNOWN)
  - LCR-per-explanation distribution
  - echo==token rate  (does the ECHO quote text equal the read token?)
  - directional-gate disagreement rate (tagged LCR/PC contradicted by content-word direction)
  - no_faithful_anchor rate + anchor-fidelity / divergence distribution (drives the stratifier)
  - injection word-count (Δwords) distribution for LCR quotes
It writes results/exp6_audit_report.json (histograms) and results/exp6_audit_rows_<key>.json
(every datapoint's parse, for the dashboard + hand inspection) and prints a compact report + a
sample of flagged rows.

Usage:  python3 exp6_audit.py
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from exp6_edits import analyze_datapoint

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
DOCS = {d["id"]: d["text"] for d in json.loads((HERE / "data" / "ufw_docs.json").read_text())}
KEYS = ["qwen7b", "gemma12b", "gemma27b"]


def _pct(n, d):
    return round(100 * n / d, 1) if d else 0.0


def _hist(vals, edges):
    c = Counter()
    for v in vals:
        for lo, hi in zip(edges, edges[1:]):
            if lo <= v < hi:
                c[f"{lo}-{hi}"] += 1
                break
        else:
            c[f">={edges[-1]}"] += 1
    return dict(c)


def run_key(key: str) -> dict:
    recs = json.loads((RES / f"records_exp3a_{key}.json").read_text())
    rows, flagged = [], []
    n_dp = n_align_bad = 0
    tagc = Counter()
    overrides = Counter()
    nblk = Counter()
    lcr_per = Counter()
    n_unbal = echo_eq = echo_tot = 0
    dir_dis = dir_tot = 0
    lcr_tot = lcr_highdiv = 0
    fids, divs, dwords = [], [], []

    for r in recs:
        if not r.get("in_range"):
            continue
        expl = r.get("expl_orig") or ""
        token = r.get("token", "")
        doc = DOCS.get(r["doc_id"], "")
        ce = r.get("char_end")
        if not expl or ce is None or not doc:
            continue
        # alignment guard (the one UTF-8 byte-boundary row is dropped + logged)
        aligned = doc[r.get("char_start", ce):ce] == token
        if not aligned:
            n_align_bad += 1
        n_dp += 1

        p = analyze_datapoint(expl, token, doc, ce)
        nblk[p["n_blocks"]] += 1
        if p["unbalanced"]:
            n_unbal += 1
        qs = p["quotes"]
        for q in qs:
            tagc[q["tag"]] += 1
            if q.get("override"):
                overrides[q["override"]] += 1
        n_lcr = sum(1 for q in qs if q["tag"] == "LCR")
        lcr_per[n_lcr] += 1

        echo = next((q for q in qs if q["tag"] == "ECHO"), None)
        if echo is not None:
            echo_tot += 1
            if echo["text"].strip() == token.strip():
                echo_eq += 1
        for q in qs:
            if q["tag"] in ("LCR", "PC") and "dir_disagree" in q:
                dir_tot += 1
                dir_dis += int(q["dir_disagree"])
            if q["tag"] == "LCR":
                lcr_tot += 1
                lcr_highdiv += int(q.get("restate_fidelity", 0.0) < 0.5)   # confabulated => has power
                fids.append(q.get("restate_fidelity", 0.0))
                divs.append(q.get("divergence", 1.0))
                dwords.append(q.get("n_words", 0))

        row = dict(idx=r["idx"], doc_id=r["doc_id"], token=token, position=r.get("position"),
                   char_start=r.get("char_start"), char_end=ce, aligned=aligned,
                   n_blocks=p["n_blocks"], unbalanced=p["unbalanced"],
                   final_block_idx=p["final_block_idx"], quotes=qs,
                   expl_orig=expl, input_display=r.get("input_display", ""))
        rows.append(row)

        # flag conditions for hand inspection
        reasons = []
        if p["n_blocks"] != 3:
            reasons.append(f"n_blocks={p['n_blocks']}")
        if p["unbalanced"]:
            reasons.append("unbalanced")
        if n_lcr == 0:
            reasons.append("no_LCR")
        if n_lcr >= 3:
            reasons.append(f"LCR={n_lcr}")
        if echo is None:
            reasons.append("no_ECHO")
        if any(q.get("dir_disagree") for q in qs):
            reasons.append("dir_disagree")
        if all(q.get("restate_fidelity", 0.0) < 0.15 for q in qs if q["tag"] == "LCR") and n_lcr:
            reasons.append("all_LCR_confab")
        if not aligned:
            reasons.append("align_bad")
        if reasons:
            flagged.append(dict(idx=r["idx"], reasons=reasons))

    (RES / f"exp6_audit_rows_{key}.json").write_text(json.dumps(rows, ensure_ascii=False))

    return dict(
        key=key, n_datapoints=n_dp, align_bad=n_align_bad,
        n_blocks=dict(nblk), unbalanced_pct=_pct(n_unbal, n_dp),
        tag_coverage=dict(tagc), overrides=dict(overrides),
        quotes_per_dp=round(sum(tagc.values()) / n_dp, 2) if n_dp else 0,
        lcr_per_expl=dict(sorted(lcr_per.items())),
        echo_present_pct=_pct(echo_tot, n_dp), echo_eq_token_pct=_pct(echo_eq, echo_tot),
        dir_disagree_pct=_pct(dir_dis, dir_tot), dir_checked=dir_tot,
        lcr_total=lcr_tot, high_div_pct=_pct(lcr_highdiv, lcr_tot),
        restate_fidelity_hist=_hist(fids, [0.0, 0.15, 0.3, 0.5, 0.7, 0.9, 1.001]),
        divergence_hist=_hist(divs, [0.0, 0.1, 0.3, 0.5, 0.7, 0.85, 1.001]),
        lcr_words_hist=_hist(dwords, [1, 4, 8, 12, 16, 24, 40, 999]),
        median_lcr_words=sorted(dwords)[len(dwords) // 2] if dwords else 0,
        n_flagged=len(flagged), flagged_sample=flagged[:60],
    )


def main() -> None:
    report = {k: run_key(k) for k in KEYS}
    (RES / "exp6_audit_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    for k in KEYS:
        s = report[k]
        print(f"\n===== {k}  (n_dp={s['n_datapoints']}, align_bad={s['align_bad']}) =====")
        print(f"  n_blocks           {s['n_blocks']}   unbalanced={s['unbalanced_pct']}%")
        print(f"  tag_coverage       {s['tag_coverage']}   (~{s['quotes_per_dp']} quotes/dp)")
        print(f"  dir overrides      {s['overrides']}")
        print(f"  LCR per expl       {s['lcr_per_expl']}")
        print(f"  ECHO present       {s['echo_present_pct']}%   echo==token {s['echo_eq_token_pct']}%")
        print(f"  dir disagree       {s['dir_disagree_pct']}%  of {s['dir_checked']} LCR/PC checked")
        print(f"  high-div stratum   {s['high_div_pct']}%  of {s['lcr_total']} LCR quotes (fid<0.5 => power)")
        print(f"  restate fidelity   {s['restate_fidelity_hist']}")
        print(f"  divergence (1-fid) {s['divergence_hist']}")
        print(f"  LCR words          {s['lcr_words_hist']}  median={s['median_lcr_words']}")
        print(f"  flagged rows       {s['n_flagged']}/{s['n_datapoints']}")
    print("\nwrote results/exp6_audit_report.json + results/exp6_audit_rows_<key>.json")


if __name__ == "__main__":
    main()
