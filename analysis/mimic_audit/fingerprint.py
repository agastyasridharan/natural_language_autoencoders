"""Step 1 of the single-prompt tuning plan: compute the empirical STYLE FINGERPRINT
of the real AV reads.

This is the objective target the one mimic prompt has to land inside. It is pure
analysis of data we already collected (corpus/corpus_all.jsonl); it does not call
any API and does not generate any mimic.

For every real AV description we compute a small set of measurable, non-fuzzy
signatures (length, block structure, quoted-fragment count, recycled-string
repetition, hedge/candidate density, quote imbalance, ellipsis / mid-thought
ending, slash-classification). We then report each signature's distribution
overall (p10 / p50 / p90) and its median per genre. The p10-p90 band is the
"realistic" window: a mimic whose outputs fall below it is too clean, above it is
caricature.

Outputs:
  corpus/style_fingerprint.json   machine-readable target profile
  STYLE_FINGERPRINT.md            human-readable report
"""
from __future__ import annotations

import json
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ALL = HERE / "corpus" / "corpus_all.jsonl"
OUT_JSON = HERE / "corpus" / "style_fingerprint.json"
OUT_MD = HERE / "STYLE_FINGERPRINT.md"

_QUOTED = re.compile(r'"([^"]{1,200})"')
_SLASH_CLASS = re.compile(r"\w\s*/\s*\w")
_HEDGES = ("likely", "probably", "maybe", "perhaps", "possibly", "might",
           "could", "seems", "appears", "presumably", "or ")
_END_PUNCT = (".", "!", "?", '"', ":", ";")


def _norm_span(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip(" .,:;\"'")


def features(av: str) -> dict:
    av = av or ""
    stripped = av.strip()
    blocks = [b for b in re.split(r"\n\s*\n", av) if b.strip()]
    lines = [l for l in av.splitlines() if l.strip()]
    spans = _QUOTED.findall(av)
    norm = [_norm_span(s) for s in spans if _norm_span(s)]
    counts = defaultdict(int)
    for n in norm:
        counts[n] += 1
    max_repeat = max(counts.values()) if counts else 0
    dq = av.count('"')
    low = av.lower()
    hedge = sum(low.count(h) for h in _HEDGES)
    ends = stripped[-3:] if stripped else ""
    ends_ellipsis = stripped.endswith("...") or stripped.endswith("…")
    last = stripped[-1] if stripped else ""
    ends_midthought = bool(stripped) and not ends_ellipsis and last not in _END_PUNCT
    return {
        "chars": len(av),
        "words": len(av.split()),
        "blocks": len(blocks),
        "lines": len(lines),
        "quoted_spans": len(spans),
        "max_span_repeat": max_repeat,
        "has_repeat": 1 if max_repeat >= 2 else 0,
        "dquotes": dq,
        "quote_unbalanced": 1 if dq % 2 == 1 else 0,
        "ellipses": av.count("...") + av.count("…"),
        "ends_ellipsis": 1 if ends_ellipsis else 0,
        "ends_midthought": 1 if ends_midthought else 0,
        "likely_count": low.count("likely"),
        "hedge_count": hedge,
        "slash_count": av.count("/"),
        "has_slash_class": 1 if _SLASH_CLASS.search(av) else 0,
    }


NUMERIC = ["chars", "words", "blocks", "lines", "quoted_spans", "max_span_repeat",
           "dquotes", "ellipses", "likely_count", "hedge_count", "slash_count"]
FRAC = ["has_repeat", "quote_unbalanced", "ends_ellipsis", "ends_midthought",
        "has_slash_class"]


def pct(vals, q):
    if not vals:
        return 0
    vals = sorted(vals)
    i = max(0, min(len(vals) - 1, int(round(q * (len(vals) - 1)))))
    return vals[i]


def summarize(feats: list[dict]) -> dict:
    out = {"n": len(feats)}
    for k in NUMERIC:
        vals = [f[k] for f in feats]
        out[k] = {
            "p10": pct(vals, 0.10), "p50": st.median(vals), "p90": pct(vals, 0.90),
            "mean": round(st.mean(vals), 2),
        }
    for k in FRAC:
        vals = [f[k] for f in feats]
        out[k] = {"frac": round(st.mean(vals), 3)}
    return out


def main():
    rows = [json.loads(l) for l in ALL.open(encoding="utf-8")]
    feats = [features(r["av_description"]) for r in rows]
    by_genre = defaultdict(list)
    for r, f in zip(rows, feats):
        by_genre[r["theme"]].append(f)

    overall = summarize(feats)
    per_genre = {g: summarize(fs) for g, fs in sorted(by_genre.items())}

    profile = {
        "n_reads": len(rows),
        "n_docs": len(set(r["doc_id"] for r in rows)),
        "n_genres": len(by_genre),
        "overall": overall,
        "per_genre": per_genre,
    }
    OUT_JSON.write_text(json.dumps(profile, indent=2), encoding="utf-8")

    # ---- human-readable report ----
    L = []
    L.append("# Real AV style fingerprint")
    L.append("")
    L.append(f"Computed from **{len(rows):,} real AV reads** across "
             f"**{len(by_genre)} genres** ({profile['n_docs']} documents) in "
             "`corpus/corpus_all.jsonl`. This is the objective target the single "
             "mimic prompt must land inside. The realistic window for each "
             "signature is its **p10 to p90** band: a mimic below the band is too "
             "clean, above it is caricature (the failure mode of the [OLD] run).")
    L.append("")
    L.append("## What each signature means")
    L.append("")
    meanings = [
        ("chars / words", "raw length of the read."),
        ("blocks", "blank-line-separated paragraphs. Real reads are typically 3: a genre/source classification, a local-structure description, then a recycled-quote candidate block."),
        ("quoted_spans", 'number of \"...\" fragments (the candidate continuations).'),
        ("max_span_repeat", "how many times the single most-repeated quoted fragment recurs. This is the AV's signature near-verbatim recycling (\"stare at Slack\" x4)."),
        ("has_repeat", "fraction of reads that recycle some quoted fragment at least twice."),
        ("dquotes / quote_unbalanced", "double-quote count; fraction with an odd count (an opened-but-unclosed quote, very common in real reads)."),
        ("ellipses / ends_ellipsis", '"..." count; fraction ending in "...".'),
        ("ends_midthought", "fraction ending without sentence punctuation (the read stops mid-thought)."),
        ("likely_count / hedge_count", 'hedge density ("likely", "probably", "or ", etc.).'),
        ("slash_count / has_slash_class", 'slash usage and the "Personal/ creative writing" classification idiom.'),
    ]
    for name, desc in meanings:
        L.append(f"- **{name}** — {desc}")
    L.append("")
    L.append("## Overall target profile")
    L.append("")
    L.append("| signature | p10 | median | p90 | mean |")
    L.append("|---|---|---|---|---|")
    for k in NUMERIC:
        s = overall[k]
        L.append(f"| {k} | {s['p10']} | {s['p50']} | {s['p90']} | {s['mean']} |")
    L.append("")
    L.append("| signature (fraction of reads) | value |")
    L.append("|---|---|")
    for k in FRAC:
        L.append(f"| {k} | {overall[k]['frac']:.0%} |")
    L.append("")
    L.append("## Per-genre medians (key signatures)")
    L.append("")
    keys = ["chars", "words", "blocks", "quoted_spans", "max_span_repeat", "hedge_count", "slash_count"]
    L.append("| genre | n | " + " | ".join(keys) + " | %repeat | %unbal | %midthought |")
    L.append("|---|---|" + "|".join(["---"] * len(keys)) + "|---|---|---|")
    for g, s in per_genre.items():
        cells = [str(s[k]["p50"]) for k in keys]
        L.append(f"| {g} | {s['n']} | " + " | ".join(cells) +
                 f" | {s['has_repeat']['frac']:.0%} | {s['quote_unbalanced']['frac']:.0%} | {s['ends_midthought']['frac']:.0%} |")
    L.append("")
    OUT_MD.write_text("\n".join(L), encoding="utf-8")

    print(f"wrote {OUT_JSON.name} and {OUT_MD.name}")
    print(f"reads={len(rows)} genres={len(by_genre)}")
    o = overall
    print("OVERALL medians: chars={} words={} blocks={} quoted_spans={} max_repeat={} hedge={} slash={}".format(
        o["chars"]["p50"], o["words"]["p50"], o["blocks"]["p50"], o["quoted_spans"]["p50"],
        o["max_span_repeat"]["p50"], o["hedge_count"]["p50"], o["slash_count"]["p50"]))
    print("OVERALL fracs: repeat={:.0%} unbalanced={:.0%} ends_ellipsis={:.0%} ends_midthought={:.0%} slash_class={:.0%}".format(
        o["has_repeat"]["frac"], o["quote_unbalanced"]["frac"], o["ends_ellipsis"]["frac"],
        o["ends_midthought"]["frac"], o["has_slash_class"]["frac"]))


if __name__ == "__main__":
    main()
