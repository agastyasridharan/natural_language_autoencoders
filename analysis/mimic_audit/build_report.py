"""Split the collected corpus (document-level, stratified by theme) and emit a
browsable, fully-categorized report so every explanation is readable by category.

Outputs under analysis/mimic_audit/corpus/:
  train.jsonl, eval.jsonl        document-level split (every theme in both)
  CORPUS.md                      overview: stats + per-theme/format/length tables + index
  by_theme/<theme>.md            one file per theme listing EVERY doc and EVERY explanation
"""
from __future__ import annotations

import json
from collections import OrderedDict, defaultdict
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
CORPUS = HERE / "corpus"
OUT = CORPUS / "corpus_all.jsonl"
BYTHEME = CORPUS / "by_theme"


def load_rows():
    return [json.loads(l) for l in OUT.open(encoding="utf-8") if l.strip()]


def group_docs(rows):
    docs = OrderedDict()
    for r in rows:
        docs.setdefault(r["doc_id"], []).append(r)
    for k in docs:
        docs[k].sort(key=lambda r: r["position"])
    return docs


def band(cos):
    return "high" if cos >= 0.9 else ("med" if cos >= 0.75 else "low")


def split_docs(docs):
    """~25% of each theme's docs to eval (>=1 if the theme has >=2 docs), deterministic."""
    theme_docs = defaultdict(list)
    for did, rs in docs.items():
        theme_docs[rs[0]["theme"]].append(did)
    train, evl = set(), set()
    for theme, dids in theme_docs.items():
        dids = sorted(dids)
        n = len(dids)
        n_eval = max(1, round(n * 0.25)) if n >= 2 else 0
        evl |= set(dids[:n_eval])
        train |= set(dids[n_eval:])
    return train, evl


def fulltext(rs):
    return max(rs, key=lambda r: r["position"])["prefix"]


def write_split(rows, ids, path):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            if r["doc_id"] in ids:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


def md_block(t):
    return "```\n" + t + "\n```"


def doc_md(did, rs, split):
    L = [f"#### {did}  `[{split}]`\n"]
    r0 = rs[0]
    lo = min(r["cosine"] for r in rs)
    hi = max(r["cosine"] for r in rs)
    L.append(f"**format** `{r0['format']}` · **length** `{r0['length_class']}` · **kind** `{r0['kind']}` "
             f"· **{r0['n_tokens']} tokens** · **{len(rs)} positions** · **cosine {lo:.2f}-{hi:.2f}**\n")
    L.append("Source text the model read:\n")
    L.append(md_block(fulltext(rs)) + "\n")
    L.append("Explanations (one per swept token):\n")
    for r in rs:
        tok = r["token"].replace("\n", "\\n")
        L.append(f"- **pos {r['position']}** · token `{tok}` · cos **{r['cosine']}** "
                 f"({band(r['cosine'])}) · mse {r['mse']}")
        L.append("  > " + r["av_description"].replace("\n", "\n  > "))
    L.append("")
    return "\n".join(L)


def main():
    rows = load_rows()
    docs = group_docs(rows)
    train_ids, eval_ids = split_docs(docs)
    split_of = {d: ("eval" if d in eval_ids else "train") for d in docs}

    write_split(rows, train_ids, CORPUS / "train.jsonl")
    write_split(rows, eval_ids, CORPUS / "eval.jsonl")

    # ---- per-theme grouping ----
    theme_docs = defaultdict(list)
    for did, rs in docs.items():
        theme_docs[rs[0]["theme"]].append(did)
    themes = sorted(theme_docs)

    BYTHEME.mkdir(parents=True, exist_ok=True)
    for theme in themes:
        dids = sorted(theme_docs[theme])
        F = [f"# Corpus theme: {theme}\n"]
        n_exp = sum(len(docs[d]) for d in dids)
        F.append(f"{len(dids)} documents · {n_exp} explanations · "
                 f"source: hosted Llama-3.3-70B NLA (layer 53)\n")
        for did in dids:
            F.append(doc_md(did, docs[did], split_of[did]))
        (BYTHEME / f"{theme}.md").write_text("\n".join(F), encoding="utf-8")

    # ---- master report ----
    n_docs = len(docs)
    n_exp = len(rows)
    all_cos = [r["cosine"] for r in rows]
    M = ["# Mimic-audit corpus (large, categorized)\n"]
    M.append(f"Real per-token NLA reads (AV text + AR cosine/mse) from the hosted "
             f"**Llama-3.3-70B / layer 53** NLA via Neuronpedia.\n")
    M.append(f"- **{n_docs} documents**, **{n_exp} explanations**, **{len(themes)} themes**\n"
             f"- cosine: mean {mean(all_cos):.3f}, min {min(all_cos):.3f}, max {max(all_cos):.3f}\n"
             f"- split: **{len(train_ids)} train docs**, **{len(eval_ids)} eval docs** "
             f"(document-level, stratified so every theme appears in both)\n")
    M.append("Raw rows: `corpus/corpus_all.jsonl` · split: `corpus/train.jsonl`, "
             "`corpus/eval.jsonl` · per-theme browsable: `corpus/by_theme/<theme>.md`\n")

    # how built
    M.append("## How it was built\n")
    M.append("Documents were generated across genres/formats, then each was tokenized once "
             "and several token positions were swept (short docs: last several tokens; longer "
             "docs: ~8 spread across the back portion, past position 50 when long enough, always "
             "including the final token). The causal-prefix property means each row carries the "
             "real AV text, AR fidelity, and the exact prefix the mimic must reproduce from. "
             "Row schema: `{doc_id, theme, format, length_class, kind, n_tokens, position, token, "
             "cosine, mse, prefix, av_description}`.\n")

    # per-theme table
    M.append("## By theme\n")
    M.append("| theme | docs | explanations | train/eval docs | formats | cosine mean (min-max) |")
    M.append("|---|---:|---:|---|---|---|")
    for theme in themes:
        dids = theme_docs[theme]
        rs = [r for d in dids for r in docs[d]]
        cos = [r["cosine"] for r in rs]
        fmts = sorted({docs[d][0]["format"] for d in dids})
        tr = sum(1 for d in dids if d in train_ids)
        ev = sum(1 for d in dids if d in eval_ids)
        M.append(f"| [{theme}](by_theme/{theme}.md) | {len(dids)} | {len(rs)} | {tr}/{ev} | "
                 f"{', '.join(fmts)} | {mean(cos):.2f} ({min(cos):.2f}-{max(cos):.2f}) |")

    # per-format and per-length tables
    def tally(key):
        t = defaultdict(lambda: [0, 0])  # docs, explanations
        seen = set()
        for did, rs in docs.items():
            k = rs[0][key]
            if did not in seen:
                t[k][0] += 1
                seen.add(did)
            t[k][1] += len(rs)
        return t

    M.append("\n## By format\n")
    M.append("| format | docs | explanations |")
    M.append("|---|---:|---:|")
    for k, (nd, ne) in sorted(tally("format").items()):
        M.append(f"| {k} | {nd} | {ne} |")

    M.append("\n## By length class\n")
    M.append("| length_class | docs | explanations |")
    M.append("|---|---:|---:|")
    for k in ["short", "medium", "long"]:
        t = tally("length_class").get(k)
        if t:
            M.append(f"| {k} | {t[0]} | {t[1]} |")

    M.append("\n## By kind\n")
    M.append("| kind | docs | explanations |")
    M.append("|---|---:|---:|")
    for k, (nd, ne) in sorted(tally("kind").items()):
        M.append(f"| {k} | {nd} | {ne} |")

    # fidelity bands
    bands = defaultdict(int)
    for r in rows:
        bands[band(r["cosine"])] += 1
    M.append("\n## Reconstruction-fidelity bands (AR cosine)\n")
    M.append("| band | count | meaning |")
    M.append("|---|---:|---|")
    M.append(f"| high (>=0.90) | {bands['high']} | text reconstructs the activation cleanly |")
    M.append(f"| med (0.75-0.90) | {bands['med']} | usable, some loss |")
    M.append(f"| low (<0.75) | {bands['low']} | noisy read (often short prompts / early tokens) |")

    M.append("\n## One sample explanation per theme\n")
    for theme in themes:
        did = sorted(theme_docs[theme])[0]
        r = docs[did][-1]
        M.append(f"\n**{theme}** ({did}, pos {r['position']}, token `{r['token'].strip() or r['token']}`, "
                 f"cos {r['cosine']}):")
        M.append("> " + r["av_description"].replace("\n", "\n> "))

    (CORPUS / "CORPUS.md").write_text("\n".join(M), encoding="utf-8")

    print(f"docs={n_docs} explanations={n_exp} themes={len(themes)} "
          f"train={len(train_ids)} eval={len(eval_ids)}")
    print(f"wrote CORPUS.md + {len(themes)} per-theme files + train/eval.jsonl")


if __name__ == "__main__":
    main()
