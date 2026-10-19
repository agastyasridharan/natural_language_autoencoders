"""Emit corpus/viewer_data.js (window.CORPUS = {...}) for the static HTML viewer.

Pulls together every collected explanation (corpus_all.jsonl), the original
document payloads (corpus_docs.json, for clean chat-turn rendering), and the
document-level split (train.jsonl / eval.jsonl). The viewer is fully offline:
it reads only this generated JS, so it always reflects ALL stored data.
"""
from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
CORPUS = HERE / "corpus"


def load_jsonl(p):
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]


def band(c):
    return "high" if c >= 0.9 else ("med" if c >= 0.75 else "low")


def main():
    rows = load_jsonl(CORPUS / "corpus_all.jsonl")
    docs_meta = {d["id"]: d for d in json.loads((CORPUS / "corpus_docs.json").read_text(encoding="utf-8"))}
    eval_ids = {json.loads(l)["doc_id"] for l in (CORPUS / "eval.jsonl").open(encoding="utf-8") if l.strip()}

    by = OrderedDict()
    for r in rows:
        by.setdefault(r["doc_id"], []).append(r)
    for k in by:
        by[k].sort(key=lambda r: r["position"])

    docs = []
    for did, rs in by.items():
        r0 = rs[0]
        source = max(rs, key=lambda r: r["position"])["prefix"]
        meta = docs_meta.get(did, {})
        kind = r0["kind"]
        turns = meta["payload"] if (kind == "chat" and isinstance(meta.get("payload"), list)) else None
        positions = [{
            "position": r["position"], "token": r["token"],
            "cosine": r["cosine"], "mse": r["mse"], "band": band(r["cosine"]),
            "prefix_tail": r["prefix"][-220:], "av": r["av_description"],
        } for r in rs]
        docs.append({
            "id": did, "theme": r0["theme"], "format": r0["format"],
            "length_class": r0["length_class"], "kind": kind,
            "split": "eval" if did in eval_ids else "train",
            "n_tokens": r0["n_tokens"],
            "cos_min": min(r["cosine"] for r in rs),
            "cos_max": max(r["cosine"] for r in rs),
            "source": source, "turns": turns, "positions": positions,
        })

    docs.sort(key=lambda d: (d["theme"], d["id"]))
    allcos = [r["cosine"] for r in rows]
    meta = {
        "docs": len(docs), "explanations": len(rows),
        "themes": len({d["theme"] for d in docs}),
        "train_docs": sum(1 for d in docs if d["split"] == "train"),
        "eval_docs": sum(1 for d in docs if d["split"] == "eval"),
        "cos_mean": round(mean(allcos), 3), "cos_min": min(allcos), "cos_max": max(allcos),
        "bands": {b: sum(1 for r in rows if band(r["cosine"]) == b) for b in ("high", "med", "low")},
    }
    out = "window.CORPUS = " + json.dumps({"meta": meta, "docs": docs}, ensure_ascii=False) + ";\n"
    (CORPUS / "viewer_data.js").write_text(out, encoding="utf-8")
    print(f"wrote viewer_data.js ({len(out):,} bytes) — {len(docs)} docs, {len(rows)} explanations, "
          f"{meta['themes']} themes, train {meta['train_docs']} / eval {meta['eval_docs']}")


if __name__ == "__main__":
    main()
