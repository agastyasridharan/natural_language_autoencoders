"""Exp 5b (shuffle sentences at random) — edit + per-datapoint records. Deterministic, no API/GPU.

Flattens the explanation into quote-aware sentences (never splitting inside a quote; abbreviation- and
list-marker-aware), permutes them with an arm-salted md5(explanation) seed, rejoins with spaces. By the
corpus structure this is ~1 sentence/block, so it mostly matches 5a except where a block holds >1
sentence. cos/FVE stay null until the AR re-score.

Usage:  python exp5b_edit.py [qwen7b|gemma12b|gemma27b ...]
"""
from __future__ import annotations

import sys

sys.path.insert(0, "/data/agastyas/nla_fve_ufw/code")
from edits import shuffle_sentences                       # noqa: E402
from records_common import build_records, load_docs        # noqa: E402

KEYS = ["qwen7b", "gemma12b", "gemma27b"]


def edit_fn(expl: str) -> dict:
    edited, meta = shuffle_sentences(expl)
    return dict(expl_edit=edited, edit_meta=meta)


def main() -> None:
    docs = load_docs()
    for key in (sys.argv[1:] or KEYS):
        n = build_records(key, "exp5b", edit_fn, docs)
        print(f"[exp5b:{key}] {n} records; wrote edited_reads_exp5b.json + records_exp5b_{key}.json",
              flush=True)


if __name__ == "__main__":
    main()
