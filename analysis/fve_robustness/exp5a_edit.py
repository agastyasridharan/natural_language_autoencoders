"""Exp 5a (shuffle whole blocks at random) — edit + per-datapoint records. Deterministic, no API/GPU.

Splits the explanation on blank-line boundaries (~3 blocks), permutes their order with an arm-salted
md5(explanation) seed (identity rejected for n>=2), and rejoins with blank lines. Characters preserved;
only order changes. cos/FVE stay null until the AR re-score.

Usage:  python exp5a_edit.py [qwen7b|gemma12b|gemma27b ...]
"""
from __future__ import annotations

import sys

sys.path.insert(0, "/data/agastyas/nla_fve_ufw/code")
from edits import shuffle_blocks                          # noqa: E402
from records_common import build_records, load_docs        # noqa: E402

KEYS = ["qwen7b", "gemma12b", "gemma27b"]


def edit_fn(expl: str) -> dict:
    edited, meta = shuffle_blocks(expl)
    return dict(expl_edit=edited, edit_meta=meta)


def main() -> None:
    docs = load_docs()
    for key in (sys.argv[1:] or KEYS):
        n = build_records(key, "exp5a", edit_fn, docs)
        print(f"[exp5a:{key}] {n} records; wrote edited_reads_exp5a.json + records_exp5a_{key}.json",
              flush=True)


if __name__ == "__main__":
    main()
