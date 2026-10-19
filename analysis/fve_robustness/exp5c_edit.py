"""Exp 5c (shuffle the quoted claims inside the final block) — edit + records. Deterministic, no API/GPU.

Within the FINAL block, permutes the CONTENTS of the quoted claim spans (the lead-in clause plus the
~6-8 predicted continuations) among their slots, keeping all connective scaffolding, the read-token
ECHO quote, and every other block fixed. This is the finest-grain claim shuffle the quotes block
enables. cos/FVE stay null until the AR re-score.

Usage:  python exp5c_edit.py [qwen7b|gemma12b|gemma27b ...]
"""
from __future__ import annotations

import sys

sys.path.insert(0, "/data/agastyas/nla_fve_ufw/code")
from edits import shuffle_quoted_claims                   # noqa: E402
from records_common import build_records, load_docs        # noqa: E402

KEYS = ["qwen7b", "gemma12b", "gemma27b"]


def edit_fn(expl: str) -> dict:
    edited, meta = shuffle_quoted_claims(expl)
    return dict(expl_edit=edited, edit_meta=meta)


def main() -> None:
    docs = load_docs()
    for key in (sys.argv[1:] or KEYS):
        n = build_records(key, "exp5c", edit_fn, docs)
        print(f"[exp5c:{key}] {n} records; wrote edited_reads_exp5c.json + records_exp5c_{key}.json",
              flush=True)


if __name__ == "__main__":
    main()
