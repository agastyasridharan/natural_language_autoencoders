"""Exp 3a (Tier A: remove input-derivable quotes, keep divergent) — edit + per-datapoint records.

Deterministic, no API. A quoted span is input-derivable iff it shares a contiguous verbatim run of
>= _MIN_CONTENT (2) content words with the input document (which the AR never sees); such quotes are
removed, all divergent quotes and connective prose are kept. Per-quote verdicts (kind, run_content)
are stored in edit_meta.quotes for auditing. cos/FVE stay null until the AR re-score.
Tier B (paraphrase/synonym-aware, API) is the sibling arm Exp 3b.

Usage:  python exp3a_edit.py                # all three models
        python exp3a_edit.py qwen7b         # one model
"""
from __future__ import annotations

import sys

sys.path.insert(0, "/data/agastyas/nla_fve_ufw/code")
from edits import strip_derivable_quotes                # noqa: E402
from records_common import build_records, load_docs      # noqa: E402

KEYS = ["qwen7b", "gemma12b", "gemma27b"]


def edit_fn(expl: str, doc: str) -> dict:
    edited, removed, meta = strip_derivable_quotes(expl, doc)
    return dict(expl_edit=edited, removed=removed, edit_meta=meta)


def main() -> None:
    docs = load_docs()
    for key in (sys.argv[1:] or KEYS):
        n = build_records(key, "exp3a", edit_fn, docs)
        print(f"[exp3a:{key}] {n} records; wrote edited_reads_exp3a.json + records_exp3a_{key}.json",
              flush=True)


if __name__ == "__main__":
    main()
