"""Exp 2 (keep only quoted content) — edit step + per-datapoint records.

Complement of Exp 1: keep only the double-quoted spans, drop the connective prose.
CPU only; reuses the shared record builder. `kept` = quote spans (tab shows them green,
prose struck). cos/FVE stay null until the AR re-score.

Usage:  python exp2_edit.py            # all three models
        python exp2_edit.py qwen7b     # one model
"""
from __future__ import annotations

import sys

sys.path.insert(0, "/data/agastyas/nla_fve_ufw/code")
from edits import keep_only_quotes                     # noqa: E402
from records_common import build_records, load_docs    # noqa: E402

KEYS = ["qwen7b", "gemma12b", "gemma27b"]


def edit_fn(expl: str) -> dict:
    edited, kept, meta = keep_only_quotes(expl)
    return dict(expl_edit=edited, kept=kept, edit_meta=meta)


def main() -> None:
    docs = load_docs()
    for key in (sys.argv[1:] or KEYS):
        n = build_records(key, "exp2", edit_fn, docs)
        print(f"[exp2:{key}] {n} records; wrote edited_reads_exp2.json + records_exp2_{key}.json",
              flush=True)


if __name__ == "__main__":
    main()
