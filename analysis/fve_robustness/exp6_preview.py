"""Exp 6 — local PREVIEW records for the dashboard (CPU, no GPU/API).

Builds results/records_exp6_<key>.json for the A2 core substitution (LCR quotes -> true pre-token
context tail, read token EXCLUDED) by reusing the frozen Exp-3a records (token, char offsets,
input_display, ctx, in_range, position) + data/ufw_docs.json. cos/FVE stay null (pending the GPU
re-score) exactly like the other tabs before their re-score. Lets the substitution quality be
inspected before spending any GPU. The real multi-arm records come later from the cluster.

Usage:  python3 exp6_preview.py            # all three models, all in-range datapoints
        python3 exp6_preview.py qwen7b     # one model
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from exp6_edits import substitute_lcr

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
DOCS = {d["id"]: d["text"] for d in json.loads((HERE / "data" / "ufw_docs.json").read_text())}
KEYS = ["qwen7b", "gemma12b", "gemma27b"]


def run_key(key: str) -> int:
    base = json.loads((RES / f"records_exp3a_{key}.json").read_text())
    out = []
    n_edit = 0
    for r in base:
        rec = dict(idx=r["idx"], doc_id=r["doc_id"], position=r.get("position"),
                   n_tokens=r.get("n_tokens"), in_range=r.get("in_range"),
                   token=r.get("token"), token_piece=r.get("token_piece"),
                   char_start=r.get("char_start"), char_end=r.get("char_end"),
                   input_display=r.get("input_display", ""), ctx=r.get("ctx"),
                   expl_orig=r.get("expl_orig", ""),
                   cos_orig=None, cos_edit=None, fve_i_orig=None, fve_i_edit=None)
        doc = DOCS.get(r["doc_id"], "")
        ce = r.get("char_end")
        if r.get("in_range") and rec["expl_orig"] and ce is not None and doc:
            ed = substitute_lcr(rec["expl_orig"], rec["token"] or "", doc, ce, r.get("char_start"),
                                include_token=False)                    # A2 (token excluded)
            rec["expl_edit"] = ed["expl_edit"]
            rec["removed"] = ed["removed"]
            rec["added"] = ed["added"]
            rec["edit_meta"] = ed["edit_meta"]
            if ed["edit_meta"]["n_removed"]:
                n_edit += 1
        else:
            rec["expl_edit"] = rec["expl_orig"]
            rec["edit_meta"] = dict(arm="exp6_A2", n_removed=0, chars_removed=0, n_lcr=0, quotes=[])
        out.append(rec)
    (RES / f"records_exp6_{key}.json").write_text(json.dumps(out, ensure_ascii=False))
    return len(out), n_edit


def main() -> None:
    for key in (sys.argv[1:] or KEYS):
        n, ne = run_key(key)
        print(f"[exp6-preview:{key}] {n} records ({ne} with >=1 LCR substituted) -> records_exp6_{key}.json")


if __name__ == "__main__":
    main()
