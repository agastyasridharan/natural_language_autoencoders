"""Exp 6 — build the A1/A2 substitution arms (records + edited_reads) LOCALLY (CPU, no GPU/API).

Two distinct arms, reusing the frozen Exp-3a records (token, char offsets, input_display, ctx) + docs:
  exp6a = A1: LCR quotes -> true pre-token context tail, read token INCLUDED (ends at char_end)
  exp6b = A2: LCR quotes -> true pre-token context tail, read token EXCLUDED (ends at char_start)  [load-bearing]

Writes per (arm,key):
  results/records_<arm>_<key>.json        dashboard records (cos/FVE null until GPU re-score)
  results/edited_reads_<arm>_<key>.json   {"reads": {idx: edited}, "arm": arm}  (pushed to cluster for rescore.py)

Usage:  python3 exp6_build_arms.py [qwen7b gemma12b gemma27b]
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
ARMS = [("exp6a", True), ("exp6b", False)]          # (arm, include_token)


def run_key(key: str) -> None:
    base = json.loads((RES / f"records_exp3a_{key}.json").read_text())
    for arm, incl in ARMS:
        recs, reads = [], {}
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
                ed = substitute_lcr(rec["expl_orig"], rec["token"] or "", doc, ce,
                                    r.get("char_start"), include_token=incl)
                rec["expl_edit"] = ed["expl_edit"]
                rec["removed"], rec["added"] = ed["removed"], ed["added"]
                rec["edit_meta"] = dict(ed["edit_meta"], arm=arm)
                reads[str(r["idx"])] = ed["expl_edit"]
                if ed["edit_meta"]["n_removed"]:
                    n_edit += 1
            else:
                rec["expl_edit"] = rec["expl_orig"]
                rec["edit_meta"] = dict(arm=arm, n_removed=0, chars_removed=0, n_lcr=0, quotes=[])
            recs.append(rec)
        (RES / f"records_{arm}_{key}.json").write_text(json.dumps(recs, ensure_ascii=False))
        (RES / f"edited_reads_{arm}_{key}.json").write_text(
            json.dumps({"reads": reads, "arm": arm}, ensure_ascii=False))
        print(f"[{arm}:{key}] {len(recs)} records, {len(reads)} edited reads ({n_edit} with >=1 LCR substituted)")


def main() -> None:
    for key in (sys.argv[1:] or KEYS):
        run_key(key)


if __name__ == "__main__":
    main()
