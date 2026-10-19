"""Exp 6 — build the control arms F / B / C_otherdoc (records + edited_reads) LOCALLY (CPU, no GPU).

All three replace the SAME LCR quotes as A1/A2, word-length-matched, so they are byte-comparable:
  exp6f = F  : shuffle the AV's OWN LCR quote words (form-OOD floor)
  exp6r = B  : random real span from the same doc's far past (rules out 'any same-doc text helps')
  exp6o = C  : true pre-token tail (token excluded) of a DIFFERENT doc (must hurt; else slots inert)

C_otherdoc pairs each datapoint with a donor via a deterministic derangement over the in-range set,
guaranteeing a different doc_id.

Usage:  python3 exp6_build_controls.py [qwen7b gemma12b gemma27b]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from edits import _seed
from exp6_edits import otherdoc_ctx, random_ctx, shuffle_lcr

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
DOCS = {d["id"]: d["text"] for d in json.loads((HERE / "data" / "ufw_docs.json").read_text())}
KEYS = ["qwen7b", "gemma12b", "gemma27b"]


def donor_map(base: list) -> dict:
    """idx -> donor row (different doc_id), via a fixed-offset rotation with forward-scan for a distinct
    doc. Deterministic (seeded by the key's id list) so the pairing is reproducible."""
    inr = [r for r in base if r.get("in_range") and r.get("char_start") is not None]
    n = len(inr)
    off = max(1, n // 2 + (_seed("exp6o") % 97))               # fixed large offset, coprime-ish
    dm = {}
    for i, r in enumerate(inr):
        j = (i + off) % n
        for _ in range(n):                                     # forward-scan to a different document
            if inr[j]["doc_id"] != r["doc_id"]:
                break
            j = (j + 1) % n
        dm[r["idx"]] = inr[j]
    return dm


def run_key(key: str) -> None:
    base = json.loads((RES / f"records_exp3a_{key}.json").read_text())
    dm = donor_map(base)
    arms = {"exp6f": "shuffle", "exp6r": "random", "exp6o": "otherdoc"}
    for arm, kind in arms.items():
        recs, reads, n_edit = [], {}, 0
        for r in base:
            rec = dict(idx=r["idx"], doc_id=r["doc_id"], position=r.get("position"),
                       n_tokens=r.get("n_tokens"), in_range=r.get("in_range"),
                       token=r.get("token"), token_piece=r.get("token_piece"),
                       char_start=r.get("char_start"), char_end=r.get("char_end"),
                       input_display=r.get("input_display", ""), ctx=r.get("ctx"),
                       expl_orig=r.get("expl_orig", ""),
                       cos_orig=None, cos_edit=None, fve_i_orig=None, fve_i_edit=None)
            doc = DOCS.get(r["doc_id"], "")
            ce, cs = r.get("char_end"), r.get("char_start")
            expl, tok = rec["expl_orig"], (rec["token"] or "")
            if r.get("in_range") and expl and ce is not None and doc:
                if kind == "shuffle":
                    ed = shuffle_lcr(expl, tok, doc, ce, cs)
                elif kind == "random":
                    ed = random_ctx(expl, tok, doc, ce, cs)
                else:
                    d = dm.get(r["idx"])
                    ddoc = DOCS.get(d["doc_id"], "") if d else ""
                    dcut = d["char_start"] if d else 0
                    ed = otherdoc_ctx(expl, tok, doc, ce, cs, ddoc, dcut)
                    rec["donor_doc_id"] = d["doc_id"] if d else None
                rec["expl_edit"] = ed["expl_edit"]
                rec["removed"], rec["added"] = ed["removed"], ed["added"]
                rec["edit_meta"] = ed["edit_meta"]
                reads[str(r["idx"])] = ed["expl_edit"]
                if ed["edit_meta"]["n_removed"]:
                    n_edit += 1
            else:
                rec["expl_edit"] = expl
                rec["edit_meta"] = dict(arm=arm, n_removed=0, chars_removed=0, n_lcr=0, quotes=[])
            recs.append(rec)
        (RES / f"records_{arm}_{key}.json").write_text(json.dumps(recs, ensure_ascii=False))
        (RES / f"edited_reads_{arm}_{key}.json").write_text(
            json.dumps({"reads": reads, "arm": arm}, ensure_ascii=False))
        print(f"[{arm}:{key}] {len(recs)} records, {len(reads)} edited reads ({n_edit} with >=1 LCR replaced)")


def main() -> None:
    for key in (sys.argv[1:] or KEYS):
        run_key(key)


if __name__ == "__main__":
    main()
