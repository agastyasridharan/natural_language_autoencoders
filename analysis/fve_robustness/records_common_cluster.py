"""Shared per-datapoint record builder for the FVE-robustness arms.

Given an `edit_fn` mapping an AV explanation -> dict(expl_edit, [removed]/[kept], edit_meta),
this re-tokenizes each doc EXACTLY as phase_extract did, recovers the exact read token +
char offsets, and emits (same schema as Exp 1; cos/FVE null until the AR re-score):

  out_<key>/edited_reads_<arm>.json     {"reads": {idx: edited_text}, "arm": arm}
  out_<key>/records_<arm>_<key>.json    rich per-datapoint records for the dashboard

CPU only. Reuse across arms so every experiment produces an identical record schema.
"""
from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

sys.path.insert(0, "/data/agastyas/nla_fve_ufw/code")
from repl_common import MODELS                 # noqa: E402
from transformers import AutoTokenizer         # noqa: E402

ROOT = Path("/data/agastyas/nla_fve_ufw")
DOCS = ROOT / "data" / "ufw_docs.json"
MAX_LEN = 2048
CTX_L, CTX_R = 12, 8


def load_docs() -> dict:
    return {d["id"]: d["text"] for d in json.loads(DOCS.read_text(encoding="utf-8"))}


def build_records(key: str, arm: str, edit_fn, docs: dict | None = None) -> int:
    if docs is None:
        docs = load_docs()
    cfg = MODELS[key]
    od = ROOT / f"out_{key}"
    manifest = json.loads((od / "manifest.json").read_text())
    reads = json.loads((od / "av_reads.json").read_text())["reads"]
    tok = AutoTokenizer.from_pretrained(cfg["base"], trust_remote_code=True)
    n_params = len(inspect.signature(edit_fn).parameters)   # edit_fn(expl) or edit_fn(expl, doc)

    edited_reads, recs = {}, []
    for m in manifest:
        idx, doc_id, pos = m["idx"], m["doc_id"], m["position"]
        expl = reads.get(str(idx), "") or ""
        doc_text = docs.get(doc_id, "")
        ed = edit_fn(expl, doc_text) if n_params >= 2 else edit_fn(expl)   # dict: expl_edit, [removed]/[kept], edit_meta
        edited = ed["expl_edit"]
        edited_reads[str(idx)] = edited
        rec = dict(idx=idx, doc_id=doc_id, position=pos, n_tokens=m["n_tokens"],
                   in_range=m["in_range"], expl_orig=expl, expl_edit=edited,
                   edit_meta=ed.get("edit_meta", {}),
                   cos_orig=None, cos_edit=None, fve_i_orig=None, fve_i_edit=None)
        if "removed" in ed:
            rec["removed"] = [[int(s), int(e)] for s, e in ed["removed"]]
        if "kept" in ed:
            rec["kept"] = [[int(s), int(e)] for s, e in ed["kept"]]
        if m["in_range"]:
            doc = doc_text
            enc = tok(doc, add_special_tokens=True, truncation=True,
                      max_length=MAX_LEN, return_offsets_mapping=True)
            ids, offs = enc["input_ids"], enc["offset_mapping"]
            if 0 <= pos < len(ids):
                s, e = offs[pos]
                max_end = max((oe for _, oe in offs), default=len(doc))
                ctx = [dict(t=tok.decode([ids[j]]), r=(j == pos))
                       for j in range(max(0, pos - CTX_L), min(len(ids), pos + CTX_R + 1))]
                rec.update(token=tok.decode([ids[pos]]),
                           token_piece=tok.convert_ids_to_tokens([ids[pos]])[0],
                           char_start=int(s), char_end=int(e),
                           input_display=doc[:max_end], ctx=ctx)
        recs.append(rec)

    (od / f"edited_reads_{arm}.json").write_text(
        json.dumps({"reads": edited_reads, "arm": arm}, ensure_ascii=False), encoding="utf-8")
    (od / f"records_{arm}_{key}.json").write_text(
        json.dumps(recs, ensure_ascii=False), encoding="utf-8")
    return len(recs)
