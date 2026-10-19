"""Exp 1 (remove all quoted content) — edit step + per-datapoint records.

CPU only (tokenizer + string ops); safe to run alongside GPU jobs. For each model:
re-tokenize each doc EXACTLY as phase_extract did (add_special_tokens, truncate 2048),
recover the exact read token at `position` and its char offsets, apply strip_all_quotes
to the baseline explanation, and emit:

  out_<key>/edited_reads_exp1.json      {"reads": {idx: edited_text}}  (fed to rescore.py)
  out_<key>/records_exp1_<key>.json     rich per-datapoint records for the dashboard
                                        (cos_* / fve_i_* are null until the GPU re-score)

Usage:  python exp1_edit.py            # all three models
        python exp1_edit.py qwen7b     # one model
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "/data/agastyas/nla_fve_ufw/code")
from repl_common import MODELS               # noqa: E402
from edits import strip_all_quotes           # noqa: E402
from transformers import AutoTokenizer       # noqa: E402

ROOT = Path("/data/agastyas/nla_fve_ufw")
DOCS = ROOT / "data" / "ufw_docs.json"
MAX_LEN = 2048
CTX_L, CTX_R = 12, 8          # token-strip window around the read position
KEYS = ["qwen7b", "gemma12b", "gemma27b"]


def build(key: str, docs: dict) -> None:
    cfg = MODELS[key]
    od = ROOT / f"out_{key}"
    manifest = json.loads((od / "manifest.json").read_text())
    reads = json.loads((od / "av_reads.json").read_text())["reads"]
    tok = AutoTokenizer.from_pretrained(cfg["base"], trust_remote_code=True)

    records, edited_reads, n_bad, tot_removed = [], {}, 0, 0
    for m in manifest:
        idx, doc_id, pos = m["idx"], m["doc_id"], m["position"]
        expl = reads.get(str(idx), "") or ""
        edited, spans, meta = strip_all_quotes(expl)
        edited_reads[str(idx)] = edited
        tot_removed += meta["n_removed"]

        rec = dict(idx=idx, doc_id=doc_id, position=pos, n_tokens=m["n_tokens"],
                   in_range=m["in_range"], expl_orig=expl, expl_edit=edited,
                   removed=[[s, e] for s, e in spans], edit_meta=meta,
                   cos_orig=None, cos_edit=None, fve_i_orig=None, fve_i_edit=None)

        if m["in_range"]:
            doc = docs[doc_id]
            enc = tok(doc, add_special_tokens=True, truncation=True,
                      max_length=MAX_LEN, return_offsets_mapping=True)
            ids, offs = enc["input_ids"], enc["offset_mapping"]
            if len(ids) != m["n_tokens"]:
                n_bad += 1                       # tokenization drift (should be 0)
            if 0 <= pos < len(ids):
                s, e = offs[pos]
                max_end = max((oe for _, oe in offs), default=len(doc))
                ctx = [dict(t=tok.decode([ids[j]]), r=(j == pos))
                       for j in range(max(0, pos - CTX_L), min(len(ids), pos + CTX_R + 1))]
                rec.update(token=tok.decode([ids[pos]]),
                           token_piece=tok.convert_ids_to_tokens([ids[pos]])[0],
                           char_start=int(s), char_end=int(e),
                           input_display=doc[:max_end], ctx=ctx)
        records.append(rec)

    (od / "edited_reads_exp1.json").write_text(
        json.dumps({"reads": edited_reads, "arm": "exp1"}, ensure_ascii=False),
        encoding="utf-8")
    (od / f"records_exp1_{key}.json").write_text(
        json.dumps(records, ensure_ascii=False), encoding="utf-8")
    avg = tot_removed / max(1, len(records))
    print(f"[exp1:{key}] {len(records)} records; avg {avg:.2f} spans removed/expl; "
          f"tokenization_drift={n_bad}; wrote edited_reads_exp1.json + records_exp1_{key}.json",
          flush=True)


def main() -> None:
    docs = {d["id"]: d["text"] for d in json.loads(DOCS.read_text(encoding="utf-8"))}
    keys = sys.argv[1:] or KEYS
    for key in keys:
        build(key, docs)


if __name__ == "__main__":
    main()
