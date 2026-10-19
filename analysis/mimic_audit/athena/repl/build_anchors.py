"""CPU step: turn each llama read position into a tokenizer-independent anchor.

For every run_input item we render the document with the llama markers, tokenize
with offset mapping, locate the read token at the stored `position`, and record
(turn_index, content_offset) plus the decoded token (sanity vs the stored token).
Other models re-anchor to the same (turn_index, content_offset). Writes anchors.json.

Run on the cluster (needs the llama tokenizer; gated, token already set). No GPU.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from repl_common import MODELS, render, content_offset_of, MAX_LEN  # noqa: E402

INPUT = "/data/agastyas/nla_mimic/data/run_input.json"
OUT = "/data/agastyas/nla_mimic/data/anchors.json"


def main() -> int:
    items = json.loads(Path(INPUT).read_text(encoding="utf-8"))
    tok = AutoTokenizer.from_pretrained(MODELS["llama70b"]["base"])
    fam = MODELS["llama70b"]["family"]

    out, ok, tokmatch = [], 0, 0
    for it in items:
        text, spans = render(it["payload"], fam)
        enc = tok(text, add_special_tokens=True, return_offsets_mapping=True,
                  truncation=True, max_length=MAX_LEN)
        offsets = enc["offset_mapping"]
        ids = enc["input_ids"]
        pos = it["position"]
        ti, co, good = content_offset_of(offsets, spans, pos)
        decoded = tok.decode([ids[pos]], skip_special_tokens=False) if pos < len(ids) else ""
        tm = decoded == it["token"]
        ok += good
        tokmatch += tm
        out.append(dict(idx=it["idx"], theme=it["theme"], is_chat=it["is_chat"],
                        turn_index=ti, content_offset=co, anchor_ok=bool(good),
                        llama_token=it["token"], decoded=decoded, token_match=bool(tm),
                        cosine_stored=it["cosine_stored"]))
        if not tm or not good:
            print(f"  [warn] idx{it['idx']} {it['theme']} pos{pos} "
                  f"anchor_ok={good} stored={it['token']!r} decoded={decoded!r} ti={ti} co={co}")

    Path(OUT).write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"\n[anchors] {len(out)} items; anchor_ok {ok}/{len(out)}, "
          f"token_match {tokmatch}/{len(out)} (sanity vs corpus)")
    print(f"[anchors] wrote {OUT}")
    return 0 if tokmatch == len(out) else 1


if __name__ == "__main__":
    raise SystemExit(main())
