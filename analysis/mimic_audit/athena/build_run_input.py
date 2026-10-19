"""Build the full input bundle for the llama70b activation-reconstruction run.

One self-contained json with every v2_3 calibration position that has a mimic:
  idx, doc_id, theme, position, token (the stored Neuronpedia/Llama token at that
  position), cosine_stored, mse_stored, real_text (real AV read), mimic_text
  (v2_3), payload (str for raw docs OR list[{role,content}] for chat docs).

extract_llama.py re-tokenizes payload the SAME way the corpus did (raw text, or
format_chat for chat, then add_special_tokens=True) and asserts the token at
`position` == this stored `token` (the alignment gate).
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
C = HERE.parent / "corpus"
OUT = HERE / "run_input.json"


def main():
    tasks = json.loads((C / "calib_sample_v2_3.json").read_text())["tasks"]
    payloads = {d["id"]: d["payload"] for d in json.loads((C / "corpus_docs.json").read_text())}
    tok = {}
    for line in (C / "eval.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        tok[(r["doc_id"], r["position"])] = r["token"]

    items, n_chat = [], 0
    for t in tasks:
        mp = C / "calib_out_v2_3" / f"mimic_{t['idx']}.txt"
        payload = payloads[t["doc_id"]]
        is_chat = not isinstance(payload, str)
        n_chat += is_chat
        items.append({
            "idx": t["idx"], "doc_id": t["doc_id"], "theme": t["theme"],
            "position": t["position"], "token": tok.get((t["doc_id"], t["position"]), ""),
            "cosine_stored": t["cosine"], "mse_stored": t["mse"],
            "is_chat": is_chat,
            "real_text": t["real_av"].strip(),
            "mimic_text": mp.read_text(encoding="utf-8").strip(),
            "payload": payload,
        })
    OUT.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT.name}: {len(items)} items ({n_chat} chat, {len(items)-n_chat} raw), "
          f"{OUT.stat().st_size:,} bytes, "
          f"cosine {min(x['cosine_stored'] for x in items):.2f}-{max(x['cosine_stored'] for x in items):.2f}")


if __name__ == "__main__":
    main()
