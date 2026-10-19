"""Build a tiny self-contained input bundle for the 7B GPU smoke test.

Picks 6 held-out positions spanning the cosine range from the v2_3 calibration
set (the ones that already have mimics), and packs everything the GPU script
needs into ONE json (so only that file + nla_inference.py get staged to /data):
  doc_id, position, llama_token, cosine_stored (Neuronpedia/Llama), real_text
  (the real AV read), mimic_text (v2_3), and the full doc payload.

NOTE: the smoke runs on QWEN, whose tokenizer differs from the Llama corpus, so
the position/token won't align — that's fine, the smoke only validates the
extract->AR-score mechanics, not corpus-matched numbers. Alignment matters only
for the real llama70b run.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
C = HERE.parent / "corpus"
OUT = HERE / "smoke_input.json"
N = 6


def main():
    tasks = json.loads((C / "calib_sample_v2_3.json").read_text())["tasks"]
    payloads = {d["id"]: d["payload"] for d in json.loads((C / "corpus_docs.json").read_text())}
    tok = {}
    for line in (C / "eval.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        tok[(r["doc_id"], r["position"])] = r["token"]

    tasks = sorted(tasks, key=lambda t: t["cosine"])
    idxs = [round(i * (len(tasks) - 1) / (N - 1)) for i in range(N)]
    items = []
    for i in idxs:
        t = tasks[i]
        mp = C / "calib_out_v2_3" / f"mimic_{t['idx']}.txt"
        items.append({
            "idx": t["idx"], "doc_id": t["doc_id"], "theme": t["theme"],
            "position": t["position"], "llama_token": tok.get((t["doc_id"], t["position"]), ""),
            "cosine_stored": t["cosine"], "mse_stored": t["mse"],
            "real_text": t["real_av"].strip(),
            "mimic_text": mp.read_text(encoding="utf-8").strip(),
            "payload": payloads[t["doc_id"]],
        })
    OUT.write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {OUT.name}: {len(items)} items, "
          f"cosine {min(x['cosine_stored'] for x in items):.2f}-{max(x['cosine_stored'] for x in items):.2f}, "
          f"{OUT.stat().st_size:,} bytes")
    for x in items:
        print(f"  idx {x['idx']:3d} {x['theme']:16s} pos {x['position']:3d} "
              f"cos {x['cosine_stored']:.2f} tok {x['llama_token']!r}")


if __name__ == "__main__":
    main()
