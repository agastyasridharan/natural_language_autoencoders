"""Cluster-side: emit the Tier-B judge input for the first N in-range datapoints of one model.

Reuses the Exp 3a records (which already carry token / expl_orig / quote spans + the Tier-A verdict)
and joins the full input document text. -> out_<key>/tierb_input_<key>.json, pulled to the Mac where
the API judge runs (keys live in the repo .env, and the Mac has direct internet).

Usage:  python build_tierb_input.py qwen7b 50
"""
import json
import sys

ROOT = "/data/agastyas/nla_fve_ufw"
KEY = sys.argv[1] if len(sys.argv) > 1 else "qwen7b"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 50

recs = json.load(open(f"{ROOT}/out_{KEY}/records_exp3a_{KEY}.json"))
docs = {d["id"]: d["text"] for d in json.load(open(f"{ROOT}/data/ufw_docs.json"))}

out = []
for r in recs:
    if not r["in_range"]:
        continue
    quotes = [{"id": i, "span": q["span"],
               "text": r["expl_orig"][q["span"][0]:q["span"][1]],
               "tierA_derivable": q["derivable"], "run_content": q["run_content"]}
              for i, q in enumerate(r["edit_meta"]["quotes"])]
    out.append(dict(idx=r["idx"], doc_id=r["doc_id"], position=r["position"],
                    token=r.get("token"), expl=r["expl_orig"], doc=docs[r["doc_id"]],
                    quotes=quotes))
    if len(out) >= N:
        break

json.dump(out, open(f"{ROOT}/out_{KEY}/tierb_input_{KEY}.json", "w"), ensure_ascii=False)
print(f"wrote {len(out)} datapoints ({sum(len(d['quotes']) for d in out)} quotes) "
      f"-> out_{KEY}/tierb_input_{KEY}.json")
