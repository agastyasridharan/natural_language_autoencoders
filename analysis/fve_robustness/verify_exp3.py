"""Ad-hoc: print the new Tier-A per-quote decisions for one datapoint (default the reported bug)."""
import json
import sys

sys.path.insert(0, "/data/agastyas/nla_fve_ufw/code")
from edits import (strip_derivable_quotes, _coalesce, _straight_spans,  # noqa: E402
                   _curly_spans, classify_quote, build_doc_index)

KEY = sys.argv[1] if len(sys.argv) > 1 else "qwen7b"
DOC_ID = sys.argv[2] if len(sys.argv) > 2 else "05f758ebaab32aed"
POS = int(sys.argv[3]) if len(sys.argv) > 3 else 1671

ROOT = "/data/agastyas/nla_fve_ufw"
man = json.load(open(f"{ROOT}/out_{KEY}/manifest.json"))
reads = json.load(open(f"{ROOT}/out_{KEY}/av_reads.json"))["reads"]
docs = {d["id"]: d["text"] for d in json.load(open(f"{ROOT}/data/ufw_docs.json"))}

m = [x for x in man if x["doc_id"] == DOC_ID and x["position"] == POS][0]
doc, expl = docs[m["doc_id"]], reads[str(m["idx"])]
idx = build_doc_index(doc)

print("=== per-quote decisions (NEW classifier) ===")
for s, e in _coalesce(_straight_spans(expl) + _curly_spans(expl)):
    seg = expl[s:e]
    v = classify_quote(seg, idx)
    tag = "REMOVE" if v["derivable"] else "keep  "
    print("  [%s] run_content=%d run_words=%d kind=%-9s :: %r"
          % (tag, v["run_content"], v["run_words"], v["kind"], seg[:70]))

edited, removed, meta = strip_derivable_quotes(expl, doc)
print("=== meta ===", {k: meta[k] for k in
                       ("n_quotes", "n_derivable", "n_divergent", "n_verbatim", "n_phrase", "chars_removed")})
print("=== EDITED explanation ===")
print(edited)
