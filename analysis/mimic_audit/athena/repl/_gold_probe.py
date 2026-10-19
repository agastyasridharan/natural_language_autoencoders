import json, numpy as np, pathlib
runs = {
 "gemma12b": ("claim_ablation_runs/gemma12b-cp7-fineweb-atoms",
              ["out_corpus_gemma12b", "out_corpus_gemma12b_fineweb"]),
 "gemma27b": ("code/runs/gemma27b-g4-fineweb-atoms",
              ["out_corpus_gemma27b_fineweb", "out_corpus_gemma27b"]),
 "llama70b": ("code/runs/llama70b-l4-fineweb-atoms",
              ["out_corpus_llama70b_fineweb", "out_corpus_llama70b"]),
}
base = pathlib.Path("/data/agastyas/nla_mimic")
for k, (run, golds) in runs.items():
    recs = [json.loads(l) for l in open(base / run / "records.jsonl")]
    r0 = recs[0]
    print("=== %s atom rec[0]: idx=%s doc_id=%s" % (k, r0.get("idx"), r0.get("doc_id")))
    for g in golds:
        p = base / g
        if not (p / "manifest.json").exists():
            print("   %s: MISSING" % g); continue
        man = json.loads((p / "manifest.json").read_text())
        gold = np.load(p / "gold.npy")
        m0 = man[0] if isinstance(man, list) else man
        print("   %s: shape=%s manifest[0].doc_id=%s keys=%s"
              % (g, gold.shape, m0.get("doc_id"), list(m0)[:8]))
