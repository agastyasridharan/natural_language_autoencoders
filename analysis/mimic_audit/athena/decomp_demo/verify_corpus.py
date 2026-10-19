#!/usr/bin/env python3
"""Independently verify that every claim in corpus_data.js is accurate and reflected in the data.

Recomputes the headline statistics straight from the raw cache (NOT via build_corpus), checks
them against what corpus_data.js asserts, checks the provenance block against run.json, and
confirms the decompositions were actually run on the run's verbalizer outputs (cache z_inner ==
records.jsonl z_inner, same read_token / doc_id). Also re-checks that every atom's char offsets
are exact and the Layer-1 partition reproduces each text. Exits non-zero on any mismatch.
"""
from __future__ import annotations
import json
from pathlib import Path
from statistics import median, mean

HERE = Path(__file__).resolve().parent
ATHENA = HERE.parent
CACHE = ATHENA / "segments" / "qwen7b.jsonl"
RUN = ATHENA / "runs" / "qwen7b-20260622-121049-fineweb"
CORPUS_JS = HERE / "corpus_data.js"
ROLES = ("describes", "quotes", "predicts")
CONTENT = ("theme", "detail")

P = []
def ck(name, ok, detail=""):
    P.append((name, bool(ok), detail))

def load_cache():
    by_idx = {}
    for line in CACHE.read_text().splitlines():
        line = line.strip()
        if line:
            r = json.loads(line)
            by_idx[int(r["idx"])] = r        # last line for an idx wins (matches segment_run)
    return [r for r in by_idx.values() if not r.get("error")]

def load_corpus_js():
    txt = CORPUS_JS.read_text().strip()
    assert txt.startswith("window.CORPUS = "), "unexpected corpus_data.js header"
    body = txt[len("window.CORPUS = "):].rstrip().rstrip(";")
    return json.loads(body)

def counter(seq):
    out = {}
    for x in seq:
        out[x] = out.get(x, 0) + 1
    return out

def main():
    recs = load_cache()
    C = load_corpus_js()
    S = C["stats"]
    run = json.load(open(RUN / "run.json"))
    records = {json.loads(l)["idx"]: json.loads(l) for l in open(RUN / "records.jsonl") if l.strip()}

    # ---- recompute headline stats straight from the cache ----
    atoms = [a for r in recs for a in r["atoms"]]
    content = [a for a in atoms if a["type"] in CONTENT]
    details = [a for a in content if a["type"] == "detail"]
    ents = [e for r in recs for e in r["entities"]]
    cpa = [r["n_content"] for r in recs]
    cov = [r["coverage"]["base_frac"] for r in recs]

    ck("n_activations", S["n_activations"] == len(recs), f"{S['n_activations']} vs {len(recs)}")
    ck("n_atoms", S["n_atoms"] == len(atoms), f"{S['n_atoms']} vs {len(atoms)}")
    ck("n_content", S["n_content"] == len(content), f"{S['n_content']} vs {len(content)}")
    ck("n_entity", S["n_entity"] == len(ents), f"{S['n_entity']} vs {len(ents)}")
    ck("n_connective", S["n_connective"] == sum(r["n_connective"] for r in recs),
       f"{S['n_connective']} vs {sum(r['n_connective'] for r in recs)}")
    ck("type_mix", S["type_mix"] == counter(a["type"] for a in content), str(S["type_mix"]))
    rm = {r: sum(1 for a in content if a["role"] == r) for r in ROLES}
    ck("role_mix", S["role_mix"] == rm, str(rm))
    ck("entity_subtypes", S["entity_subtypes"] == counter(e["subtype"] for e in ents), str(S["entity_subtypes"]))
    ck("claims/act median", S["claims_per_act"]["median"] == median(cpa),
       f"{S['claims_per_act']['median']} vs {median(cpa)}")
    ck("claims/act min/max", S["claims_per_act"]["min"] == min(cpa) and S["claims_per_act"]["max"] == max(cpa),
       f"{S['claims_per_act']['min']}-{S['claims_per_act']['max']} vs {min(cpa)}-{max(cpa)}")
    ck("coverage mean", abs(S["coverage"]["mean"] - round(mean(cov), 4)) < 1e-9,
       f"{S['coverage']['mean']} vs {round(mean(cov),4)}")
    ck("regex-confirmed detail count", S["regex_confirmed"]["n"] == sum(1 for a in details if a.get("regex_confirmed"))
       and S["regex_confirmed"]["total"] == len(details), str(S["regex_confirmed"]))
    ck("hedged count", S["hedged"]["n"] == sum(1 for a in content if a.get("hedged"))
       and S["hedged"]["total"] == len(content), str(S["hedged"]))
    ck("claims/act mean", S["claims_per_act"]["mean"] == round(mean(cpa), 1), str(S["claims_per_act"]["mean"]))
    epa = [r["n_entity"] for r in recs]
    ck("entities/act median/mean", S["entities_per_act"]["median"] == median(epa)
       and S["entities_per_act"]["mean"] == round(mean(epa), 1), str(S["entities_per_act"]))
    cfrac = [r["coverage"]["content_frac"] for r in recs]
    ck("content_frac mean", abs(S["content_frac"]["mean"] - round(mean(cfrac), 4)) < 1e-9,
       str(S["content_frac"]["mean"]))
    ck("run_id + tool verbatim", C["provenance"]["run_id"] == run["run_id"]
       and C["provenance"]["tool"] == "emit_layers")
    ck("sum theme+detail == n_content", sum(S["type_mix"].values()) == S["n_content"])
    ck("sum roles == n_content", sum(S["role_mix"].values()) == S["n_content"])
    ck("sum entity subtypes == n_entity", sum(S["entity_subtypes"].values()) == S["n_entity"])

    # ---- provenance block matches run.json ----
    pr = C["provenance"]
    ck("prov verbalizer_av", pr["verbalizer_av"] == run["av"], pr["verbalizer_av"])
    ck("prov reconstructor_ar", pr["reconstructor_ar"] == run["ar"], pr["reconstructor_ar"])
    ck("prov base", pr["base"] == run["base"], pr["base"])
    ck("prov layer", pr["layer"] == run["layer"], str(pr["layer"]))
    ck("prov d_model", pr["d_model"] == run["d_model"], str(pr["d_model"]))
    ck("prov dataset", pr["dataset"] == run["dataset"], pr["dataset"])
    ck("prov injection_scale", pr["injection_scale"] == run["injection_scale"], str(pr["injection_scale"]))
    ck("prov av_temperature", pr["av_temperature"] == run["temperature"], str(pr["av_temperature"]))
    ck("prov n_activations == run n_records", pr["n_activations"] == run["n_records"], str(pr["n_activations"]))
    ck("prov n_cjk", pr["n_cjk"] == run.get("n_cjk"), str(pr["n_cjk"]))
    ck("prov segmenter_model is claude-sonnet-4-6", pr["segmenter_model"] == "claude-sonnet-4-6", pr["segmenter_model"])
    ck("prov segmenter_models matches cache", pr.get("segmenter_models") == counter(r.get("model") for r in recs),
       str(pr.get("segmenter_models")))
    ck("no degenerate tiling (one atom covering >60% of any text)",
       all(max((a["end"] - a["start"]) / (len(r["z_inner"]) or 1) for a in r["atoms"]) <= 0.6 for r in recs),
       "a text collapsed to a single atom")
    # every referenced file exists
    for label, rel in pr["files"].items():
        path = (ATHENA / rel.split("  ")[0].split(" (")[0].replace("athena/", "")).resolve()
        ck(f"prov file exists: {label}", path.exists(), str(path))

    # ---- the decompositions were run on the run's actual verbalizer outputs ----
    zmis = tokmis = docmis = 0
    for r in recs:
        rec = records.get(r["idx"])
        if rec is None:
            zmis += 1; continue
        if r["z_inner"] != rec["z_inner"]:
            zmis += 1
        if r["read_token"] != rec["read_token"]:
            tokmis += 1
        if r["doc_id"] != rec["doc_id"]:
            docmis += 1
    ck("cache z_inner == run records z_inner (segmented the real AV outputs)", zmis == 0, f"{zmis} mismatches")
    ck("cache read_token matches run", tokmis == 0, f"{tokmis} mismatches")
    ck("cache doc_id matches run", docmis == 0, f"{docmis} mismatches")

    # ---- offsets exact + partition reproduces text (all activations) ----
    bad_a = bad_e = tile = 0
    for r in recs:
        t = r["z_inner"]
        for a in r["atoms"]:
            if t[a["start"]:a["end"]] != a["text"]:
                bad_a += 1
        if "".join(t[a["start"]:a["end"]] for a in sorted(r["atoms"], key=lambda x: x["start"])) != t:
            tile += 1
        for e in r["entities"]:
            for s, en in e["spans"]:
                if t[s:en] != e["text"]:
                    bad_e += 1
    ck("all atom offsets exact", bad_a == 0, f"{bad_a} bad")
    ck("all entity occurrences exact", bad_e == 0, f"{bad_e} bad")
    ck("partition reproduces every text", tile == 0, f"{tile} fail")

    # ---- the per-activation strip matches the cache ----
    ck("rows count == n_activations", len(C["rows"]) == len(recs), f"{len(C['rows'])} vs {len(recs)}")
    by_idx = {r["idx"]: r for r in recs}
    rowbad = 0
    for row in C["rows"]:
        r = by_idx.get(row["idx"])
        if r is None:
            rowbad += 1; continue
        exp_roles = {rr: sum(1 for a in r["atoms"] if a["type"] != "connective" and a["role"] == rr) for rr in ROLES}
        exp_types = {t: sum(1 for a in r["atoms"] if a["type"] == t) for t in CONTENT}
        if row["roles"] != exp_roles or row["types"] != exp_types or row["n_content"] != r["n_content"] \
           or row["n_entity"] != r["n_entity"]:
            rowbad += 1
    ck("every strip row matches its cache record", rowbad == 0, f"{rowbad} bad rows")

    # ---- corpus_detail.js (click-to-expand full decompositions) matches the cache exactly ----
    dtxt = (HERE / "corpus_detail.js").read_text().strip()
    ck("detail: file header", dtxt.startswith("window.CORPUS_DETAIL = "))
    D = json.loads(dtxt[len("window.CORPUS_DETAIL = "):].rstrip().rstrip(";"))
    ck("detail: one entry per activation, keyed by idx", len(D) == len(recs), f"{len(D)} vs {len(recs)}")
    dbad = 0
    for r in recs:
        e = D.get(str(r["idx"]))
        if e is None or e["text"] != r["z_inner"] or e["atoms"] != r["atoms"] \
           or e["entities"] != r["entities"] or e["read_token"] != r["read_token"] \
           or e["doc_id"] != r["doc_id"] or e["coverage"] != r["coverage"]:
            dbad += 1
    ck("detail: every entry matches its cache record (text/atoms/entities/coverage)", dbad == 0, f"{dbad} bad")

    npass = sum(1 for _, ok, _ in P if ok)
    nfail = len(P) - npass
    for name, ok, detail in P:
        print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  [{detail}]"))
    print(f"\n{npass} passed, {nfail} failed")
    raise SystemExit(1 if nfail else 0)


if __name__ == "__main__":
    main()
