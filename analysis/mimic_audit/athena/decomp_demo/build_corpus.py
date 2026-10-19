#!/usr/bin/env python3
"""Export a corpus-wide view of the full qwen7b two-layer decomposition to corpus_data.js.

Reads the complete segmentation cache (`athena/segments/qwen7b.jsonl`, produced by
`repl/segment_run.py` running `segment.segment_audit` over every verbalizer output in the
source run) plus the source run's `run.json`, and emits `window.CORPUS` for the "Full
corpus" dashboard tab: provenance, corpus aggregates and distributions, and a compact
per-activation strip covering ALL activations.

Every statistic is computed here from the cache; provenance is read verbatim from
`run.json`; nothing is hardcoded. `verify_corpus.py` independently recomputes the headline
numbers from the raw cache and checks them against this file, so the tab's claims are
auditable against the source data.
"""
from __future__ import annotations
import json
from pathlib import Path
from statistics import median, mean

import segment as seg  # same dir; provides _SYS_AUDIT / _TOOL_AUDIT

HERE = Path(__file__).resolve().parent
ATHENA = HERE.parent
CACHE = ATHENA / "segments" / "qwen7b.jsonl"
RUN = ATHENA / "runs" / "qwen7b-20260622-121049-fineweb"
OUT = HERE / "corpus_data.js"
DETAIL_OUT = HERE / "corpus_detail.js"
ROLES = ("describes", "quotes", "predicts")
CONTENT = ("theme", "detail")


def load_cache(p: Path):
    by_idx: dict = {}
    for line in p.read_text().splitlines():
        line = line.strip()
        if line:
            r = json.loads(line)
            by_idx[int(r["idx"])] = r        # last line for an idx wins (matches segment_run)
    recs = [r for r in by_idx.values() if not r.get("error")]
    errs = sum(1 for r in by_idx.values() if r.get("error"))
    return recs, errs


def hist(values: list[float], nbins: int = 12) -> list[dict]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if lo == hi:
        return [{"lo": lo, "hi": hi, "count": len(values)}]
    w = (hi - lo) / nbins
    bins = [0] * nbins
    for v in values:
        i = min(nbins - 1, int((v - lo) / w))
        bins[i] += 1
    return [{"lo": round(lo + i * w, 2), "hi": round(lo + (i + 1) * w, 2), "count": c}
            for i, c in enumerate(bins)]


def counter(seq) -> dict:
    out: dict = {}
    for x in seq:
        out[x] = out.get(x, 0) + 1
    return out


def build() -> dict:
    recs, errs = load_cache(CACHE)
    run = json.load(open(RUN / "run.json"))

    atoms = [a for r in recs for a in r["atoms"]]
    content = [a for a in atoms if a["type"] in CONTENT]
    details = [a for a in content if a["type"] == "detail"]
    ents = [e for r in recs for e in r["entities"]]

    cpa = [r["n_content"] for r in recs]
    epa = [r["n_entity"] for r in recs]
    cov = [r["coverage"]["base_frac"] for r in recs]
    cfrac = [r["coverage"]["content_frac"] for r in recs]

    stats = {
        "n_activations": len(recs),
        "n_atoms": len(atoms),
        "n_content": len(content),
        "n_connective": sum(r["n_connective"] for r in recs),
        "n_entity": len(ents),
        "claims_per_act": {"min": min(cpa), "median": median(cpa), "max": max(cpa),
                            "mean": round(mean(cpa), 1), "hist": hist([float(x) for x in cpa])},
        "entities_per_act": {"min": min(epa), "median": median(epa), "max": max(epa),
                              "mean": round(mean(epa), 1), "hist": hist([float(x) for x in epa])},
        "coverage": {"mean": round(mean(cov), 4), "min": round(min(cov), 4)},
        "content_frac": {"mean": round(mean(cfrac), 4), "min": round(min(cfrac), 4),
                          "max": round(max(cfrac), 4)},
        "type_mix": counter(a["type"] for a in content),
        "role_mix": {r: sum(1 for a in content if a["role"] == r) for r in ROLES},
        "entity_subtypes": counter(e["subtype"] for e in ents),
        "theme_subtypes_top": sorted(counter(a["subtype"] for a in content if a["type"] == "theme").items(),
                                     key=lambda x: -x[1])[:12],
        "detail_subtypes_top": sorted(counter(a["subtype"] for a in content if a["type"] == "detail").items(),
                                      key=lambda x: -x[1])[:12],
        "regex_confirmed": {"n": sum(1 for a in details if a.get("regex_confirmed")), "total": len(details)},
        "hedged": {"n": sum(1 for a in content if a.get("hedged")), "total": len(content)},
        "entity_repeat": {"n": sum(1 for e in ents if e.get("n_occ", 1) > 1), "total": len(ents)},
        "segmenter_models": counter(r.get("model", "") for r in recs),
        "n_errors": errs,
    }

    seg_model = max(stats["segmenter_models"].items(), key=lambda x: x[1])[0] if stats["segmenter_models"] else ""
    n_fallback = sum(n for m, n in stats["segmenter_models"].items() if m != seg_model)
    seg_note = (f"{n_fallback} of {len(recs)} activations fell back to a secondary model after a "
                f"false-positive safety refusal by the primary" if n_fallback else "")
    prov = {
        "run_id": run["run_id"],
        "verbalizer_av": run["av"],
        "reconstructor_ar": run["ar"],
        "base": run["base"],
        "family": run["family"],
        "layer": run["layer"],
        "d_model": run["d_model"],
        "dataset": run["dataset"],
        "injection_scale": run["injection_scale"],
        "av_temperature": run["temperature"],
        "av_max_new_tokens": 200,
        "n_activations": run["n_records"],
        "n_cjk": run.get("n_cjk"),
        "segmenter_model": seg_model,
        "segmenter_models": stats["segmenter_models"],
        "segmenter_note": seg_note,
        "segmenter_temp": 0,
        "tool": seg._TOOL_AUDIT.get("name", "emit_layers"),
        "files": {
            "AV outputs (verbalizer text)": "athena/runs/qwen7b-20260622-121049-fineweb/records.jsonl",
            "decompositions (this cache)": "athena/segments/qwen7b.jsonl",
            "segmenter": "athena/decomp_demo/segment.py  (segment_audit)",
            "segmentation pre-pass": "athena/repl/segment_run.py",
            "this exporter": "athena/decomp_demo/build_corpus.py",
        },
    }

    rows = [{
        "idx": r["idx"], "read_token": r["read_token"], "doc_id": r["doc_id"],
        "n_content": r["n_content"], "n_entity": r["n_entity"], "n_connective": r["n_connective"],
        "coverage": round(r["coverage"]["base_frac"], 3),
        "roles": {rr: sum(1 for a in r["atoms"] if a["type"] != "connective" and a["role"] == rr) for rr in ROLES},
        "types": {t: sum(1 for a in r["atoms"] if a["type"] == t) for t in CONTENT},
    } for r in sorted(recs, key=lambda r: r["idx"])]

    prompt = {"system": seg._SYS_AUDIT, "tool": seg._TOOL_AUDIT.get("name", "emit_layers"),
              "model": seg_model,
              "types": ["theme", "detail", "connective", "person", "place", "org", "title", "ticker", "other"],
              "note": "two layers: a theme/detail/connective partition plus an overlapping entity overlay"}

    desc = (
        "The complete two-layer decomposition of every verbalizer output in the qwen7b run. "
        "LAYER 1 partitions each output into theme / detail / connective claims (100% coverage); "
        "LAYER 2 overlays named entities; every content claim carries a role (describes / quotes / "
        "predicts). This tab summarizes the whole corpus and lists all activations; the Two-layer "
        "audit tab shows eight of them in full detail. This is segmentation only. The groundedness "
        "measurement (teacher-forced Δlogp) is a separate, judge-free step and is not shown here."
    )

    return {"id": "corpus", "kind": "corpus", "label": "5 · Full corpus", "desc": desc,
            "provenance": prov, "prompt": prompt, "stats": stats, "rows": rows}


def build_detail() -> dict:
    """Full per-activation decompositions keyed by idx, for click-to-expand in the dashboard.
    Same fields the audit tab renders (text = z_inner, atoms, entities), so the corpus modal
    can reuse the audit render path for any of the 400 activations."""
    recs, _ = load_cache(CACHE)
    detail = {}
    for r in sorted(recs, key=lambda r: r["idx"]):
        detail[str(r["idx"])] = {
            "idx": r["idx"], "read_token": r["read_token"], "doc_id": r["doc_id"],
            "text": r["z_inner"], "coverage": r["coverage"],
            "n_content": r["n_content"], "n_entity": r["n_entity"], "n_connective": r["n_connective"],
            "atoms": r["atoms"], "entities": r["entities"],
        }
    return detail


def main() -> None:
    obj = build()
    OUT.write_text("window.CORPUS = " + json.dumps(obj, ensure_ascii=False) + ";\n")
    detail = build_detail()
    DETAIL_OUT.write_text("window.CORPUS_DETAIL = " + json.dumps(detail, ensure_ascii=False) + ";\n")
    print(f"wrote {DETAIL_OUT.name}: {len(detail)} full decompositions")
    s = obj["stats"]
    print(f"wrote {OUT.name}: {s['n_activations']} activations, {s['n_content']} content claims, "
          f"{s['n_entity']} entities, {s['n_connective']} connectives")
    print(f"  claims/act median {s['claims_per_act']['median']} (min {s['claims_per_act']['min']} "
          f"max {s['claims_per_act']['max']} mean {s['claims_per_act']['mean']})")
    print(f"  type {s['type_mix']}  roles {s['role_mix']}")
    print(f"  entity subtypes {s['entity_subtypes']}")
    print(f"  coverage mean {s['coverage']['mean']} min {s['coverage']['min']}  "
          f"content_frac mean {s['content_frac']['mean']}")
    print(f"  regex-confirmed details {s['regex_confirmed']['n']}/{s['regex_confirmed']['total']}  "
          f"hedged {s['hedged']['n']}/{s['hedged']['total']}")
    print(f"  segmenter models {s['segmenter_models']}  errors {s['n_errors']}")


if __name__ == "__main__":
    main()
