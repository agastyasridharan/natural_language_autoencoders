#!/usr/bin/env python3
"""Export the two-layer audit segmentation cache to a JS data file for the dashboard.

Reads the resumable segmentation cache written by `repl/segment_run.py`
(`athena/segments/<key>.jsonl`, one record per activation, produced by
`segment.segment_audit`) and emits `audit_data.js` = `window.AUDIT = {...}` for the
"Two-layer audit" tab in `index.html`.

The tab illustrates a small curated set of REAL run activations (not the demo genre
corpus the v1/v2/v3 tabs use), so by default we pick a fixed set of eight indices; if
any are missing we fall back to the first eight cached records. Stats and the exact
segmentation prompt are computed / pulled from the data and `segment.py`, never
hardcoded, so the tab stays accurate to what the segmenter actually produced.
"""
from __future__ import annotations
import json
from pathlib import Path
from statistics import median, mean

import segment as seg  # same dir; provides _SYS_AUDIT / _TOOL_AUDIT

HERE = Path(__file__).resolve().parent
CACHE = HERE.parent / "segments" / "qwen7b.jsonl"
OUT = HERE / "audit_data.js"
DEMO_IDXS = [0, 20, 50, 100, 137, 200, 300, 380]
ROLES = ("describes", "quotes", "predicts")


def load(path: Path) -> list[dict]:
    recs = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if not r.get("error"):
            recs.append(r)
    return recs


def to_example(r: dict) -> dict:
    """Shape one cache record into the tab's example schema (text=z_inner)."""
    return {
        "idx": r["idx"],
        "doc_id": r["doc_id"],
        "read_token": r["read_token"],
        "text": r["z_inner"],
        "model": r.get("model", ""),
        "coverage": r["coverage"],
        "n_content": r["n_content"],
        "n_connective": r["n_connective"],
        "n_entity": r["n_entity"],
        "atoms": r["atoms"],        # {text,type,subtype,start,end,hedged,assertion,role,regex_confirmed}
        "entities": r["entities"],  # {text,subtype,start,end,spans,n_occ,parent}
    }


def compute_stats(examples: list[dict]) -> dict:
    content = [e["n_content"] for e in examples]
    atoms = [a for e in examples for a in e["atoms"]]
    ents = [x for e in examples for x in e["entities"]]
    content_atoms = [a for a in atoms if a["type"] in ("theme", "detail")]

    def count(seq, key):
        out: dict[str, int] = {}
        for x in seq:
            out[x[key]] = out.get(x[key], 0) + 1
        return out

    return {
        "n_activations": len(examples),
        "content_total": sum(content),
        "content_median": median(content) if content else 0,
        "content_min": min(content) if content else 0,
        "content_max": max(content) if content else 0,
        "coverage_mean": mean(e["coverage"]["base_frac"] for e in examples) if examples else 0,
        "content_frac_mean": mean(e["coverage"]["content_frac"] for e in examples) if examples else 0,
        "type_mix": count(content_atoms, "type"),                 # theme / detail
        "roles": {r: sum(1 for a in content_atoms if a["role"] == r) for r in ROLES},
        "entity_total": len(ents),
        "entity_median": median([e["n_entity"] for e in examples]) if examples else 0,
        "entity_subtypes": count(ents, "subtype"),
        "connective_total": sum(e["n_connective"] for e in examples),
    }


def build() -> dict:
    recs = load(CACHE)
    by_idx = {r["idx"]: r for r in recs}
    chosen = [by_idx[i] for i in DEMO_IDXS if i in by_idx]
    if len(chosen) < len(DEMO_IDXS):
        chosen = recs[:8]                                          # fallback: first eight cached
    chosen.sort(key=lambda r: r["idx"])
    examples = [to_example(r) for r in chosen]
    model = next((e["model"] for e in examples if e["model"]), "claude-sonnet-4-6")

    prompt = {
        "system": seg._SYS_AUDIT,
        "tool": seg._TOOL_AUDIT.get("name", "emit_layers"),
        "model": model,
        "types": ["theme", "detail", "connective", "person", "place", "org", "title", "ticker", "other"],
        "note": "two layers: a theme/detail/connective partition plus an overlapping entity overlay",
    }

    desc = (
        "Two overlapping layers on real run activations. LAYER 1 partitions the whole verbalizer "
        "output into theme / detail / connective claims and tiles it to 100% coverage. LAYER 2 "
        "overlays named entities (person / place / org / title / ticker / other) that may sit inside "
        "a Layer-1 claim, so the specific referent can be ablated apart from the claim around it. "
        "Every content claim also carries a role: describes (the AV's own description of the source), "
        "quotes (a literal string the AV says is in the source), or predicts (the AV's guess at the "
        "next token). These outputs are sampled from the qwen7b FineWeb run and are the exact "
        "claim-set feeding the atomic claim-ablation replication. Segmentation is LLM-only; the "
        "groundedness measurement (teacher-forced Δlogp) is a separate, judge-free step."
    )

    return {
        "id": "audit",
        "kind": "audit",
        "label": "4 · Two-layer audit",
        "desc": desc,
        "prompt": prompt,
        "stats": compute_stats(examples),
        "data": examples,
    }


def main() -> None:
    obj = build()
    OUT.write_text("window.AUDIT = " + json.dumps(obj, ensure_ascii=False) + ";\n")
    s = obj["stats"]
    print(f"wrote {OUT.name}: {len(obj['data'])} activations")
    print(f"  content median {s['content_median']} (total {s['content_total']}), "
          f"coverage {s['coverage_mean']*100:.0f}% / content {s['content_frac_mean']*100:.0f}%")
    print(f"  type mix {s['type_mix']}  roles {s['roles']}")
    print(f"  entities {s['entity_total']} {s['entity_subtypes']}  connectives {s['connective_total']}")
    print(f"  idxs {[e['idx'] for e in obj['data']]}")


if __name__ == "__main__":
    main()
