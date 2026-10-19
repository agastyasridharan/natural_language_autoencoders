"""Aggregate the Axis-1 content audit from corpus/axis1_out/.

Headline: what fraction of real AV claims is derivable from the text-only mimic
(overall, per genre, per category). And the payload candidates: claims the real
AV makes that are neither in the text nor in the mimic, cross-checked by the
Sonnet verifier and split into confab_specific vs substantive.

Writes AXIS1_REPORT.md and corpus/axis1_result.json. No API calls.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
SAMPLE = HERE / "corpus" / "axis1_sample.json"
OUTDIR = HERE / "corpus" / "axis1_out"
REPORT = HERE / "AXIS1_REPORT.md"
RESULT = HERE / "corpus" / "axis1_result.json"

CATS = ["classification", "attribution", "local_structure", "continuation", "other"]


def load_json(p):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def norm(s):
    return " ".join((s or "").lower().split())


def deriv_score(present):
    return {"yes": 1.0, "partial": 0.5, "no": 0.0}.get(present, 0.0)


def main():
    items = {it["idx"]: it for it in json.loads(SAMPLE.read_text(encoding="utf-8"))["items"]}

    per_item = []
    missing_claims = missing_verify = 0
    for idx, it in sorted(items.items()):
        cj = load_json(OUTDIR / f"claims_{idx}.json")
        if not cj or "claims" not in cj:
            missing_claims += 1
            continue
        claims = cj["claims"]
        vj = load_json(OUTDIR / f"verify_{idx}.json")
        vmap = {}
        if vj and "verdicts" in vj:
            for v in vj["verdicts"]:
                vmap[norm(v.get("claim"))] = v
        elif any(c["present_in_mimic"] == "no" and c["supported_by_text"] == "no" for c in claims):
            missing_verify += 1
        per_item.append({"idx": idx, "theme": it["theme"], "kind": it["kind"],
                         "cosine": it["cosine"], "claims": claims, "verify": vmap})

    # ---- aggregates ----
    tot = 0
    pres = defaultdict(int)
    supp = defaultdict(int)
    cat_tot = defaultdict(int)
    cat_deriv = defaultdict(float)
    grid = defaultdict(int)            # (supported_bucket, present_bucket) -> n
    by_genre_tot = defaultdict(int)
    by_genre_deriv = defaultdict(float)
    payload_candidates = []            # present=no & support=no
    confirmed_payload = []             # + verifier says really_not_* both true

    def bucket(x):
        return "pos" if x in ("yes", "partial") else "neg"

    for r in per_item:
        for c in r["claims"]:
            tot += 1
            pres[c["present_in_mimic"]] += 1
            supp[c["supported_by_text"]] += 1
            cat = c["category"] if c["category"] in CATS else "other"
            cat_tot[cat] += 1
            cat_deriv[cat] += deriv_score(c["present_in_mimic"])
            grid[(bucket(c["supported_by_text"]), bucket(c["present_in_mimic"]))] += 1
            by_genre_tot[r["theme"]] += 1
            by_genre_deriv[r["theme"]] += deriv_score(c["present_in_mimic"])
            if c["present_in_mimic"] == "no" and c["supported_by_text"] == "no":
                v = r["verify"].get(norm(c["text"]))
                rec = {"idx": r["idx"], "theme": r["theme"], "category": cat,
                       "claim": c["text"],
                       "verdict": v if v else None}
                payload_candidates.append(rec)
                if v and v.get("really_not_in_mimic") and v.get("really_not_in_text"):
                    confirmed_payload.append({**rec, "kind": v.get("kind"), "reason": v.get("reason")})

    derivable = sum(deriv_score(c["present_in_mimic"]) for r in per_item for c in r["claims"]) / (tot or 1)
    strict = pres["yes"] / (tot or 1)

    conf_kinds = defaultdict(int)
    for c in confirmed_payload:
        conf_kinds[c.get("kind", "other")] += 1

    # ---- report ----
    L = []
    L.append("# Axis 1: how much of each NLA explanation is recoverable from text alone")
    L.append("")
    L.append(f"Decomposed **{tot} atomic claims** from {len(per_item)} held-out real AV reads "
             f"(locked prompt v1.1 as the text-only mimic). "
             f"{missing_claims} pairs missing claims, {missing_verify} pairs missing verifier output.")
    L.append("")
    L.append("## Headline")
    L.append("")
    L.append(f"- **Derivable-by-mimic: {derivable:.1%}** of real AV claims (yes=1, partial=0.5). "
             f"Strict (yes only): {strict:.1%}.")
    L.append(f"- present_in_mimic: yes {pres['yes']}, partial {pres['partial']}, no {pres['no']}.")
    L.append(f"- supported_by_text: yes {supp['yes']}, partial {supp['partial']}, no {supp['no']}.")
    L.append("")
    L.append("## The 2x2 that defines the payload")
    L.append("")
    L.append("(pos = yes/partial, neg = no)")
    L.append("")
    L.append("| | in mimic (pos) | NOT in mimic (neg) |")
    L.append("|---|---|---|")
    L.append(f"| **in text (pos)** | {grid[('pos','pos')]} (trivially recoverable) | {grid[('pos','neg')]} (mimic recall gap) |")
    L.append(f"| **NOT in text (neg)** | {grid[('neg','pos')]} (shared prior/confab) | **{grid[('neg','neg')]} (PAYLOAD CANDIDATES)** |")
    L.append("")
    L.append("Only the bottom-right cell can contain genuine activation payload. Everything "
             "else is recoverable from the text prior or is a shared guess.")
    L.append("")
    L.append("## Derivability by claim category")
    L.append("")
    L.append("| category | n | derivable |")
    L.append("|---|---|---|")
    for cat in CATS:
        if cat_tot[cat]:
            L.append(f"| {cat} | {cat_tot[cat]} | {cat_deriv[cat]/cat_tot[cat]:.0%} |")
    L.append("")
    L.append("## Derivability by genre")
    L.append("")
    L.append("| genre | claims | derivable |")
    L.append("|---|---|---|")
    for g in sorted(by_genre_tot):
        L.append(f"| {g} | {by_genre_tot[g]} | {by_genre_deriv[g]/by_genre_tot[g]:.0%} |")
    L.append("")
    L.append("## Payload candidates (not in text AND not in mimic)")
    L.append("")
    L.append(f"- Raw candidates: **{len(payload_candidates)}** ({len(payload_candidates)/(tot or 1):.1%} of all claims).")
    L.append(f"- Confirmed by Sonnet verifier (really not in mimic AND really not in text): "
             f"**{len(confirmed_payload)}**.")
    L.append(f"  - confab_specific (fabricated detail, likely hallucination): {conf_kinds['confab_specific']}")
    L.append(f"  - substantive (meaningful non-text claim, the real payload suspects): {conf_kinds['substantive']}")
    L.append(f"  - other: {conf_kinds['other']}")
    L.append("")
    subs = [c for c in confirmed_payload if c.get("kind") == "substantive"]
    if subs:
        L.append("### Substantive payload suspects (need Axis-2 grounding to confirm)")
        L.append("")
        for c in subs[:40]:
            L.append(f"- *{c['theme']}* [{c['category']}]: \"{c['claim']}\" - {c.get('reason','')}")
        L.append("")
    REPORT.write_text("\n".join(L), encoding="utf-8")

    result = {
        "n_items": len(per_item), "total_claims": tot,
        "derivable_fraction": derivable, "derivable_strict": strict,
        "present": dict(pres), "supported": dict(supp),
        "grid": {f"{k[0]}_{k[1]}": v for k, v in grid.items()},
        "by_category": {c: {"n": cat_tot[c], "derivable": (cat_deriv[c]/cat_tot[c] if cat_tot[c] else None)} for c in CATS},
        "by_genre": {g: {"n": by_genre_tot[g], "derivable": by_genre_deriv[g]/by_genre_tot[g]} for g in sorted(by_genre_tot)},
        "n_payload_candidates": len(payload_candidates),
        "n_confirmed_payload": len(confirmed_payload),
        "confirmed_kinds": dict(conf_kinds),
        "confirmed_payload": confirmed_payload,
        "per_item": per_item,
    }
    RESULT.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {REPORT.name} and {RESULT.name}")
    print(f"claims={tot} derivable={derivable:.1%} strict={strict:.1%} "
          f"payload_raw={len(payload_candidates)} confirmed={len(confirmed_payload)} "
          f"(substantive={conf_kinds['substantive']}, confab={conf_kinds['confab_specific']})")
    print(f"GRID neg_neg(payload)={grid[('neg','neg')]} neg_pos={grid[('neg','pos')]} "
          f"pos_neg={grid[('pos','neg')]} pos_pos={grid[('pos','pos')]}")


if __name__ == "__main__":
    main()
