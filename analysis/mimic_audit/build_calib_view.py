"""Emit corpus/calib_view_<tag>.js (window.CALIB_<tag>) for a calibration+audit tab.

Usage: python build_calib_view.py v1_1     (the [OLD] tab)
       python build_calib_view.py v2        (the revised-prompt tab)

Reads, for the given tag:
  prompt_<tag>.txt, corpus/calib_result_<tag>.json, corpus/axis1_result_<tag>.json
Shared:
  corpus/style_fingerprint.json, and all calib_result_*.json for the comparison row.
Pairs carry the FULL prefix so the viewer can show the conversation context.
No API calls.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
C = HERE / "corpus"

NUMERIC = ["chars", "words", "blocks", "quoted_spans", "max_span_repeat",
           "dquotes", "ellipses", "likely_count", "hedge_count", "slash_count"]
FRAC = ["has_repeat", "quote_unbalanced", "ends_ellipsis", "ends_midthought", "has_slash_class"]
RUN_FILES = [("v1", "calib_result.json"), ("v1.1", "calib_result_v1_1.json"),
             ("v2", "calib_result_v2.json"), ("v2.1", "calib_result_v2_1.json"),
             ("v2.2", "calib_result_v2_2.json"), ("v2.3", "calib_result_v2_3.json")]

# versions charted in the block-2 anchoring trajectory (the v2-line iterations)
B2_VERSIONS = ["v2_1", "v2_2", "v2_3"]


def _blocks(t):
    return [b.strip() for b in re.split(r"\n\s*\n", (t or "").strip()) if b.strip()]


def _b2_quotes(t):
    b = _blocks(t)
    return None if len(b) < 2 else len(re.findall(r'"[^"]{2,}"', b[1]))


def _b2_rate(rows, key):
    """Fraction of block-2 paragraphs with >=1 verbatim quote, and avg quotes/block."""
    xs = [_b2_quotes(r[key]) for r in rows]
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None
    return sum(1 for x in xs if x >= 1) / len(xs), sum(xs) / len(xs)


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "v1_1"
    fp = load(C / "style_fingerprint.json")
    calres = load(C / f"calib_result_{tag}.json")
    ax_path = C / f"axis1_result_{tag}.json"
    ax = load(ax_path) if ax_path.exists() else None  # Axis-1 audit is optional per tag
    locked = (HERE / f"prompt_{tag}.txt").read_text(encoding="utf-8")

    # map each calibration row to the exact token the AV read (idx -> doc_id -> token)
    idx2doc = {t["idx"]: t["doc_id"] for t in load(C / f"calib_sample_{tag}.json")["tasks"]}
    tok_by_pos = {}
    with (C / "eval.jsonl").open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            tok_by_pos[(r["doc_id"], r["position"])] = r["token"]

    # fingerprint table (mimic fracs come from THIS tag's calib result)
    fingerprint = {
        "numeric": [{"sig": k, "p10": fp["overall"][k]["p10"], "p50": fp["overall"][k]["p50"],
                     "p90": fp["overall"][k]["p90"], "mimic": calres["mimic_fp"][k]["p50"]} for k in NUMERIC],
        "frac": [{"sig": k, "real": fp["overall"][k]["frac"], "mimic": calres["mimic_fp"][k]["frac"]} for k in FRAC],
        "n_reads": fp["n_reads"], "n_genres": fp["n_genres"],
    }

    # discriminator comparison across all runs that exist
    runs = []
    for name, fn in RUN_FILES:
        p = C / fn
        if p.exists():
            r = load(p)
            runs.append({"name": name, "acc": r["overall_acc"], "two_sided": r["two_sided"],
                         "is_this": (fn == f"calib_result_{tag}.json")})

    calib = {"acc": calres["overall_acc"], "two_sided": calres["two_sided"], "runs": runs}

    pairs = []
    for r in calres["rows"]:
        doc = idx2doc.get(r["idx"])
        pairs.append({
            "genre": r["theme"], "cosine": r["cosine"], "position": r["position"],
            "prefix": r.get("prefix", ""),
            "token": tok_by_pos.get((doc, r["position"]), ""),
            "nla": r["real_av"], "mimic": r["mimic"],
            "pick": r["pick"], "real_is": r["real_is"], "correct": r["correct"], "tell": r["tell"],
        })

    axis1, samples = None, []
    if ax:
        grid = ax["grid"]
        axis1 = {
            "total_claims": ax["total_claims"],
            "derivable": ax["derivable_fraction"], "strict": ax["derivable_strict"],
            "present": ax["present"], "supported": ax["supported"],
            "grid": {k: grid.get(k, 0) for k in ["pos_pos", "pos_neg", "neg_pos", "neg_neg"]},
            "by_category": ax["by_category"], "by_genre": ax["by_genre"],
            "n_payload": ax["n_payload_candidates"], "n_confirmed": ax["n_confirmed_payload"],
            "confirmed_kinds": ax["confirmed_kinds"],
            "substantive": [{"genre": c["theme"], "category": c["category"], "claim": c["claim"],
                             "reason": (c.get("reason") or "")} for c in ax["confirmed_payload"]
                            if c.get("kind") == "substantive"],
        }
        seen = set()
        for it in ax["per_item"]:
            has_cand = any(c["present_in_mimic"] == "no" and c["supported_by_text"] == "no" for c in it["claims"])
            if has_cand and it["theme"] not in seen:
                seen.add(it["theme"])
                samples.append({"genre": it["theme"], "cosine": it["cosine"],
                                "claims": [{"claim": c["text"], "category": c["category"],
                                            "supported": c["supported_by_text"], "in_mimic": c["present_in_mimic"]}
                                           for c in it["claims"]]})
            if len(samples) >= 16:
                break

    # block-2 anchoring trajectory: real-AV vs each v2-line mimic, plus the
    # total quote-count bands for THIS tag (over-band = caricature).
    real_rate, real_avg = _b2_rate(calres["rows"], "real_av")
    b2_versions = []
    for v in B2_VERSIONS:
        vp = C / f"calib_result_{v}.json"
        if vp.exists():
            rate, avg = _b2_rate(load(vp)["rows"], "mimic")
            b2_versions.append({"name": v, "rate": rate, "avg": avg})
    block2 = {
        "real_rate": real_rate, "real_avg": real_avg, "versions": b2_versions,
        "bands": {k: {"p10": fp["overall"][k]["p10"], "p50": fp["overall"][k]["p50"],
                      "p90": fp["overall"][k]["p90"], "mimic": calres["mimic_fp"][k]["p50"],
                      "verdict": calres.get("band_verdicts", {}).get(k, "?")}
                  for k in ["quoted_spans", "dquotes"]},
    }

    CALIB = {
        "meta": {"tag": tag, "locked_prompt": locked, "locked_chars": len(locked),
                 "goal": ("Measure how much of an NLA explanation is recoverable from the text prior "
                          "alone versus genuinely carried by the injected activation. A single fixed "
                          "prompt turns a strong model into a text-only 'mimic'; it is calibrated until "
                          "a cross-model discriminator cannot tell it from real AV output, then used to "
                          "audit each claim of the real reads.")},
        "fingerprint": fingerprint, "calib": calib, "pairs": pairs,
        "axis1": axis1, "samples": samples, "block2": block2,
    }
    var = "CALIB_" + tag
    out = C / f"calib_view_{tag}.js"
    out.write_text(f"window.{var} = " + json.dumps(CALIB, ensure_ascii=False) + ";\n", encoding="utf-8")
    n_subst = len(axis1["substantive"]) if axis1 else 0
    print(f"wrote {out.name} (window.{var}, {out.stat().st_size:,} bytes): "
          f"{len(pairs)} pairs, axis1={'yes' if axis1 else 'no'} ({n_subst} substantive), "
          f"{len(samples)} samples, block2 versions={[v['name'] for v in b2_versions]}, "
          f"runs={[r['name'] for r in runs]}")


if __name__ == "__main__":
    main()
