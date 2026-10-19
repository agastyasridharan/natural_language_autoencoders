"""Attach a Tier-3 (band-widening variant) before/after block to each committed
P90 cycle_report.json, so the studio cycle tab can render the comparison panel.

Tier-3 is a documented VARIANT: the committed P90 runs remain primary, and this
block is comparison metadata only (it never changes a P90 verdict). The panel reads
report["tier3"]; this script writes it from the consolidated tier3/tier3_summary.json
produced by tier3_consolidate.py on the box.

Idempotent. Run from the athena dir (or pass --athena).

Usage:
  python repl/tier3_provenance.py [--athena <dir>] [--summary <path>] [--apply]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# model key -> committed P90 report (relative to the athena dir)
REPORTS = {
    "qwen7b":   "cycle_runs/qwen7b-d4g24-full/qwen7b-cp6-fineweb-atoms/cycle_report.json",
    "gemma12b": "cycle_runs/gemma12b-d4g24/gemma12b-cp7-fineweb-atoms/cycle_report.json",
    "gemma27b": "cycle_runs/gemma27b-d4g24/gemma27b-g4-fineweb-atoms/cycle_report.json",
    "llama70b": "cycle_runs/llama70b-d4g24/llama70b-l4-fineweb-atoms/cycle_report.json",
}
# qwen (N=1196) passed the smell screen; the three N~298 models flagged but were
# cleared by the mechanistic spot-check (see tier3/TIER3_REPORT.md).
SMELL_FLAGGED = {"gemma12b", "gemma27b", "llama70b"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--athena", default=".")
    ap.add_argument("--summary", default="tier3/tier3_summary.json")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    athena = Path(args.athena)
    summary = json.loads((athena / args.summary).read_text(encoding="utf-8"))
    models = summary["models"]

    for key, rel in REPORTS.items():
        m = models[key]
        block = {
            "applied": True,
            "variant": True,             # P90 stays primary; comparison only
            "band": "P90->P95",
            "n_claims": m["N"],
            "pre_distribution": m["dist_base"],
            "post_distribution": m["dist_tier3"],
            "unresolved_pct_base": m["unresolved_pct_base"],
            "unresolved_pct_tier3": m["unresolved_pct_tier3"],
            "recovered": m["recovered"],
            "lost": m["lost"],
            "recovered_dist": m["recovered_dist"],
            "fp_full": m["fp_full"],
            "fp_oos": m["fp_oos"],
            "fp_transfers": m["fp_transfers"],
            "mbb_drift": m["mbb_drift"],
            "kappa_0": m["kappa_0"],
            "smell_flagged": key in SMELL_FLAGGED,
            "smell_cleared": True,       # every model cleared by the spot-check
            "ge_recoveries": m["ge_recoveries"],
            "ge_rec_base_power_true": m["ge_rec_base_power_true"],
            "ge_rec_nodata": m["ge_rec_nodata"],
        }
        path = athena / rel
        rep = json.loads(path.read_text(encoding="utf-8"))
        had = "tier3" in rep
        rep["tier3"] = block
        u0, u1 = block["unresolved_pct_base"], block["unresolved_pct_tier3"]
        print(f"{key:9s} {'update' if had else 'add   '} tier3: "
              f"unresolved {u0}->{u1}%  recovered {block['recovered']}  "
              f"FP {block['fp_full']}  drift {block['mbb_drift']}  "
              f"smell={'flagged/cleared' if block['smell_flagged'] else 'ok'}")
        if args.apply:
            path.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    if not args.apply:
        print("\n(dry run - pass --apply to write)")


if __name__ == "__main__":
    main()
