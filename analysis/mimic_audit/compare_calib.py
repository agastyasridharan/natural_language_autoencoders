"""Compare two calibration runs (v1 vs v1.1) on the identical held-out tasks and
recommend which single prompt to lock.

Lock criteria, in order:
  1. no numeric signature OVER the real band (caricature is disqualifying);
  2. discriminator two-sided |acc - 0.5| closest to 0 (indistinguishable);
  3. tie-break: smaller total fraction-signature drift (tighter style match).

Usage: python compare_calib.py            (compares v1 unsuffixed vs v1_1)
       python compare_calib.py v1 v1_1
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from fingerprint import NUMERIC, FRAC

HERE = Path(__file__).resolve().parent
OUT = HERE / "COMPARE_v1_vs_v1_1.md"


def load(tag):
    name = "calib_result.json" if tag == "v1" else f"calib_result_{tag}.json"
    return json.loads((HERE / "corpus" / name).read_text(encoding="utf-8"))


def frac_drift(res):
    real = res["real_fp"]
    mim = res["mimic_fp"]
    return sum(abs(mim[k]["frac"] - real[k]["frac"]) for k in FRAC)


def main():
    a_tag = sys.argv[1] if len(sys.argv) > 2 else "v1"
    b_tag = sys.argv[2] if len(sys.argv) > 2 else "v1_1"
    A, B = load(a_tag), load(b_tag)
    real = A["real_fp"]

    L = []
    L.append(f"# Lock decision: {a_tag} vs {b_tag}")
    L.append("")
    L.append("Identical 128 held-out tasks. Discriminator = Sonnet 4.6, style only. "
             "Target = 50% (two-sided |acc-0.5| = 0).")
    L.append("")
    L.append("## Discriminator")
    L.append("")
    L.append(f"| | {a_tag} | {b_tag} |")
    L.append("|---|---|---|")
    L.append(f"| disc acc | {A['overall_acc']:.1%} | {B['overall_acc']:.1%} |")
    L.append(f"| two-sided \\|acc-0.5\\| | {A['two_sided']:.3f} | {B['two_sided']:.3f} |")
    L.append("")
    L.append("## Style fraction signatures (mimic vs real; closer to real = better)")
    L.append("")
    L.append(f"| signature | real | {a_tag} | {b_tag} |")
    L.append("|---|---|---|---|")
    for k in FRAC:
        L.append(f"| {k} | {real[k]['frac']:.0%} | {A['mimic_fp'][k]['frac']:.0%} "
                 f"| {B['mimic_fp'][k]['frac']:.0%} |")
    da, db = frac_drift(A), frac_drift(B)
    L.append(f"| **total drift** | 0 | **{da:.2f}** | **{db:.2f}** |")
    L.append("")
    L.append("## Numeric band check (OVER = caricature, disqualifying)")
    L.append("")
    over_a = [k for k in NUMERIC if A["band_verdicts"][k] == "OVER"]
    over_b = [k for k in NUMERIC if B["band_verdicts"][k] == "OVER"]
    L.append(f"- {a_tag} OVER: {over_a or 'none'}")
    L.append(f"- {b_tag} OVER: {over_b or 'none'}")
    L.append("")

    # decision
    def disq(over):
        return len(over) > 0
    cand = []
    if not disq(over_a):
        cand.append((a_tag, A["two_sided"], da))
    if not disq(over_b):
        cand.append((b_tag, B["two_sided"], db))
    if not cand:
        winner = "neither (both caricature) - revisit"
    else:
        cand.sort(key=lambda x: (round(x[1], 3), x[2]))
        winner = cand[0][0]
    L.append("## Recommendation")
    L.append("")
    L.append(f"**Lock: {winner}** (lowest two-sided discriminator distance with no "
             f"OVER signatures; fraction drift as tie-break).")
    L.append("")
    OUT.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
