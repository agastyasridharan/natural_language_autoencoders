"""Rescore a cycle_run output dir under the corrected Gate-3 rule (2026-07-22).

The correction (user): a non-positive own-effect (M_BB <= 0) is confabulated
outright, so 'seeded' now requires a POSITIVE own-effect. Scoped to resolved
claims: the rule is applied inside classify_claim AFTER the power/validity
pre-checks, so broken-control claims stay unresolved.

Verdicts are a pure function of the stored readings (cycle_consistency.
classify_claim), so this rescore reconstructs the EXACT inputs cycle_run.py fed
classify_claim (the 'scored' candidate subset), re-runs the now-corrected
classifier, and rewrites only the classify-derived fields. The measured readings
(M_BB, X_m, D*, supp, cos, npz/l0 checks, candidates) are untouched.

Safety: every claim with M_BB > 0 (or non-finite) MUST keep its stored verdict
(the rule does not touch that path); any deviation there means the input
reconstruction or the edit is wrong, and the script aborts. The only legal
changes are, among pre-check passers with M_BB <= 0:
  seeded -> confabulated
  unresolved(no finite X_m at Gate 3) -> confabulated
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import cycle_consistency as cc

# fields classify_claim owns in the row (mirror of cycle_run.py's merge set)
_MERGE = ("verdict", "gate", "reason", "grounding_estimate",
          "q95_dstar", "med_dstar", "gate1_bar", "med_x")


def _inputs(row: dict):
    """Reconstruct the exact (power, validity, m_bb, dstars, x_ms, m_ib_max,
    kappa_0) cycle_run.py passed classify_claim for this claim."""
    cands = row.get("candidates", [])
    scored = [c for c in cands
              if c.get("supp_star") is not None and c.get("dstar") is not None]
    dstars = [c["dstar"] for c in scored]
    x_ms = [c["x_m"] for c in scored if c.get("x_m") is not None]
    return (row.get("power") or {"passed": False},
            row.get("validity") or {"passed": False},
            row.get("m_bb", float("nan")), dstars, x_ms,
            row.get("m_ib_max"), row.get("kappa_0") or 0.0)


def rescore_dir(run_dir: Path, apply: bool) -> dict:
    claims_dir = run_dir / "claims"
    files = sorted(p for p in claims_dir.glob("*.json") if p.name != "status.json")
    trans: dict[str, int] = {}
    changed = 0
    for p in files:
        row = json.loads(p.read_text(encoding="utf-8"))
        if "power" not in row or "validity" not in row:
            continue                                    # never went through classify
        power, validity, m_bb, dstars, x_ms, m_ib_max, kappa_0 = _inputs(row)
        new = cc.classify_claim(power, validity,
                                m_bb if m_bb is not None else float("nan"),
                                dstars, x_ms, m_ib_max, kappa_0)
        old_v = row.get("verdict")
        new_v = new["verdict"]

        # SAFETY: the rule cannot touch M_BB>0 or non-finite claims.
        mb_ok = (m_bb is not None and np.isfinite(m_bb))
        if not (mb_ok and m_bb <= 0.0):
            assert new_v == old_v, (
                f"UNEXPECTED change on M_BB>0/non-finite claim {p.name}: "
                f"{old_v} -> {new_v} (m_bb={m_bb})")
            # this path is never rewritten (no --apply write below), so the row
            # stays byte-identical to the original run: faithful by construction.
            continue

        # M_BB <= 0 and pre-checks passed: legal transitions only.
        if new_v != old_v:
            legal = (new_v == "confabulated" and old_v in ("seeded", "unresolved"))
            assert legal, (f"ILLEGAL transition {p.name}: {old_v} -> {new_v} "
                           f"(m_bb={m_bb})")
            trans[f"{old_v}->{new_v}"] = trans.get(f"{old_v}->{new_v}", 0) + 1
            changed += 1
        if apply:
            for kk in _MERGE:
                if kk in new:
                    row[kk] = new[kk]
                elif kk in row and kk not in new:
                    # med_x is absent from the new dict on the M_BB<=0 early
                    # return; drop the stale value so the row matches classify.
                    row.pop(kk, None)
            tmp = p.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(row, indent=1), encoding="utf-8")
            tmp.replace(p)
    return dict(n_files=len(files), changed=changed, transitions=trans)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="cycle_runs/<tag>/<run> with claims/")
    ap.add_argument("--apply", action="store_true",
                    help="write the corrected verdicts (default: dry-run)")
    args = ap.parse_args()
    res = rescore_dir(Path(args.run_dir), args.apply)
    mode = "APPLIED" if args.apply else "DRY-RUN"
    print(f"[{mode}] {args.run_dir}")
    print(f"  files={res['n_files']} changed={res['changed']}")
    for k, v in sorted(res["transitions"].items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
