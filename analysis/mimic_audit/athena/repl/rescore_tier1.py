"""Rescore a cycle_run dir under the Tier-1 unresolved-reduction change (2026-07-22).

Tier-1 is TWO edits to cycle_consistency.py, both judge-free and GPU-free:
  (1) precheck_validity DECOUPLES: a flagged decoy (encoded / high-side D* outlier)
      is trimmed from the clean-placebo COUNT but kept in the Gate-1 null; the
      claim passes iff >= min_k CLEAN placebos remain (monotone-null lock).
  (2) classify_claim resolves a power-fail with a VALID family and M_BB <= 0 to
      confabulated (MDE floor = a valid family; emits only confabulated).

Verdicts are a pure function of the stored readings, so this rescore re-runs the
NOW-CORRECTED cc.precheck_validity + cc.classify_claim on the exact inputs
cycle_run.py fed them (the `scored` survivor subset), and rewrites only the
classify/validity-derived fields. The measured readings (M_BB, X_m, D*, supp*,
cos, npz/l0 checks, candidates) are untouched.

Faithfulness guards (abort on violation):
  - the re-run validity FLAGS (supp_viol_idx, dstar_outlier_idx) must equal the
    stored ones: the flagging math is unchanged, so a mismatch means the input
    reconstruction drifted.
  - CHURN LOCK: a claim resolved before Tier-1 must keep its verdict (Tier-1 only
    RECOVERS unresolved claims; it never re-judges a resolved one).
  - a power-reason unresolved claim may only become confabulated or stay
    unresolved (the power rule emits only confabulated; power itself is unchanged).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import cycle_consistency as cc

_MERGE = ("verdict", "gate", "reason", "grounding_estimate",
          "q95_dstar", "med_dstar", "gate1_bar", "med_x")


def _scored(row: dict) -> list[dict]:
    return [c for c in row.get("candidates", [])
            if c.get("supp_star") is not None and c.get("dstar") is not None]


def _rederive(row: dict, tau_val: float, min_k: int):
    """Re-run the corrected validity + classifier on the stored survivor readings.
    Returns (new_validity, classify_result) or (None, None) if the claim never
    went through classify (no readings)."""
    scored = _scored(row)
    if "power" not in row or "validity" not in row:
        return None, None
    supp_stars = [c["supp_star"] for c in scored]
    dstars = [c["dstar"] for c in scored]
    x_ms = [c["x_m"] for c in scored if c.get("x_m") is not None]
    new_val = cc.precheck_validity(supp_stars, tau_val, dstar=dstars, min_k=min_k)

    # faithfulness: the re-run flags must reproduce the stored ones exactly.
    old_val = row.get("validity") or {}
    for key in ("supp_viol_idx", "dstar_outlier_idx"):
        if sorted(old_val.get(key, []) or []) != sorted(new_val.get(key, []) or []):
            raise AssertionError(
                f"validity flag drift on claim {row.get('idx')}_{row.get('claim_i')} "
                f"[{key}]: stored {old_val.get(key)} vs re-run {new_val.get(key)}")

    m_bb = row.get("m_bb")
    res = cc.classify_claim(row.get("power") or {"passed": False}, new_val,
                            m_bb if m_bb is not None else float("nan"),
                            dstars, x_ms, row.get("m_ib_max"),
                            row.get("kappa_0") or 0.0)
    return new_val, res


def rescore_dir(run_dir: Path, tau_val: float, min_k: int, apply: bool) -> dict:
    files = sorted(p for p in (run_dir / "claims").glob("*.json")
                   if p.name != "status.json")
    trans: dict[str, int] = {}
    changed = 0
    for p in files:
        row = json.loads(p.read_text(encoding="utf-8"))
        new_val, res = _rederive(row, tau_val, min_k)
        if res is None:
            continue
        old_v = row.get("verdict")
        new_v = res["verdict"]

        if new_v != old_v:
            # CHURN LOCK: a resolved claim must not be re-judged.
            assert old_v in (None, "unresolved"), (
                f"CHURN on resolved claim {p.name}: {old_v} -> {new_v}")
            # power-reason claims may only reach confabulated.
            if (row.get("reason") == "power") and new_v not in ("unresolved", "confabulated"):
                raise AssertionError(
                    f"ILLEGAL power-fail transition {p.name}: -> {new_v}")
            trans[f"{old_v or 'None'}->{new_v}"] = trans.get(f"{old_v or 'None'}->{new_v}", 0) + 1
            changed += 1

        if apply:
            for kk in _MERGE:
                if kk in res:
                    row[kk] = res[kk]
                elif kk in row and kk not in res:
                    row.pop(kk, None)          # e.g. med_x absent on non-Gate-3 paths
            row["validity"] = new_val          # carries n_clean / dropped_idx
            tmp = p.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(row, indent=1), encoding="utf-8")
            tmp.replace(p)
    return dict(n_files=len(files), changed=changed, transitions=trans)


def _tau_val(calib_path: Path | None) -> float:
    if calib_path and calib_path.exists():
        return float(json.loads(calib_path.read_text())["tau_val"])
    raise SystemExit("need --calib calibration_signed.json for tau_val")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--calib", required=True,
                    help="calibration_signed.json (for tau_val)")
    ap.add_argument("--min-k", type=int, default=5)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    tau = _tau_val(Path(args.calib))
    res = rescore_dir(Path(args.run_dir), tau, args.min_k, args.apply)
    mode = "APPLIED" if args.apply else "DRY-RUN"
    print(f"[{mode}] {args.run_dir}  (tau_val={tau:.2f}, min_k={args.min_k})")
    print(f"  files={res['n_files']} changed={res['changed']}")
    for k, v in sorted(res["transitions"].items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
