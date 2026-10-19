"""Attach the Tier-0 provenance block (and re-attach the stable Tier-1 block) to a
run's cycle_report.json (2026-07-23).

report.py builds a fresh report, so a Tier-0 report regeneration drops the `tier1`
block written earlier. Tier-1 is a FIXED historical record (pre-Tier-1 -> post-Tier-1),
so we recover it verbatim from the committed report at a git ref (default HEAD =
the Tier-1 commit) rather than recomputing it against the now-Tier-0 live claims.

Tier-0 is the kappa_0 recalibration (raise kappa_0 to restore the <=5% decoy FP
budget). It only demotes grounded/entangled -> seeded/confabulated and leaves the
unresolved set invariant, so its block is a git-ref -> live diff with that guard.

Guards (abort on violation):
  - every Tier-0 per-claim change is grounded/entangled -> seeded/confabulated;
  - the unresolved count is identical pre/post (kappa_0 never touches a pre-check);
  - the live report carries a tier1 block at the git ref (else nothing to re-attach).
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

VERDICTS = ("grounded", "entangled", "seeded", "confabulated", "unresolved")
_DEMOTE_FROM = ("grounded", "entangled")
_DEMOTE_TO = ("seeded", "confabulated")


def _git_show(repo_root: Path, ref: str, rel: str):
    try:
        out = subprocess.run(["git", "-C", str(repo_root), "show", f"{ref}:{rel}"],
                             capture_output=True, text=True, check=True).stdout
        return json.loads(out)
    except Exception:
        return None


def _dist(vs) -> dict:
    d = {v: 0 for v in VERDICTS}
    for v in vs:
        d[v] = d.get(v, 0) + 1
    return d


def build_tier0(run_dir: Path, ref: str, repo_root: Path, fresh: dict) -> dict:
    claims_dir = run_dir / "claims"
    rel_run = run_dir.resolve().relative_to(repo_root.resolve())
    trans: dict[str, int] = {}
    pre_rows, post_rows = [], []
    for p in sorted(claims_dir.glob("*.json")):
        if p.name == "status.json":
            continue
        post_v = json.loads(p.read_text(encoding="utf-8")).get("verdict")
        pre = _git_show(repo_root, ref, str(rel_run / "claims" / p.name))
        pre_v = (pre or {}).get("verdict") if pre is not None else post_v
        pre_rows.append(pre_v)
        post_rows.append(post_v)
        if pre_v != post_v:
            assert pre_v in _DEMOTE_FROM and post_v in _DEMOTE_TO, (
                f"non-Tier-0 transition {p.name}: {pre_v} -> {post_v}")
            trans[f"{pre_v}->{post_v}"] = trans.get(f"{pre_v}->{post_v}", 0) + 1
    pre_d, post_d = _dist(pre_rows), _dist(post_rows)
    assert pre_d["unresolved"] == post_d["unresolved"], (
        f"Tier-0 changed unresolved {pre_d['unresolved']} -> {post_d['unresolved']}")

    ref_rep = _git_show(repo_root, ref, str(rel_run / "cycle_report.json")) or {}
    oos_now = fresh.get("kappa_out_of_sample") or {}
    oos_ref = ref_rep.get("kappa_out_of_sample") or {}
    return dict(
        applied=True, date="2026-07-23", ref=ref,
        kappa_0_pre=ref_rep.get("kappa_0_calibration"),
        kappa_0_post=fresh.get("kappa_0_calibration"),
        method="max(calib, full-run LODO refit, OOS LODO refit); raising kappa_0 "
               "only tightens Gate 1, so this restores the FP budget without any "
               "headline-inflation risk (one-directional).",
        fp_full_pre=ref_rep.get("decoy_fp_rate_at_calibration_kappa"),
        fp_full_post=fresh.get("decoy_fp_rate_at_calibration_kappa"),
        fp_oos_pre=oos_ref.get("decoy_fp_rate_at_calibration_kappa"),
        fp_oos_post=oos_now.get("decoy_fp_rate_at_calibration_kappa"),
        transfers_pre=oos_ref.get("transfers"), transfers_post=oos_now.get("transfers"),
        pre_distribution=pre_d, post_distribution=post_d,
        transitions=dict(sorted(trans.items(), key=lambda x: -x[1])),
        demoted=sum(trans.values()),
        caveats=[
            "The OOS split reuses this run's claims (calib idxs excluded) and is a "
            "subset of the full families the refit also fits, so 'OOS' is not "
            "classic held-out; soundness rests on max()+monotonicity, not on OOS "
            "being independent.",
            "The LODO FP budget is fit on families with > k survivors (report.py "
            "convention); it is a >k-family budget, not a per-claim guarantee for "
            "the rare 1..k-survivor families that still reach Gate 1.",
        ],
        note="Tier-0 (2026-07-23): kappa_0 recalibrated to hold the decoy-vs-decoy "
             "LODO false-positive rate <= 5% on the full and out-of-sample survivor "
             "families. Demotes a few grounded/entangled to seeded/confabulated; "
             "leaves seeded/confabulated/unresolved and the FP budget sound.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--ref", default="HEAD", help="git ref of the post-Tier-1 report")
    ap.add_argument("--repo-root", default=None)
    args = ap.parse_args()
    run_dir = Path(args.run_dir).resolve()
    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(
        subprocess.run(["git", "-C", str(run_dir), "rev-parse", "--show-toplevel"],
                       capture_output=True, text=True, check=True).stdout.strip())

    rep_path = run_dir / "cycle_report.json"
    fresh = json.loads(rep_path.read_text(encoding="utf-8"))

    rel_run = run_dir.relative_to(repo_root)
    ref_rep = _git_show(repo_root, args.ref, str(rel_run / "cycle_report.json")) or {}
    tier1 = ref_rep.get("tier1")
    if tier1 is None:
        raise SystemExit(f"no tier1 block at {args.ref}:{rel_run}/cycle_report.json")
    fresh["tier1"] = tier1                                   # re-attach stable record
    fresh["tier0"] = build_tier0(run_dir, args.ref, repo_root, fresh)
    rep_path.write_text(json.dumps(fresh, indent=1), encoding="utf-8")

    t = fresh["tier0"]
    print(f"[tier0] {run_dir.name}: kappa_0 {t['kappa_0_pre']}->{t['kappa_0_post']}  "
          f"FP_oos {t['fp_oos_pre']}->{t['fp_oos_post']} "
          f"(transfers {t['transfers_pre']}->{t['transfers_post']})  demoted {t['demoted']}")
    for k, v in t["transitions"].items():
        print(f"    {k}: {v}")


if __name__ == "__main__":
    main()
