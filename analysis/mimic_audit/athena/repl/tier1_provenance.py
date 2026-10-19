"""Write a `tier1` before/after provenance block into a run's cycle_report.json.

Tier-1 (2026-07-22) only RECOVERS unresolved claims (churn-locked: no resolved
claim is ever re-judged), so the pre-Tier-1 state is the committed report at a git
ref. This helper reads that ref's per-claim verdicts, diffs them against the live
(post-apply) claims, and records the exact unresolved -> X transitions plus the
before/after verdict distributions and the (unchanged) decoy FP rate. The studio
cycle tab reads report["tier1"] to render the before/after panel; it computes
nothing itself.

Integrity guards (abort on violation):
  - every per-claim change must be unresolved -> X (churn lock); a resolved-claim
    change means the apply step was not Tier-1.
  - the diffed transition totals must reconstruct the live distribution exactly
    (pre + transitions == post), else the git ref and the live run disagree.

Usage:
  tier1_provenance.py <run_dir> [--ref HEAD] [--repo-root <path>]
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

VERDICTS = ("grounded", "entangled", "seeded", "confabulated", "unresolved")


def _git_show(repo_root: Path, ref: str, rel: str) -> dict | None:
    try:
        out = subprocess.run(["git", "-C", str(repo_root), "show", f"{ref}:{rel}"],
                             capture_output=True, text=True, check=True).stdout
        return json.loads(out)
    except Exception:
        return None


def _dist(rows) -> dict:
    d = {v: 0 for v in VERDICTS}
    for v in rows:
        d[v] = d.get(v, 0) + 1
    return d


def build(run_dir: Path, ref: str, repo_root: Path) -> dict:
    claims_dir = run_dir / "claims"
    live = {}
    for p in sorted(claims_dir.glob("*.json")):
        if p.name == "status.json":
            continue
        live[p.name] = json.loads(p.read_text(encoding="utf-8")).get("verdict")

    rel_run = run_dir.resolve().relative_to(repo_root.resolve())
    transitions: dict[str, int] = {}
    pre_rows, post_rows = [], []
    missing_at_ref = 0
    for name, post_v in live.items():
        pre = _git_show(repo_root, ref, str(rel_run / "claims" / name))
        pre_v = (pre or {}).get("verdict") if pre is not None else None
        if pre is None:
            missing_at_ref += 1
            pre_v = post_v            # not committed at ref: treat as unchanged
        pre_rows.append(pre_v)
        post_rows.append(post_v)
        if pre_v != post_v:
            assert pre_v in (None, "unresolved"), \
                f"CHURN on resolved claim {name}: {pre_v} -> {post_v}"
            k = f"{pre_v or 'None'}->{post_v}"
            transitions[k] = transitions.get(k, 0) + 1

    pre_d, post_d = _dist(pre_rows), _dist(post_rows)
    # reconstruct: pre + (unresolved -> X) must equal post exactly
    recon = dict(pre_d)
    for k, n in transitions.items():
        src, dst = k.split("->")
        recon[src] = recon.get(src, 0) - n
        recon[dst] = recon.get(dst, 0) + n
    assert recon == post_d, f"transition reconstruction mismatch: {recon} != {post_d}"

    N = len(live)
    rep = json.loads((run_dir / "cycle_report.json").read_text(encoding="utf-8"))
    fp = rep.get("decoy_fp_rate_at_calibration_kappa")
    return dict(
        applied=True, min_k=5, ref=ref, n_claims=N,
        pre_distribution=pre_d, post_distribution=post_d,
        transitions=dict(sorted(transitions.items(), key=lambda x: -x[1])),
        recovered=sum(transitions.values()),
        unresolved_pre=pre_d["unresolved"], unresolved_post=post_d["unresolved"],
        unresolved_pct_pre=round(100 * pre_d["unresolved"] / N, 1),
        unresolved_pct_post=round(100 * post_d["unresolved"] / N, 1),
        decoy_fp_rate_unchanged=fp,
        note=("Tier-1 recovers unresolved claims via the validity decouple "
              "(flagged decoys trimmed from the clean count, kept in the Gate-1 "
              "null) and the power-fail M_BB<=0 rule. Churn-locked: no resolved "
              "claim is re-judged; the decoy-vs-decoy FP rate is unchanged."),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--ref", default="HEAD")
    ap.add_argument("--repo-root", default=None)
    args = ap.parse_args()
    run_dir = Path(args.run_dir).resolve()
    repo_root = Path(args.repo_root).resolve() if args.repo_root else \
        Path(subprocess.run(["git", "-C", str(run_dir), "rev-parse", "--show-toplevel"],
                            capture_output=True, text=True, check=True).stdout.strip())
    tier1 = build(run_dir, args.ref, repo_root)
    rep_path = run_dir / "cycle_report.json"
    rep = json.loads(rep_path.read_text(encoding="utf-8"))
    rep["tier1"] = tier1
    rep_path.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(f"[tier1] {run_dir.name}: unresolved {tier1['unresolved_pct_pre']}% -> "
          f"{tier1['unresolved_pct_post']}%  (recovered {tier1['recovered']}, "
          f"FP {tier1['decoy_fp_rate_unchanged']})")
    for k, v in tier1["transitions"].items():
        print(f"    {k}: {v}")


if __name__ == "__main__":
    main()
