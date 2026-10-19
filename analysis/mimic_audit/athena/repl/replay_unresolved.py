"""replay_unresolved.py - CPU replay of Tier-1 unresolved-reduction variants.

JUDGE-FREE + NO GPU: re-derives verdicts from STORED readings only (the same
per-claim M_BB / D* / X_m / supp* the run already wrote), reusing
cycle_consistency.classify_claim so the gate arithmetic is byte-identical to the
published run. This is the arithmetic-replay A/B the design panel called for: it
measures what each variant would do to the unresolved rate BEFORE spending any
GPU, and self-adjudicates the contested D*-outlier arm via a false-positive
smell test.

Variants (composable via opts):
  encoded : decouple the validity ENCODED-decoy arm. A decoy the activation
            actually supports (|supp*|>tau_val, flagged in supp_viol_idx) is not
            a placebo; drop it from the min_k COUNT but keep ALL survivor D* in
            the Gate-1 null. Recover the claim iff >=min_k clean survivors remain.
  outlier : additionally decouple the D*-outlier arm (dstar_outlier_idx). This is
            the CONTESTED slice (the never-guess skeptic wants it fail-closed);
            it is a toggle so the smell test can admit or reject it on the data.
  power   : a power-fail with a VALID decoy family (validity passes) and a finite
            non-positive own-effect M_BB<=0 -> confabulated. The MDE sensitivity
            floor is 'validity passes': a >=min_k clean placebo family is what
            gives the measurement the power to have detected a real effect, so a
            clear M_BB<=0 against it is a null, not an underpowered guess. Emits
            ONLY confabulated (never grounded/seeded), so zero false-grounding.
  naive   : DIAGNOSTIC ONLY. Drops the dropped decoys from the Gate-1 null too,
            lowering Q95(D*). Quantifies the false-grounded verdicts the
            monotone-null lock prevents. NEVER ship this.

Guards baked in (all non-naive variants):
  - monotone-null lock : Gate-1 Q95(D*) is always over the FULL survivor set,
    so a recovered claim clears a bar at least as strict as fail-closed would.
  - X_m median over the FULL survivor family (never the trimmed subset).
  - the power supp* null is untouched; we only flip validity.passed.

Report (on the in-sample 25 / out-of-sample rest split, per report.py):
  unresolved drop by reason, churn on already-resolved claims (MUST be 0 for
  every non-naive variant), and the false-positive smell test (verdict mix of
  the NEWLY-resolved claims vs the baseline-resolved population).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import cycle_consistency as cc

VERDICTS = ("grounded", "entangled", "seeded", "confabulated", "unresolved")


# ── input reconstruction (identical filter to cycle_run.py's `scored`) ────────

def _scored(row: dict) -> list[dict]:
    """The survivor candidates classify saw, in order: supp_star AND dstar set.
    dstar_outlier_idx / supp_viol_idx index into THIS list (they were computed
    from arrays built off it), so order and membership must match exactly."""
    return [c for c in row.get("candidates", [])
            if c.get("supp_star") is not None and c.get("dstar") is not None]


def _finite(xs):
    a = np.asarray([x for x in xs if x is not None], dtype=np.float64)
    return a[np.isfinite(a)]


def baseline_verdict(row: dict) -> dict:
    """Re-run classify_claim on the stored inputs unchanged. Reproduces the run's
    verdict (verified against verdict_stored); the control arm of every A/B."""
    scored = _scored(row)
    dstars = [c["dstar"] for c in scored]
    x_ms = [c["x_m"] for c in scored if c.get("x_m") is not None]
    return cc.classify_claim(row.get("power") or {"passed": False},
                             row.get("validity") or {"passed": False},
                             row.get("m_bb", float("nan")) if row.get("m_bb") is not None else float("nan"),
                             dstars, x_ms, row.get("m_ib_max"),
                             row.get("kappa_0") or 0.0)


def variant_verdict(row: dict, *, encoded=False, outlier=False, power=False,
                    naive=False, min_k=5) -> dict:
    """Verdict under the chosen Tier-1 variant, from stored readings only."""
    scored = _scored(row)
    n = len(scored)
    dstars_full = [c["dstar"] for c in scored]
    x_full = [c["x_m"] for c in scored if c.get("x_m") is not None]   # X_m guard: FULL family
    pw = row.get("power") or {"passed": False}
    val = row.get("validity") or {"passed": False}
    m_bb = row.get("m_bb")
    m_bb = float(m_bb) if m_bb is not None else float("nan")
    m_ib_max = row.get("m_ib_max")
    kappa_0 = row.get("kappa_0") or 0.0

    # ── power-fail branch ────────────────────────────────────────────────────
    if not pw.get("passed"):
        if (power and val.get("passed") and np.isfinite(m_bb) and m_bb <= 0.0
                and _finite(dstars_full).size >= min_k):
            return {"verdict": "confabulated", "gate": 3,
                    "reason": "power-fail non-positive own-effect "
                              "(MDE floor: valid >=min_k decoy family)",
                    "new_rule": "power_mbb"}
        reason = "no-decoy-family" if pw.get("n_family") == 0 else "power"
        return {"verdict": "unresolved", "reason": reason}

    # ── power passes: decouple validity, keep the null full (monotone lock) ───
    sv = [i for i in (val.get("supp_viol_idx") or []) if i < n]
    do = [i for i in (val.get("dstar_outlier_idx") or []) if i < n]
    dropped = set()
    if encoded:
        dropped |= set(sv)
    if outlier:
        dropped |= set(do)

    val2 = dict(val)
    if not val.get("passed"):
        n_clean = n - len(dropped)
        remaining = [i for i in (sv + do) if i not in dropped]
        if n_clean >= min_k and not remaining:
            val2["passed"] = True            # recovered on the clean subset

    if naive:                                # DIAGNOSTIC: bias the null downward
        dstars_use = [scored[i]["dstar"] for i in range(n) if i not in dropped]
    else:                                    # monotone-null lock: full survivor set
        dstars_use = dstars_full

    out = cc.classify_claim(pw, val2, m_bb, dstars_use, x_full, m_ib_max, kappa_0)
    return out


# ── run-level replay + reporting ─────────────────────────────────────────────

def _load_claims(run_dir: Path) -> list[dict]:
    out = []
    for p in sorted((run_dir / "claims").glob("*.json")):
        if p.name == "status.json":
            continue
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            pass
    return out


def _in_sample_idxs(calib_claims_dir: Path | None) -> set[int]:
    """Activation idxs that were IN the calibration split (report.py convention).
    Verdicts on these are in-sample; the rest are the honest OOS test."""
    idxs: set[int] = set()
    if calib_claims_dir and calib_claims_dir.exists():
        for p in calib_claims_dir.glob("*.json"):
            if p.name == "status.json":
                continue
            try:
                idxs.add(int(json.loads(p.read_text())["idx"]))
            except Exception:
                pass
    return idxs


def _dist(claims, verdicts):
    d = {v: 0 for v in VERDICTS}
    for c, v in zip(claims, verdicts):
        d[v["verdict"]] = d.get(v["verdict"], 0) + 1
    return d


def replay(run_dir: Path, calib_claims_dir: Path | None, min_k: int) -> dict:
    claims = _load_claims(run_dir)
    in_samp = _in_sample_idxs(calib_claims_dir)
    is_oos = [int(c.get("idx", -1)) not in in_samp for c in claims]

    base = [baseline_verdict(c) for c in claims]
    # faithfulness: baseline must reproduce the stored verdict.
    mism = [(c.get("idx"), c.get("claim_i"), c.get("verdict"), b["verdict"])
            for c, b in zip(claims, base)
            if c.get("verdict") is not None and b["verdict"] != c.get("verdict")]

    VARIANTS = {
        "encoded (safe)":            dict(encoded=True),
        "encoded+outlier":           dict(encoded=True, outlier=True),
        "power (M_BB<=0)":           dict(power=True),
        "encoded+power (safe)":      dict(encoded=True, power=True),
        "FULL (enc+out+power)":      dict(encoded=True, outlier=True, power=True),
        "NAIVE enc+out (diag only)": dict(encoded=True, outlier=True, naive=True),
    }

    base_resolved = [b["verdict"] != "unresolved" for b in base]
    n = len(claims)
    n_oos = sum(is_oos)
    base_unres = sum(1 for b in base if b["verdict"] == "unresolved")
    base_unres_oos = sum(1 for b, o in zip(base, is_oos)
                         if o and b["verdict"] == "unresolved")

    report = {
        "run": run_dir.name, "n_claims": n, "n_oos": n_oos,
        "min_k": min_k, "n_in_sample_acts": len(in_samp),
        "baseline_reproduces_stored": not mism,
        "baseline_mismatches": mism[:10],
        "baseline_unresolved": base_unres,
        "baseline_unresolved_pct": round(100 * base_unres / n, 1),
        "baseline_unresolved_oos_pct": round(100 * base_unres_oos / max(n_oos, 1), 1),
        "baseline_verdict_distribution": _dist(claims, base),
        "variants": {},
    }

    for name, opts in VARIANTS.items():
        var = [variant_verdict(c, min_k=min_k, **opts) for c in claims]
        # newly resolved (baseline unresolved -> now resolved)
        newly = [(i, v) for i, (b, v) in enumerate(zip(base, var))
                 if b["verdict"] == "unresolved" and v["verdict"] != "unresolved"]
        newly_oos = [(i, v) for i, v in newly if is_oos[i]]
        # churn on already-resolved (MUST be 0 for non-naive)
        churn = [(claims[i].get("idx"), claims[i].get("claim_i"),
                  base[i]["verdict"], var[i]["verdict"])
                 for i in range(n)
                 if base_resolved[i] and var[i]["verdict"] != base[i]["verdict"]]
        # smell test: verdict mix of newly-resolved vs baseline-resolved
        newly_mix = {v: 0 for v in VERDICTS}
        for _, v in newly:
            newly_mix[v["verdict"]] += 1
        base_res_mix = {v: 0 for v in VERDICTS}
        for b in base:
            if b["verdict"] != "unresolved":
                base_res_mix[b["verdict"]] += 1
        n_new = len(newly)
        n_base_res = sum(base_res_mix.values())
        grounded_new = newly_mix["grounded"] + newly_mix["entangled"]
        grounded_base = base_res_mix["grounded"] + base_res_mix["entangled"]
        var_unres_oos = sum(1 for v, o in zip(var, is_oos)
                            if o and v["verdict"] == "unresolved")
        report["variants"][name] = {
            "newly_resolved": n_new,
            "newly_resolved_oos": len(newly_oos),
            "unresolved_after": sum(1 for v in var if v["verdict"] == "unresolved"),
            "unresolved_after_pct": round(100 * sum(1 for v in var if v["verdict"] == "unresolved") / n, 1),
            "unresolved_after_oos_pct": round(100 * var_unres_oos / max(n_oos, 1), 1),
            "newly_resolved_mix": newly_mix,
            "newly_grounded_frac": round(grounded_new / n_new, 3) if n_new else None,
            "baseline_resolved_grounded_frac": round(grounded_base / n_base_res, 3) if n_base_res else None,
            "churn_count": len(churn),
            "churn_examples": churn[:8],
        }
    return report


def _fmt(report: dict) -> str:
    L = []
    r = report
    L.append(f"### {r['run']}   n={r['n_claims']} (OOS {r['n_oos']}, in-sample acts {r['n_in_sample_acts']})")
    ok = "OK" if r["baseline_reproduces_stored"] else "FAIL"
    L.append(f"baseline reproduces stored verdicts: {ok}"
             + ("" if r["baseline_reproduces_stored"] else f"  MISMATCHES {r['baseline_mismatches']}"))
    L.append(f"baseline unresolved: {r['baseline_unresolved']} "
             f"({r['baseline_unresolved_pct']}% all, {r['baseline_unresolved_oos_pct']}% OOS)")
    L.append("")
    L.append(f"{'variant':26} {'unres%(OOS)':>11} {'newly':>6} {'grounded/entangled of newly':>28} {'churn':>6}")
    base_gf = None
    for name, v in r["variants"].items():
        gf = v["newly_grounded_frac"]
        base_gf = v["baseline_resolved_grounded_frac"]
        gtxt = (f"{v['newly_resolved_mix']['grounded']}g+{v['newly_resolved_mix']['entangled']}e "
                f"({'-' if gf is None else f'{100*gf:.0f}%'})")
        flag = ""
        if gf is not None and base_gf is not None and gf > base_gf + 0.02 and "NAIVE" not in name:
            flag = "  <-- SMELL: grounded spike"
        if "NAIVE" in name:
            flag = "  (diagnostic: false-grounded the lock prevents)"
        L.append(f"{name:26} {v['unresolved_after_oos_pct']:>10.1f}% {v['newly_resolved']:>6} "
                 f"{gtxt:>28} {v['churn_count']:>6}{flag}")
    L.append("")
    L.append(f"(baseline resolved grounded/entangled frac = {base_gf}; "
             f"a variant is clean if newly-resolved grounded frac is <= that and churn=0)")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--calib-claims-dir", default=None,
                    help="calib_d4g24/<run>/claims - its activation idxs are in-sample")
    ap.add_argument("--min-k", type=int, default=5)
    ap.add_argument("--json", default=None, help="write the full report JSON here")
    args = ap.parse_args()
    rep = replay(Path(args.run_dir),
                 Path(args.calib_claims_dir) if args.calib_claims_dir else None,
                 args.min_k)
    print(_fmt(rep))
    if args.json:
        Path(args.json).write_text(json.dumps(rep, indent=1), encoding="utf-8")
        print(f"\n[wrote {args.json}]")


if __name__ == "__main__":
    main()
