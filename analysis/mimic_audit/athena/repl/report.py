"""Cycle-consistency run report (CPU, anywhere). Reads a cycle_run.py output
directory (claims/<idx>_<ci>.json) and produces the verdict distribution, the
per-gate flow, the atom-vs-cycle verdict-shift cross-tab, the grounding-estimate
summary, and a LEAVE-ONE-DECOY-OUT re-fit of kappa_0 on THIS run's survivor D*
families - with the false-positive (decoy-vs-decoy exceedance) rate reported at
the calibration kappa_0 so Gate 1's null is re-validated on the run it is used on.

  python report.py --run-dir .../cycle_runs/qwen7b-d4g24/qwen7b-cp6-fineweb-atoms \
                   --calib .../calibration_signed.json
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

VERDICTS = ("grounded", "entangled", "seeded", "confabulated", "unresolved")


def _atomic_write(path: Path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    if isinstance(obj, str):
        tmp.write_text(obj, encoding="utf-8")
    else:
        tmp.write_text(json.dumps(obj, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def _pct(a, q):
    a = np.asarray([x for x in a if x is not None and np.isfinite(x)], np.float64)
    return float(np.percentile(a, q)) if a.size else float("nan")


def load_claims(run_dir: Path):
    rows = [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted((run_dir / "claims").glob("*.json"))
            if p.name != "status.json"]
    if not rows:
        raise SystemExit(f"no cycle claims under {run_dir}/claims")
    return rows


def kappa_lodo(rows, k, grid_hi=5.01, step=0.1, alpha=0.05):
    """Re-fit kappa_0 on this run's survivor D* families (spec Gate 1 null).
    For each claim with > k surviving decoys, leave-one-out: does D*_j exceed
    Q95 of the OTHER survivors' D* by more than kappa? Sweep kappa; the fit is
    the smallest value holding the pooled exceedance rate <= alpha. Returns the
    grid, the fit, and the number of contributing families."""
    fams = []
    for r in rows:
        d = [c.get("dstar") for c in r.get("candidates", [])
             if c.get("dstar") is not None and np.isfinite(c.get("dstar"))]
        if len(d) > k:
            fams.append(np.asarray(d, np.float64))
    grid, fit = [], None
    for kap in np.arange(0.0, grid_hi, step):
        n_pass = n_tot = 0
        for d in fams:
            for j in range(d.size):
                rest = np.delete(d, j)
                n_pass += int(d[j] > np.percentile(rest, 95) + kap)
                n_tot += 1
        rate = (n_pass / n_tot) if n_tot else None
        grid.append(dict(kappa=round(float(kap), 2),
                         lodo_pass_rate=round(rate, 4) if rate is not None else None))
        if fit is None and rate is not None and rate <= alpha:
            fit = float(round(kap, 2))
    return dict(kappa_0_refit=fit, n_families=len(fams), grid=grid, alpha=alpha)


def rate_at(grid, kappa):
    """Decoy-vs-decoy exceedance rate at a specific kappa (nearest grid point)."""
    if kappa is None or not grid:
        return None
    g = min(grid, key=lambda x: abs(x["kappa"] - kappa))
    return g["lodo_pass_rate"]


def build_report(rows, calib, calib_idxs=None):
    k = calib.get("k", 5)
    kappa_cal = calib.get("kappa_0")
    calib_idxs = set(calib_idxs or [])

    dist = {v: 0 for v in VERDICTS}
    unresolved_reasons: dict[str, int] = {}
    for r in rows:
        v = r.get("verdict", "unresolved")
        dist[v] = dist.get(v, 0) + 1
        if v == "unresolved":
            reason = r.get("reason") or "unspecified"
            unresolved_reasons[reason] = unresolved_reasons.get(reason, 0) + 1

    # per-gate flow (gate field: 2 -> grounded/entangled, 3 -> seeded/confabulated)
    gated = [r for r in rows if r.get("verdict") in
             ("grounded", "entangled", "seeded", "confabulated")]
    gate1_pass = [r for r in gated if r.get("gate") == 2]
    gate1_fail = [r for r in gated if r.get("gate") == 3]
    flow = dict(
        total=len(rows), unresolved=dist["unresolved"], gated=len(gated),
        gate1_pass=len(gate1_pass), gate1_fail=len(gate1_fail),
        gate2_grounded=sum(1 for r in gate1_pass if r["verdict"] == "grounded"),
        gate2_entangled=sum(1 for r in gate1_pass if r["verdict"] == "entangled"),
        gate3_seeded=sum(1 for r in gate1_fail if r["verdict"] == "seeded"),
        gate3_confabulated=sum(1 for r in gate1_fail if r["verdict"] == "confabulated"))

    # atom(verdict_stored) -> cycle(verdict) shift cross-tab
    atom_labels = sorted({(r.get("verdict_stored") or "none") for r in rows})
    shift = {a: {v: 0 for v in VERDICTS} for a in atom_labels}
    for r in rows:
        a = r.get("verdict_stored") or "none"
        shift[a][r.get("verdict", "unresolved")] += 1

    # power / validity pass rates
    n_pow = sum(1 for r in rows if (r.get("power") or {}).get("passed"))
    n_val = sum(1 for r in rows if (r.get("validity") or {}).get("passed"))

    # grounding estimate = M_BB - med D*. Only interpretable where the ablation
    # genuinely hurt B (M_BB > 0), i.e. the Gate-1 passers (grounded+entangled);
    # for a Gate-3 claim with M_BB <= 0 the number can be positive purely
    # because the decoy null is more negative, so summarising it there would
    # misread as grounding signal. Restrict the summary to Gate-1 passers.
    g_est = [r.get("grounding_estimate") for r in gate1_pass
             if r.get("grounding_estimate") is not None]

    kfit = kappa_lodo(rows, k)
    fp_at_cal = rate_at(kfit["grid"], kappa_cal)
    refit = kfit["kappa_0_refit"]
    diverges = (kappa_cal is not None and refit is not None
                and abs(refit - kappa_cal) > 0.1)

    # genuine OUT-OF-SAMPLE Gate-1 validation: re-fit + FP rate on the families
    # from activations DISJOINT from the calibration split. This is the number
    # that actually tests whether kappa_0=0.8 transfers; the global kfit above
    # is contaminated by the in-sample calibration activations.
    oos = None
    if calib_idxs:
        oos_rows = [r for r in rows if r.get("idx") not in calib_idxs]
        in_rows = [r for r in rows if r.get("idx") in calib_idxs]
        ok = kappa_lodo(oos_rows, k)
        oos = dict(
            n_activations_out_of_sample=len({r["idx"] for r in oos_rows}),
            n_activations_in_sample=len({r["idx"] for r in in_rows}),
            n_families=ok["n_families"],
            kappa_0_refit=ok["kappa_0_refit"],
            decoy_fp_rate_at_calibration_kappa=rate_at(ok["grid"], kappa_cal),
            transfers=bool(rate_at(ok["grid"], kappa_cal) is not None
                           and rate_at(ok["grid"], kappa_cal) <= 0.05))

    return dict(
        n_claims=len(rows), k=k,
        verdict_distribution=dist,
        unresolved_reasons=unresolved_reasons,
        gate_flow=flow,
        atom_vs_cycle_shift=shift,
        power_pass=n_pow, validity_pass=n_val,
        grounding_estimate=dict(
            scope="Gate-1 passers only (grounded+entangled; M_BB > 0)",
            n=len(g_est),
            median=round(_pct(g_est, 50), 4) if g_est else None,
            p10=round(_pct(g_est, 10), 4) if g_est else None,
            p90=round(_pct(g_est, 90), 4) if g_est else None,
            min=round(float(np.min(g_est)), 4) if g_est else None,
            max=round(float(np.max(g_est)), 4) if g_est else None),
        kappa_0_calibration=kappa_cal,
        kappa_0_refit=refit,
        kappa_0_n_families=kfit["n_families"],
        decoy_fp_rate_at_calibration_kappa=fp_at_cal,
        kappa_0_diverges_from_calibration=bool(diverges),
        kappa_out_of_sample=oos,
        kappa_validation_note=(
            "This re-fit is IN-SAMPLE when the run activations are the "
            "calibration split (e.g. the S4 small-N run reuses the 25 "
            "calibration activations): the LODO families and grid are then "
            "identical to calib_cycle.py's, so the FP rate is a consistency "
            "check, NOT out-of-sample validation. Genuine out-of-sample "
            "validation of kappa_0 requires a run on activations disjoint from "
            "calibration (S5's 375 new activations). "
            + ("WARNING: kappa_0_refit disagrees with the applied calibration "
               "kappa_0 by > 0.1 - the applied kappa_0 is stale for this set."
               if diverges else
               "kappa_0_refit agrees with the applied calibration kappa_0.")),
        kappa_grid=kfit["grid"])


def render_md(rep, rows, run_dir):
    d = rep["verdict_distribution"]
    n = rep["n_claims"]
    f = rep["gate_flow"]
    lines = [
        "# Cycle-consistency verdict report", "",
        f"run `{run_dir.name}`  claims {n}  k {rep['k']}", "",
        "## Verdict distribution", "",
        "| verdict | n | % |", "|---|---|---|"]
    for v in VERDICTS:
        lines.append(f"| {v} | {d[v]} | {100 * d[v] / n:.1f} |")
    if rep["unresolved_reasons"]:
        lines += ["", "unresolved by reason: " + ", ".join(
            f"{r} {c}" for r, c in sorted(rep["unresolved_reasons"].items(),
                                          key=lambda x: -x[1]))]
    lines += [
        "", "## Gate flow (spec Step 7)", "",
        f"- {f['total']} claims -> {f['unresolved']} unresolved (pre-checks), "
        f"{f['gated']} reached the gates",
        f"- Gate 1 (specificity): {f['gate1_pass']} pass -> Gate 2, "
        f"{f['gate1_fail']} fail -> Gate 3",
        f"- Gate 2 (ownership): {f['gate2_grounded']} grounded, "
        f"{f['gate2_entangled']} entangled",
        f"- Gate 3 (shared signal): {f['gate3_seeded']} seeded, "
        f"{f['gate3_confabulated']} confabulated",
        "", "## Atom -> cycle verdict shift", "",
        "rows = published atom verdict, columns = cycle verdict", "",
        "| atom \\ cycle | " + " | ".join(VERDICTS) + " |",
        "|---|" + "|".join("---" for _ in VERDICTS) + "|"]
    for a, row in sorted(rep["atom_vs_cycle_shift"].items()):
        lines.append(f"| {a} | " + " | ".join(str(row[v]) for v in VERDICTS) + " |")
    ge = rep["grounding_estimate"]
    lines += [
        "", "## Pre-checks + grounding", "",
        f"- power passed {rep['power_pass']}/{n}, validity passed "
        f"{rep['validity_pass']}/{n}",
        f"- grounding estimate (M_BB - med D*, {ge['scope']}, n={ge['n']}): "
        f"median {ge['median']}, p10 {ge['p10']}, p90 {ge['p90']}",
        "", "## kappa_0 re-fit (Gate 1 null on THIS run)", "",
        f"- calibration kappa_0 = {rep['kappa_0_calibration']} (applied)",
        f"- re-fit on {rep['kappa_0_n_families']} survivor D* families = "
        f"{rep['kappa_0_refit']}"
        + ("  **DIVERGES > 0.1**" if rep["kappa_0_diverges_from_calibration"] else ""),
        f"- decoy-vs-decoy exceedance rate at the calibration kappa_0 = "
        f"{rep['decoy_fp_rate_at_calibration_kappa']} (target <= 0.05, "
        f"whole run incl. in-sample)",
    ]
    oos = rep.get("kappa_out_of_sample")
    if oos:
        lines += [
            f"- **out-of-sample** ({oos['n_activations_out_of_sample']} activations "
            f"disjoint from the {oos['n_activations_in_sample']} calibration ones, "
            f"{oos['n_families']} families): re-fit kappa_0 = {oos['kappa_0_refit']}, "
            f"FP rate at calibration kappa_0 = "
            f"{oos['decoy_fp_rate_at_calibration_kappa']} -> "
            f"{'TRANSFERS (<= 0.05)' if oos['transfers'] else 'DOES NOT TRANSFER (> 0.05)'}"]
    lines += [
        f"- _{rep['kappa_validation_note']}_",
        "", "## Per-claim", "",
        "| idx.ci | type | atom | cycle | gate | M_BB | gate1_bar | ground_est "
        "| surv | pw | vd |",
        "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda x: (x["idx"], x["claim_i"])):
        def fmt(x):
            return "" if x is None else (f"{x:.2f}" if isinstance(x, float) else str(x))
        lines.append(
            f"| {r['idx']}.{r['claim_i']} | {r.get('claim_type', '')} "
            f"| {r.get('verdict_stored', '')} | {r.get('verdict', '')} "
            f"| {r.get('gate') or ''} | {fmt(r.get('m_bb'))} "
            f"| {fmt(r.get('gate1_bar'))} | {fmt(r.get('grounding_estimate'))} "
            f"| {r.get('n_surv', '')} "
            f"| {'P' if (r.get('power') or {}).get('passed') else '.'} "
            f"| {'P' if (r.get('validity') or {}).get('passed') else '.'} |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-dir", required=True,
                    help="cycle_run output dir (.../<run_id>) with claims/")
    ap.add_argument("--calib", required=True, help="calibration_signed.json")
    ap.add_argument("--calib-claims-dir", default=None,
                    help="calibration claims/ dir (.../calib_d4g24/<run>/claims); "
                         "its activation idxs are treated as IN-SAMPLE so the "
                         "report can compute an out-of-sample kappa_0 validation")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    calib = json.loads(Path(args.calib).read_text(encoding="utf-8"))
    rows = load_claims(run_dir)
    calib_idxs = None
    if args.calib_claims_dir:
        calib_idxs = sorted({int(p.name.split("_")[0])
                             for p in Path(args.calib_claims_dir).glob("*.json")
                             if p.name != "status.json"})
        print(f"in-sample calibration activations: {len(calib_idxs)}")
    rep = build_report(rows, calib, calib_idxs=calib_idxs)
    rep["run_dir"] = str(run_dir)
    _atomic_write(run_dir / "cycle_report.json", rep)
    _atomic_write(run_dir / "CYCLE_REPORT.md", render_md(rep, rows, run_dir))
    print(json.dumps({k: v for k, v in rep.items() if k != "kappa_grid"}, indent=1))
    print(f"\nwrote {run_dir}/cycle_report.json and CYCLE_REPORT.md")


if __name__ == "__main__":
    main()
