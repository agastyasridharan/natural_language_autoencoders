"""Phase-0 validation gate: reproduce a logged decode dump with the white-box
harness, proving the wiring (extraction layer, injection, AR scoring) is right
before any experiment is trusted.

Three independent checks, cheapest/most-diagnostic first:

  EXTRACTION  (no AR, no AV) — re-extract the dump's conversation and compare
      each position's RAW activation norm to the logged ||v||. A mismatch means
      the layer index, chat template, or extraction model is wrong. This alone
      catches the loudest footgun (layer 20 = hidden_states[21] = block-20
      output, NOT hidden_states[20]).

  L1 / AR    (deterministic) — feed the dump's OWN explanation text back through
      the AR against the re-extracted activation; compare (mse_nrm, cos) to the
      log. Isolates AR + extraction; no sampling, so it should match to ~fp
      tolerance regardless of the AV decode engine.

  L2 / end-to-end (optional) — re-verbalize each activation with the AV
      (greedy) and re-score. Text won't match the SGLang-logged decode
      token-for-token, so this compares the SUMMARY distribution (mean cos, FVE
      counts) within a looser tolerance.

Pass/fail thresholds are arguments; the defaults are calibrated to the
Qwen-2.5-7B L20 dump.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from nla.whitebox import example_log as el
from nla.whitebox.harness import WhiteBoxNLA


@dataclass
class PositionCheck:
    idx: int
    region: str
    token: str
    file_raw_norm: float
    got_raw_norm: float
    file_mse: float
    got_mse: float | None
    file_cos: float
    got_cos: float | None


@dataclass
class GateReport:
    level: str
    n_positions: int
    min_position: int
    # extraction check
    max_norm_rel_err: float
    norm_check_passed: bool
    # AR / end-to-end checks
    per_position: list[PositionCheck] = field(default_factory=list)
    file_summary: el.SummaryStats | None = None
    got_summary: el.SummaryStats | None = None
    median_abs_dcos: float | None = None
    passed: bool = False
    notes: list[str] = field(default_factory=list)


def _extract_aligned(
    nla: WhiteBoxNLA,
    log: el.ParsedLog,
    conv: el.Conversation | None,
    *,
    norm_rel_tol: float,
    max_new_tokens: int,
    notes: list[str],
):
    """Re-extract the dump's conversation and align to the logged positions via
    raw-norm cross-check. Returns (ConversationActivations, max_rel_err, ok)."""
    n_log = len(log.positions)
    messages = [{"role": "user", "content": conv.user_message if conv else "What are you hiding?"}]

    # First try faithful greedy regeneration (reproduces the original tokens on
    # the same model). Fall back to verbatim reply reconstruction if the length
    # doesn't line up with the log.
    acts = nla.extract_conversation(messages, generate_reply=True, max_new_tokens=max_new_tokens)
    if len(acts) != n_log and conv is not None and conv.reply_text:
        notes.append(
            f"greedy regen gave {len(acts)} positions != {n_log} logged; "
            f"falling back to verbatim reply reconstruction."
        )
        acts = nla.extract_conversation(messages, generate_reply=False, reply_text=conv.reply_text)

    n = min(len(acts), n_log)
    if len(acts) != n_log:
        notes.append(f"length mismatch: re-extracted {len(acts)} vs logged {n_log}; comparing first {n}.")

    rel_errs = []
    for p in range(n):
        got = float(acts.residuals[p].norm())
        ref = log.positions[p].raw_norm
        if ref > 1e-6:
            rel_errs.append(abs(got - ref) / ref)
    max_rel = max(rel_errs) if rel_errs else float("inf")
    return acts, max_rel, max_rel <= norm_rel_tol


def run_gate(
    nla: WhiteBoxNLA,
    log_path: str,
    *,
    level: str = "L1",
    baseline: float = 0.7335,
    min_position: int = 24,
    norm_rel_tol: float = 0.02,
    l1_median_dcos_tol: float = 0.02,
    summary_cos_tol: float = 0.03,
    summary_count_tol: int = 2,
    l2_summary_cos_tol: float = 0.06,
    max_new_tokens: int = 200,
    verbose: bool = True,
) -> GateReport:
    """Run the gate. `level` in {"extract", "L1", "L2"} (each includes the
    cheaper ones). `baseline` is the FVE denominator (Var(v_nrm))."""
    log = el.parse_log(log_path)
    conv = el.parse_conversation(log_path)
    notes: list[str] = []

    acts, max_rel, norm_ok = _extract_aligned(
        nla, log, conv, norm_rel_tol=norm_rel_tol, max_new_tokens=max_new_tokens, notes=notes
    )
    n = min(len(acts), len(log.positions))
    report = GateReport(
        level=level,
        n_positions=n,
        min_position=min_position,
        max_norm_rel_err=max_rel,
        norm_check_passed=norm_ok,
        notes=notes,
    )
    if verbose:
        print(f"[gate] extraction: max raw-norm rel err = {max_rel:.4f} "
              f"({'PASS' if norm_ok else 'FAIL'} @ tol {norm_rel_tol})")
    if level == "extract":
        report.passed = norm_ok
        return report

    # ── L1 / L2 per-position scoring ──────────────────────────────────────────
    end_to_end = level == "L2"
    checks: list[PositionCheck] = []
    got_records: list[el.PositionRecord] = []
    dcos: list[float] = []

    for p in range(n):
        rec = log.positions[p]
        activation = acts.residuals[p]
        if end_to_end:
            # Re-verbalize with the AV (greedy) then score — only for the
            # summarized region (saves ~24 generations).
            if p < min_position:
                checks.append(PositionCheck(p, acts.region(p), rec.token, rec.raw_norm,
                                            float(activation.norm()), rec.mse_nrm, None, rec.cos, None))
                continue
            expl = nla.verbalize(activation, greedy=True, max_new_tokens=max_new_tokens)
        else:
            expl = rec.explanation_clean
        mse, cos = nla.score(expl, activation)
        fve = WhiteBoxNLA.fve_nrm(mse, baseline)
        checks.append(PositionCheck(p, acts.region(p), rec.token, rec.raw_norm,
                                    float(activation.norm()), rec.mse_nrm, mse, rec.cos, cos))
        got_records.append(el.PositionRecord(p, acts.region(p), rec.token, rec.raw_norm,
                                             mse, cos, fve, expl))
        if not end_to_end:
            dcos.append(abs(cos - rec.cos))
        if verbose and (p % 10 == 0 or p == n - 1):
            tag = "L2" if end_to_end else "L1"
            print(f"[gate:{tag}] pos {p:3d} {acts.region(p):6s} "
                  f"file(mse={rec.mse_nrm:.3f} cos={rec.cos:.3f})  "
                  f"got(mse={mse:.3f} cos={cos:.3f})")

    report.per_position = checks

    # Aggregate over the summarized region and compare to the log's own summary.
    region_got = [r for r in got_records if r.idx >= min_position]
    file_region = log.from_position(min_position)
    report.file_summary = el.summarize(file_region)
    report.got_summary = el.summarize(region_got) if region_got else None

    if not end_to_end:
        report.median_abs_dcos = (sorted(dcos)[len(dcos) // 2] if dcos else float("inf"))

    report.passed = _decide(report, end_to_end, l1_median_dcos_tol,
                            summary_cos_tol, summary_count_tol, l2_summary_cos_tol)
    return report


def _decide(report, end_to_end, l1_median_dcos_tol, summary_cos_tol,
            summary_count_tol, l2_summary_cos_tol) -> bool:
    fs, gs = report.file_summary, report.got_summary
    if not report.norm_check_passed:
        report.notes.append("extraction norm check failed — fix layer/template before trusting AR/AV numbers.")
        return False
    if gs is None or fs is None:
        return False
    cos_tol = l2_summary_cos_tol if end_to_end else summary_cos_tol
    cos_ok = abs(gs.cos_mean - fs.cos_mean) <= cos_tol
    count_ok = (
        abs(gs.n_fve_gt_0_5 - fs.n_fve_gt_0_5) <= summary_count_tol
        and abs(gs.n_fve_gt_0_8 - fs.n_fve_gt_0_8) <= summary_count_tol
    )
    if not cos_ok:
        report.notes.append(
            f"summary cos_mean {gs.cos_mean:.3f} vs file {fs.cos_mean:.3f} "
            f"(Δ={abs(gs.cos_mean - fs.cos_mean):.3f} > tol {cos_tol})."
        )
    if not end_to_end:
        l1_ok = report.median_abs_dcos is not None and report.median_abs_dcos <= l1_median_dcos_tol
        if not l1_ok:
            report.notes.append(
                f"L1 median |Δcos| {report.median_abs_dcos:.3f} > tol {l1_median_dcos_tol} — "
                f"AR or extraction drift (HF vs the logging stack)."
            )
        return bool(cos_ok and count_ok and l1_ok)
    if end_to_end:
        report.notes.append(
            "L2 compares distributions only — HF greedy ≠ the SGLang-logged "
            "decode token-for-token, so per-position text differs by design."
        )
    return bool(cos_ok and count_ok)


def format_report(report: GateReport) -> str:
    lines = []
    lines.append("═" * 72)
    lines.append(f"  WHITE-BOX GATE — level={report.level}  "
                 f"{'PASS ✓' if report.passed else 'FAIL ✗'}")
    lines.append("═" * 72)
    lines.append(f"  positions compared : {report.n_positions}")
    lines.append(f"  extraction norm    : max rel err {report.max_norm_rel_err:.4f}  "
                 f"[{'PASS' if report.norm_check_passed else 'FAIL'}]")
    if report.median_abs_dcos is not None:
        lines.append(f"  L1 median |Δcos|   : {report.median_abs_dcos:.4f}")
    if report.file_summary and report.got_summary:
        fs, gs = report.file_summary, report.got_summary
        lines.append(f"  summary (pos {report.min_position}+, n={gs.n}):")
        lines.append(f"    cos_mean   file={fs.cos_mean:.3f}   got={gs.cos_mean:.3f}")
        lines.append(f"    mse_mean   file={fs.mse_mean:.3f}   got={gs.mse_mean:.3f}")
        lines.append(f"    fve_mean   file={fs.fve_mean:.3f}   got={gs.fve_mean:.3f}")
        lines.append(f"    fve>0.5    file={fs.n_fve_gt_0_5}      got={gs.n_fve_gt_0_5}")
        lines.append(f"    fve>0.8    file={fs.n_fve_gt_0_8}      got={gs.n_fve_gt_0_8}")
    if report.notes:
        lines.append("  notes:")
        for nt in report.notes:
            lines.append(f"    - {nt}")
    lines.append("═" * 72)
    return "\n".join(lines)
