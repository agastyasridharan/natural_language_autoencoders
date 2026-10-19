"""Cycle-consistency data layer for the Claim-Ablation Studio.

Read-only. Auto-discovers cycle_run.py outputs under <athena>/cycle_runs/*/*/ and
serves the report + per-claim drill-downs for the studio's cycle tab. Never loads
a model, never mutates a run. The verdicts, gate flow, readings and decoy panels
are all mechanical artifacts produced upstream by cycle_run.py + report.py; this
module only shapes them for the page.

Cycle-run layout it expects:
  <athena>/cycle_runs/<tag>/<run>/cycle_report.json          (report.py output)
  <athena>/cycle_runs/<tag>/<run>/claims/<idx>_<ci>.json     (cycle_run.py output)
  <athena>/calib_d4g24/<run>/calibration_signed.json         (thresholds)
where <tag> is "<key>-d4g24[-suffix]" and <key> is a repl_common.MODELS key.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ATHENA = HERE.parent
CYCLE_ROOT = ATHENA / "cycle_runs"
CALIB_ROOT = ATHENA / "calib_d4g24"

try:
    from repl_common import MODELS            # repl is on sys.path via server.py
except Exception:                             # pragma: no cover - defensive
    MODELS = {}

# fixed presentation order (scale-ascending, family-grouped)
_ORDER = ["qwen7b", "gemma12b", "gemma27b", "llama70b"]
VERDICTS = ("grounded", "entangled", "seeded", "confabulated", "unresolved")


def _key_of_tag(tag: str) -> str:
    return tag.split("-d4g24")[0]


def _load(p: Path, default=None):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return default


def discover() -> dict:
    """Map model key -> {tag, run, report_path, claims_dir}. When a key has more
    than one cycle run (e.g. qwen7b's smoke / small / full), keep the one with the
    most scored claims (the full run)."""
    best: dict[str, dict] = {}
    if not CYCLE_ROOT.exists():
        return best
    for rep in CYCLE_ROOT.glob("*/*/cycle_report.json"):
        run_dir = rep.parent
        tag = run_dir.parent.name
        key = _key_of_tag(tag)
        report = _load(rep, {})
        n = int(report.get("n_claims") or 0)
        cur = best.get(key)
        if cur is None or n > cur["n_claims"]:
            best[key] = dict(key=key, tag=tag, run=run_dir.name,
                             report_path=str(rep), claims_dir=str(run_dir / "claims"),
                             n_claims=n)
    return best


def _model_meta(key: str) -> dict:
    m = MODELS.get(key, {})
    return dict(key=key, label=m.get("label", key), family=m.get("family"),
                layer=m.get("layer"), d_model=m.get("d_model"))


def list_cycle_runs() -> list[dict]:
    """One summary row per model, in _ORDER, for the cross-model overview."""
    found = discover()
    out = []
    ordered = [k for k in _ORDER if k in found] + \
              [k for k in found if k not in _ORDER]
    for key in ordered:
        info = found[key]
        rep = _load(Path(info["report_path"]), {})
        meta = _model_meta(key)
        oos = rep.get("kappa_out_of_sample") or {}
        out.append(dict(
            **meta, tag=info["tag"], run=info["run"],
            n_claims=rep.get("n_claims"),
            verdict_distribution=rep.get("verdict_distribution", {}),
            gate_flow=rep.get("gate_flow", {}),
            atom_vs_cycle_shift=rep.get("atom_vs_cycle_shift", {}),
            grounding_estimate=rep.get("grounding_estimate", {}),
            kappa_0_calibration=rep.get("kappa_0_calibration"),
            kappa_0_oos_refit=oos.get("kappa_0_refit"),
            kappa_oos_fp=oos.get("decoy_fp_rate_at_calibration_kappa"),
            kappa_oos_transfers=oos.get("transfers"),
            unresolved_reasons=rep.get("unresolved_reasons", {}),
            tier1=rep.get("tier1"),
            tier0=rep.get("tier0"),
            tier3=rep.get("tier3"),
        ))
    return out


def cycle_report(key: str) -> dict | None:
    info = discover().get(key)
    if not info:
        return None
    rep = _load(Path(info["report_path"]), {})
    rep["_meta"] = _model_meta(key)
    rep["_tag"] = info["tag"]
    rep["_run"] = info["run"]
    return rep


def _claim_row(c: dict) -> dict:
    """Lightweight per-claim summary for the browsable table."""
    return dict(
        idx=c.get("idx"), ci=c.get("claim_i"),
        verdict=c.get("verdict"), reason=c.get("reason"),
        claim_text=c.get("claim_text"), claim_type=c.get("claim_type"),
        taxo=c.get("taxo"),
        m_bb=c.get("m_bb"), m_bb_stored=c.get("m_bb_stored"),
        m_ib_max=c.get("m_ib_max"), n_text_pass=c.get("n_text_pass"),
        supp_b=c.get("supp_b"), verdict_stored=c.get("verdict_stored"),
        located_fc=c.get("located_fc"),
    )


def cycle_claims(key: str) -> dict | None:
    info = discover().get(key)
    if not info:
        return None
    cdir = Path(info["claims_dir"])
    rows = []
    for p in sorted(cdir.glob("*.json")):
        if p.name == "status.json":
            continue
        c = _load(p, {})
        if c:
            rows.append(_claim_row(c))
    rows.sort(key=lambda r: (r["idx"] if r["idx"] is not None else 1e9,
                             r["ci"] if r["ci"] is not None else 0))
    return dict(key=key, meta=_model_meta(key), tag=info["tag"], run=info["run"],
                n=len(rows), claims=rows)


def cycle_claim(key: str, idx: int, ci: int) -> dict | None:
    info = discover().get(key)
    if not info:
        return None
    p = Path(info["claims_dir"]) / f"{idx}_{ci}.json"
    c = _load(p, None)
    if c is None:
        return None
    c["_meta"] = _model_meta(key)
    return c
