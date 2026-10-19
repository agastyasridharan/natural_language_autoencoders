"""Recompute dashboard uncertainty from the stored per-document records.

No model inference is performed here.  Every manipulation is paired to the
same document and frozen activation, so the existing cosine/D_i records are
sufficient for the registered estimands.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from inference_stats import bh_adjust, bootstrap_bundle, one_sample_t

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
N_BOOT = 20_000
MODELS = ("qwen7b", "gemma12b", "gemma27b")
ARMS = ("exp1", "exp2", "exp3a", "exp3b", "exp4a", "exp4b", "exp5a", "exp5b", "exp5c",
        "exp6a", "exp6b", "exp6f", "exp6r", "exp6o", "exp6p", "exp6t")


def seed(label: str) -> int:
    return int.from_bytes(hashlib.sha256(label.encode()).digest()[:8], "big") & 0x7FFFFFFF


def load_records(arm: str, key: str) -> list[dict]:
    return json.loads((RES / f"records_{arm}_{key}.json").read_text())


def arrays(rows: list[dict], selector=lambda _r: True, arm: bool = True):
    kept = [r for r in rows if selector(r) and r.get("D_i") is not None
            and r.get("cos_orig") is not None and (not arm or r.get("cos_edit") is not None)]
    d = np.asarray([r["D_i"] for r in kept], dtype=np.float64)
    cb = np.asarray([r["cos_orig"] for r in kept], dtype=np.float64)
    ca = np.asarray([r["cos_edit"] for r in kept], dtype=np.float64) if arm else None
    return kept, d, cb, ca


def p_fields(test: dict) -> dict:
    return {k: test[k] for k in ("t", "df", "p", "log10_p")}


def baseline_result(rows: list[dict], key: str) -> dict:
    kept, d, cb, _ = arrays(rows, arm=False)
    boot = bootstrap_bundle(d, cb, None, n_boot=N_BOOT, seed=seed(f"baseline|{key}"))
    loss = 2.0 * (1.0 - cb)
    return {
        "n": len(kept), "n_docs": len({r["doc_id"] for r in kept}),
        "FVE": float(1.0 - loss.sum() / d.sum()), "FVE_ci95": boot["FVE_base"],
        "FVE_test": p_fields(one_sample_t(d - loss)),
        "mean_cos": float(cb.mean()), "mean_cos_ci95": boot["mean_cos_base"],
    }


def arm_result(rows: list[dict], key: str, arm: str, selector=lambda _r: True,
               suffix: str = "full") -> dict:
    kept, d, cb, ca = arrays(rows, selector=selector, arm=True)
    assert ca is not None
    boot = bootstrap_bundle(d, cb, ca, n_boot=N_BOOT,
                            seed=seed(f"arm|{arm}|{key}|{suffix}"))
    lb, la = 2.0 * (1.0 - cb), 2.0 * (1.0 - ca)
    fb, fa = 1.0 - lb.sum() / d.sum(), 1.0 - la.sum() / d.sum()
    test = p_fields(one_sample_t(2.0 * (ca - cb)))
    return {
        "n": len(kept), "n_docs": len({r["doc_id"] for r in kept}),
        "FVE_base": float(fb), "FVE_base_ci95": boot["FVE_base"],
        "FVE_arm": float(fa), "FVE_arm_ci95": boot["FVE_arm"],
        "dFVE": float(fa - fb), "dFVE_ci95": boot["dFVE"], "p_test": test,
        "mean_cos_base": float(cb.mean()), "mean_cos_base_ci95": boot["mean_cos_base"],
        "mean_cos_arm": float(ca.mean()), "mean_cos_arm_ci95": boot["mean_cos_arm"],
        "mean_dcos": float((ca - cb).mean()), "mean_dcos_ci95": boot["mean_dcos"],
    }


def contrast(rows_a: list[dict], rows_b: list[dict], key: str, label: str,
             selector=lambda _r: True) -> dict:
    by_b = {r["idx"]: r for r in rows_b if r.get("cos_edit") is not None}
    paired = [(r, by_b[r["idx"]]) for r in rows_a if selector(r) and r.get("D_i") is not None
              and r.get("cos_edit") is not None and r["idx"] in by_b]
    d = np.asarray([a["D_i"] for a, _ in paired], dtype=np.float64)
    cb = np.asarray([b["cos_edit"] for _, b in paired], dtype=np.float64)
    ca = np.asarray([a["cos_edit"] for a, _ in paired], dtype=np.float64)
    boot = bootstrap_bundle(d, cb, ca, n_boot=N_BOOT, seed=seed(f"contrast|{key}|{label}"))
    value = float(2.0 * (ca - cb).sum() / d.sum())
    return {
        "label": label, "n": len(paired), "n_docs": len({a["doc_id"] for a, _ in paired}),
        "dFVE": value, "dFVE_ci95": boot["dFVE"],
        "p_test": p_fields(one_sample_t(2.0 * (ca - cb))),
    }


def high_div(row: dict) -> bool:
    return any(float(q.get("divergence", -1)) >= 0.5
               for q in row.get("edit_meta", {}).get("quotes", []))


def changed(row: dict) -> bool:
    return row.get("expl_orig") != row.get("expl_edit")


def continuation_wrong(row: dict) -> bool:
    return row.get("edit_meta", {}).get("continuation_wrong") is True


def main() -> int:
    baselines = {}
    arms = {arm: {} for arm in ARMS}
    arm_family = []
    for key in MODELS:
        base_rows = load_records("exp1", key)
        baselines[key] = baseline_result(base_rows, key)
        for arm in ARMS:
            result = arm_result(load_records(arm, key), key, arm)
            arms[arm][key] = result
            arm_family.append(result)
    llama_rows = json.loads((RES / "records_baseline_llama70b.json").read_text())
    baselines["llama70b"] = baseline_result(llama_rows, "llama70b")
    bh_adjust(arm_family)

    contrasts = {}
    contrast_family = []
    for key in MODELS:
        raw = {arm: load_records(arm, key) for arm in
               ("exp5a", "exp5b", "exp5c", "exp6b", "exp6p", "exp6t")}
        rows = {
            "exp5b_full": arm_result(raw["exp5b"], key, "exp5b", suffix="contrast"),
            "exp5c_full": arm_result(raw["exp5c"], key, "exp5c", suffix="contrast"),
            "exp5c_minus_exp5a": contrast(raw["exp5c"], raw["exp5a"], key, "5c - 5a"),
            "exp6p_full": arm_result(raw["exp6p"], key, "exp6p", suffix="contrast-full"),
            "exp6p_changed": arm_result(raw["exp6p"], key, "exp6p", changed, "changed"),
            "exp6p_high_div_changed": arm_result(
                raw["exp6p"], key, "exp6p", lambda r: changed(r) and high_div(r), "high-div-changed"),
            "exp6p_minus_a2_full": contrast(raw["exp6p"], raw["exp6b"], key, "P - A2"),
            "exp6p_minus_a2_high_div_changed": contrast(
                raw["exp6p"], raw["exp6b"], key, "P - A2, high-div changed",
                lambda r: changed(r) and high_div(r)),
            "exp6t_full": arm_result(raw["exp6t"], key, "exp6t", suffix="contrast-full"),
            "exp6t_wrong": arm_result(raw["exp6t"], key, "exp6t", continuation_wrong, "wrong"),
        }
        contrasts[key] = rows
        contrast_family.extend(rows.values())
    bh_adjust(contrast_family)

    output = {
        "method": {
            "sampling_unit": "document", "n_boot": N_BOOT,
            "ci": "seeded nonparametric document-bootstrap percentile 95% interval",
            "p": "two-sided one-sample Student-t test on the document-level estimating equation",
            "multiplicity": "Benjamini-Hochberg q within the 48 manipulation tests or displayed contrast family",
        },
        "baseline": baselines, "arms": arms, "contrasts": contrasts,
    }
    (RES / "dashboard_inference.json").write_text(json.dumps(output, indent=2))
    (RES / "remaining_contrasts.json").write_text(json.dumps(contrasts, indent=2))
    print(f"wrote dashboard_inference.json; {len(arm_family)} arm tests; {len(contrast_family)} contrasts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
