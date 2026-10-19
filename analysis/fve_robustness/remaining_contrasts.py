"""Paired set-level contrasts for the completed Exp. 5 and Exp. 6 arms.

All arms use the same frozen gold activations.  This script recomputes FVE on identical selected rows
and bootstraps document clusters, allowing direct arm-minus-arm contrasts without mixing denominators.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
KEYS = ("qwen7b", "gemma12b", "gemma27b")
N_BOOT = 5000


def load(arm: str, key: str) -> list[dict]:
    return json.loads((RES / f"records_{arm}_{key}.json").read_text())


def fve(rows: list[dict], field: str) -> float:
    denominator = sum(float(row["D_i"]) for row in rows)
    loss = sum(2.0 * (1.0 - float(row[field])) for row in rows)
    return 1.0 - loss / denominator


def usable(row: dict) -> bool:
    return all(row.get(field) is not None for field in ("D_i", "cos_orig", "cos_edit"))


def high_div(row: dict) -> bool:
    return any(float(quote.get("divergence", -1)) >= 0.5
               for quote in row.get("edit_meta", {}).get("quotes", []))


def changed(row: dict) -> bool:
    return row.get("expl_orig") != row.get("expl_edit")


def continuation_wrong(row: dict) -> bool:
    return row.get("edit_meta", {}).get("continuation_wrong") is True


def _cluster_boot(rows: list[dict], stat, seed: int) -> list[float]:
    groups = defaultdict(list)
    for row in rows:
        groups[row["doc_id"]].append(row)
    docs = sorted(groups)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(N_BOOT):
        sample = rng.integers(0, len(docs), len(docs))
        sampled = [row for index in sample for row in groups[docs[index]]]
        values.append(float(stat(sampled)))
    return values


def arm_result(rows: list[dict], selector, seed: int) -> dict:
    selected = [row for row in rows if usable(row) and selector(row)]
    base, arm = fve(selected, "cos_orig"), fve(selected, "cos_edit")
    boots = _cluster_boot(selected, lambda sample: fve(sample, "cos_edit") - fve(sample, "cos_orig"), seed)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return dict(n=len(selected), n_docs=len({row["doc_id"] for row in selected}),
                FVE_base=round(base, 5), FVE_arm=round(arm, 5), dFVE=round(arm - base, 5),
                dFVE_ci95=[round(float(lo), 5), round(float(hi), 5)])


def contrast(rows_a: list[dict], rows_b: list[dict], selector, seed: int,
             label: str) -> dict:
    by_b = {row["idx"]: row for row in rows_b if usable(row)}
    paired = []
    for row_a in rows_a:
        row_b = by_b.get(row_a["idx"])
        if not usable(row_a) or row_b is None or not selector(row_a):
            continue
        paired.append(dict(
            doc_id=row_a["doc_id"], D_i=row_a["D_i"],
            cos_a=row_a["cos_edit"], cos_b=row_b["cos_edit"],
        ))

    def paired_fve(rows: list[dict], field: str) -> float:
        denominator = sum(float(row["D_i"]) for row in rows)
        return 1.0 - sum(2.0 * (1.0 - float(row[field])) for row in rows) / denominator

    value = paired_fve(paired, "cos_a") - paired_fve(paired, "cos_b")
    boots = _cluster_boot(
        paired, lambda sample: paired_fve(sample, "cos_a") - paired_fve(sample, "cos_b"), seed)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return dict(label=label, n=len(paired), n_docs=len({row["doc_id"] for row in paired}),
                dFVE=round(value, 5), dFVE_ci95=[round(float(lo), 5), round(float(hi), 5)])


def main() -> int:
    output = {}
    for model_index, key in enumerate(KEYS):
        arms = {arm: load(arm, key) for arm in (
            "exp5a", "exp5b", "exp5c", "exp6a", "exp6b", "exp6p", "exp6t")}
        seed = 9100 + 100 * model_index
        output[key] = {
            "exp5b_full": arm_result(arms["exp5b"], lambda _row: True, seed + 1),
            "exp5c_full": arm_result(arms["exp5c"], lambda _row: True, seed + 2),
            "exp5c_minus_exp5a": contrast(
                arms["exp5c"], arms["exp5a"], lambda _row: True, seed + 3, "5c - 5a"),
            "exp6p_full": arm_result(arms["exp6p"], lambda _row: True, seed + 4),
            "exp6p_changed": arm_result(arms["exp6p"], changed, seed + 5),
            "exp6p_high_div_changed": arm_result(
                arms["exp6p"], lambda row: changed(row) and high_div(row), seed + 6),
            "exp6p_minus_a2_full": contrast(
                arms["exp6p"], arms["exp6b"], lambda _row: True, seed + 7, "P - A2"),
            "exp6p_minus_a2_high_div_changed": contrast(
                arms["exp6p"], arms["exp6b"],
                lambda row: changed(row) and high_div(row), seed + 8, "P - A2, high-div changed"),
            "exp6t_full": arm_result(arms["exp6t"], lambda _row: True, seed + 9),
            "exp6t_wrong": arm_result(arms["exp6t"], continuation_wrong, seed + 10),
        }
    (RES / "remaining_contrasts.json").write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
