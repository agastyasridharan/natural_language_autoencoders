from __future__ import annotations

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1] / "fve_robustness"
sys.path.insert(0, str(HERE))

from inference_stats import exact_binomial_p_half, two_sided_t_pvalue, wilson_interval  # noqa: E402


def test_student_t_tail_reference_values():
    cases = [
        (0.0, 10, 1.0),
        (1.96, 100, 0.052778901366229654),
        (5.0, 30, 2.3296685467007762e-05),
        (20.0, 1999, 2.8942856239141775e-81),
    ]
    for t_value, df, expected in cases:
        actual = two_sided_t_pvalue(t_value, df)["p"]
        assert actual is not None
        assert math.isclose(actual, expected, rel_tol=2e-12, abs_tol=0.0)


def test_binomial_and_wilson_helpers():
    assert exact_binomial_p_half(12, 24) == 1.0
    assert math.isclose(exact_binomial_p_half(0, 24), 2 / 2**24)
    lo, hi = wilson_interval(0, 24)
    assert lo == 0.0
    assert 0.13 < hi < 0.14


def test_dashboard_inference_artifact_is_complete():
    data = json.loads((HERE / "results" / "dashboard_inference.json").read_text())
    assert set(data["baseline"]) == {"qwen7b", "gemma12b", "gemma27b", "llama70b"}
    assert len(data["arms"]) == 16
    for result in data["baseline"].values():
        assert len(result["FVE_ci95"]) == 2
        assert result["FVE_test"]["log10_p"] is not None
    for arm in data["arms"].values():
        assert set(arm) == {"qwen7b", "gemma12b", "gemma27b"}
        for result in arm.values():
            for key in ("FVE_base_ci95", "FVE_arm_ci95", "dFVE_ci95",
                        "mean_cos_base_ci95", "mean_cos_arm_ci95", "mean_dcos_ci95"):
                assert len(result[key]) == 2
            assert result["p_test"]["log10_p"] is not None
            assert result["p_test"]["log10_q"] is not None
            assert result["p_test"]["log10_q"] + 1e-12 >= result["p_test"]["log10_p"]


def test_exp7_inference_artifact_is_complete():
    data = json.loads((HERE / "results" / "exp7_metrics.json").read_text())
    for model in data.values():
        for tag in ("genre", "LCR", "PC"):
            for T in ("0.4", "0.7", "1.0", "1.5"):
                row = model["curves"][tag][T]
                assert row["core"][1] is not None
                assert row["delta_vs_null"]["p_test"]["log10_q"] is not None
            for name in ("delta_vs_floor", "delta_vs_shuffle", "delta_vs_null"):
                test = model["contrasts"][tag][name]["p_test"]
                if test["log10_p"] is not None:  # invalid no-injection floor can have zero paired docs
                    assert test["log10_q"] is not None
            for name in ("meaning", "surface", "meaning_minus_surface"):
                assert model["entropy_changes"][tag][name]["p_test"]["log10_q"] is not None
            assert model["greedy_in_core"][tag]["p_test_vs_one"]["log10_q"] is not None
        for arm in model["quality"].values():
            for row in arm.values():
                assert len(row["valid_rate_ci95"]) == 2
                assert len(row["docs_effective_rate_ci95"]) == 2
