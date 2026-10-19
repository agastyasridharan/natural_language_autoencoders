from __future__ import annotations

import unittest

from analysis.warm_start_robustness.claim_agreement.core import (
    ALIGNMENT_BENCHMARKS,
    CAPABILITY_BENCHMARKS,
)
from analysis.warm_start_robustness.claim_agreement.statistics import summarize_with_bootstrap


def score_row(
    activation: str,
    trajectory: str,
    family: str,
    benchmark: str,
    value: float,
    *,
    analysis: str = "complete_case",
) -> dict:
    return {
        "activation_id": activation,
        "distribution": "ood",
        "benchmark": benchmark,
        "family": family,
        "pair_type": "cross_initialization",
        "stratum": "all",
        "analysis": analysis,
        "document_id": None,
        "trajectory_id": trajectory,
        "pair_id": f"{activation}-{analysis}",
        "included": True,
        "complete": True,
        "p_left_to_right": value,
        "p_right_to_left": value,
        "a_cross": value,
        "failure_reason": None,
    }


class BootstrapTests(unittest.TestCase):
    def test_primary_macro_weights_16_benchmark_families_not_two_groups(self) -> None:
        # Alignment families are all 1 and capability families all 0. The primary equal-16
        # estimand is therefore 7/16; the explicitly secondary two-group macro is 1/2.
        rows = [
            score_row(f"a{i}", f"ta{i}", "alignment", benchmark, 1.0)
            for i, benchmark in enumerate(ALIGNMENT_BENCHMARKS)
        ] + [
            score_row(f"c{i}", f"tc{i}", "capability", benchmark, 0.0)
            for i, benchmark in enumerate(CAPABILITY_BENCHMARKS)
        ]
        summaries, arrays, metadata = summarize_with_bootstrap(rows, n_replicates=1000, seed=7)
        primary = next(record for record in summaries
                       if record["scope"] == "ood_equal_benchmark_family_macro")
        secondary = next(record for record in summaries
                         if record["scope"] == "ood_equal_alignment_capability_group_macro")
        self.assertAlmostEqual(primary["a_cross"], 7 / 16)
        self.assertEqual(primary["a_cross_ci95"], [7 / 16, 7 / 16])
        self.assertTrue(primary["primary_ood_estimand"])
        self.assertAlmostEqual(secondary["a_cross"], 0.5)
        self.assertFalse(secondary["primary_ood_estimand"])
        alignment = next(record for record in summaries
                         if record["scope"] == "ood_family" and record["scope_value"] == "alignment")
        self.assertEqual(alignment["n_clusters"], 7)
        self.assertEqual(metadata["bootstrap_unit_ood"], "trajectory_id")
        self.assertEqual(metadata["primary_ood_macro"], "equal_weight_across_16_benchmark_families")
        self.assertTrue(arrays)

    def test_bootstrap_is_deterministic(self) -> None:
        rows = [
            score_row("a0", "t0", "alignment", "blackmail", 0.0),
            score_row("a1", "t1", "alignment", "blackmail", 1.0),
            score_row("c0", "u0", "capability", "gpqa_diamond", 0.25),
        ]
        first, _, _ = summarize_with_bootstrap(rows, n_replicates=1000, seed=11)
        second, _, _ = summarize_with_bootstrap(rows, n_replicates=1000, seed=11)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
