from __future__ import annotations

import unittest

from analysis.groundedness_report.build_toolemu_tasks import (
    TOOLEMU_NONCYBER_EXCLUDED_TASK_IDS,
    select_toolemu_cases,
)
from analysis.groundedness_report.sampling import sha256_rank


class ToolEmuTaskSelectionTests(unittest.TestCase):
    def test_selection_matches_protocol_ranking(self):
        rows = [{"name": f"official_{index}"} for index in range(60)]
        expected = sha256_rank(
            rows,
            seed=17,
            namespace="ood-task:alignment_overly_agentic",
            identity=lambda row: row["name"],
        )[3:8]
        observed = select_toolemu_cases(rows, seed=17, rank_start=3, count=5)
        self.assertEqual(observed, expected)

    def test_duplicate_names_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            select_toolemu_cases(
                [{"name": "official_0"}, {"name": "official_0"}],
                seed=1,
                count=1,
            )

    def test_out_of_range_interval_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            select_toolemu_cases(
                [{"name": "official_0"}], seed=1, rank_start=1, count=1
            )

    def test_noncyber_exclusions_are_replaced_by_next_ranked_cases(self):
        rows = [{"name": f"official_{index}"} for index in range(144)]
        ranked = sha256_rank(
            rows,
            seed=20260827,
            namespace="ood-task:alignment_overly_agentic",
            identity=lambda row: row["name"],
        )
        expected = [
            row for row in ranked
            if row["name"] not in TOOLEMU_NONCYBER_EXCLUDED_TASK_IDS
        ][:50]
        observed = select_toolemu_cases(rows, seed=20260827, count=50)
        self.assertEqual(observed, expected)
        self.assertTrue(set(TOOLEMU_NONCYBER_EXCLUDED_TASK_IDS).isdisjoint(
            row["name"] for row in observed
        ))


if __name__ == "__main__":
    unittest.main()
