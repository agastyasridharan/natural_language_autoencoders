from __future__ import annotations

import unittest

from analysis.groundedness_report.build_tau2_tasks import select_retail_tasks
from analysis.groundedness_report.sampling import sha256_rank


class Tau2TaskSelectionTests(unittest.TestCase):
    def test_selection_matches_protocol_ranking(self):
        rows = [{"id": str(index)} for index in range(12)]
        ranked = sha256_rank(
            rows,
            seed=17,
            namespace="ood-task:capability_tau2_retail",
            identity=lambda row: row["id"],
        )
        self.assertEqual(
            select_retail_tasks(rows, seed=17, rank_start=2, count=5), ranked[2:7]
        )

    def test_duplicate_ids_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            select_retail_tasks([{"id": "1"}, {"id": "1"}], seed=3, count=1)

    def test_out_of_range_interval_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            select_retail_tasks([{"id": "1"}], seed=3, rank_start=1, count=1)


if __name__ == "__main__":
    unittest.main()
