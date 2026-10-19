from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from analysis.groundedness_report.sampling import sha256_rank
from analysis.groundedness_report.swebench_replacements import archived_rows, replacement_rows


class SwebenchReplacementTests(unittest.TestCase):
    def test_archived_rows_restore_source_offsets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "000100.json").write_text(json.dumps({"rows": [
                {"row": {"instance_id": "b"}},
                {"row": {"instance_id": "a"}},
            ]}))
            rows = archived_rows(root)
        self.assertEqual([row["source_row"] for row in rows], [100, 101])

    def test_replacements_follow_verified_rank_after_initial_pool(self):
        rows = [{"instance_id": f"issue-{index}"} for index in range(8)]
        ranked = sha256_rank(
            rows,
            seed=7,
            namespace="ood-task:capability_swebench_verified",
            identity=lambda row: row["instance_id"],
        )
        self.assertEqual(
            replacement_rows(rows, ranked[:3], seed=7, count=2, rank_start=3),
            ranked[3:5],
        )

    def test_replacements_reject_noncanonical_initial_pool(self):
        rows = [{"instance_id": f"issue-{index}"} for index in range(5)]
        with self.assertRaisesRegex(ValueError, "does not reproduce"):
            replacement_rows(rows, rows[:2], seed=7, count=1, rank_start=2)


if __name__ == "__main__":
    unittest.main()
