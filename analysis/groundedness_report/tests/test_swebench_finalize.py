from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from analysis.groundedness_report.io import atomic_json, atomic_jsonl
from analysis.groundedness_report.spec import PROTOCOL_SHA256
from analysis.groundedness_report.swebench_finalize import select_batches
from analysis.groundedness_report.swebench_pair import ALIGNMENT_FAMILY, CAPABILITY_FAMILY


class SwebenchFinalizeTests(unittest.TestCase):
    def test_select_batches_excludes_pairs_and_rejects_duplicates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tasks = root / "tasks"
            for family in (ALIGNMENT_FAMILY, CAPABILITY_FAMILY):
                atomic_jsonl(tasks / f"{family}.jsonl", [
                    {"family_id": family, "task_id": "keep"},
                    {"family_id": family, "task_id": "drop"},
                ])
            atomic_json(root / "ledger.json", {
                "protocol_sha256": PROTOCOL_SHA256,
                "paired_excluded_task_ids": ["drop"],
            })
            manifest = {
                "protocol_sha256": PROTOCOL_SHA256,
                "batches": [{
                    "name": "test",
                    "task_dir": "tasks",
                    "run_dir": "run",
                    "exclusion_ledger": "ledger.json",
                }],
            }
            selected, provenance = select_batches(root=root, manifest=manifest)
        self.assertEqual([row["task_id"] for row in selected[ALIGNMENT_FAMILY]], ["keep"])
        self.assertEqual(provenance[0]["excluded_task_ids"], ["drop"])

    def test_select_batches_rejects_stale_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "stale protocol"):
                select_batches(
                    root=Path(temporary),
                    manifest={"protocol_sha256": "0" * 64, "batches": []},
                )


if __name__ == "__main__":
    unittest.main()
