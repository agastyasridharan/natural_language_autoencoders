from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from analysis.groundedness_report.gpu_ood_static import _slug, _trajectory_task_id
from analysis.groundedness_report.io import atomic_json, atomic_jsonl, sha256_file
from analysis.groundedness_report.spec import PROTOCOL_SHA256
from analysis.groundedness_report.tau2_finalize import select_batches


def _task(task_id: str, rank: int) -> dict:
    row = {
        "schema_version": 1,
        "protocol_sha256": PROTOCOL_SHA256,
        "replication_label": "test",
        "family_id": "capability_tau2_retail",
        "task_id": task_id,
        "interaction_kind": "test",
        "source": {"sha256_rank_zero_based": rank},
    }
    row["task_sha256"] = hashlib.sha256(
        json.dumps(row, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return row


class Tau2FinalizeTests(unittest.TestCase):
    def test_embedded_task_id_is_supported_by_static_assembler(self):
        self.assertEqual(_trajectory_task_id({"task": {"task_id": "7"}}), "7")

    def test_selects_valid_and_preserves_declared_exclusion(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tasks = [_task("keep", 0), _task("drop", 1)]
            task_path = root / "tasks.jsonl"
            atomic_jsonl(task_path, tasks)
            run = root / "run"
            keep_slug = _slug(tasks[0])
            drop_slug = _slug(tasks[1])
            atomic_json(
                run / "trajectories/capability_tau2_retail" / f"{keep_slug}.json",
                {
                    "protocol_sha256": PROTOCOL_SHA256,
                    "status": "complete",
                    "family_id": "capability_tau2_retail",
                    "trajectory_id": "capability_tau2_retail:keep",
                    "task": tasks[0],
                },
            )
            failure_path = (
                run / "failures/capability_tau2_retail" / f"{drop_slug}.json"
            )
            atomic_json(
                failure_path,
                {
                    "protocol_sha256": PROTOCOL_SHA256,
                    "task_id": "drop",
                    "task": tasks[1],
                },
            )
            ledger_path = root / "ledger.json"
            atomic_json(
                ledger_path,
                {
                    "protocol_sha256": PROTOCOL_SHA256,
                    "exclusions": [
                        {
                            "task_id": "drop",
                            "failure_artifact_sha256": sha256_file(failure_path),
                        }
                    ],
                },
            )
            manifest = {
                "protocol_sha256": PROTOCOL_SHA256,
                "family_id": "capability_tau2_retail",
                "expected_trajectories": 1,
                "exclusion_ledger": "ledger.json",
                "batches": [
                    {
                        "name": "test",
                        "task_artifact": "tasks.jsonl",
                        "task_artifact_sha256": sha256_file(task_path),
                        "run_dir": "run",
                    }
                ],
            }
            selected, excluded, _ = select_batches(root=root, manifest=manifest)
        self.assertEqual([row["task"]["task_id"] for row in selected], ["keep"])
        self.assertEqual([row["task"]["task_id"] for row in excluded], ["drop"])

    def test_rejects_stale_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "stale protocol"):
                select_batches(
                    root=Path(temporary),
                    manifest={"protocol_sha256": "0" * 64},
                )


if __name__ == "__main__":
    unittest.main()
