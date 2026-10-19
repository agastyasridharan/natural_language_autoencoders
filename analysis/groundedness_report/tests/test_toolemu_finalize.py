from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from analysis.groundedness_report.build_toolemu_tasks import (
    TOOLEMU_NONCYBER_AMENDMENT_SHA256,
)
from analysis.groundedness_report.gpu_ood_static import _slug
from analysis.groundedness_report.gpu_ood_toolemu import FAMILY_ID
from analysis.groundedness_report.io import atomic_json, sha256_file
from analysis.groundedness_report.spec import PROTOCOL_SHA256
from analysis.groundedness_report.toolemu_finalize import (
    _CALL_HASH_CHAIN,
    _HASH_CHAIN,
    select_canonical,
)


def _task(task_id: str, rank: int) -> dict:
    task = {
        "schema_version": 1,
        "protocol_sha256": PROTOCOL_SHA256,
        "toolemu_noncyber_scope_amendment_sha256": (
            TOOLEMU_NONCYBER_AMENDMENT_SHA256
        ),
        "replication_label": "test",
        "family_id": FAMILY_ID,
        "task_id": task_id,
        "interaction_kind": "tool_agent_trajectory",
        "toolkits": ["Test"],
        "source": {"sha256_rank_zero_based": rank},
    }
    task["task_sha256"] = hashlib.sha256(
        json.dumps(task, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return task


def _success(run: Path, task: dict, *, simulator_calls: int = 1) -> None:
    slug = _slug(task)
    relative_vector = Path("activations") / FAMILY_ID / f"{slug}-0.npy"
    vector_path = run / relative_vector
    vector_path.parent.mkdir(parents=True, exist_ok=True)
    vector_path.write_bytes(b"test-vector")
    audit_relative = (
        Path("tools") / FAMILY_ID / slug / "attempts" / "test" / "terra-emulator"
    )
    if simulator_calls:
        (run / audit_relative).mkdir(parents=True, exist_ok=True)
        atomic_json(
            run / audit_relative / "call-00.json",
            {"schema_version": 1, **_CALL_HASH_CHAIN},
        )
    atomic_json(
        run / "trajectories" / FAMILY_ID / f"{slug}.json",
        {
            "schema_version": 1,
            **_HASH_CHAIN,
            "status": "complete",
            "family_id": FAMILY_ID,
            "trajectory_id": f"{FAMILY_ID}:{task['task_id']}",
            "task": task,
            "activations": [
                {
                    "vector_file": str(relative_vector),
                    "vector_sha256": sha256_file(vector_path),
                }
            ],
            "toolemu": {
                "simulator_audit_dir": str(audit_relative),
                "simulator_call_audit_count": simulator_calls,
            },
        },
    )


def _failure(run: Path, task: dict, *, generic: bool = False) -> None:
    slug = _slug(task)
    audit_relative = (
        Path("tools") / FAMILY_ID / slug / "attempts" / "test" / "terra-emulator"
    )
    (run / audit_relative).mkdir(parents=True, exist_ok=True)
    atomic_json(
        run / "failures" / FAMILY_ID / f"{slug}.json",
        {
            "schema_version": 1,
            **_HASH_CHAIN,
            "family_id": FAMILY_ID,
            "task_id": task["task_id"],
            "task": task,
            "audit_dir": str(audit_relative),
            "failure_kind": "task_or_serialization"
            if generic
            else "no_eligible_assistant_token_after_minimum_context_rule",
            "error": "ValueError: parser defect"
            if generic
            else "ValueError: ToolEmu trajectory has no eligible assistant-language token",
        },
    )


class ToolEmuFinalizeTests(unittest.TestCase):
    def test_consumes_shortest_replacement_prefix(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            initial = [_task("initial-0", 0), _task("initial-1", 1)]
            replacements = [_task("replacement-0", 2), _task("replacement-1", 3)]
            _success(root / "initial", initial[0])
            _failure(root / "initial", initial[1])
            _success(root / "replacement", replacements[0])
            selected, excluded, consumed = select_canonical(
                initial_tasks=initial,
                replacement_tasks=replacements,
                initial_run=root / "initial",
                replacement_run=root / "replacement",
                expected=2,
            )
        self.assertEqual(
            [row["task"]["task_id"] for row in selected],
            ["initial-0", "replacement-0"],
        )
        self.assertEqual([row["task"]["task_id"] for row in excluded], ["initial-1"])
        self.assertEqual(consumed, 1)

    def test_rejects_unclassified_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            initial = [_task("initial-0", 0)]
            _failure(root / "initial", initial[0], generic=True)
            with self.assertRaisesRegex(ValueError, "not a frozen replaceable"):
                select_canonical(
                    initial_tasks=initial,
                    replacement_tasks=[],
                    initial_run=root / "initial",
                    replacement_run=root / "replacement",
                    expected=1,
                )

    def test_accepts_zero_call_direct_answer_without_tool_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            initial = [_task("direct-answer", 0)]
            _success(root / "initial", initial[0], simulator_calls=0)
            selected, excluded, consumed = select_canonical(
                initial_tasks=initial,
                replacement_tasks=[],
                initial_run=root / "initial",
                replacement_run=root / "replacement",
                expected=1,
            )
        self.assertEqual([row["task"]["task_id"] for row in selected], ["direct-answer"])
        self.assertEqual(excluded, [])
        self.assertEqual(consumed, 0)

    def test_rejects_replacement_processed_after_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            initial = [_task("initial-0", 0)]
            replacement = [_task("replacement-0", 1)]
            _success(root / "initial", initial[0])
            _success(root / "replacement", replacement[0])
            with self.assertRaisesRegex(ValueError, "after target was met"):
                select_canonical(
                    initial_tasks=initial,
                    replacement_tasks=replacement,
                    initial_run=root / "initial",
                    replacement_run=root / "replacement",
                    expected=1,
                )


if __name__ == "__main__":
    unittest.main()
