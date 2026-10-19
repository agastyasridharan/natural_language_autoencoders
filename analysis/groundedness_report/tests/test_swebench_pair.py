from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from analysis.groundedness_report.gpu_ood_swebench import (
    ContainerShell,
    _is_infrastructure_failure,
    _podman,
    _select_tasks,
    _parse_shell_response,
    _validate_podman_paths,
)
from analysis.groundedness_report.io import atomic_json, atomic_jsonl, sha256_file
from analysis.groundedness_report.spec import PROTOCOL_SHA256
from analysis.groundedness_report.swebench_pair import (
    ALIGNMENT_FAMILY,
    CAPABILITY_FAMILY,
    _task_sha256,
    audit_output,
    build_plan,
    validate_pair,
    validate_storage_paths,
)


def _task(task_id: str, family_id: str) -> dict:
    value = {
        "schema_version": 1,
        "protocol_sha256": PROTOCOL_SHA256,
        "replication_label": "test-replication",
        "family_id": family_id,
        "task_id": task_id,
        "interaction_kind": "official_swebench_container_shell_agent",
        "messages": [{"role": "user", "content": f"Repair {task_id}"}],
        "source": {"dataset": "princeton-nlp/SWE-bench_Verified"},
        "container_image": f"docker.io/swebench/sweb.eval.x86_64.repo_1776_{task_id}:latest",
        "reference": {"instance_id": task_id, "base_commit": "abc"},
    }
    value["task_sha256"] = _task_sha256(value)
    return value


class PairValidationTests(unittest.TestCase):
    def _files(self, directory: Path, count: int = 2) -> tuple[Path, Path]:
        task_ids = [f"issue-{index:02d}" for index in range(count)]
        alignment = directory / f"{ALIGNMENT_FAMILY}.jsonl"
        capability = directory / f"{CAPABILITY_FAMILY}.jsonl"
        atomic_jsonl(alignment, [_task(task_id, ALIGNMENT_FAMILY) for task_id in task_ids])
        atomic_jsonl(capability, [_task(task_id, CAPABILITY_FAMILY) for task_id in task_ids])
        return alignment, capability

    def test_pair_validation_selects_explicit_lexicographic_smoke(self):
        with tempfile.TemporaryDirectory() as temporary:
            alignment, capability = self._files(Path(temporary))
            result = validate_pair(alignment, capability, expected_count=2)
        self.assertEqual(result["smoke_task_id"], "issue-00")
        self.assertEqual(result["paired_instances"], 2)
        self.assertEqual(result["unique_container_images"], 2)

    def test_pair_validation_rejects_any_noncondition_difference(self):
        with tempfile.TemporaryDirectory() as temporary:
            alignment, capability = self._files(Path(temporary))
            rows = [json.loads(line) for line in capability.read_text().splitlines()]
            rows[0]["messages"][0]["content"] = "different prompt"
            rows[0]["task_sha256"] = _task_sha256(rows[0])
            atomic_jsonl(capability, rows)
            with self.assertRaisesRegex(ValueError, "differs in fields.*messages"):
                validate_pair(alignment, capability, expected_count=2)

    def test_pair_validation_rejects_corrupt_task_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            alignment, capability = self._files(Path(temporary))
            rows = [json.loads(line) for line in alignment.read_text().splitlines()]
            rows[0]["task_sha256"] = "0" * 64
            atomic_jsonl(alignment, rows)
            with self.assertRaisesRegex(ValueError, "invalid task_sha256"):
                validate_pair(alignment, capability, expected_count=2)

    def test_plan_is_two_sequential_paired_smokes_then_four_parallel_shards(self):
        with tempfile.TemporaryDirectory() as temporary:
            alignment, capability = self._files(Path(temporary))
            plan = build_plan(
                alignment_tasks=alignment,
                capability_tasks=capability,
                smoke_output="/data/agastyas/out-smoke",
                full_output="/data/agastyas/out-full",
                graphroot="/data/agastyas/podman-sweb7/graphroot",
                runroot="/data/agastyas/podrun-sweb7",
                tmpdir="/data/agastyas/podtmp-sweb7",
                python="/data/agastyas/Miniconda3/bin/python",
                gpu_indices=["0", "1", "2", "3"],
                expected_count=2,
            )
        smokes = plan["smoke_commands_sequential"]
        self.assertEqual(len(smokes), 2)
        self.assertTrue(all("--task-id" in row["argv"] for row in smokes))
        self.assertTrue(all("issue-00" in row["argv"] for row in smokes))
        self.assertTrue(all(row["argv"].count("--podman-root") == 1 for row in smokes))
        full = plan["full_commands_parallel"]
        self.assertEqual([row["cuda_visible_devices"] for row in full], ["0", "1", "2", "3"])
        self.assertTrue(all(row["argv"][row["argv"].index("--num-shards") + 1] == "2" for row in full))
        self.assertTrue(all(row["argv"].count("--podman-root") == 1 for row in full))


class RunnerPreflightTests(unittest.TestCase):
    def test_only_environment_failures_are_retryable(self):
        self.assertTrue(_is_infrastructure_failure("RuntimeError: podman pull failed: x"))
        self.assertTrue(_is_infrastructure_failure("TimeoutExpired: command timed out"))
        self.assertFalse(_is_infrastructure_failure(
            "ValueError: assistant did not give a final answer within 20 turns"
        ))

    def test_podman_capture_replaces_non_utf8_bytes(self):
        result = _podman(
            [
                sys.executable,
                "-c",
                "import sys; sys.stdout.buffer.write(bytes([0xe0]))",
            ],
            timeout=5,
        )
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "\ufffd")

    def test_athena_storage_paths_reject_home_and_long_runroot(self):
        with self.assertRaisesRegex(ValueError, "under /data"):
            validate_storage_paths("/home/user/graph", "/data/user/run", "/data/user/tmp")
        with self.assertRaisesRegex(ValueError, "at most 50"):
            validate_storage_paths(
                "/data/user/graph", "/data/user/" + "r" * 50, "/data/user/tmp"
            )
        validate_storage_paths(
            "/data/agastyas/podman-sweb7/graphroot",
            "/data/agastyas/podrun-sweb7",
            "/data/agastyas/podtmp-sweb7",
        )

    def test_runner_rejects_long_runroot_before_podman(self):
        with self.assertRaisesRegex(ValueError, "at most 50"):
            _validate_podman_paths(
                Path("/tmp/graph"), Path("/tmp/" + "x" * 50), Path("/tmp/image-tmp")
            )

    def test_explicit_task_id_is_exact_and_incompatible_with_sharding(self):
        rows = [_task(f"issue-{index:02d}", ALIGNMENT_FAMILY) for index in range(8)]
        selected = _select_tasks(
            rows, num_shards=1, shard_index=0, limit=None, task_id="issue-03"
        )
        self.assertEqual([row["task_id"] for row in selected], ["issue-03"])
        with self.assertRaisesRegex(ValueError, "requires --num-shards 1"):
            _select_tasks(
                rows, num_shards=2, shard_index=0, limit=None, task_id="issue-03"
            )

    def test_malformed_tool_call_is_returned_as_auditable_parse_error(self):
        calls, error = _parse_shell_response(
            '<tool_call>{"name":"run_shell","arguments":{"command":"rg \\s"}}</tool_call>'
        )
        self.assertEqual(calls, [])
        self.assertIn("JSONDecodeError", error)
        calls, error = _parse_shell_response(
            '<tool_call>{"name":"run_shell","arguments":{"command":"pwd"}}</tool_call>'
        )
        self.assertEqual(calls[0]["arguments"]["command"], "pwd")
        self.assertIsNone(error)
        calls, error = _parse_shell_response(
            '<tool_call>{"name":"run_shell","arguments":{"command":"pwd"}}</tool_call>'
            '<tool_call>{"name":"run_shell","arguments":{"command":"ls"}}</tool_call>'
        )
        self.assertEqual(calls, [])
        self.assertIn("more than one shell call", error)

    def test_shards_are_disjoint_and_complete(self):
        rows = [_task(f"issue-{index:02d}", ALIGNMENT_FAMILY) for index in range(50)]
        shard_zero = _select_tasks(
            rows, num_shards=2, shard_index=0, limit=None, task_id=None
        )
        shard_one = _select_tasks(
            rows, num_shards=2, shard_index=1, limit=None, task_id=None
        )
        ids_zero = {row["task_id"] for row in shard_zero}
        ids_one = {row["task_id"] for row in shard_one}
        self.assertFalse(ids_zero & ids_one)
        self.assertEqual(ids_zero | ids_one, {row["task_id"] for row in rows})

    def test_container_records_immutable_image_identity_and_disables_container_gpu(self):
        completed = subprocess.CompletedProcess
        calls: list[tuple[str, ...]] = []

        def fake_podman(_base, *arguments, timeout):
            del timeout
            calls.append(arguments)
            if arguments[:2] == ("image", "exists"):
                return completed([], 0, "", "")
            if arguments[:2] == ("image", "inspect"):
                return completed([], 0, json.dumps([{
                    "Id": "sha256:immutable",
                    "Digest": "sha256:manifest",
                    "RepoDigests": ["example@sha256:manifest"],
                    "Created": "2026-01-01T00:00:00Z",
                    "Size": 123,
                }]), "")
            return completed([], 0, "container-id", "")

        with tempfile.TemporaryDirectory(dir="/tmp", prefix="sweb-") as temporary:
            root = Path(temporary)
            graph = root / "g"
            runroot = root / "r"
            image_tmp = root / "t"
            shell = ContainerShell(
                podman_root=graph,
                podman_runroot=runroot,
                podman_tmpdir=image_tmp,
                image="example:latest",
                name="test-container",
                output_dir=root / "output",
                timeout_seconds=5,
                output_characters=100,
            )
            self.assertEqual(
                shell.base[-2:],
                ["--storage-opt", "overlay.ignore_chown_errors=true"],
            )
            with patch(
                "analysis.groundedness_report.gpu_ood_swebench._podman",
                side_effect=fake_podman,
            ):
                shell.start()
                shell.close()
            image_record = json.loads((root / "output" / "image.json").read_text())
        self.assertEqual(image_record["id"], "sha256:immutable")
        run_call = next(call for call in calls if call and call[0] == "run")
        self.assertIn("CUDA_VISIBLE_DEVICES=", run_call)
        self.assertIn("NVIDIA_VISIBLE_DEVICES=void", run_call)


class OutputAuditTests(unittest.TestCase):
    def _trajectory(
        self,
        root: Path,
        family: str,
        *,
        image_id: str = "sha256:" + "a" * 64,
    ) -> None:
        vector_file = Path("activations") / family / "vector.npy"
        (root / vector_file).parent.mkdir(parents=True, exist_ok=True)
        np.save(root / vector_file, np.asarray([1.0, 2.0], dtype=np.float32))
        atomic_json(root / "trajectories" / family / "trajectory.json", {
            "protocol_sha256": PROTOCOL_SHA256,
            "status": "complete",
            "family_id": family,
            "task_id": "issue-00",
            "container_image": "example:latest",
            "container_image_metadata": {
                "id": image_id, "requested_reference": "example:latest",
            },
            "container_runtime": {
                "engine": "rootless_podman",
                "graphroot": "/data/test/graph",
                "runroot": "/data/test/run",
                "image_copy_tmpdir": "/data/test/tmp",
                "network": "none",
                "container_gpu_visible": False,
            },
            "turns": [{"tool_calls": [], "response": "Done"}],
            "tool_calls": [],
            "invalid_tool_call_attempts": 0,
            "repository_state": {"returncode": 0, "stdout_sha256": "a" * 64},
            "activations": [{
                "vector_file": str(vector_file),
                "vector_sha256": sha256_file(root / vector_file),
            }],
        })
        atomic_json(root / f"{family}-shard-00-meta.json", {
            "status": "done",
            "failures": [],
            "assigned_task_ids": ["issue-00"],
        })

    def test_output_audit_requires_exact_paired_immutable_images(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._trajectory(root, ALIGNMENT_FAMILY)
            self._trajectory(root, CAPABILITY_FAMILY)
            result = audit_output(root, expected_per_family=1)
            self.assertEqual(result["paired_instances"], 1)
            capability = root / "trajectories" / CAPABILITY_FAMILY / "trajectory.json"
            row = json.loads(capability.read_text())
            row["container_image_metadata"]["id"] = "sha256:" + "b" * 64
            atomic_json(capability, row)
            with self.assertRaisesRegex(ValueError, "immutable image IDs differ"):
                audit_output(root, expected_per_family=1)

    def test_output_audit_accepts_podman_raw_hex_image_id(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_id = "c" * 64
            self._trajectory(root, ALIGNMENT_FAMILY, image_id=image_id)
            self._trajectory(root, CAPABILITY_FAMILY, image_id=image_id)
            result = audit_output(root, expected_per_family=1)
            self.assertEqual(result["paired_instances"], 1)

    def test_output_audit_rejects_non_digest_image_id(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._trajectory(root, ALIGNMENT_FAMILY, image_id="not-a-digest")
            self._trajectory(root, CAPABILITY_FAMILY, image_id="not-a-digest")
            with self.assertRaisesRegex(ValueError, "missing immutable container image ID"):
                audit_output(root, expected_per_family=1)


if __name__ == "__main__":
    unittest.main()
