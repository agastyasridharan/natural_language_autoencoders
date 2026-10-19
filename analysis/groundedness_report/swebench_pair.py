"""Validate and plan the paired SWE-bench / destructive-coding container run.

This module is deliberately CPU-only.  It freezes the exact paired smoke and
full-run commands, and can inspect an explicitly located rootless Podman store,
but it never pulls an image, starts a container, or imports the target model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from .io import atomic_json, read_json, read_jsonl, sha256_file
from .spec import PROTOCOL_SHA256


ALIGNMENT_FAMILY = "alignment_destructive_coding"
CAPABILITY_FAMILY = "capability_swebench_verified"
FAMILIES = (ALIGNMENT_FAMILY, CAPABILITY_FAMILY)


def _task_sha256(task: dict[str, Any]) -> str:
    payload = dict(task)
    payload.pop("task_sha256", None)
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _validate_task_file(
    path: Path,
    *,
    family_id: str,
    expected_count: int,
) -> list[dict[str, Any]]:
    rows = list(read_jsonl(path))
    if len(rows) != expected_count:
        raise ValueError(f"{path} must contain exactly {expected_count} tasks; found {len(rows)}")
    task_ids = [str(row.get("task_id")) for row in rows]
    if task_ids != sorted(task_ids):
        raise ValueError(f"{path} is not sorted by task_id")
    if len(set(task_ids)) != len(task_ids):
        raise ValueError(f"{path} contains duplicate task IDs")
    for row in rows:
        if row.get("family_id") != family_id:
            raise ValueError(f"{path} contains a task outside {family_id}")
        if row.get("protocol_sha256") != PROTOCOL_SHA256:
            raise ValueError(f"{path} contains a stale protocol hash")
        if row.get("interaction_kind") != "official_swebench_container_shell_agent":
            raise ValueError(f"{path} contains a non-container interaction")
        if (row.get("reference") or {}).get("instance_id") != row.get("task_id"):
            raise ValueError(f"{path} has a task/reference instance mismatch")
        image = str(row.get("container_image") or "")
        if not image.startswith("docker.io/swebench/sweb.eval.x86_64."):
            raise ValueError(f"{path} contains a non-official SWE-bench image")
        if row.get("task_sha256") != _task_sha256(row):
            raise ValueError(f"{path} contains an invalid task_sha256 for {row.get('task_id')}")
    return rows


def validate_pair(
    alignment_tasks: str | Path,
    capability_tasks: str | Path,
    *,
    expected_count: int = 50,
) -> dict[str, Any]:
    """Validate exact instance-level pairing and frozen artifact metadata."""
    alignment_path = Path(alignment_tasks)
    capability_path = Path(capability_tasks)
    alignment = _validate_task_file(
        alignment_path, family_id=ALIGNMENT_FAMILY, expected_count=expected_count
    )
    capability = _validate_task_file(
        capability_path, family_id=CAPABILITY_FAMILY, expected_count=expected_count
    )
    alignment_by_id = {str(row["task_id"]): row for row in alignment}
    capability_by_id = {str(row["task_id"]): row for row in capability}
    if alignment_by_id.keys() != capability_by_id.keys():
        only_alignment = sorted(alignment_by_id.keys() - capability_by_id.keys())
        only_capability = sorted(capability_by_id.keys() - alignment_by_id.keys())
        raise ValueError(
            "paired task IDs differ: "
            f"alignment_only={only_alignment}, capability_only={only_capability}"
        )
    paired_fields = (
        "schema_version",
        "protocol_sha256",
        "replication_label",
        "task_id",
        "interaction_kind",
        "messages",
        "source",
        "container_image",
        "reference",
    )
    for task_id in sorted(alignment_by_id):
        left = alignment_by_id[task_id]
        right = capability_by_id[task_id]
        mismatched = [field for field in paired_fields if left.get(field) != right.get(field)]
        if mismatched:
            raise ValueError(f"paired task {task_id} differs in fields {mismatched}")

    task_hashes = {
        ALIGNMENT_FAMILY: sha256_file(alignment_path),
        CAPABILITY_FAMILY: sha256_file(capability_path),
    }
    if alignment_path.parent == capability_path.parent:
        meta_path = alignment_path.parent / "task_meta.json"
        if meta_path.exists():
            meta = read_json(meta_path)
            if meta.get("protocol_sha256") != PROTOCOL_SHA256:
                raise ValueError("paired task_meta.json has a stale protocol hash")
            if meta.get("paired_instances") != expected_count:
                raise ValueError("paired task_meta.json has the wrong instance count")
            if meta.get("task_artifact_sha256") != task_hashes:
                raise ValueError("paired task_meta.json hashes do not match the task files")

    smoke_task_id = min(alignment_by_id)
    return {
        "schema_version": 1,
        "protocol_sha256": PROTOCOL_SHA256,
        "paired_instances": expected_count,
        "task_artifact_sha256": task_hashes,
        "smoke_task_id": smoke_task_id,
        "smoke_container_image": alignment_by_id[smoke_task_id]["container_image"],
        "unique_container_images": len({row["container_image"] for row in alignment}),
    }


def validate_storage_paths(
    graphroot: str | Path,
    runroot: str | Path,
    tmpdir: str | Path,
) -> None:
    """Enforce Athena-safe storage and Podman 4.9's runroot length limit."""
    graph = PurePosixPath(str(graphroot))
    run = PurePosixPath(str(runroot))
    temporary = PurePosixPath(str(tmpdir))
    if not all(path.is_absolute() for path in (graph, run, temporary)):
        raise ValueError("Podman graphroot, runroot, and image-copy tmpdir must be absolute")
    data = PurePosixPath("/data")
    if not all(path.is_relative_to(data) for path in (graph, run, temporary)):
        raise ValueError(
            "Athena Podman graphroot, runroot, and image-copy tmpdir must all be under /data"
        )
    if len({graph, run, temporary}) != 3:
        raise ValueError("Podman graphroot, runroot, and image-copy tmpdir must be distinct")
    if len(os.fsencode(str(run))) > 50:
        raise ValueError("Podman runroot must be at most 50 bytes")


def inspect_podman_store(
    graphroot: str | Path,
    runroot: str | Path,
    tmpdir: str | Path,
) -> dict[str, Any]:
    """Initialize and inspect only the explicit rootless store; start nothing."""
    validate_storage_paths(graphroot, runroot, tmpdir)
    graph = Path(graphroot)
    run = Path(runroot)
    temporary = Path(tmpdir)
    graph.mkdir(parents=True, exist_ok=True)
    run.mkdir(parents=True, exist_ok=True)
    temporary.mkdir(parents=True, exist_ok=True)
    base = ["podman", "--root", str(graph), "--runroot", str(run)]
    result = subprocess.run(
        [*base, "info", "--format", "json"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=120,
        env={**os.environ, "TMPDIR": str(temporary)},
    )
    if result.returncode != 0:
        raise RuntimeError(f"podman info failed: {result.stderr[-2000:]}")
    try:
        info = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"podman info returned invalid JSON: {error}") from error
    host = info.get("host") or {}
    store = info.get("store") or {}
    if (host.get("security") or {}).get("rootless") is not True:
        raise ValueError("Podman store is not rootless")
    if store.get("graphRoot") != str(graph) or store.get("runRoot") != str(run):
        raise ValueError("Podman did not honor the explicit graphroot/runroot")
    if store.get("imageCopyTmpDir") != str(temporary):
        raise ValueError("Podman did not honor the /data image-copy tmpdir")
    return {
        "version": (info.get("version") or {}).get("Version"),
        "rootless": True,
        "graph_root": store.get("graphRoot"),
        "run_root": store.get("runRoot"),
        "image_copy_tmpdir": store.get("imageCopyTmpDir"),
        "graph_driver": store.get("graphDriverName"),
        "backing_filesystem": (store.get("graphStatus") or {}).get("Backing Filesystem"),
        "images": (store.get("imageStore") or {}).get("number"),
        "containers": (store.get("containerStore") or {}).get("number"),
    }


def audit_output(
    output: str | Path,
    *,
    expected_per_family: int,
    expected_tasks: dict[str, dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Fail closed on incomplete, unpaired, or internally corrupt run artifacts."""
    if expected_per_family < 1:
        raise ValueError("expected_per_family must be positive")
    root = Path(output)
    failure_files = sorted((root / "failures").glob("*.json"))
    cleanup_failures = sorted((root / "tools").glob("**/cleanup-failure.json"))
    if failure_files or cleanup_failures:
        raise ValueError(
            f"run contains {len(failure_files)} task failures and "
            f"{len(cleanup_failures)} cleanup failures"
        )
    trajectories = [
        read_json(path) for path in sorted((root / "trajectories").glob("*/*.json"))
    ]
    expected_total = expected_per_family * len(FAMILIES)
    if len(trajectories) != expected_total:
        raise ValueError(f"expected {expected_total} trajectories; found {len(trajectories)}")
    by_family: dict[str, dict[str, dict[str, Any]]] = {family: {} for family in FAMILIES}
    activation_count = 0
    invalid_tool_call_attempts: dict[str, int] = {family: 0 for family in FAMILIES}
    vector_widths: set[int] = set()
    for trajectory in trajectories:
        family = str(trajectory.get("family_id"))
        task_id = str(trajectory.get("task_id"))
        if family not in by_family:
            raise ValueError(f"unexpected family in trajectory: {family}")
        if task_id in by_family[family]:
            raise ValueError(f"duplicate trajectory for {family}:{task_id}")
        if trajectory.get("protocol_sha256") != PROTOCOL_SHA256:
            raise ValueError(f"stale trajectory protocol for {family}:{task_id}")
        if trajectory.get("status") != "complete":
            raise ValueError(f"incomplete trajectory for {family}:{task_id}")
        image = trajectory.get("container_image_metadata") or {}
        if re.fullmatch(r"(?:sha256:)?[0-9a-f]{64}", str(image.get("id") or "")) is None:
            raise ValueError(f"missing immutable container image ID for {family}:{task_id}")
        if image.get("requested_reference") != trajectory.get("container_image"):
            raise ValueError(f"container image identity/reference mismatch for {family}:{task_id}")
        runtime = trajectory.get("container_runtime") or {}
        if (
            runtime.get("engine") != "rootless_podman"
            or runtime.get("network") != "none"
            or runtime.get("container_gpu_visible") is not False
            or not str(runtime.get("graphroot") or "").startswith("/data/")
            or not str(runtime.get("runroot") or "").startswith("/data/")
            or not str(runtime.get("image_copy_tmpdir") or "").startswith("/data/")
        ):
            raise ValueError(f"unsafe or unrecorded container runtime for {family}:{task_id}")
        turns = trajectory.get("turns") or []
        if not 1 <= len(turns) <= 20:
            raise ValueError(f"invalid assistant turn count for {family}:{task_id}")
        if any(len(turn.get("tool_calls") or []) > 1 for turn in turns):
            raise ValueError(f"multiple shell calls in one turn for {family}:{task_id}")
        if (
            turns[-1].get("tool_calls")
            or turns[-1].get("tool_call_error")
            or not str(turns[-1].get("response") or "").strip()
        ):
            raise ValueError(f"missing final non-tool assistant answer for {family}:{task_id}")
        invalid_attempts = sum("tool_call_error" in turn for turn in turns)
        if trajectory.get("invalid_tool_call_attempts") != invalid_attempts:
            raise ValueError(f"invalid-tool-call count mismatch for {family}:{task_id}")
        if any(
            turn.get("tool_call_error") and turn.get("tool_calls")
            for turn in turns
        ):
            raise ValueError(f"invalid tool attempt also executed a call for {family}:{task_id}")
        invalid_tool_call_attempts[family] += invalid_attempts
        calls = trajectory.get("tool_calls") or []
        if len(calls) > 20 or [call.get("call_index") for call in calls] != list(range(len(calls))):
            raise ValueError(f"invalid shell-call sequence for {family}:{task_id}")
        if sum(len(turn.get("tool_calls") or []) for turn in turns) != len(calls):
            raise ValueError(f"turn/tool-call record mismatch for {family}:{task_id}")
        repository_state = trajectory.get("repository_state") or {}
        if repository_state.get("returncode") != 0 or not repository_state.get("stdout_sha256"):
            raise ValueError(f"invalid final repository state for {family}:{task_id}")
        activations = trajectory.get("activations") or []
        if not 1 <= len(activations) <= 4:
            raise ValueError(f"invalid activation count for {family}:{task_id}")
        for activation in activations:
            relative = Path(str(activation.get("vector_file") or ""))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"unsafe activation path for {family}:{task_id}")
            vector_path = root / relative
            if not vector_path.is_file() or sha256_file(vector_path) != activation.get("vector_sha256"):
                raise ValueError(f"missing or corrupt activation for {family}:{task_id}")
            vector = np.load(vector_path, allow_pickle=False)
            if vector.ndim != 1 or vector.dtype != np.float32 or not bool(np.isfinite(vector).all()):
                raise ValueError(f"invalid activation vector for {family}:{task_id}")
            vector_widths.add(int(vector.shape[0]))
        activation_count += len(activations)
        by_family[family][task_id] = trajectory
    counts = {family: len(rows) for family, rows in by_family.items()}
    if any(count != expected_per_family for count in counts.values()):
        raise ValueError(f"incorrect family counts: {counts}")
    if len(vector_widths) != 1:
        raise ValueError(f"activation widths are inconsistent: {sorted(vector_widths)}")
    if by_family[ALIGNMENT_FAMILY].keys() != by_family[CAPABILITY_FAMILY].keys():
        raise ValueError("completed trajectory task IDs are not exactly paired")
    if expected_tasks is not None:
        for family in FAMILIES:
            observed = {
                task_id: str(row.get("task_sha256"))
                for task_id, row in by_family[family].items()
            }
            if observed != expected_tasks[family]:
                raise ValueError(f"completed trajectories do not match frozen tasks for {family}")
    for task_id in by_family[ALIGNMENT_FAMILY]:
        left = by_family[ALIGNMENT_FAMILY][task_id]
        right = by_family[CAPABILITY_FAMILY][task_id]
        if left.get("container_image") != right.get("container_image"):
            raise ValueError(f"paired trajectory image references differ for {task_id}")
        left_id = (left.get("container_image_metadata") or {}).get("id")
        right_id = (right.get("container_image_metadata") or {}).get("id")
        if left_id != right_id:
            raise ValueError(f"paired trajectory immutable image IDs differ for {task_id}")
    for family in FAMILIES:
        meta_paths = sorted(root.glob(f"{family}-shard-*-meta.json"))
        if not meta_paths:
            raise ValueError(f"missing completed shard metadata for {family}")
        assigned: list[str] = []
        for meta_path in meta_paths:
            meta = read_json(meta_path)
            if meta.get("status") != "done" or meta.get("failures"):
                raise ValueError(f"incomplete or failed shard metadata: {meta_path}")
            assigned.extend(str(task_id) for task_id in meta.get("assigned_task_ids") or [])
        if len(assigned) != len(set(assigned)) or set(assigned) != set(by_family[family]):
            raise ValueError(f"shard assignment metadata does not cover {family} exactly once")
    return {
        "status": "passed",
        "protocol_sha256": PROTOCOL_SHA256,
        "family_trajectories": counts,
        "paired_instances": expected_per_family,
        "activations": activation_count,
        "activation_width": next(iter(vector_widths)),
        "invalid_tool_call_attempts": invalid_tool_call_attempts,
    }


def _command(
    *,
    python: str,
    tasks: Path,
    output: Path,
    graphroot: Path,
    runroot: Path,
    tmpdir: Path,
    gpu: str,
    num_shards: int = 1,
    shard_index: int = 0,
    task_id: str | None = None,
) -> dict[str, Any]:
    argv = [
        python,
        "-m",
        "analysis.groundedness_report.gpu_ood_swebench",
        "--tasks",
        str(tasks),
        "--out",
        str(output),
        "--podman-root",
        str(graphroot),
        "--podman-runroot",
        str(runroot),
        "--podman-tmpdir",
        str(tmpdir),
    ]
    if task_id is not None:
        argv.extend(["--task-id", task_id])
    if num_shards != 1 or shard_index != 0:
        argv.extend(["--num-shards", str(num_shards), "--shard-index", str(shard_index)])
    return {"cuda_visible_devices": gpu, "argv": argv, "shell": shlex.join(argv)}


def build_plan(
    *,
    alignment_tasks: str | Path,
    capability_tasks: str | Path,
    smoke_output: str | Path,
    full_output: str | Path,
    graphroot: str | Path,
    runroot: str | Path,
    tmpdir: str | Path,
    python: str,
    gpu_indices: list[str],
    expected_count: int = 50,
) -> dict[str, Any]:
    pair = validate_pair(alignment_tasks, capability_tasks, expected_count=expected_count)
    validate_storage_paths(graphroot, runroot, tmpdir)
    if len(gpu_indices) != 4 or len(set(gpu_indices)) != 4:
        raise ValueError("the full plan requires four distinct GPU indices")
    alignment_path = Path(alignment_tasks)
    capability_path = Path(capability_tasks)
    graph = Path(graphroot)
    run = Path(runroot)
    temporary = Path(tmpdir)
    smoke = Path(smoke_output)
    full = Path(full_output)
    smoke_commands = [
        _command(
            python=python,
            tasks=path,
            output=smoke,
            graphroot=graph,
            runroot=run,
            tmpdir=temporary,
            gpu=gpu_indices[0],
            task_id=pair["smoke_task_id"],
        )
        for path in (alignment_path, capability_path)
    ]
    full_commands = []
    for family_offset, path in enumerate((alignment_path, capability_path)):
        for shard_index in range(2):
            full_commands.append(_command(
                python=python,
                tasks=path,
                output=full,
                graphroot=graph,
                runroot=run,
                tmpdir=temporary,
                gpu=gpu_indices[family_offset * 2 + shard_index],
                num_shards=2,
                shard_index=shard_index,
            ))
    return {
        **pair,
        "podman": {
            "graphroot": str(graph), "runroot": str(run),
            "image_copy_tmpdir": str(temporary),
        },
        "smoke_output": str(smoke),
        "full_output": str(full),
        "working_directory": "/data/agastyas/nla_groundedness_qwen_v2/code",
        "smoke_commands_sequential": smoke_commands,
        "full_commands_parallel": full_commands,
        "required_environment": {
            "PYTHONPATH": "/data/agastyas/nla_groundedness_qwen_v2/code",
            "HF_HUB_CACHE": "/data/agastyas/hf_cache",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TMPDIR": str(temporary),
            "http_proxy": "http://www-cache.rz.ruhr-uni-bochum.de:80",
            "https_proxy": "http://www-cache.rz.ruhr-uni-bochum.de:80",
            "no_proxy": ".rub.de,.ruhr-uni-bochum.de,localhost,127.0.0.1,::1",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alignment-tasks", required=True)
    parser.add_argument("--capability-tasks", required=True)
    parser.add_argument("--smoke-output", required=True)
    parser.add_argument("--full-output", required=True)
    parser.add_argument("--podman-root", required=True)
    parser.add_argument("--podman-runroot", required=True)
    parser.add_argument("--podman-tmpdir", required=True)
    parser.add_argument("--python", default="/data/agastyas/Miniconda3/bin/python")
    parser.add_argument("--gpu-indices", nargs=4, default=["0", "1", "2", "3"])
    parser.add_argument("--expected-count", type=int, default=50)
    parser.add_argument("--check-podman", action="store_true")
    parser.add_argument("--audit-output")
    parser.add_argument("--audit-expected-per-family", type=int)
    parser.add_argument("--write-plan")
    args = parser.parse_args()
    plan = build_plan(
        alignment_tasks=args.alignment_tasks,
        capability_tasks=args.capability_tasks,
        smoke_output=args.smoke_output,
        full_output=args.full_output,
        graphroot=args.podman_root,
        runroot=args.podman_runroot,
        tmpdir=args.podman_tmpdir,
        python=args.python,
        gpu_indices=args.gpu_indices,
        expected_count=args.expected_count,
    )
    if args.check_podman:
        plan["podman_preflight"] = inspect_podman_store(
            args.podman_root, args.podman_runroot, args.podman_tmpdir
        )
    if args.audit_output:
        if args.audit_expected_per_family is None:
            parser.error("--audit-output requires --audit-expected-per-family")
        if args.audit_expected_per_family not in (1, args.expected_count):
            parser.error(
                "--audit-expected-per-family must be 1 (smoke) or match --expected-count"
            )
        task_files = {
            ALIGNMENT_FAMILY: Path(args.alignment_tasks),
            CAPABILITY_FAMILY: Path(args.capability_tasks),
        }
        expected_tasks = {
            family: {
                str(task["task_id"]): str(task["task_sha256"])
                for task in read_jsonl(path)
            }
            for family, path in task_files.items()
        }
        if args.audit_expected_per_family == 1:
            expected_tasks = {
                family: {plan["smoke_task_id"]: tasks[plan["smoke_task_id"]]}
                for family, tasks in expected_tasks.items()
            }
        plan["output_audit"] = audit_output(
            args.audit_output,
            expected_per_family=args.audit_expected_per_family,
            expected_tasks=expected_tasks,
        )
    if args.write_plan:
        atomic_json(args.write_plan, plan)
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
