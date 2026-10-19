"""Build the canonical tau2 Retail run from initial and replacement batches."""
from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path
from typing import Any

from .gpu_ood_static import _slug
from .gpu_ood_tau2 import FAMILY_ID, _validate_task
from .io import atomic_json, atomic_jsonl, read_json, read_jsonl, sha256_file
from .spec import PROTOCOL_SHA256


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _aggregate_hash(rows: list[tuple[str, str, str]]) -> str:
    payload = "".join(
        f"{task_id}\0{relative}\0{digest}\n"
        for task_id, relative, digest in sorted(rows)
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def select_batches(
    *, root: Path, manifest: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Resolve and validate canonical tasks without writing an output."""
    if manifest.get("protocol_sha256") != PROTOCOL_SHA256:
        raise ValueError("canonical tau2 manifest has a stale protocol hash")
    if manifest.get("family_id") != FAMILY_ID:
        raise ValueError("canonical tau2 manifest has the wrong family")
    expected = int(manifest.get("expected_trajectories") or 0)
    if expected < 1:
        raise ValueError("canonical tau2 manifest must set expected_trajectories")

    ledger_path = _resolve(root, str(manifest["exclusion_ledger"]))
    ledger = read_json(ledger_path)
    if ledger.get("protocol_sha256") != PROTOCOL_SHA256:
        raise ValueError("tau2 exclusion ledger has a stale protocol hash")
    excluded_by_id = {
        str(row["task_id"]): row for row in ledger.get("exclusions") or []
    }
    if len(excluded_by_id) != len(ledger.get("exclusions") or []):
        raise ValueError("tau2 exclusion ledger contains duplicate task IDs")

    selected: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    seen: set[str] = set()
    for batch in manifest.get("batches") or []:
        task_path = _resolve(root, str(batch["task_artifact"]))
        run_dir = _resolve(root, str(batch["run_dir"]))
        observed_task_hash = sha256_file(task_path)
        expected_task_hash = str(batch["task_artifact_sha256"])
        if observed_task_hash != expected_task_hash:
            raise ValueError(f"tau2 task artifact hash mismatch: {task_path}")
        batch_ids = []
        for task in read_jsonl(task_path):
            _validate_task(task)
            task_id = str(task["task_id"])
            if task_id in seen:
                raise ValueError(f"duplicate tau2 task ID across batches: {task_id}")
            seen.add(task_id)
            batch_ids.append(task_id)
            slug = _slug(task)
            if task_id in excluded_by_id:
                failure_path = run_dir / "failures" / FAMILY_ID / f"{slug}.json"
                if not failure_path.is_file():
                    raise ValueError(f"missing excluded tau2 failure: {task_id}")
                failure = read_json(failure_path)
                ledger_row = excluded_by_id[task_id]
                if (
                    failure.get("protocol_sha256") != PROTOCOL_SHA256
                    or str(failure.get("task_id")) != task_id
                    or failure.get("task") != task
                    or sha256_file(failure_path)
                    != str(ledger_row["failure_artifact_sha256"])
                ):
                    raise ValueError(f"incompatible excluded tau2 failure: {task_id}")
                excluded.append(
                    {
                        "task": task,
                        "run_dir": run_dir,
                        "failure_path": failure_path,
                        "ledger": ledger_row,
                    }
                )
                continue
            trajectory_path = run_dir / "trajectories" / FAMILY_ID / f"{slug}.json"
            if not trajectory_path.is_file():
                raise ValueError(f"missing canonical tau2 trajectory: {task_id}")
            trajectory = read_json(trajectory_path)
            if (
                trajectory.get("protocol_sha256") != PROTOCOL_SHA256
                or trajectory.get("status") != "complete"
                or trajectory.get("family_id") != FAMILY_ID
                or trajectory.get("trajectory_id") != f"{FAMILY_ID}:{task_id}"
                or trajectory.get("task") != task
            ):
                raise ValueError(f"incompatible canonical tau2 trajectory: {task_id}")
            selected.append(
                {
                    "task": task,
                    "run_dir": run_dir,
                    "trajectory_path": trajectory_path,
                    "trajectory": trajectory,
                }
            )
        provenance.append(
            {
                "name": str(batch["name"]),
                "task_artifact": str(task_path),
                "task_artifact_sha256": observed_task_hash,
                "run_dir": str(run_dir),
                "task_ids": batch_ids,
            }
        )

    observed_exclusions = {str(row["task"]["task_id"]) for row in excluded}
    if observed_exclusions != set(excluded_by_id):
        raise ValueError("tau2 canonical batches do not exactly cover the exclusions")
    if len(selected) != expected:
        raise ValueError(
            f"canonical tau2 count {len(selected)} differs from expected {expected}"
        )
    selected.sort(key=lambda row: int(row["task"]["source"]["sha256_rank_zero_based"]))
    excluded.sort(key=lambda row: int(row["task"]["source"]["sha256_rank_zero_based"]))
    return selected, excluded, provenance


def _copy_tree(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise ValueError(f"missing tau2 tool-audit directory: {source}")
    shutil.copytree(source, destination)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-tasks", required=True)
    parser.add_argument("--out-run", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    manifest_path = Path(args.manifest)
    manifest = read_json(manifest_path)
    selected, excluded, provenance = select_batches(root=root, manifest=manifest)
    task_output = Path(args.out_tasks)
    run_output = Path(args.out_run)
    if task_output.exists() or run_output.exists():
        raise FileExistsError("canonical tau2 outputs must not already exist")

    tasks = [row["task"] for row in selected]
    task_path = task_output / f"{FAMILY_ID}.jsonl"
    atomic_jsonl(task_path, tasks)
    atomic_json(
        task_output / "task_meta.json",
        {
            "schema_version": 1,
            "protocol_sha256": PROTOCOL_SHA256,
            "family_id": FAMILY_ID,
            "trajectories": len(tasks),
            "selected_task_ids_in_rank_order": [str(row["task_id"]) for row in tasks],
            "selected_ranks": [
                int(row["source"]["sha256_rank_zero_based"]) for row in tasks
            ],
            "task_artifact_sha256": sha256_file(task_path),
            "canonical_batch_manifest_sha256": sha256_file(manifest_path),
        },
    )

    trajectory_hashes: list[tuple[str, str, str]] = []
    activation_hashes: list[tuple[str, str, str]] = []
    tool_files = 0
    for row in selected:
        task = row["task"]
        task_id = str(task["task_id"])
        slug = _slug(task)
        relative_trajectory = Path("trajectories") / FAMILY_ID / f"{slug}.json"
        destination_trajectory = run_output / relative_trajectory
        destination_trajectory.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(row["trajectory_path"], destination_trajectory)
        trajectory_hashes.append(
            (task_id, str(relative_trajectory), sha256_file(destination_trajectory))
        )
        for activation in row["trajectory"]["activations"]:
            relative = Path(str(activation["vector_file"]))
            source = row["run_dir"] / relative
            if sha256_file(source) != str(activation["vector_sha256"]):
                raise ValueError(f"tau2 activation hash mismatch: {source}")
            destination = run_output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            activation_hashes.append(
                (task_id, str(relative), sha256_file(destination))
            )
        source_tools = row["run_dir"] / "tools" / FAMILY_ID / slug
        destination_tools = run_output / "tools" / FAMILY_ID / slug
        _copy_tree(source_tools, destination_tools)
        tool_files += sum(path.is_file() for path in destination_tools.rglob("*"))

    excluded_rows = []
    for row in excluded:
        task = row["task"]
        task_id = str(task["task_id"])
        slug = _slug(task)
        failure_destination = (
            run_output / "excluded" / "failures" / FAMILY_ID / f"{slug}.json"
        )
        failure_destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(row["failure_path"], failure_destination)
        tools_source = row["run_dir"] / "tools" / FAMILY_ID / slug
        tools_destination = run_output / "excluded" / "tools" / FAMILY_ID / slug
        _copy_tree(tools_source, tools_destination)
        excluded_rows.append(
            {
                "task_id": task_id,
                "rank": int(task["source"]["sha256_rank_zero_based"]),
                "failure_file": str(failure_destination.relative_to(run_output)),
                "failure_sha256": sha256_file(failure_destination),
                "tool_files": sum(path.is_file() for path in tools_destination.rglob("*")),
            }
        )

    meta_path = run_output / f"{FAMILY_ID}-shard-00-meta.json"
    atomic_json(
        meta_path,
        {
            "status": "done",
            "schema_version": 1,
            "protocol_sha256": PROTOCOL_SHA256,
            "family_id": FAMILY_ID,
            "assigned_tasks": len(selected),
            "assigned_task_ids": [str(row["task"]["task_id"]) for row in selected],
            "written": len(selected),
            "skipped": 0,
            "failures": [],
            "canonical_merge": True,
            "canonical_task_artifact_sha256": sha256_file(task_path),
        },
    )
    atomic_json(
        run_output / "canonicalization.json",
        {
            "schema_version": 1,
            "protocol_sha256": PROTOCOL_SHA256,
            "family_id": FAMILY_ID,
            "batch_manifest": str(manifest_path),
            "batch_manifest_sha256": sha256_file(manifest_path),
            "exclusion_ledger": str(_resolve(root, str(manifest["exclusion_ledger"]))),
            "exclusion_ledger_sha256": sha256_file(
                _resolve(root, str(manifest["exclusion_ledger"]))
            ),
            "batches": provenance,
            "trajectories": len(selected),
            "activations": len(activation_hashes),
            "tool_audit_files": tool_files,
            "excluded": excluded_rows,
            "canonical_task_artifact_sha256": sha256_file(task_path),
            "trajectory_aggregate_sha256": _aggregate_hash(trajectory_hashes),
            "activation_aggregate_sha256": _aggregate_hash(activation_hashes),
        },
    )
    print(read_json(run_output / "canonicalization.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
