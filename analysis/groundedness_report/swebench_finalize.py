"""Build and audit the canonical paired SWE-bench set from replacement passes."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

from .gpu_ood_static import _slug
from .io import atomic_json, atomic_jsonl, read_json, read_jsonl, sha256_file
from .spec import PROTOCOL_SHA256
from .swebench_pair import ALIGNMENT_FAMILY, CAPABILITY_FAMILY, audit_output


FAMILIES = (ALIGNMENT_FAMILY, CAPABILITY_FAMILY)


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def select_batches(
    *, root: Path, manifest: dict[str, Any]
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    if manifest.get("protocol_sha256") != PROTOCOL_SHA256:
        raise ValueError("canonical batch manifest has a stale protocol hash")
    selected: dict[str, list[dict[str, Any]]] = {family: [] for family in FAMILIES}
    provenance = []
    seen: set[str] = set()
    for batch in manifest.get("batches") or []:
        task_dir = _resolve(root, str(batch["task_dir"]))
        run_dir = _resolve(root, str(batch["run_dir"]))
        ledger_value = batch.get("exclusion_ledger")
        ledger_path = _resolve(root, str(ledger_value)) if ledger_value else None
        excluded: set[str] = set()
        if ledger_path is not None:
            ledger = read_json(ledger_path)
            if ledger.get("protocol_sha256") != PROTOCOL_SHA256:
                raise ValueError(f"stale exclusion ledger: {ledger_path}")
            excluded = {str(value) for value in ledger.get("paired_excluded_task_ids") or []}
        by_family = {
            family: list(read_jsonl(task_dir / f"{family}.jsonl"))
            for family in FAMILIES
        }
        ids = {
            family: {str(task["task_id"]) for task in tasks}
            for family, tasks in by_family.items()
        }
        if ids[ALIGNMENT_FAMILY] != ids[CAPABILITY_FAMILY]:
            raise ValueError(f"unpaired source batch: {task_dir}")
        if not excluded <= ids[ALIGNMENT_FAMILY]:
            raise ValueError(f"batch exclusions are absent from {task_dir}")
        retained_ids = ids[ALIGNMENT_FAMILY] - excluded
        duplicate = seen & retained_ids
        if duplicate:
            raise ValueError(f"task IDs occur in multiple retained batches: {sorted(duplicate)}")
        seen.update(retained_ids)
        for family in FAMILIES:
            selected[family].extend(
                task for task in by_family[family] if str(task["task_id"]) in retained_ids
            )
        provenance.append({
            "name": str(batch["name"]),
            "task_dir": str(task_dir),
            "run_dir": str(run_dir),
            "exclusion_ledger": str(ledger_path) if ledger_path else None,
            "exclusion_ledger_sha256": sha256_file(ledger_path) if ledger_path else None,
            "excluded_task_ids": sorted(excluded),
            "retained_task_ids": sorted(retained_ids),
        })
    for family in FAMILIES:
        selected[family].sort(key=lambda task: str(task["task_id"]))
    return selected, provenance


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
    expected = int(manifest.get("expected_pairs") or 0)
    if expected < 1:
        raise ValueError("canonical batch manifest must set a positive expected_pairs")
    task_output = Path(args.out_tasks)
    run_output = Path(args.out_run)
    if task_output.exists() or run_output.exists():
        raise FileExistsError("canonical task and run outputs must not already exist")

    selected, provenance = select_batches(root=root, manifest=manifest)
    if any(len(selected[family]) != expected for family in FAMILIES):
        raise ValueError(
            f"canonical family counts do not equal {expected}: "
            f"{ {family: len(selected[family]) for family in FAMILIES} }"
        )
    task_output.mkdir(parents=True)
    task_paths = {}
    for family in FAMILIES:
        path = task_output / f"{family}.jsonl"
        atomic_jsonl(path, selected[family])
        task_paths[family] = path
    task_hashes = {family: sha256_file(path) for family, path in task_paths.items()}
    atomic_json(task_output / "task_meta.json", {
        "schema_version": 1,
        "protocol_sha256": PROTOCOL_SHA256,
        "paired_instances": expected,
        "families": list(FAMILIES),
        "task_artifact_sha256": task_hashes,
        "canonical_batch_manifest_sha256": sha256_file(manifest_path),
    })

    run_output.mkdir(parents=True)
    batch_by_task = {
        task_id: batch
        for batch in provenance
        for task_id in batch["retained_task_ids"]
    }
    for family in FAMILIES:
        for task in selected[family]:
            task_id = str(task["task_id"])
            batch = batch_by_task[task_id]
            source_root = Path(batch["run_dir"])
            slug = _slug(task)
            relative_trajectory = Path("trajectories") / family / f"{slug}.json"
            source_trajectory = source_root / relative_trajectory
            trajectory = read_json(source_trajectory)
            if (
                trajectory.get("protocol_sha256") != PROTOCOL_SHA256
                or trajectory.get("status") != "complete"
                or trajectory.get("task_id") != task_id
                or trajectory.get("task_sha256") != task.get("task_sha256")
            ):
                raise ValueError(f"incompatible trajectory for {family}:{task_id}")
            destination_trajectory = run_output / relative_trajectory
            destination_trajectory.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_trajectory, destination_trajectory)
            for activation in trajectory.get("activations") or []:
                relative = Path(str(activation["vector_file"]))
                destination = run_output / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_root / relative, destination)
            source_tools = source_root / "tools" / family / slug
            if not source_tools.is_dir():
                raise ValueError(f"missing tool records for {family}:{task_id}")
            shutil.copytree(source_tools, run_output / "tools" / family / slug)

    expected_tasks = {
        family: {str(task["task_id"]): str(task["task_sha256"]) for task in tasks}
        for family, tasks in selected.items()
    }
    for family in FAMILIES:
        atomic_json(run_output / f"{family}-shard-00-meta.json", {
            "status": "done",
            "protocol_sha256": PROTOCOL_SHA256,
            "assigned_tasks": expected,
            "assigned_task_ids": sorted(expected_tasks[family]),
            "written": expected,
            "skipped": 0,
            "failures": [],
            "canonical_merge": True,
            "canonical_task_artifact_sha256": task_hashes[family],
        })
    audit = audit_output(
        run_output,
        expected_per_family=expected,
        expected_tasks=expected_tasks,
    )
    atomic_json(run_output / "canonicalization.json", {
        "schema_version": 1,
        "protocol_sha256": PROTOCOL_SHA256,
        "batch_manifest": str(manifest_path),
        "batch_manifest_sha256": sha256_file(manifest_path),
        "batches": provenance,
        "canonical_task_artifact_sha256": task_hashes,
        "audit": audit,
    })
    print(audit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
