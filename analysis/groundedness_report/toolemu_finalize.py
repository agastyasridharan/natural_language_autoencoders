"""Deterministically build the canonical non-Cyber ToolEmu replication set."""
from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path
from typing import Any

from .build_toolemu_tasks import TOOLEMU_NONCYBER_AMENDMENT_SHA256
from .gpu_ood_static import _slug
from .gpu_ood_toolemu import (
    FAMILY_ID,
    TOOLEMU_BOUNDED_SIMULATOR_AMENDMENT_SHA256,
    TOOLEMU_FAILURE_CLASSIFICATION_AMENDMENT_SHA256,
    TOOLEMU_FORMAT_BUDGET_AMENDMENT_SHA256,
    TOOLEMU_PILOT_GATE_RESOLUTION_SHA256,
    TOOLEMU_SHARED_STATE_AMENDMENT_SHA256,
    _validate_replaceable_failure,
    _validate_task,
)
from .io import atomic_json, atomic_jsonl, read_json, read_jsonl, sha256_file
from .spec import PROTOCOL_SHA256, TEXT_MODEL_CORRECTION_AMENDMENT_SHA256


INITIAL_TASK_ARTIFACT_SHA256 = (
    "6807c340e91d881a4c61dba00a5644eddde5f144793acb214ec5086c7b04d8ca"
)
REPLACEMENT_TASK_ARTIFACT_SHA256 = (
    "8edef2a35d39b93858ace423e8622c2bd5b748443112cd04f882271c09eb8f51"
)
CANONICAL_TARGET = 50
_HASH_CHAIN = {
    "protocol_sha256": PROTOCOL_SHA256,
    "text_model_correction_amendment_sha256": TEXT_MODEL_CORRECTION_AMENDMENT_SHA256,
    "toolemu_noncyber_scope_amendment_sha256": TOOLEMU_NONCYBER_AMENDMENT_SHA256,
    "toolemu_bounded_simulator_amendment_sha256": TOOLEMU_BOUNDED_SIMULATOR_AMENDMENT_SHA256,
    "toolemu_format_budget_amendment_sha256": TOOLEMU_FORMAT_BUDGET_AMENDMENT_SHA256,
    "toolemu_shared_state_amendment_sha256": TOOLEMU_SHARED_STATE_AMENDMENT_SHA256,
    "toolemu_pilot_gate_resolution_sha256": TOOLEMU_PILOT_GATE_RESOLUTION_SHA256,
    "toolemu_failure_classification_amendment_sha256": TOOLEMU_FAILURE_CLASSIFICATION_AMENDMENT_SHA256,
}
_CALL_HASH_CHAIN = {
    key: value
    for key, value in _HASH_CHAIN.items()
    if key != "toolemu_noncyber_scope_amendment_sha256"
}


def _validate_chain(
    record: dict[str, Any],
    label: str,
    expected_chain: dict[str, str] | None = None,
) -> None:
    for key, expected in (expected_chain or _HASH_CHAIN).items():
        if record.get(key) != expected:
            raise ValueError(f"{label} has incompatible {key}")


def _outcome_paths(task: dict[str, Any], run: Path) -> tuple[Path, Path, Path]:
    slug = _slug(task)
    return (
        run / "trajectories" / FAMILY_ID / f"{slug}.json",
        run / "failures" / FAMILY_ID / f"{slug}.json",
        run / "tools" / FAMILY_ID / slug,
    )


def _relative_artifact(value: str, prefix: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError(f"unsafe ToolEmu artifact path: {value}")
    if relative.parts[0] != prefix:
        raise ValueError(f"ToolEmu artifact is outside {prefix}: {value}")
    return relative


def _read_outcome(task: dict[str, Any], run: Path) -> dict[str, Any] | None:
    task_id = str(task["task_id"])
    trajectory_path, failure_path, tools_path = _outcome_paths(task, run)
    if trajectory_path.is_file() and failure_path.is_file():
        raise ValueError(f"ToolEmu task has both trajectory and failure: {task_id}")
    if trajectory_path.is_file():
        trajectory = read_json(trajectory_path)
        _validate_chain(trajectory, f"ToolEmu trajectory {task_id}")
        if (
            trajectory.get("status") != "complete"
            or trajectory.get("family_id") != FAMILY_ID
            or trajectory.get("trajectory_id") != f"{FAMILY_ID}:{task_id}"
            or trajectory.get("task") != task
        ):
            raise ValueError(f"incompatible ToolEmu trajectory: {task_id}")
        activations = trajectory.get("activations") or []
        if not 1 <= len(activations) <= 4:
            raise ValueError(f"invalid ToolEmu activation count: {task_id}")
        for activation in activations:
            relative = _relative_artifact(str(activation["vector_file"]), "activations")
            vector_path = run / relative
            if not vector_path.is_file() or sha256_file(vector_path) != str(
                activation["vector_sha256"]
            ):
                raise ValueError(f"ToolEmu activation hash mismatch: {task_id}")
        audit_relative = _relative_artifact(
            str(trajectory["toolemu"]["simulator_audit_dir"]), "tools"
        )
        audit_path = run / audit_relative
        call_audits = sorted(audit_path.glob("call-*.json"))
        if len(call_audits) != int(
            trajectory["toolemu"]["simulator_call_audit_count"]
        ):
            raise ValueError(f"ToolEmu call-audit count mismatch: {task_id}")
        for call_path in call_audits:
            _validate_chain(
                read_json(call_path),
                f"ToolEmu call audit {task_id}",
                _CALL_HASH_CHAIN,
            )
        if call_audits and not tools_path.is_dir():
            raise ValueError(f"missing ToolEmu tool-audit tree: {task_id}")
        return {
            "kind": "trajectory",
            "task": task,
            "run": run,
            "path": trajectory_path,
            "record": trajectory,
            "tools": tools_path if tools_path.is_dir() else None,
        }
    if failure_path.is_file():
        failure = read_json(failure_path)
        _validate_chain(failure, f"ToolEmu failure {task_id}")
        if (
            failure.get("family_id") != FAMILY_ID
            or str(failure.get("task_id")) != task_id
            or failure.get("task") != task
        ):
            raise ValueError(f"incompatible ToolEmu failure: {task_id}")
        _validate_replaceable_failure(failure)
        audit_relative = _relative_artifact(str(failure["audit_dir"]), "tools")
        if audit_relative.parts[:3] != ("tools", FAMILY_ID, _slug(task)):
            raise ValueError(f"ToolEmu failure audit path has wrong task: {task_id}")
        completed_calls = int(
            ((failure.get("simulator_budget") or {}).get("completed_calls")) or 0
        )
        if completed_calls and not tools_path.is_dir():
            raise ValueError(f"missing excluded ToolEmu tool-audit tree: {task_id}")
        return {
            "kind": "failure",
            "task": task,
            "run": run,
            "path": failure_path,
            "record": failure,
            "tools": tools_path if tools_path.is_dir() else None,
        }
    return None


def select_canonical(
    *,
    initial_tasks: list[dict[str, Any]],
    replacement_tasks: list[dict[str, Any]],
    initial_run: Path,
    replacement_run: Path,
    expected: int = CANONICAL_TARGET,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """Apply the frozen outcome/exclusion/replacement rule without writing."""
    if len(initial_tasks) != expected:
        raise ValueError("ToolEmu initial artifact does not equal the canonical target")
    for task in [*initial_tasks, *replacement_tasks]:
        _validate_task(task)
        if (
            task.get("toolemu_noncyber_scope_amendment_sha256")
            != TOOLEMU_NONCYBER_AMENDMENT_SHA256
        ):
            raise ValueError("ToolEmu canonicalizer received a pre-scope task")
    task_ids = [str(task["task_id"]) for task in [*initial_tasks, *replacement_tasks]]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("duplicate ToolEmu task across initial and replacement pools")

    selected: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for task in initial_tasks:
        outcome = _read_outcome(task, initial_run)
        if outcome is None:
            raise ValueError(f"missing initial ToolEmu outcome: {task['task_id']}")
        (selected if outcome["kind"] == "trajectory" else excluded).append(outcome)

    replacement_count = 0
    for index, task in enumerate(replacement_tasks):
        if len(selected) >= expected:
            break
        outcome = _read_outcome(task, replacement_run)
        if outcome is None:
            raise ValueError(f"missing next ToolEmu replacement outcome: {task['task_id']}")
        replacement_count = index + 1
        (selected if outcome["kind"] == "trajectory" else excluded).append(outcome)

    if len(selected) != expected:
        raise ValueError(
            f"canonical ToolEmu count {len(selected)} differs from expected {expected}"
        )
    for task in replacement_tasks[replacement_count:]:
        trajectory_path, failure_path, tools_path = _outcome_paths(task, replacement_run)
        if trajectory_path.exists() or failure_path.exists() or tools_path.exists():
            raise ValueError(
                f"ToolEmu replacement processed after target was met: {task['task_id']}"
            )
    return selected, excluded, replacement_count


def _validate_run_meta(
    *, run: Path, task_hash: str, assigned_ids: list[str], label: str
) -> dict[str, Any]:
    meta_path = run / f"{FAMILY_ID}-shard-00-meta.json"
    meta = read_json(meta_path)
    _validate_chain(meta, f"{label} metadata")
    if (
        meta.get("pilot") is not False
        or meta.get("task_artifact_sha256") != task_hash
        or meta.get("assigned_task_ids") != assigned_ids
        or int(meta.get("assigned_tasks", -1)) != len(assigned_ids)
        or meta.get("status") not in {"done", "failed"}
    ):
        raise ValueError(f"incompatible {label} metadata")
    return {"path": str(meta_path), "sha256": sha256_file(meta_path), "value": meta}


def _aggregate_hash(rows: list[tuple[str, str, str]]) -> str:
    payload = "".join(
        f"{task_id}\0{relative}\0{digest}\n"
        for task_id, relative, digest in sorted(rows)
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _copy_tree(source: Path, destination: Path) -> int:
    if not source.is_dir():
        raise ValueError(f"missing ToolEmu audit directory: {source}")
    shutil.copytree(source, destination)
    return sum(path.is_file() for path in destination.rglob("*"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-tasks", required=True)
    parser.add_argument("--replacement-tasks", required=True)
    parser.add_argument("--initial-run", required=True)
    parser.add_argument("--replacement-run", required=True)
    parser.add_argument("--out-tasks", required=True)
    parser.add_argument("--out-run", required=True)
    args = parser.parse_args()

    initial_task_path = Path(args.initial_tasks)
    replacement_task_path = Path(args.replacement_tasks)
    if sha256_file(initial_task_path) != INITIAL_TASK_ARTIFACT_SHA256:
        raise ValueError("initial ToolEmu task artifact hash mismatch")
    if sha256_file(replacement_task_path) != REPLACEMENT_TASK_ARTIFACT_SHA256:
        raise ValueError("replacement ToolEmu task artifact hash mismatch")
    initial_tasks = list(read_jsonl(initial_task_path))
    replacement_tasks = list(read_jsonl(replacement_task_path))
    initial_run = Path(args.initial_run)
    replacement_run = Path(args.replacement_run)
    selected, excluded, replacement_count = select_canonical(
        initial_tasks=initial_tasks,
        replacement_tasks=replacement_tasks,
        initial_run=initial_run,
        replacement_run=replacement_run,
    )
    initial_meta = _validate_run_meta(
        run=initial_run,
        task_hash=INITIAL_TASK_ARTIFACT_SHA256,
        assigned_ids=[str(task["task_id"]) for task in initial_tasks],
        label="initial ToolEmu run",
    )
    replacement_meta = None
    if replacement_count:
        replacement_meta = _validate_run_meta(
            run=replacement_run,
            task_hash=REPLACEMENT_TASK_ARTIFACT_SHA256,
            assigned_ids=[
                str(task["task_id"])
                for task in replacement_tasks[:replacement_count]
            ],
            label="replacement ToolEmu run",
        )

    task_output = Path(args.out_tasks)
    run_output = Path(args.out_run)
    if task_output.exists() or run_output.exists():
        raise FileExistsError("canonical ToolEmu outputs must not already exist")
    canonical_tasks = [row["task"] for row in selected]
    canonical_task_path = task_output / f"{FAMILY_ID}.jsonl"
    atomic_jsonl(canonical_task_path, canonical_tasks)

    trajectory_hashes: list[tuple[str, str, str]] = []
    activation_hashes: list[tuple[str, str, str]] = []
    tool_files = 0
    for row in selected:
        task_id = str(row["task"]["task_id"])
        slug = _slug(row["task"])
        relative_trajectory = Path("trajectories") / FAMILY_ID / f"{slug}.json"
        destination_trajectory = run_output / relative_trajectory
        destination_trajectory.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(row["path"], destination_trajectory)
        trajectory_hashes.append(
            (task_id, str(relative_trajectory), sha256_file(destination_trajectory))
        )
        for activation in row["record"]["activations"]:
            relative = _relative_artifact(str(activation["vector_file"]), "activations")
            destination = run_output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(row["run"] / relative, destination)
            activation_hashes.append((task_id, str(relative), sha256_file(destination)))
        if row["tools"] is not None:
            tool_files += _copy_tree(
                row["tools"], run_output / "tools" / FAMILY_ID / slug
            )

    excluded_rows = []
    for row in excluded:
        task_id = str(row["task"]["task_id"])
        slug = _slug(row["task"])
        relative_failure = Path("excluded") / "failures" / FAMILY_ID / f"{slug}.json"
        destination_failure = run_output / relative_failure
        destination_failure.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(row["path"], destination_failure)
        if row["tools"] is not None:
            tool_files += _copy_tree(
                row["tools"], run_output / "tools" / FAMILY_ID / slug
            )
        excluded_rows.append(
            {
                "task_id": task_id,
                "source_sha256_rank_zero_based": int(
                    row["task"]["source"]["sha256_rank_zero_based"]
                ),
                "failure_kind": row["record"]["failure_kind"],
                "error": row["record"]["error"],
                "failure_file": str(relative_failure),
                "failure_sha256": sha256_file(destination_failure),
            }
        )

    canonical_hash = sha256_file(canonical_task_path)
    atomic_json(
        task_output / "task_meta.json",
        {
            "schema_version": 1,
            **_HASH_CHAIN,
            "family_id": FAMILY_ID,
            "trajectories": len(canonical_tasks),
            "selected_task_ids_in_frozen_order": [
                str(task["task_id"]) for task in canonical_tasks
            ],
            "initial_task_artifact_sha256": INITIAL_TASK_ARTIFACT_SHA256,
            "replacement_task_artifact_sha256": REPLACEMENT_TASK_ARTIFACT_SHA256,
            "replacement_tasks_consumed": replacement_count,
            "canonical_task_artifact_sha256": canonical_hash,
        },
    )
    atomic_json(
        run_output / f"{FAMILY_ID}-shard-00-meta.json",
        {
            "status": "done",
            "schema_version": 1,
            **_HASH_CHAIN,
            "family_id": FAMILY_ID,
            "assigned_tasks": len(selected),
            "assigned_task_ids": [str(row["task"]["task_id"]) for row in selected],
            "written": len(selected),
            "skipped": 0,
            "failures": [],
            "pilot": False,
            "canonical_merge": True,
            "canonical_task_artifact_sha256": canonical_hash,
        },
    )
    atomic_json(
        run_output / "canonicalization.json",
        {
            "schema_version": 1,
            **_HASH_CHAIN,
            "family_id": FAMILY_ID,
            "selection_rule": "initial frozen order followed by the shortest replacement prefix needed to reach 50 analyzable trajectories",
            "initial_meta": {
                "path": initial_meta["path"],
                "sha256": initial_meta["sha256"],
            },
            "replacement_meta": None
            if replacement_meta is None
            else {
                "path": replacement_meta["path"],
                "sha256": replacement_meta["sha256"],
            },
            "trajectories": len(selected),
            "activations": len(activation_hashes),
            "excluded": excluded_rows,
            "replacement_tasks_consumed": replacement_count,
            "tool_audit_files": tool_files,
            "canonical_task_artifact_sha256": canonical_hash,
            "trajectory_aggregate_sha256": _aggregate_hash(trajectory_hashes),
            "activation_aggregate_sha256": _aggregate_hash(activation_hashes),
        },
    )
    print(read_json(run_output / "canonicalization.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
