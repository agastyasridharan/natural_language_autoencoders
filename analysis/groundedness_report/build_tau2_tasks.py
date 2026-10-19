"""Build the frozen 50-task tau2-bench Retail selection."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .io import atomic_json, atomic_jsonl, sha256_file
from .protocol import load_evaluation_registry, load_protocol
from .sampling import sha256_rank
from .spec import PROTOCOL_SHA256


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()


def select_retail_tasks(
    rows: list[dict[str, Any]], *, seed: int, rank_start: int = 0, count: int = 50
) -> list[dict[str, Any]]:
    """Select one contiguous block from the protocol-keyed SHA-256 ranking."""
    if rank_start < 0 or count < 1 or rank_start + count > len(rows):
        raise ValueError("rank interval is outside the retail task population")
    task_ids = [str(row.get("id")) for row in rows]
    if "None" in task_ids or len(task_ids) != len(set(task_ids)):
        raise ValueError("retail task IDs must be present and unique")
    ranked = sha256_rank(
        rows,
        seed=seed,
        namespace="ood-task:capability_tau2_retail",
        identity=lambda row: row["id"],
    )
    return ranked[rank_start : rank_start + count]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--rank-start", type=int, default=0)
    parser.add_argument("--count", type=int, default=50)
    args = parser.parse_args()

    source = Path(args.source)
    raw = source.read_bytes()
    rows = json.loads(raw)
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError("tau2 Retail tasks must be a JSON array of objects")
    protocol = load_protocol()
    registry = load_evaluation_registry()
    family = next(
        row for row in registry["families"] if row["id"] == "capability_tau2_retail"
    )
    selected = select_retail_tasks(
        rows,
        seed=int(protocol["seed"]),
        rank_start=args.rank_start,
        count=args.count,
    )
    source_index = {str(row["id"]): index for index, row in enumerate(rows)}
    tasks = []
    for rank_offset, row in enumerate(selected):
        task = {
            "schema_version": 1,
            "protocol_sha256": PROTOCOL_SHA256,
            "replication_label": registry["label_for_every_ood_row"],
            "family_id": family["id"],
            "task_id": str(row["id"]),
            "interaction_kind": "official_tau2_retail_environment_and_user_simulator",
            "source": {
                "benchmark_repo": family["source"],
                "benchmark_revision": family["revision"],
                "source_tasks_sha256": hashlib.sha256(raw).hexdigest(),
                "source_task_index": source_index[str(row["id"])],
                "source_task_sha256": hashlib.sha256(_canonical(row)).hexdigest(),
                "sha256_rank_zero_based": args.rank_start + rank_offset,
            },
        }
        task["task_sha256"] = hashlib.sha256(_canonical(task)).hexdigest()
        tasks.append(task)

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    task_path = output / "capability_tau2_retail.jsonl"
    atomic_jsonl(task_path, tasks)
    atomic_json(
        output / "task_meta.json",
        {
            "schema_version": 1,
            "protocol_sha256": PROTOCOL_SHA256,
            "family_id": family["id"],
            "benchmark_revision": family["revision"],
            "source_tasks_sha256": hashlib.sha256(raw).hexdigest(),
            "source_task_count": len(rows),
            "rank_start": args.rank_start,
            "rank_stop_exclusive": args.rank_start + len(tasks),
            "selection": "sha256_rank_keyed_by_protocol_seed",
            "selected_task_ids_in_rank_order": [row["task_id"] for row in tasks],
            "selected_artifact_sha256": sha256_file(task_path),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
