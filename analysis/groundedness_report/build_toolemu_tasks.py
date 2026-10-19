"""Build the frozen 50-case ToolEmu selection."""
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


FAMILY_ID = "alignment_overly_agentic"
TOOLEMU_NONCYBER_AMENDMENT_SHA256 = (
    "655e8d38f2a2e1c51167e12c56baf30fa0c2193e9f2663c73bd113b2c39c5b19"
)
TOOLEMU_NONCYBER_EXCLUDED_TASK_IDS = (
    "official_110",
    "official_112",
    "official_76",
)
_NONCYBER_AMENDMENT_PATH = (
    Path(__file__).with_name("audits") / "v7" / "toolemu_noncyber_scope_amendment.json"
)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()


def select_toolemu_cases(
    rows: list[dict[str, Any]], *, seed: int, rank_start: int = 0, count: int = 50
) -> list[dict[str, Any]]:
    """Select from the frozen ranking after the explicit non-Cyber exclusions."""
    names = [str(row.get("name")) for row in rows]
    if "None" in names or len(names) != len(set(names)):
        raise ValueError("ToolEmu case names must be present and unique")
    ranked = sha256_rank(
        rows,
        seed=seed,
        namespace=f"ood-task:{FAMILY_ID}",
        identity=lambda row: row["name"],
    )
    allowed = [
        row for row in ranked
        if str(row["name"]) not in TOOLEMU_NONCYBER_EXCLUDED_TASK_IDS
    ]
    if rank_start < 0 or count < 1 or rank_start + count > len(allowed):
        raise ValueError("rank interval is outside the permitted ToolEmu case population")
    return allowed[rank_start : rank_start + count]


def _validate_noncyber_amendment() -> None:
    observed = sha256_file(_NONCYBER_AMENDMENT_PATH)
    if observed != TOOLEMU_NONCYBER_AMENDMENT_SHA256:
        raise ValueError(
            "ToolEmu non-Cyber scope amendment hash drift: "
            f"expected {TOOLEMU_NONCYBER_AMENDMENT_SHA256}, observed {observed}"
        )


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
        raise ValueError("ToolEmu cases must be a JSON array of objects")
    _validate_noncyber_amendment()
    available_names = {str(row.get("name")) for row in rows}
    missing_exclusions = sorted(set(TOOLEMU_NONCYBER_EXCLUDED_TASK_IDS) - available_names)
    if missing_exclusions:
        raise ValueError(f"ToolEmu source is missing frozen excluded cases: {missing_exclusions}")
    protocol = load_protocol()
    registry = load_evaluation_registry()
    family = next(row for row in registry["families"] if row["id"] == FAMILY_ID)
    selected = select_toolemu_cases(
        rows,
        seed=int(protocol["seed"]),
        rank_start=args.rank_start,
        count=args.count,
    )
    source_index = {str(row["name"]): index for index, row in enumerate(rows)}
    source_rank = {
        str(row["name"]): rank
        for rank, row in enumerate(sha256_rank(
            rows,
            seed=int(protocol["seed"]),
            namespace=f"ood-task:{FAMILY_ID}",
            identity=lambda row: row["name"],
        ))
    }
    tasks = []
    for row in selected:
        task = {
            "schema_version": 1,
            "protocol_sha256": PROTOCOL_SHA256,
            "toolemu_noncyber_scope_amendment_sha256": (
                TOOLEMU_NONCYBER_AMENDMENT_SHA256
            ),
            "replication_label": registry["label_for_every_ood_row"],
            "family_id": family["id"],
            "task_id": str(row["name"]),
            "interaction_kind": "official_toolemu_adversarial_emulator",
            "toolkits": list(row["Toolkits"]),
            "source": {
                "benchmark_repo": family["source"],
                "benchmark_revision": family["revision"],
                "source_cases_sha256": hashlib.sha256(raw).hexdigest(),
                "source_case_index": source_index[str(row["name"])],
                "source_case_sha256": hashlib.sha256(_canonical(row)).hexdigest(),
                "sha256_rank_zero_based": source_rank[str(row["name"])],
            },
        }
        task["task_sha256"] = hashlib.sha256(_canonical(task)).hexdigest()
        tasks.append(task)

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    task_path = output / f"{FAMILY_ID}.jsonl"
    atomic_jsonl(task_path, tasks)
    atomic_json(
        output / "task_meta.json",
        {
            "schema_version": 1,
            "protocol_sha256": PROTOCOL_SHA256,
            "toolemu_noncyber_scope_amendment_sha256": (
                TOOLEMU_NONCYBER_AMENDMENT_SHA256
            ),
            "family_id": family["id"],
            "benchmark_revision": family["revision"],
            "source_cases_sha256": hashlib.sha256(raw).hexdigest(),
            "source_case_count": len(rows),
            "permitted_rank_start": args.rank_start,
            "permitted_rank_stop_exclusive": args.rank_start + len(tasks),
            "selection": "sha256_rank_keyed_by_protocol_seed_after_exact_noncyber_exclusions",
            "excluded_task_ids": list(TOOLEMU_NONCYBER_EXCLUDED_TASK_IDS),
            "selected_source_ranks_zero_based": [
                task["source"]["sha256_rank_zero_based"] for task in tasks
            ],
            "simulator_type": "adv_thought",
            "agent_type": "naive",
            "selected_task_ids_in_rank_order": [row["task_id"] for row in tasks],
            "selected_artifact_sha256": sha256_file(task_path),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
