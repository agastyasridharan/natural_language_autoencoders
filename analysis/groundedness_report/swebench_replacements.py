"""Select the next frozen SHA-ranked SWE-bench replacement rows offline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .io import atomic_json, atomic_jsonl, read_json, read_jsonl, sha256_file
from .protocol import load_protocol
from .sampling import sha256_rank
from .spec import PROTOCOL_SHA256


def archived_rows(response_dir: str | Path) -> list[dict[str, Any]]:
    """Reconstruct dataset rows from the immutable datasets-server responses."""
    rows: list[dict[str, Any]] = []
    paths = sorted(Path(response_dir).glob("*.json"))
    if not paths:
        raise ValueError("no archived SWE-bench response pages found")
    for path in paths:
        try:
            offset = int(path.stem)
        except ValueError as error:
            raise ValueError(f"non-numeric response page name: {path.name}") from error
        payload = json.loads(path.read_text())
        returned = payload.get("rows") or []
        for within, wrapper in enumerate(returned):
            row = wrapper.get("row") if isinstance(wrapper, dict) else None
            if not isinstance(row, dict):
                raise ValueError(f"invalid archived row in {path}")
            rows.append({"source_row": offset + within, **row})
    identities = [str(row.get("instance_id") or "") for row in rows]
    if any(not value for value in identities) or len(identities) != len(set(identities)):
        raise ValueError("archived SWE-bench instance IDs are missing or duplicated")
    return rows


def replacement_rows(
    rows: list[dict[str, Any]],
    initial: list[dict[str, Any]],
    *,
    seed: int,
    count: int,
    rank_start: int,
) -> list[dict[str, Any]]:
    if count < 1 or rank_start < len(initial):
        raise ValueError("replacement count must be positive and rank-start must follow the initial pool")
    ranked = sha256_rank(
        rows,
        seed=seed,
        namespace="ood-task:capability_swebench_verified",
        identity=lambda row: str(row.get("instance_id") or row.get("source_row")),
    )
    if ranked[: len(initial)] != initial:
        raise ValueError("archived ranking does not reproduce the frozen initial selection")
    selected = ranked[rank_start : rank_start + count]
    if len(selected) != count:
        raise ValueError("replacement rank interval exceeds the archived dataset")
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--responses", required=True)
    parser.add_argument("--initial-selected", required=True)
    parser.add_argument("--exclusion-ledger", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--rank-start", type=int, default=50)
    args = parser.parse_args()

    protocol = load_protocol()
    ledger = read_json(args.exclusion_ledger)
    if ledger.get("protocol_sha256") != PROTOCOL_SHA256:
        raise ValueError("exclusion ledger has a stale protocol hash")
    excluded = [str(value) for value in ledger.get("paired_excluded_task_ids") or []]
    if not excluded or len(excluded) != len(set(excluded)):
        raise ValueError("exclusion ledger must contain unique paired exclusions")
    rows = archived_rows(args.responses)
    initial = list(read_jsonl(args.initial_selected))
    selected = replacement_rows(
        rows,
        initial,
        seed=int(protocol["seed"]),
        count=len(excluded),
        rank_start=args.rank_start,
    )
    output = Path(args.out)
    atomic_jsonl(output, selected)
    atomic_json(output.with_suffix(".meta.json"), {
        "schema_version": 1,
        "protocol_sha256": PROTOCOL_SHA256,
        "seed": int(protocol["seed"]),
        "namespace": "ood-task:capability_swebench_verified",
        "initial_selected_sha256": sha256_file(args.initial_selected),
        "exclusion_ledger_sha256": sha256_file(args.exclusion_ledger),
        "paired_excluded_task_ids": excluded,
        "rank_start_zero_based": args.rank_start,
        "rank_stop_exclusive": args.rank_start + len(selected),
        "replacement_task_ids": [str(row["instance_id"]) for row in selected],
        "replacement_selected_sha256": sha256_file(output),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
