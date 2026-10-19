#!/usr/bin/env python3
"""Inventory and resumably transfer the public warm-start NLA bundles.

This script is intended to run on Athena with gdown installed outside the
repository. It deliberately downloads only evaluation-relevant actor and
critic files: actor model DCP shards plus metadata, and the critic's exported
HF checkpoint. Optimizer, scheduler, RNG, and rollout-state files are excluded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import gdown


SOURCES = {
    "control": {
        "folder_id": "1xUX_fD5Stz0LR4fZf9W0gEfIvQJ5Wr1X",
        "directory": "control_corrected",
        "iterations": (200, 400, 600, 780),
        "latest": 780,
    },
    "anti_carthage": {
        "folder_id": "1g6i7NjjEAcMb3ccCDA0jT_Ea1ybPqfrw",
        "iterations": (100,),
        "latest": 100,
    },
    "pro_carthage": {
        "folder_id": "1GELESnqyl4k9JGhBLbztbcYfUVlIgXu3",
        "iterations": (400, 600, 780),
        "latest": 780,
    },
    "false_but_related": {
        "folder_id": "1lDNeptXg1Qx5QvCZxkMP2bQwULEukET8",
        "iterations": (200, 400, 600, 780),
        "latest": 780,
    },
}


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def enumerate_source(condition: str, output_root: Path) -> list[dict[str, str]]:
    source = SOURCES[condition]
    condition_root = output_root / "checkpoints" / str(source.get("directory", condition))
    condition_root.mkdir(parents=True, exist_ok=True)
    rows = gdown.download_folder(
        id=source["folder_id"],
        output=str(condition_root),
        quiet=True,
        remaining_ok=True,
        use_cookies=False,
        skip_download=True,
    )
    if rows is None:
        raise RuntimeError(f"could not enumerate Drive folder for {condition}")
    return [
        {
            "drive_file_id": row.id,
            "path": row.path,
            "local_path": row.local_path,
        }
        for row in rows
    ]


def iteration_from_path(path: str) -> int | None:
    match = re.search(r"/iter_(\d{7})/", f"/{path}")
    return int(match.group(1)) if match else None


def selected(path: str, condition: str, selection: str) -> bool:
    source = SOURCES[condition]
    iteration = iteration_from_path(path)
    allowed = (
        {int(source["latest"])}
        if selection in {"latest", "metadata_latest"}
        else set(source["iterations"])
    )
    if iteration not in allowed:
        return False
    if path.startswith("actor/"):
        if path.endswith(("/meta.json", "/nla_meta.yaml", "/model/.metadata")):
            return True
        return selection in {"latest", "all_candidates"} and "/model/" in path
    if path.startswith("critic/"):
        if path.endswith(("/meta.json", "/nla_meta.yaml")):
            return True
        if "/hf/" not in path:
            return False
        if selection in {"latest", "all_candidates"}:
            return True
        return not path.endswith((".safetensors",))
    return False


def download_one(row: dict[str, str], *, quiet: bool) -> dict[str, object]:
    destination = Path(row["local_path"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    result = gdown.download(
        id=row["drive_file_id"],
        output=str(destination),
        quiet=quiet,
        use_cookies=False,
        resume=True,
    )
    if result is None or not destination.is_file():
        raise RuntimeError(f"download failed: {row['path']}")
    return {
        **row,
        "bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
        "download_seconds": round(time.time() - started, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", required=True, choices=tuple(SOURCES))
    parser.add_argument(
        "--selection",
        choices=("inventory", "metadata_latest", "metadata_all", "latest", "all_candidates"),
        default="inventory",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/data/agastyas/nla_warmstart_drive_nlas"),
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    rows = enumerate_source(args.condition, args.root)
    manifest_path = args.root / "manifests" / f"{args.condition}_drive_inventory.json"
    atomic_json(
        manifest_path,
        {
            "schema": "nla.warm_start.drive_inventory.v1",
            "condition": args.condition,
            "drive_folder_id": SOURCES[args.condition]["folder_id"],
            "file_count": len(rows),
            "files": rows,
        },
    )
    print(f"inventory condition={args.condition} files={len(rows)} path={manifest_path}", flush=True)
    if args.selection == "inventory":
        return

    chosen = [row for row in rows if selected(row["path"], args.condition, args.selection)]
    receipt_path = args.root / "manifests" / f"{args.condition}_{args.selection}_receipt.json"
    receipt: dict[str, object] = {
        "schema": "nla.warm_start.drive_transfer_receipt.v1",
        "condition": args.condition,
        "selection": args.selection,
        "planned_file_count": len(chosen),
        "completed": [],
    }
    if receipt_path.is_file():
        old = json.loads(receipt_path.read_text())
        if old.get("condition") == args.condition and old.get("selection") == args.selection:
            receipt["completed"] = old.get("completed", [])
    completed_by_path = {row["path"]: row for row in receipt["completed"]}

    for index, row in enumerate(chosen, start=1):
        previous = completed_by_path.get(row["path"])
        destination = Path(row["local_path"])
        if previous and destination.is_file() and destination.stat().st_size == previous.get("bytes"):
            print(f"skip verified {index}/{len(chosen)} {row['path']}", flush=True)
            continue
        print(f"download {index}/{len(chosen)} {row['path']}", flush=True)
        completed_by_path[row["path"]] = download_one(row, quiet=args.quiet)
        receipt["completed"] = [completed_by_path[p["path"]] for p in chosen if p["path"] in completed_by_path]
        receipt["completed_file_count"] = len(receipt["completed"])
        receipt["completed_bytes"] = sum(int(item["bytes"]) for item in receipt["completed"])
        atomic_json(receipt_path, receipt)
    receipt["complete"] = len(completed_by_path) == len(chosen)
    atomic_json(receipt_path, receipt)
    print(
        f"complete condition={args.condition} selection={args.selection} "
        f"files={len(completed_by_path)} bytes={receipt.get('completed_bytes', 0)}",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"fatal: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
