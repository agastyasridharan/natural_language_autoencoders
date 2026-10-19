"""Atomic self-contained checkpoint-history transactions."""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path


def publish_immutable_checkpoint_snapshot(
    save_dir: Path,
    actor_dir: Path,
    step: int,
    *,
    run_log_path: str | None = None,
) -> Path:
    """Atomically publish actor/critic/optimizer/metrics/config provenance.

    Large immutable files are hard-linked within the same run filesystem, so
    preserving every 20-step critic/optimizer does not duplicate local blocks.
    Files that continue changing (metrics and the live log) are copied instead.
    """
    history_root = save_dir / "history"
    history_root.mkdir(parents=True, exist_ok=True)
    destination = history_root / f"step_{step:06d}"
    if destination.exists():
        raise RuntimeError(
            f"immutable checkpoint history already exists at {destination}; refusing overwrite"
        )
    staging = history_root / f".step_{step:06d}.tmp.{os.getpid()}"
    if staging.exists():
        raise RuntimeError(f"unexpected immutable-history staging collision: {staging}")
    staging.mkdir()

    required = [
        actor_dir,
        save_dir / "optim_latest.pt",
        save_dir / "metrics.jsonl",
        save_dir / "run_config.yaml",
        save_dir / "nla_meta.yaml",
        save_dir / "provenance" / "code_snapshot.tar.gz",
    ]
    if not run_log_path:
        raise RuntimeError(
            f"immutable checkpoint step {step}: NLA_RUN_LOG_PATH is required"
        )
    required.append(Path(run_log_path))
    critic_dir = save_dir / "critic_latest"
    if critic_dir.exists():
        required.append(critic_dir)
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"immutable checkpoint step {step}: missing {path}")

    # Actor includes both the trainable default and frozen reference adapters.
    # Hard links preserve current critic/optimizer inodes after latest advances.
    shutil.copytree(actor_dir, staging / "actor", copy_function=os.link)
    if critic_dir.exists():
        shutil.copytree(critic_dir, staging / "critic", copy_function=os.link)
    os.link(save_dir / "optim_latest.pt", staging / "optim.pt")
    shutil.copy2(save_dir / "metrics.jsonl", staging / "metrics.jsonl")
    shutil.copy2(save_dir / "run_config.yaml", staging / "run_config.yaml")
    shutil.copy2(save_dir / "nla_meta.yaml", staging / "nla_meta.yaml")
    provenance = save_dir / "provenance"
    shutil.copytree(provenance, staging / "provenance", copy_function=os.link)
    live_log = Path(run_log_path)
    shutil.copy2(live_log, staging / "training.log")

    for path in (
        staging / "actor" / "adapter_model.safetensors",
        staging / "actor" / "reference" / "adapter_model.safetensors",
    ):
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(
                f"immutable checkpoint step {step}: missing actor/reference adapter {path}"
            )
    if critic_dir.exists():
        for path in (
            staging / "critic" / "value_head.safetensors",
            staging / "critic" / "saved_at_step.txt",
        ):
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError(
                    f"immutable checkpoint step {step}: incomplete critic snapshot {path}"
                )

    rows = [
        json.loads(line)
        for line in (staging / "metrics.jsonl").read_text().splitlines()
        if line
    ]
    actual_metric_steps = [int(row["step"]) for row in rows]
    if (
        not actual_metric_steps
        or actual_metric_steps != sorted(set(actual_metric_steps))
        or actual_metric_steps[-1] != step - 1
        or any(value < 0 or value >= step for value in actual_metric_steps)
    ):
        raise RuntimeError(
            f"immutable checkpoint step {step}: metrics must be unique, increasing, "
            f"bounded below {step}, and end at {step - 1}; got "
            f"{len(actual_metric_steps)} rows ending at "
            f"{actual_metric_steps[-1] if actual_metric_steps else None}"
        )
    if critic_dir.exists():
        marker = int((staging / "critic" / "saved_at_step.txt").read_text())
        if marker != step:
            raise RuntimeError(f"critic marker {marker} != checkpoint step {step}")

    inventory = []
    for path in sorted(staging.rglob("*")):
        if path.is_file():
            inventory.append({"path": str(path.relative_to(staging)), "size": path.stat().st_size})
    complete = {
        "schema_version": 1,
        "step": step,
        "actor_source": actor_dir.name,
        "metrics_rows": len(rows),
        "files_before_marker": len(inventory),
        "logical_bytes_before_marker": sum(row["size"] for row in inventory),
        "inventory": inventory,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    marker_path = staging / "COMPLETE.json"
    with marker_path.open("w") as handle:
        json.dump(complete, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    for path in sorted(staging.rglob("*")):
        if path.is_file():
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
    for directory in sorted(
        [path for path in staging.rglob("*") if path.is_dir()] + [staging],
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        directory_fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    os.rename(staging, destination)
    history_fd = os.open(history_root, os.O_RDONLY)
    try:
        os.fsync(history_fd)
    finally:
        os.close(history_fd)
    print(f"[save] immutable transaction published -> {destination}", flush=True)
    return destination
