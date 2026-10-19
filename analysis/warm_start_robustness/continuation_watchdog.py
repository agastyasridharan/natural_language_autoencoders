"""Record continuation health and restart only idempotent durability workers.

The trainer is intentionally never restarted by this watchdog.  A trainer
failure requires an audit of the latest complete transaction before resuming.
The snapshotter and uploader are safe to restart because both use immutable
completion markers and validate their inputs before doing work.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any


ROOT = Path("/data/agastyas/nla_warmstart_robustness")
RUN = ROOT / "qwen25_7b_seed42"
SAVE = RUN / "checkpoints/implausible/rl_vllm"
SNAPSHOTS = RUN / "checkpoints/implausible/continuation_snapshots"
OPS = ROOT / "ops"
LOGS = ROOT / "logs"
HEALTH = OPS / "CONTINUATION_HEALTH.json"
HEARTBEATS = OPS / "CONTINUATION_HEARTBEATS.jsonl"
EXPECTED_STEPS = tuple(range(540, 781, 20))
TARGET_ROWS = 780
POLL_SECONDS = 60
STALL_SECONDS = 12 * 60

TRAIN_PATTERN = r"nla\.train_rl_vllm.*--resume-from-lora.*/iter_000540"
SNAPSHOTTER_PATH = OPS / "continuation_snapshotter.py"
UPLOADER_PATH = OPS / "continuation_uploader.py"
SNAPSHOTTER_PATTERN = str(SNAPSHOTTER_PATH)
UPLOADER_PATTERN = str(UPLOADER_PATH)

SNAPSHOTTER_COMMAND = (
    "exec /data/agastyas/envs/vllm-lens/bin/python -u "
    f"{SNAPSHOTTER_PATH} >> {LOGS / 'continuation_snapshotter.log'} 2>&1"
)
UPLOADER_COMMAND = (
    "export http_proxy=http://www-cache.rz.ruhr-uni-bochum.de:80; "
    "export https_proxy=$http_proxy; "
    "export no_proxy=.rub.de,.ruhr-uni-bochum.de,localhost,127.0.0.1,::1; "
    "export HF_HOME=/data/agastyas/hf_home; "
    "export HF_HUB_CACHE=/data/agastyas/hf_cache; "
    "export HF_TOKEN_PATH=/home/agastyas/.cache/huggingface/token; "
    f"exec /data/agastyas/Miniconda3/bin/python -u {UPLOADER_PATH} "
    f">> {LOGS / 'continuation_uploader.log'} 2>&1"
)


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _process_count(pattern: str) -> int:
    result = subprocess.run(
        ["pgrep", "-fc", pattern],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode not in {0, 1}:
        raise RuntimeError(f"pgrep failed for {pattern!r} with {result.returncode}")
    return int(result.stdout.strip() or 0)


def _metric_state() -> tuple[int, int | None]:
    if not (SAVE / "metrics.jsonl").is_file():
        return 0, None
    complete = [
        line
        for line in (SAVE / "metrics.jsonl").read_text().splitlines()
        if line.strip()
    ]
    if not complete:
        return 0, None
    last = json.loads(complete[-1])
    return len(complete), int(last["step"])


def _step_markers(name: str) -> list[int]:
    marked = []
    for step in EXPECTED_STEPS:
        path = SNAPSHOTS / f"step_{step:06d}" / name
        if path.is_file():
            marked.append(step)
    return marked


def _failed_markers() -> list[str]:
    if not SNAPSHOTS.exists():
        return []
    return sorted(str(path.relative_to(SNAPSHOTS)) for path in SNAPSHOTS.rglob("*FAILED.txt"))


def _start_worker(name: str, command: str) -> None:
    result = subprocess.run(
        ["screen", "-dmS", name, "bash", "-lc", command],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"could not start {name}: {(result.stderr or result.stdout).strip()}"
        )


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> int:
    OPS.mkdir(parents=True, exist_ok=True)
    last_rows = -1
    last_advance = time.monotonic()
    restart_counts = {"snapshotter": 0, "uploader": 0}
    while True:
        checked_at = _utc()
        alerts: list[str] = []
        actions: list[str] = []
        try:
            rows, latest_step = _metric_state()
            if rows > last_rows:
                last_rows = rows
                last_advance = time.monotonic()
            trainer_processes = _process_count(TRAIN_PATTERN)
            snapshotter_processes = _process_count(SNAPSHOTTER_PATTERN)
            uploader_processes = _process_count(UPLOADER_PATTERN)
            ready = _step_markers("READY.json")
            uploaded = _step_markers("UPLOADED.json")
            receipts = _step_markers("RECEIPT_PUBLISHED.json")
            failures = _failed_markers()

            if trainer_processes and time.monotonic() - last_advance > STALL_SECONDS:
                alerts.append(f"metrics have not advanced for at least {STALL_SECONDS} seconds")
            if rows < TARGET_ROWS and not trainer_processes:
                alerts.append("trainer is inactive before 780 metric rows; manual audit required")
            if failures:
                alerts.append(f"durability failure markers present: {failures}")

            if len(ready) < len(EXPECTED_STEPS) and not snapshotter_processes:
                _start_worker("nla_continuation_snapshotter", SNAPSHOTTER_COMMAND)
                restart_counts["snapshotter"] += 1
                actions.append("restarted idempotent snapshotter")
                snapshotter_processes = _process_count(SNAPSHOTTER_PATTERN)
            if len(uploaded) < len(EXPECTED_STEPS) and not uploader_processes:
                _start_worker("nla_continuation_uploader", UPLOADER_COMMAND)
                restart_counts["uploader"] += 1
                actions.append("restarted idempotent uploader")
                uploader_processes = _process_count(UPLOADER_PATTERN)

            disk = shutil.disk_usage(ROOT)
            if disk.free < 500 * 1024**3:
                alerts.append(f"Athena free space below 500 GiB: {disk.free} bytes")
            state = {
                "schema_version": 1,
                "checked_at_utc": checked_at,
                "trainer": {
                    "processes": trainer_processes,
                    "metric_rows": rows,
                    "latest_metric_step": latest_step,
                    "target_metric_rows": TARGET_ROWS,
                },
                "durability_workers": {
                    "snapshotter_processes": snapshotter_processes,
                    "uploader_processes": uploader_processes,
                    "restart_counts": restart_counts,
                },
                "continuation_snapshots": {
                    "expected_steps": list(EXPECTED_STEPS),
                    "ready_steps": ready,
                    "uploaded_steps": uploaded,
                    "receipt_published_steps": receipts,
                    "failure_markers": failures,
                },
                "disk": {
                    "total_bytes": disk.total,
                    "used_bytes": disk.used,
                    "free_bytes": disk.free,
                },
                "code_sha256": {
                    "watchdog": _sha256(Path(__file__)),
                    "snapshotter": _sha256(SNAPSHOTTER_PATH),
                    "uploader": _sha256(UPLOADER_PATH),
                },
                "actions": actions,
                "alerts": alerts,
            }
        except Exception as exc:
            state = {
                "schema_version": 1,
                "checked_at_utc": checked_at,
                "actions": [],
                "alerts": [f"watchdog check failed: {type(exc).__name__}: {exc}"],
            }
        _atomic_json(HEALTH, state)
        with HEARTBEATS.open("a") as handle:
            handle.write(json.dumps(state, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        done = (
            state.get("trainer", {}).get("metric_rows", 0) >= TARGET_ROWS
            and len(state.get("continuation_snapshots", {}).get("uploaded_steps", []))
            == len(EXPECTED_STEPS)
            and len(state.get("continuation_snapshots", {}).get("receipt_published_steps", []))
            == len(EXPECTED_STEPS)
        )
        if done:
            return 0
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    raise SystemExit(main())
