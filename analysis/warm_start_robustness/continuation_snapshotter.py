"""Create immutable, versioned restart snapshots for the step-540 continuation.

The trainer intentionally keeps only ``critic_latest`` and ``optim_latest.pt``.
This watcher hard-links each complete transaction into a step-specific directory
before the next transaction can replace those paths.  Actor checkpoints are
also linked into the bundle so every snapshot is independently restartable.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import torch


ROOT = Path("/data/agastyas/nla_warmstart_robustness")
RUN = ROOT / "qwen25_7b_seed42"
SAVE = RUN / "checkpoints/implausible/rl_vllm"
SNAPSHOTS = RUN / "checkpoints/implausible/continuation_snapshots"
CODE = ROOT / "easynla"
LOG = ROOT / "logs/qwen25_rl_vllm_implausible_resume540.log"
EXPECTED_STEPS = tuple(range(540, 781, 20))
POLL_SECONDS = 15


def _link_or_copy(source: str, destination: str) -> str:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
    return destination


def _copy_complete_lines(source: Path, destination: Path) -> None:
    data = source.read_bytes() if source.exists() else b""
    if data and not data.endswith(b"\n"):
        data = data.rsplit(b"\n", 1)[0] + b"\n"
    destination.write_bytes(data)


def _checkpoint_ready(step: int) -> tuple[bool, list[str]]:
    actor = SAVE / f"iter_{step:06d}"
    critic = SAVE / "critic_latest"
    required = (
        actor / "adapter_model.safetensors",
        actor / "adapter_config.json",
        actor / "reference/adapter_model.safetensors",
        actor / "reference/adapter_config.json",
        critic / "saved_at_step.txt",
        critic / "model.safetensors.index.json",
        critic / "value_head.safetensors",
        SAVE / "optim_latest.pt",
        SAVE / "metrics.jsonl",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        return False, missing
    if any(path.stat().st_size == 0 for path in required):
        return False, ["zero-byte required file"]
    if (critic / "saved_at_step.txt").read_text().strip() != str(step):
        return False, ["critic transaction not at requested step"]
    try:
        state = torch.load(SAVE / "optim_latest.pt", map_location="cpu", weights_only=True)
        saved_step = int(state.get("step", -1))
        del state
        gc.collect()
    except Exception as exc:  # transaction may still be replacing the file
        return False, [f"optimizer unreadable: {exc}"]
    if saved_step != step:
        return False, [f"optimizer at step {saved_step}"]
    try:
        rows = [
            json.loads(line)
            for line in (SAVE / "metrics.jsonl").read_text().splitlines()
            if line.strip()
        ]
    except Exception as exc:
        return False, [f"metrics unreadable: {exc}"]
    if len(rows) < step or [int(row["step"]) for row in rows[:step]] != list(range(step)):
        return False, ["metrics are not contiguous through checkpoint"]
    for suffix in ("critic_latest.tmp", "critic_latest.old", "optim_latest.pt.tmp"):
        if (SAVE / suffix).exists():
            return False, [f"active transaction remnant: {suffix}"]
    return True, []


def _snapshot(step: int) -> Path:
    final = SNAPSHOTS / f"step_{step:06d}"
    if (final / "READY.json").is_file():
        return final
    tmp = SNAPSHOTS / f".step_{step:06d}.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    if final.exists():
        shutil.rmtree(final)
    tmp.mkdir(parents=True)

    shutil.copytree(
        SAVE / f"iter_{step:06d}",
        tmp / "actor",
        copy_function=_link_or_copy,
    )
    shutil.copytree(
        SAVE / "critic_latest",
        tmp / "critic",
        copy_function=_link_or_copy,
    )
    _link_or_copy(str(SAVE / "optim_latest.pt"), str(tmp / "optim.pt"))

    metric_lines = [
        line
        for line in (SAVE / "metrics.jsonl").read_text().splitlines()
        if line.strip()
    ][:step]
    (tmp / "metrics.jsonl").write_text("\n".join(metric_lines) + "\n")
    for name in ("nla_meta.yaml", "run_config.yaml"):
        if (SAVE / name).is_file():
            shutil.copy2(SAVE / name, tmp / name)
    _copy_complete_lines(LOG, tmp / "training.log")

    code_dir = tmp / "code"
    code_dir.mkdir()
    code_files = {
        "train_rl_vllm.py": CODE / "nla/train_rl_vllm.py",
        "rl_logging.py": CODE / "nla/utils/rl_logging.py",
        "resume.py": CODE / "nla/utils/resume.py",
        "rl_config.yaml": CODE / "configs/rl_qwen25_7b_ufw20k_implausible.yaml",
        "continuation_launcher.sh": CODE / "scripts/continue_qwen25_7b_ufw20k_implausible_from_540.sh",
    }
    code_sha256 = {}
    for destination, source in code_files.items():
        shutil.copy2(source, code_dir / destination)
        code_sha256[destination] = hashlib.sha256(source.read_bytes()).hexdigest()

    metadata = {
        "schema_version": 1,
        "checkpoint_step": step,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_save_dir": str(SAVE),
        "wandb_run_id": "7jcur2ou",
        "metric_rows": step,
        "critic_saved_at_step": int((SAVE / "critic_latest/saved_at_step.txt").read_text()),
        "optimizer_saved_step": step,
        "code_sha256": code_sha256,
        "storage": "hardlinks_on_same_Athena_filesystem_plus_queued_private_Hugging_Face_upload",
    }
    (tmp / "READY.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    os.rename(tmp, final)
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} snapshotted step {step}", flush=True)
    return final


def main() -> int:
    SNAPSHOTS.mkdir(parents=True, exist_ok=True)
    remaining = set(EXPECTED_STEPS)
    while remaining:
        for step in sorted(remaining):
            final = SNAPSHOTS / f"step_{step:06d}"
            if (final / "READY.json").is_file():
                remaining.remove(step)
                continue
            ready, _ = _checkpoint_ready(step)
            if ready:
                _snapshot(step)
                remaining.remove(step)
        if remaining:
            time.sleep(POLL_SECONDS)
    print("all continuation checkpoints snapshotted", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
