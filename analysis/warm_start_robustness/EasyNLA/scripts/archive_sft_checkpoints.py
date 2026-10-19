"""Continuously upload and SHA-verify immutable SFT checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError

from archive_immutable_rl_history import _atomic_json, _sha256, preflight, verify_remote


class RemoteManifestConflict(RuntimeError):
    pass


def build_manifest(checkpoint: Path, mode: str) -> dict[str, Any]:
    marker_path = checkpoint / "COMPLETE.json"
    if not marker_path.is_file():
        raise RuntimeError(f"checkpoint lacks COMPLETE.json: {checkpoint}")
    marker = json.loads(marker_path.read_text())
    step = int(checkpoint.name.removeprefix("iter_"))
    if int(marker.get("completed_steps", -1)) != step or marker.get("mode") != mode:
        raise RuntimeError(f"checkpoint marker disagrees with path/mode: {checkpoint}")
    required = {
        "COMPLETE.json",
        "trainer_state.pt",
        "metrics.jsonl",
        "run_config.yaml",
        "nla_meta.yaml",
        "training.log",
        "provenance/code_snapshot.tar.gz",
    }
    required.add(
        "adapter_model.safetensors" if mode == "av" else "ar_lora_value_head.safetensors"
    )
    files = []
    for path in sorted(checkpoint.rglob("*")):
        if not path.is_file() or path.name.endswith((".tmp", ".partial", ".old")):
            continue
        files.append(
            {
                "path": str(path.relative_to(checkpoint)),
                "size": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    names = {row["path"] for row in files}
    missing = required - names
    if missing:
        raise RuntimeError(f"incomplete SFT checkpoint {checkpoint}; missing={sorted(missing)}")
    metrics = [
        json.loads(line)
        for line in (checkpoint / "metrics.jsonl").read_text().splitlines()
        if line
    ]
    if not metrics or int(metrics[-1]["step"]) != step - 1:
        raise RuntimeError(f"{checkpoint}: metrics do not end at step {step - 1}")
    return {
        "schema_version": 1,
        "mode": mode,
        "step": step,
        "source": str(checkpoint.resolve()),
        "file_count": len(files),
        "logical_bytes": sum(row["size"] for row in files),
        "files": files,
    }


def _existing_manifest(repo_id: str, remote_dir: str) -> dict[str, Any] | None:
    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=f"{remote_dir}/SHA256_MANIFEST.json",
            repo_type="model",
        )
    except EntryNotFoundError:
        return None
    return json.loads(Path(path).read_text())


def archive_one(
    api: HfApi,
    repo_id: str,
    path_prefix: str,
    checkpoint: Path,
    receipt_dir: Path,
) -> None:
    step = int(checkpoint.name.removeprefix("iter_"))
    receipt = receipt_dir / f"iter_{step:07d}.json"
    if receipt.is_file() and json.loads(receipt.read_text()).get("verified"):
        return
    mode = str(json.loads((checkpoint / "COMPLETE.json").read_text())["mode"])
    manifest = build_manifest(checkpoint, mode)
    remote_dir = f"{path_prefix}/iter_{step:07d}"
    existing = _existing_manifest(repo_id, remote_dir)
    upload = None
    if existing is not None:
        if existing != manifest:
            raise RemoteManifestConflict(
                f"remote immutable SFT archive {remote_dir} has a different manifest"
            )
    else:
        upload = api.upload_folder(
            folder_path=str(checkpoint),
            path_in_repo=remote_dir,
            repo_id=repo_id,
            repo_type="model",
            ignore_patterns=["**/*.tmp", "**/*.partial", "**/*.old"],
            commit_message=f"Archive {mode.upper()} SFT step {step}",
        )
        api.upload_file(
            path_or_fileobj=json.dumps(manifest, indent=2, sort_keys=True).encode(),
            path_in_repo=f"{remote_dir}/SHA256_MANIFEST.json",
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Add manifest for {mode.upper()} SFT step {step}",
        )
    revision = api.repo_info(repo_id=repo_id, repo_type="model").sha
    verify_remote(api, repo_id, remote_dir, manifest, revision)
    _atomic_json(
        receipt,
        {
            "schema_version": 1,
            "verified": True,
            "mode": mode,
            "step": step,
            "repo_id": repo_id,
            "remote_dir": remote_dir,
            "commit": revision,
            "folder_upload_commit": (
                getattr(upload, "oid", None) if upload is not None else None
            ),
            "manifest_sha256": hashlib.sha256(
                json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "verified_at": datetime.now(timezone.utc).isoformat(),
        },
    )


def _complete_checkpoints(save_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in save_dir.glob("iter_[0-9][0-9][0-9][0-9][0-9][0-9][0-9]")
        if path.is_dir() and (path / "COMPLETE.json").is_file()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--mode", choices=("av", "ar"), required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--path-prefix", required=True)
    parser.add_argument("--receipt-dir", required=True)
    parser.add_argument("--status-path", required=True)
    parser.add_argument("--stop-file", required=True)
    parser.add_argument("--stop-step", type=int, default=774)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.poll_seconds <= 60:
        raise ValueError("--poll-seconds must be in [1, 60]")

    save_dir = Path(args.save_dir)
    receipt_dir = Path(args.receipt_dir)
    status_path = Path(args.status_path)
    stop_file = Path(args.stop_file)
    expected = list(range(args.save_every, args.stop_step, args.save_every)) + [args.stop_step]
    api = HfApi()
    preflight(api, args.repo_id, args.path_prefix.split("/")[0])
    if args.preflight_only:
        return

    last_error = None
    while True:
        checkpoints = _complete_checkpoints(save_dir)
        try:
            for checkpoint in checkpoints:
                marker = json.loads((checkpoint / "COMPLETE.json").read_text())
                if marker.get("mode") != args.mode:
                    raise RuntimeError(f"wrong mode checkpoint in {save_dir}: {checkpoint}")
                archive_one(api, args.repo_id, args.path_prefix, checkpoint, receipt_dir)
            last_error = None
        except RemoteManifestConflict as exc:
            _atomic_json(
                status_path,
                {
                    "schema_version": 1,
                    "state": "fatal_remote_conflict",
                    "mode": args.mode,
                    "error": str(exc),
                    "heartbeat_unix": time.time(),
                },
            )
            raise
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"

        verified = {
            int(path.stem.removeprefix("iter_"))
            for path in receipt_dir.glob("iter_*.json")
            if json.loads(path.read_text()).get("verified")
        } if receipt_dir.is_dir() else set()
        local_steps = [int(path.name.removeprefix("iter_")) for path in checkpoints]
        pending = [step for step in local_steps if step not in verified]
        state = "healthy" if last_error is None else "retrying"
        if args.stop_step in verified and all(step in verified for step in expected):
            state = "complete"
        elif stop_file.exists() and not pending:
            state = "stopped_after_failure"
        _atomic_json(
            status_path,
            {
                "schema_version": 1,
                "state": state,
                "mode": args.mode,
                "repo_id": args.repo_id,
                "path_prefix": args.path_prefix,
                "local_complete_steps": local_steps,
                "verified_off_cluster_steps": sorted(verified & set(local_steps)),
                "pending_steps": pending,
                "expected_steps": expected,
                "last_error": last_error,
                "heartbeat_unix": time.time(),
            },
        )
        if state in {"complete", "stopped_after_failure"}:
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
