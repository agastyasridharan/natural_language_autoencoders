"""Upload and independently verify every immutable RL checkpoint transaction.

The trainer publishes ``history/step_NNNNNN`` only after actor/reference,
critic/value head, optimizer, point-in-time metrics/config/provenance/log are all
locally durable. This watcher processes every such directory in order, uploads
it to a private Hugging Face model repository, verifies every remote file by
size and SHA-256 (LFS metadata for large files, download+hash for Git files),
and atomically writes a local receipt. It is safe to restart: existing remote
objects deduplicate and verified receipts suppress repeated work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError


_EXCLUDED_NAMES = {"README.md", ".DS_Store"}


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    parent_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(snapshot: Path, step: int) -> dict[str, Any]:
    complete = snapshot / "COMPLETE.json"
    if not complete.is_file():
        raise RuntimeError(f"snapshot lacks COMPLETE.json: {snapshot}")
    transaction = json.loads(complete.read_text())
    if int(transaction["step"]) != step:
        raise RuntimeError(f"{snapshot}: COMPLETE step {transaction['step']} != {step}")
    files = []
    for path in sorted(snapshot.rglob("*")):
        if not path.is_file() or path.name in _EXCLUDED_NAMES or path.name.endswith(".tmp"):
            continue
        files.append(
            {
                "path": str(path.relative_to(snapshot)),
                "size": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    required = {
        "COMPLETE.json",
        "actor/adapter_model.safetensors",
        "actor/reference/adapter_model.safetensors",
        "critic/value_head.safetensors",
        "critic/saved_at_step.txt",
        "optim.pt",
        "metrics.jsonl",
        "run_config.yaml",
        "nla_meta.yaml",
        "training.log",
        "provenance/code_snapshot.tar.gz",
    }
    paths = {row["path"] for row in files}
    missing = required - paths
    if missing or not any(path.startswith("actor/") for path in paths):
        raise RuntimeError(f"{snapshot}: incomplete immutable transaction; missing={sorted(missing)}")
    return {
        "schema_version": 1,
        "step": step,
        "source": str(snapshot.resolve()),
        "file_count": len(files),
        "logical_bytes": sum(row["size"] for row in files),
        "files": files,
    }


def _remote_sha(item: Any) -> str | None:
    lfs = getattr(item, "lfs", None)
    if lfs is None:
        return None
    if isinstance(lfs, dict):
        return lfs.get("sha256") or lfs.get("oid")
    return getattr(lfs, "sha256", None) or getattr(lfs, "oid", None)


def verify_remote(
    api: HfApi,
    repo_id: str,
    remote_dir: str,
    manifest: dict[str, Any],
    revision: str,
) -> None:
    remote_files: dict[str, Any] = {}
    prefix = remote_dir.rstrip("/") + "/"
    for item in api.list_repo_tree(
        repo_id=repo_id,
        path_in_repo=remote_dir,
        recursive=True,
        expand=True,
        repo_type="model",
        revision=revision,
    ):
        path = getattr(item, "path", None) or getattr(item, "rfilename", None)
        if path and not str(path).endswith("/"):
            rel = str(path)[len(prefix) :] if str(path).startswith(prefix) else str(path)
            remote_files[rel] = item

    for row in manifest["files"]:
        rel = row["path"]
        item = remote_files.get(rel)
        if item is None:
            raise RuntimeError(f"remote snapshot lacks {remote_dir}/{rel}")
        remote_size = getattr(item, "size", None)
        if remote_size is not None and int(remote_size) != int(row["size"]):
            raise RuntimeError(
                f"remote size mismatch {rel}: {remote_size} != {row['size']}"
            )
        lfs_sha = _remote_sha(item)
        if lfs_sha is not None:
            lfs_sha = str(lfs_sha).removeprefix("sha256:")
            if lfs_sha != row["sha256"]:
                raise RuntimeError(f"remote LFS digest mismatch for {rel}")
            continue
        if int(row["size"]) > 10 * 1024 * 1024:
            raise RuntimeError(
                f"large remote file {rel} has no LFS SHA metadata; refusing an "
                "unverified multi-gigabyte download fallback"
            )
        downloaded = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=f"{remote_dir}/{rel}",
                repo_type="model",
                revision=revision,
            )
        )
        if _sha256(downloaded) != row["sha256"]:
            raise RuntimeError(f"downloaded Git-file digest mismatch for {rel}")

    remote_manifest = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=f"{remote_dir}/SHA256_MANIFEST.json",
            repo_type="model",
            revision=revision,
        )
    )
    if json.loads(remote_manifest.read_text()) != manifest:
        raise RuntimeError(f"remote manifest mismatch for {remote_dir}")


def preflight(api: HfApi, repo_id: str, path_prefix: str) -> None:
    api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
    info = api.repo_info(repo_id=repo_id, repo_type="model")
    if not bool(getattr(info, "private", False)):
        raise RuntimeError(f"refusing archive: {repo_id} is not private")
    payload = {
        "schema_version": 1,
        "purpose": "plausible matched warm-start immutable checkpoint archive",
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }
    commit = api.upload_file(
        path_or_fileobj=json.dumps(payload, indent=2, sort_keys=True).encode(),
        path_in_repo=f"{path_prefix}/_preflight.json",
        repo_id=repo_id,
        repo_type="model",
        commit_message="Preflight plausible RL immutable-checkpoint archive",
    )
    revision = getattr(commit, "oid", None) or getattr(commit, "commit_id", None)
    if not revision:
        raise RuntimeError("archive preflight upload returned no commit id")
    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename=f"{path_prefix}/_preflight.json",
        repo_type="model",
        revision=revision,
    )
    if json.loads(Path(downloaded).read_text())["purpose"] != payload["purpose"]:
        raise RuntimeError("archive preflight round-trip verification failed")
    print(f"[archive] private repo preflight verified: {repo_id}@{revision}", flush=True)


def archive_one(
    api: HfApi,
    repo_id: str,
    path_prefix: str,
    snapshot: Path,
    receipts: Path,
) -> None:
    step = int(snapshot.name.removeprefix("step_"))
    receipt_path = receipts / f"step_{step:06d}.json"
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("verified") and int(receipt.get("step", -1)) == step:
            return

    print(f"[archive] hashing immutable step {step}: {snapshot}", flush=True)
    manifest = build_manifest(snapshot, step)
    manifests_dir = receipts.parent / "archive_manifests"
    manifest_path = manifests_dir / f"step_{step:06d}.json"
    _atomic_json(manifest_path, manifest)
    remote_dir = f"{path_prefix}/step_{step:06d}"
    upload = None
    try:
        existing_manifest_path = hf_hub_download(
            repo_id=repo_id,
            filename=f"{remote_dir}/SHA256_MANIFEST.json",
            repo_type="model",
        )
    except EntryNotFoundError:
        existing_manifest_path = None
    if existing_manifest_path is not None:
        existing_manifest = json.loads(Path(existing_manifest_path).read_text())
        if existing_manifest != manifest:
            raise RuntimeError(
                f"remote immutable path {remote_dir} already has a different manifest; "
                "refusing overwrite"
            )
        print(f"[archive] remote step {step} already has the exact manifest; verifying", flush=True)
    else:
        upload = api.upload_folder(
            folder_path=str(snapshot),
            path_in_repo=remote_dir,
            repo_id=repo_id,
            repo_type="model",
            ignore_patterns=["**/README.md", "**/*.tmp", "**/.DS_Store"],
            commit_message=f"Archive plausible RL immutable step {step}",
        )
        api.upload_file(
            path_or_fileobj=str(manifest_path),
            path_in_repo=f"{remote_dir}/SHA256_MANIFEST.json",
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Add SHA-256 manifest for plausible RL step {step}",
        )
    # Verify against the head after the manifest commit, not the preceding
    # folder commit. No other writer should mutate this step path; if one does,
    # exact verification still catches it.
    revision = api.repo_info(repo_id=repo_id, repo_type="model").sha
    verify_remote(api, repo_id, remote_dir, manifest, revision)
    receipt = {
        "schema_version": 1,
        "step": step,
        "verified": True,
        "repo_id": repo_id,
        "remote_dir": remote_dir,
        "commit": revision,
        "folder_upload_commit": getattr(upload, "oid", None) if upload is not None else None,
        "file_count": manifest["file_count"],
        "logical_bytes": manifest["logical_bytes"],
        "manifest_sha256": _sha256(manifest_path),
        "verified_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_json(receipt_path, receipt)
    print(
        f"[archive] VERIFIED off-cluster step {step}: {manifest['file_count']} files, "
        f"{manifest['logical_bytes'] / 1e9:.2f} GB, commit={revision}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--path-prefix", default="rl_history")
    parser.add_argument("--stop-step", type=int, default=780)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--stop-file", default=None)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    if args.poll_seconds <= 0 or args.poll_seconds > 60:
        raise ValueError("--poll-seconds must be in (0, 60]")
    save_dir = Path(args.save_dir)
    history = save_dir / "history"
    receipts = save_dir / "archive_receipts"
    stop_file = Path(args.stop_file) if args.stop_file else save_dir / "ARCHIVER_STOP"
    api = HfApi()
    preflight(api, args.repo_id, args.path_prefix)
    if args.preflight_only:
        return

    while True:
        snapshots = sorted(
            path
            for path in history.glob("step_[0-9][0-9][0-9][0-9][0-9][0-9]")
            if path.is_dir() and (path / "COMPLETE.json").is_file()
        ) if history.is_dir() else []
        for snapshot in snapshots:
            step = int(snapshot.name.removeprefix("step_"))
            receipt = receipts / f"step_{step:06d}.json"
            if receipt.is_file():
                continue
            try:
                archive_one(api, args.repo_id, args.path_prefix, snapshot, receipts)
            except Exception as exc:
                # Stay alive and retry. A transient Hub/proxy failure must not
                # discard the local immutable transaction or silently end the
                # redundancy process.
                print(
                    f"[archive] step {step} upload/verification failed; will retry: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
                break

        final_receipt = receipts / f"step_{args.stop_step:06d}.json"
        if final_receipt.is_file():
            print(f"[archive] final step {args.stop_step} verified; watcher complete", flush=True)
            return
        if stop_file.exists():
            pending = [
                path
                for path in snapshots
                if not (receipts / f"{path.name}.json").is_file()
            ]
            if not pending:
                print("[archive] stop requested and all published snapshots are verified", flush=True)
                return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
