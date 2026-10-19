"""Continuously archive completed OpenAI Batch records off-cluster.

Each content-addressed Batch directory is immutable once ``usage.json`` exists.
The watcher uploads input/state/raw output/errors/usage to a private Hugging Face
model repo, verifies every SHA-256, and writes an atomic local receipt. Mutable
``cumulative_usage.json`` is archived as content-addressed immutable revisions.
Conflicting remote manifests are fatal; transient network failures are retried.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError

from archive_immutable_rl_history import _atomic_json, _sha256, preflight, verify_remote


_HASH_DIR = re.compile(r"^[0-9a-f]{64}$")
_TERMINAL = {"completed", "failed", "expired", "cancelled"}


class RemoteManifestConflict(RuntimeError):
    pass


def _contains_secret(path: Path, secret: bytes | None) -> bool:
    if not secret:
        return False
    overlap = b""
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            value = overlap + chunk
            if secret in value:
                return True
            tail = len(secret) - 1
            overlap = value[-tail:] if tail > 0 else b""
    return False


def build_batch_manifest(run_dir: Path, secret: bytes | None) -> dict[str, Any]:
    state = json.loads((run_dir / "state.json").read_text())
    if state.get("status") != "completed":
        raise RuntimeError(f"Batch is not completed: {run_dir}")
    required = {"input.jsonl", "state.json", "output.jsonl", "usage.json"}
    files = []
    for path in sorted(run_dir.iterdir()):
        if not path.is_file() or path.name.startswith(".") or path.name.endswith(".tmp"):
            continue
        if _contains_secret(path, secret):
            raise RuntimeError(f"refusing to archive API credential found in {path}")
        files.append(
            {
                "path": path.name,
                "size": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    names = {row["path"] for row in files}
    missing = required - names
    if missing:
        raise RuntimeError(f"completed Batch cache {run_dir} lacks {sorted(missing)}")
    usage = json.loads((run_dir / "usage.json").read_text())
    return {
        "schema_version": 1,
        "content_key": run_dir.name,
        "batch_id": state.get("batch_id"),
        "status": "completed",
        "requests_succeeded": int(usage.get("requests_succeeded", 0)),
        "requests_failed": int(usage.get("requests_failed", 0)),
        "cost_usd": float(usage.get("cost_usd", 0.0)),
        "file_count": len(files),
        "logical_bytes": sum(row["size"] for row in files),
        "files": files,
    }


def build_incomplete_manifest(run_dir: Path, secret: bytes | None) -> dict[str, Any]:
    """Inventory a non-completed Batch without treating its cost as reconciled."""
    state_path = run_dir / "state.json"
    input_path = run_dir / "input.jsonl"
    if not input_path.is_file():
        raise RuntimeError(f"incomplete Batch evidence lacks input: {run_dir}")
    state = json.loads(state_path.read_text()) if state_path.is_file() else {}
    if state.get("status") == "completed" and (run_dir / "usage.json").is_file():
        raise RuntimeError(f"completed Batch was misclassified as incomplete: {run_dir}")
    files = []
    for path in sorted(run_dir.iterdir()):
        if not path.is_file() or path.name.startswith(".") or path.name.endswith(".tmp"):
            continue
        if _contains_secret(path, secret):
            raise RuntimeError(f"refusing to archive API credential found in {path}")
        files.append(
            {"path": path.name, "size": path.stat().st_size, "sha256": _sha256(path)}
        )
    names = {row["path"] for row in files}
    if "input.jsonl" not in names or (state_path.is_file() and "state.json" not in names):
        raise RuntimeError(f"incomplete Batch evidence changed while inventorying: {run_dir}")
    return {
        "schema_version": 1,
        "content_key": run_dir.name,
        "batch_id": state.get("batch_id"),
        "status": state.get("status", "reservation_rejected_or_not_submitted"),
        "cost_reconciled": False,
        "pessimistic_cost_usd": float(state.get("pessimistic_cost_usd", 0.0)),
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


def _archive_directory(
    api: HfApi,
    repo_id: str,
    source: Path,
    remote_dir: str,
    manifest: dict[str, Any],
    receipt: Path,
) -> None:
    if receipt.is_file():
        existing_receipt = json.loads(receipt.read_text())
        if existing_receipt.get("verified"):
            return
    existing = _existing_manifest(repo_id, remote_dir)
    upload = None
    if existing is not None:
        if existing != manifest:
            raise RemoteManifestConflict(
                f"remote immutable API archive {remote_dir} has a different manifest"
            )
    else:
        upload = api.upload_folder(
            folder_path=str(source),
            path_in_repo=remote_dir,
            repo_id=repo_id,
            repo_type="model",
            ignore_patterns=["**/.*", "**/*.tmp"],
            commit_message=f"Archive completed OpenAI Batch {source.name}",
        )
        api.upload_file(
            path_or_fileobj=json.dumps(manifest, indent=2, sort_keys=True).encode(),
            path_in_repo=f"{remote_dir}/SHA256_MANIFEST.json",
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Add manifest for OpenAI Batch {source.name}",
        )
    revision = api.repo_info(repo_id=repo_id, repo_type="model").sha
    verify_remote(api, repo_id, remote_dir, manifest, revision)
    _atomic_json(
        receipt,
        {
            "schema_version": 1,
            "verified": True,
            "content_key": source.name,
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


def archive_ledger(
    api: HfApi,
    repo_id: str,
    path_prefix: str,
    ledger_path: Path,
    receipts: Path,
    secret: bytes | None,
) -> str | None:
    if not ledger_path.is_file():
        return None
    payload = ledger_path.read_bytes()
    if secret and secret in payload:
        raise RuntimeError(f"refusing to archive API credential found in {ledger_path}")
    ledger = json.loads(payload)
    digest = hashlib.sha256(payload).hexdigest()
    batch_count = int(ledger.get("batch_count", -1))
    remote_dir = f"{path_prefix}/ledgers/batch_{batch_count:06d}_{digest[:16]}"
    receipt = receipts / "ledgers" / f"batch_{batch_count:06d}_{digest[:16]}.json"
    if receipt.is_file() and json.loads(receipt.read_text()).get("verified"):
        return digest
    manifest = {
        "schema_version": 1,
        "ledger_sha256": digest,
        "batch_count": batch_count,
        "file_count": 1,
        "logical_bytes": len(payload),
        "files": [
            {
                "path": "cumulative_usage.json",
                "size": len(payload),
                "sha256": digest,
            }
        ],
    }
    existing = _existing_manifest(repo_id, remote_dir)
    if existing is not None and existing != manifest:
        raise RemoteManifestConflict(
            f"remote immutable ledger {remote_dir} has a different manifest"
        )
    if existing is None:
        api.upload_file(
            path_or_fileobj=payload,
            path_in_repo=f"{remote_dir}/cumulative_usage.json",
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Archive OpenAI cumulative ledger {digest[:16]}",
        )
        api.upload_file(
            path_or_fileobj=json.dumps(manifest, indent=2, sort_keys=True).encode(),
            path_in_repo=f"{remote_dir}/SHA256_MANIFEST.json",
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Add manifest for OpenAI ledger {digest[:16]}",
        )
    revision = api.repo_info(repo_id=repo_id, repo_type="model").sha
    verify_remote(api, repo_id, remote_dir, manifest, revision)
    _atomic_json(
        receipt,
        {
            "schema_version": 1,
            "verified": True,
            "ledger_sha256": digest,
            "batch_count": batch_count,
            "repo_id": repo_id,
            "remote_dir": remote_dir,
            "commit": revision,
            "verified_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    return digest


def _local_completed(cache_dir: Path) -> list[Path]:
    completed = []
    for path in sorted(cache_dir.iterdir() if cache_dir.is_dir() else []):
        if not path.is_dir() or not _HASH_DIR.fullmatch(path.name):
            continue
        state_path = path / "state.json"
        if not state_path.is_file():
            continue
        state = json.loads(state_path.read_text())
        if state.get("status") == "completed" and (path / "usage.json").is_file():
            completed.append(path)
    return completed


def _local_incomplete(cache_dir: Path) -> list[Path]:
    incomplete = []
    for path in sorted(cache_dir.iterdir() if cache_dir.is_dir() else []):
        if not path.is_dir() or not _HASH_DIR.fullmatch(path.name):
            continue
        if not (path / "input.jsonl").is_file():
            continue
        state_path = path / "state.json"
        state = json.loads(state_path.read_text()) if state_path.is_file() else {}
        if state.get("status") != "completed" or not (path / "usage.json").is_file():
            incomplete.append(path)
    return incomplete


def archive_stop_ready(
    *,
    stop_requested: bool,
    pending_completed: list[str],
    unarchived_incomplete: list[str],
    latest_ledger_sha: str | None,
    current_ledger_sha: str | None,
) -> bool:
    return (
        stop_requested
        and not pending_completed
        and not unarchived_incomplete
        and current_ledger_sha is not None
        and latest_ledger_sha == current_ledger_sha
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--path-prefix", default="teacher/api_cache")
    parser.add_argument("--receipts-dir", required=True)
    parser.add_argument("--status-path", required=True)
    parser.add_argument("--stop-file", required=True)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.poll_seconds <= 60:
        raise ValueError("--poll-seconds must be in [1, 60]")

    cache_dir = Path(args.cache_dir)
    receipts = Path(args.receipts_dir)
    status_path = Path(args.status_path)
    stop_file = Path(args.stop_file)
    secret_value = os.environ.get("OPENAI_API_KEY_2")
    secret = secret_value.encode() if secret_value else None
    api = HfApi()
    preflight(api, args.repo_id, args.path_prefix.split("/")[0])
    if args.preflight_only:
        return

    last_error = None
    latest_ledger_sha = None
    while True:
        completed = _local_completed(cache_dir)
        incomplete = _local_incomplete(cache_dir)
        incomplete_receipts: dict[str, str] = {}
        try:
            for run_dir in completed:
                receipt = receipts / "batches" / f"{run_dir.name}.json"
                if receipt.is_file() and json.loads(receipt.read_text()).get("verified"):
                    continue
                manifest = build_batch_manifest(run_dir, secret)
                _archive_directory(
                    api,
                    args.repo_id,
                    run_dir,
                    f"{args.path_prefix}/batches/{run_dir.name}",
                    manifest,
                    receipt,
                )
            # Terminal failures are archived immediately. On an explicit stop,
            # also snapshot submitted/in-progress evidence so a local worker
            # failure cannot strand its input/state/reservation only on Athena.
            for run_dir in incomplete:
                state_path = run_dir / "state.json"
                state = json.loads(state_path.read_text()) if state_path.is_file() else {}
                if state.get("status") not in _TERMINAL and not stop_file.exists():
                    continue
                manifest = build_incomplete_manifest(run_dir, secret)
                manifest_digest = hashlib.sha256(
                    json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest()
                receipt = (
                    receipts
                    / "incomplete"
                    / f"{run_dir.name}_{manifest_digest[:16]}.json"
                )
                _archive_directory(
                    api,
                    args.repo_id,
                    run_dir,
                    f"{args.path_prefix}/incomplete/{run_dir.name}/{manifest_digest}",
                    manifest,
                    receipt,
                )
                incomplete_receipts[run_dir.name] = manifest_digest
            latest_ledger_sha = archive_ledger(
                api,
                args.repo_id,
                args.path_prefix,
                cache_dir / "cumulative_usage.json",
                receipts,
                secret,
            )
            last_error = None
        except RemoteManifestConflict as exc:
            _atomic_json(
                status_path,
                {
                    "schema_version": 1,
                    "state": "fatal_remote_conflict",
                    "error": str(exc),
                    "heartbeat_unix": time.time(),
                },
            )
            raise
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"

        verified = {
            path.stem
            for path in (receipts / "batches").glob("*.json")
            if json.loads(path.read_text()).get("verified")
        } if (receipts / "batches").is_dir() else set()
        pending = [path.name for path in completed if path.name not in verified]
        unarchived_incomplete = []
        for run_dir in incomplete:
            try:
                manifest = build_incomplete_manifest(run_dir, secret)
                digest = hashlib.sha256(
                    json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest()
                receipt = receipts / "incomplete" / f"{run_dir.name}_{digest[:16]}.json"
                if not receipt.is_file() or not json.loads(receipt.read_text()).get("verified"):
                    unarchived_incomplete.append(run_dir.name)
                else:
                    incomplete_receipts[run_dir.name] = digest
            except Exception:
                unarchived_incomplete.append(run_dir.name)
        ledger_path = cache_dir / "cumulative_usage.json"
        current_ledger_sha = _sha256(ledger_path) if ledger_path.is_file() else None
        state = "healthy" if last_error is None else "retrying"
        if archive_stop_ready(
            stop_requested=stop_file.exists(),
            pending_completed=pending,
            unarchived_incomplete=unarchived_incomplete,
            latest_ledger_sha=latest_ledger_sha,
            current_ledger_sha=current_ledger_sha,
        ):
            state = "complete"
        _atomic_json(
            status_path,
            {
                "schema_version": 1,
                "state": state,
                "repo_id": args.repo_id,
                "path_prefix": args.path_prefix,
                "completed_local": len(completed),
                "verified_off_cluster": len([p for p in completed if p.name in verified]),
                "pending_content_keys": pending,
                "incomplete_local": [path.name for path in incomplete],
                "archived_incomplete": incomplete_receipts,
                "unarchived_incomplete": unarchived_incomplete,
                "latest_cumulative_ledger_sha256": latest_ledger_sha,
                "current_cumulative_ledger_sha256": current_ledger_sha,
                "last_error": last_error,
                "heartbeat_unix": time.time(),
            },
        )
        if state == "complete":
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
