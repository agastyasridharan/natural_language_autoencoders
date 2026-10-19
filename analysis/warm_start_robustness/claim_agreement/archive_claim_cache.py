"""Periodically archive claim-agreement cache/results to a private HF repo.

Snapshots are content addressed, include every non-lock cache/result file, and
are packaged as deterministic tar.gz bundles.  The watcher verifies exact
remote size/SHA receipts and is restartable: an unchanged local tree maps to
the same snapshot ID, while verified local receipts suppress duplicate work.
No OpenAI endpoint is used by this module.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import shutil
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

from .api_cache import _atomic_json


MANIFEST_SCHEMA = "nla.claim_agreement.archive_manifest.v1"
RECEIPT_SCHEMA = "nla.claim_agreement.archive_receipt.v1"
STATUS_SCHEMA = "nla.claim_agreement.archive_status.v1"
_ROOT_NAMES = ("cache", "results")


class TreeChanged(RuntimeError):
    """A source file changed while a point-in-time snapshot was materialized."""


class RemoteConflict(RuntimeError):
    """A content-addressed remote path contains different content."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _excluded(path: Path) -> bool:
    name = path.name
    return (
        name.endswith(".lock")
        or name.endswith(".tmp")
        or ".tmp." in name
        or name.startswith(".nfs")
        or name in {".DS_Store"}
    )


def _secret_values() -> list[bytes]:
    values = []
    for name in ("OPENAI_API_KEY_2", "OPENAI_API_KEY"):
        value = os.environ.get(name)
        if value:
            encoded = value.encode()
            if encoded not in values:
                values.append(encoded)
    return values


def _hash_stable_file(path: Path, secrets: Iterable[bytes]) -> tuple[str, int]:
    before = path.stat()
    digest = hashlib.sha256()
    overlap = b""
    maximum_secret = max((len(secret) for secret in secrets), default=0)
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
            combined = overlap + chunk
            for secret in secrets:
                if secret in combined:
                    raise RuntimeError(f"refusing to archive API credential found in {path}")
            overlap = combined[-(maximum_secret - 1):] if maximum_secret > 1 else b""
    after = path.stat()
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after:
        raise TreeChanged(f"source file changed while hashing: {path}")
    return digest.hexdigest(), int(after.st_size)


def _cache_summary(cache_root: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "requests": 0,
        "attempts": 0,
        "results": 0,
        "preflight_journals": 0,
        "response_journals": 0,
        "active_budget_reservations": None,
        "budget_charges": None,
        "accounted_spend_usd": None,
        "maximum_committed_usd": None,
    }
    if not cache_root.is_dir():
        return summary
    summary["requests"] = sum(1 for _ in cache_root.glob("*/*/request.json"))
    summary["attempts"] = sum(1 for _ in cache_root.glob("*/*/attempts/*.json"))
    summary["results"] = sum(1 for _ in cache_root.glob("*/*/result.json"))
    summary["preflight_journals"] = sum(
        1 for _ in cache_root.glob("*/*/journal/*_preflight.json")
    )
    summary["response_journals"] = sum(
        1 for _ in cache_root.glob("*/*/journal/*_response.json")
    )
    ledger_path = cache_root / "budget_ledger.json"
    if ledger_path.is_file():
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        summary.update({
            "active_budget_reservations": len(ledger.get("reservations") or {}),
            "budget_charges": len(ledger.get("charges") or {}),
            "accounted_spend_usd": ledger.get("accounted_spend_usd"),
            "maximum_committed_usd": ledger.get("maximum_committed_usd"),
        })
    return summary


def build_manifest(
    cache_root: Path,
    results_root: Path,
    *,
    protocol_sha256: str,
    secrets: Iterable[bytes] = (),
) -> dict[str, Any]:
    """Inventory a stable point-in-time tree with exact per-file receipts."""
    secret_list = list(secrets)
    files: list[dict[str, Any]] = []
    for root_name, root in (("cache", cache_root), ("results", results_root)):
        if not root.exists():
            continue
        if root.is_symlink() or not root.is_dir():
            raise RuntimeError(f"archive source must be a real directory: {root}")
        for path in sorted(root.rglob("*")):
            if path.is_symlink():
                raise RuntimeError(f"refusing symlink in archive source: {path}")
            if not path.is_file() or _excluded(path):
                continue
            relative = path.relative_to(root)
            digest, size = _hash_stable_file(path, secret_list)
            files.append({
                "root": root_name,
                "path": relative.as_posix(),
                "bytes": size,
                "sha256": digest,
            })
    files.sort(key=lambda row: (str(row["root"]), str(row["path"])))
    payload = {
        "schema_version": MANIFEST_SCHEMA,
        "protocol_sha256": protocol_sha256,
        "file_count": len(files),
        "logical_bytes": sum(int(row["bytes"]) for row in files),
        "cache_summary": _cache_summary(cache_root),
        "files": files,
    }
    snapshot_id = hashlib.sha256(_canonical(payload)).hexdigest()
    return {**payload, "snapshot_id": snapshot_id}


def _manifest_bytes(manifest: Mapping[str, Any]) -> bytes:
    return (json.dumps(dict(manifest), ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise RuntimeError("unsupported claim archive manifest schema")
    payload = dict(manifest)
    snapshot_id = str(payload.pop("snapshot_id", ""))
    if hashlib.sha256(_canonical(payload)).hexdigest() != snapshot_id:
        raise RuntimeError("claim archive snapshot ID does not match manifest content")
    files = manifest.get("files")
    if not isinstance(files, list) or len(files) != int(manifest.get("file_count", -1)):
        raise RuntimeError("claim archive manifest file count is inconsistent")
    if sum(int(record["bytes"]) for record in files) != int(manifest.get("logical_bytes", -1)):
        raise RuntimeError("claim archive manifest byte count is inconsistent")
    names = []
    for record in files:
        if record.get("root") not in _ROOT_NAMES:
            raise RuntimeError("claim archive manifest contains an unknown root")
        relative = PurePosixPath(str(record.get("path") or ""))
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise RuntimeError("claim archive manifest contains an unsafe path")
        digest = str(record.get("sha256") or "")
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise RuntimeError("claim archive manifest contains an invalid SHA-256")
        names.append((record["root"], relative.as_posix()))
    if len(names) != len(set(names)):
        raise RuntimeError("claim archive manifest contains duplicate paths")


def _source_path(
    record: Mapping[str, Any], cache_root: Path, results_root: Path
) -> Path:
    roots = {"cache": cache_root, "results": results_root}
    root_name = str(record["root"])
    if root_name not in roots:
        raise RuntimeError(f"unknown archive root {root_name}")
    relative = PurePosixPath(str(record["path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(f"unsafe archive path {relative}")
    return roots[root_name].joinpath(*relative.parts)


def build_bundle(
    manifest: Mapping[str, Any],
    cache_root: Path,
    results_root: Path,
    staging_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    """Create a deterministic, self-describing tar.gz and verify sources again."""
    validate_manifest(manifest)
    snapshot_id = str(manifest["snapshot_id"])
    staging_dir.mkdir(parents=True, exist_ok=True)
    target = staging_dir / f"{snapshot_id}.tar.gz"
    if target.is_file():
        verify_bundle(target, manifest)
        return target, {"bytes": target.stat().st_size, "sha256": _sha256(target)}
    temporary = target.with_name(target.name + f".tmp.{os.getpid()}")
    try:
        with temporary.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
                with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
                    manifest_payload = _manifest_bytes(manifest)
                    info = tarfile.TarInfo("SHA256_MANIFEST.json")
                    info.size = len(manifest_payload)
                    info.mtime = 0
                    info.mode = 0o600
                    archive.addfile(info, io.BytesIO(manifest_payload))
                    for record in manifest["files"]:
                        path = _source_path(record, cache_root, results_root)
                        before = path.stat()
                        if before.st_size != int(record["bytes"]):
                            raise TreeChanged(f"source size changed before bundling: {path}")
                        info = tarfile.TarInfo(
                            f"{record['root']}/{PurePosixPath(str(record['path'])).as_posix()}"
                        )
                        info.size = int(record["bytes"])
                        info.mtime = 0
                        info.mode = 0o600
                        with path.open("rb") as handle:
                            archive.addfile(info, handle)
                        after = path.stat()
                        if (
                            before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns
                        ) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
                            raise TreeChanged(f"source changed while bundling: {path}")
                        if _sha256(path) != record["sha256"]:
                            raise TreeChanged(f"source digest changed while bundling: {path}")
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(temporary, target)
        parent_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    finally:
        temporary.unlink(missing_ok=True)
    verify_bundle(target, manifest)
    return target, {"bytes": target.stat().st_size, "sha256": _sha256(target)}


def _safe_destination(root: Path, relative: str) -> Path:
    parsed = PurePosixPath(relative)
    if parsed.is_absolute() or ".." in parsed.parts:
        raise RuntimeError(f"unsafe restore path {relative}")
    return root.joinpath(*parsed.parts)


def verify_bundle(bundle: Path, manifest: Mapping[str, Any]) -> None:
    """Verify every member directly from the bundle before upload or restore."""
    validate_manifest(manifest)
    expected = {
        f"{record['root']}/{record['path']}": record for record in manifest["files"]
    }
    with tarfile.open(bundle, "r:gz") as archive:
        members = {member.name: member for member in archive.getmembers() if member.isfile()}
        if set(members) != {"SHA256_MANIFEST.json", *expected}:
            raise RuntimeError("bundle members do not exactly match the manifest")
        manifest_stream = archive.extractfile(members["SHA256_MANIFEST.json"])
        if manifest_stream is None or json.load(manifest_stream) != dict(manifest):
            raise RuntimeError("embedded archive manifest differs from verified manifest")
        for member_name, record in expected.items():
            member = members[member_name]
            if member.size != int(record["bytes"]):
                raise RuntimeError(f"bundle member size mismatch: {member_name}")
            stream = archive.extractfile(member)
            if stream is None:
                raise RuntimeError(f"bundle member is unreadable: {member_name}")
            digest = hashlib.sha256()
            while chunk := stream.read(8 * 1024 * 1024):
                digest.update(chunk)
            if digest.hexdigest() != record["sha256"]:
                raise RuntimeError(f"bundle member SHA-256 mismatch: {member_name}")


def restore_bundle(
    bundle: Path,
    manifest: Mapping[str, Any],
    cache_destination: Path,
    results_destination: Path,
) -> None:
    """Restore and verify a snapshot into two empty destination trees."""
    for destination in (cache_destination, results_destination):
        if destination.exists() and any(destination.iterdir()):
            raise RuntimeError(f"restore destination is not empty: {destination}")
        destination.mkdir(parents=True, exist_ok=True)
    verify_bundle(bundle, manifest)
    expected = {
        f"{record['root']}/{record['path']}": record for record in manifest["files"]
    }
    with tarfile.open(bundle, "r:gz") as archive:
        members = {member.name: member for member in archive.getmembers() if member.isfile()}
        if set(members) != {"SHA256_MANIFEST.json", *expected}:
            raise RuntimeError("bundle members do not exactly match the manifest")
        embedded = json.loads(archive.extractfile(members["SHA256_MANIFEST.json"]).read())
        if embedded != dict(manifest):
            raise RuntimeError("embedded archive manifest differs from verified manifest")
        roots = {"cache": cache_destination, "results": results_destination}
        for member_name, record in expected.items():
            destination = _safe_destination(roots[str(record["root"])], str(record["path"]))
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(destination.name + ".tmp.restore")
            with archive.extractfile(members[member_name]) as source, temporary.open("wb") as sink:
                shutil.copyfileobj(source, sink, length=8 * 1024 * 1024)
                sink.flush()
                os.fsync(sink.fileno())
            if temporary.stat().st_size != int(record["bytes"]) or _sha256(temporary) != record["sha256"]:
                temporary.unlink(missing_ok=True)
                raise RuntimeError(f"restored file failed verification: {member_name}")
            os.replace(temporary, destination)


def _remote_paths(path_prefix: str, snapshot_id: str) -> dict[str, str]:
    base = f"{path_prefix.strip('/')}/snapshots/{snapshot_id}"
    return {
        "bundle": f"{base}/snapshot.tar.gz",
        "manifest": f"{base}/SHA256_MANIFEST.json",
        "record": f"{base}/ARCHIVE_RECORD.json",
    }


def _download_json(
    *, repo_id: str, repo_type: str, filename: str, revision: str | None, cache_dir: Path
) -> dict[str, Any] | None:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError, HfHubHTTPError

    try:
        path = hf_hub_download(
            repo_id=repo_id,
            repo_type=repo_type,
            filename=filename,
            revision=revision,
            cache_dir=str(cache_dir),
        )
    except (EntryNotFoundError, HfHubHTTPError) as error:
        status = getattr(getattr(error, "response", None), "status_code", None)
        if isinstance(error, EntryNotFoundError) or status == 404:
            return None
        raise
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"remote JSON is not an object: {filename}")
    return value


def ensure_private_repo(api: Any, repo_id: str, repo_type: str) -> None:
    api.create_repo(repo_id=repo_id, repo_type=repo_type, private=True, exist_ok=True)
    info = api.repo_info(repo_id=repo_id, repo_type=repo_type)
    if not bool(getattr(info, "private", False)):
        raise RuntimeError(f"refusing archive because Hugging Face repo is not private: {repo_id}")


def archive_snapshot(
    api: Any,
    *,
    repo_id: str,
    repo_type: str,
    path_prefix: str,
    manifest: Mapping[str, Any],
    bundle: Path,
    bundle_receipt: Mapping[str, Any],
    receipts_dir: Path,
    hf_cache_dir: Path,
) -> dict[str, Any]:
    snapshot_id = str(manifest["snapshot_id"])
    local_receipt_path = receipts_dir / "snapshots" / f"{snapshot_id}.json"
    if local_receipt_path.is_file():
        local = json.loads(local_receipt_path.read_text(encoding="utf-8"))
        if local.get("verified") and local.get("bundle_sha256") == bundle_receipt["sha256"]:
            return local
    paths = _remote_paths(path_prefix, snapshot_id)
    manifest_payload = _manifest_bytes(manifest)
    manifest_sha = hashlib.sha256(manifest_payload).hexdigest()
    remote_record = {
        "schema_version": RECEIPT_SCHEMA,
        "snapshot_id": snapshot_id,
        "manifest_bytes": len(manifest_payload),
        "manifest_sha256": manifest_sha,
        "bundle_bytes": int(bundle_receipt["bytes"]),
        "bundle_sha256": str(bundle_receipt["sha256"]),
        "file_count": int(manifest["file_count"]),
        "logical_bytes": int(manifest["logical_bytes"]),
    }
    existing = _download_json(
        repo_id=repo_id,
        repo_type=repo_type,
        filename=paths["record"],
        revision=None,
        cache_dir=hf_cache_dir,
    )
    upload_commit = None
    if existing is not None and existing != remote_record:
        raise RemoteConflict(f"remote archive record conflicts at {paths['record']}")
    if existing is None:
        api.upload_file(
            path_or_fileobj=str(bundle),
            path_in_repo=paths["bundle"],
            repo_id=repo_id,
            repo_type=repo_type,
            commit_message=f"Archive claim cache snapshot {snapshot_id[:16]}",
        )
        api.upload_file(
            path_or_fileobj=manifest_payload,
            path_in_repo=paths["manifest"],
            repo_id=repo_id,
            repo_type=repo_type,
            commit_message=f"Add claim snapshot manifest {snapshot_id[:16]}",
        )
        upload_commit = api.upload_file(
            path_or_fileobj=_canonical(remote_record),
            path_in_repo=paths["record"],
            repo_id=repo_id,
            repo_type=repo_type,
            commit_message=f"Verify claim snapshot record {snapshot_id[:16]}",
        )
    revision = str(api.repo_info(repo_id=repo_id, repo_type=repo_type).sha)
    verified_record = _download_json(
        repo_id=repo_id,
        repo_type=repo_type,
        filename=paths["record"],
        revision=revision,
        cache_dir=hf_cache_dir,
    )
    verified_manifest = _download_json(
        repo_id=repo_id,
        repo_type=repo_type,
        filename=paths["manifest"],
        revision=revision,
        cache_dir=hf_cache_dir,
    )
    if verified_record != remote_record or verified_manifest != dict(manifest):
        raise RuntimeError("remote claim snapshot metadata failed exact verification")
    # Bundle verification is intentionally an independent download. Claim JSON
    # caches are expected to remain small enough that avoiding an unverifiable
    # metadata-only shortcut is preferable to saving bandwidth.
    from huggingface_hub import hf_hub_download

    downloaded_bundle = Path(hf_hub_download(
        repo_id=repo_id,
        repo_type=repo_type,
        filename=paths["bundle"],
        revision=revision,
        cache_dir=str(hf_cache_dir),
    ))
    if downloaded_bundle.stat().st_size != remote_record["bundle_bytes"]:
        raise RuntimeError("remote claim snapshot bundle size mismatch")
    if _sha256(downloaded_bundle) != remote_record["bundle_sha256"]:
        raise RuntimeError("remote claim snapshot bundle SHA-256 mismatch")
    receipt = {
        **remote_record,
        "verified": True,
        "repo_id": repo_id,
        "repo_type": repo_type,
        "remote_paths": paths,
        "commit": revision,
        "upload_commit": (
            getattr(upload_commit, "oid", None) or getattr(upload_commit, "commit_id", None)
            if upload_commit is not None else None
        ),
        "verified_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_json(local_receipt_path, receipt)
    return receipt


def _status_payload(
    *, state: str, manifest: Mapping[str, Any] | None, receipt: Mapping[str, Any] | None,
    last_error: str | None, repo_id: str, repo_type: str, path_prefix: str,
) -> dict[str, Any]:
    return {
        "schema_version": STATUS_SCHEMA,
        "state": state,
        "repo_id": repo_id,
        "repo_type": repo_type,
        "path_prefix": path_prefix,
        "snapshot_id": None if manifest is None else manifest.get("snapshot_id"),
        "local_file_count": None if manifest is None else manifest.get("file_count"),
        "local_logical_bytes": None if manifest is None else manifest.get("logical_bytes"),
        "cache_summary": None if manifest is None else manifest.get("cache_summary"),
        "verified_off_cluster": bool(receipt and receipt.get("verified")),
        "remote_commit": None if receipt is None else receipt.get("commit"),
        "last_error": last_error,
        "heartbeat_unix": time.time(),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("once", "watch"):
        command = commands.add_parser(name)
        command.add_argument("--cache-dir", type=Path, required=True)
        command.add_argument("--results-dir", type=Path, required=True)
        command.add_argument("--protocol", type=Path, required=True)
        command.add_argument("--repo-id", required=True)
        command.add_argument("--repo-type", choices=("dataset", "model"), default="dataset")
        command.add_argument("--path-prefix", default="claim_agreement")
        command.add_argument("--receipts-dir", type=Path, required=True)
        command.add_argument("--staging-dir", type=Path, required=True)
        command.add_argument("--hf-cache-dir", type=Path, required=True)
        command.add_argument("--status-path", type=Path, required=True)
        command.add_argument("--stop-file", type=Path)
        command.add_argument("--poll-seconds", type=float, default=30.0)
    restore = commands.add_parser("restore")
    restore.add_argument("--repo-id", required=True)
    restore.add_argument("--repo-type", choices=("dataset", "model"), default="dataset")
    restore.add_argument("--path-prefix", default="claim_agreement")
    restore.add_argument("--snapshot-id", required=True)
    restore.add_argument("--hf-cache-dir", type=Path, required=True)
    restore.add_argument("--cache-destination", type=Path, required=True)
    restore.add_argument("--results-destination", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command in ("once", "watch") and not 1 <= args.poll_seconds <= 60:
        parser.error("--poll-seconds must be in [1, 60]")
    return args


def _is_within(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _validate_layout(args: argparse.Namespace) -> None:
    sources = (args.cache_dir, args.results_dir)
    if args.cache_dir.resolve() == args.results_dir.resolve():
        raise RuntimeError("cache and results directories must be distinct")
    external_paths = (
        args.receipts_dir,
        args.staging_dir,
        args.hf_cache_dir,
        args.status_path,
    )
    for external in external_paths:
        if any(_is_within(external, source) for source in sources):
            raise RuntimeError(
                f"archive runtime path must be outside cache/results to avoid snapshot churn: {external}"
            )


def _restore_remote(args: argparse.Namespace) -> int:
    paths = _remote_paths(args.path_prefix, args.snapshot_id)
    record = _download_json(
        repo_id=args.repo_id, repo_type=args.repo_type, filename=paths["record"],
        revision=None, cache_dir=args.hf_cache_dir,
    )
    manifest = _download_json(
        repo_id=args.repo_id, repo_type=args.repo_type, filename=paths["manifest"],
        revision=None, cache_dir=args.hf_cache_dir,
    )
    if record is None or manifest is None or record.get("snapshot_id") != args.snapshot_id:
        raise RuntimeError("remote snapshot metadata is missing or inconsistent")
    from huggingface_hub import hf_hub_download

    bundle = Path(hf_hub_download(
        repo_id=args.repo_id, repo_type=args.repo_type, filename=paths["bundle"],
        cache_dir=str(args.hf_cache_dir),
    ))
    if bundle.stat().st_size != record["bundle_bytes"] or _sha256(bundle) != record["bundle_sha256"]:
        raise RuntimeError("remote bundle failed receipt verification")
    restore_bundle(bundle, manifest, args.cache_destination, args.results_destination)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "restore":
        return _restore_remote(args)
    _validate_layout(args)
    protocol_sha = _sha256(args.protocol)
    from huggingface_hub import HfApi

    api = HfApi()
    ensure_private_repo(api, args.repo_id, args.repo_type)
    last_error = None
    latest_manifest: dict[str, Any] | None = None
    latest_receipt: dict[str, Any] | None = None
    while True:
        try:
            latest_manifest = build_manifest(
                args.cache_dir,
                args.results_dir,
                protocol_sha256=protocol_sha,
                secrets=_secret_values(),
            )
            if latest_manifest["file_count"]:
                bundle, bundle_receipt = build_bundle(
                    latest_manifest, args.cache_dir, args.results_dir, args.staging_dir
                )
                latest_receipt = archive_snapshot(
                    api,
                    repo_id=args.repo_id,
                    repo_type=args.repo_type,
                    path_prefix=args.path_prefix,
                    manifest=latest_manifest,
                    bundle=bundle,
                    bundle_receipt=bundle_receipt,
                    receipts_dir=args.receipts_dir,
                    hf_cache_dir=args.hf_cache_dir,
                )
                state = "healthy"
            else:
                latest_receipt = None
                state = "waiting_for_artifacts"
            last_error = None
        except RemoteConflict as error:
            last_error = f"{type(error).__name__}: {error}"
            _atomic_json(
                args.status_path,
                _status_payload(
                    state="fatal_remote_conflict", manifest=latest_manifest,
                    receipt=latest_receipt, last_error=last_error,
                    repo_id=args.repo_id, repo_type=args.repo_type,
                    path_prefix=args.path_prefix,
                ),
            )
            raise
        except Exception as error:
            last_error = f"{type(error).__name__}: {error}"
            state = "retrying"
        _atomic_json(
            args.status_path,
            _status_payload(
                state=state, manifest=latest_manifest, receipt=latest_receipt,
                last_error=last_error, repo_id=args.repo_id,
                repo_type=args.repo_type, path_prefix=args.path_prefix,
            ),
        )
        if args.command == "once":
            return 0 if last_error is None else 1
        if args.stop_file and args.stop_file.exists() and last_error is None:
            final_manifest = build_manifest(
                args.cache_dir,
                args.results_dir,
                protocol_sha256=protocol_sha,
                secrets=_secret_values(),
            )
            if (
                latest_receipt
                and latest_receipt.get("verified")
                and final_manifest["snapshot_id"] == latest_manifest["snapshot_id"]
            ):
                _atomic_json(
                    args.status_path,
                    _status_payload(
                        state="complete", manifest=latest_manifest,
                        receipt=latest_receipt, last_error=None,
                        repo_id=args.repo_id, repo_type=args.repo_type,
                        path_prefix=args.path_prefix,
                    ),
                )
                return 0
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
