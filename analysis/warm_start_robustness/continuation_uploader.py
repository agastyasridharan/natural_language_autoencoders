"""Upload and byte-verify every immutable continuation snapshot off Athena."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download


ROOT = Path("/data/agastyas/nla_warmstart_robustness/qwen25_7b_seed42")
SNAPSHOTS = ROOT / "checkpoints/implausible/continuation_snapshots"
REPO_ID = "agastyasridharan/nla-qwen25-7b-implausible-warmstart-step540"
EXPECTED_STEPS = tuple(range(540, 781, 20))
POLL_SECONDS = 30
VERIFIED_INDEX = SNAPSHOTS / "VERIFIED_INDEX.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _files(snapshot: Path) -> list[Path]:
    return sorted(
        path
        for path in snapshot.rglob("*")
        if path.is_file()
        and ".cache" not in path.parts
        and path.name
        not in {
            "SHA256SUMS.json",
            "UPLOADED.json",
            "RECEIPT_PUBLISHED.json",
            "UPLOAD_FAILED.txt",
        }
    )


def _manifest(snapshot: Path) -> dict[str, dict[str, Any]]:
    existing = snapshot / "SHA256SUMS.json"
    if existing.is_file():
        return json.loads(existing.read_text())["files"]
    files = {
        str(path.relative_to(snapshot)): {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in _files(snapshot)
    }
    payload = {
        "schema_version": 1,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "files": files,
        "logical_bytes": sum(row["bytes"] for row in files.values()),
    }
    tmp = snapshot / "SHA256SUMS.json.tmp"
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, existing)
    return files


def _lfs_sha(lfs: Any) -> str | None:
    if lfs is None:
        return None
    if isinstance(lfs, dict):
        value = lfs.get("sha256") or lfs.get("oid")
    else:
        value = getattr(lfs, "sha256", None) or getattr(lfs, "oid", None)
    if isinstance(value, str) and value.startswith("sha256:"):
        return value.split(":", 1)[1]
    return value


def _verify(api: HfApi, snapshot: Path, prefix: str, revision: str, files: dict) -> None:
    info = api.model_info(REPO_ID, revision=revision, files_metadata=True)
    assert info.private, "archive repository is not private"
    siblings = {
        sibling.rfilename: sibling
        for sibling in info.siblings
        if sibling.rfilename.startswith(prefix + "/")
    }
    expected = {f"{prefix}/{name}" for name in files} | {f"{prefix}/SHA256SUMS.json"}
    assert set(siblings) == expected, (sorted(expected - set(siblings)), sorted(set(siblings) - expected))
    for name, row in files.items():
        remote_name = f"{prefix}/{name}"
        sibling = siblings[remote_name]
        assert int(sibling.size) == int(row["bytes"]), remote_name
        lfs_sha = _lfs_sha(sibling.lfs)
        if lfs_sha:
            assert lfs_sha == row["sha256"], remote_name
        else:
            downloaded = Path(hf_hub_download(
                repo_id=REPO_ID,
                filename=remote_name,
                revision=revision,
            ))
            assert _sha256(downloaded) == row["sha256"], remote_name
    manifest_name = f"{prefix}/SHA256SUMS.json"
    manifest_local = snapshot / "SHA256SUMS.json"
    assert siblings[manifest_name].size == manifest_local.stat().st_size
    manifest_remote = Path(hf_hub_download(
        repo_id=REPO_ID,
        filename=manifest_name,
        revision=revision,
    ))
    assert _sha256(manifest_remote) == _sha256(manifest_local), manifest_name


def _upload(api: HfApi, step: int, snapshot: Path) -> None:
    files = _manifest(snapshot)
    prefix = f"continuation_snapshots/step_{step:06d}"
    commit = api.upload_folder(
        repo_id=REPO_ID,
        repo_type="model",
        folder_path=str(snapshot),
        path_in_repo=prefix,
        ignore_patterns=[".cache/**", "UPLOADED.json", "UPLOAD_FAILED.txt", "*.tmp"],
        commit_message=f"Archive and verify continuation checkpoint step {step}",
    )
    revision = commit.oid
    _verify(api, snapshot, prefix, revision, files)
    record = {
        "schema_version": 1,
        "checkpoint_step": step,
        "repo_id": REPO_ID,
        "commit": revision,
        "verified_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "verified_files": len(files),
        "verified_logical_bytes": sum(row["bytes"] for row in files.values()),
        "verification": "size_and_SHA256_for_every_file",
    }
    (snapshot / "UPLOADED.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(f"uploaded and verified step {step} at {revision}", flush=True)


def _write_verified_index() -> dict[str, Any]:
    records = []
    for step in EXPECTED_STEPS:
        receipt = SNAPSHOTS / f"step_{step:06d}/UPLOADED.json"
        if receipt.is_file():
            records.append(json.loads(receipt.read_text()))
    payload = {
        "schema_version": 1,
        "updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "repo_id": REPO_ID,
        "verification": "size_and_SHA256_for_every_file",
        "records": records,
    }
    tmp = VERIFIED_INDEX.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, VERIFIED_INDEX)
    return payload


def _publish_receipt(api: HfApi, step: int, snapshot: Path) -> None:
    """Replicate the compact verification evidence after payload verification.

    The receipt points to the immutable payload commit.  The later receipt
    commit deliberately is not self-referential; its ID is retained in the
    Athena-side marker and the dashboard's synced copy.
    """
    index = _write_verified_index()
    commit = api.create_commit(
        repo_id=REPO_ID,
        repo_type="model",
        operations=[
            CommitOperationAdd(
                path_in_repo=f"continuation_snapshots/step_{step:06d}/UPLOADED.json",
                path_or_fileobj=str(snapshot / "UPLOADED.json"),
            ),
            CommitOperationAdd(
                path_in_repo="continuation_snapshots/VERIFIED_INDEX.json",
                path_or_fileobj=str(VERIFIED_INDEX),
            ),
        ],
        commit_message=f"Record verified continuation checkpoint step {step}",
    )
    marker = {
        "schema_version": 1,
        "checkpoint_step": step,
        "receipt_commit": commit.oid,
        "published_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "verified_records_in_index": len(index["records"]),
    }
    marker_path = snapshot / "RECEIPT_PUBLISHED.json"
    marker_path.write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n")
    print(f"published verification receipt for step {step} at {commit.oid}", flush=True)


def main() -> int:
    api = HfApi()
    info = api.model_info(REPO_ID)
    assert info.private, "refusing to upload restart state to a public repository"
    completed: set[int] = set()
    while len(completed) < len(EXPECTED_STEPS):
        for step in EXPECTED_STEPS:
            snapshot = SNAPSHOTS / f"step_{step:06d}"
            uploaded = snapshot / "UPLOADED.json"
            if uploaded.is_file():
                receipt_published = snapshot / "RECEIPT_PUBLISHED.json"
                if not receipt_published.is_file():
                    try:
                        _publish_receipt(api, step, snapshot)
                    except Exception as exc:
                        (snapshot / "UPLOAD_FAILED.txt").write_text(
                            f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} "
                            f"{type(exc).__name__}: receipt publication failed: {exc}\n"
                        )
                        print(
                            f"step {step} receipt publication failed; will retry: {exc}",
                            flush=True,
                        )
                        time.sleep(POLL_SECONDS)
                        break
                    (snapshot / "UPLOAD_FAILED.txt").unlink(missing_ok=True)
                completed.add(step)
                continue
            if not (snapshot / "READY.json").is_file():
                continue
            try:
                _upload(api, step, snapshot)
                completed.add(step)
                (snapshot / "UPLOAD_FAILED.txt").unlink(missing_ok=True)
            except Exception as exc:
                (snapshot / "UPLOAD_FAILED.txt").write_text(
                    f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {type(exc).__name__}: {exc}\n"
                )
                print(f"step {step} upload/verification failed; will retry: {exc}", flush=True)
                time.sleep(POLL_SECONDS)
                break
        if len(completed) < len(EXPECTED_STEPS):
            time.sleep(POLL_SECONDS)
    print("all continuation snapshots uploaded and verified", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
