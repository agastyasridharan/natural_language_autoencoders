"""Publish and byte-verify the non-secret continuation control plane offsite."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download


SOURCE = Path("/data/agastyas/nla_warmstart_robustness/ops/control_source")
REPO_ID = "agastyasridharan/nla-qwen25-7b-implausible-warmstart-step540"
PREFIX = "run_control/continuation_20260828"
FILES = (
    "RUN_MANIFEST.yaml",
    "STATUS.md",
    "build_dashboard.py",
    "sync_live_dashboard.sh",
    "serve_dashboard.sh",
    "continuation_snapshotter.py",
    "continuation_uploader.py",
    "continuation_watchdog.py",
    "continue_qwen25_7b_ufw20k_implausible_from_540.sh",
    "publish_control_bundle.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, value: object) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def main() -> int:
    missing = [name for name in FILES if not (SOURCE / name).is_file()]
    if missing:
        raise FileNotFoundError(f"control bundle is incomplete: {missing}")
    records = {
        name: {"bytes": (SOURCE / name).stat().st_size, "sha256": _sha256(SOURCE / name)}
        for name in FILES
    }
    manifest = {
        "schema_version": 1,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "repo_id": REPO_ID,
        "path_in_repo": PREFIX,
        "files": records,
    }
    manifest_path = SOURCE / "CONTROL_SHA256SUMS.json"
    _atomic_write_json(manifest_path, manifest)

    api = HfApi()
    assert api.model_info(REPO_ID).private, "refusing to publish control state to a public repo"
    operations = [
        CommitOperationAdd(
            path_in_repo=f"{PREFIX}/{name}",
            path_or_fileobj=str(SOURCE / name),
        )
        for name in FILES
    ]
    operations.append(CommitOperationAdd(
        path_in_repo=f"{PREFIX}/{manifest_path.name}",
        path_or_fileobj=str(manifest_path),
    ))
    commit = api.create_commit(
        repo_id=REPO_ID,
        repo_type="model",
        operations=operations,
        commit_message="Archive continuation control plane and durability manifest",
    )
    expected = {**records, manifest_path.name: {
        "bytes": manifest_path.stat().st_size,
        "sha256": _sha256(manifest_path),
    }}
    info = api.model_info(REPO_ID, revision=commit.oid, files_metadata=True)
    siblings = {
        row.rfilename: row
        for row in info.siblings
        if row.rfilename.startswith(PREFIX + "/")
    }
    assert set(siblings) == {f"{PREFIX}/{name}" for name in expected}
    for name, record in expected.items():
        remote_name = f"{PREFIX}/{name}"
        assert int(siblings[remote_name].size) == int(record["bytes"]), remote_name
        downloaded = Path(hf_hub_download(
            repo_id=REPO_ID,
            filename=remote_name,
            revision=commit.oid,
        ))
        assert _sha256(downloaded) == record["sha256"], remote_name
    receipt = {
        "schema_version": 1,
        "commit": commit.oid,
        "verified_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "verified_files": len(expected),
        "verification": "downloaded_size_and_SHA256_for_every_file",
    }
    _atomic_write_json(SOURCE / "CONTROL_PUBLISHED.json", receipt)
    print(json.dumps(receipt, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
