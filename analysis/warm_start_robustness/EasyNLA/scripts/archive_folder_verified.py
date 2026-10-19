"""Upload one durable folder to a private HF repo and SHA-256 verify it."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError

from archive_immutable_rl_history import (
    _atomic_json,
    _sha256,
    preflight,
    verify_remote,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--remote-dir", required=True)
    parser.add_argument("--receipt", required=True)
    args = parser.parse_args()

    source = Path(args.source)
    if not source.is_dir():
        raise FileNotFoundError(source)
    bad = [
        path
        for path in source.rglob("*")
        if path.name.endswith((".tmp", ".partial", ".old"))
    ]
    if bad:
        raise RuntimeError(f"refusing to archive transaction remnants: {bad[:5]}")

    receipt = Path(args.receipt)
    if receipt.is_file() and json.loads(receipt.read_text()).get("verified"):
        print(f"[archive-once] existing verified receipt: {receipt}")
        return

    files = []
    for path in sorted(source.rglob("*")):
        if not path.is_file() or path.name in {"README.md", ".DS_Store"}:
            continue
        files.append(
            {
                "path": str(path.relative_to(source)),
                "size": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    if not files:
        raise RuntimeError(f"source folder is empty: {source}")
    manifest = {
        "schema_version": 1,
        "source": str(source.resolve()),
        "file_count": len(files),
        "logical_bytes": sum(row["size"] for row in files),
        "files": files,
    }
    manifest_path = receipt.with_name(receipt.stem + ".manifest.json")
    _atomic_json(manifest_path, manifest)

    api = HfApi()
    preflight(api, args.repo_id, args.remote_dir.split("/")[0])
    upload = None
    try:
        existing_path = hf_hub_download(
            repo_id=args.repo_id,
            filename=f"{args.remote_dir}/SHA256_MANIFEST.json",
            repo_type="model",
        )
    except EntryNotFoundError:
        existing_path = None
    if existing_path is not None:
        existing = json.loads(Path(existing_path).read_text())
        if existing != manifest:
            raise RuntimeError(
                f"remote archive path {args.remote_dir} already has a different "
                "manifest; refusing overwrite"
            )
        print(f"[archive-once] exact remote manifest exists; verifying {args.remote_dir}")
    else:
        upload = api.upload_folder(
            folder_path=str(source),
            path_in_repo=args.remote_dir,
            repo_id=args.repo_id,
            repo_type="model",
            ignore_patterns=["**/README.md", "**/*.tmp", "**/*.partial", "**/*.old"],
            commit_message=f"Archive {args.remote_dir}",
        )
        api.upload_file(
            path_or_fileobj=str(manifest_path),
            path_in_repo=f"{args.remote_dir}/SHA256_MANIFEST.json",
            repo_id=args.repo_id,
            repo_type="model",
            commit_message=f"Add SHA-256 manifest for {args.remote_dir}",
        )
    revision = api.repo_info(repo_id=args.repo_id, repo_type="model").sha
    verify_remote(api, args.repo_id, args.remote_dir, manifest, revision)
    _atomic_json(
        receipt,
        {
            "schema_version": 1,
            "verified": True,
            "repo_id": args.repo_id,
            "remote_dir": args.remote_dir,
            "commit": revision,
            "folder_upload_commit": getattr(upload, "oid", None) if upload is not None else None,
            "file_count": manifest["file_count"],
            "logical_bytes": manifest["logical_bytes"],
            "manifest_sha256": _sha256(manifest_path),
            "verified_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    print(
        f"[archive-once] VERIFIED {args.remote_dir}: {manifest['file_count']} files, "
        f"{manifest['logical_bytes'] / 1e9:.2f} GB, commit={revision}"
    )


if __name__ == "__main__":
    main()
