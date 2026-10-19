from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from analysis.warm_start_robustness.claim_agreement.archive_claim_cache import (
    _secret_values,
    archive_snapshot,
    build_bundle,
    build_manifest,
    ensure_private_repo,
    restore_bundle,
    validate_manifest,
    verify_bundle,
)


class FakeApi:
    def __init__(self, *, private: bool = True):
        self.private = private
        self.uploads = []

    def create_repo(self, **kwargs):
        self.created = kwargs

    def repo_info(self, **kwargs):
        return SimpleNamespace(private=self.private, sha="commit-verified")

    def upload_file(self, **kwargs):
        self.uploads.append(kwargs)
        return SimpleNamespace(oid=f"upload-{len(self.uploads)}")


class ClaimArchiveTests(unittest.TestCase):
    def _tree(self, root: Path) -> tuple[Path, Path]:
        cache = root / "cache"
        results = root / "results"
        call = cache / "segment" / "abc"
        (call / "attempts").mkdir(parents=True)
        (call / "journal").mkdir()
        (call / "request.json").write_text('{"request":"redacted"}\n')
        (call / "attempts" / "0000.json").write_text('{"outcome":"success"}\n')
        (call / "result.json").write_text('{"parsed_output":{"claims":[]}}\n')
        (call / "call.lock").write_text("ignored")
        (call / "result.json.tmp.123").write_text("ignored")
        cache.mkdir(exist_ok=True)
        (cache / "budget_ledger.json").write_text(json.dumps({
            "reservations": {},
            "charges": {"call": {"accounted_usd": 0.01}},
            "accounted_spend_usd": 0.01,
            "maximum_committed_usd": 0.01,
        }))
        results.mkdir()
        (results / "segmented_explanations.jsonl").write_text('{"ok":true}\n')
        return cache, results

    def test_manifest_bundle_and_restore_are_exact_and_deterministic(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache, results = self._tree(root)
            manifest = build_manifest(
                cache, results, protocol_sha256="a" * 64, secrets=[]
            )
            paths = {(row["root"], row["path"]) for row in manifest["files"]}
            self.assertNotIn(("cache", "segment/abc/call.lock"), paths)
            self.assertNotIn(("cache", "segment/abc/result.json.tmp.123"), paths)
            self.assertEqual(manifest["cache_summary"]["requests"], 1)
            first, first_receipt = build_bundle(
                manifest, cache, results, root / "staging-one"
            )
            second, second_receipt = build_bundle(
                manifest, cache, results, root / "staging-two"
            )
            self.assertEqual(first_receipt, second_receipt)
            verify_bundle(first, manifest)
            restored_cache = root / "restored-cache"
            restored_results = root / "restored-results"
            restore_bundle(first, manifest, restored_cache, restored_results)
            for row in manifest["files"]:
                destination_root = (
                    restored_cache if row["root"] == "cache" else restored_results
                )
                self.assertTrue((destination_root / row["path"]).is_file())

    def test_secret_is_rejected_before_bundle(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(
            os.environ, {"OPENAI_API_KEY_2": "sk-secret-value"}, clear=True
        ):
            root = Path(temporary)
            cache, results = self._tree(root)
            (cache / "leak.json").write_text('{"key":"sk-secret-value"}')
            with self.assertRaisesRegex(RuntimeError, "credential"):
                build_manifest(
                    cache, results, protocol_sha256="a" * 64,
                    secrets=_secret_values(),
                )

    def test_manifest_snapshot_id_is_self_authenticating(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache, results = self._tree(root)
            manifest = build_manifest(cache, results, protocol_sha256="a" * 64)
            validate_manifest(manifest)
            manifest["logical_bytes"] += 1
            with self.assertRaisesRegex(RuntimeError, "snapshot ID"):
                validate_manifest(manifest)

    def test_private_repo_is_mandatory(self):
        with self.assertRaisesRegex(RuntimeError, "not private"):
            ensure_private_repo(FakeApi(private=False), "owner/repo", "dataset")

    def test_remote_archive_writes_verified_sha_size_receipt_and_resumes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache, results = self._tree(root)
            manifest = build_manifest(cache, results, protocol_sha256="a" * 64)
            bundle, bundle_receipt = build_bundle(
                manifest, cache, results, root / "staging"
            )
            api = FakeApi()
            remote_record = {
                "schema_version": "nla.claim_agreement.archive_receipt.v1",
                "snapshot_id": manifest["snapshot_id"],
                "manifest_bytes": len(
                    (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
                ),
                "manifest_sha256": __import__("hashlib").sha256(
                    (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
                ).hexdigest(),
                "bundle_bytes": bundle_receipt["bytes"],
                "bundle_sha256": bundle_receipt["sha256"],
                "file_count": manifest["file_count"],
                "logical_bytes": manifest["logical_bytes"],
            }
            with patch(
                "analysis.warm_start_robustness.claim_agreement.archive_claim_cache._download_json",
                side_effect=[None, remote_record, manifest],
            ), patch("huggingface_hub.hf_hub_download", return_value=str(bundle)):
                receipt = archive_snapshot(
                    api,
                    repo_id="owner/private",
                    repo_type="dataset",
                    path_prefix="claim",
                    manifest=manifest,
                    bundle=bundle,
                    bundle_receipt=bundle_receipt,
                    receipts_dir=root / "receipts",
                    hf_cache_dir=root / "hf-cache",
                )
            self.assertTrue(receipt["verified"])
            self.assertEqual(receipt["bundle_sha256"], bundle_receipt["sha256"])
            self.assertEqual(receipt["bundle_bytes"], bundle_receipt["bytes"])
            self.assertEqual(len(api.uploads), 3)
            # A verified local receipt makes a restart idempotent and network-free.
            resumed = archive_snapshot(
                api,
                repo_id="owner/private",
                repo_type="dataset",
                path_prefix="claim",
                manifest=manifest,
                bundle=bundle,
                bundle_receipt=bundle_receipt,
                receipts_dir=root / "receipts",
                hf_cache_dir=root / "hf-cache",
            )
            self.assertEqual(resumed["commit"], "commit-verified")
            self.assertEqual(len(api.uploads), 3)


if __name__ == "__main__":
    unittest.main()
