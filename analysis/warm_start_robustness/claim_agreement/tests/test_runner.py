from __future__ import annotations

import contextlib
import io
import json
import tempfile
import time
import unittest
from pathlib import Path

from analysis.warm_start_robustness.claim_agreement.core import EXPLANATION_SCHEMA_VERSION, ValidationError
from analysis.warm_start_robustness.claim_agreement.run_pipeline import (
    DEFAULT_PROTOCOL,
    load_protocol,
    main,
    parse_args,
    validate_archive_guard,
)


class RunnerTests(unittest.TestCase):
    def _write_input(self, root: Path) -> Path:
        path = root / "explanations.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for condition in ("plausible", "implausible"):
                for sample in (0, 1):
                    row = {
                        "schema_version": EXPLANATION_SCHEMA_VERSION,
                        "activation_id": "fineweb:a0",
                        "document_id": "doc0",
                        "trajectory_id": None,
                        "distribution": "fineweb",
                        "benchmark": "fineweb",
                        "family": "fineweb",
                        "condition": condition,
                        "sample_index": sample,
                        "model_id": "model",
                        "checkpoint_id": condition,
                        "temperature": 0.7,
                        "explanation": f"{condition} sample {sample}",
                        "generation_ok": True,
                        "failure_reason": None,
                    }
                    handle.write(json.dumps(row) + "\n")
        return path

    def test_protocol_freezes_global_remaining_cap_and_key(self) -> None:
        protocol, _ = load_protocol(DEFAULT_PROTOCOL)
        self.assertEqual(protocol["api"]["maximum_hard_budget_usd"], 12.0)
        self.assertEqual(protocol["api"]["api_key_environment_variable"], "OPENAI_API_KEY_2")
        self.assertFalse(protocol["api"]["store"])
        self.assertFalse(protocol["api"]["automatic_model_fallback"])

    def test_default_status_is_no_api_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self._write_input(root)
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = main([
                    "--input", str(path), "--stage", "status",
                    "--cache-dir", str(root / "cache"), "--output-dir", str(root / "out"),
                ])
            result = json.loads(stdout.getvalue())
            self.assertEqual(code, 0)
            self.assertFalse(result["api_calls_launched"])
            self.assertEqual(result["reports"][0]["pending_unique_calls"], 4)
            self.assertFalse((root / "out" / "segmented_explanations.jsonl").exists())

    def test_paid_execution_requires_live_archive_guard(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(["--input", "unused.jsonl", "--stage", "segment", "--execute"])
        with tempfile.TemporaryDirectory() as directory:
            status_path = Path(directory) / "archive-status.json"
            status = {
                "schema_version": "nla.claim_agreement.archive_status.v1",
                "state": "healthy",
                "repo_id": "owner/private",
                "repo_type": "dataset",
                "last_error": None,
                "heartbeat_unix": time.time(),
            }
            status_path.write_text(json.dumps(status))
            self.assertEqual(validate_archive_guard(status_path)["state"], "healthy")
            status["heartbeat_unix"] = time.time() - 1000
            status_path.write_text(json.dumps(status))
            with self.assertRaisesRegex(ValidationError, "stale"):
                validate_archive_guard(status_path)


if __name__ == "__main__":
    unittest.main()
