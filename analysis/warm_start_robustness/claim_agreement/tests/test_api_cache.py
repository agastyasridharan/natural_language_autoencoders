from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from analysis.warm_start_robustness.claim_agreement.api_cache import (
    ApiConfig,
    BudgetExceeded,
    CachedCallFailure,
    CachedResponsesClient,
)


def config(budget: float = 12.0) -> ApiConfig:
    return ApiConfig(
        endpoint="https://api.openai.com/v1/responses",
        model="gpt-5-nano-2025-08-07",
        reasoning_effort="minimal",
        service_tier="default",
        input_usd_per_million=0.05,
        cached_input_usd_per_million=0.005,
        output_usd_per_million=0.40,
        hard_budget_usd=budget,
    )


SCHEMA = {
    "type": "object",
    "properties": {"ok": {"type": "boolean"}},
    "required": ["ok"],
    "additionalProperties": False,
}


class ApiCacheTests(unittest.TestCase):
    def test_success_is_fully_cached_and_reused(self) -> None:
        calls = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            self.assertEqual(request.headers["authorization"], "Bearer authorized-key")
            body = json.loads(request.content)
            self.assertIs(body["store"], False)
            response = {
                "id": "resp_test",
                "status": "completed",
                "output": [{"type": "message", "content": [
                    {"type": "output_text", "text": '{"ok":true}'}
                ]}],
                "usage": {"input_tokens": 100, "output_tokens": 20,
                          "input_tokens_details": {"cached_tokens": 40}},
            }
            return httpx.Response(200, json=response, headers={"x-request-id": "request-test"})

        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, {"OPENAI_API_KEY_2": "authorized-key", "OPENAI_API_KEY": "wrong-key"}, clear=True
        ):
            client = CachedResponsesClient(Path(directory), config(), transport=httpx.MockTransport(handler))
            kwargs = dict(
                stage="segment", cache_key="abc123", instructions="instructions", input_text="input",
                schema_name="test_schema", schema=SCHEMA, max_output_tokens=100,
                validator=lambda value: dict(value),
            )
            parsed, _ = client.call(**kwargs)
            parsed_cached, _ = client.call(**kwargs)
            self.assertEqual(parsed, {"ok": True})
            self.assertEqual(parsed_cached, parsed)
            self.assertEqual(len(calls), 1)
            root = Path(directory) / "segment" / "abc123"
            request = json.loads((root / "request.json").read_text())
            attempt = json.loads((root / "attempts" / "0000.json").read_text())
            self.assertEqual(request["headers"]["authorization"], "REDACTED_OPENAI_API_KEY_2")
            self.assertEqual(attempt["response"]["headers"]["x-request-id"], "request-test")
            self.assertIn('resp_test', attempt["response"]["body_text"])
            self.assertEqual(
                attempt["response"]["json"]["output"][0]["content"][0]["text"],
                '{"ok":true}',
            )
            self.assertEqual(attempt["usage"]["input_tokens"], 100)
            expected = ((60 * 0.05) + (40 * 0.005) + (20 * 0.40)) / 1_000_000
            self.assertAlmostEqual(attempt["charge"]["accounted_usd"], expected)

    def test_missing_authorized_key_creates_no_reservation(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, {"OPENAI_API_KEY": "forbidden-fallback"}, clear=True
        ):
            client = CachedResponsesClient(Path(directory), config(), transport=httpx.MockTransport(lambda _: None))
            with self.assertRaisesRegex(RuntimeError, "OPENAI_API_KEY_2"):
                client.call(
                    stage="segment", cache_key="nokey", instructions="i", input_text="x",
                    schema_name="s", schema=SCHEMA, max_output_tokens=10,
                    validator=lambda value: dict(value),
                )
            ledger = client.ledger.snapshot()
            self.assertFalse(ledger["reservations"])
            self.assertFalse(ledger["charges"])

    def test_unknown_usage_failure_retains_full_reservation(self) -> None:
        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="server error")

        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, {"OPENAI_API_KEY_2": "key"}, clear=True
        ):
            client = CachedResponsesClient(Path(directory), config(), transport=httpx.MockTransport(handler))
            with self.assertRaises(CachedCallFailure):
                client.call(
                    stage="segment", cache_key="failure", instructions="i", input_text="x",
                    schema_name="s", schema=SCHEMA, max_output_tokens=10,
                    validator=lambda value: dict(value),
                )
            ledger = client.ledger.snapshot()
            charge = next(iter(ledger["charges"].values()))
            self.assertEqual(charge["accounting_basis"], "conservative_full_reservation_missing_usage")
            self.assertEqual(charge["accounted_usd"], charge["reserved_usd"])

    def test_preflight_hard_budget(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, {"OPENAI_API_KEY_2": "key"}, clear=True
        ):
            client = CachedResponsesClient(Path(directory), config(0.000001),
                                           transport=httpx.MockTransport(lambda _: None))
            with self.assertRaises(BudgetExceeded):
                client.call(
                    stage="segment", cache_key="budget", instructions="long" * 100,
                    input_text="x" * 100, schema_name="s", schema=SCHEMA,
                    max_output_tokens=100, validator=lambda value: dict(value),
                )

    def test_interrupted_preflight_is_conservatively_recovered_without_replay(self) -> None:
        calls = 0

        def interrupted(_: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            raise KeyboardInterrupt

        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, {"OPENAI_API_KEY_2": "key"}, clear=True
        ):
            root = Path(directory)
            kwargs = dict(
                stage="segment", cache_key="interrupted", instructions="i", input_text="x",
                schema_name="s", schema=SCHEMA, max_output_tokens=10,
                validator=lambda value: dict(value),
            )
            first = CachedResponsesClient(root, config(), transport=httpx.MockTransport(interrupted))
            with self.assertRaises(KeyboardInterrupt):
                first.call(**kwargs)
            second = CachedResponsesClient(
                root, config(),
                transport=httpx.MockTransport(lambda _: self.fail("ambiguous request was replayed")),
            )
            with self.assertRaises(CachedCallFailure) as captured:
                second.call(**kwargs)
            self.assertEqual(captured.exception.code, "interrupted_unknown_usage")
            self.assertEqual(calls, 1)
            ledger = second.ledger.snapshot()
            charge = next(iter(ledger["charges"].values()))
            self.assertEqual(charge["accounting_basis"], "conservative_full_reservation_missing_usage")


if __name__ == "__main__":
    unittest.main()
