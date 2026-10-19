from __future__ import annotations

import unittest

import httpx

from analysis.groundedness_report.gpu_ood_tau2 import (
    TerraInfrastructureError,
    _parse_qwen_tool_response,
    _preflight_openai_endpoint,
)


class Tau2QwenToolParserTests(unittest.TestCase):
    def test_accepts_one_pure_call(self):
        call, error = _parse_qwen_tool_response(
            '<tool_call>{"name":"get_order_details","arguments":{"order_id":"x"}}</tool_call>'
        )
        self.assertIsNone(error)
        self.assertEqual(call["name"], "get_order_details")
        self.assertEqual(call["arguments"], {"order_id": "x"})

    def test_plain_text_is_not_a_call(self):
        self.assertEqual(_parse_qwen_tool_response("How can I help?"), (None, None))

    def test_rejects_mixed_text_and_call(self):
        call, error = _parse_qwen_tool_response(
            'I will check. <tool_call>{"name":"x","arguments":{}}</tool_call>'
        )
        self.assertIsNone(call)
        self.assertIn("exactly one pure call", error)

    def test_rejects_non_object_arguments(self):
        call, error = _parse_qwen_tool_response(
            '<tool_call>{"name":"x","arguments":[]}</tool_call>'
        )
        self.assertIsNone(call)
        self.assertIn("object arguments", error)

    def test_endpoint_timeout_is_infrastructure_error(self):
        def timeout(_request):
            raise httpx.ConnectTimeout("timed out")

        with self.assertRaisesRegex(
            TerraInfrastructureError, "OpenAI endpoint HTTPS preflight failed"
        ):
            _preflight_openai_endpoint(
                timeout_seconds=0.01,
                transport=httpx.MockTransport(timeout),
            )

    def test_endpoint_http_response_passes_preflight(self):
        transport = httpx.MockTransport(
            lambda request: httpx.Response(401, request=request)
        )
        observed = _preflight_openai_endpoint(
            timeout_seconds=0.01, transport=transport
        )
        self.assertEqual(observed["http_status_without_credentials"], 401)
        self.assertFalse(observed["proxy_configured"])


if __name__ == "__main__":
    unittest.main()
