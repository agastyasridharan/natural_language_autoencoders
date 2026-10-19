from __future__ import annotations

import io
import hashlib
import json
import tempfile
import unittest
import os
from argparse import Namespace
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import httpx
import numpy as np

from analysis.groundedness_report.spec import (
    Calibration,
    PROTOCOL_SHA256,
    RECONSTRUCTION_ERROR_FORMULA,
    RECONSTRUCTION_METRIC_AMENDMENT_SHA256,
    TEXT_MODEL_CORRECTION_AMENDMENT_SHA256,
    claim_direction,
    classify_claim,
    injection_vector,
    mean_logp,
    project_out,
    reconstruction_error,
    select_decoys,
    verdict_summary,
)
from analysis.groundedness_report.workflow import (
    make_document_split,
    make_evaluation_split,
    validate_split,
)
from analysis.groundedness_report.gpu_measure import _assert_claim_tiling, _text_pass_candidates
from analysis.groundedness_report.cli import cmd_classify
from analysis.groundedness_report.protocol import (
    load_evaluation_registry,
    load_protocol,
    load_reconstruction_metric_amendment,
    load_text_model_correction_amendment,
)
from analysis.groundedness_report.sampling import cyclic_derangement, sample_claims, sample_positions
from analysis.groundedness_report.stats import clustered_intervals, weighted_percentages
from analysis.groundedness_report.openai_text import JSONResponse, TerraResponses, load_exclusive_api_key
from analysis.groundedness_report.text_pipeline import (
    _segments_with_spans,
    _validate_judgments,
    generate_decoys,
    segment_explanation,
)
from analysis.groundedness_report.prepare_study_text import (
    _balanced_allocation,
    _ood_audit_indices,
)
from analysis.groundedness_report.build_browsecomp_tasks import _derive_key, decrypt
from analysis.groundedness_report.gpu_ood_browser import (
    RecordedWeb,
    _bing_results,
    _parse_calls,
    _SearchParser,
)
from analysis.groundedness_report.gpu_ood_swebench import _parse_shell_calls
from analysis.groundedness_report.calibrate import freeze_thresholds


def calibration(**overrides) -> Calibration:
    values = {
        "model_key": "fake",
        "run_id": "smoke",
        "split_sha256": "1" * 64,
        "reference_sha256": "2" * 64,
        "negative_control_sha256": "3" * 64,
        "protocol_sha256": PROTOCOL_SHA256,
        "reconstruction_metric_amendment_sha256": RECONSTRUCTION_METRIC_AMENDMENT_SHA256,
        "text_model_correction_amendment_sha256": TEXT_MODEL_CORRECTION_AMENDMENT_SHA256,
        "k": 3,
        "eps_overlap_ladder": [0.1, 0.25, 0.5],
        "eps_base": 0.25,
        "eps_reconstruction": 0.1,
        "tau_decoy_support": 0.4,
        "kappa_spec": 0.1,
        "kappa_theme": 0.2,
        "kappa_support": 0.3,
        "rho_own": 1.5,
        "specificity_quantile": 0.95,
        "quantile_method": "linear",
    }
    values.update(overrides)
    return Calibration.from_mapping(values)


def real(**overrides):
    value = {
        "overlap": 2.0,
        "reference_score": -2.0,
        "reconstruction_error": 10.0,
        "activation_support": 0.1,
        "self_effect": 2.0,
    }
    value.update(overrides)
    return value


def candidates(*, d=(0.10, 0.15, 0.20), x=(0.0, 0.0, 0.0)):
    overlaps = (-2.0, 2.1, -1.9)
    return [
        {
            "candidate_index": i,
            "text_pass": True,
            "overlap": overlaps[i],
            "reference_score": -2.0 + 0.05 * i,
            "reconstruction_error": 10.0 + 0.2 * i,
            "activation_support": 0.1,
            "self_effect": d[i],
            "cross_effect": x[i],
        }
        for i in range(3)
    ]


class GeometryTests(unittest.TestCase):
    def test_direction_is_normalized_difference_of_normalized_reconstructions(self):
        direction = claim_direction([2.0, 0.0], [0.0, 3.0])
        self.assertTrue(np.allclose(direction, [2**-0.5, -(2**-0.5)]))

    def test_projection_is_not_renormalized_before_injection(self):
        h = np.array([1.0, 0.0])
        direction = np.array([2**-0.5, 2**-0.5])
        projected = project_out(h, direction)
        self.assertAlmostEqual(float(projected @ direction), 0.0, places=12)
        self.assertLess(float(np.linalg.norm(projected)), 1.0)
        injected = injection_vector(projected, 150.0)
        self.assertAlmostEqual(float(np.linalg.norm(injected)), 150.0 * np.linalg.norm(projected), places=9)
        self.assertNotAlmostEqual(float(np.linalg.norm(injected)), 150.0, places=6)

    def test_reconstruction_error_is_directional_squared_l2(self):
        self.assertAlmostEqual(
            reconstruction_error([1.0, 0.0], [2.0, 2.0]),
            2.0 - 2.0**0.5,
        )

    def test_reconstruction_error_is_invariant_to_positive_scale(self):
        expected = reconstruction_error([1.0, 2.0], [3.0, -4.0])
        self.assertAlmostEqual(
            reconstruction_error([10.0, 20.0], [300.0, -400.0]), expected
        )

    def test_reconstruction_error_rejects_degenerate_vectors(self):
        with self.assertRaisesRegex(ValueError, "zero-norm"):
            reconstruction_error([1.0, 0.0], [0.0, 0.0])

    def test_claim_score_is_length_normalized(self):
        self.assertEqual(mean_logp([-1.0, -2.0, -3.0, -4.0], [1, 3]), -2.5)

    def test_measurement_injector_scans_each_live_input(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")
        from analysis.groundedness_report.gpu_measure import ScanningEmbeddingInjector

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(20, 3)

            def get_input_embeddings(self):
                return self.embedding

        model = Model()
        injector = ScanningEmbeddingInjector(
            model, {"inj_id": 7, "left": 6, "right": 8}
        )
        injector.v = torch.tensor([10.0, 11.0, 12.0])
        first = model.embedding(torch.tensor([[6, 7, 8, 1]]))
        second = model.embedding(torch.tensor([[1, 6, 7, 8]]))
        self.assertTrue(torch.equal(first[0, 1], injector.v))
        self.assertTrue(torch.equal(second[0, 2], injector.v))


class CalibrationTests(unittest.TestCase):
    def test_unfrozen_calibration_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "not frozen"):
            Calibration.from_mapping({"k": 5})

    def test_opposite_signed_overlap_matches_by_magnitude(self):
        selected = select_decoys(real(), candidates(), calibration())
        self.assertTrue(selected["passed"])
        self.assertEqual(selected["eps_overlap_used"], 0.1)
        self.assertEqual(selected["selected_indices"], [0, 1, 2])

    def test_filter_ladder_never_disables_overlap_matching(self):
        rows = candidates()
        for row in rows:
            row["overlap"] = 20.0
        selected = select_decoys(real(), rows, calibration())
        self.assertFalse(selected["passed"])
        self.assertEqual(selected["selected_indices"], [])

    def test_positive_decoy_support_above_threshold_is_rejected(self):
        rows = candidates()
        rows[0]["activation_support"] = 0.5
        selected = select_decoys(real(), rows, calibration())
        self.assertFalse(selected["passed"])
        rows[0]["activation_support"] = -100.0
        selected = select_decoys(real(), rows, calibration())
        self.assertTrue(selected["passed"], "report rejects high positive, not large negative, support")

    def test_thresholds_are_frozen_from_shuffled_null_by_fixed_rule(self):
        rows = []
        for claim_index in range(3):
            controls = []
            null_controls = []
            for candidate_index in range(64):
                base = {
                    "candidate_index": candidate_index,
                    "text_pass": True,
                    "overlap": 2.0,
                    "reference_score": -2.0,
                    "reconstruction_error": 10.0,
                    "activation_support": 0.0,
                    "self_effect": 0.1,
                    "cross_effect": 0.05,
                }
                controls.append(dict(base))
                null_controls.append(dict(base, activation_support=0.01 * candidate_index / 64))
            rows.append({
                "reconstruction_metric_amendment_sha256": (
                    RECONSTRUCTION_METRIC_AMENDMENT_SHA256
                ),
                "reconstruction_error_formula": RECONSTRUCTION_ERROR_FORMULA,
                "text_model_correction_amendment_sha256": (
                    TEXT_MODEL_CORRECTION_AMENDMENT_SHA256
                ),
                "real": real(),
                "candidates": controls,
                "negative_control": {
                    "real": real(self_effect=0.2, activation_support=0.1),
                    "candidates": null_controls,
                },
            })
        frozen, audit = freeze_thresholds(
            rows,
            provenance={
                "model_key": "fake",
                "run_id": "smoke",
                "split_sha256": "1" * 64,
                "reference_sha256": "2" * 64,
                "negative_control_sha256": "3" * 64,
            },
            minimum_null_claims=1,
        )
        self.assertEqual(frozen.k, 20)
        self.assertEqual(frozen.eps_base, 0.25)
        self.assertEqual(frozen.eps_reconstruction, 0.10)
        self.assertTrue(audit["resolvability_target_met"])
        self.assertAlmostEqual(frozen.kappa_spec, 0.1)


class VerdictTests(unittest.TestCase):
    def classify(self, source, controls, off=(0.5,)):
        return classify_claim(
            real=source,
            candidates=controls,
            off_diagonal_effects=off,
            calibration=calibration(),
        )

    def test_grounded(self):
        row = self.classify(real(self_effect=2.0), candidates())
        self.assertEqual(row["verdict"], "grounded")
        self.assertTrue(row["specificity_pass"])
        self.assertTrue(row["ownership_pass"])

    def test_entangled_uses_zero_clamped_largest_competitor(self):
        row = self.classify(real(self_effect=2.0), candidates(), off=(1.5, -4.0))
        self.assertEqual(row["verdict"], "entangled")
        negative = self.classify(real(self_effect=0.5), candidates(d=(0.0, 0.0, 0.0)), off=(-8.0, -2.0))
        self.assertEqual(negative["ownership_competitor"], 0.0)
        self.assertEqual(negative["verdict"], "grounded")

    def test_seeded(self):
        row = self.classify(
            real(self_effect=0.1),
            candidates(d=(0.1, 0.2, 0.3), x=(0.3, 0.4, 0.5)),
        )
        self.assertEqual(row["verdict"], "seeded")

    def test_confabulated(self):
        row = self.classify(
            real(self_effect=0.1, activation_support=0.2),
            candidates(d=(0.1, 0.2, 0.3), x=(-0.1, 0.0, 0.1)),
        )
        self.assertEqual(row["verdict"], "confabulated")

    def test_missing_report_branch_fails_closed(self):
        row = self.classify(
            real(self_effect=0.1, activation_support=0.35),
            candidates(d=(0.1, 0.2, 0.3), x=(-0.1, 0.0, 0.1)),
        )
        self.assertEqual(row["verdict"], "unresolved")
        self.assertIn("unclassified", row["reason"])

    def test_percentages_include_unresolved(self):
        summary = verdict_summary([
            {"verdict": "grounded"},
            {"verdict": "seeded"},
            {"verdict": "unresolved"},
            {"verdict": "unresolved"},
        ])
        self.assertEqual(summary["percentages_all_claims"]["grounded"], 25.0)
        self.assertEqual(summary["percentages_resolved_only"]["grounded"], 50.0)


class SplitTests(unittest.TestCase):
    def test_document_level_split_has_no_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = root / "records.jsonl"
            rows = [
                {"idx": 0, "doc_id": "a"},
                {"idx": 1, "doc_id": "a"},
                {"idx": 2, "doc_id": "b"},
                {"idx": 3, "doc_id": "c"},
            ]
            records.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            split = make_document_split(records, root / "split.json", calibration_fraction=0.34, seed=7)
            validate_split(split)
            partition = {idx: "cal" for idx in split["calibration_indices"]}
            partition.update({idx: "eval" for idx in split["evaluation_indices"]})
            self.assertEqual(partition[0], partition[1])

    def test_keyed_sampling_is_order_independent_and_weighted(self):
        claims = [{"text": str(index)} for index in range(7)]
        sampled = sample_claims(claims, activation_id="a", seed=9)
        self.assertEqual(len(sampled), 3)
        self.assertTrue(all(row["inclusion_probability"] == 3 / 7 for row in sampled))
        self.assertTrue(all(row["analysis_weight"] == 7 / 3 for row in sampled))
        positions = sample_positions([90, 60, 70, 80, 50], trajectory_id="t", seed=9)
        reversed_positions = sample_positions([50, 80, 70, 60, 90], trajectory_id="t", seed=9)
        self.assertEqual(positions, reversed_positions)
        mapping = cyclic_derangement(["a", "b", "c", "d"], seed=9)
        self.assertTrue(all(source != target for source, target in mapping.items()))
        self.assertEqual(set(mapping), set(mapping.values()))

    def test_ood_evaluation_split_has_no_fake_calibration_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = root / "records.jsonl"
            records.write_text("".join(
                json.dumps({"idx": index, "doc_id": f"trajectory-{index // 2}"}) + "\n"
                for index in range(4)
            ), encoding="utf-8")
            split = make_evaluation_split(records, root / "split.json")
            validate_split(split)
            self.assertEqual(split["split_kind"], "evaluation_only")
            self.assertEqual(split["calibration_indices"], [])
            self.assertEqual(split["evaluation_indices"], [0, 1, 2, 3])

    def test_preregistration_hash_and_invariants_are_frozen(self):
        protocol = load_protocol()
        self.assertEqual(protocol["decoys"]["k"], 20)
        self.assertEqual(protocol["openai_text_pipeline"]["model"], "gpt-5.6-luna")
        amendment = load_reconstruction_metric_amendment()
        self.assertEqual(amendment["active_operational_formula"], RECONSTRUCTION_ERROR_FORMULA)
        text_amendment = load_text_model_correction_amendment()
        self.assertEqual(text_amendment["active_operational_model"], "gpt-5.6-terra")
        registry = load_evaluation_registry()
        self.assertEqual(len(registry["families"]), 16)

    def test_ood_manual_audit_allocation_is_balanced_and_clustered(self):
        families = [str(row["id"]) for row in load_evaluation_registry()["families"]]
        allocation = _balanced_allocation(
            families, total=100, seed=20260827, namespace="test"
        )
        self.assertEqual(sum(allocation.values()), 100)
        self.assertEqual(set(allocation.values()), {6, 7})
        manifest = []
        idx = 0
        for family in families:
            for trajectory in range(10):
                for position in range(2):
                    manifest.append({
                        "idx": idx,
                        "family_id": family,
                        "trajectory_id": f"{family}:{trajectory}",
                        "position": position,
                    })
                    idx += 1
        selected = _ood_audit_indices(manifest, seed=20260827, count=100)
        selected_rows = [manifest[index] for index in selected]
        self.assertEqual(len(selected), 100)
        self.assertEqual(len({row["trajectory_id"] for row in selected_rows}), 100)


class ClusteredStatisticsTests(unittest.TestCase):
    def test_weights_and_clusters_not_claim_iid(self):
        rows = [
            {"verdict": "grounded", "analysis_weight": 2.0, "document_id": "a"},
            {"verdict": "confabulated", "analysis_weight": 1.0, "document_id": "b"},
            {"verdict": "confabulated", "analysis_weight": 1.0, "document_id": "b"},
        ]
        normalized = [{**row, "cluster_id": row["document_id"]} for row in rows]
        point = weighted_percentages(normalized)
        self.assertEqual(point["grounded"], 50.0)
        result = clustered_intervals(rows, cluster_key="document_id", replicates=100, seed=4)
        self.assertEqual(result["n_clusters"], 2)
        self.assertEqual(result["point_percentages"]["grounded"], 50.0)


class OpenAITextPipelineTests(unittest.TestCase):
    def test_generic_key_is_never_a_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            dotenv = Path(directory) / ".env"
            dotenv.write_text("OPENAI_API_KEY=forbidden\n", encoding="utf-8")
            with patch.dict(os.environ, {"OPENAI_API_KEY": "also-forbidden"}, clear=True):
                with self.assertRaisesRegex(ValueError, "OPENAI_API_KEY_2"):
                    load_exclusive_api_key(dotenv)
            dotenv.write_text("OPENAI_API_KEY=forbidden\nOPENAI_API_KEY_2='allowed'\n", encoding="utf-8")
            with patch.dict(os.environ, {}, clear=True):
                self.assertEqual(load_exclusive_api_key(dotenv), "allowed")

    def test_response_audit_records_resolved_model_without_key(self):
        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.headers["authorization"], "Bearer test-key-2")
            return httpx.Response(200, json={
                "id": "resp_test",
                "model": "gpt-5.6-terra-2026-09-01",
                "system_fingerprint": "fp_test",
                "created_at": 1,
                "usage": {"input_tokens": 1, "output_tokens": 1},
                "output": [{"content": [{"type": "output_text", "text": '{"ok":true}'}]}],
            })

        transport = httpx.MockTransport(handler)
        http_client = httpx.Client(transport=transport)
        with patch.dict(os.environ, {"OPENAI_API_KEY_2": "test-key-2"}, clear=True):
            client = TerraResponses(client=http_client)
            result = client.request_json(
                task_name="test_schema",
                system_prompt="system",
                user_prompt="user",
                schema={
                    "type": "object",
                    "properties": {"ok": {"type": "boolean"}},
                    "required": ["ok"],
                    "additionalProperties": False,
                },
                max_output_tokens=20,
            )
        http_client.close()
        self.assertEqual(result.value, {"ok": True})
        self.assertEqual(result.audit["resolved_model"], "gpt-5.6-terra-2026-09-01")
        self.assertIn("text_model_correction_amendment_sha256", result.audit)
        self.assertNotIn("api_key", result.audit)

    def test_non_terra_resolved_model_is_rejected(self):
        transport = httpx.MockTransport(lambda request: httpx.Response(200, json={
            "id": "resp_wrong_model",
            "model": "gpt-5.6-luna",
            "output": [{"content": [{"type": "output_text", "text": '{"ok":true}'}]}],
        }, request=request))
        http_client = httpx.Client(transport=transport)
        with patch.dict(os.environ, {"OPENAI_API_KEY_2": "test-key-2"}, clear=True):
            client = TerraResponses(client=http_client)
            with self.assertRaisesRegex(RuntimeError, "differs from required Terra model"):
                client.request_json(
                    task_name="test_schema",
                    system_prompt="system",
                    user_prompt="user",
                    schema={
                        "type": "object",
                        "properties": {"ok": {"type": "boolean"}},
                        "required": ["ok"],
                        "additionalProperties": False,
                    },
                    max_output_tokens=20,
                )
        http_client.close()

    def test_claim_ranges_derive_an_exact_tiling(self):
        rows = _segments_with_spans("alpha, beta", {"claims": [
            {"first_token": 0, "last_token": 0},
            {"first_token": 3, "last_token": 3},
        ]})
        self.assertEqual(rows[1], {
            "kind": "connective",
            "text": ", ",
            "char_span": [5, 7],
        })
        self.assertEqual(rows[-1]["char_span"], [7, 11])
        self.assertEqual("".join(row["text"] for row in rows), "alpha, beta")

    def test_claim_ranges_are_not_repaired_or_reordered(self):
        with self.assertRaisesRegex(ValueError, "invalid range"):
            _segments_with_spans("alpha, beta", {"claims": [
                {"first_token": 3, "last_token": 2},
            ]})
        with self.assertRaisesRegex(ValueError, "overlaps, nests, or is out of order"):
            _segments_with_spans("alpha, beta", {"claims": [
                {"first_token": 3, "last_token": 3},
                {"first_token": 0, "last_token": 0},
            ]})
        with self.assertRaisesRegex(ValueError, "whitespace boundary"):
            _segments_with_spans("alpha, beta", {"claims": [
                {"first_token": 0, "last_token": 2},
            ]})

    def test_repeated_claim_ranges_are_unambiguous(self):
        rows = _segments_with_spans("x / x", {"claims": [
            {"first_token": 0, "last_token": 0},
            {"first_token": 4, "last_token": 4},
        ]})
        claims = [row for row in rows if row["kind"] == "claim"]
        self.assertEqual([row["char_span"] for row in claims], [[0, 1], [4, 5]])

    def test_invalid_primary_inventory_gets_one_complete_adjudication(self):
        class FakeClient:
            def __init__(self):
                self.calls = []

            def request_json(self, **kwargs):
                self.calls.append(kwargs)
                if len(self.calls) == 1:
                    value = {"claims": [
                        {"first_token": 0, "last_token": 3},
                        {"first_token": 3, "last_token": 3},
                    ]}
                else:
                    value = {"claims": [
                        {"first_token": 0, "last_token": 0},
                        {"first_token": 3, "last_token": 3},
                    ]}
                return JSONResponse(value=value, audit={"call": len(self.calls)})

        client = FakeClient()
        rows, audit = segment_explanation(client, "alpha, beta", cache_path="primary.json")
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(client.calls[1]["cache_path"], "primary.adjudication.json")
        self.assertTrue(audit["adjudication_used"])
        self.assertEqual("".join(row["text"] for row in rows), "alpha, beta")

    def test_judgment_indices_are_assigned_by_input_order(self):
        rows = [{"label": "pass", "reason": str(index)} for index in range(64)]
        result = _validate_judgments(JSONResponse(value={"judgments": rows}, audit={}))
        self.assertEqual(
            [row["candidate_index"] for row in result.value["judgments"]],
            list(range(64)),
        )

    def test_duplicate_decoy_pool_gets_one_complete_regeneration(self):
        class FakeClient:
            def __init__(self):
                self.calls = []

            def request_json(self, **kwargs):
                self.calls.append(kwargs)
                if len(self.calls) == 1:
                    candidates = [f"candidate {index}" for index in range(63)] + ["candidate 0 "]
                else:
                    candidates = [f"replacement {index}" for index in range(64)]
                return JSONResponse(value={"candidates": candidates}, audit={"call": len(self.calls)})

        client = FakeClient()
        result = generate_decoys(
            client,
            explanation="example",
            claim="claim",
            claim_class="detail",
            subtype="other",
            cache_path="generation.json",
        )
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(client.calls[1]["cache_path"], "generation.regeneration.json")
        self.assertTrue(result.audit["regeneration_used"])
        self.assertEqual(result.value["candidates"][0], "replacement 0")


class BrowseCompReplicationTests(unittest.TestCase):
    def test_official_xor_decryption_round_trip(self):
        import base64

        clear = "a public benchmark question"
        password = "canary"
        key = _derive_key(password, len(clear.encode()))
        encrypted = bytes(left ^ right for left, right in zip(clear.encode(), key))
        self.assertEqual(decrypt(base64.b64encode(encrypted).decode(), password), clear)

    def test_qwen_tool_call_and_search_result_parsing(self):
        calls = _parse_calls(
            '<tool_call>{"name":"search","arguments":{"query":"test"}}</tool_call>'
        )
        self.assertEqual(calls, [{"name": "search", "arguments": {"query": "test"}}])
        parser = _SearchParser()
        parser.feed(
            '<a class="result__a" href="https://example.com">Example</a>'
            '<a class="result__snippet">Result text</a>'
        )
        self.assertEqual(parser.results, [{
            "title": "Example", "url": "https://example.com", "snippet": "Result text",
        }])
        self.assertEqual(_bing_results(
            '<li class="b_algo"><h2><a href="https://example.org">Result</a></h2>'
            '<p>Useful <strong>snippet</strong>.</p></li>'
        ), [{
            "title": "Result", "url": "https://example.org", "snippet": "Useful snippet .",
        }])

    def test_search_probe_can_archive_http_error_before_fallback(self):
        class FakeResponse:
            status_code = 403
            headers = {"content-type": "text/html"}

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return None

            def iter_bytes(self):
                yield b"rate limited"

        class FakeClient:
            def stream(self, *args, **kwargs):
                return FakeResponse()

            def close(self):
                return None

        with tempfile.TemporaryDirectory() as directory:
            recorder = RecordedWeb(Path(directory), byte_limit=1000, text_limit=1000)
            recorder.client.close()
            recorder.client = FakeClient()
            with patch(
                "analysis.groundedness_report.gpu_ood_browser._safe_url",
                side_effect=lambda value: value,
            ):
                body, metadata = recorder._request(
                    "https://example.com", "probe", allow_http_error=True
                )
                self.assertEqual(body, b"rate limited")
                self.assertEqual(metadata["redirect_chain"][-1]["status_code"], 403)
                with self.assertRaisesRegex(ValueError, "HTTP 403"):
                    recorder._request("https://example.com", "ordinary")
            recorder.close()

    def test_qwen_shell_call_parser_rejects_other_tools(self):
        self.assertEqual(_parse_shell_calls(
            '<tool_call>{"name":"run_shell","arguments":{"command":"pwd"}}</tool_call>'
        ), [{"name": "run_shell", "arguments": {"command": "pwd"}}])
        with self.assertRaisesRegex(ValueError, "invalid shell"):
            _parse_shell_calls(
                '<tool_call>{"name":"fetch","arguments":{"url":"https://example.com"}}</tool_call>'
            )


class ArtifactContractTests(unittest.TestCase):
    def test_claims_and_connectives_must_exactly_tile(self):
        _assert_claim_tiling(
            "alpha, beta",
            [
                {"text": "alpha", "char_span": [0, 5]},
                {"text": " beta", "char_span": [6, 11]},
            ],
            [{"text": ",", "char_span": [5, 6]}],
        )
        with self.assertRaisesRegex(ValueError, "exactly tile"):
            _assert_claim_tiling(
                "alpha, beta",
                [{"text": "alpha", "char_span": [0, 5]}],
                [{"text": ",", "char_span": [5, 6]}],
            )

    def test_decoy_already_in_explanation_is_discarded(self):
        cache = {"candidates": [
            {"text": "beta", "kept": True, "absence": "absent", "natural_pass": True},
            {"text": "gamma", "kept": True, "absence": "absent", "natural_pass": True},
        ]}
        self.assertEqual(_text_pass_candidates(cache, "alpha BETA"), [1])

    def test_cpu_cli_writes_verdicts_and_percentages(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calibration_path = root / "calibration.json"
            calibration_path.write_text(json.dumps(calibration().as_dict()), encoding="utf-8")
            measurement_path = root / "measurements.jsonl"
            measurement = {
                "reconstruction_metric_amendment_sha256": (
                    RECONSTRUCTION_METRIC_AMENDMENT_SHA256
                ),
                "reconstruction_error_formula": RECONSTRUCTION_ERROR_FORMULA,
                "text_model_correction_amendment_sha256": (
                    TEXT_MODEL_CORRECTION_AMENDMENT_SHA256
                ),
                "status": "ok",
                "model_key": "fake",
                "run_id": "smoke",
                "partition": "calibration",
                "split_sha256": "1" * 64,
                "reference_activation": {"sha256": "2" * 64},
                "idx": 0,
                "doc_id": "doc",
                "claim_i": 0,
                "claim_text": "claim",
                "claim_type": "detail",
                "claim_subtype": "other",
                "real": real(),
                "off_diagonal_effects": [0.5],
                "candidates": candidates(),
            }
            measurement_path.write_text(json.dumps(measurement) + "\n", encoding="utf-8")
            output = root / "verdicts.jsonl"
            summary = root / "summary.json"
            with redirect_stdout(io.StringIO()):
                cmd_classify(Namespace(
                    measurements=str(measurement_path),
                    calibration=str(calibration_path),
                    out=str(output),
                    summary=str(summary),
                ))
            verdict = json.loads(output.read_text(encoding="utf-8"))
            aggregate = json.loads(summary.read_text(encoding="utf-8"))
            self.assertEqual(verdict["verdict"], "grounded")
            self.assertEqual(aggregate["percentages_all_claims"]["grounded"], 100.0)

    def test_cpu_cli_rejects_cross_model_calibration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calibration_path = root / "calibration.json"
            calibration_path.write_text(json.dumps(calibration().as_dict()), encoding="utf-8")
            measurement_path = root / "measurements.jsonl"
            measurement_path.write_text(json.dumps({
                "reconstruction_metric_amendment_sha256": (
                    RECONSTRUCTION_METRIC_AMENDMENT_SHA256
                ),
                "reconstruction_error_formula": RECONSTRUCTION_ERROR_FORMULA,
                "text_model_correction_amendment_sha256": (
                    TEXT_MODEL_CORRECTION_AMENDMENT_SHA256
                ),
                "status": "unresolved",
                "model_key": "wrong-model",
                "run_id": "smoke",
                "partition": "evaluation",
                "split_sha256": "1" * 64,
                "reference_activation": {"sha256": "2" * 64},
                "idx": 0,
                "claim_i": 0,
                "calibration_fingerprint": "bad",
            }) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "provenance mismatch"):
                cmd_classify(Namespace(
                    measurements=str(measurement_path),
                    calibration=str(calibration_path),
                    out=str(root / "out.jsonl"),
                    summary=None,
                ))

    def test_cpu_cli_accepts_distinct_ood_run_with_frozen_fineweb_calibration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frozen = calibration()
            calibration_path = root / "calibration.json"
            calibration_path.write_text(json.dumps(frozen.as_dict()), encoding="utf-8")
            fingerprint = hashlib.sha256(
                json.dumps(
                    frozen.as_dict(), sort_keys=True, separators=(",", ":")
                ).encode()
            ).hexdigest()
            measurement_path = root / "measurements.jsonl"
            measurement_path.write_text(json.dumps({
                "reconstruction_metric_amendment_sha256": (
                    RECONSTRUCTION_METRIC_AMENDMENT_SHA256
                ),
                "reconstruction_error_formula": RECONSTRUCTION_ERROR_FORMULA,
                "text_model_correction_amendment_sha256": (
                    TEXT_MODEL_CORRECTION_AMENDMENT_SHA256
                ),
                "status": "unresolved",
                "model_key": "fake",
                "run_id": "ood-static-v7",
                "partition": "evaluation",
                "split_sha256": "9" * 64,
                "reference_activation": {"sha256": "2" * 64},
                "idx": 0,
                "claim_i": 0,
                "calibration_fingerprint": fingerprint,
            }) + "\n", encoding="utf-8")
            with redirect_stdout(io.StringIO()):
                cmd_classify(Namespace(
                    measurements=str(measurement_path),
                    calibration=str(calibration_path),
                    out=str(root / "out.jsonl"),
                    summary=None,
                ))
            verdict = json.loads((root / "out.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(verdict["verdict"], "unresolved")


if __name__ == "__main__":
    unittest.main()
