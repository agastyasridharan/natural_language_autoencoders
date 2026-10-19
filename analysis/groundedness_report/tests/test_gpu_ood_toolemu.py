from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from analysis.groundedness_report.gpu_ood_toolemu import (
    TOOLEMU_BOUNDED_SIMULATOR_AMENDMENT_SHA256,
    TOOLEMU_FORMAT_BUDGET_AMENDMENT_SHA256,
    TOOLEMU_FAILURE_CLASSIFICATION_AMENDMENT_SHA256,
    TOOLEMU_PILOT_GATE_RESOLUTION_SHA256,
    TOOLEMU_SHARED_STATE_AMENDMENT_SHA256,
    ToolEmuBudgetExceeded,
    _load_bounded_simulator_amendment,
    _load_failure_classification_amendment,
    _load_format_budget_amendment,
    _load_pilot_gate_resolution,
    _load_shared_state_amendment,
    _natural_language_character_end,
    _normalize_simulator_headers,
    _position_rank,
    _persist_interruption,
    _replaceable_failure_kind,
    _retain_position,
    _simulator_budget_violation,
    _truncate_at_stop,
    _validate_replaceable_failure,
)
from analysis.groundedness_report.sampling import sample_positions


class ToolEmuRunnerPureTests(unittest.TestCase):
    def test_interruption_record_is_persistent_and_task_scoped(self):
        task = {
            "family_id": "alignment_overly_agentic",
            "task_id": "official_23",
            "task_sha256": "a" * 64,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record, relative = _persist_interruption(
                output=root,
                selected=task,
                attempt_id="attempt-1",
                audit_rel=Path("tools/alignment_overly_agentic/x/attempts/attempt-1"),
                error=RuntimeError("transport failed"),
                interruption_kind="infrastructure",
            )
            self.assertTrue((root / relative).is_file())
        self.assertEqual(record["task_id"], "official_23")
        self.assertEqual(record["error"], "RuntimeError: transport failed")

    def test_bounded_simulator_amendment_is_hash_pinned(self):
        amendment = _load_bounded_simulator_amendment()
        self.assertEqual(amendment["configuration"]["maximum_iterations"], 5)
        self.assertEqual(amendment["configuration"]["simulator_model"], "gpt-5.6-terra")
        self.assertEqual(len(TOOLEMU_BOUNDED_SIMULATOR_AMENDMENT_SHA256), 64)

    def test_format_budget_amendment_is_hash_pinned(self):
        amendment = _load_format_budget_amendment()
        correction = amendment["correction"]
        self.assertEqual(correction["maximum_simulator_calls_per_trajectory"], 10)
        self.assertEqual(
            correction["maximum_cumulative_simulator_input_tokens_per_trajectory"],
            250000,
        )
        self.assertEqual(
            correction["maximum_serialized_simulator_request_bytes_per_call"],
            600000,
        )
        self.assertEqual(len(TOOLEMU_FORMAT_BUDGET_AMENDMENT_SHA256), 64)

    def test_shared_state_amendment_is_hash_pinned(self):
        amendment = _load_shared_state_amendment()
        self.assertEqual(
            amendment["toolemu_format_budget_amendment_sha256"],
            TOOLEMU_FORMAT_BUDGET_AMENDMENT_SHA256,
        )
        self.assertEqual(
            amendment["corrected_pilot"]["task_ids"],
            ["official_99", "official_92", "official_136"],
        )
        self.assertEqual(len(TOOLEMU_SHARED_STATE_AMENDMENT_SHA256), 64)

    def test_pilot_gate_resolution_is_hash_pinned(self):
        resolution = _load_pilot_gate_resolution()
        self.assertEqual(resolution["resolution"]["canonical_target"], 50)
        self.assertEqual(
            resolution["controlling_preexisting_rule"]["replacement_rule"],
            "take_next_sha256_ranked_task_until_50_analyzable_trajectories",
        )
        self.assertEqual(len(TOOLEMU_PILOT_GATE_RESOLUTION_SHA256), 64)

    def test_failure_classification_amendment_is_hash_pinned(self):
        amendment = _load_failure_classification_amendment()
        self.assertIn(
            "unexpected_exception_rule",
            amendment["correction"],
        )
        self.assertEqual(len(TOOLEMU_FAILURE_CLASSIFICATION_AMENDMENT_SHA256), 64)

    def test_only_frozen_failures_are_replaceable(self):
        no_eligible = ValueError(
            "ToolEmu trajectory has no eligible assistant-language token"
        )
        nonfinite = ValueError("captured activation contains nonfinite values")
        budget = ToolEmuBudgetExceeded("simulator-call ceiling reached: 10>=10")
        self.assertEqual(
            _replaceable_failure_kind(no_eligible),
            "no_eligible_assistant_token_after_minimum_context_rule",
        )
        self.assertEqual(
            _replaceable_failure_kind(nonfinite),
            "nonfinite_activation",
        )
        self.assertEqual(
            _replaceable_failure_kind(budget),
            "bounded_budget_exceeded",
        )
        self.assertIsNone(_replaceable_failure_kind(ValueError("parser defect")))

    def test_replaceable_failure_record_validation_is_exact(self):
        _validate_replaceable_failure(
            {
                "failure_kind": "no_eligible_assistant_token_after_minimum_context_rule",
                "error": "ValueError: ToolEmu trajectory has no eligible assistant-language token",
            }
        )
        with self.assertRaisesRegex(ValueError, "not a frozen replaceable"):
            _validate_replaceable_failure(
                {
                    "failure_kind": "task_or_serialization",
                    "error": "ValueError: parser defect",
                }
            )

    def test_simulator_header_newline_is_normalized_only_at_exact_headers(self):
        raw = (
            "Simulator Thought:\nunchanged\n"
            "Simulator Log Summary:\nsummary\n"
            "Observation:\n{\"success\":true}"
        )
        observed, headers = _normalize_simulator_headers(raw)
        self.assertEqual(
            observed,
            "Simulator Thought:\nunchanged\n"
            "Simulator Log Summary: summary\n"
            "Observation: {\"success\":true}",
        )
        self.assertEqual(headers, ["Simulator Log Summary:", "Observation:"])

    def test_simulator_budget_boundaries(self):
        common = {
            "maximum_calls": 10,
            "maximum_cumulative_input_tokens": 250000,
            "maximum_request_bytes": 600000,
        }
        self.assertIsNone(
            _simulator_budget_violation(
                completed_calls=9,
                cumulative_input_tokens=249999,
                serialized_request_bytes=600000,
                **common,
            )
        )
        self.assertIn(
            "call ceiling",
            _simulator_budget_violation(
                completed_calls=10,
                cumulative_input_tokens=1,
                serialized_request_bytes=1,
                **common,
            ),
        )
        self.assertIn(
            "byte ceiling",
            _simulator_budget_violation(
                completed_calls=0,
                cumulative_input_tokens=0,
                serialized_request_bytes=600001,
                **common,
            ),
        )
        self.assertIn(
            "input-token ceiling",
            _simulator_budget_violation(
                completed_calls=1,
                cumulative_input_tokens=250000,
                serialized_request_bytes=1,
                after_completed_call=True,
                **common,
            ),
        )

    def test_action_suffix_is_excluded(self):
        text = 'Thought: I should inspect first.\nAction: Search\nAction Input: {"q":"x"}'
        end = _natural_language_character_end(text)
        self.assertEqual(text[:end], "Thought: I should inspect first.\n")

    def test_final_answer_is_entirely_natural_language(self):
        text = "Thought: done.\nFinal Answer: I finished."
        self.assertEqual(_natural_language_character_end(text), len(text))

    def test_stop_truncation_uses_earliest_match(self):
        self.assertEqual(
            _truncate_at_stop("answer<later>tail<first>end", ["<first>", "<later>"]),
            "answer",
        )

    def test_streaming_position_reservoir_matches_protocol_sampler(self):
        positions = [91, 4, 1002, 8, 77, 13, 905]
        retained = []
        for position in positions:
            _retain_position(
                retained,
                {"position_key": position},
                seed=20260827,
                trajectory_id="alignment_overly_agentic:official_0",
            )
        observed = sorted(row["position_key"] for _, _, row in retained)
        expected = sample_positions(
            positions,
            seed=20260827,
            trajectory_id="alignment_overly_agentic:official_0",
            maximum=4,
        )
        self.assertEqual(observed, expected)
        for digest, identity, row in retained:
            self.assertEqual(
                digest,
                _position_rank(
                    20260827,
                    "alignment_overly_agentic:official_0",
                    row["position_key"],
                ),
            )
            self.assertEqual(identity, str(row["position_key"]))


if __name__ == "__main__":
    unittest.main()
