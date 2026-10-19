from __future__ import annotations

import unittest

from analysis.warm_start_robustness.claim_agreement.core import (
    EXPLANATION_SCHEMA_VERSION,
    PairScoreInput,
    ValidationError,
    build_pair_plan,
    score_pair,
    validate_equivalence_output,
    validate_segmentation_output,
)


def explanation_row(activation: str, condition: str, sample_index: int) -> dict:
    return {
        "schema_version": EXPLANATION_SCHEMA_VERSION,
        "activation_id": activation,
        "document_id": "doc-1",
        "trajectory_id": None,
        "distribution": "fineweb",
        "benchmark": "fineweb",
        "family": "fineweb",
        "condition": condition,
        "sample_index": sample_index,
        "model_id": "model",
        "checkpoint_id": "checkpoint",
        "temperature": 0.7,
        "explanation": "text",
        "generation_ok": True,
        "failure_reason": None,
    }


def segmentation(claims: list[tuple[str, str]]) -> dict:
    return {
        "claims": [
            {"claim_id": f"c{i:04d}", "claim": text, "specificity": specificity}
            for i, (text, specificity) in enumerate(claims)
        ]
    }


class SegmentationTests(unittest.TestCase):
    def test_lossless_tiling_and_local_offsets(self) -> None:
        text = "Broad biology; Alice ran."
        raw = {
            "units": [
                {"source_text": "Broad biology", "kind": "claim", "claim": "The topic is biology.",
                 "specificity": "broad_theme"},
                {"source_text": "; ", "kind": "nonclaim", "claim": "", "specificity": "not_applicable"},
                {"source_text": "Alice ran", "kind": "claim", "claim": "Alice ran.",
                 "specificity": "entity_detail"},
                {"source_text": ".", "kind": "nonclaim", "claim": "", "specificity": "not_applicable"},
            ]
        }
        result = validate_segmentation_output(text, raw)
        self.assertTrue(result["lossless"])
        self.assertEqual(result["n_claims"], 2)
        self.assertEqual(result["claims"][0]["char_start"], 0)
        self.assertEqual(result["claims"][1]["char_start"], len("Broad biology; "))

    def test_nonlossless_tiling_fails_closed(self) -> None:
        raw = {"units": [{"source_text": "changed", "kind": "claim", "claim": "A claim.",
                           "specificity": "entity_detail"}]}
        with self.assertRaisesRegex(ValidationError, "not lossless"):
            validate_segmentation_output("source", raw)

    def test_zero_claim_output_fails_closed(self) -> None:
        raw = {"units": [{"source_text": "heading", "kind": "nonclaim", "claim": "",
                           "specificity": "not_applicable"}]}
        with self.assertRaisesRegex(ValidationError, "zero substantive claims"):
            validate_segmentation_output("heading", raw)


class EquivalenceAndScoringTests(unittest.TestCase):
    def test_cross_specificity_edge_is_rejected(self) -> None:
        left = segmentation([("Biology topic.", "broad_theme")])["claims"]
        right = segmentation([("Alice studies biology.", "entity_detail")])["claims"]
        raw = {"all_pairs_reviewed": True,
               "matches": [{"left_id": "c0000", "right_id": "c0000", "rationale": "same"}]}
        with self.assertRaisesRegex(ValidationError, "crosses specificity"):
            validate_equivalence_output(raw, left, right)

    def test_existential_matching_is_not_one_to_one(self) -> None:
        pair = {
            "pair_id": "p", "pair_type": "cross_initialization", "activation_id": "a",
            "document_id": "d", "trajectory_id": None, "distribution": "fineweb",
            "benchmark": "fineweb", "family": "fineweb", "left_condition": "plausible",
            "left_sample_index": 0, "right_condition": "implausible", "right_sample_index": 0,
        }
        left = segmentation([("Alice ran.", "entity_detail"), ("Alice ran.", "entity_detail")])
        right = segmentation([("Alice ran.", "entity_detail")])
        equivalence = {"matches": [
            {"left_id": "c0000", "right_id": "c0000", "rationale": "same proposition"},
            {"left_id": "c0001", "right_id": "c0000", "rationale": "repeated proposition"},
        ]}
        scores = score_pair(PairScoreInput(pair, left, right, equivalence))
        complete_all = next(row for row in scores if row["analysis"] == "complete_case" and row["stratum"] == "all")
        self.assertEqual(complete_all["p_left_to_right"], 1.0)
        self.assertEqual(complete_all["p_right_to_left"], 1.0)
        self.assertEqual(complete_all["a_cross"], 1.0)

    def test_failure_excluded_primary_and_zero_in_sensitivity(self) -> None:
        pair = {
            "pair_id": "p", "pair_type": "cross_initialization", "activation_id": "a",
            "document_id": "d", "trajectory_id": None, "distribution": "fineweb",
            "benchmark": "fineweb", "family": "fineweb", "left_condition": "plausible",
            "left_sample_index": 0, "right_condition": "implausible", "right_sample_index": 0,
        }
        rows = score_pair(PairScoreInput(pair, None, None, None, "api_failure"))
        primary = next(row for row in rows if row["analysis"] == "complete_case" and row["stratum"] == "all")
        sensitivity = next(row for row in rows if row["analysis"] == "zero_failure" and row["stratum"] == "all")
        self.assertFalse(primary["included"])
        self.assertIsNone(primary["a_cross"])
        self.assertTrue(sensitivity["included"])
        self.assertEqual(sensitivity["a_cross"], 0.0)


class PairPlanTests(unittest.TestCase):
    def test_two_samples_produce_four_cross_and_two_baselines(self) -> None:
        rows = [
            explanation_row("a", condition, sample)
            for condition in ("plausible", "implausible") for sample in (0, 1)
        ]
        pairs = build_pair_plan(rows)
        counts = {kind: sum(pair["pair_type"] == kind for pair in pairs) for kind in (
            "cross_initialization", "within_plausible", "within_implausible"
        )}
        self.assertEqual(counts, {
            "cross_initialization": 4,
            "within_plausible": 1,
            "within_implausible": 1,
        })
        for pair in pairs:
            if pair["pair_type"] == "cross_initialization":
                self.assertEqual(pair["left_condition"], "plausible")
                self.assertEqual(pair["right_condition"], "implausible")


if __name__ == "__main__":
    unittest.main()
