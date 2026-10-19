"""Score completed blinded segmentation and decoy-review sheets."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from .io import atomic_json, read_jsonl, sha256_file
from .spec import PROTOCOL_SHA256, TEXT_MODEL_CORRECTION_AMENDMENT_SHA256


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _label(value: str, *, allowed: set[str], field: str) -> str:
    normalized = value.strip().casefold()
    if normalized not in allowed:
        raise ValueError(f"{field} must be one of {sorted(allowed)}, observed {value!r}")
    return normalized


def score_manual_audit(
    *,
    segmentation_review: str | Path,
    decoy_review: str | Path,
    decoy_answer_key: str | Path,
    exact_tiling_failures: int = 0,
    audit_half: str = "fineweb",
) -> dict[str, Any]:
    if audit_half not in {"fineweb", "ood"}:
        raise ValueError("audit_half must be fineweb or ood")
    segment_rows = _read_csv(segmentation_review)
    decoy_rows = _read_csv(decoy_review)
    answer_rows = list(read_jsonl(decoy_answer_key))
    if len(segment_rows) != 100:
        raise ValueError(f"FineWeb segmentation review must contain 100 rows, observed {len(segment_rows)}")
    if len(decoy_rows) != 200 or len(answer_rows) != 200:
        raise ValueError("FineWeb decoy review and answer key must each contain 200 rows")

    atomic = [
        _label(
            row["review_atomic_claims_acceptable_yes_no"],
            allowed={"yes", "no"},
            field="review_atomic_claims_acceptable_yes_no",
        )
        for row in segment_rows
    ]
    complete = [
        _label(
            row["review_all_semantic_claims_captured_yes_no"],
            allowed={"yes", "no"},
            field="review_all_semantic_claims_captured_yes_no",
        )
        for row in segment_rows
    ]

    answer_by_id = {str(row["audit_id"]): row for row in answer_rows}
    review_ids = [str(row["audit_id"]) for row in decoy_rows]
    if len(answer_by_id) != 200 or len(set(review_ids)) != 200 or set(review_ids) != set(answer_by_id):
        raise ValueError("decoy review and answer-key audit IDs must match one-to-one")
    absence_matches = 0
    natural_matches = 0
    for row in decoy_rows:
        answer = answer_by_id[str(row["audit_id"])]
        manual_absence = _label(
            row["review_source_absent_pass_fail_uncertain"],
            allowed={"pass", "fail", "uncertain"},
            field="review_source_absent_pass_fail_uncertain",
        )
        manual_natural = _label(
            row["review_splice_natural_pass_fail_uncertain"],
            allowed={"pass", "fail", "uncertain"},
            field="review_splice_natural_pass_fail_uncertain",
        )
        automated_absence = {
            "absent": "pass",
            "fail": "fail",
            "uncertain": "uncertain",
        }[str(answer["automated_absence"])]
        automated_natural = str(answer["automated_natural_label"])
        absence_matches += int(manual_absence == automated_absence)
        natural_matches += int(manual_natural == automated_natural)

    atomic_rate = atomic.count("yes") / len(atomic)
    completeness_rate = complete.count("yes") / len(complete)
    absence_agreement = absence_matches / len(decoy_rows)
    naturalness_agreement = natural_matches / len(decoy_rows)
    gates = {
        "exact_tiling": int(exact_tiling_failures) == 0,
        "atomic_claim_acceptability": atomic_rate >= 0.90,
        "claim_inventory_completeness": completeness_rate >= 0.90,
        "source_absence_agreement": absence_agreement >= 0.90,
        "splice_naturalness_agreement": naturalness_agreement >= 0.90,
    }
    return {
        "schema_version": 1,
        "protocol_sha256": PROTOCOL_SHA256,
        "text_model_correction_amendment_sha256": (
            TEXT_MODEL_CORRECTION_AMENDMENT_SHA256
        ),
        "audit_half": audit_half,
        "segmentation_items": len(segment_rows),
        "decoy_items": len(decoy_rows),
        "exact_tiling_failures": int(exact_tiling_failures),
        "atomic_claim_acceptability": atomic_rate,
        "claim_inventory_completeness": completeness_rate,
        "source_absence_agreement": absence_agreement,
        "splice_naturalness_agreement": naturalness_agreement,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "artifact_sha256": {
            "segmentation_review": sha256_file(segmentation_review),
            "decoy_review": sha256_file(decoy_review),
            "decoy_answer_key": sha256_file(decoy_answer_key),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segmentation-review", required=True)
    parser.add_argument("--decoy-review", required=True)
    parser.add_argument("--decoy-answer-key", required=True)
    parser.add_argument("--exact-tiling-failures", type=int, default=0)
    parser.add_argument("--audit-half", choices=("fineweb", "ood"), default="fineweb")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = score_manual_audit(
        segmentation_review=args.segmentation_review,
        decoy_review=args.decoy_review,
        decoy_answer_key=args.decoy_answer_key,
        exact_tiling_failures=args.exact_tiling_failures,
        audit_half=args.audit_half,
    )
    atomic_json(args.out, result)
    print(f"all_gates_pass={result['all_gates_pass']}")


if __name__ == "__main__":
    main()
