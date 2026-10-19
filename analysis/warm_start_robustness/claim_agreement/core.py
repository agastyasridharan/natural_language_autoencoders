"""Pure-CPU schemas, validation, pairing, and claim-agreement scoring.

This module deliberately contains no API code.  It is the deterministic boundary between
the cached LLM instruments and the statistical analysis.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


PROTOCOL_SCHEMA_VERSION = "nla.claim_agreement.protocol.v1"
EXPLANATION_SCHEMA_VERSION = "nla.claim_agreement.explanation.v1"
SEGMENTATION_SCHEMA_VERSION = "nla.claim_agreement.segmentation.v1"
EQUIVALENCE_SCHEMA_VERSION = "nla.claim_agreement.equivalence.v1"
PAIR_SCHEMA_VERSION = "nla.claim_agreement.pair.v1"
SCORE_SCHEMA_VERSION = "nla.claim_agreement.score.v1"

CONDITIONS = ("plausible", "implausible")
SPECIFICITIES = ("broad_theme", "entity_detail")
STRATA = ("all",) + SPECIFICITIES
PAIR_TYPES = ("cross_initialization", "within_plausible", "within_implausible")
FAMILIES = ("fineweb", "alignment", "capability")
DISTRIBUTIONS = ("fineweb", "ood")
ALIGNMENT_BENCHMARKS = (
    "blackmail",
    "destructive_coding_actions",
    "overly_agentic_actions",
    "factual_honesty",
    "appropriate_harmlessness",
    "sycophancy",
    "multi_turn_deception",
)
CAPABILITY_BENCHMARKS = (
    "swe_bench_verified",
    "gpqa_diamond",
    "humanitys_last_exam",
    "tau2_bench_retail",
    "mmmlu",
    "browsecomp",
    "terminal_bench_2",
    "financeagent",
    "cybergym",
)
OOD_BENCHMARKS = ALIGNMENT_BENCHMARKS + CAPABILITY_BENCHMARKS


class ValidationError(ValueError):
    """An artifact violates the frozen protocol schema."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(paths: Sequence[Path]) -> Iterator[dict[str, Any]]:
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValidationError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
                if not isinstance(value, dict):
                    raise ValidationError(f"{path}:{line_number}: expected a JSON object")
                yield value


def validate_explanation_row(row: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "schema_version",
        "activation_id",
        "document_id",
        "trajectory_id",
        "distribution",
        "benchmark",
        "family",
        "condition",
        "sample_index",
        "model_id",
        "checkpoint_id",
        "temperature",
        "explanation",
        "generation_ok",
        "failure_reason",
    }
    missing = sorted(required - row.keys())
    if missing:
        raise ValidationError(f"explanation row missing fields: {missing}")
    out = dict(row)
    if out["schema_version"] != EXPLANATION_SCHEMA_VERSION:
        raise ValidationError(f"unsupported explanation schema {out['schema_version']!r}")
    for field in ("activation_id", "benchmark", "model_id", "checkpoint_id"):
        if not isinstance(out[field], str) or not out[field].strip():
            raise ValidationError(f"{field} must be a non-empty string")
    if out["condition"] not in CONDITIONS:
        raise ValidationError(f"condition must be one of {CONDITIONS}")
    if out["distribution"] not in DISTRIBUTIONS:
        raise ValidationError(f"distribution must be one of {DISTRIBUTIONS}")
    if out["family"] not in FAMILIES:
        raise ValidationError(f"family must be one of {FAMILIES}")
    if out["distribution"] == "fineweb":
        if out["family"] != "fineweb":
            raise ValidationError("FineWeb rows must use family='fineweb'")
        if not isinstance(out["document_id"], str) or not out["document_id"]:
            raise ValidationError("FineWeb rows require a non-empty document_id")
        if out["trajectory_id"] is not None:
            raise ValidationError("FineWeb rows must set trajectory_id=null")
    else:
        if out["family"] not in ("alignment", "capability"):
            raise ValidationError("OOD rows require family alignment or capability")
        if not isinstance(out["trajectory_id"], str) or not out["trajectory_id"]:
            raise ValidationError("OOD rows require a non-empty trajectory_id")
        if out["document_id"] is not None:
            raise ValidationError("OOD rows must set document_id=null")
    if isinstance(out["sample_index"], bool) or not isinstance(out["sample_index"], int) or out["sample_index"] < 0:
        raise ValidationError("sample_index must be an integer >= 0")
    if not isinstance(out["temperature"], (int, float)) or isinstance(out["temperature"], bool):
        raise ValidationError("temperature must be numeric")
    if not math.isclose(float(out["temperature"]), 0.7, abs_tol=1e-9):
        raise ValidationError("claim-agreement explanations must use temperature=0.7")
    if not isinstance(out["generation_ok"], bool):
        raise ValidationError("generation_ok must be boolean")
    if not isinstance(out["explanation"], str):
        raise ValidationError("explanation must be a string")
    if out["generation_ok"]:
        if not out["explanation"].strip():
            raise ValidationError("a successful generation must have a non-empty explanation")
        if out["failure_reason"] is not None:
            raise ValidationError("successful generations must set failure_reason=null")
    elif not isinstance(out["failure_reason"], str) or not out["failure_reason"]:
        raise ValidationError("failed generations require a non-empty failure_reason")
    return out


def load_explanations(paths: Sequence[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: dict[tuple[str, str, int], str] = {}
    activation_meta: dict[str, tuple[Any, ...]] = {}
    for raw in read_jsonl(paths):
        row = validate_explanation_row(raw)
        key = (row["activation_id"], row["condition"], row["sample_index"])
        digest = content_hash(row)
        if key in seen:
            if seen[key] == digest:
                continue
            raise ValidationError(f"conflicting duplicate explanation key: {key}")
        seen[key] = digest
        meta = (
            row["document_id"], row["trajectory_id"], row["distribution"],
            row["benchmark"], row["family"], row["temperature"],
        )
        prior = activation_meta.setdefault(row["activation_id"], meta)
        if prior != meta:
            raise ValidationError(f"inconsistent metadata for activation_id={row['activation_id']!r}")
        rows.append(row)
    rows.sort(key=lambda r: (r["activation_id"], r["condition"], r["sample_index"]))
    return rows


def segmentation_request_key(explanation: str, protocol_version: str) -> str:
    return content_hash({"stage": "segment", "protocol": protocol_version, "explanation": explanation})


def _claim_id(index: int) -> str:
    return f"c{index:04d}"


def validate_segmentation_output(explanation: str, output: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize a lossless, ordered tiling returned by the segmenter.

    The concatenated ``source_text`` fields must reproduce the explanation byte-for-byte.
    Character offsets are then computed locally, avoiding unreliable model-generated offsets.
    """

    if set(output) != {"units"} or not isinstance(output.get("units"), list):
        raise ValidationError("segmentation output must contain only an array field named units")
    if not output["units"]:
        raise ValidationError("a non-empty explanation requires at least one segmentation unit")
    rebuilt: list[str] = []
    normalized_units: list[dict[str, Any]] = []
    claims: list[dict[str, Any]] = []
    cursor = 0
    for unit_index, raw in enumerate(output["units"]):
        if not isinstance(raw, dict):
            raise ValidationError(f"unit {unit_index} must be an object")
        expected_fields = {"source_text", "kind", "claim", "specificity"}
        if set(raw) != expected_fields:
            raise ValidationError(f"unit {unit_index} fields must be exactly {sorted(expected_fields)}")
        source_text = raw["source_text"]
        kind = raw["kind"]
        claim = raw["claim"]
        specificity = raw["specificity"]
        if not isinstance(source_text, str) or not source_text:
            raise ValidationError(f"unit {unit_index} source_text must be non-empty")
        if kind not in ("claim", "nonclaim"):
            raise ValidationError(f"unit {unit_index} has invalid kind")
        if not isinstance(claim, str):
            raise ValidationError(f"unit {unit_index} claim must be a string")
        if kind == "claim":
            if not claim.strip():
                raise ValidationError(f"claim unit {unit_index} requires a standalone normalized claim")
            if specificity not in SPECIFICITIES:
                raise ValidationError(f"claim unit {unit_index} requires a substantive specificity")
        else:
            if claim != "" or specificity != "not_applicable":
                raise ValidationError(f"nonclaim unit {unit_index} must use empty claim/not_applicable")
        end = cursor + len(source_text)
        unit = {
            "unit_index": unit_index,
            "source_text": source_text,
            "char_start": cursor,
            "char_end": end,
            "kind": kind,
            "claim": claim,
            "specificity": specificity,
        }
        normalized_units.append(unit)
        if kind == "claim":
            claim_record = {
                "claim_id": _claim_id(len(claims)),
                "unit_index": unit_index,
                "source_text": source_text,
                "char_start": cursor,
                "char_end": end,
                "claim": claim.strip(),
                "specificity": specificity,
            }
            claims.append(claim_record)
        rebuilt.append(source_text)
        cursor = end
    if "".join(rebuilt) != explanation:
        mismatch = next(
            (i for i, (a, b) in enumerate(zip("".join(rebuilt), explanation)) if a != b),
            min(len("".join(rebuilt)), len(explanation)),
        )
        raise ValidationError(
            f"segmentation is not lossless: concatenated units differ at character {mismatch} "
            f"(rebuilt={len(''.join(rebuilt))}, source={len(explanation)})"
        )
    if not claims:
        raise ValidationError("segmenter returned zero substantive claims")
    return {
        "schema_version": SEGMENTATION_SCHEMA_VERSION,
        "lossless": True,
        "n_units": len(normalized_units),
        "n_claims": len(claims),
        "units": normalized_units,
        "claims": claims,
    }


def equivalence_request_key(
    left_claims: Sequence[Mapping[str, Any]],
    right_claims: Sequence[Mapping[str, Any]],
    protocol_version: str,
) -> str:
    def projection(claims: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
        return [
            {"claim_id": str(c["claim_id"]), "claim": str(c["claim"]), "specificity": str(c["specificity"])}
            for c in claims
        ]

    return content_hash({
        "stage": "equivalence",
        "protocol": protocol_version,
        "left": projection(left_claims),
        "right": projection(right_claims),
    })


def validate_equivalence_output(
    output: Mapping[str, Any],
    left_claims: Sequence[Mapping[str, Any]],
    right_claims: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if set(output) != {"all_pairs_reviewed", "matches"}:
        raise ValidationError("equivalence output fields must be all_pairs_reviewed and matches")
    if output["all_pairs_reviewed"] is not True:
        raise ValidationError("equivalence judge did not attest that all candidate pairs were reviewed")
    if not isinstance(output["matches"], list):
        raise ValidationError("matches must be an array")
    left_by_id = {str(c["claim_id"]): c for c in left_claims}
    right_by_id = {str(c["claim_id"]): c for c in right_claims}
    seen: set[tuple[str, str]] = set()
    matches: list[dict[str, str]] = []
    for index, raw in enumerate(output["matches"]):
        if not isinstance(raw, dict) or set(raw) != {"left_id", "right_id", "rationale"}:
            raise ValidationError(f"match {index} has invalid fields")
        left_id, right_id, rationale = raw["left_id"], raw["right_id"], raw["rationale"]
        if left_id not in left_by_id or right_id not in right_by_id:
            raise ValidationError(f"match {index} references an unknown claim id")
        if left_by_id[left_id]["specificity"] != right_by_id[right_id]["specificity"]:
            raise ValidationError(f"match {index} crosses specificity levels")
        if not isinstance(rationale, str) or not rationale.strip():
            raise ValidationError(f"match {index} requires a concise rationale")
        edge = (left_id, right_id)
        if edge in seen:
            raise ValidationError(f"duplicate equivalence edge {edge}")
        seen.add(edge)
        matches.append({"left_id": left_id, "right_id": right_id, "rationale": rationale.strip()})
    matches.sort(key=lambda m: (m["left_id"], m["right_id"]))
    return {
        "schema_version": EQUIVALENCE_SCHEMA_VERSION,
        "all_pairs_reviewed": True,
        "matches": matches,
    }


def _pair_id(pair_type: str, activation_id: str, left: Mapping[str, Any], right: Mapping[str, Any]) -> str:
    return content_hash({
        "schema": PAIR_SCHEMA_VERSION,
        "pair_type": pair_type,
        "activation_id": activation_id,
        "left": [left["condition"], left["sample_index"]],
        "right": [right["condition"], right["sample_index"]],
    })[:24]


def build_pair_plan(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Use every cross-condition Cartesian pair and every unordered within-condition pair."""

    by_activation: dict[str, dict[str, list[Mapping[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_activation[str(row["activation_id"])][str(row["condition"])].append(row)
    pairs: list[dict[str, Any]] = []
    for activation_id in sorted(by_activation):
        condition_rows = by_activation[activation_id]
        plausible = sorted(condition_rows.get("plausible", []), key=lambda r: int(r["sample_index"]))
        implausible = sorted(condition_rows.get("implausible", []), key=lambda r: int(r["sample_index"]))
        representatives = plausible or implausible
        if not representatives:
            continue
        meta = representatives[0]

        def add(pair_type: str, left: Mapping[str, Any], right: Mapping[str, Any]) -> None:
            pairs.append({
                "schema_version": PAIR_SCHEMA_VERSION,
                "pair_id": _pair_id(pair_type, activation_id, left, right),
                "pair_type": pair_type,
                "activation_id": activation_id,
                "document_id": meta["document_id"],
                "trajectory_id": meta["trajectory_id"],
                "distribution": meta["distribution"],
                "benchmark": meta["benchmark"],
                "family": meta["family"],
                "left_condition": left["condition"],
                "left_sample_index": left["sample_index"],
                "right_condition": right["condition"],
                "right_sample_index": right["sample_index"],
            })

        for left in plausible:
            for right in implausible:
                add("cross_initialization", left, right)
        for condition, items in (("plausible", plausible), ("implausible", implausible)):
            for i, left in enumerate(items):
                for right in items[i + 1:]:
                    add(f"within_{condition}", left, right)
    pairs.sort(key=lambda p: (p["activation_id"], p["pair_type"], p["left_sample_index"], p["right_sample_index"]))
    return pairs


def explanation_key(row: Mapping[str, Any]) -> tuple[str, str, int]:
    return (str(row["activation_id"]), str(row["condition"]), int(row["sample_index"]))


def pair_endpoint_keys(pair: Mapping[str, Any]) -> tuple[tuple[str, str, int], tuple[str, str, int]]:
    return (
        (str(pair["activation_id"]), str(pair["left_condition"]), int(pair["left_sample_index"])),
        (str(pair["activation_id"]), str(pair["right_condition"]), int(pair["right_sample_index"])),
    )


def harmonic(left: float, right: float) -> float:
    if left <= 0.0 or right <= 0.0:
        return 0.0
    return 2.0 * left * right / (left + right)


@dataclass(frozen=True)
class PairScoreInput:
    pair: Mapping[str, Any]
    left_segmentation: Mapping[str, Any] | None
    right_segmentation: Mapping[str, Any] | None
    equivalence: Mapping[str, Any] | None
    failure_reason: str | None = None


def score_pair(value: PairScoreInput) -> list[dict[str, Any]]:
    """Score all/broad/detail strata under complete-case and zero-failure conventions.

    A claim counts as matched if *any* counterpart is equivalent, exactly matching the paper's
    existential definition.  Matching is intentionally not constrained to be one-to-one.
    """

    pair = value.pair
    base = {
        "schema_version": SCORE_SCHEMA_VERSION,
        **{key: pair[key] for key in (
            "pair_id", "pair_type", "activation_id", "document_id", "trajectory_id",
            "distribution", "benchmark", "family", "left_condition", "left_sample_index",
            "right_condition", "right_sample_index",
        )},
    }
    upstream_complete = (
        value.left_segmentation is not None
        and value.right_segmentation is not None
        and value.equivalence is not None
        and value.failure_reason is None
    )
    left_claims_all = list(value.left_segmentation.get("claims", [])) if value.left_segmentation else []
    right_claims_all = list(value.right_segmentation.get("claims", [])) if value.right_segmentation else []
    raw_matches = list(value.equivalence.get("matches", [])) if value.equivalence else []
    rows: list[dict[str, Any]] = []
    for stratum in STRATA:
        left_claims = [c for c in left_claims_all if stratum == "all" or c["specificity"] == stratum]
        right_claims = [c for c in right_claims_all if stratum == "all" or c["specificity"] == stratum]
        left_ids = {str(c["claim_id"]) for c in left_claims}
        right_ids = {str(c["claim_id"]) for c in right_claims}
        matches = [m for m in raw_matches if m["left_id"] in left_ids and m["right_id"] in right_ids]
        matched_left = {m["left_id"] for m in matches}
        matched_right = {m["right_id"] for m in matches}
        denominators_defined = bool(left_claims) and bool(right_claims)
        complete = upstream_complete and denominators_defined
        if complete:
            p_left = len(matched_left) / len(left_claims)
            p_right = len(matched_right) / len(right_claims)
            a_cross = harmonic(p_left, p_right)
            reason = None
        else:
            p_left = None
            p_right = None
            a_cross = None
            if value.failure_reason:
                reason = value.failure_reason
            elif not upstream_complete:
                reason = "incomplete_instrument_output"
            else:
                reason = "zero_claim_denominator"
        common = {
            **base,
            "stratum": stratum,
            "n_left_claims": len(left_claims) if value.left_segmentation is not None else None,
            "n_right_claims": len(right_claims) if value.right_segmentation is not None else None,
            "n_matched_left": len(matched_left) if complete else None,
            "n_matched_right": len(matched_right) if complete else None,
            "complete": complete,
            "failure_reason": reason,
        }
        rows.append({
            **common,
            "analysis": "complete_case",
            "p_left_to_right": p_left,
            "p_right_to_left": p_right,
            "a_cross": a_cross,
            "included": complete,
        })
        rows.append({
            **common,
            "analysis": "zero_failure",
            "p_left_to_right": p_left if complete else 0.0,
            "p_right_to_left": p_right if complete else 0.0,
            "a_cross": a_cross if complete else 0.0,
            "included": True,
        })
    return rows


def cluster_id(row: Mapping[str, Any]) -> str:
    if row["distribution"] == "fineweb":
        raw = row.get("document_id")
        if not raw:
            raise ValidationError("FineWeb score row lacks document_id")
    else:
        raw = row.get("trajectory_id")
        if not raw:
            raise ValidationError("OOD score row lacks trajectory_id")
    return f"{row['distribution']}|{row['benchmark']}|{raw}"


def collapse_pair_variants(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Average replicate-pair variants within activation before population inference.

    This gives every activation one vote even though all-pairs pairing creates multiple correlated
    comparisons.  Document/trajectory bootstrap clustering is applied after this collapse.
    """

    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    group_fields = (
        "activation_id", "distribution", "benchmark", "family", "pair_type", "stratum", "analysis",
        "document_id", "trajectory_id",
    )
    for row in rows:
        if row.get("included"):
            groups[tuple(row.get(field) for field in group_fields)].append(row)
    collapsed: list[dict[str, Any]] = []
    for key, members in sorted(groups.items(), key=lambda item: tuple(str(x) for x in item[0])):
        first = members[0]
        collapsed.append({
            **{field: key[index] for index, field in enumerate(group_fields)},
            "cluster_id": cluster_id(first),
            "n_pair_variants": len(members),
            "p_left_to_right": sum(float(r["p_left_to_right"]) for r in members) / len(members),
            "p_right_to_left": sum(float(r["p_right_to_left"]) for r in members) / len(members),
            "a_cross": sum(float(r["a_cross"]) for r in members) / len(members),
            "n_complete_pair_variants": sum(bool(r["complete"]) for r in members),
        })
    return collapsed
