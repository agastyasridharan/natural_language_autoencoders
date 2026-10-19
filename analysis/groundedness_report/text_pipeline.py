"""Prompts and strict schemas for segmentation, taxonomy, and matched decoys."""
from __future__ import annotations

import json
import re
from typing import Any, Mapping, Sequence

from .openai_text import JSONResponse, TerraResponses


_CHUNK_PATTERN = re.compile(r"\w+|[^\w\s]|\s+", re.UNICODE)


def _claim_ranges_schema(last_token: int) -> dict[str, Any]:
    if last_token < 0:
        raise ValueError("an explanation must contain at least one chunk")
    index = {"type": "integer", "minimum": 0, "maximum": last_token}
    return {
        "type": "object",
        "properties": {
            "claims": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "properties": {
                        "first_token": dict(index),
                        "last_token": dict(index),
                    },
                    "required": ["first_token", "last_token"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["claims"],
        "additionalProperties": False,
    }

TAXONOMY_SCHEMA = {
    "type": "object",
    "properties": {
        "claim_class": {"type": "string", "enum": ["theme", "detail"]},
        "subtype": {
            "type": "string",
            "enum": ["genre", "topic", "entity", "structure", "era", "quote", "date", "number", "value", "other"],
        },
    },
    "required": ["claim_class", "subtype"],
    "additionalProperties": False,
}

DECOY_SCHEMA = {
    "type": "object",
    "properties": {
        "candidates": {
            "type": "array",
            "minItems": 64,
            "maxItems": 64,
            "items": {"type": "string", "minLength": 1},
        }
    },
    "required": ["candidates"],
    "additionalProperties": False,
}

JUDGMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "judgments": {
            "type": "array",
            "minItems": 64,
            "maxItems": 64,
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string", "enum": ["pass", "fail", "uncertain"]},
                    "reason": {"type": "string"},
                },
                "required": ["label", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["judgments"],
    "additionalProperties": False,
}


def _lossless_chunks(explanation: str) -> list[dict[str, Any]]:
    chunks = [
        {"token": token, "text": match.group(), "char_span": [match.start(), match.end()]}
        for token, match in enumerate(_CHUNK_PATTERN.finditer(explanation))
    ]
    if not chunks or "".join(str(chunk["text"]) for chunk in chunks) != explanation:
        raise ValueError("lossless chunking invariant failed")
    return chunks


def _segments_with_spans(explanation: str, value: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Slice exact ordered claim ranges and make untouched gaps connectives."""
    chunks = _lossless_chunks(explanation)
    cursor = 0
    previous_last_token = -1
    outputs: list[dict[str, Any]] = []
    claims = value.get("claims") or []
    if not claims:
        raise ValueError("segmentation produced no claims")
    for claim_index, claim in enumerate(claims):
        first_token = int(claim["first_token"])
        last_token = int(claim["last_token"])
        if not (0 <= first_token <= last_token < len(chunks)):
            raise ValueError(
                f"claim {claim_index} has invalid range {first_token}-{last_token}"
            )
        if first_token <= previous_last_token:
            raise ValueError(f"claim {claim_index} overlaps, nests, or is out of order")
        if str(chunks[first_token]["text"]).isspace() or str(chunks[last_token]["text"]).isspace():
            raise ValueError(f"claim {claim_index} has a whitespace boundary")
        start = int(chunks[first_token]["char_span"][0])
        end = int(chunks[last_token]["char_span"][1])
        if start > cursor:
            outputs.append({
                "kind": "connective",
                "text": explanation[cursor:start],
                "char_span": [cursor, start],
            })
        outputs.append({
            "kind": "claim",
            "text": explanation[start:end],
            "char_span": [start, end],
            "token_span_inclusive": [first_token, last_token],
        })
        cursor = end
        previous_last_token = last_token
    if cursor < len(explanation):
        outputs.append({
            "kind": "connective",
            "text": explanation[cursor:],
            "char_span": [cursor, len(explanation)],
        })
    if "".join(segment["text"] for segment in outputs) != explanation:
        raise AssertionError("internal error: derived segments do not exactly tile the explanation")
    return outputs


def segment_explanation(
    client: TerraResponses,
    explanation: str,
    *,
    cache_path: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    system = (
        "Identify every atomic semantic claim in an NLA explanation. The input is a lossless ordered "
        "list of indexed source chunks. Return one inclusive token-index range per claim, in source "
        "order. Ranges must be nonoverlapping, must not nest, and each range must express one "
        "independently testable semantic assertion. Start and end on non-whitespace chunks. Leave "
        "headings, list markers, discourse transitions, and purely grammatical material outside "
        "ranges. Include uncertain or incorrect assertions. Never return the same content at "
        "multiple levels of granularity."
    )
    chunks = _lossless_chunks(explanation)
    chunk_payload = [{"token": chunk["token"], "text": chunk["text"]} for chunk in chunks]
    result = client.request_json(
        task_name="nla_indexed_claim_ranges_v5",
        system_prompt=system,
        user_prompt=json.dumps({
            "instruction": "Return claim ranges over this exact chunk list.",
            "chunks": chunk_payload,
        }, ensure_ascii=False),
        schema=_claim_ranges_schema(len(chunks) - 1),
        max_output_tokens=4096,
        cache_path=cache_path,
    )
    try:
        segments = _segments_with_spans(explanation, result.value)
    except ValueError as error:
        adjudication_cache_path = None
        if cache_path is not None:
            source = str(cache_path)
            adjudication_cache_path = (
                source[:-5] + ".adjudication.json" if source.endswith(".json")
                else source + ".adjudication.json"
            )
        adjudication = client.request_json(
            task_name="nla_indexed_claim_ranges_adjudication_v5",
            system_prompt=(
                "Correct a structurally invalid proposed inventory of atomic semantic claims over "
                "a lossless indexed source. Return a complete replacement inventory, not a patch. "
                "Use one inclusive token-index range per atomic claim in source order. Ranges must "
                "be nonoverlapping and nonnested, and must start and end on non-whitespace chunks. "
                "Remove redundant granularity: never represent the same content both as a broad "
                "range and a nested subclaim. Preserve all distinct semantic claims, including "
                "uncertain or incorrect assertions."
            ),
            user_prompt=json.dumps({
                "validation_error": str(error),
                "invalid_proposal": result.value,
                "chunks": chunk_payload,
            }, ensure_ascii=False),
            schema=_claim_ranges_schema(len(chunks) - 1),
            max_output_tokens=4096,
            cache_path=adjudication_cache_path,
        )
        segments = _segments_with_spans(explanation, adjudication.value)
        return segments, {
            "primary": result.audit,
            "primary_validation_error": str(error),
            "adjudication_used": True,
            "adjudication": adjudication.audit,
        }
    return segments, {
        "primary": result.audit,
        "primary_validation_error": None,
        "adjudication_used": False,
        "adjudication": None,
    }


def classify_claim_text(
    client: TerraResponses,
    *,
    explanation: str,
    claim: str,
    cache_path: str | None = None,
) -> JSONResponse:
    system = (
        "Classify one atomic NLA-explanation claim. Theme claims describe genre, topic, entity, "
        "structure, era, or another broad semantic property. Detail claims assert a quote, date, "
        "number, value, or another specific detail. Theme subtypes are genre/topic/entity/structure/"
        "era/other; detail subtypes are quote/date/number/value/other. Use only a subtype valid for "
        "the selected class."
    )
    result = client.request_json(
        task_name="nla_claim_taxonomy_v1",
        system_prompt=system,
        user_prompt=json.dumps({"explanation": explanation, "claim": claim}, ensure_ascii=False),
        schema=TAXONOMY_SCHEMA,
        max_output_tokens=256,
        cache_path=cache_path,
    )
    valid = {
        "theme": {"genre", "topic", "entity", "structure", "era", "other"},
        "detail": {"quote", "date", "number", "value", "other"},
    }
    if result.value["subtype"] not in valid[result.value["claim_class"]]:
        raise ValueError("taxonomy subtype is invalid for its claim class")
    return result


def generate_decoys(
    client: TerraResponses,
    *,
    explanation: str,
    claim: str,
    claim_class: str,
    subtype: str,
    cache_path: str | None = None,
) -> JSONResponse:
    system = (
        "Generate exactly 64 distinct sibling replacements for the marked atomic claim. Each must "
        "have the same claim class and subtype, comparable specificity and length, and fit the same "
        "grammatical/discourse slot. Change the substantive content while retaining the broad theme. "
        "Do not reproduce or paraphrase any claim already in the explanation. Before returning, "
        "strip and casefold every candidate and verify that all 64 are nonempty and pairwise "
        "distinct. Return only candidate text for the exact replacement span, without quotation "
        "marks or commentary."
    )
    result = client.request_json(
        task_name="nla_decoy_generation_v2",
        system_prompt=system,
        user_prompt=json.dumps({
            "explanation": explanation,
            "claim": claim,
            "claim_class": claim_class,
            "subtype": subtype,
        }, ensure_ascii=False),
        schema=DECOY_SCHEMA,
        max_output_tokens=8192,
        cache_path=cache_path,
    )
    folded = [str(value).strip().casefold() for value in result.value["candidates"]]
    if len(set(folded)) == 64 and all(folded):
        return JSONResponse(value=result.value, audit={
            "primary": result.audit,
            "primary_validation_error": None,
            "regeneration_used": False,
            "regeneration": None,
        })
    validation_error = "primary pool contains an empty or strip-and-casefold-duplicate candidate"
    regeneration_cache_path = None
    if cache_path is not None:
        source = str(cache_path)
        regeneration_cache_path = (
            source[:-5] + ".regeneration.json" if source.endswith(".json")
            else source + ".regeneration.json"
        )
    regeneration = client.request_json(
        task_name="nla_decoy_generation_regeneration_v2",
        system_prompt=(
            "Replace an invalid pool with exactly 64 distinct sibling replacements for the marked "
            "atomic claim. Return a complete replacement pool, not a patch. Each replacement must "
            "have the same claim class and subtype, comparable specificity and length, and fit the "
            "same grammatical and discourse slot. Change substantive content while retaining the "
            "broad theme. Do not reproduce or paraphrase any claim already in the explanation or "
            "any candidate in the invalid pool. Before returning, strip and casefold every "
            "candidate and verify that all 64 are nonempty and pairwise distinct."
        ),
        user_prompt=json.dumps({
            "explanation": explanation,
            "claim": claim,
            "claim_class": claim_class,
            "subtype": subtype,
            "invalid_pool": result.value["candidates"],
            "validation_error": validation_error,
        }, ensure_ascii=False),
        schema=DECOY_SCHEMA,
        max_output_tokens=8192,
        cache_path=regeneration_cache_path,
    )
    replacement_folded = [
        str(value).strip().casefold() for value in regeneration.value["candidates"]
    ]
    if len(set(replacement_folded)) != 64 or any(not value for value in replacement_folded):
        raise ValueError("regenerated decoy pool contains a duplicate or empty candidate")
    return JSONResponse(value=regeneration.value, audit={
        "primary": result.audit,
        "primary_validation_error": validation_error,
        "regeneration_used": True,
        "regeneration": regeneration.audit,
    })


def _validate_judgments(result: JSONResponse) -> JSONResponse:
    rows = result.value["judgments"]
    if len(rows) != 64:
        raise ValueError("judge must return exactly 64 ordered judgments")
    for candidate_index, row in enumerate(rows):
        row["candidate_index"] = candidate_index
    return result


def judge_source_absence(
    client: TerraResponses,
    *,
    source_text: str,
    candidates: Sequence[str],
    cache_path: str | None = None,
) -> JSONResponse:
    if len(candidates) != 64:
        raise ValueError("absence judge requires exactly 64 candidates")
    system = (
        "For each candidate claim in input order, decide whether its specific content is absent from the source. "
        "PASS only if no source span states, paraphrases, or directly entails it. FAIL if stated, "
        "paraphrased, or directly entailed. UNCERTAIN if the evidence is genuinely ambiguous; "
        "uncertain candidates are rejected. Judge observable source support, not what a model might "
        "infer from world knowledge. Return exactly one judgment per candidate in the same order."
    )
    return _validate_judgments(client.request_json(
        task_name="nla_source_absence_v2",
        system_prompt=system,
        user_prompt=json.dumps({"source_text": source_text, "candidates": list(candidates)}, ensure_ascii=False),
        schema=JUDGMENT_SCHEMA,
        max_output_tokens=8192,
        cache_path=cache_path,
    ))


def judge_splice_naturalness(
    client: TerraResponses,
    *,
    original_explanation: str,
    edited_explanations: Sequence[str],
    cache_path: str | None = None,
) -> JSONResponse:
    if len(edited_explanations) != 64:
        raise ValueError("naturalness judge requires exactly 64 edited explanations")
    system = (
        "Compare each edited NLA explanation with the original, in input order. PASS only when the replacement "
        "preserves grammar, local fluency, discourse coherence, and structural well-formedness. "
        "Do not judge factual truth or activation support. FAIL for a substantial degradation. "
        "UNCERTAIN is rejected. Return exactly one judgment per edited explanation in the same order."
    )
    return _validate_judgments(client.request_json(
        task_name="nla_splice_naturalness_v2",
        system_prompt=system,
        user_prompt=json.dumps({
            "original_explanation": original_explanation,
            "edited_explanations": list(edited_explanations),
        }, ensure_ascii=False),
        schema=JUDGMENT_SCHEMA,
        max_output_tokens=8192,
        cache_path=cache_path,
    ))
