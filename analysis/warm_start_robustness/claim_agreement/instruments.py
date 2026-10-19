"""Versioned LLM instrument prompts and strict JSON schemas."""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence


SEGMENTATION_INSTRUMENT_VERSION = "lossless-atomic-v1"
EQUIVALENCE_INSTRUMENT_VERSION = "strict-mutual-entailment-v1"


SEGMENTATION_INSTRUCTIONS = r"""
You are a measurement instrument that decomposes one NLA explanation into atomic semantic claims.

Return an ORDERED LOSSLESS TILING of the entire explanation. Concatenating every unit's source_text,
in order, must reproduce the input explanation character-for-character, including every whitespace,
newline, punctuation mark, quote mark, and formatting character. Copy source_text verbatim. Do not
normalize it. Every input character must occur exactly once across the ordered units.

Make every substantive claim unit atomic: it must express only one proposition. Split coordinated or
compound statements into separate units whenever each part could independently be true or false. A
unit may rely on a subject from an earlier unit; write its normalized claim as a standalone proposition
with the subject resolved. Preserve negation, uncertainty, modality, temporal relations, quantities,
and named-entity identity in the normalized claim. Repeated claims remain repeated occurrences.

Mark punctuation, headings, discourse scaffolding, and connective text that asserts no semantic
content as nonclaim. A claim unit receives exactly one specificity label:

- broad_theme: a high-level claim about overall topic, domain, genre, task, register, tone, style, or
  evaluation framing. It must remain broad and cannot identify a particular entity, local event,
  concrete action, exact relationship, number, code behavior, answer, or predicted continuation.
- entity_detail: every more specific proposition, including named or described entities, attributes,
  events, actions, relations, quantities, quotations, local textual facts, answers, and predictions.

Do not assess whether a claim is plausible, true, grounded in the activation, or shared with another
explanation. Segmentation and specificity labeling only.

For claim units, claim is a concise standalone normalized proposition and specificity is broad_theme
or entity_detail. For nonclaim units, claim is exactly the empty string and specificity is
not_applicable. Return only the required structured output.
""".strip()


SEGMENTATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "units": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source_text": {"type": "string"},
                    "kind": {"type": "string", "enum": ["claim", "nonclaim"]},
                    "claim": {"type": "string"},
                    "specificity": {
                        "type": "string",
                        "enum": ["broad_theme", "entity_detail", "not_applicable"],
                    },
                },
                "required": ["source_text", "kind", "claim", "specificity"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["units"],
    "additionalProperties": False,
}


EQUIVALENCE_INSTRUCTIONS = r"""
You are a conservative semantic-equivalence measurement instrument. Compare every LEFT claim with
every RIGHT claim that has the same specificity label. Return only pairs that express the SAME
underlying proposition at the SAME specificity.

Equivalence requires mutual entailment in context, allowing paraphrase, synonyms, grammatical voice,
or minor non-semantic wording differences. The following are NOT equivalent:

- merely sharing a topic, domain, sentiment, style, or entity;
- one claim being broader, narrower, more general, or more specific;
- different named entities, quantities, times, locations, actions, relations, answers, or predictions;
- compatible claims for which neither entails the other;
- negation, material modality, certainty, or temporal differences;
- a broad_theme claim matched to an entity_detail claim.

When uncertain, do not match. Review all eligible Cartesian pairs. A claim may match more than one
claim on the other side only when those claim occurrences are genuinely semantically equivalent; do
not impose one-to-one matching. Return a concise rationale for every equivalence edge. Do not return
non-equivalent pairs. Set all_pairs_reviewed=true only after considering every eligible pair. Do not
judge truth, plausibility, or activation-grounding. Return only the required structured output.
""".strip()


EQUIVALENCE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "all_pairs_reviewed": {"type": "boolean"},
        "matches": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "left_id": {"type": "string"},
                    "right_id": {"type": "string"},
                    "rationale": {"type": "string"},
                },
                "required": ["left_id", "right_id", "rationale"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["all_pairs_reviewed", "matches"],
    "additionalProperties": False,
}


def segmentation_input(explanation: str) -> str:
    # JSON encoding makes leading/trailing whitespace unambiguous to the instrument.
    return "EXPLANATION_JSON_STRING:\n" + json.dumps(explanation, ensure_ascii=False)


def _claim_projection(claim: Mapping[str, Any]) -> dict[str, str]:
    return {
        "claim_id": str(claim["claim_id"]),
        "specificity": str(claim["specificity"]),
        "claim": str(claim["claim"]),
    }


def equivalence_input(
    left_claims: Sequence[Mapping[str, Any]], right_claims: Sequence[Mapping[str, Any]]
) -> str:
    payload = {
        "left_claims": [_claim_projection(claim) for claim in left_claims],
        "right_claims": [_claim_projection(claim) for claim in right_claims],
    }
    return "CLAIM_SETS:\n" + json.dumps(payload, ensure_ascii=False, sort_keys=True)
