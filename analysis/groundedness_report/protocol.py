"""Load and enforce the frozen preregistration."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from .spec import (
    PROTOCOL_SHA256,
    RECONSTRUCTION_ERROR_FORMULA,
    RECONSTRUCTION_METRIC_AMENDMENT_SHA256,
    TEXT_MODEL_CORRECTION_AMENDMENT_SHA256,
)


DEFAULT_PATH = Path(__file__).with_name("preregistration.yaml")
REGISTRY_PATH = Path(__file__).with_name("evaluation_registry.yaml")
RECONSTRUCTION_METRIC_AMENDMENT_PATH = (
    Path(__file__).with_name("audits") / "v7" / "reconstruction_metric_amendment.json"
)
TEXT_MODEL_CORRECTION_AMENDMENT_PATH = (
    Path(__file__).with_name("audits") / "v7" / "text_model_correction_amendment.json"
)


def protocol_sha256(path: str | Path = DEFAULT_PATH) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_protocol(path: str | Path = DEFAULT_PATH) -> dict[str, Any]:
    source = Path(path)
    digest = protocol_sha256(source)
    if digest != PROTOCOL_SHA256:
        raise ValueError(
            f"preregistration hash drift: expected {PROTOCOL_SHA256}, observed {digest}"
        )
    value = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("preregistration must be a YAML mapping")
    validate_protocol(value)
    load_reconstruction_metric_amendment()
    load_text_model_correction_amendment()
    return value


def load_reconstruction_metric_amendment(
    path: str | Path = RECONSTRUCTION_METRIC_AMENDMENT_PATH,
) -> dict[str, Any]:
    source = Path(path)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != RECONSTRUCTION_METRIC_AMENDMENT_SHA256:
        raise ValueError(
            "reconstruction-metric amendment hash drift: "
            f"expected {RECONSTRUCTION_METRIC_AMENDMENT_SHA256}, observed {digest}"
        )
    value = json.loads(source.read_text(encoding="utf-8"))
    if value.get("protocol_sha256") != PROTOCOL_SHA256:
        raise ValueError("reconstruction-metric amendment names a different preregistration")
    if value.get("active_operational_formula") != RECONSTRUCTION_ERROR_FORMULA:
        raise ValueError("reconstruction-metric amendment formula drift")
    return value


def load_text_model_correction_amendment(
    path: str | Path = TEXT_MODEL_CORRECTION_AMENDMENT_PATH,
) -> dict[str, Any]:
    source = Path(path)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != TEXT_MODEL_CORRECTION_AMENDMENT_SHA256:
        raise ValueError(
            "text-model correction amendment hash drift: "
            f"expected {TEXT_MODEL_CORRECTION_AMENDMENT_SHA256}, observed {digest}"
        )
    value = json.loads(source.read_text(encoding="utf-8"))
    if value.get("protocol_sha256") != PROTOCOL_SHA256:
        raise ValueError("text-model amendment names a different preregistration")
    if value.get("active_operational_model") != "gpt-5.6-terra":
        raise ValueError("text-model amendment model drift")
    if value.get("api_key_environment_variable") != "OPENAI_API_KEY_2":
        raise ValueError("text-model amendment key-name drift")
    if value.get("allow_fallback_key") or value.get("allow_fallback_model"):
        raise ValueError("text-model amendment permits a forbidden fallback")
    return value


def validate_protocol(value: Mapping[str, Any]) -> None:
    if value.get("status") != "frozen_before_outcome_generation":
        raise ValueError("protocol is not frozen")
    if value.get("protocol_name") != "qwen_nla_groundedness_replication_v7":
        raise ValueError("unexpected protocol name")
    decoys = value.get("decoys") or {}
    if int(decoys.get("k", -1)) != 20:
        raise ValueError("protocol fixes k=20")
    if int(decoys.get("candidates_generated_per_scored_claim", -1)) != 64:
        raise ValueError("protocol fixes 64 candidate decoys")
    if tuple(float(x) for x in decoys.get("relative_overlap_ladder") or ()) != (
        0.10,
        0.20,
        0.35,
        0.50,
        0.75,
        1.00,
    ):
        raise ValueError("overlap ladder differs from the frozen protocol")
    quantile = decoys.get("quantile") or {}
    if float(quantile.get("probability", -1)) != 0.95 or quantile.get("convention") != (
        "Hyndman_Fan_type_7_linear"
    ):
        raise ValueError("quantile convention differs from the frozen protocol")
    llm = value.get("openai_text_pipeline") or {}
    # The immutable base file records the erroneous Luna substitution. The
    # separately hashed pre-outcome correction is loaded by ``load_protocol``
    # and defines Terra as the only operational model.
    if llm.get("model") != "gpt-5.6-luna":
        raise ValueError("base text-pipeline provenance differs from V7")
    if llm.get("api_key_environment_variable") != "OPENAI_API_KEY_2":
        raise ValueError("text pipeline key name differs from the frozen protocol")
    if llm.get("allow_fallback_key") or llm.get("allow_fallback_model"):
        raise ValueError("model and API key fallback are forbidden")


def load_evaluation_registry(path: str | Path = REGISTRY_PATH) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("evaluation registry must be a YAML mapping")
    if value.get("protocol_sha256") != PROTOCOL_SHA256:
        raise ValueError("evaluation registry preregistration hash mismatch")
    families = value.get("families") or []
    identifiers = [str(row.get("id") or "") for row in families]
    if len(families) != 16 or len(set(identifiers)) != 16 or "" in identifiers:
        raise ValueError("evaluation registry must contain 16 uniquely named families")
    groups = [str(row.get("group")) for row in families]
    if groups.count("alignment") != 7 or groups.count("capability") != 9:
        raise ValueError("evaluation registry must contain seven alignment and nine capability families")
    if any(not row.get("fidelity") or not row.get("environment") for row in families):
        raise ValueError("every family must state fidelity and environment")
    return value
