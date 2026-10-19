"""CPU workflow helpers: document split, calibration audit, and summaries."""
from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .io import atomic_json, read_jsonl
from .sampling import sample_claims
from .spec import (
    PROTOCOL_SHA256,
    TEXT_MODEL_CORRECTION_AMENDMENT_SHA256,
    VERDICTS,
    verdict_summary,
)


def make_document_split(
    records_path: str | Path,
    output_path: str | Path,
    *,
    calibration_fraction: float,
    seed: int,
) -> dict[str, Any]:
    """Make a stable document-level calibration/evaluation partition."""
    if not 0 < calibration_fraction < 1:
        raise ValueError("calibration_fraction must lie strictly between zero and one")
    records = list(read_jsonl(records_path))
    by_doc: dict[str, list[int]] = defaultdict(list)
    for row in records:
        if row.get("doc_id") in (None, "") or row.get("idx") is None:
            raise ValueError("every record must contain non-empty doc_id and idx")
        by_doc[str(row["doc_id"])].append(int(row["idx"]))
    if len(by_doc) < 2:
        raise ValueError("at least two distinct documents are required")

    def key(doc_id: str) -> str:
        return hashlib.sha256(f"{seed}\0{doc_id}".encode()).hexdigest()

    docs = sorted(by_doc, key=key)
    n_cal = min(len(docs) - 1, max(1, round(len(docs) * calibration_fraction)))
    cal_docs = set(docs[:n_cal])
    calibration_indices = sorted(i for doc in cal_docs for i in by_doc[doc])
    evaluation_indices = sorted(i for doc in docs[n_cal:] for i in by_doc[doc])
    result = {
        "schema_version": 1,
        "split_unit": "doc_id",
        "seed": int(seed),
        "calibration_fraction_requested": float(calibration_fraction),
        "n_documents": len(docs),
        "calibration_doc_ids": sorted(cal_docs),
        "evaluation_doc_ids": sorted(set(docs) - cal_docs),
        "calibration_indices": calibration_indices,
        "evaluation_indices": evaluation_indices,
    }
    validate_split(result)
    atomic_json(output_path, result)
    return result


def make_evaluation_split(
    records_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Place every record in evaluation for an OOD-only run.

    The thresholds and reference activation remain those frozen on the
    separate FineWeb calibration run.  An explicit evaluation-only split
    avoids relabeling any OOD activation as calibration merely to satisfy a
    file-format constraint.
    """
    records = list(read_jsonl(records_path))
    if not records:
        raise ValueError("evaluation records are empty")
    by_doc: dict[str, list[int]] = defaultdict(list)
    for row in records:
        if row.get("doc_id") in (None, "") or row.get("idx") is None:
            raise ValueError("every record must contain non-empty doc_id and idx")
        by_doc[str(row["doc_id"])].append(int(row["idx"]))
    indices = sorted(index for values in by_doc.values() for index in values)
    if len(indices) != len(set(indices)):
        raise ValueError("evaluation records contain duplicate activation indices")
    result = {
        "schema_version": 1,
        "split_kind": "evaluation_only",
        "split_unit": "doc_id",
        "calibration_source": "external_frozen_fineweb_calibration",
        "n_documents": len(by_doc),
        "calibration_doc_ids": [],
        "evaluation_doc_ids": sorted(by_doc),
        "calibration_indices": [],
        "evaluation_indices": indices,
    }
    validate_split(result)
    atomic_json(output_path, result)
    return result


def validate_split(value: Mapping[str, Any]) -> None:
    cal_docs = set(str(x) for x in value.get("calibration_doc_ids", []))
    eval_docs = set(str(x) for x in value.get("evaluation_doc_ids", []))
    cal_idx = set(int(x) for x in value.get("calibration_indices", []))
    eval_idx = set(int(x) for x in value.get("evaluation_indices", []))
    evaluation_only = value.get("split_kind") == "evaluation_only"
    if evaluation_only:
        if cal_docs or cal_idx or not eval_docs or not eval_idx:
            raise ValueError(
                "evaluation-only split requires an empty calibration partition "
                "and a non-empty evaluation partition"
            )
    elif not cal_docs or not eval_docs or not cal_idx or not eval_idx:
        raise ValueError("both split partitions must be non-empty")
    if cal_docs & eval_docs:
        raise ValueError("document leakage between calibration and evaluation")
    if cal_idx & eval_idx:
        raise ValueError("activation leakage between calibration and evaluation")


def make_claim_sample(
    records_path: str | Path,
    split_path: str | Path,
    output_path: str | Path,
    *,
    seed: int,
    maximum: int = 3,
) -> dict[str, Any]:
    """Freeze the preregistered per-activation uniform claim sample."""
    if maximum != 3:
        raise ValueError("preregistration fixes at most three claims per activation")
    import json

    split = json.loads(Path(split_path).read_text(encoding="utf-8"))
    validate_split(split)
    partitions = {
        int(idx): partition
        for partition in ("calibration", "evaluation")
        for idx in split[f"{partition}_indices"]
    }
    rows: list[dict[str, Any]] = []
    seen_indices: set[int] = set()
    for record in read_jsonl(records_path):
        idx = int(record["idx"])
        if idx not in partitions or record.get("status") != "ok":
            continue
        if record.get(
            "text_model_correction_amendment_sha256"
        ) != TEXT_MODEL_CORRECTION_AMENDMENT_SHA256:
            raise ValueError(f"record {idx} predates the Terra correction")
        seen_indices.add(idx)
        activation_id = f"{record.get('doc_id')}:{idx}"
        for sampled in sample_claims(
            list(record.get("snippet_results") or []),
            activation_id=activation_id,
            seed=seed,
            maximum=maximum,
        ):
            rows.append({
                "idx": idx,
                "doc_id": str(record.get("doc_id")),
                "partition": partitions[idx],
                **sampled,
            })
    absent = sorted(set(partitions) - seen_indices)
    if absent:
        raise ValueError(f"split contains missing or non-ok records: {absent[:10]}")
    rows.sort(key=lambda row: (row["idx"], row["claim_i"]))
    result = {
        "schema_version": 1,
        "protocol_sha256": PROTOCOL_SHA256,
        "text_model_correction_amendment_sha256": (
            TEXT_MODEL_CORRECTION_AMENDMENT_SHA256
        ),
        "seed": int(seed),
        "maximum_claims_per_activation": maximum,
        "method": "uniform_without_replacement_keyed_by_seed_and_activation_id",
        "n_activations": len(seen_indices),
        "n_sampled_claims": len(rows),
        "rows": rows,
    }
    atomic_json(output_path, result)
    return result


def _quantiles(values: Sequence[float]) -> dict[str, Any]:
    finite = np.asarray([float(x) for x in values if np.isfinite(float(x))], dtype=np.float64)
    if finite.size == 0:
        return {"n": 0, "quantiles": {}}
    qs = (0.0, 0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
    return {
        "n": int(finite.size),
        "quantiles": {f"q{int(q * 100):02d}": float(np.quantile(finite, q)) for q in qs},
    }


def calibration_audit(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Describe calibration distributions without inventing threshold choices."""
    overlap_mismatch: list[float] = []
    baseline_mismatch: list[float] = []
    reconstruction_relative_mismatch: list[float] = []
    decoy_support: list[float] = []
    real_support: list[float] = []
    real_self: list[float] = []
    decoy_self: list[float] = []
    cross_effect: list[float] = []
    n_rows = 0
    for row in rows:
        n_rows += 1
        real = row.get("real") or {}
        if _isfinite(real.get("activation_support")):
            real_support.append(float(real["activation_support"]))
        if _isfinite(real.get("self_effect")):
            real_self.append(float(real["self_effect"]))
        for candidate in row.get("candidates") or []:
            if not candidate.get("text_pass"):
                continue
            if _isfinite(real.get("overlap")) and _isfinite(candidate.get("overlap")):
                denom = abs(float(real["overlap"]))
                if denom > 0:
                    overlap_mismatch.append(abs(abs(float(candidate["overlap"])) - denom) / denom)
            if _isfinite(real.get("reference_score")) and _isfinite(candidate.get("reference_score")):
                baseline_mismatch.append(abs(float(candidate["reference_score"]) - float(real["reference_score"])))
            if _isfinite(real.get("reconstruction_error")) and _isfinite(candidate.get("reconstruction_error")):
                denom = float(real["reconstruction_error"])
                if denom > 0:
                    reconstruction_relative_mismatch.append(
                        abs(float(candidate["reconstruction_error"]) - denom) / denom
                    )
            if _isfinite(candidate.get("activation_support")):
                decoy_support.append(float(candidate["activation_support"]))
            if _isfinite(candidate.get("self_effect")):
                decoy_self.append(float(candidate["self_effect"]))
            if _isfinite(candidate.get("cross_effect")):
                cross_effect.append(float(candidate["cross_effect"]))
    return {
        "n_claims": n_rows,
        "note": "Descriptive only: the attached report does not specify threshold-selection rules.",
        "relative_absolute_overlap_mismatch": _quantiles(overlap_mismatch),
        "absolute_reference_score_mismatch": _quantiles(baseline_mismatch),
        "relative_reconstruction_error_mismatch": _quantiles(reconstruction_relative_mismatch),
        "decoy_activation_support": _quantiles(decoy_support),
        "real_activation_support": _quantiles(real_support),
        "real_self_effect": _quantiles(real_self),
        "decoy_self_effect": _quantiles(decoy_self),
        "decoy_cross_effect": _quantiles(cross_effect),
    }


def _isfinite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def stratified_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    overall = verdict_summary(rows)
    by_type: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_subtype: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    reasons: dict[str, int] = defaultdict(int)
    for row in rows:
        by_type[str(row.get("claim_type") or "unknown")].append(row)
        by_subtype[str(row.get("claim_subtype") or "unknown")].append(row)
        if row.get("verdict") == "unresolved":
            reasons[str(row.get("reason") or "unspecified")] += 1
    return {
        **overall,
        "by_claim_type": {key: verdict_summary(value) for key, value in sorted(by_type.items())},
        "by_claim_subtype": {key: verdict_summary(value) for key, value in sorted(by_subtype.items())},
        "unresolved_reasons": dict(sorted(reasons.items(), key=lambda item: (-item[1], item[0]))),
        "verdict_order": list(VERDICTS),
    }
