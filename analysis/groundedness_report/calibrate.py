"""Freeze all report thresholds from FineWeb and deranged activations only."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .spec import (
    Calibration,
    PROTOCOL_SHA256,
    RECONSTRUCTION_ERROR_FORMULA,
    RECONSTRUCTION_METRIC_AMENDMENT_SHA256,
    REPORT_SHA256,
    SCHEMA_VERSION,
    TEXT_MODEL_CORRECTION_AMENDMENT_SHA256,
    select_decoys,
)


EPS_BASE_GRID = (0.25, 0.50, 0.75, 1.00, 1.50, 2.00)
EPS_RECONSTRUCTION_GRID = (0.10, 0.20, 0.35, 0.50, 0.75, 1.00)
OVERLAP_LADDER = (0.10, 0.20, 0.35, 0.50, 0.75, 1.00)


def _finite(values: Sequence[Any], name: str) -> np.ndarray:
    result = np.asarray([float(value) for value in values if np.isfinite(float(value))], dtype=np.float64)
    if result.size == 0:
        raise ValueError(f"no finite values for {name}")
    return result


def q95(values: Sequence[Any], name: str) -> float:
    return float(np.quantile(_finite(values, name), 0.95, method="linear"))


def _calibration(
    provenance: Mapping[str, str],
    *,
    eps_base: float,
    eps_reconstruction: float,
    tau_decoy_support: float,
    kappa_spec: float = 0.0,
    kappa_theme: float = 0.0,
    kappa_support: float = 0.0,
) -> Calibration:
    return Calibration.from_mapping({
        "schema_version": SCHEMA_VERSION,
        "report_sha256": REPORT_SHA256,
        "protocol_sha256": PROTOCOL_SHA256,
        "reconstruction_metric_amendment_sha256": RECONSTRUCTION_METRIC_AMENDMENT_SHA256,
        "text_model_correction_amendment_sha256": TEXT_MODEL_CORRECTION_AMENDMENT_SHA256,
        "model_key": provenance["model_key"],
        "run_id": provenance["run_id"],
        "split_sha256": provenance["split_sha256"],
        "reference_sha256": provenance["reference_sha256"],
        "negative_control_sha256": provenance["negative_control_sha256"],
        "k": 20,
        "eps_overlap_ladder": list(OVERLAP_LADDER),
        "eps_base": eps_base,
        "eps_reconstruction": eps_reconstruction,
        "tau_decoy_support": tau_decoy_support,
        "kappa_spec": kappa_spec,
        "kappa_theme": kappa_theme,
        "kappa_support": kappa_support,
        "rho_own": 1.5,
        "specificity_quantile": 0.95,
        "quantile_method": "linear",
    })


def freeze_thresholds(
    rows: Sequence[Mapping[str, Any]],
    *,
    provenance: Mapping[str, str],
    minimum_null_claims: int = 100,
) -> tuple[Calibration, dict[str, Any]]:
    """Apply the fully preregistered threshold-selection algorithm.

    Each input row must contain the ordinary FineWeb calibration measurement in
    ``real``/``candidates`` and the same measurement under its deranged source
    activation in ``negative_control``.
    """
    if not rows:
        raise ValueError("calibration rows are empty")
    incompatible_metric_rows = [
        index
        for index, row in enumerate(rows)
        if row.get("reconstruction_metric_amendment_sha256")
        != RECONSTRUCTION_METRIC_AMENDMENT_SHA256
        or row.get("reconstruction_error_formula") != RECONSTRUCTION_ERROR_FORMULA
        or row.get("text_model_correction_amendment_sha256")
        != TEXT_MODEL_CORRECTION_AMENDMENT_SHA256
    ]
    if incompatible_metric_rows:
        raise ValueError(
            "calibration rows use an absent or incompatible reconstruction metric; "
            f"first row indices: {incompatible_metric_rows[:5]}"
        )
    required_provenance = {
        "model_key", "run_id", "split_sha256", "reference_sha256", "negative_control_sha256"
    }
    if set(provenance) != required_provenance:
        raise ValueError(f"provenance keys must be exactly {sorted(required_provenance)}")
    negative_decoy_support = [
        candidate.get("activation_support")
        for row in rows
        for candidate in (row.get("negative_control") or {}).get("candidates") or []
        if candidate.get("text_pass") and candidate.get("activation_support") is not None
    ]
    tau = q95(negative_decoy_support, "negative decoy activation support")

    feasibility: list[dict[str, Any]] = []
    for base_rank, eps_base in enumerate(EPS_BASE_GRID):
        for rec_rank, eps_reconstruction in enumerate(EPS_RECONSTRUCTION_GRID):
            provisional = _calibration(
                provenance,
                eps_base=eps_base,
                eps_reconstruction=eps_reconstruction,
                tau_decoy_support=tau,
            )
            passed = sum(
                select_decoys(row["real"], row.get("candidates") or [], provisional)["passed"]
                for row in rows
            )
            feasibility.append({
                "eps_base": eps_base,
                "eps_reconstruction": eps_reconstruction,
                "base_rank": base_rank,
                "reconstruction_rank": rec_rank,
                "rank_sum": base_rank + rec_rank,
                "n_claims": len(rows),
                "n_resolvable": passed,
                "resolvability": passed / len(rows),
            })
    feasible = [row for row in feasibility if row["resolvability"] >= 0.90]
    if feasible:
        choice = min(
            feasible,
            key=lambda row: (
                row["rank_sum"], row["eps_base"], row["eps_reconstruction"]
            ),
        )
        target_met = True
    else:
        choice = next(
            row for row in feasibility
            if row["eps_base"] == EPS_BASE_GRID[-1]
            and row["eps_reconstruction"] == EPS_RECONSTRUCTION_GRID[-1]
        )
        target_met = False
    provisional = _calibration(
        provenance,
        eps_base=float(choice["eps_base"]),
        eps_reconstruction=float(choice["eps_reconstruction"]),
        tau_decoy_support=tau,
    )

    specificity_excess: list[float] = []
    theme_effects: list[float] = []
    real_support: list[float] = []
    negative_resolved = 0
    for row in rows:
        negative = row.get("negative_control") or {}
        real = negative.get("real") or {}
        if real.get("activation_support") is not None:
            real_support.append(float(real["activation_support"]))
        selection = select_decoys(real, negative.get("candidates") or [], provisional)
        if not selection["passed"]:
            continue
        selected = [negative["candidates"][index] for index in selection["selected_indices"]]
        readings = [real.get("self_effect")]
        readings.extend(candidate.get("self_effect") for candidate in selected)
        readings.extend(candidate.get("cross_effect") for candidate in selected)
        if not all(value is not None and np.isfinite(float(value)) for value in readings):
            continue
        d = np.asarray([float(candidate["self_effect"]) for candidate in selected])
        x = np.asarray([float(candidate["cross_effect"]) for candidate in selected])
        specificity_excess.append(
            float(real["self_effect"]) - float(np.quantile(d, 0.95, method="linear"))
        )
        theme_effects.append(float(np.median(x)))
        negative_resolved += 1
    if negative_resolved < minimum_null_claims:
        raise ValueError(
            f"only {negative_resolved} complete shuffled-null claims; require {minimum_null_claims}"
        )
    result = _calibration(
        provenance,
        eps_base=float(choice["eps_base"]),
        eps_reconstruction=float(choice["eps_reconstruction"]),
        tau_decoy_support=tau,
        kappa_spec=q95(specificity_excess, "shuffled specificity excess"),
        kappa_theme=q95(theme_effects, "shuffled theme effect"),
        kappa_support=q95(real_support, "shuffled real-claim activation support"),
    )
    audit = {
        "schema_version": SCHEMA_VERSION,
        "protocol_sha256": PROTOCOL_SHA256,
        "reconstruction_metric_amendment_sha256": RECONSTRUCTION_METRIC_AMENDMENT_SHA256,
        "text_model_correction_amendment_sha256": TEXT_MODEL_CORRECTION_AMENDMENT_SHA256,
        "reconstruction_error_formula": RECONSTRUCTION_ERROR_FORMULA,
        "report_sha256": REPORT_SHA256,
        "data_scope": "FineWeb calibration plus fixed deranged activations only",
        "n_calibration_claims": len(rows),
        "n_negative_claims_complete": negative_resolved,
        "n_negative_decoy_support_values": len(negative_decoy_support),
        "quantile_convention": "Hyndman-Fan type 7 / numpy method=linear",
        "nuisance_grid": feasibility,
        "nuisance_choice": choice,
        "resolvability_target": 0.90,
        "resolvability_target_met": target_met,
        "calibration": result.as_dict(),
    }
    return result, audit
