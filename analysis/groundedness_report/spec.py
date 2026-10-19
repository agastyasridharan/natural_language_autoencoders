"""Pure, CPU-only implementation of the attached groundedness procedure.

This module deliberately contains no model-loading or LLM-judge code.  It is the
frozen arithmetic contract shared by calibration, GPU measurement, rescoring,
tests, and reporting.

The supplied main text ends after the confabulation gate without assigning the
remaining branch.  The only fail-closed completion made here is that a
non-specific claim with weak theme evidence but activation support above
``kappa_support`` is ``unresolved``.  No claim is silently forced into a
substantive category that the manuscript does not define.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


REPORT_MAIN_SHA256 = "6cd2b69b9172c975d71cccf04e18527110386454dcc3dc4d326517433236046f"
REPORT_APPENDIX_SHA256 = "b6b0cedb25771ec0110cfca9f9e137a7838792dd48877cf8f7fc0870bf87211f"
# SHA-256 of the exact main-text bytes followed immediately by the exact
# appendix bytes.  The individual hashes above make the bundle reproducible.
REPORT_SHA256 = "ea7d2316e7a23ea1cf51eaf326627902cc150da0ba523b2538a1b8ac9b2adf97"
SCHEMA_VERSION = 3
PROTOCOL_SHA256 = "20a87044c5706a3de0d9deb098a740739c979a6d7a9ae21f010eb850cd329637"
RECONSTRUCTION_METRIC_AMENDMENT_SHA256 = (
    "3ec1a6fb1f6668f00d83caf48fd033d2d8e155b1021a25806a3587403c839057"
)
TEXT_MODEL_CORRECTION_AMENDMENT_SHA256 = (
    "fcd5d50f6f863337d9e9009b1129354afad1ce6dbff17b0510132d0b5d423966"
)
RECONSTRUCTION_ERROR_FORMULA = "squared_l2(unit(h), unit(AR(z)))"
VERDICTS = ("grounded", "entangled", "seeded", "confabulated", "unresolved")
_EPS = 1e-12


def _finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def unit(vector: Sequence[float]) -> np.ndarray:
    """Return the L2-unit vector; reject a non-finite or degenerate vector."""
    value = np.asarray(vector, dtype=np.float64)
    if value.ndim != 1 or not np.all(np.isfinite(value)):
        raise ValueError("vector must be a finite rank-1 array")
    norm = float(np.linalg.norm(value))
    if norm <= _EPS:
        raise ValueError("cannot normalize a zero-norm vector")
    return value / norm


def claim_direction(ar_with: Sequence[float], ar_without: Sequence[float]) -> np.ndarray:
    """Eq. 4-6: unit(unit(AR(z)) - unit(AR(z without C_i)))."""
    return unit(unit(ar_with) - unit(ar_without))


def project_out(h: Sequence[float], direction: Sequence[float]) -> np.ndarray:
    """Eq. 7: remove one unit direction without re-normalizing the result."""
    source = np.asarray(h, dtype=np.float64)
    if source.ndim != 1 or not np.all(np.isfinite(source)):
        raise ValueError("h must be a finite rank-1 array")
    delta = unit(direction)
    if source.shape != delta.shape:
        raise ValueError(f"shape mismatch: h={source.shape}, direction={delta.shape}")
    return source - float(source @ delta) * delta


def injection_vector(h: Sequence[float], injection_scale: float) -> np.ndarray:
    """Appendix contract: multiply by alpha; do not normalize inside this function."""
    value = np.asarray(h, dtype=np.float64)
    if value.ndim != 1 or not np.all(np.isfinite(value)):
        raise ValueError("injected activation must be a finite rank-1 array")
    if not _finite(injection_scale) or float(injection_scale) <= 0:
        raise ValueError("injection_scale must be positive and finite")
    return value * float(injection_scale)


def mean_logp(token_logp: Sequence[float], token_range: Sequence[int]) -> float:
    """Eq. 8: length-normalized mean log probability over [lo, hi)."""
    lo, hi = (int(token_range[0]), int(token_range[1]))
    values = np.asarray(token_logp, dtype=np.float64)
    if lo < 0 or hi <= lo or hi > values.size:
        raise ValueError(f"invalid token range [{lo}, {hi}) for {values.size} values")
    span = values[lo:hi]
    if not np.all(np.isfinite(span)):
        raise ValueError("claim token log probabilities must be finite")
    return float(span.mean())


def reconstruction_error(h: Sequence[float], ar_output: Sequence[float]) -> float:
    """Approved amendment: ||unit(h) - unit(AR(z))||_2^2."""
    source = np.asarray(h, dtype=np.float64)
    pred = np.asarray(ar_output, dtype=np.float64)
    if source.shape != pred.shape or source.ndim != 1:
        raise ValueError(f"shape mismatch: h={source.shape}, AR={pred.shape}")
    source_hat = unit(source)
    pred_hat = unit(pred)
    return float(np.square(source_hat - pred_hat).sum())


@dataclass(frozen=True)
class Calibration:
    """Thresholds that the report requires to be frozen on a calibration split."""

    model_key: str
    run_id: str
    split_sha256: str
    reference_sha256: str
    negative_control_sha256: str
    protocol_sha256: str
    reconstruction_metric_amendment_sha256: str
    text_model_correction_amendment_sha256: str
    k: int
    eps_overlap_ladder: tuple[float, ...]
    eps_base: float
    eps_reconstruction: float
    tau_decoy_support: float
    kappa_spec: float
    kappa_theme: float
    kappa_support: float
    rho_own: float = 1.5
    specificity_quantile: float = 0.95
    quantile_method: str = "linear"
    report_sha256: str = REPORT_SHA256
    schema_version: int = SCHEMA_VERSION

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "Calibration":
        required = (
            "model_key",
            "run_id",
            "split_sha256",
            "reference_sha256",
            "negative_control_sha256",
            "protocol_sha256",
            "reconstruction_metric_amendment_sha256",
            "text_model_correction_amendment_sha256",
            "k",
            "eps_overlap_ladder",
            "eps_base",
            "eps_reconstruction",
            "tau_decoy_support",
            "kappa_spec",
            "kappa_theme",
            "kappa_support",
        )
        missing = [key for key in required if value.get(key) is None]
        if missing:
            raise ValueError("calibration is not frozen; missing: " + ", ".join(missing))
        result = cls(
            model_key=str(value["model_key"]),
            run_id=str(value["run_id"]),
            split_sha256=str(value["split_sha256"]),
            reference_sha256=str(value["reference_sha256"]),
            negative_control_sha256=str(value["negative_control_sha256"]),
            protocol_sha256=str(value["protocol_sha256"]),
            reconstruction_metric_amendment_sha256=str(
                value["reconstruction_metric_amendment_sha256"]
            ),
            text_model_correction_amendment_sha256=str(
                value["text_model_correction_amendment_sha256"]
            ),
            k=int(value["k"]),
            eps_overlap_ladder=tuple(float(x) for x in value["eps_overlap_ladder"]),
            eps_base=float(value["eps_base"]),
            eps_reconstruction=float(value["eps_reconstruction"]),
            tau_decoy_support=float(value["tau_decoy_support"]),
            kappa_spec=float(value["kappa_spec"]),
            kappa_theme=float(value["kappa_theme"]),
            kappa_support=float(value["kappa_support"]),
            rho_own=float(value.get("rho_own", 1.5)),
            specificity_quantile=float(value.get("specificity_quantile", 0.95)),
            quantile_method=str(value.get("quantile_method", "linear")),
            report_sha256=str(value.get("report_sha256", REPORT_SHA256)),
            schema_version=int(value.get("schema_version", SCHEMA_VERSION)),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported calibration schema {self.schema_version}")
        if self.report_sha256 != REPORT_SHA256:
            raise ValueError("calibration was frozen against a different report")
        if not self.model_key or not self.run_id:
            raise ValueError("model_key and run_id must be non-empty")
        for name in (
            "split_sha256",
            "reference_sha256",
            "negative_control_sha256",
            "protocol_sha256",
            "reconstruction_metric_amendment_sha256",
            "text_model_correction_amendment_sha256",
        ):
            value = getattr(self, name)
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if self.protocol_sha256 != PROTOCOL_SHA256:
            raise ValueError("calibration was frozen against a different preregistration")
        if self.reconstruction_metric_amendment_sha256 != RECONSTRUCTION_METRIC_AMENDMENT_SHA256:
            raise ValueError("calibration was frozen against a different reconstruction metric")
        if self.text_model_correction_amendment_sha256 != TEXT_MODEL_CORRECTION_AMENDMENT_SHA256:
            raise ValueError("calibration was frozen against a different text model")
        if self.k < 1:
            raise ValueError("k must be positive")
        ladder = self.eps_overlap_ladder
        if not ladder or tuple(sorted(ladder)) != ladder or any(x < 0 or not _finite(x) for x in ladder):
            raise ValueError("eps_overlap_ladder must be finite, non-negative, and ascending")
        names = (
            "eps_base",
            "eps_reconstruction",
            "tau_decoy_support",
            "kappa_spec",
            "kappa_theme",
            "kappa_support",
            "rho_own",
        )
        for name in names:
            if not _finite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if self.eps_base < 0 or self.eps_reconstruction < 0 or self.rho_own <= 1:
            raise ValueError("matching tolerances must be non-negative and rho_own must exceed 1")
        if not (0 < self.specificity_quantile < 1):
            raise ValueError("specificity_quantile must lie strictly between zero and one")
        if self.specificity_quantile != 0.95 or self.quantile_method != "linear":
            raise ValueError("preregistration fixes Q0.95 to Hyndman-Fan type 7 / linear")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "report_sha256": self.report_sha256,
            "model_key": self.model_key,
            "run_id": self.run_id,
            "split_sha256": self.split_sha256,
            "reference_sha256": self.reference_sha256,
            "negative_control_sha256": self.negative_control_sha256,
            "protocol_sha256": self.protocol_sha256,
            "reconstruction_metric_amendment_sha256": (
                self.reconstruction_metric_amendment_sha256
            ),
            "text_model_correction_amendment_sha256": (
                self.text_model_correction_amendment_sha256
            ),
            "k": self.k,
            "eps_overlap_ladder": list(self.eps_overlap_ladder),
            "eps_base": self.eps_base,
            "eps_reconstruction": self.eps_reconstruction,
            "tau_decoy_support": self.tau_decoy_support,
            "kappa_spec": self.kappa_spec,
            "kappa_theme": self.kappa_theme,
            "kappa_support": self.kappa_support,
            "rho_own": self.rho_own,
            "specificity_quantile": self.specificity_quantile,
            "quantile_method": self.quantile_method,
        }


def _candidate_pre_overlap(real: Mapping[str, Any], candidate: Mapping[str, Any], c: Calibration) -> bool:
    if not candidate.get("text_pass", False):
        return False
    fields = (
        candidate.get("overlap"),
        candidate.get("reference_score"),
        candidate.get("reconstruction_error"),
        candidate.get("activation_support"),
        real.get("overlap"),
        real.get("reference_score"),
        real.get("reconstruction_error"),
    )
    if not all(_finite(x) for x in fields):
        return False
    if abs(float(candidate["reference_score"]) - float(real["reference_score"])) > c.eps_base:
        return False
    e_real = float(real["reconstruction_error"])
    if e_real < 0:
        return False
    if abs(float(candidate["reconstruction_error"]) - e_real) > c.eps_reconstruction * e_real:
        return False
    # The report rejects only unusually large POSITIVE activation support.
    if float(candidate["activation_support"]) > c.tau_decoy_support:
        return False
    return True


def select_decoys(
    real: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    calibration: Calibration,
) -> dict[str, Any]:
    """Apply report filters and deterministically select exactly k decoys.

    Matching is on ``||h.D*| - |h.D||``.  The overlap tolerance is widened only
    along the preregistered finite ladder.  If its final rung yields fewer than
    k candidates, the claim is unresolved; the magnitude filter is never
    disabled.  Among more than k survivors, generation order (candidate_index)
    chooses the first k and therefore cannot depend on D* or X readings.
    """
    calibration.validate()
    pre = [i for i, cand in enumerate(candidates) if _candidate_pre_overlap(real, cand, calibration)]
    real_overlap = abs(float(real["overlap"])) if _finite(real.get("overlap")) else float("nan")
    eps_used: float | None = None
    survivors: list[int] = []
    if np.isfinite(real_overlap):
        for eps in calibration.eps_overlap_ladder:
            survivors = [
                i
                for i in pre
                if abs(abs(float(candidates[i]["overlap"])) - real_overlap) <= eps * real_overlap
            ]
            if len(survivors) >= calibration.k:
                eps_used = float(eps)
                break
    if eps_used is None:
        return {
            "passed": False,
            "reason": f"fewer than k={calibration.k} valid matched decoys",
            "eps_overlap_used": None,
            "n_text_pass": sum(bool(x.get("text_pass")) for x in candidates),
            "n_pre_overlap": len(pre),
            "n_overlap": len(survivors),
            "selected_indices": [],
        }
    ranked = sorted(survivors, key=lambda i: (int(candidates[i].get("candidate_index", i)), i))
    selected = ranked[: calibration.k]
    return {
        "passed": True,
        "reason": "",
        "eps_overlap_used": eps_used,
        "n_text_pass": sum(bool(x.get("text_pass")) for x in candidates),
        "n_pre_overlap": len(pre),
        "n_overlap": len(survivors),
        "selected_indices": selected,
    }


def classify_claim(
    *,
    real: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    off_diagonal_effects: Iterable[float],
    calibration: Calibration,
) -> dict[str, Any]:
    """Apply the report's specificity, ownership, theme, and support gates."""
    selection = select_decoys(real, candidates, calibration)
    base: dict[str, Any] = {
        "verdict": "unresolved",
        "reason": selection["reason"],
        "selection": selection,
    }
    if not selection["passed"]:
        return base
    selected = [candidates[i] for i in selection["selected_indices"]]
    readings = [real.get("self_effect"), real.get("activation_support")]
    readings.extend(c.get("self_effect") for c in selected)
    readings.extend(c.get("cross_effect") for c in selected)
    if not all(_finite(x) for x in readings):
        return {**base, "reason": "non-finite or missing causal reading"}

    r_i = float(real["self_effect"])
    a_i = float(real["activation_support"])
    d = np.asarray([float(c["self_effect"]) for c in selected], dtype=np.float64)
    x = np.asarray([float(c["cross_effect"]) for c in selected], dtype=np.float64)
    q95 = float(np.quantile(d, calibration.specificity_quantile, method=calibration.quantile_method))
    specificity_bar = q95 + calibration.kappa_spec
    specificity = bool(r_i > specificity_bar)

    off = [float(v) for v in off_diagonal_effects if _finite(v)]
    ownership_competitor = max(0.0, max(off) if off else 0.0)
    ownership_bar = calibration.rho_own * ownership_competitor
    ownership = bool(r_i > ownership_bar)
    theme_effect = float(np.median(x))
    audit = {
        "r_i": r_i,
        "a_i": a_i,
        "q_decoy": q95,
        "specificity_bar": specificity_bar,
        "specificity_pass": specificity,
        "ownership_competitor": ownership_competitor,
        "ownership_bar": ownership_bar,
        "ownership_pass": ownership,
        "theme_effect": theme_effect,
        "grounding_estimate": r_i - float(np.median(d)),
    }
    if specificity:
        verdict = "grounded" if ownership else "entangled"
        return {**base, **audit, "verdict": verdict, "reason": ""}
    if theme_effect > calibration.kappa_theme:
        return {**base, **audit, "verdict": "seeded", "reason": ""}
    if a_i <= calibration.kappa_support:
        return {**base, **audit, "verdict": "confabulated", "reason": ""}
    return {
        **base,
        **audit,
        "reason": "report leaves weak-theme/high-activation-support branch unclassified",
    }


def verdict_summary(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Counts and percentages over all claims, including unresolved claims."""
    counts = {name: 0 for name in VERDICTS}
    total = 0
    for row in rows:
        verdict = str(row.get("verdict", "unresolved"))
        if verdict not in counts:
            verdict = "unresolved"
        counts[verdict] += 1
        total += 1
    percentages = {name: (100.0 * value / total if total else 0.0) for name, value in counts.items()}
    resolved = total - counts["unresolved"]
    resolved_percentages = {
        name: (100.0 * counts[name] / resolved if resolved else 0.0)
        for name in VERDICTS
        if name != "unresolved"
    }
    return {
        "n_claims": total,
        "n_resolved": resolved,
        "counts": counts,
        "percentages_all_claims": percentages,
        "percentages_resolved_only": resolved_percentages,
    }
