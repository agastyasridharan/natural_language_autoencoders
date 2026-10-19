"""Weighted category estimates with document/transcript-clustered intervals."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np

from .spec import VERDICTS


def _weight(row: Mapping[str, Any]) -> float:
    value = float(row.get("analysis_weight", 1.0))
    if not np.isfinite(value) or value <= 0:
        raise ValueError("analysis weights must be finite and positive")
    return value


def weighted_percentages(
    rows: Sequence[Mapping[str, Any]],
    *,
    multipliers: Mapping[str, int] | None = None,
    cluster_key: str = "cluster_id",
) -> dict[str, float]:
    totals = {category: 0.0 for category in VERDICTS}
    denominator = 0.0
    for row in rows:
        multiplier = 1 if multipliers is None else int(multipliers.get(str(row[cluster_key]), 0))
        if multiplier <= 0:
            continue
        weight = _weight(row) * multiplier
        verdict = str(row.get("verdict", "unresolved"))
        if verdict not in totals:
            verdict = "unresolved"
        totals[verdict] += weight
        denominator += weight
    if denominator <= 0:
        raise ValueError("weighted denominator is zero")
    return {category: 100.0 * value / denominator for category, value in totals.items()}


def clustered_intervals(
    rows: Sequence[Mapping[str, Any]],
    *,
    cluster_key: str,
    replicates: int = 10_000,
    seed: int = 20_260_827,
) -> dict[str, Any]:
    if replicates < 1:
        raise ValueError("replicates must be positive")
    cluster_ids = sorted({str(row.get(cluster_key) or "") for row in rows})
    if "" in cluster_ids or len(cluster_ids) < 2:
        raise ValueError("at least two non-empty clusters are required")
    normalized = [{**row, "cluster_id": str(row[cluster_key])} for row in rows]
    point = weighted_percentages(normalized)
    rng = np.random.default_rng(seed)
    draws = {category: np.empty(replicates, dtype=np.float64) for category in VERDICTS}
    for replicate in range(replicates):
        sampled = rng.choice(cluster_ids, size=len(cluster_ids), replace=True)
        counts: dict[str, int] = defaultdict(int)
        for cluster_id in sampled:
            counts[str(cluster_id)] += 1
        estimate = weighted_percentages(normalized, multipliers=counts)
        for category in VERDICTS:
            draws[category][replicate] = estimate[category]
    return {
        "method": "percentile_cluster_bootstrap",
        "confidence_level": 0.95,
        "replicates": replicates,
        "seed": seed,
        "cluster_key": cluster_key,
        "n_clusters": len(cluster_ids),
        "n_sampled_claims": len(rows),
        "point_percentages": point,
        "intervals": {
            category: {
                "low": float(np.quantile(values, 0.025, method="linear")),
                "high": float(np.quantile(values, 0.975, method="linear")),
            }
            for category, values in draws.items()
        },
    }


def ood_macro_intervals(
    rows: Sequence[Mapping[str, Any]],
    *,
    family_key: str = "evaluation_family",
    cluster_key: str = "trajectory_id",
    replicates: int = 10_000,
    seed: int = 20_260_827,
) -> dict[str, Any]:
    """Equal-family macro estimate with within-family transcript resampling."""
    by_family: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        family = str(row.get(family_key) or "")
        if not family:
            raise ValueError(f"missing {family_key}")
        by_family[family].append(row)
    if len(by_family) != 16:
        raise ValueError(f"preregistration requires 16 OOD families, observed {len(by_family)}")
    rng = np.random.default_rng(seed)
    point_by_family = {
        family: weighted_percentages(
            [{**row, "cluster_id": str(row[cluster_key])} for row in family_rows]
        )
        for family, family_rows in by_family.items()
    }
    point = {
        category: float(np.mean([values[category] for values in point_by_family.values()]))
        for category in VERDICTS
    }
    draws = {category: np.empty(replicates, dtype=np.float64) for category in VERDICTS}
    prepared: dict[str, tuple[list[dict[str, Any]], list[str]]] = {}
    for family, family_rows in by_family.items():
        normalized = [{**row, "cluster_id": str(row[cluster_key])} for row in family_rows]
        clusters = sorted({row["cluster_id"] for row in normalized})
        if len(clusters) < 2 or "" in clusters:
            raise ValueError(f"family {family} has fewer than two valid transcript clusters")
        prepared[family] = (normalized, clusters)
    for replicate in range(replicates):
        family_estimates = []
        for normalized, clusters in prepared.values():
            sampled = rng.choice(clusters, size=len(clusters), replace=True)
            counts: dict[str, int] = defaultdict(int)
            for cluster_id in sampled:
                counts[str(cluster_id)] += 1
            family_estimates.append(weighted_percentages(normalized, multipliers=counts))
        for category in VERDICTS:
            draws[category][replicate] = float(
                np.mean([estimate[category] for estimate in family_estimates])
            )
    return {
        "method": "equal_family_macro_percentile_cluster_bootstrap",
        "confidence_level": 0.95,
        "replicates": replicates,
        "seed": seed,
        "n_families": len(by_family),
        "point_percentages": point,
        "by_family_point_percentages": dict(sorted(point_by_family.items())),
        "intervals": {
            category: {
                "low": float(np.quantile(values, 0.025, method="linear")),
                "high": float(np.quantile(values, 0.975, method="linear")),
            }
            for category, values in draws.items()
        },
    }
