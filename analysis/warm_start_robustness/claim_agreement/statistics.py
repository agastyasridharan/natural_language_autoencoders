"""Cluster-bootstrap inference for claim agreement.

Pair variants are first averaged within activation.  FineWeb uncertainty resamples documents;
OOD uncertainty resamples whole trajectories.  The primary OOD macro gives each of the 16
evaluation families equal weight.  Alignment/capability grouping is reported secondarily.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .core import OOD_BENCHMARKS, PAIR_TYPES, STRATA, collapse_pair_variants


METRICS = ("p_left_to_right", "p_right_to_left", "a_cross")
ANALYSES = ("complete_case", "zero_failure")


def _seed(base_seed: int, label: str) -> int:
    raw = hashlib.sha256(f"{base_seed}|{label}".encode()).digest()[:8]
    return int.from_bytes(raw, "big")


def _cluster_arrays(rows: Sequence[Mapping[str, Any]]) -> tuple[list[str], dict[str, np.ndarray], np.ndarray]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["cluster_id"])].append(row)
    cluster_ids = sorted(grouped)
    counts = np.asarray([len(grouped[key]) for key in cluster_ids], dtype=np.float64)
    sums = {
        metric: np.asarray(
            [sum(float(row[metric]) for row in grouped[key]) for key in cluster_ids],
            dtype=np.float64,
        )
        for metric in METRICS
    }
    return cluster_ids, sums, counts


def _bootstrap_cluster_means(
    rows: Sequence[Mapping[str, Any]], *, n_replicates: int, seed: int, batch_size: int = 512
) -> dict[str, np.ndarray]:
    cluster_ids, sums, counts = _cluster_arrays(rows)
    if not cluster_ids:
        raise ValueError("cannot bootstrap an empty group")
    n_clusters = len(cluster_ids)
    rng = np.random.default_rng(seed)
    outputs = {metric: np.empty(n_replicates, dtype=np.float64) for metric in METRICS}
    probabilities = np.full(n_clusters, 1.0 / n_clusters, dtype=np.float64)
    for start in range(0, n_replicates, batch_size):
        stop = min(start + batch_size, n_replicates)
        weights = rng.multinomial(n_clusters, probabilities, size=stop - start).astype(np.float64)
        denominator = weights @ counts
        for metric in METRICS:
            outputs[metric][start:stop] = (weights @ sums[metric]) / denominator
    return outputs


def _point(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    return {metric: float(np.mean([float(row[metric]) for row in rows])) for metric in METRICS}


def _interval(values: np.ndarray) -> list[float]:
    low, high = np.quantile(values, [0.025, 0.975])
    return [float(low), float(high)]


def _scope_groups(rows: Sequence[Mapping[str, Any]]) -> list[tuple[str, str, list[Mapping[str, Any]]]]:
    groups: list[tuple[str, str, list[Mapping[str, Any]]]] = []
    fineweb = [row for row in rows if row["distribution"] == "fineweb"]
    if fineweb:
        groups.append(("fineweb_overall", "FineWeb", fineweb))
    ood = [row for row in rows if row["distribution"] == "ood"]
    if ood:
        groups.append(("ood_micro", "all_ood", ood))
        for family in ("alignment", "capability"):
            selected = [row for row in ood if row["family"] == family]
            if selected:
                groups.append(("ood_family", family, selected))
        for benchmark in sorted({str(row["benchmark"]) for row in ood}):
            selected = [row for row in ood if row["benchmark"] == benchmark]
            groups.append(("ood_benchmark_family", benchmark, selected))
    return groups


def _completion_counts(
    score_rows: Sequence[Mapping[str, Any]],
    *,
    scope: str,
    scope_value: str,
    pair_type: str,
    stratum: str,
) -> dict[str, int | float]:
    # The complete_case and zero_failure copies have identical completion flags; count one copy only.
    rows = [
        row for row in score_rows
        if row["analysis"] == "complete_case"
        and row["pair_type"] == pair_type
        and row["stratum"] == stratum
    ]
    if scope == "fineweb_overall":
        rows = [row for row in rows if row["distribution"] == "fineweb"]
    elif scope == "ood_micro":
        rows = [row for row in rows if row["distribution"] == "ood"]
    elif scope == "ood_family":
        rows = [row for row in rows if row["distribution"] == "ood" and row["family"] == scope_value]
    elif scope == "ood_benchmark_family":
        rows = [row for row in rows if row["distribution"] == "ood" and row["benchmark"] == scope_value]
    elif scope in ("ood_equal_benchmark_family_macro", "ood_equal_alignment_capability_group_macro"):
        rows = [row for row in rows if row["distribution"] == "ood"]
    expected = len(rows)
    complete = sum(bool(row["complete"]) for row in rows)
    return {
        "n_expected_pair_variants": expected,
        "n_complete_pair_variants": complete,
        "complete_rate": complete / expected if expected else 0.0,
    }


def _summary_record(
    rows: Sequence[Mapping[str, Any]],
    score_rows: Sequence[Mapping[str, Any]],
    *,
    scope: str,
    scope_value: str,
    pair_type: str,
    stratum: str,
    analysis: str,
    n_replicates: int,
    base_seed: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    label = "|".join((scope, scope_value, pair_type, stratum, analysis))
    boot = _bootstrap_cluster_means(rows, n_replicates=n_replicates, seed=_seed(base_seed, label))
    point = _point(rows)
    record = {
        "scope": scope,
        "scope_value": scope_value,
        "pair_type": pair_type,
        "stratum": stratum,
        "analysis": analysis,
        "estimand": "activation_macro_mean_after_within_activation_pair_variant_averaging",
        "n_activations": len(rows),
        "n_clusters": len({row["cluster_id"] for row in rows}),
        "n_bootstrap_replicates": n_replicates,
        **_completion_counts(
            score_rows, scope=scope, scope_value=scope_value, pair_type=pair_type, stratum=stratum
        ),
    }
    if pair_type == "cross_initialization":
        record["left_condition"] = "plausible"
        record["right_condition"] = "implausible"
    elif pair_type == "within_plausible":
        record["left_condition"] = record["right_condition"] = "plausible"
    else:
        record["left_condition"] = record["right_condition"] = "implausible"
    for metric in METRICS:
        record[metric] = point[metric]
        record[f"{metric}_ci95"] = _interval(boot[metric])
    return record, boot


def _set_condition_labels(record: dict[str, Any], pair_type: str) -> None:
    if pair_type == "cross_initialization":
        record["left_condition"] = "plausible"
        record["right_condition"] = "implausible"
    elif pair_type == "within_plausible":
        record["left_condition"] = record["right_condition"] = "plausible"
    else:
        record["left_condition"] = record["right_condition"] = "implausible"


def _equal_group_macro(
    rows: Sequence[Mapping[str, Any]],
    score_rows: Sequence[Mapping[str, Any]],
    *,
    grouped_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    group_completion_scope: str,
    scope: str,
    scope_value: str,
    estimand: str,
    completion_by_key: str,
    primary_ood_estimand: bool,
    pair_type: str,
    stratum: str,
    analysis: str,
    n_replicates: int,
    base_seed: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]] | None:
    if any(not selected for selected in grouped_rows.values()):
        return None
    group_boot: dict[str, dict[str, np.ndarray]] = {}
    for group, selected in grouped_rows.items():
        label = f"{scope}|{group}|{pair_type}|{stratum}|{analysis}"
        group_boot[group] = _bootstrap_cluster_means(
            selected, n_replicates=n_replicates, seed=_seed(base_seed, label)
        )
    weight = 1.0 / len(grouped_rows)
    boot = {
        metric: sum(group_boot[group][metric] for group in grouped_rows) * weight
        for metric in METRICS
    }
    group_points = {group: _point(selected) for group, selected in grouped_rows.items()}
    point = {
        metric: sum(group_points[group][metric] for group in grouped_rows) * weight
        for metric in METRICS
    }
    pooled_completion = _completion_counts(
        score_rows,
        scope=scope,
        scope_value=scope_value,
        pair_type=pair_type,
        stratum=stratum,
    )
    group_completion = {
        group: _completion_counts(
            score_rows, scope=group_completion_scope, scope_value=group,
            pair_type=pair_type, stratum=stratum,
        )
        for group in grouped_rows
    }
    record = {
        "scope": scope,
        "scope_value": scope_value,
        "pair_type": pair_type,
        "stratum": stratum,
        "analysis": analysis,
        "estimand": estimand,
        "primary_ood_estimand": primary_ood_estimand,
        "n_equal_weight_groups": len(grouped_rows),
        "n_activations": sum(len(selected) for selected in grouped_rows.values()),
        "n_clusters": sum(len({row["cluster_id"] for row in selected}) for selected in grouped_rows.values()),
        "n_bootstrap_replicates": n_replicates,
        "n_expected_pair_variants": pooled_completion["n_expected_pair_variants"],
        "n_complete_pair_variants": pooled_completion["n_complete_pair_variants"],
        "pooled_complete_rate": pooled_completion["complete_rate"],
        "complete_rate": sum(float(values["complete_rate"]) for values in group_completion.values()) * weight,
        completion_by_key: {
            group: values["complete_rate"] for group, values in group_completion.items()
        },
    }
    _set_condition_labels(record, pair_type)
    for metric in METRICS:
        record[metric] = point[metric]
        record[f"{metric}_ci95"] = _interval(boot[metric])
    return record, boot


def _equal_benchmark_family_macro(
    rows: Sequence[Mapping[str, Any]],
    score_rows: Sequence[Mapping[str, Any]],
    **kwargs: Any,
) -> tuple[dict[str, Any], dict[str, np.ndarray]] | None:
    grouped = {
        benchmark: [
            row for row in rows
            if row["distribution"] == "ood" and row["benchmark"] == benchmark
        ]
        for benchmark in OOD_BENCHMARKS
    }
    return _equal_group_macro(
        rows,
        score_rows,
        grouped_rows=grouped,
        group_completion_scope="ood_benchmark_family",
        scope="ood_equal_benchmark_family_macro",
        scope_value="16_benchmark_families_equal_weight",
        estimand="equal_weight_mean_of_16_benchmark_family_activation_macro_means",
        completion_by_key="complete_rate_by_benchmark_family",
        primary_ood_estimand=True,
        **kwargs,
    )


def _equal_alignment_capability_group_macro(
    rows: Sequence[Mapping[str, Any]],
    score_rows: Sequence[Mapping[str, Any]],
    **kwargs: Any,
) -> tuple[dict[str, Any], dict[str, np.ndarray]] | None:
    grouped = {
        family: [row for row in rows if row["distribution"] == "ood" and row["family"] == family]
        for family in ("alignment", "capability")
    }
    return _equal_group_macro(
        rows,
        score_rows,
        grouped_rows=grouped,
        group_completion_scope="ood_family",
        scope="ood_equal_alignment_capability_group_macro",
        scope_value="alignment_capability_groups_equal_weight",
        estimand="equal_weight_mean_of_alignment_and_capability_group_activation_macro_means",
        completion_by_key="complete_rate_by_alignment_capability_group",
        primary_ood_estimand=False,
        **kwargs,
    )


def summarize_with_bootstrap(
    score_rows: Sequence[Mapping[str, Any]],
    *,
    n_replicates: int = 20_000,
    seed: int = 17371,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], dict[str, Any]]:
    if n_replicates < 1000:
        raise ValueError("conference analysis requires at least 1,000 bootstrap replicates")
    collapsed = collapse_pair_variants(score_rows)
    summaries: list[dict[str, Any]] = []
    replicate_arrays: dict[str, np.ndarray] = {}
    replicate_index: dict[str, Any] = {}
    for pair_type in PAIR_TYPES:
        for stratum in STRATA:
            for analysis in ANALYSES:
                selected = [
                    row for row in collapsed
                    if row["pair_type"] == pair_type
                    and row["stratum"] == stratum
                    and row["analysis"] == analysis
                ]
                if not selected:
                    continue
                for scope, scope_value, group in _scope_groups(selected):
                    record, boot = _summary_record(
                        group,
                        score_rows,
                        scope=scope,
                        scope_value=scope_value,
                        pair_type=pair_type,
                        stratum=stratum,
                        analysis=analysis,
                        n_replicates=n_replicates,
                        base_seed=seed,
                    )
                    summary_index = len(summaries)
                    summaries.append(record)
                    for metric, values in boot.items():
                        key = f"s{summary_index:04d}_{metric}"
                        replicate_arrays[key] = values
                        replicate_index[key] = {
                            "summary_index": summary_index,
                            "metric": metric,
                            "scope": scope,
                            "scope_value": scope_value,
                            "pair_type": pair_type,
                            "stratum": stratum,
                            "analysis": analysis,
                        }
                for macro_builder in (
                    _equal_benchmark_family_macro,
                    _equal_alignment_capability_group_macro,
                ):
                    macro = macro_builder(
                        selected,
                        score_rows,
                        pair_type=pair_type,
                        stratum=stratum,
                        analysis=analysis,
                        n_replicates=n_replicates,
                        base_seed=seed,
                    )
                    if macro is None:
                        continue
                    record, boot = macro
                    summary_index = len(summaries)
                    summaries.append(record)
                    for metric, values in boot.items():
                        key = f"s{summary_index:04d}_{metric}"
                        replicate_arrays[key] = values
                        replicate_index[key] = {
                            "summary_index": summary_index,
                            "metric": metric,
                            "scope": record["scope"],
                            "scope_value": record["scope_value"],
                            "pair_type": pair_type,
                            "stratum": stratum,
                            "analysis": analysis,
                        }
    metadata = {
        "bootstrap_unit_fineweb": "document_id",
        "bootstrap_unit_ood": "trajectory_id",
        "primary_ood_macro": "equal_weight_across_16_benchmark_families",
        "secondary_ood_group_macro": "equal_weight_alignment_capability_groups",
        "n_replicates": n_replicates,
        "seed": seed,
        "replicate_index": replicate_index,
    }
    return summaries, replicate_arrays, metadata


def write_bootstrap_artifacts(
    output_dir: Path,
    summaries: Sequence[Mapping[str, Any]],
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = output_dir / "bootstrap_replicates.npz.tmp"
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(output_dir / "bootstrap_replicates.npz")
    (output_dir / "bootstrap_index.json").write_text(
        json.dumps(metadata, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(
        json.dumps({"summaries": list(summaries), "bootstrap": dict(metadata)},
                   ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
