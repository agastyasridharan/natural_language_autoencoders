"""Resumable claim-segmentation, equivalence, scoring, and inference runner."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml

from .api_cache import (
    ApiConfig,
    BudgetExceeded,
    CachedCallFailure,
    CachedResponsesClient,
    get_api_key,
    make_request_body,
)
from .core import (
    ALIGNMENT_BENCHMARKS,
    CAPABILITY_BENCHMARKS,
    CONDITIONS,
    PROTOCOL_SCHEMA_VERSION,
    PairScoreInput,
    ValidationError,
    build_pair_plan,
    canonical_json,
    content_hash,
    equivalence_request_key,
    explanation_key,
    file_sha256,
    load_explanations,
    pair_endpoint_keys,
    score_pair,
    segmentation_request_key,
    validate_equivalence_output,
    validate_segmentation_output,
)
from .instruments import (
    EQUIVALENCE_INSTRUCTIONS,
    EQUIVALENCE_INSTRUMENT_VERSION,
    EQUIVALENCE_SCHEMA,
    SEGMENTATION_INSTRUCTIONS,
    SEGMENTATION_INSTRUMENT_VERSION,
    SEGMENTATION_SCHEMA,
    equivalence_input,
    segmentation_input,
)
from .statistics import summarize_with_bootstrap, write_bootstrap_artifacts


HERE = Path(__file__).resolve().parent
DEFAULT_PROTOCOL = HERE / "protocol.yaml"
DEFAULT_CACHE = HERE / "cache"
DEFAULT_OUTPUT = HERE / "results"
ARCHIVE_STATUS_SCHEMA = "nla.claim_agreement.archive_status.v1"
ARCHIVE_HEARTBEAT_MAX_AGE_SECONDS = 150.0


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def validate_archive_guard(path: Path) -> dict[str, Any]:
    """Require a live, private-HF archive watcher before any paid call."""
    if not path.is_file():
        raise ValidationError(f"claim archive watcher status is missing: {path}")
    try:
        status = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise ValidationError(f"claim archive watcher status is unreadable: {error}") from error
    if not isinstance(status, dict) or status.get("schema_version") != ARCHIVE_STATUS_SCHEMA:
        raise ValidationError("claim archive watcher status has an unsupported schema")
    if status.get("state") not in {"waiting_for_artifacts", "healthy"}:
        raise ValidationError(f"claim archive watcher is not healthy: {status.get('state')!r}")
    if status.get("last_error") is not None:
        raise ValidationError(f"claim archive watcher reports an error: {status['last_error']}")
    if not status.get("repo_id") or status.get("repo_type") not in {"dataset", "model"}:
        raise ValidationError("claim archive watcher status lacks a verified private-repo target")
    heartbeat = status.get("heartbeat_unix")
    if not isinstance(heartbeat, (int, float)):
        raise ValidationError("claim archive watcher status lacks a heartbeat")
    age = time.time() - float(heartbeat)
    if age < -30 or age > ARCHIVE_HEARTBEAT_MAX_AGE_SECONDS:
        raise ValidationError(f"claim archive watcher heartbeat is stale ({age:.1f} seconds)")
    return dict(status)


def load_protocol(path: Path) -> tuple[dict[str, Any], str]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != PROTOCOL_SCHEMA_VERSION:
        raise ValidationError(f"unsupported protocol in {path}")
    if raw.get("frozen") is not True:
        raise ValidationError("claim-agreement protocol must be frozen")
    families = raw.get("design", {}).get("ood", {}).get("families", {})
    if tuple(families.get("alignment", ())) != ALIGNMENT_BENCHMARKS:
        raise ValidationError("protocol alignment benchmark families drifted from the scoring code")
    if tuple(families.get("capability", ())) != CAPABILITY_BENCHMARKS:
        raise ValidationError("protocol capability benchmark families drifted from the scoring code")
    if raw.get("design", {}).get("ood", {}).get("primary_macro") != "equal_weight_16_benchmark_families":
        raise ValidationError("primary OOD macro must weight all 16 benchmark families equally")
    return raw, content_hash(raw)


def api_config(protocol: Mapping[str, Any], budget_usd: float | None) -> ApiConfig:
    api = protocol["api"]
    prices = api["pricing_usd_per_million_tokens"]
    maximum = float(api["maximum_hard_budget_usd"])
    selected = float(api["default_hard_budget_usd"] if budget_usd is None else budget_usd)
    if selected <= 0 or selected > maximum:
        raise ValidationError(f"--budget-usd must be in (0, {maximum}]")
    if api["api_key_environment_variable"] != "OPENAI_API_KEY_2" or api["store"] is not False:
        raise ValidationError("protocol API privacy settings drifted")
    if api.get("automatic_model_fallback") is not False or api.get("automatic_retry") is not False:
        raise ValidationError("protocol must disable model fallback and automatic retries")
    return ApiConfig(
        endpoint=str(api["endpoint"]),
        model=str(api["model"]),
        reasoning_effort=str(api["reasoning_effort"]),
        service_tier=str(api["service_tier"]),
        input_usd_per_million=float(prices["input"]),
        cached_input_usd_per_million=float(prices["cached_input"]),
        output_usd_per_million=float(prices["output"]),
        hard_budget_usd=selected,
    )


def validate_design(
    rows: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any], *, finalize: bool
) -> dict[str, Any]:
    required_samples = int(protocol["design"]["required_samples_per_condition"])
    counts: dict[tuple[str, str], list[int]] = defaultdict(list)
    for row in rows:
        counts[(str(row["activation_id"]), str(row["condition"]))].append(int(row["sample_index"]))
    activation_ids = sorted({str(row["activation_id"]) for row in rows})
    sample_errors: list[str] = []
    for activation_id in activation_ids:
        for condition in CONDITIONS:
            observed = sorted(counts.get((activation_id, condition), []))
            expected = list(range(required_samples))
            if observed != expected:
                sample_errors.append(f"{activation_id}:{condition}: observed={observed}, expected={expected}")
    family_for_benchmark: dict[str, str] = {"fineweb": "fineweb"}
    for family, benchmarks in protocol["design"]["ood"]["families"].items():
        for benchmark in benchmarks:
            family_for_benchmark[str(benchmark)] = str(family)
    unknown: list[str] = []
    for row in rows:
        expected_family = family_for_benchmark.get(str(row["benchmark"]))
        if expected_family is None or expected_family != row["family"]:
            unknown.append(f"{row['benchmark']}:{row['family']}")
    observed_benchmarks = sorted({str(row["benchmark"]) for row in rows})
    missing_benchmarks = sorted(set(family_for_benchmark) - set(observed_benchmarks))
    audit = {
        "n_explanation_rows": len(rows),
        "n_activations": len(activation_ids),
        "required_samples_per_condition": required_samples,
        "sample_design_errors": sample_errors,
        "unknown_or_misclassified_benchmarks": sorted(set(unknown)),
        "observed_benchmarks": observed_benchmarks,
        "missing_benchmarks": missing_benchmarks,
        "final_design_complete": not sample_errors and not unknown and not missing_benchmarks,
    }
    if sample_errors or unknown or (finalize and missing_benchmarks):
        raise ValidationError("design validation failed: " + canonical_json(audit))
    return audit


def _cache_state(cache_root: Path, stage: str, key: str) -> tuple[str, dict[str, Any] | None]:
    key_dir = cache_root / stage / key
    result_path = key_dir / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        return "success", result
    attempts = sorted((key_dir / "attempts").glob("*.json")) if (key_dir / "attempts").exists() else []
    if attempts:
        attempt = json.loads(attempts[-1].read_text(encoding="utf-8"))
        return str(attempt.get("outcome", "instrument_failure")), attempt
    return "pending", None


def _protocol_key(protocol: Mapping[str, Any], protocol_hash: str, instrument_version: str) -> str:
    return f"{protocol['protocol_id']}:{protocol_hash}:{instrument_version}"


def _segmentation_key(row: Mapping[str, Any], protocol: Mapping[str, Any], protocol_hash: str) -> str:
    return segmentation_request_key(
        str(row["explanation"]), _protocol_key(protocol, protocol_hash, SEGMENTATION_INSTRUMENT_VERSION)
    )


def _segment_record(
    row: Mapping[str, Any], cache_root: Path, protocol: Mapping[str, Any], protocol_hash: str
) -> dict[str, Any]:
    base = dict(row)
    if not row["generation_ok"]:
        return {
            **base,
            "segmentation_status": "upstream_generation_failure",
            "segmentation_failure_reason": row["failure_reason"],
            "segmentation_cache_key": None,
            "segmentation": None,
            "api_result": None,
        }
    key = _segmentation_key(row, protocol, protocol_hash)
    state, record = _cache_state(cache_root, "segment", key)
    if state == "success" and record is not None:
        segmentation = record["parsed_output"]
        if "".join(unit["source_text"] for unit in segmentation["units"]) != row["explanation"]:
            raise ValidationError(f"cached segmentation no longer reconstructs explanation for {key}")
        return {
            **base,
            "segmentation_status": "success",
            "segmentation_failure_reason": None,
            "segmentation_cache_key": key,
            "segmentation": segmentation,
            "api_result": {
                "attempt_number": record["attempt_number"],
                "usage": record.get("usage"),
                "charge": record.get("charge"),
            },
        }
    return {
        **base,
        "segmentation_status": state,
        "segmentation_failure_reason": None if record is None else record.get("error"),
        "segmentation_cache_key": key,
        "segmentation": None,
        "api_result": None,
    }


def build_segment_records(
    rows: Sequence[Mapping[str, Any]], cache_root: Path, protocol: Mapping[str, Any], protocol_hash: str
) -> list[dict[str, Any]]:
    return [_segment_record(row, cache_root, protocol, protocol_hash) for row in rows]


def _segment_work(
    rows: Sequence[Mapping[str, Any]],
    cache_root: Path,
    protocol: Mapping[str, Any],
    protocol_hash: str,
    *,
    retry_failures: bool,
) -> list[tuple[str, str]]:
    work: dict[str, str] = {}
    for row in rows:
        if not row["generation_ok"]:
            continue
        key = _segmentation_key(row, protocol, protocol_hash)
        state, _ = _cache_state(cache_root, "segment", key)
        if state == "success" or (state != "pending" and not retry_failures):
            continue
        work.setdefault(key, str(row["explanation"]))
    return sorted(work.items())


def _segment_request(config: ApiConfig, explanation: str, max_output_tokens: int) -> dict[str, Any]:
    return make_request_body(
        config,
        instructions=SEGMENTATION_INSTRUCTIONS,
        input_text=segmentation_input(explanation),
        schema_name="nla_claim_segmentation",
        schema=SEGMENTATION_SCHEMA,
        max_output_tokens=max_output_tokens,
    )


def _run_calls(
    work: Sequence[Any], call: Callable[[Any], None], *, workers: int
) -> tuple[int, Counter[str]]:
    outcomes: Counter[str] = Counter()
    if not work:
        return 0, outcomes
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(call, item): item for item in work}
        for future in as_completed(futures):
            try:
                future.result()
                outcomes["success"] += 1
            except CachedCallFailure as exc:
                outcomes[exc.code] += 1
            except BudgetExceeded:
                outcomes["budget_exceeded"] += 1
            except Exception as exc:  # preserve batch progress, then report exact class
                outcomes[f"unexpected_{type(exc).__name__}"] += 1
    return len(work), outcomes


def dry_run_segment(
    rows: Sequence[Mapping[str, Any]],
    cache_root: Path,
    client: CachedResponsesClient,
    protocol: Mapping[str, Any],
    protocol_hash: str,
    *,
    retry_failures: bool,
    max_calls: int,
) -> dict[str, Any]:
    work = _segment_work(
        rows, cache_root, protocol, protocol_hash, retry_failures=retry_failures
    )
    max_output = int(protocol["api"]["max_output_tokens"]["segmentation"])
    selected = work[:max_calls]
    all_costs: list[float] = []
    all_token_bounds: list[int] = []
    for _, explanation in work:
        body = _segment_request(client.config, explanation, max_output)
        cost, token_bound = client.ledger.worst_case_cost(body, max_output)
        all_costs.append(cost)
        all_token_bounds.append(token_bound)
    records = build_segment_records(rows, cache_root, protocol, protocol_hash)
    return {
        "stage": "segment",
        "mode": "dry_run",
        "cache_status": dict(Counter(record["segmentation_status"] for record in records)),
        "pending_unique_calls": len(work),
        "calls_selected_by_max_calls": len(selected),
        "conservative_input_token_upper_bound_selected": sum(all_token_bounds[:len(selected)]),
        "conservative_cost_upper_bound_selected_usd": sum(all_costs[:len(selected)]),
        "conservative_input_token_upper_bound_all_pending": sum(all_token_bounds),
        "conservative_cost_upper_bound_all_pending_usd": sum(all_costs),
        "note": "Upper bound reserves max output for every selected call; actual reported-token cost should be lower.",
    }


def execute_segment(
    rows: Sequence[Mapping[str, Any]],
    cache_root: Path,
    output_dir: Path,
    client: CachedResponsesClient,
    protocol: Mapping[str, Any],
    protocol_hash: str,
    *,
    retry_failures: bool,
    max_calls: int,
    workers: int,
) -> tuple[dict[str, Any], int]:
    work = _segment_work(rows, cache_root, protocol, protocol_hash, retry_failures=retry_failures)[:max_calls]
    max_output = int(protocol["api"]["max_output_tokens"]["segmentation"])

    def call(item: tuple[str, str]) -> None:
        key, explanation = item
        client.call(
            stage="segment",
            cache_key=key,
            instructions=SEGMENTATION_INSTRUCTIONS,
            input_text=segmentation_input(explanation),
            schema_name="nla_claim_segmentation",
            schema=SEGMENTATION_SCHEMA,
            max_output_tokens=max_output,
            validator=lambda raw: validate_segmentation_output(explanation, raw),
            retry_failure=retry_failures,
        )

    attempted, outcomes = _run_calls(work, call, workers=workers)
    records = build_segment_records(rows, cache_root, protocol, protocol_hash)
    _atomic_jsonl(output_dir / "segmented_explanations.jsonl", records)
    return {
        "stage": "segment",
        "mode": "execute",
        "attempted_calls": attempted,
        "attempt_outcomes": dict(outcomes),
        "cache_status": dict(Counter(record["segmentation_status"] for record in records)),
    }, attempted


def _segment_map(records: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str, int], Mapping[str, Any]]:
    return {explanation_key(row): row for row in records}


def _equivalence_key(
    pair: Mapping[str, Any],
    segments: Mapping[tuple[str, str, int], Mapping[str, Any]],
    protocol: Mapping[str, Any],
    protocol_hash: str,
) -> tuple[str, Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]]] | None:
    left_key, right_key = pair_endpoint_keys(pair)
    left = segments[left_key]
    right = segments[right_key]
    if left["segmentation_status"] != "success" or right["segmentation_status"] != "success":
        return None
    left_claims = left["segmentation"]["claims"]
    right_claims = right["segmentation"]["claims"]
    key = equivalence_request_key(
        left_claims,
        right_claims,
        _protocol_key(protocol, protocol_hash, EQUIVALENCE_INSTRUMENT_VERSION),
    )
    return key, left_claims, right_claims


def _equivalence_work(
    pairs: Sequence[Mapping[str, Any]],
    segments: Mapping[tuple[str, str, int], Mapping[str, Any]],
    cache_root: Path,
    protocol: Mapping[str, Any],
    protocol_hash: str,
    *,
    retry_failures: bool,
) -> list[tuple[str, Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]]]]:
    work: dict[str, tuple[Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]]]] = {}
    for pair in pairs:
        value = _equivalence_key(pair, segments, protocol, protocol_hash)
        if value is None:
            continue
        key, left_claims, right_claims = value
        state, _ = _cache_state(cache_root, "equivalence", key)
        if state == "success" or (state != "pending" and not retry_failures):
            continue
        work.setdefault(key, (left_claims, right_claims))
    return [(key, *work[key]) for key in sorted(work)]


def build_equivalence_records(
    pairs: Sequence[Mapping[str, Any]],
    segments: Mapping[tuple[str, str, int], Mapping[str, Any]],
    cache_root: Path,
    protocol: Mapping[str, Any],
    protocol_hash: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for pair in pairs:
        value = _equivalence_key(pair, segments, protocol, protocol_hash)
        if value is None:
            left_key, right_key = pair_endpoint_keys(pair)
            left_state = segments[left_key]["segmentation_status"]
            right_state = segments[right_key]["segmentation_status"]
            records.append({
                **pair,
                "equivalence_status": "blocked_by_segmentation",
                "equivalence_failure_reason": f"left={left_state};right={right_state}",
                "equivalence_cache_key": None,
                "equivalence": None,
                "api_result": None,
            })
            continue
        key, _, _ = value
        state, record = _cache_state(cache_root, "equivalence", key)
        if state == "success" and record is not None:
            records.append({
                **pair,
                "equivalence_status": "success",
                "equivalence_failure_reason": None,
                "equivalence_cache_key": key,
                "equivalence": record["parsed_output"],
                "api_result": {
                    "attempt_number": record["attempt_number"],
                    "usage": record.get("usage"),
                    "charge": record.get("charge"),
                },
            })
        else:
            records.append({
                **pair,
                "equivalence_status": state,
                "equivalence_failure_reason": None if record is None else record.get("error"),
                "equivalence_cache_key": key,
                "equivalence": None,
                "api_result": None,
            })
    return records


def dry_run_equivalence(
    pairs: Sequence[Mapping[str, Any]],
    segments: Mapping[tuple[str, str, int], Mapping[str, Any]],
    cache_root: Path,
    client: CachedResponsesClient,
    protocol: Mapping[str, Any],
    protocol_hash: str,
    *,
    retry_failures: bool,
    max_calls: int,
) -> dict[str, Any]:
    work = _equivalence_work(
        pairs, segments, cache_root, protocol, protocol_hash, retry_failures=retry_failures
    )
    selected = work[:max_calls]
    max_output = int(protocol["api"]["max_output_tokens"]["equivalence"])
    all_costs: list[float] = []
    all_token_bounds: list[int] = []
    for _, left, right in work:
        body = make_request_body(
            client.config,
            instructions=EQUIVALENCE_INSTRUCTIONS,
            input_text=equivalence_input(left, right),
            schema_name="nla_claim_equivalence",
            schema=EQUIVALENCE_SCHEMA,
            max_output_tokens=max_output,
        )
        cost, token_bound = client.ledger.worst_case_cost(body, max_output)
        all_costs.append(cost)
        all_token_bounds.append(token_bound)
    records = build_equivalence_records(pairs, segments, cache_root, protocol, protocol_hash)
    return {
        "stage": "equivalence",
        "mode": "dry_run",
        "cache_status": dict(Counter(record["equivalence_status"] for record in records)),
        "pending_unique_calls": len(work),
        "calls_selected_by_max_calls": len(selected),
        "conservative_input_token_upper_bound_selected": sum(all_token_bounds[:len(selected)]),
        "conservative_cost_upper_bound_selected_usd": sum(all_costs[:len(selected)]),
        "conservative_input_token_upper_bound_all_pending": sum(all_token_bounds),
        "conservative_cost_upper_bound_all_pending_usd": sum(all_costs),
        "note": (
            "Only pairs with cached successful segmentations are estimable; rerun after segmentation "
            "for the full judge estimate."
        ),
    }


def execute_equivalence(
    pairs: Sequence[Mapping[str, Any]],
    segments: Mapping[tuple[str, str, int], Mapping[str, Any]],
    cache_root: Path,
    output_dir: Path,
    client: CachedResponsesClient,
    protocol: Mapping[str, Any],
    protocol_hash: str,
    *,
    retry_failures: bool,
    max_calls: int,
    workers: int,
) -> tuple[dict[str, Any], int]:
    work = _equivalence_work(
        pairs, segments, cache_root, protocol, protocol_hash, retry_failures=retry_failures
    )[:max_calls]
    max_output = int(protocol["api"]["max_output_tokens"]["equivalence"])

    def call(item: tuple[str, Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]]]) -> None:
        key, left_claims, right_claims = item
        client.call(
            stage="equivalence",
            cache_key=key,
            instructions=EQUIVALENCE_INSTRUCTIONS,
            input_text=equivalence_input(left_claims, right_claims),
            schema_name="nla_claim_equivalence",
            schema=EQUIVALENCE_SCHEMA,
            max_output_tokens=max_output,
            validator=lambda raw: validate_equivalence_output(raw, left_claims, right_claims),
            retry_failure=retry_failures,
        )

    attempted, outcomes = _run_calls(work, call, workers=workers)
    records = build_equivalence_records(pairs, segments, cache_root, protocol, protocol_hash)
    _atomic_jsonl(output_dir / "pair_plan.jsonl", pairs)
    _atomic_jsonl(output_dir / "equivalence.jsonl", records)
    return {
        "stage": "equivalence",
        "mode": "execute",
        "attempted_calls": attempted,
        "attempt_outcomes": dict(outcomes),
        "cache_status": dict(Counter(record["equivalence_status"] for record in records)),
    }, attempted


def score_from_cache(
    rows: Sequence[Mapping[str, Any]],
    cache_root: Path,
    output_dir: Path,
    protocol: Mapping[str, Any],
    protocol_hash: str,
    *,
    finalize: bool,
) -> dict[str, Any]:
    segment_records = build_segment_records(rows, cache_root, protocol, protocol_hash)
    segments = _segment_map(segment_records)
    pairs = build_pair_plan(rows)
    equivalence_records = build_equivalence_records(pairs, segments, cache_root, protocol, protocol_hash)
    equivalence_by_pair = {record["pair_id"]: record for record in equivalence_records}
    score_rows: list[dict[str, Any]] = []
    pending_pairs: list[str] = []
    for pair in pairs:
        left_key, right_key = pair_endpoint_keys(pair)
        left_record, right_record = segments[left_key], segments[right_key]
        left_seg = left_record["segmentation"] if left_record["segmentation_status"] == "success" else None
        right_seg = right_record["segmentation"] if right_record["segmentation_status"] == "success" else None
        eq_record = equivalence_by_pair[pair["pair_id"]]
        equivalence = eq_record["equivalence"] if eq_record["equivalence_status"] == "success" else None
        if left_seg is None:
            failure = f"left_segmentation_{left_record['segmentation_status']}"
        elif right_seg is None:
            failure = f"right_segmentation_{right_record['segmentation_status']}"
        elif equivalence is None:
            failure = f"equivalence_{eq_record['equivalence_status']}"
        else:
            failure = None
        if failure and ("pending" in failure or "blocked_by_segmentation" in failure):
            pending_pairs.append(str(pair["pair_id"]))
        score_rows.extend(score_pair(PairScoreInput(pair, left_seg, right_seg, equivalence, failure)))
    if finalize and pending_pairs:
        raise ValidationError(f"cannot finalize with {len(pending_pairs)} pending pair instruments")
    _atomic_jsonl(output_dir / "segmented_explanations.jsonl", segment_records)
    _atomic_jsonl(output_dir / "pair_plan.jsonl", pairs)
    _atomic_jsonl(output_dir / "equivalence.jsonl", equivalence_records)
    _atomic_jsonl(output_dir / "pair_scores.jsonl", score_rows)
    inference = protocol["inference"]
    summaries, arrays, bootstrap_metadata = summarize_with_bootstrap(
        score_rows,
        n_replicates=int(inference["bootstrap_replicates"]),
        seed=int(inference["bootstrap_seed"]),
    )
    write_bootstrap_artifacts(output_dir, summaries, arrays, bootstrap_metadata)
    return {
        "stage": "score",
        "n_pairs": len(pairs),
        "n_score_rows": len(score_rows),
        "n_summaries": len(summaries),
        "pending_pairs": len(pending_pairs),
        "complete_pair_status": dict(Counter(
            "complete" if row["complete"] else str(row["failure_reason"])
            for row in score_rows if row["analysis"] == "complete_case" and row["stratum"] == "all"
        )),
    }


def _write_run_metadata(
    output_dir: Path,
    protocol_path: Path,
    protocol_hash: str,
    input_paths: Sequence[Path],
    design_audit: Mapping[str, Any],
    stage_reports: Sequence[Mapping[str, Any]],
    client: CachedResponsesClient,
    archive_guard: Mapping[str, Any] | None,
) -> None:
    artifact_names = (
        "segmented_explanations.jsonl", "pair_plan.jsonl", "equivalence.jsonl",
        "pair_scores.jsonl", "summary.json", "bootstrap_replicates.npz", "bootstrap_index.json",
    )
    artifacts = {
        name: {"sha256": file_sha256(output_dir / name), "bytes": (output_dir / name).stat().st_size}
        for name in artifact_names if (output_dir / name).exists()
    }
    metadata = {
        "schema_version": "nla.claim_agreement.run_metadata.v1",
        "created_at": _utc_now(),
        "protocol_path": str(protocol_path.resolve()),
        "protocol_sha256": protocol_hash,
        "inputs": [
            {"path": str(path.resolve()), "sha256": file_sha256(path), "bytes": path.stat().st_size}
            for path in input_paths
        ],
        "design_audit": dict(design_audit),
        "stage_reports": list(stage_reports),
        "budget_ledger": client.ledger.snapshot(),
        "archive_guard_at_launch": None if archive_guard is None else dict(archive_guard),
        "artifacts": artifacts,
    }
    _atomic_json(output_dir / "run_metadata.json", metadata)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True, help="Explanation JSONL; repeatable")
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--archive-status",
        type=Path,
        help="Live private-HF archive watcher status; required for paid API attempts.",
    )
    parser.add_argument("--stage", choices=("segment", "equivalence", "score", "all", "status"), default="status")
    parser.add_argument("--execute", action="store_true", help="Authorize uncached API calls for API stages")
    parser.add_argument("--max-calls", type=int, default=100, help="Maximum new API attempts in this invocation")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retry-failures", action="store_true", help="Explicit same-model retry; old attempts remain")
    parser.add_argument(
        "--budget-usd", type=float, default=None,
        help="Hard claim-pipeline cap, at most protocol maximum",
    )
    parser.add_argument(
        "--finalize", action="store_true",
        help="Require all 17 benchmark suites and no pending instruments",
    )
    args = parser.parse_args(argv)
    if args.max_calls < 0:
        parser.error("--max-calls must be >= 0")
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    if args.execute and args.stage in ("score", "status"):
        parser.error("--execute is only valid for segment, equivalence, or all")
    if args.execute and args.max_calls > 0 and args.archive_status is None:
        parser.error("--execute with new calls requires --archive-status")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    protocol, protocol_hash = load_protocol(args.protocol)
    rows = load_explanations(args.input)
    design_audit = validate_design(rows, protocol, finalize=args.finalize)
    config = api_config(protocol, args.budget_usd)
    client = CachedResponsesClient(args.cache_dir, config)
    archive_guard = None
    if args.execute and args.max_calls > 0:
        archive_guard = validate_archive_guard(args.archive_status)
        get_api_key()  # fail before scheduling workers or creating a budget reservation
    pairs = build_pair_plan(rows)
    reports: list[dict[str, Any]] = []
    remaining_calls = args.max_calls

    if args.stage in ("segment", "all", "status"):
        if args.execute:
            report, attempted = execute_segment(
                rows, args.cache_dir, args.output_dir, client, protocol, protocol_hash,
                retry_failures=args.retry_failures, max_calls=remaining_calls, workers=args.workers,
            )
            remaining_calls -= attempted
        else:
            report = dry_run_segment(
                rows, args.cache_dir, client, protocol, protocol_hash,
                retry_failures=args.retry_failures, max_calls=remaining_calls,
            )
        reports.append(report)

    segment_records = build_segment_records(rows, args.cache_dir, protocol, protocol_hash)
    segments = _segment_map(segment_records)
    if args.stage in ("equivalence", "all", "status"):
        if args.execute:
            report, attempted = execute_equivalence(
                pairs, segments, args.cache_dir, args.output_dir, client, protocol, protocol_hash,
                retry_failures=args.retry_failures, max_calls=remaining_calls, workers=args.workers,
            )
            remaining_calls -= attempted
        else:
            report = dry_run_equivalence(
                pairs, segments, args.cache_dir, client, protocol, protocol_hash,
                retry_failures=args.retry_failures, max_calls=remaining_calls,
            )
        reports.append(report)

    if args.stage in ("score", "all"):
        reports.append(score_from_cache(
            rows, args.cache_dir, args.output_dir, protocol, protocol_hash, finalize=args.finalize
        ))
        _write_run_metadata(
            args.output_dir, args.protocol, protocol_hash, args.input,
            design_audit, reports, client, archive_guard,
        )

    output = {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_hash,
        "design_audit": design_audit,
        "reports": reports,
        "remaining_max_calls_this_invocation": remaining_calls,
        "budget": client.ledger.snapshot(),
        "api_calls_launched": bool(args.execute),
        "archive_guard": archive_guard,
    }
    print(json.dumps(output, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValidationError, BudgetExceeded, CachedCallFailure) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
