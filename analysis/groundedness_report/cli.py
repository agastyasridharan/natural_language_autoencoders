"""CPU control plane for the report-faithful groundedness experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .gpu_measure import _assert_claim_tiling
from .io import atomic_json, atomic_jsonl, load_calibration, measurement_rows, read_json, read_jsonl
from .calibrate import freeze_thresholds
from .sampling import cyclic_derangement
from .spec import (
    PROTOCOL_SHA256,
    RECONSTRUCTION_ERROR_FORMULA,
    RECONSTRUCTION_METRIC_AMENDMENT_SHA256,
    REPORT_SHA256,
    SCHEMA_VERSION,
    TEXT_MODEL_CORRECTION_AMENDMENT_SHA256,
    classify_claim,
)
from .stats import clustered_intervals, ood_macro_intervals
from .workflow import (
    calibration_audit,
    make_claim_sample,
    make_document_split,
    make_evaluation_split,
    stratified_summary,
    validate_split,
)


def cmd_split(args: argparse.Namespace) -> None:
    result = make_document_split(
        args.records,
        args.out,
        calibration_fraction=args.calibration_fraction,
        seed=args.seed,
    )
    print(json.dumps({"out": str(args.out), "n_documents": result["n_documents"]}, indent=2))


def cmd_evaluation_split(args: argparse.Namespace) -> None:
    result = make_evaluation_split(args.records, args.out)
    print(json.dumps({
        "out": str(args.out),
        "n_documents": result["n_documents"],
        "n_evaluation": len(result["evaluation_indices"]),
    }, indent=2))


def cmd_validate_calibration(args: argparse.Namespace) -> None:
    calibration = load_calibration(args.calibration)
    print(json.dumps(calibration.as_dict(), indent=2))


def cmd_claim_sample(args: argparse.Namespace) -> None:
    result = make_claim_sample(
        args.records,
        args.split,
        args.out,
        seed=args.seed,
        maximum=3,
    )
    print(json.dumps({
        "out": str(args.out),
        "n_activations": result["n_activations"],
        "n_sampled_claims": result["n_sampled_claims"],
    }, indent=2))


def cmd_negative_map(args: argparse.Namespace) -> None:
    split = read_json(args.split)
    validate_split(split)
    calibration_indices = set(int(value) for value in split["calibration_indices"])
    records = {
        int(row["idx"]): row
        for row in read_jsonl(args.records)
        if int(row["idx"]) in calibration_indices
    }
    if set(records) != calibration_indices:
        raise ValueError("calibration split and records do not contain identical indices")
    index_by_doc: dict[str, int] = {}
    for idx, row in records.items():
        doc_id = str(row.get("doc_id") or "")
        if not doc_id or doc_id in index_by_doc:
            raise ValueError("negative map requires one FineWeb activation per unique document")
        index_by_doc[doc_id] = idx
    doc_map = cyclic_derangement(
        list(index_by_doc), seed=args.seed, namespace="fineweb-calibration-negative-control"
    )
    mapping = {
        str(index_by_doc[source_doc]): index_by_doc[donor_doc]
        for source_doc, donor_doc in doc_map.items()
    }
    result = {
        "schema_version": 1,
        "protocol_sha256": PROTOCOL_SHA256,
        "kind": "one_seeded_cyclic_derangement_after_sha256_rank",
        "seed": args.seed,
        "split_sha256": hashlib.sha256(Path(args.split).read_bytes()).hexdigest(),
        "records_sha256": hashlib.sha256(Path(args.records).read_bytes()).hexdigest(),
        "n": len(mapping),
        "mapping": dict(sorted(mapping.items(), key=lambda item: int(item[0]))),
    }
    atomic_json(args.out, result)
    print(json.dumps({"out": str(args.out), "n": len(mapping)}, indent=2))


def cmd_calibration_audit(args: argparse.Namespace) -> None:
    rows = measurement_rows(args.measurements)
    incompatible_metric_rows = [
        (row.get("idx"), row.get("claim_i"))
        for row in rows
        if row.get("reconstruction_metric_amendment_sha256")
        != RECONSTRUCTION_METRIC_AMENDMENT_SHA256
        or row.get("reconstruction_error_formula") != RECONSTRUCTION_ERROR_FORMULA
        or row.get("text_model_correction_amendment_sha256")
        != TEXT_MODEL_CORRECTION_AMENDMENT_SHA256
    ]
    if incompatible_metric_rows:
        raise ValueError(
            "calibration measurements use an absent or incompatible reconstruction metric: "
            f"{incompatible_metric_rows[:5]}"
        )
    audit = calibration_audit(rows)
    audit.update({
        "reconstruction_metric_amendment_sha256": (
            RECONSTRUCTION_METRIC_AMENDMENT_SHA256
        ),
        "reconstruction_error_formula": RECONSTRUCTION_ERROR_FORMULA,
        "text_model_correction_amendment_sha256": (
            TEXT_MODEL_CORRECTION_AMENDMENT_SHA256
        ),
    })
    provenance_fields = {
        "model_key": {str(row.get("model_key")) for row in rows},
        "run_id": {str(row.get("run_id")) for row in rows},
        "split_sha256": {str(row.get("split_sha256")) for row in rows},
        "reference_sha256": {
            str((row.get("reference_activation") or {}).get("sha256")) for row in rows
        },
    }
    conflicts = {
        key: sorted(values)
        for key, values in provenance_fields.items()
        if len(values) != 1 or next(iter(values), "") in {"", "None"}
    }
    if conflicts:
        raise ValueError(f"calibration measurements mix provenance: {conflicts}")
    audit["provenance"] = {key: next(iter(values)) for key, values in provenance_fields.items()}
    atomic_json(args.out, audit)
    print(json.dumps({"out": str(args.out), "n_claims": audit["n_claims"]}, indent=2))


def cmd_freeze_calibration(args: argparse.Namespace) -> None:
    rows = measurement_rows(args.measurements)
    negative_sha256 = hashlib.sha256(Path(args.measurements).read_bytes()).hexdigest() \
        if Path(args.measurements).is_file() else args.negative_control_sha256
    if not negative_sha256:
        raise ValueError("directory inputs require --negative-control-sha256")
    provenance = {
        "model_key": args.model_key,
        "run_id": args.run_id,
        "split_sha256": hashlib.sha256(Path(args.split).read_bytes()).hexdigest(),
        "reference_sha256": hashlib.sha256(np.asarray(np.load(args.reference), dtype=np.float32).tobytes()).hexdigest(),
        "negative_control_sha256": negative_sha256,
    }
    calibration, audit = freeze_thresholds(rows, provenance=provenance)
    atomic_json(args.out, calibration.as_dict())
    audit_path = Path(args.audit) if args.audit else Path(args.out).with_suffix(".audit.json")
    atomic_json(audit_path, audit)
    print(json.dumps({
        "out": str(args.out),
        "audit": str(audit_path),
        "n_calibration_claims": audit["n_calibration_claims"],
        "n_negative_claims_complete": audit["n_negative_claims_complete"],
        "resolvability_target_met": audit["resolvability_target_met"],
    }, indent=2))


def cmd_classify(args: argparse.Namespace) -> None:
    calibration = load_calibration(args.calibration)
    calibration_fingerprint = hashlib.sha256(
        json.dumps(calibration.as_dict(), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    measurement_values = measurement_rows(args.measurements)
    mismatches = []
    for row in measurement_values:
        observed = {
            "model_key": row.get("model_key"),
            "run_id": row.get("run_id"),
            "split_sha256": row.get("split_sha256"),
            "reference_sha256": (row.get("reference_activation") or {}).get("sha256"),
        }
        expected = {
            "model_key": calibration.model_key,
            "run_id": calibration.run_id,
            "split_sha256": calibration.split_sha256,
            "reference_sha256": calibration.reference_sha256,
        }
        # A frozen FineWeb calibration is intentionally reused for OOD runs.
        # Their local run and split are distinct provenance, while model and
        # reference activation must remain identical.  Calibration rows, when
        # classified descriptively, still have to originate from the exact
        # run and split on which the thresholds were frozen.
        required_keys = {"model_key", "reference_sha256"}
        if row.get("partition") != "evaluation":
            required_keys.update({"run_id", "split_sha256"})
        bad = {
            key: (observed[key], expected[key])
            for key in required_keys
            if observed[key] != expected[key]
        }
        if (
            row.get("reconstruction_metric_amendment_sha256")
            != RECONSTRUCTION_METRIC_AMENDMENT_SHA256
        ):
            bad["reconstruction_metric_amendment_sha256"] = (
                row.get("reconstruction_metric_amendment_sha256"),
                RECONSTRUCTION_METRIC_AMENDMENT_SHA256,
            )
        if row.get("reconstruction_error_formula") != RECONSTRUCTION_ERROR_FORMULA:
            bad["reconstruction_error_formula"] = (
                row.get("reconstruction_error_formula"),
                RECONSTRUCTION_ERROR_FORMULA,
            )
        if row.get(
            "text_model_correction_amendment_sha256"
        ) != TEXT_MODEL_CORRECTION_AMENDMENT_SHA256:
            bad["text_model_correction_amendment_sha256"] = (
                row.get("text_model_correction_amendment_sha256"),
                TEXT_MODEL_CORRECTION_AMENDMENT_SHA256,
            )
        if row.get("partition") == "evaluation" and row.get("calibration_fingerprint") != calibration_fingerprint:
            bad["calibration_fingerprint"] = (
                row.get("calibration_fingerprint"),
                calibration_fingerprint,
            )
        if bad:
            mismatches.append((row.get("idx"), row.get("claim_i"), bad))
    if mismatches:
        raise ValueError(f"measurement/calibration provenance mismatch: {mismatches[:5]}")
    outputs = []
    for row in measurement_values:
        identity = {
            key: row.get(key)
            for key in (
                "model_key",
                "run_id",
                "idx",
                "doc_id",
                "trajectory_id",
                "evaluation_family",
                "claim_i",
                "claim_text",
                "claim_type",
                "claim_subtype",
                "inclusion_probability",
                "analysis_weight",
            )
        }
        if row.get("status") != "ok":
            verdict = {"verdict": "unresolved", "reason": str(row.get("reason") or row.get("status"))}
        else:
            try:
                verdict = classify_claim(
                    real=row["real"],
                    candidates=row.get("candidates") or [],
                    off_diagonal_effects=row.get("off_diagonal_effects") or [],
                    calibration=calibration,
                )
            except (KeyError, TypeError, ValueError) as error:
                verdict = {"verdict": "unresolved", "reason": f"malformed measurement: {error}"}
        outputs.append({
            "schema_version": SCHEMA_VERSION,
            "report_sha256": REPORT_SHA256,
            "reconstruction_metric_amendment_sha256": (
                RECONSTRUCTION_METRIC_AMENDMENT_SHA256
            ),
            "reconstruction_error_formula": RECONSTRUCTION_ERROR_FORMULA,
            "text_model_correction_amendment_sha256": (
                TEXT_MODEL_CORRECTION_AMENDMENT_SHA256
            ),
            **identity,
            **verdict,
        })
    outputs.sort(key=lambda row: (str(row.get("model_key")), int(row.get("idx") or -1), int(row.get("claim_i") or -1)))
    atomic_jsonl(args.out, outputs)
    summary = stratified_summary(outputs)
    summary.update({
        "schema_version": SCHEMA_VERSION,
        "report_sha256": REPORT_SHA256,
        "reconstruction_metric_amendment_sha256": (
            RECONSTRUCTION_METRIC_AMENDMENT_SHA256
        ),
        "reconstruction_error_formula": RECONSTRUCTION_ERROR_FORMULA,
        "text_model_correction_amendment_sha256": (
            TEXT_MODEL_CORRECTION_AMENDMENT_SHA256
        ),
        "calibration": calibration.as_dict(),
    })
    summary_path = Path(args.summary) if args.summary else Path(args.out).with_suffix(".summary.json")
    atomic_json(summary_path, summary)
    print(json.dumps({"out": str(args.out), "summary": str(summary_path), **summary["counts"]}, indent=2))


def cmd_preflight(args: argparse.Namespace) -> None:
    split = read_json(args.split)
    validate_split(split)
    calibration = load_calibration(args.calibration) if args.calibration else None
    records = list(read_jsonl(args.records))
    record_indices = {int(row["idx"]) for row in records}
    split_indices = set(int(x) for x in split["calibration_indices"] + split["evaluation_indices"])
    absent_indices = sorted(split_indices - record_indices)
    if absent_indices:
        raise SystemExit(f"split contains indices absent from records: {absent_indices[:10]}")
    records_by_idx = {int(row["idx"]): row for row in records}
    calibration_docs = {
        str(records_by_idx[idx].get("doc_id")) for idx in split["calibration_indices"]
    }
    evaluation_docs = {
        str(records_by_idx[idx].get("doc_id")) for idx in split["evaluation_indices"]
    }
    if calibration_docs & evaluation_docs:
        raise SystemExit("document leakage implied by split indices and records")
    required = [
        Path(args.records),
        Path(args.gold_dir) / "gold.npy",
        Path(args.gold_dir) / "manifest.json",
        Path(args.decoys_dir) / args.run,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("missing required inputs:\n" + "\n".join(missing))
    gold = np.load(Path(args.gold_dir) / "gold.npy", mmap_mode="r")
    manifest = read_json(Path(args.gold_dir) / "manifest.json")
    if gold.ndim != 2 or gold.shape[0] != len(manifest):
        raise SystemExit(f"gold/manifest mismatch: {gold.shape} versus {len(manifest)}")
    if max(split_indices) >= gold.shape[0]:
        raise SystemExit("split contains an index beyond gold.npy")
    tiling_failures = []
    for row in records:
        if row.get("status") != "ok" or int(row["idx"]) not in split_indices:
            continue
        try:
            _assert_claim_tiling(
                str(row["z_inner"]),
                list(row.get("snippet_results") or []),
                list(row.get("connective_results") or []),
            )
        except ValueError as error:
            tiling_failures.append((int(row["idx"]), str(error)))
    if tiling_failures:
        raise SystemExit(f"claim segmentation failures: {tiling_failures[:5]}")
    decoy_files = list((Path(args.decoys_dir) / args.run).glob(f"*_*.{args.decoy_model}.json"))
    if not decoy_files:
        raise SystemExit("no matching decoy caches")
    for path in decoy_files:
        if read_json(path).get(
            "text_model_correction_amendment_sha256"
        ) != TEXT_MODEL_CORRECTION_AMENDMENT_SHA256:
            raise SystemExit(f"decoy cache predates the Terra correction: {path}")
    decoy_keys = {
        tuple(int(value) for value in path.name.split(".", 1)[0].split("_"))
        for path in decoy_files
    }
    claim_sample = read_json(args.claim_sample)
    if claim_sample.get("protocol_sha256") != PROTOCOL_SHA256:
        raise SystemExit("claim sample preregistration hash mismatch")
    if claim_sample.get(
        "text_model_correction_amendment_sha256"
    ) != TEXT_MODEL_CORRECTION_AMENDMENT_SHA256:
        raise SystemExit("claim sample predates the Terra correction")
    required_decoy_keys = {
        (int(row["idx"]), int(row["claim_i"]))
        for row in claim_sample.get("rows") or []
        if int(row["idx"]) in split_indices
    }
    missing_decoys = sorted(required_decoy_keys - decoy_keys)
    if missing_decoys:
        raise SystemExit(
            f"missing decoy caches for {len(missing_decoys)} located claims; "
            f"first keys: {missing_decoys[:10]}"
        )
    reference = None
    if args.reference_vector:
        reference_value = np.asarray(np.load(args.reference_vector), dtype=np.float64)
        reference = {
            "shape": list(reference_value.shape),
            "norm": float(np.linalg.norm(reference_value)),
        }
        if reference_value.shape != (gold.shape[1],) or abs(reference["norm"] - 1.0) > 1e-4:
            raise SystemExit(f"reference vector is not unit [{gold.shape[1]}]: {reference}")
    cuda = None
    if args.check_cuda:
        import torch

        cuda = {"available": torch.cuda.is_available(), "device_count": torch.cuda.device_count()}
        if not cuda["available"]:
            raise SystemExit("CUDA is not available")
    print(json.dumps({
        "status": "ready",
        "report_sha256": REPORT_SHA256,
        "reconstruction_metric_amendment_sha256": (
            RECONSTRUCTION_METRIC_AMENDMENT_SHA256
        ),
        "reconstruction_error_formula": RECONSTRUCTION_ERROR_FORMULA,
        "text_model_correction_amendment_sha256": (
            TEXT_MODEL_CORRECTION_AMENDMENT_SHA256
        ),
        "n_calibration": len(split["calibration_indices"]),
        "n_evaluation": len(split["evaluation_indices"]),
        "n_decoy_caches": len(decoy_files),
        "n_required_decoy_caches": len(required_decoy_keys),
        "n_tiling_failures": 0,
        "calibration_frozen": calibration is not None,
        "reference": reference,
        "cuda": cuda,
    }, indent=2))


def cmd_make_reference(args: argparse.Namespace) -> None:
    """Materialize one explicit h_ref operationalization for preregistration."""
    split = read_json(args.split)
    validate_split(split)
    indices = [int(x) for x in split["calibration_indices"]]
    gold = np.load(Path(args.gold_dir) / "gold.npy", mmap_mode="r")
    if max(indices) >= gold.shape[0]:
        raise ValueError("calibration split index exceeds gold rows")
    mean = np.asarray(gold[indices], dtype=np.float64).mean(axis=0)
    norm = float(np.linalg.norm(mean))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("calibration mean is degenerate")
    reference = (mean / norm).astype(np.float32)
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp.npy")
    np.save(temporary, reference)
    temporary.replace(target)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "report_sha256": REPORT_SHA256,
        "kind": "unit_normalized_mean_of_calibration_activations",
        "n_activations": len(indices),
        "split": str(Path(args.split).resolve()),
        "split_sha256": hashlib.sha256(Path(args.split).read_bytes()).hexdigest(),
        "sha256": hashlib.sha256(reference.tobytes()).hexdigest(),
    }
    atomic_json(target.with_suffix(".json"), metadata)
    print(json.dumps({"out": str(target), **metadata}, indent=2))


def cmd_combine(args: argparse.Namespace) -> None:
    rows = []
    for source in args.verdicts:
        rows.extend(measurement_rows(source))
    models = sorted({str(row.get("model_key") or "unknown") for row in rows})
    output = {
        "schema_version": SCHEMA_VERSION,
        "report_sha256": REPORT_SHA256,
        "reconstruction_metric_amendment_sha256": (
            RECONSTRUCTION_METRIC_AMENDMENT_SHA256
        ),
        "reconstruction_error_formula": RECONSTRUCTION_ERROR_FORMULA,
        "text_model_correction_amendment_sha256": (
            TEXT_MODEL_CORRECTION_AMENDMENT_SHA256
        ),
        "n_models": len(models),
        "by_model": {
            model: stratified_summary([row for row in rows if str(row.get("model_key") or "unknown") == model])
            for model in models
        },
    }
    atomic_json(args.out, output)
    print(json.dumps({"out": str(args.out), "models": models}, indent=2))


def cmd_clustered_ci(args: argparse.Namespace) -> None:
    rows = measurement_rows(args.verdicts)
    if args.kind == "fineweb":
        result = clustered_intervals(
            rows,
            cluster_key="doc_id",
            replicates=args.replicates,
            seed=args.seed,
        )
    else:
        result = ood_macro_intervals(
            rows,
            replicates=args.replicates,
            seed=args.seed,
        )
    result.update({
        "schema_version": SCHEMA_VERSION,
        "protocol_sha256": PROTOCOL_SHA256,
        "report_sha256": REPORT_SHA256,
        "reconstruction_metric_amendment_sha256": (
            RECONSTRUCTION_METRIC_AMENDMENT_SHA256
        ),
        "reconstruction_error_formula": RECONSTRUCTION_ERROR_FORMULA,
        "text_model_correction_amendment_sha256": (
            TEXT_MODEL_CORRECTION_AMENDMENT_SHA256
        ),
        "claim_iid_intervals_used": False,
    })
    atomic_json(args.out, result)
    print(json.dumps({"out": str(args.out), **result["point_percentages"]}, indent=2))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)

    split = sub.add_parser("make-split", help="create a document-level calibration/evaluation split")
    split.add_argument("--records", required=True)
    split.add_argument("--out", required=True)
    split.add_argument("--calibration-fraction", type=float, default=0.2)
    split.add_argument("--seed", type=int, default=0)
    split.set_defaults(func=cmd_split)

    evaluation_split = sub.add_parser(
        "make-evaluation-split",
        help="place every OOD record in evaluation under an external FineWeb calibration",
    )
    evaluation_split.add_argument("--records", required=True)
    evaluation_split.add_argument("--out", required=True)
    evaluation_split.set_defaults(func=cmd_evaluation_split)

    claim_sample = sub.add_parser("make-claim-sample", help="freeze uniform per-activation claim sampling")
    claim_sample.add_argument("--records", required=True)
    claim_sample.add_argument("--split", required=True)
    claim_sample.add_argument("--out", required=True)
    claim_sample.add_argument("--seed", type=int, default=20260827)
    claim_sample.set_defaults(func=cmd_claim_sample)

    negative_map = sub.add_parser(
        "make-negative-map", help="freeze the FineWeb calibration activation derangement"
    )
    negative_map.add_argument("--records", required=True)
    negative_map.add_argument("--split", required=True)
    negative_map.add_argument("--out", required=True)
    negative_map.add_argument("--seed", type=int, default=20260827)
    negative_map.set_defaults(func=cmd_negative_map)

    validate = sub.add_parser("validate-calibration", help="fail unless every report threshold is frozen")
    validate.add_argument("--calibration", required=True)
    validate.set_defaults(func=cmd_validate_calibration)

    audit = sub.add_parser("calibration-audit", help="summarize calibration measurements without choosing thresholds")
    audit.add_argument("--measurements", required=True)
    audit.add_argument("--out", required=True)
    audit.set_defaults(func=cmd_calibration_audit)

    freeze = sub.add_parser("freeze-calibration", help="freeze thresholds from FineWeb shuffled controls")
    freeze.add_argument("--measurements", required=True)
    freeze.add_argument("--model-key", required=True)
    freeze.add_argument("--run-id", required=True)
    freeze.add_argument("--split", required=True)
    freeze.add_argument("--reference", required=True)
    freeze.add_argument("--negative-control-sha256")
    freeze.add_argument("--out", required=True)
    freeze.add_argument("--audit")
    freeze.set_defaults(func=cmd_freeze_calibration)

    classify = sub.add_parser("classify", help="apply the frozen report gates and write percentages")
    classify.add_argument("--measurements", required=True)
    classify.add_argument("--calibration", required=True)
    classify.add_argument("--out", required=True)
    classify.add_argument("--summary")
    classify.set_defaults(func=cmd_classify)

    preflight = sub.add_parser("preflight", help="validate artifacts before requesting GPU time")
    preflight.add_argument("--records", required=True)
    preflight.add_argument("--run", required=True)
    preflight.add_argument("--gold-dir", required=True)
    preflight.add_argument("--decoys-dir", required=True)
    preflight.add_argument("--decoy-model", default="terra")
    preflight.add_argument("--split", required=True)
    preflight.add_argument("--claim-sample", required=True)
    preflight.add_argument("--calibration")
    preflight.add_argument("--reference-vector")
    preflight.add_argument("--check-cuda", action="store_true")
    preflight.set_defaults(func=cmd_preflight)

    reference = sub.add_parser(
        "make-reference",
        help="explicitly choose h_ref as the unit-normalized calibration-activation mean",
    )
    reference.add_argument("--gold-dir", required=True)
    reference.add_argument("--split", required=True)
    reference.add_argument("--out", required=True)
    reference.set_defaults(func=cmd_make_reference)

    combine = sub.add_parser("combine", help="combine per-model verdict files into one comparison artifact")
    combine.add_argument("--verdicts", required=True, nargs="+")
    combine.add_argument("--out", required=True)
    combine.set_defaults(func=cmd_combine)

    intervals = sub.add_parser("clustered-ci", help="preregistered clustered confidence intervals")
    intervals.add_argument("--verdicts", required=True)
    intervals.add_argument("--kind", choices=("fineweb", "ood-macro"), required=True)
    intervals.add_argument("--replicates", type=int, default=10_000)
    intervals.add_argument("--seed", type=int, default=20_260_827)
    intervals.add_argument("--out", required=True)
    intervals.set_defaults(func=cmd_clustered_ci)
    return root


def main() -> None:
    args = parser().parse_args()
    try:
        args.func(args)
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(str(error)) from None


if __name__ == "__main__":
    main()
