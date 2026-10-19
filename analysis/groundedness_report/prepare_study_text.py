"""Prepare OOD audits and gated full-study Terra text artifacts.

The audit commands produce blinded 100-row segmentation and 200-row decoy
review sheets for the OOD half.  Full segmentation and sampled-claim decoy
generation refuse to start until both the FineWeb and OOD manual audit result
files pass every preregistered gate.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

from .io import atomic_json, atomic_jsonl, read_json, read_jsonl, sha256_file
from .prepare_text import PROMPT_VERSION, _decoy_one, _rank, _segment_one
from .protocol import load_evaluation_registry, load_protocol
from .spec import PROTOCOL_SHA256, TEXT_MODEL_CORRECTION_AMENDMENT_SHA256


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty review sheet to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _balanced_allocation(
    keys: Iterable[str], *, total: int, seed: int, namespace: str
) -> dict[str, int]:
    ordered = sorted(set(str(key) for key in keys))
    if not ordered or total < len(ordered):
        raise ValueError("balanced allocation requires at least one item per family")
    base, remainder = divmod(total, len(ordered))
    extra = set(sorted(ordered, key=lambda key: _rank(seed, namespace, key))[:remainder])
    return {key: base + int(key in extra) for key in ordered}


def _manifest(path: str | Path) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    rows = list(read_json(path))
    by_idx = {int(row["idx"]): row for row in rows}
    if len(rows) != len(by_idx):
        raise ValueError("manifest contains duplicate activation indices")
    return rows, by_idx


def _ood_audit_indices(
    manifest: list[dict[str, Any]], *, seed: int, count: int
) -> list[int]:
    registry = load_evaluation_registry()
    families = [str(row["id"]) for row in registry["families"]]
    allocation = _balanced_allocation(
        families, total=count, seed=seed, namespace="ood-segmentation-audit-allocation"
    )
    by_family: dict[str, list[dict[str, Any]]] = {family: [] for family in families}
    for row in manifest:
        family = str(row.get("family_id") or "")
        if family in by_family:
            by_family[family].append(row)
    selected: list[int] = []
    for family in families:
        # Prefer distinct transcript clusters.  Each selected transcript
        # contributes one deterministically chosen activation to the audit.
        by_trajectory: dict[str, list[dict[str, Any]]] = {}
        for row in by_family[family]:
            by_trajectory.setdefault(str(row.get("trajectory_id") or row.get("doc_id")), []).append(row)
        trajectory_rows = []
        for trajectory_id, values in by_trajectory.items():
            chosen = min(
                values,
                key=lambda row: _rank(seed, "ood-segmentation-audit-position", f"{trajectory_id}:{row['idx']}"),
            )
            trajectory_rows.append(chosen)
        trajectory_rows.sort(
            key=lambda row: _rank(
                seed,
                "ood-segmentation-audit-trajectory",
                f"{family}:{row.get('trajectory_id') or row.get('doc_id')}",
            )
        )
        required = allocation[family]
        if len(trajectory_rows) < required:
            raise ValueError(
                f"family {family} has {len(trajectory_rows)} trajectories; require {required}"
            )
        selected.extend(int(row["idx"]) for row in trajectory_rows[:required])
    if len(selected) != count or len(set(selected)) != count:
        raise AssertionError("OOD audit selection is not the requested unique size")
    return sorted(selected)


def _record_extra(row: dict[str, Any], *, kind: str) -> dict[str, Any]:
    if kind == "fineweb":
        return {}
    family = str(row.get("family_id") or "")
    trajectory = str(row.get("trajectory_id") or row.get("doc_id") or "")
    if not family or not trajectory:
        raise ValueError("OOD manifest row lacks family_id or trajectory_id")
    return {
        "trajectory_id": trajectory,
        "evaluation_family": family,
        "task_id": row.get("task_id"),
        "source_scope": "exact_serialized_target_model_context_through_the_activation_token",
    }


def _validated_audit_gate(paths: list[str]) -> dict[str, str]:
    if len(paths) != 2:
        raise ValueError("full expansion requires exactly two --audit-result files")
    observed: dict[str, str] = {}
    for path in paths:
        result = read_json(path)
        if result.get("protocol_sha256") != PROTOCOL_SHA256:
            raise ValueError(f"manual audit result {path} has a stale protocol hash")
        if result.get(
            "text_model_correction_amendment_sha256"
        ) != TEXT_MODEL_CORRECTION_AMENDMENT_SHA256:
            raise ValueError(f"manual audit result {path} predates the Terra correction")
        half = str(result.get("audit_half") or "")
        if half not in {"fineweb", "ood"} or half in observed:
            raise ValueError("manual audit results must contain one fineweb and one ood half")
        if not result.get("all_gates_pass"):
            raise ValueError(f"manual {half} audit did not pass every gate")
        observed[half] = sha256_file(path)
    if set(observed) != {"fineweb", "ood"}:
        raise ValueError("manual audit results must contain both halves")
    return observed


def _segment(
    args: argparse.Namespace, *, selected: list[int], audit_gate: dict[str, str] | None
) -> None:
    protocol = load_protocol()
    _, by_idx = _manifest(args.manifest)
    output = Path(args.out)
    records_dir = output / "records"
    api_dir = output / "api" / "segmentation"
    records_dir.mkdir(parents=True, exist_ok=True)
    api_dir.mkdir(parents=True, exist_ok=True)
    compatible = (
        str(protocol["compatible_upstream_artifacts"]["av_generation"]["protocol_sha256"])
        if args.kind == "fineweb"
        else PROTOCOL_SHA256
    )
    atomic_json(output / "selection.json", {
        "schema_version": 1,
        "protocol_sha256": PROTOCOL_SHA256,
        "text_model_correction_amendment_sha256": (
            TEXT_MODEL_CORRECTION_AMENDMENT_SHA256
        ),
        "kind": f"{args.kind}_{'audit' if audit_gate is None else 'full'}_segmentation",
        "indices": selected,
        "manifest_sha256": sha256_file(args.manifest),
        "manual_audit_results": audit_gate,
    })
    work = []
    for idx in selected:
        if idx not in by_idx:
            raise ValueError(f"selected activation {idx} is absent from manifest")
        destination = records_dir / f"{idx:06d}.json"
        if destination.exists() and not args.force:
            cached = read_json(destination)
            if cached.get("protocol_sha256") != PROTOCOL_SHA256:
                raise ValueError(f"stale segmented record at {destination}")
            if cached.get(
                "text_model_correction_amendment_sha256"
            ) != TEXT_MODEL_CORRECTION_AMENDMENT_SHA256:
                raise ValueError(f"pre-Terra segmented record at {destination}")
            continue
        read_path = Path(args.av_dir) / "reads" / f"{idx:06d}.json"
        if not read_path.exists():
            raise ValueError(f"missing AV read {read_path}")
        work.append((idx, read_path, destination))
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _segment_one,
                idx=idx,
                manifest_row=by_idx[idx],
                read_path=read_path,
                output_path=destination,
                api_cache_dir=api_dir,
                dotenv_path=Path(args.dotenv),
                compatible_av_protocol_sha256=compatible,
                record_extra=_record_extra(by_idx[idx], kind=args.kind),
            ): idx
            for idx, read_path, destination in work
        }
        for position, future in enumerate(as_completed(futures), start=1):
            idx = futures[future]
            try:
                future.result()
            except Exception as error:
                failures.append({"idx": idx, "error": f"{type(error).__name__}: {error}"})
            if position % 10 == 0 or position == len(futures):
                print(f"[segment {position}/{len(futures)}] failures={len(failures)}", flush=True)
    atomic_json(output / "segment_status.json", {
        "protocol_sha256": PROTOCOL_SHA256,
        "text_model_correction_amendment_sha256": (
            TEXT_MODEL_CORRECTION_AMENDMENT_SHA256
        ),
        "selected": len(selected),
        "attempted": len(work),
        "complete_records": len(list(records_dir.glob("*.json"))),
        "failures": failures,
    })
    if failures:
        raise RuntimeError(f"segmentation has {len(failures)} failures")


def run_segment_ood_audit(args: argparse.Namespace) -> None:
    protocol = load_protocol()
    manifest, _ = _manifest(args.manifest)
    selected = _ood_audit_indices(
        manifest, seed=int(protocol["seed"]), count=args.count
    )
    args.kind = "ood"
    _segment(args, selected=selected, audit_gate=None)
    records_dir = Path(args.out) / "records"
    rows = []
    for idx in selected:
        record = read_json(records_dir / f"{idx:06d}.json")
        rendered = "\n".join(
            f"[{claim['char_span'][0]}:{claim['char_span'][1]}] {claim['text']}"
            for claim in record.get("snippet_results") or []
        )
        rows.append({
            "audit_id": hashlib.sha256(f"ood-segment:{idx}".encode()).hexdigest()[:12],
            "evaluation_family": record.get("evaluation_family"),
            "explanation": record.get("z_inner"),
            "proposed_atomic_claims": rendered,
            "review_atomic_claims_acceptable_yes_no": "",
            "review_all_semantic_claims_captured_yes_no": "",
            "review_notes": "",
        })
    _write_csv(Path(args.out) / "manual_segmentation_review.csv", rows)


def run_segment_all(args: argparse.Namespace) -> None:
    gate = _validated_audit_gate(args.audit_result)
    manifest, _ = _manifest(args.manifest)
    selected = sorted(int(row["idx"]) for row in manifest)
    _segment(args, selected=selected, audit_gate=gate)


def _claims_by_family(records_dir: Path) -> dict[str, list[tuple[int, int]]]:
    result: dict[str, list[tuple[int, int]]] = {}
    for path in sorted(records_dir.glob("*.json")):
        record = read_json(path)
        if record.get("status") != "ok":
            continue
        family = str(record.get("evaluation_family") or "")
        if not family:
            raise ValueError(f"OOD audit record {path} has no evaluation_family")
        for claim_i, _ in enumerate(record.get("snippet_results") or []):
            result.setdefault(family, []).append((int(record["idx"]), claim_i))
    return result


def _ood_audit_claims(
    records_dir: Path, *, seed: int, count: int
) -> list[tuple[int, int]]:
    by_family = _claims_by_family(records_dir)
    expected = [str(row["id"]) for row in load_evaluation_registry()["families"]]
    allocation = _balanced_allocation(
        expected, total=count, seed=seed, namespace="ood-decoy-audit-allocation"
    )
    selected = []
    for family in expected:
        values = sorted(
            by_family.get(family) or [],
            key=lambda key: _rank(seed, "ood-decoy-audit-claim", f"{family}:{key[0]}:{key[1]}"),
        )
        if len(values) < allocation[family]:
            raise ValueError(
                f"family {family} has {len(values)} audit claims; require {allocation[family]}"
            )
        selected.extend(values[: allocation[family]])
    return sorted(selected)


def _causal_source(kind: str, row: dict[str, Any], documents: dict[str, str] | None) -> tuple[str, str]:
    if kind == "ood":
        source = str(row.get("causal_context") or "")
        if not source:
            raise ValueError(f"OOD manifest row {row.get('idx')} has no causal_context")
        return source, "exact_serialized_target_model_context_through_the_activation_token"
    if documents is None:
        raise ValueError("FineWeb decoys require source documents")
    doc_id = str(row["doc_id"])
    if doc_id not in documents:
        raise ValueError(f"FineWeb source document {doc_id} is missing")
    return documents[doc_id][: int(row["read_char"])], "exact_causal_document_prefix_through_the_activation_token"


def _documents(path: str | None) -> dict[str, str] | None:
    if not path:
        return None
    result = {}
    for row in read_jsonl(path):
        doc_id = str(row["doc_id"])
        if doc_id in result:
            raise ValueError(f"duplicate source document {doc_id}")
        result[doc_id] = str(row["text"])
    return result


def _run_decoys(
    args: argparse.Namespace,
    *,
    selected: list[tuple[int, int]],
    caches_dir: Path,
    audit_gate: dict[str, str] | None,
) -> None:
    protocol = load_protocol()
    _, manifest = _manifest(args.manifest)
    documents = _documents(args.source_documents)
    records_dir = Path(args.records_dir)
    api_dir = Path(args.out) / "api" / str(getattr(args, "run", "audit"))
    caches_dir.mkdir(parents=True, exist_ok=True)
    api_dir.mkdir(parents=True, exist_ok=True)
    compatible = str(protocol["compatible_upstream_artifacts"]["segmentation"]["protocol_sha256"])
    work = []
    for idx, claim_i in selected:
        destination = caches_dir / f"{idx}_{claim_i}.terra.json"
        if destination.exists() and not args.force:
            cached = read_json(destination)
            if cached.get("protocol_sha256") != PROTOCOL_SHA256:
                raise ValueError(f"stale decoy cache at {destination}")
            if cached.get(
                "text_model_correction_amendment_sha256"
            ) != TEXT_MODEL_CORRECTION_AMENDMENT_SHA256:
                raise ValueError(f"pre-Terra decoy cache at {destination}")
            continue
        work.append((idx, claim_i, destination))
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for idx, claim_i, destination in work:
            source, scope = _causal_source(args.kind, manifest[idx], documents)
            futures[executor.submit(
                _decoy_one,
                idx=idx,
                claim_i=claim_i,
                record_path=records_dir / f"{idx:06d}.json",
                manifest_row=manifest[idx],
                source_text=source,
                output_path=destination,
                api_dir=api_dir,
                dotenv_path=Path(args.dotenv),
                compatible_segmentation_protocol_sha256=compatible,
                source_scope=scope,
            )] = (idx, claim_i)
        for position, future in enumerate(as_completed(futures), start=1):
            key = futures[future]
            try:
                future.result()
            except Exception as error:
                failures.append({
                    "idx": key[0], "claim_i": key[1],
                    "error": f"{type(error).__name__}: {error}",
                })
            print(f"[decoys {position}/{len(futures)}] failures={len(failures)}", flush=True)
    atomic_json(Path(args.out) / "decoy_status.json", {
        "protocol_sha256": PROTOCOL_SHA256,
        "text_model_correction_amendment_sha256": (
            TEXT_MODEL_CORRECTION_AMENDMENT_SHA256
        ),
        "prompt_version": PROMPT_VERSION,
        "selected_claims": len(selected),
        "attempted": len(work),
        "complete_caches": len(list(caches_dir.glob("*.terra.json"))),
        "manual_audit_results": audit_gate,
        "failures": failures,
    })
    if failures:
        raise RuntimeError(f"decoy generation has {len(failures)} failures")


def run_decoys_ood_audit(args: argparse.Namespace) -> None:
    protocol = load_protocol()
    records_dir = Path(args.records_dir)
    selected = _ood_audit_claims(
        records_dir, seed=int(protocol["seed"]), count=args.claims
    )
    args.kind = "ood"
    caches_dir = Path(args.out) / "caches"
    _run_decoys(args, selected=selected, caches_dir=caches_dir, audit_gate=None)
    review_rows = []
    answers = []
    for idx, claim_i in selected:
        cache = read_json(caches_dir / f"{idx}_{claim_i}.terra.json")
        family = str(read_json(records_dir / f"{idx:06d}.json")["evaluation_family"])
        candidates = sorted(
            cache["candidates"],
            key=lambda row: _rank(
                int(protocol["seed"]),
                "ood-decoy-audit-candidate",
                f"{family}:{idx}:{claim_i}:{row['candidate_index']}:{row['text']}",
            ),
        )[: args.manual_candidates_per_claim]
        for candidate in candidates:
            audit_id = hashlib.sha256(
                f"ood-decoy:{idx}:{claim_i}:{candidate['candidate_index']}".encode()
            ).hexdigest()[:12]
            review_rows.append({
                "audit_id": audit_id,
                "evaluation_family": family,
                "original_claim": cache["claim"]["text"],
                "candidate_decoy": candidate["text"],
                "review_source_absent_pass_fail_uncertain": "",
                "review_splice_natural_pass_fail_uncertain": "",
                "review_notes": "",
            })
            answers.append({
                "audit_id": audit_id,
                "evaluation_family": family,
                "idx": idx,
                "claim_i": claim_i,
                "candidate_index": candidate["candidate_index"],
                "automated_absence": candidate["absence"],
                "automated_natural_label": candidate["natural_label"],
            })
    expected = args.claims * args.manual_candidates_per_claim
    if len(review_rows) != expected:
        raise ValueError(f"expected {expected} OOD decoy audit rows, observed {len(review_rows)}")
    output = Path(args.out)
    _write_csv(output / "manual_decoy_review.csv", review_rows)
    answer_path = output / ".answer_keys" / "ood_manual_decoy_answer_key.jsonl"
    atomic_jsonl(answer_path, answers)
    atomic_json(output / "manual_audit_meta.json", {
        "schema_version": 1,
        "protocol_sha256": PROTOCOL_SHA256,
        "text_model_correction_amendment_sha256": (
            TEXT_MODEL_CORRECTION_AMENDMENT_SHA256
        ),
        "blinded": True,
        "ood_segmentation_items": len(list(records_dir.glob("*.json"))),
        "ood_decoy_items": len(review_rows),
        "answer_key_sha256": sha256_file(answer_path),
    })


def run_assemble_records(args: argparse.Namespace) -> None:
    rows = [read_json(path) for path in sorted(Path(args.records_dir).glob("*.json"))]
    if not rows:
        raise ValueError("no segmented records to assemble")
    rows.sort(key=lambda row: int(row["idx"]))
    indices = [int(row["idx"]) for row in rows]
    if len(indices) != len(set(indices)):
        raise ValueError("segmented records contain duplicate indices")
    atomic_jsonl(args.out, rows)
    print(json.dumps({"out": args.out, "records": len(rows)}, indent=2))


def run_decoys_sampled(args: argparse.Namespace) -> None:
    gate = _validated_audit_gate(args.audit_result)
    sample = read_json(args.claim_sample)
    if sample.get("protocol_sha256") != PROTOCOL_SHA256:
        raise ValueError("claim sample has a stale protocol hash")
    if sample.get(
        "text_model_correction_amendment_sha256"
    ) != TEXT_MODEL_CORRECTION_AMENDMENT_SHA256:
        raise ValueError("claim sample predates the Terra correction")
    selected = sorted(
        (int(row["idx"]), int(row["claim_i"]))
        for row in sample.get("rows") or []
    )
    if len(selected) != len(set(selected)):
        raise ValueError("claim sample contains duplicate keys")
    _run_decoys(
        args,
        selected=selected,
        caches_dir=Path(args.out) / args.run,
        audit_gate=gate,
    )


def _segmentation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--av-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dotenv", default=".env")
    parser.add_argument("--force", action="store_true")


def _decoy_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--records-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--source-documents")
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--dotenv", default=".env")
    parser.add_argument("--force", action="store_true")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)
    segment_audit = sub.add_parser("segment-ood-audit")
    _segmentation_arguments(segment_audit)
    segment_audit.add_argument("--count", type=int, default=100)
    segment_audit.set_defaults(func=run_segment_ood_audit)

    decoy_audit = sub.add_parser("decoys-ood-audit")
    _decoy_arguments(decoy_audit)
    decoy_audit.add_argument("--claims", type=int, default=25)
    decoy_audit.add_argument("--manual-candidates-per-claim", type=int, default=8)
    decoy_audit.set_defaults(func=run_decoys_ood_audit)

    segment_all = sub.add_parser("segment-all")
    _segmentation_arguments(segment_all)
    segment_all.add_argument("--kind", choices=("fineweb", "ood"), required=True)
    segment_all.add_argument("--audit-result", action="append", required=True)
    segment_all.set_defaults(func=run_segment_all)

    assemble = sub.add_parser("assemble-records")
    assemble.add_argument("--records-dir", required=True)
    assemble.add_argument("--out", required=True)
    assemble.set_defaults(func=run_assemble_records)

    decoys = sub.add_parser("decoys-sampled")
    _decoy_arguments(decoys)
    decoys.add_argument("--kind", choices=("fineweb", "ood"), required=True)
    decoys.add_argument("--claim-sample", required=True)
    decoys.add_argument("--run", required=True)
    decoys.add_argument("--audit-result", action="append", required=True)
    decoys.set_defaults(func=run_decoys_sampled)
    return root


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
