"""Prepare Terra-segmented records, decoy caches, and blinded audit subsets."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

from .io import atomic_json, atomic_jsonl, read_json, read_jsonl, sha256_file
from .openai_text import TerraResponses
from .protocol import load_protocol
from .spec import PROTOCOL_SHA256, TEXT_MODEL_CORRECTION_AMENDMENT_SHA256
from .text_pipeline import (
    classify_claim_text,
    generate_decoys,
    judge_source_absence,
    judge_splice_naturalness,
    segment_explanation,
)


PROMPT_VERSION = "terra-v8"


def _rank(seed: int, namespace: str, value: str) -> str:
    return hashlib.sha256(f"{seed}:{namespace}:{value}".encode()).hexdigest()


def _assistant_text(read: dict[str, Any]) -> str:
    continuation = str(read.get("continuation") or "")
    opening = continuation.find("<explanation>")
    if opening < 0:
        return ""
    closing = continuation.find("</explanation>", opening)
    if closing < 0:
        return continuation[opening:].strip()
    return continuation[opening : closing + len("</explanation>")].strip()


def select_fineweb_audit_indices(
    manifest: list[dict[str, Any]], *, seed: int, count: int
) -> list[int]:
    eligible = [
        (int(row["idx"]), str(row["doc_id"]))
        for row in manifest
        if row.get("in_range")
    ]
    eligible.sort(key=lambda row: _rank(seed, "fineweb-segmentation-audit", f"{row[1]}:{row[0]}"))
    if len(eligible) < count:
        raise ValueError(f"only {len(eligible)} eligible activations for audit count {count}")
    return sorted(index for index, _ in eligible[:count])


def _segment_one(
    *,
    idx: int,
    manifest_row: dict[str, Any],
    read_path: Path,
    output_path: Path,
    api_cache_dir: Path,
    dotenv_path: Path,
    compatible_av_protocol_sha256: str,
    record_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    read = read_json(read_path)
    source_av_protocol_sha256 = str(read.get("protocol_sha256") or "")
    if source_av_protocol_sha256 not in {PROTOCOL_SHA256, compatible_av_protocol_sha256}:
        raise ValueError(f"AV read {read_path} has a different protocol hash")
    if int(read["idx"]) != idx or str(read["doc_id"]) != str(manifest_row["doc_id"]):
        raise ValueError(f"AV read/manifest identity mismatch for {idx}")
    explanation = str(read.get("explanation") or "")
    assistant_text = _assistant_text(read)
    if read.get("parse_status") == "missing_opening_tag" or not assistant_text:
        record = {
            "schema_version": 1,
            "protocol_sha256": PROTOCOL_SHA256,
            "text_model_correction_amendment_sha256": (
                TEXT_MODEL_CORRECTION_AMENDMENT_SHA256
            ),
            "idx": idx,
            "doc_id": str(manifest_row["doc_id"]),
            "status": "unresolved",
            "reason": "AV generation missing opening explanation tag",
            "explanation": assistant_text,
            "z_inner": explanation,
            "snippet_results": [],
            "connective_results": [],
            "av_read_sha256": sha256_file(read_path),
            "source_av_protocol_sha256": source_av_protocol_sha256,
            **(record_extra or {}),
        }
        atomic_json(output_path, record)
        return record
    with TerraResponses(dotenv_path=dotenv_path) as client:
        segments, audit = segment_explanation(
            client,
            explanation,
            cache_path=str(api_cache_dir / f"{idx:06d}.json"),
        )
    claims = []
    connectives = []
    for segment in segments:
        if segment["kind"] == "claim":
            claims.append({
                "text": segment["text"],
                "char_span": segment["char_span"],
                "located": True,
                "scored": True,
                "type": None,
                "subtype": None,
            })
        else:
            connectives.append({
                "text": segment["text"],
                "char_span": segment["char_span"],
            })
    if not claims:
        raise ValueError(f"segmentation for {idx} produced no claims")
    record = {
        "schema_version": 1,
        "protocol_sha256": PROTOCOL_SHA256,
        "text_model_correction_amendment_sha256": (
            TEXT_MODEL_CORRECTION_AMENDMENT_SHA256
        ),
        "idx": idx,
        "doc_id": str(manifest_row["doc_id"]),
        "status": "ok",
        "reason": "",
        "explanation": assistant_text,
        "z_inner": explanation,
        "snippet_results": claims,
        "connective_results": connectives,
        "av_parse_status": read["parse_status"],
        "non_marker_cjk_characters": read.get("non_marker_cjk_characters") or [],
        "av_read_sha256": sha256_file(read_path),
        "source_av_protocol_sha256": source_av_protocol_sha256,
        "segmentation_audit": audit,
        **(record_extra or {}),
    }
    atomic_json(output_path, record)
    return record


def run_segment(args: argparse.Namespace) -> None:
    protocol = load_protocol()
    compatible_av_protocol_sha256 = str(
        protocol["compatible_upstream_artifacts"]["av_generation"]["protocol_sha256"]
    )
    manifest = read_json(args.manifest)
    by_idx = {int(row["idx"]): row for row in manifest}
    selected = select_fineweb_audit_indices(
        manifest, seed=int(protocol["seed"]), count=args.count
    )
    output_dir = Path(args.out)
    records_dir = output_dir / "records"
    api_cache_dir = output_dir / "api" / "segmentation"
    records_dir.mkdir(parents=True, exist_ok=True)
    api_cache_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "selection.json", {
        "schema_version": 1,
        "protocol_sha256": PROTOCOL_SHA256,
        "text_model_correction_amendment_sha256": (
            TEXT_MODEL_CORRECTION_AMENDMENT_SHA256
        ),
        "kind": "fineweb_segmentation_audit_half",
        "count": len(selected),
        "indices": selected,
        "manifest_sha256": sha256_file(args.manifest),
    })
    work = []
    for idx in selected:
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
        work.append((idx, destination, read_path))
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _segment_one,
                idx=idx,
                manifest_row=by_idx[idx],
                read_path=read_path,
                output_path=destination,
                api_cache_dir=api_cache_dir,
                dotenv_path=Path(args.dotenv),
                compatible_av_protocol_sha256=compatible_av_protocol_sha256,
                record_extra=None,
            ): idx
            for idx, destination, read_path in work
        }
        for position, future in enumerate(as_completed(futures), start=1):
            idx = futures[future]
            try:
                future.result()
            except Exception as error:  # retain every failure for the audit gate
                failures.append({"idx": idx, "error": f"{type(error).__name__}: {error}"})
            if position % 10 == 0 or position == len(futures):
                print(f"[segment {position}/{len(futures)}] failures={len(failures)}", flush=True)
    atomic_json(output_dir / "segment_status.json", {
        "protocol_sha256": PROTOCOL_SHA256,
        "text_model_correction_amendment_sha256": (
            TEXT_MODEL_CORRECTION_AMENDMENT_SHA256
        ),
        "selected": len(selected),
        "attempted": len(work),
        "failures": failures,
        "complete_records": len(list(records_dir.glob("*.json"))),
    })
    if failures:
        raise RuntimeError(f"segmentation gate has {len(failures)} failures")


def _load_source_documents(path: str | Path) -> dict[str, str]:
    documents: dict[str, str] = {}
    for row in read_jsonl(path):
        doc_id = str(row["doc_id"])
        if doc_id in documents:
            raise ValueError(f"duplicate source doc_id {doc_id}")
        documents[doc_id] = str(row["text"])
    return documents


def _all_claim_keys(records_dir: Path) -> list[tuple[int, int, str, str]]:
    keys = []
    for path in sorted(records_dir.glob("*.json")):
        record = read_json(path)
        if record.get(
            "text_model_correction_amendment_sha256"
        ) != TEXT_MODEL_CORRECTION_AMENDMENT_SHA256:
            raise ValueError(f"pre-Terra segmented record at {path}")
        if record.get("status") != "ok":
            continue
        for claim_i, claim in enumerate(record.get("snippet_results") or []):
            keys.append((
                int(record["idx"]),
                claim_i,
                str(record["doc_id"]),
                str(claim["text"]),
            ))
    return keys


def select_fineweb_decoy_audit_claims(
    records_dir: Path, *, seed: int, count: int
) -> list[tuple[int, int]]:
    keys = _all_claim_keys(records_dir)
    keys.sort(key=lambda row: _rank(seed, "fineweb-decoy-audit-claim", ":".join(map(str, row))))
    if len(keys) < count:
        raise ValueError(f"only {len(keys)} claims for requested audit count {count}")
    return sorted((idx, claim_i) for idx, claim_i, _, _ in keys[:count])


def _decoy_one(
    *,
    idx: int,
    claim_i: int,
    record_path: Path,
    manifest_row: dict[str, Any],
    source_text: str,
    output_path: Path,
    api_dir: Path,
    dotenv_path: Path,
    compatible_segmentation_protocol_sha256: str,
    source_scope: str = "exact_causal_document_prefix_through_the_activation_token",
) -> dict[str, Any]:
    record = read_json(record_path)
    if str(record.get("protocol_sha256") or "") not in {
        PROTOCOL_SHA256,
        compatible_segmentation_protocol_sha256,
    }:
        raise ValueError(f"segmented record {record_path} has an incompatible protocol hash")
    if record.get(
        "text_model_correction_amendment_sha256"
    ) != TEXT_MODEL_CORRECTION_AMENDMENT_SHA256:
        raise ValueError(f"pre-Terra segmented record at {record_path}")
    source_segmentation_sha256 = sha256_file(record_path)
    claim = record["snippet_results"][claim_i]
    start, end = (int(value) for value in claim["char_span"])
    if record["z_inner"][start:end] != claim["text"]:
        raise ValueError(f"claim span drift for {idx}_{claim_i}")
    prefix_end = int(manifest_row["read_char"])
    causal_source = source_text[:prefix_end]
    with TerraResponses(dotenv_path=dotenv_path) as client:
        taxonomy = classify_claim_text(
            client,
            explanation=record["z_inner"],
            claim=claim["text"],
            cache_path=str(api_dir / f"{idx:06d}_{claim_i}.taxonomy.json"),
        )
        generated = generate_decoys(
            client,
            explanation=record["z_inner"],
            claim=claim["text"],
            claim_class=taxonomy.value["claim_class"],
            subtype=taxonomy.value["subtype"],
            cache_path=str(api_dir / f"{idx:06d}_{claim_i}.generation.json"),
        )
        candidates = list(generated.value["candidates"])
        absence = judge_source_absence(
            client,
            source_text=causal_source,
            candidates=candidates,
            cache_path=str(api_dir / f"{idx:06d}_{claim_i}.absence.json"),
        )
        edited = [record["z_inner"][:start] + text + record["z_inner"][end:] for text in candidates]
        naturalness = judge_splice_naturalness(
            client,
            original_explanation=record["z_inner"],
            edited_explanations=edited,
            cache_path=str(api_dir / f"{idx:06d}_{claim_i}.naturalness.json"),
        )
    absence_by_index = {
        int(row["candidate_index"]): row for row in absence.value["judgments"]
    }
    natural_by_index = {
        int(row["candidate_index"]): row for row in naturalness.value["judgments"]
    }
    existing_claims = {
        str(row["text"]).strip().casefold() for row in record["snippet_results"]
    }
    seen: set[str] = set()
    rows = []
    for candidate_i, text in enumerate(candidates):
        folded = text.strip().casefold()
        kept = bool(folded) and folded not in seen and folded not in existing_claims
        if folded:
            seen.add(folded)
        absence_row = absence_by_index[candidate_i]
        natural_row = natural_by_index[candidate_i]
        rows.append({
            "candidate_index": candidate_i,
            "text": text,
            "kept": kept,
            "reason": "" if kept else "empty_duplicate_or_existing_claim",
            "absence": "absent" if absence_row["label"] == "pass" else absence_row["label"],
            "absence_reason": absence_row["reason"],
            "natural_pass": natural_row["label"] == "pass",
            "natural_label": natural_row["label"],
            "natural_reason": natural_row["reason"],
            "natural_score": {"pass": 5, "uncertain": 3, "fail": 1}[natural_row["label"]],
        })
    output = {
        "schema_version": 1,
        "protocol_sha256": PROTOCOL_SHA256,
        "text_model_correction_amendment_sha256": (
            TEXT_MODEL_CORRECTION_AMENDMENT_SHA256
        ),
        "prompt_version": PROMPT_VERSION,
        "idx": idx,
        "claim_i": claim_i,
        "doc_id": record["doc_id"],
        "source_segmentation_protocol_sha256": record["protocol_sha256"],
        "source_segmentation_sha256": source_segmentation_sha256,
        "z_inner_sha": hashlib.sha256(record["z_inner"].encode()).hexdigest()[:16],
        "claim": {
            "text": claim["text"],
            "type": taxonomy.value["claim_class"],
            "subtype": taxonomy.value["subtype"],
        },
        "taxo": taxonomy.value["subtype"],
        "source_scope": source_scope,
        "source_prefix_characters": len(causal_source),
        "source_prefix_sha256": hashlib.sha256(causal_source.encode()).hexdigest(),
        "candidates": rows,
        "audits": {
            "taxonomy": taxonomy.audit,
            "generation": generated.audit,
            "absence": absence.audit,
            "naturalness": naturalness.audit,
        },
    }
    atomic_json(output_path, output)
    return output


def run_decoys(args: argparse.Namespace) -> None:
    protocol = load_protocol()
    compatible_segmentation_protocol_sha256 = str(
        protocol["compatible_upstream_artifacts"]["segmentation"]["protocol_sha256"]
    )
    records_dir = Path(args.records_dir)
    selected = select_fineweb_decoy_audit_claims(
        records_dir, seed=int(protocol["seed"]), count=args.claims
    )
    manifest = read_json(args.manifest)
    manifest_by_idx = {int(row["idx"]): row for row in manifest}
    source_documents = _load_source_documents(args.source_documents)
    output_dir = Path(args.out)
    caches_dir = output_dir / "caches"
    api_dir = output_dir / "api"
    caches_dir.mkdir(parents=True, exist_ok=True)
    api_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "selection.json", {
        "schema_version": 1,
        "protocol_sha256": PROTOCOL_SHA256,
        "text_model_correction_amendment_sha256": (
            TEXT_MODEL_CORRECTION_AMENDMENT_SHA256
        ),
        "kind": "fineweb_decoy_audit_half",
        "claim_count": len(selected),
        "manual_candidates_per_claim": args.manual_candidates_per_claim,
        "claims": [{"idx": idx, "claim_i": claim_i} for idx, claim_i in selected],
    })
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
            manifest_row = manifest_by_idx[idx]
            doc_id = str(manifest_row["doc_id"])
            futures[executor.submit(
                _decoy_one,
                idx=idx,
                claim_i=claim_i,
                record_path=records_dir / f"{idx:06d}.json",
                manifest_row=manifest_row,
                source_text=source_documents[doc_id],
                output_path=destination,
                api_dir=api_dir,
                dotenv_path=Path(args.dotenv),
                compatible_segmentation_protocol_sha256=compatible_segmentation_protocol_sha256,
                source_scope="exact_causal_document_prefix_through_the_activation_token",
            )] = (idx, claim_i)
        for position, future in enumerate(as_completed(futures), start=1):
            key = futures[future]
            try:
                future.result()
            except Exception as error:
                failures.append({"idx": key[0], "claim_i": key[1], "error": f"{type(error).__name__}: {error}"})
            print(f"[decoys {position}/{len(futures)}] failures={len(failures)}", flush=True)
    atomic_json(output_dir / "decoy_status.json", {
        "protocol_sha256": PROTOCOL_SHA256,
        "text_model_correction_amendment_sha256": (
            TEXT_MODEL_CORRECTION_AMENDMENT_SHA256
        ),
        "selected_claims": len(selected),
        "attempted_this_invocation": len(work),
        "failures_this_invocation": failures,
        "complete_caches": len(list(caches_dir.glob("*.terra.json"))),
    })
    if failures:
        raise RuntimeError(f"decoy audit preparation has {len(failures)} failures")
    write_fineweb_audit_csvs(
        records_dir=records_dir,
        caches_dir=caches_dir,
        selected=selected,
        output_dir=output_dir,
        seed=int(protocol["seed"]),
        candidates_per_claim=args.manual_candidates_per_claim,
    )


def _write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_fineweb_audit_csvs(
    *,
    records_dir: Path,
    caches_dir: Path,
    selected: list[tuple[int, int]],
    output_dir: Path,
    seed: int,
    candidates_per_claim: int,
) -> None:
    segment_rows = []
    for record_path in sorted(records_dir.glob("*.json")):
        record = read_json(record_path)
        if record.get("status") != "ok":
            continue
        rendered = "\n".join(
            f"[{row['char_span'][0]}:{row['char_span'][1]}] {row['text']}"
            for row in record["snippet_results"]
        )
        segment_rows.append({
            "audit_id": hashlib.sha256(f"segment:{record['idx']}".encode()).hexdigest()[:12],
            "explanation": record["z_inner"],
            "proposed_atomic_claims": rendered,
            "review_atomic_claims_acceptable_yes_no": "",
            "review_all_semantic_claims_captured_yes_no": "",
            "review_notes": "",
        })
    _write_csv(
        output_dir / "manual_segmentation_review.csv",
        list(segment_rows[0]) if segment_rows else [],
        segment_rows,
    )

    decoy_rows = []
    answer_rows = []
    for idx, claim_i in selected:
        cache = read_json(caches_dir / f"{idx}_{claim_i}.terra.json")
        candidates = list(cache["candidates"])
        candidates.sort(key=lambda row: _rank(
            seed,
            "fineweb-decoy-audit-candidate",
            f"{idx}:{claim_i}:{row['candidate_index']}:{row['text']}",
        ))
        for candidate in candidates[:candidates_per_claim]:
            audit_id = hashlib.sha256(
                f"decoy:{idx}:{claim_i}:{candidate['candidate_index']}".encode()
            ).hexdigest()[:12]
            decoy_rows.append({
                "audit_id": audit_id,
                "original_claim": cache["claim"]["text"],
                "candidate_decoy": candidate["text"],
                "review_source_absent_pass_fail_uncertain": "",
                "review_splice_natural_pass_fail_uncertain": "",
                "review_notes": "",
            })
            answer_rows.append({
                "audit_id": audit_id,
                "idx": idx,
                "claim_i": claim_i,
                "candidate_index": candidate["candidate_index"],
                "automated_absence": candidate["absence"],
                "automated_natural_label": candidate["natural_label"],
            })
    _write_csv(
        output_dir / "manual_decoy_review.csv",
        list(decoy_rows[0]) if decoy_rows else [],
        decoy_rows,
    )
    atomic_jsonl(output_dir / "manual_decoy_answer_key.jsonl", answer_rows)
    atomic_json(output_dir / "manual_audit_meta.json", {
        "schema_version": 1,
        "protocol_sha256": PROTOCOL_SHA256,
        "text_model_correction_amendment_sha256": (
            TEXT_MODEL_CORRECTION_AMENDMENT_SHA256
        ),
        "blinded": True,
        "fineweb_segmentation_items": len(segment_rows),
        "fineweb_decoy_items": len(decoy_rows),
        "answer_key_sha256": sha256_file(output_dir / "manual_decoy_answer_key.jsonl"),
    })


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)
    segment = sub.add_parser("segment-fineweb-audit")
    segment.add_argument("--av-dir", required=True)
    segment.add_argument("--manifest", required=True)
    segment.add_argument("--out", required=True)
    segment.add_argument("--count", type=int, default=100)
    segment.add_argument("--workers", type=int, default=4)
    segment.add_argument("--dotenv", default=".env")
    segment.add_argument("--force", action="store_true")
    segment.set_defaults(func=run_segment)

    decoys = sub.add_parser("decoys-fineweb-audit")
    decoys.add_argument("--records-dir", required=True)
    decoys.add_argument("--manifest", required=True)
    decoys.add_argument("--source-documents", required=True)
    decoys.add_argument("--out", required=True)
    decoys.add_argument("--claims", type=int, default=25)
    decoys.add_argument("--manual-candidates-per-claim", type=int, default=8)
    decoys.add_argument("--workers", type=int, default=2)
    decoys.add_argument("--dotenv", default=".env")
    decoys.add_argument("--force", action="store_true")
    decoys.set_defaults(func=run_decoys)
    return root


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
