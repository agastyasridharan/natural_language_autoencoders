"""Reorder a stage-3 parquet to an exact reference key order and verify vectors.

Used by the matched plausible warm-start preparation: explanations differ by
condition, but activation rows, provenance, and SFT presentation order must be
identical to the implausible run.  Publication is atomic and idempotent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pyarrow as pa
import pyarrow.parquet as pq

from nla.datagen.sidecar import read_sidecar_local, serialize_sidecar
from nla.schema import sidecar_path_for, wrap_explanation


KEY_COLUMNS = ("doc_id", "n_raw_tokens", "activation_layer")
EXACT_COLUMNS = (*KEY_COLUMNS, "detokenized_text_truncated", "activation_vector")
Key = tuple[str, int, int]


def _keys(table: pa.Table) -> list[Key]:
    cols = [table.column(name).to_pylist() for name in KEY_COLUMNS]
    return [
        (str(doc_id), int(n_raw_tokens), int(layer))
        for doc_id, n_raw_tokens, layer in zip(*cols, strict=True)
    ]


def _unique(keys: Iterable[Key], context: str) -> list[Key]:
    rows = list(keys)
    if len(set(rows)) != len(rows):
        raise RuntimeError(f"{context} contains duplicate activation keys")
    return rows


def _digest(keys: Iterable[Key]) -> str:
    h = hashlib.sha256()
    for key in keys:
        h.update(json.dumps(key, ensure_ascii=False, separators=(",", ":")).encode())
        h.update(b"\n")
    return h.hexdigest()


def _atomic_text(path: Path, value: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    parent_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)


def _assert_exact_payload(candidate: pa.Table, reference: pa.Table, context: str) -> None:
    for column in EXACT_COLUMNS:
        if column not in candidate.column_names or column not in reference.column_names:
            raise RuntimeError(f"{context}: required comparison column {column!r} is absent")
        if not candidate.column(column).equals(reference.column(column)):
            raise RuntimeError(
                f"{context}: {column} differs despite matched activation keys; "
                "refusing a confounded SFT dataset"
            )


def _validate_published(output: Path, reference: pa.Table, reference_keys: list[Key]) -> bool:
    sidecar = sidecar_path_for(output)
    if not output.exists() and not sidecar.exists():
        return False
    if not output.exists() and sidecar.is_file():
        print(f"[match-order] found sidecar-only interrupted publication; rebuilding {output}")
        return False
    if not output.is_file() or not sidecar.is_file():
        raise RuntimeError(
            f"partial publication: output={output.exists()} sidecar={sidecar.exists()}"
        )
    candidate = pq.read_table(output)
    if _keys(candidate) != reference_keys:
        raise RuntimeError(f"existing output {output} has the wrong key order")
    _assert_exact_payload(candidate, reference, str(output))
    meta = read_sidecar_local(output)
    if meta.row_count != len(reference_keys):
        raise RuntimeError(f"existing {output} sidecar row count is stale")
    print(f"[match-order] verified existing complete output: {output}")
    return True


def _assert_stage3_matches_teacher(
    source: pa.Table, source_path: Path, teacher_path: Path, stage: str
) -> None:
    teacher = pq.read_table(teacher_path, columns=[*KEY_COLUMNS, "api_explanation"])
    teacher_keys = _unique(_keys(teacher), str(teacher_path))
    source_keys = _unique(_keys(source), str(source_path))
    if source_keys != teacher_keys:
        raise RuntimeError(
            f"{source_path} is not in the exact key order of its matched teacher {teacher_path}"
        )
    explanations = teacher.column("api_explanation").to_pylist()
    if stage == "av_sft":
        expected = [wrap_explanation(str(value)) for value in explanations]
        actual = source.column("response").to_pylist()
    else:
        meta = read_sidecar_local(source_path)
        template = meta.prompt_templates.get("critic")
        if not template:
            raise RuntimeError(f"{source_path} lacks the critic prompt template in its sidecar")
        expected = [template.format(explanation=str(value)) for value in explanations]
        actual = source.column("prompt").to_pylist()
    if actual != expected:
        for i, (got, want) in enumerate(zip(actual, expected, strict=True)):
            if got != want:
                raise RuntimeError(
                    f"{source_path} does not encode its matched teacher at row {i}, "
                    f"key={source_keys[i]}"
                )
        raise RuntimeError(f"{source_path} teacher payload mismatch")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="unshuffled plausible stage-3 parquet")
    parser.add_argument("--teacher", required=True, help="matched plausible explained parquet")
    parser.add_argument("--stage", required=True, choices=("av_sft", "ar_sft"))
    parser.add_argument("--reference", required=True, help="implausible shuffled stage-3 parquet")
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-rows", required=True, type=int)
    args = parser.parse_args()

    input_path = Path(args.input)
    teacher_path = Path(args.teacher)
    reference_path = Path(args.reference)
    output = Path(args.output)
    for path in (
        input_path,
        sidecar_path_for(input_path),
        teacher_path,
        sidecar_path_for(teacher_path),
        reference_path,
        sidecar_path_for(reference_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    reference = pq.read_table(reference_path)
    reference_keys = _unique(_keys(reference), str(reference_path))
    if len(reference_keys) != args.expected_rows:
        raise RuntimeError(
            f"reference has {len(reference_keys)} rows, expected {args.expected_rows}"
        )
    source = pq.read_table(input_path)
    _assert_stage3_matches_teacher(source, input_path, teacher_path, args.stage)
    source_keys = _unique(_keys(source), str(input_path))
    if set(source_keys) != set(reference_keys):
        missing = list(set(reference_keys) - set(source_keys))[:5]
        extra = list(set(source_keys) - set(reference_keys))[:5]
        raise RuntimeError(
            f"input/reference key sets differ: input={len(source_keys)} "
            f"reference={len(reference_keys)} missing_examples={missing} extra_examples={extra}"
        )
    if _validate_published(output, reference, reference_keys):
        return
    source_index = {key: i for i, key in enumerate(source_keys)}
    order = pa.array([source_index[key] for key in reference_keys], type=pa.int64())
    reordered = source.take(order)
    _assert_exact_payload(reordered, reference, "reordered plausible/reference")

    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = output.with_name(output.name + ".tmp")
    pq.write_table(reordered, tmp_output, row_group_size=65536)
    with tmp_output.open("rb") as handle:
        os.fsync(handle.fileno())

    source_meta = read_sidecar_local(input_path)
    reference_meta = read_sidecar_local(reference_path)
    out_meta = replace(
        source_meta,
        dataset_id=f"{source_meta.dataset_id}__matched_implausible_order",
        row_count=len(reference_keys),
        parent_datasets=[source_meta.dataset_id, reference_meta.dataset_id],
        created_by="scripts.datagen.reorder_to_reference",
        created_at="",
        git_commit="",
    )
    # Sidecar first, parquet last: the parquet is the publication marker.
    _atomic_text(sidecar_path_for(output), serialize_sidecar(out_meta))
    os.replace(tmp_output, output)
    output_fd = os.open(output.parent, os.O_RDONLY)
    try:
        os.fsync(output_fd)
    finally:
        os.close(output_fd)
    manifest = {
        "schema_version": 1,
        "input": str(input_path.resolve()),
        "reference": str(reference_path.resolve()),
        "output": str(output.resolve()),
        "row_count": len(reference_keys),
        "ordered_key_sha256": _digest(reference_keys),
        "exact_columns_verified": list(EXACT_COLUMNS),
        "published_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_text(
        Path(f"{output}.match_manifest.json"),
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    print(
        f"[match-order] published {len(reference_keys)} rows in exact reference order -> {output}"
    )


if __name__ == "__main__":
    main()
