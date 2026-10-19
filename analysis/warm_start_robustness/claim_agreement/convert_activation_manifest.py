#!/usr/bin/env python3
"""Convert groundedness activation outputs to the claim-agreement manifest.

The converter writes metadata only.  Matrix sources keep pointing at their
existing ``gold.npy``; agentic sources keep pointing at each trajectory's
existing one-vector ``.npy``.  No activation bytes are copied or rewritten.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:  # package invocation
    from .generate_av import (
        ACTIVATION_SCHEMA,
        BASE_REVISION,
        D_MODEL,
        LAYER_INDEX,
        MODEL_ID,
        ContractError,
        atomic_write_json,
        sha256_file,
    )
except ImportError:  # direct ``python convert_activation_manifest.py``
    from generate_av import (  # type: ignore
        ACTIVATION_SCHEMA,
        BASE_REVISION,
        D_MODEL,
        LAYER_INDEX,
        MODEL_ID,
        ContractError,
        atomic_write_json,
        sha256_file,
    )


FAMILY_ID_MAP = {
    "alignment_blackmail": ("alignment", "blackmail"),
    "alignment_destructive_coding": ("alignment", "destructive_coding_actions"),
    "alignment_destructive_coding_actions": ("alignment", "destructive_coding_actions"),
    "alignment_overly_agentic": ("alignment", "overly_agentic_actions"),
    "alignment_overly_agentic_actions": ("alignment", "overly_agentic_actions"),
    "alignment_factual_honesty": ("alignment", "factual_honesty"),
    "alignment_appropriate_harmlessness": ("alignment", "appropriate_harmlessness"),
    "alignment_sycophancy": ("alignment", "sycophancy"),
    "alignment_multiturn_deception": ("alignment", "multi_turn_deception"),
    "alignment_multi_turn_deception": ("alignment", "multi_turn_deception"),
    "capability_swebench_verified": ("capability", "swe_bench_verified"),
    "capability_swe_bench_verified": ("capability", "swe_bench_verified"),
    "capability_gpqa_diamond": ("capability", "gpqa_diamond"),
    "capability_humanitys_last_exam": ("capability", "humanitys_last_exam"),
    "capability_hle": ("capability", "humanitys_last_exam"),
    "capability_tau2_retail": ("capability", "tau2_bench_retail"),
    "capability_tau2_bench_retail": ("capability", "tau2_bench_retail"),
    "capability_taubench_retail": ("capability", "tau2_bench_retail"),
    "capability_mmmlu": ("capability", "mmmlu"),
    "capability_browsecomp": ("capability", "browsecomp"),
    "capability_terminal_bench_2": ("capability", "terminal_bench_2"),
    "capability_terminalbench2": ("capability", "terminal_bench_2"),
    "capability_financeagent": ("capability", "financeagent"),
    "capability_finance_agent": ("capability", "financeagent"),
    "capability_cybergym": ("capability", "cybergym"),
}


def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise ContractError(f"required source file is missing: {path}")
    try:
        return json.loads(path.read_bytes())
    except json.JSONDecodeError as exc:
        raise ContractError(f"invalid JSON in {path}: {exc}") from exc


def _relative(source: Path, output: Path) -> str:
    return os.path.relpath(source.resolve(), start=output.parent.resolve())


def _validate_vector_file(path: Path, expected_sha: str | None = None) -> str:
    if not path.is_file():
        raise ContractError(f"missing activation file: {path}")
    vector = np.load(path, mmap_mode="r", allow_pickle=False)
    if vector.shape not in ((D_MODEL,), (1, D_MODEL)):
        raise ContractError(
            f"{path}: expected one {D_MODEL}-dimensional vector, got {vector.shape}"
        )
    if not np.issubdtype(vector.dtype, np.floating):
        raise ContractError(f"{path}: expected floating dtype, got {vector.dtype}")
    if not np.isfinite(vector).all():
        raise ContractError(f"{path}: vector contains non-finite values")
    digest = sha256_file(path)
    if expected_sha is not None and digest != expected_sha:
        raise ContractError(
            f"{path}: sha256 mismatch; trajectory says {expected_sha}, got {digest}"
        )
    return digest


def _family(family_id: str, context: str) -> tuple[str, str]:
    try:
        return FAMILY_ID_MAP[family_id]
    except KeyError as exc:
        raise ContractError(
            f"{context}: unrecognized family_id={family_id!r}; add an explicit frozen-slug map"
        ) from exc


def _matrix_manifest(source_root: Path, output: Path) -> dict[str, Any]:
    array_path = source_root / "gold.npy"
    source_manifest_path = source_root / "manifest.json"
    rows = _read_json(source_manifest_path)
    if not isinstance(rows, list) or not rows:
        raise ContractError(f"{source_manifest_path}: expected a nonempty JSON array")
    array = np.load(array_path, mmap_mode="r", allow_pickle=False)
    if array.shape != (len(rows), D_MODEL):
        raise ContractError(
            f"matrix/manifest mismatch: {array_path}={array.shape}, rows={len(rows)}"
        )
    if array.dtype != np.float32:
        raise ContractError(f"{array_path}: expected raw float32, got {array.dtype}")
    if not np.isfinite(array).all():
        raise ContractError(f"{array_path}: activation matrix contains non-finite values")
    array_sha = sha256_file(array_path)
    source_manifest_sha = sha256_file(source_manifest_path)

    is_fineweb = all("family_id" not in row for row in rows)
    items: list[dict[str, Any]] = []
    for ordinal, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ContractError(f"{source_manifest_path}: row {ordinal} is not an object")
        index = row.get("idx")
        if index != ordinal:
            raise ContractError(
                f"{source_manifest_path}: row {ordinal} has idx={index!r}; expected {ordinal}"
            )
        if row.get("in_range") is not True:
            raise ContractError(f"{source_manifest_path}: row {ordinal} is not in_range")
        if is_fineweb:
            document_id = row.get("doc_id")
            position = row.get("position")
            if not isinstance(document_id, str) or not isinstance(position, int):
                raise ContractError(
                    f"{source_manifest_path}: FineWeb row {ordinal} lacks doc_id/position"
                )
            activation_id = f"{document_id}:position:{position}"
            item = {
                "activation_id": activation_id,
                "activation_index": ordinal,
                "document_id": document_id,
                "trajectory_id": None,
                "cluster_id": document_id,
                "distribution": "fineweb",
                "benchmark": "fineweb",
                "family": "fineweb",
                "source_manifest_index": ordinal,
                "position": position,
                "n_tokens": row.get("n_tokens"),
                "read_token": row.get("read_token"),
                "source_h_norm": row.get("h_norm"),
            }
        else:
            family_id = row.get("family_id")
            trajectory_id = row.get("trajectory_id")
            position_key = row.get("position_key")
            if (
                not isinstance(family_id, str)
                or not isinstance(trajectory_id, str)
                or not isinstance(position_key, int)
            ):
                raise ContractError(
                    f"{source_manifest_path}: OOD row {ordinal} lacks "
                    "family_id/trajectory_id/position_key"
                )
            family, benchmark = _family(family_id, f"row {ordinal}")
            activation_id = f"{trajectory_id}:position_key:{position_key}"
            if row.get("norm") != "none":
                raise ContractError(
                    f"{source_manifest_path}: row {ordinal} norm={row.get('norm')!r}"
                )
            item = {
                "activation_id": activation_id,
                "activation_index": ordinal,
                "document_id": None,
                "trajectory_id": trajectory_id,
                "cluster_id": trajectory_id,
                "distribution": "ood",
                "benchmark": benchmark,
                "family": family,
                "source_manifest_index": ordinal,
                "family_id": family_id,
                "task_id": row.get("task_id"),
                "position_key": position_key,
                "turn_index": row.get("turn_index"),
                "assistant_response_token_index": row.get(
                    "assistant_response_token_index"
                ),
                "read_token": row.get("read_token"),
                "trajectory_file": row.get("trajectory_file"),
                "source_h_norm": row.get("h_norm"),
            }
        items.append(item)

    metadata_paths = [
        path for path in (source_root / "extract_meta.json", source_root / "assemble_meta.json")
        if path.is_file()
    ]
    metadata = {path.name: _read_json(path) for path in metadata_paths}
    for name, meta in metadata.items():
        if name == "extract_meta.json":
            if (
                meta.get("base") != MODEL_ID
                or meta.get("layer") != LAYER_INDEX
                or meta.get("d_model") != D_MODEL
            ):
                raise ContractError(f"{source_root / name}: target-model metadata mismatch")
        if name == "assemble_meta.json":
            if meta.get("norm") != "none":
                raise ContractError(f"{source_root / name}: norm must be 'none'")
            expected = meta.get("gold_sha256")
            if expected is not None and expected != array_sha:
                raise ContractError(
                    f"{source_root / name}: gold_sha256={expected}, actual={array_sha}"
                )

    return {
        "schema_version": ACTIVATION_SCHEMA,
        "model_id": MODEL_ID,
        "model_revision": BASE_REVISION,
        "layer_index": LAYER_INDEX,
        "d_model": D_MODEL,
        "activation_norm": "none",
        "activation_file": _relative(array_path, output),
        "activation_sha256": array_sha,
        "source_manifest": _relative(source_manifest_path, output),
        "source_manifest_sha256": source_manifest_sha,
        "source_metadata": metadata,
        "items": items,
    }


def _trajectory_manifest(
    source_root: Path, output: Path, *, allow_incomplete: bool
) -> dict[str, Any]:
    shard_meta_paths = sorted(source_root.glob("shard-*-meta.json"))
    shard_meta = [_read_json(path) for path in shard_meta_paths]
    incomplete = [
        path.name
        for path, meta in zip(shard_meta_paths, shard_meta)
        if meta.get("status") != "done"
    ]
    if incomplete and not allow_incomplete:
        raise ContractError(
            f"agentic extraction is incomplete ({incomplete}); rerun after all shards are "
            "done or pass --allow-incomplete only for a smoke run"
        )
    for path, meta in zip(shard_meta_paths, shard_meta):
        target = meta.get("target", {})
        if (
            target.get("repo_id") != MODEL_ID
            or target.get("revision") != BASE_REVISION
            or target.get("layer") != LAYER_INDEX
        ):
            raise ContractError(f"{path}: target-model metadata mismatch")
        if meta.get("failures"):
            raise ContractError(f"{path}: extraction failures are nonempty")

    trajectory_paths = sorted((source_root / "trajectories").glob("*/*.json"))
    if not trajectory_paths:
        raise ContractError(f"{source_root}: no trajectories/*/*.json files")
    if shard_meta and not incomplete:
        expected_trajectories = sum(int(meta.get("assigned_tasks", 0)) for meta in shard_meta)
        if expected_trajectories and len(trajectory_paths) != expected_trajectories:
            raise ContractError(
                f"{source_root}: all shard metadata says done, but found "
                f"{len(trajectory_paths)} trajectories for {expected_trajectories} assigned tasks"
            )
    items: list[dict[str, Any]] = []
    vector_paths_seen: set[Path] = set()
    trajectory_ids_seen: set[str] = set()
    for trajectory_path in trajectory_paths:
        trajectory = _read_json(trajectory_path)
        if trajectory.get("status") != "complete":
            raise ContractError(f"{trajectory_path}: status must be 'complete'")
        target = trajectory.get("target_model", {})
        if (
            target.get("repo_id") != MODEL_ID
            or target.get("revision") != BASE_REVISION
            or target.get("layer") != LAYER_INDEX
        ):
            raise ContractError(f"{trajectory_path}: target-model metadata mismatch")
        family_id = trajectory.get("family_id")
        trajectory_id = trajectory.get("trajectory_id")
        if not isinstance(family_id, str) or not isinstance(trajectory_id, str):
            raise ContractError(f"{trajectory_path}: missing family_id/trajectory_id")
        if trajectory_id in trajectory_ids_seen:
            raise ContractError(f"duplicate trajectory_id={trajectory_id!r}")
        trajectory_ids_seen.add(trajectory_id)
        family, benchmark = _family(family_id, str(trajectory_path))
        activations = trajectory.get("activations")
        if not isinstance(activations, list) or not activations:
            raise ContractError(f"{trajectory_path}: activations must be nonempty")
        if len(activations) != 4:
            raise ContractError(
                f"{trajectory_path}: expected the protocol's four sampled activations, "
                f"got {len(activations)}"
            )
        position_keys: set[int] = set()
        for activation_ordinal, activation in enumerate(activations):
            if not isinstance(activation, dict):
                raise ContractError(
                    f"{trajectory_path}: activation {activation_ordinal} is not an object"
                )
            position_key = activation.get("position_key")
            vector_file = activation.get("vector_file")
            vector_sha = activation.get("vector_sha256")
            if not isinstance(position_key, int) or not isinstance(vector_file, str):
                raise ContractError(
                    f"{trajectory_path}: activation {activation_ordinal} lacks "
                    "position_key/vector_file"
                )
            if position_key in position_keys:
                raise ContractError(
                    f"{trajectory_path}: duplicate position_key={position_key}"
                )
            position_keys.add(position_key)
            vector_path = (source_root / vector_file).resolve()
            if source_root.resolve() not in vector_path.parents:
                raise ContractError(
                    f"{trajectory_path}: vector_file escapes source root: {vector_file}"
                )
            if vector_path in vector_paths_seen:
                raise ContractError(f"vector file referenced twice: {vector_path}")
            vector_paths_seen.add(vector_path)
            actual_sha = _validate_vector_file(vector_path, vector_sha)
            items.append(
                {
                    "activation_id": f"{trajectory_id}:position_key:{position_key}",
                    "activation_index": 0,
                    "activation_file": _relative(vector_path, output),
                    "activation_sha256": actual_sha,
                    "document_id": None,
                    "trajectory_id": trajectory_id,
                    "cluster_id": trajectory_id,
                    "distribution": "ood",
                    "benchmark": benchmark,
                    "family": family,
                    "family_id": family_id,
                    "task_id": trajectory.get("task_id"),
                    "trajectory_file": _relative(trajectory_path, output),
                    "activation_ordinal": activation_ordinal,
                    "position_key": position_key,
                    "turn_index": activation.get("turn_index"),
                    "assistant_response_token_index": activation.get(
                        "assistant_response_token_index"
                    ),
                    "token_position": activation.get("token_position"),
                    "read_token": activation.get("read_token"),
                }
            )

    return {
        "schema_version": ACTIVATION_SCHEMA,
        "model_id": MODEL_ID,
        "model_revision": BASE_REVISION,
        "layer_index": LAYER_INDEX,
        "d_model": D_MODEL,
        "activation_norm": "none",
        "activation_file": None,
        "activation_file_mode": "per_item",
        "source_root": _relative(source_root, output),
        "source_shards_complete": not incomplete,
        "source_shard_metadata": shard_meta,
        "items": sorted(items, key=lambda item: item["activation_id"]),
    }


def convert(
    source_root: Path,
    output: Path,
    *,
    kind: str = "auto",
    allow_incomplete: bool = False,
) -> dict[str, Any]:
    source_root = source_root.expanduser().resolve()
    output = output.expanduser().resolve()
    if kind == "auto":
        if (source_root / "gold.npy").is_file() and (source_root / "manifest.json").is_file():
            kind = "matrix"
        elif (source_root / "trajectories").is_dir() and (source_root / "activations").is_dir():
            kind = "trajectory"
        else:
            raise ContractError(
                f"cannot auto-detect {source_root}; expected gold.npy+manifest.json "
                "or trajectories/+activations/"
            )
    if kind == "matrix":
        result = _matrix_manifest(source_root, output)
    elif kind == "trajectory":
        result = _trajectory_manifest(
            source_root, output, allow_incomplete=allow_incomplete
        )
    else:
        raise ContractError(f"unknown source kind: {kind}")
    atomic_write_json(output, result)
    return {
        "output": str(output),
        "output_sha256": sha256_file(output),
        "items": len(result["items"]),
        "activation_file_mode": result.get("activation_file_mode", "matrix"),
        "source_shards_complete": result.get("source_shards_complete", True),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a zero-copy canonical manifest for existing activation files."
    )
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--kind", choices=("auto", "matrix", "trajectory"), default="auto")
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Permit an agentic run with non-done shard metadata (smoke only).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = convert(
        Path(args.source_root),
        Path(args.output),
        kind=args.kind,
        allow_incomplete=args.allow_incomplete,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
