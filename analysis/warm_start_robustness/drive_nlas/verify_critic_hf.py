#!/usr/bin/env python3
"""Validate a supplied NLA critic HF export and emit a hash-sealed receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

import torch
import yaml
from safetensors import safe_open


SCHEMA = "nla.warm_start.critic_hf_verification.v1"
EXPECTED_SHARDS = {
    "model-00001-of-00003.safetensors",
    "model-00002-of-00003.safetensors",
    "model-00003-of-00003.safetensors",
}
EXPECTED_CONTRACT = {
    "d_model": 3584,
    "injection_token_id": 149705,
    "injection_left_neighbor_id": 29,
    "injection_right_neighbor_id": 522,
}


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    payload = canonical_json(value)
    with tmp.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def require_finite(tensor: torch.Tensor, context: str) -> None:
    flat = tensor.reshape(-1)
    chunk = 16 * 1024 * 1024
    for start in range(0, flat.numel(), chunk):
        if not torch.isfinite(flat[start : start + chunk]).all().item():
            raise RuntimeError(f"non-finite values in {context}")


def sidecar_contract(path: Path, expected_role: str) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text())
    tokens = value.get("tokens", {})
    actual = {
        "d_model": value.get("d_model"),
        "injection_token_id": tokens.get("injection_token_id"),
        "injection_left_neighbor_id": tokens.get("injection_left_neighbor_id"),
        "injection_right_neighbor_id": tokens.get("injection_right_neighbor_id"),
    }
    if actual != EXPECTED_CONTRACT:
        raise RuntimeError(f"sidecar contract mismatch: {actual}")
    if value.get("role") != expected_role:
        raise RuntimeError(f"{path}: role={value.get('role')!r}, expected {expected_role!r}")
    extraction_layer_index = value.get("critic", {}).get("extraction_layer_index")
    if expected_role == "critic" and extraction_layer_index != 20:
        raise RuntimeError(
            f"{path}: critic extraction_layer_index={extraction_layer_index!r}"
        )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "role": expected_role,
        "extraction_layer_index": extraction_layer_index,
        **actual,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--critic-dir", type=Path, required=True)
    parser.add_argument("--actor-sidecar", type=Path, required=True)
    parser.add_argument("--critic-sidecar", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.critic_dir.expanduser().resolve()
    index_path = root / "model.safetensors.index.json"
    config_path = root / "config.json"
    value_head_path = root / "value_head.safetensors"
    for path in (index_path, config_path, value_head_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)

    config = json.loads(config_path.read_text())
    expected_config = {
        "model_type": "qwen2",
        "hidden_size": 3584,
        "num_hidden_layers": 21,
        "vocab_size": 152064,
    }
    actual_config = {key: config.get(key) for key in expected_config}
    if actual_config != expected_config:
        raise RuntimeError(f"critic config mismatch: {actual_config}")

    index = json.loads(index_path.read_text())
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise RuntimeError("critic index has no empty weight_map")
    shard_names = set(weight_map.values())
    if shard_names != EXPECTED_SHARDS:
        raise RuntimeError(f"unexpected critic shards: {sorted(shard_names)}")

    files: list[dict[str, Any]] = []
    all_keys: set[str] = set()
    logical_bytes = 0
    tensor_count = 0
    for shard_name in sorted(shard_names):
        path = root / shard_name
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
        expected_keys = {key for key, value in weight_map.items() if value == shard_name}
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            actual_keys = set(handle.keys())
            if actual_keys != expected_keys:
                raise RuntimeError(
                    f"index/header key mismatch for {shard_name}: "
                    f"missing={sorted(expected_keys - actual_keys)[:10]} "
                    f"extra={sorted(actual_keys - expected_keys)[:10]}"
                )
            if all_keys & actual_keys:
                raise RuntimeError(f"duplicate tensor keys in {shard_name}")
            all_keys.update(actual_keys)
            for key in sorted(actual_keys):
                tensor = handle.get_tensor(key)
                require_finite(tensor, f"{shard_name}:{key}")
                logical_bytes += tensor.numel() * tensor.element_size()
                tensor_count += 1
        files.append(
            {"path": shard_name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        )

    if all_keys != set(weight_map):
        raise RuntimeError("critic shard union does not equal index key set")
    declared_bytes = index.get("metadata", {}).get("total_size")
    if logical_bytes != declared_bytes:
        raise RuntimeError(
            f"logical tensor bytes {logical_bytes} != index total_size {declared_bytes}"
        )

    with safe_open(str(value_head_path), framework="pt", device="cpu") as handle:
        if set(handle.keys()) != {"weight"}:
            raise RuntimeError(f"unexpected value-head keys: {sorted(handle.keys())}")
        head = handle.get_tensor("weight")
        # These supplied checkpoints serialize the head in bf16. EasyNLA loads
        # it and deliberately promotes it to fp32 for critic inference/training.
        if head.dtype != torch.bfloat16 or tuple(head.shape) != (3584, 3584):
            raise RuntimeError(f"unexpected value-head tensor: {head.dtype} {tuple(head.shape)}")
        require_finite(head, "value_head.weight")
    files.append(
        {
            "path": value_head_path.name,
            "bytes": value_head_path.stat().st_size,
            "sha256": sha256_file(value_head_path),
        }
    )

    actor_contract = sidecar_contract(args.actor_sidecar.expanduser().resolve(), "actor")
    critic_contract = sidecar_contract(args.critic_sidecar.expanduser().resolve(), "critic")
    comparable_actor = {key: actor_contract[key] for key in EXPECTED_CONTRACT}
    comparable_critic = {key: critic_contract[key] for key in EXPECTED_CONTRACT}
    if comparable_actor != comparable_critic:
        raise RuntimeError("actor and critic sidecar contracts differ")

    payload = {
        "schema": SCHEMA,
        "complete": True,
        "verified_at_unix": time.time(),
        "critic_dir": str(root),
        "config": actual_config,
        "index_sha256": sha256_file(index_path),
        "index_tensor_count": len(weight_map),
        "verified_tensor_count": tensor_count,
        "logical_tensor_bytes": logical_bytes,
        "all_tensors_finite": True,
        "value_head": {"dtype": str(head.dtype), "shape": list(head.shape)},
        "actor_sidecar": actor_contract,
        "critic_sidecar": critic_contract,
        "files": files,
    }
    payload["payload_sha256"] = hashlib.sha256(canonical_json(payload)).hexdigest()
    atomic_json(args.output.expanduser().resolve(), payload)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
