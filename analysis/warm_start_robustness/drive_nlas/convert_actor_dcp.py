#!/usr/bin/env python3
"""Convert a supplied actor DCP checkpoint into a verified full-HF checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import time
import uuid
from pathlib import Path
from typing import Any


MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
BASE_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
SCHEMA = "nla.warm_start.actor_dcp_conversion.v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def atomic_json(path: Path, value: Any) -> None:
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    payload = canonical_json(value)
    with tmp.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def inventory(root: Path, names: list[str] | None = None) -> list[dict[str, Any]]:
    paths = [root / name for name in names] if names is not None else sorted(
        path for path in root.rglob("*") if path.is_file()
    )
    rows = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.append(
            {
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return rows


def validate_existing(output: Path, source_digest: str) -> bool:
    marker = output / "conversion_manifest.json"
    if not marker.is_file():
        return False
    record = json.loads(marker.read_text())
    if record.get("schema") != SCHEMA or record.get("source_digest") != source_digest:
        return False
    for row in record.get("output_files", []):
        path = output / row["path"]
        if not path.is_file() or path.stat().st_size != row["bytes"]:
            return False
        if sha256_file(path) != row["sha256"]:
            return False
    return bool(record.get("complete"))


def verify_saved_tensors(model: Any, output: Path) -> dict[str, Any]:
    """Require bitwise identity between the loaded DCP state and saved HF tensors."""
    import torch
    from safetensors import safe_open

    index_path = output / "model.safetensors.index.json"
    if index_path.is_file():
        weight_map = json.loads(index_path.read_text())["weight_map"]
    else:
        single = output / "model.safetensors"
        if not single.is_file():
            raise RuntimeError(f"saved actor has no safetensors weights: {output}")
        with safe_open(str(single), framework="pt", device="cpu") as handle:
            weight_map = {key: single.name for key in handle.keys()}

    expected = model.state_dict()
    if set(weight_map) != set(expected):
        missing = sorted(set(expected) - set(weight_map))
        unexpected = sorted(set(weight_map) - set(expected))
        raise RuntimeError(
            f"saved HF/model key mismatch: missing={missing[:10]} "
            f"unexpected={unexpected[:10]}"
        )

    checked_elements = 0
    for shard_name in sorted(set(weight_map.values())):
        shard_keys = sorted(key for key, value in weight_map.items() if value == shard_name)
        with safe_open(
            str(output / shard_name), framework="pt", device="cpu"
        ) as handle:
            if set(handle.keys()) != set(shard_keys):
                raise RuntimeError(f"safetensors index disagrees with shard {shard_name}")
            for key in shard_keys:
                saved = handle.get_tensor(key)
                source = expected[key].detach().cpu()
                if saved.dtype != source.dtype or tuple(saved.shape) != tuple(source.shape):
                    raise RuntimeError(
                        f"saved tensor metadata mismatch for {key}: "
                        f"{saved.dtype}/{tuple(saved.shape)} != "
                        f"{source.dtype}/{tuple(source.shape)}"
                    )
                if not torch.equal(saved, source):
                    raise RuntimeError(f"saved tensor is not bitwise equal for {key}")
                checked_elements += source.numel()
    return {
        "method": "torch.equal for every tensor after safetensors serialization",
        "tensor_count": len(expected),
        "parameter_elements": checked_elements,
        "bitwise_equal": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True, help="DCP model directory")
    parser.add_argument("--output", type=Path, required=True, help="Final full-HF directory")
    parser.add_argument("--base-model", default=MODEL_ID)
    parser.add_argument("--base-revision", default=BASE_REVISION)
    parser.add_argument("--max-shard-size", default="5GB")
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    metadata_path = source / ".metadata"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    metadata = pickle.loads(metadata_path.read_bytes())
    storage_names = sorted(
        {
            str(info.relative_path)
            for info in metadata.storage_data.values()
            if getattr(info, "relative_path", None)
        }
    )
    if storage_names != ["__0_0.distcp", "__1_0.distcp"]:
        raise RuntimeError(f"unexpected actor DCP storage files: {storage_names}")
    source_files = inventory(source, [".metadata", *storage_names])
    source_digest = hashlib.sha256(canonical_json(source_files)).hexdigest()
    if output.exists() and validate_existing(output, source_digest):
        print(f"already complete output={output} source_digest={source_digest}", flush=True)
        return

    import torch
    import torch.distributed.checkpoint as dcp
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"load base model={args.base_model} revision={args.base_revision}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        revision=args.base_revision,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, revision=args.base_revision)
    model_state = model.state_dict()
    prefix = "model_state.model."
    dcp_metadata = metadata.state_dict_metadata
    dcp_keys = {key.removeprefix(prefix) for key in dcp_metadata if key.startswith(prefix)}
    model_keys = set(model_state)
    if dcp_keys != model_keys:
        missing = sorted(model_keys - dcp_keys)
        unexpected = sorted(dcp_keys - model_keys)
        raise RuntimeError(
            f"DCP/model key mismatch: missing={missing[:10]} unexpected={unexpected[:10]}"
        )
    for key, tensor in model_state.items():
        saved = dcp_metadata[prefix + key]
        if tuple(saved.size) != tuple(tensor.shape):
            raise RuntimeError(f"shape mismatch for {key}: {saved.size} != {tensor.shape}")
        if saved.properties.dtype != tensor.dtype:
            raise RuntimeError(
                f"dtype mismatch for {key}: {saved.properties.dtype} != {tensor.dtype}"
            )

    nested = {"model_state": {"model": model_state}}
    print(f"load DCP source={source}", flush=True)
    dcp.load(nested, checkpoint_id=str(source))
    nonfinite = []
    total = 0
    for name, tensor in model.state_dict().items():
        total += tensor.numel()
        if not torch.isfinite(tensor).all().item():
            nonfinite.append(name)
    if nonfinite:
        raise RuntimeError(f"non-finite actor tensors: {nonfinite[:10]}")

    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.parent / f".{output.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    tmp.mkdir()
    print(f"save full-HF temp={tmp}", flush=True)
    model.save_pretrained(
        tmp,
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )
    tokenizer.save_pretrained(tmp)
    roundtrip_verification = verify_saved_tensors(model, tmp)
    output_files = inventory(tmp)
    manifest = {
        "schema": SCHEMA,
        "complete": True,
        "source": str(source),
        "source_files": source_files,
        "source_digest": source_digest,
        "base_model": args.base_model,
        "base_revision": args.base_revision,
        "parameter_elements": total,
        "all_parameters_finite": True,
        "roundtrip_verification": roundtrip_verification,
        "output_files": output_files,
        "created_at_unix": time.time(),
    }
    atomic_json(tmp / "conversion_manifest.json", manifest)
    if output.exists():
        backup = output.parent / f"{output.name}.incomplete-{int(time.time())}"
        os.replace(output, backup)
        print(f"preserved prior incomplete output={backup}", flush=True)
    os.replace(tmp, output)
    print(
        f"complete output={output} source_digest={source_digest} files={len(output_files)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
