#!/usr/bin/env python3
"""Generate matched NLA AV samples from saved raw layer-20 activations.

Inference follows the checkpoint's native contract.  The stored activation is
kept raw on disk, rescaled to ``extraction.injection_scale`` at inference, and
replaces the canonical marker token's input embedding via
``nla.injection.inject_at_marked_positions``.  Generation receives
``inputs_embeds``; no residual-addition hook is installed.

Durability model
----------------
Each (activation, condition, sample-index) result is first committed as its own
fsync + rename JSON file.  A restart validates and reuses those immutable
records.  At shard completion, deterministic ``explanations.jsonl`` (the scorer
contract) and ``raw_generations.jsonl`` audit exports are themselves written by
fsync + rename.  Thus a kill can lose at most the generation currently in GPU
memory, never an already committed result.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import json
import math
import os
import re
import sys
import time
import traceback
import unicodedata
import uuid
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np


ACTIVATION_SCHEMA = "nla.claim_agreement.activation_manifest.v1"
GENERATION_SCHEMA = "nla.claim_agreement.generation.v2"
EXPLANATION_SCHEMA = "nla.claim_agreement.explanation.v1"
SHARD_SCHEMA = "nla.claim_agreement.generation_shard.v2"

# Experiment-locked values.  They are constants rather than CLI defaults so a
# launch cannot silently create a condition with a different sampling policy.
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
BASE_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
LAYER_INDEX = 20
D_MODEL = 3584
N_SAMPLES = 2
TEMPERATURE = 0.7
TOP_P = 1.0
MAX_NEW_TOKENS = 150
CJK_SUSPECT_FRACTION = 0.20
CJK_SHARD_ALERT_FRACTION = 0.05
CJK_SHARD_ALERT_MIN_COUNT = 3

_ALLOWED_DISTRIBUTIONS = {"fineweb", "ood"}
_ALLOWED_FAMILIES = {"fineweb", "alignment", "capability"}
_BENCHMARKS_BY_FAMILY = {
    "fineweb": {"fineweb"},
    "alignment": {
        "blackmail",
        "destructive_coding_actions",
        "overly_agentic_actions",
        "factual_honesty",
        "appropriate_harmlessness",
        "sycophancy",
        "multi_turn_deception",
    },
    "capability": {
        "swe_bench_verified",
        "gpqa_diamond",
        "humanitys_last_exam",
        "tau2_bench_retail",
        "mmmlu",
        "browsecomp",
        "terminal_bench_2",
        "financeagent",
        "cybergym",
    },
}
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class ContractError(ValueError):
    """An input or resume artifact violates the experiment contract."""


@dataclasses.dataclass(frozen=True)
class ActivationSource:
    manifest_path: Path
    manifest_sha256: str
    activation_path: Path
    activation_sha256: str
    array_key: str | None
    array: np.ndarray


@dataclasses.dataclass(frozen=True)
class ActivationItem:
    activation_id: str
    activation_index: int
    document_id: str | None
    trajectory_id: str | None
    cluster_id: str
    distribution: str
    benchmark: str
    family: str
    source: ActivationSource
    source_metadata: Mapping[str, Any]


@dataclasses.dataclass(frozen=True)
class ParsedExplanation:
    explanation: str
    status: str
    open_tag_count: int
    close_tag_count: int


@dataclasses.dataclass(frozen=True)
class RuntimeOutput:
    raw_text: str
    clean_text: str
    token_ids: tuple[int, ...]
    stop_reason: str
    eos_token_id: int | None
    injection_apply_count_before: int
    injection_apply_count_after: int
    injection_target_l2_norm: float
    injected_l2_norm: float
    marker_row_matches_scaled_activation: bool


def canonical_json(value: Any) -> bytes:
    """Stable UTF-8 JSON used for fingerprints and atomic artifacts."""
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def sha256_file(path: Path, chunk_bytes: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_bytes):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _fsync_directory(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Commit one file by fsyncing a same-directory temp then replacing it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        with tmp.open("xb") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        _fsync_directory(path.parent)
    finally:
        # If the write itself failed before replace, do not leave a confusing
        # transaction remnant for a later integrity audit.
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_bytes(path, canonical_json(value))


def atomic_write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    payload = b"".join(canonical_json(row) for row in rows)
    atomic_write_bytes(path, payload)


def _required_str(obj: Mapping[str, Any], key: str, context: str) -> str:
    value = obj.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{context}: {key!r} must be a nonempty string")
    return value


def _optional_id(obj: Mapping[str, Any], key: str, context: str) -> str | None:
    value = obj.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{context}: {key!r} must be null or a nonempty string")
    return value


def _activation_index(item: Mapping[str, Any], context: str) -> int:
    aliases = [k for k in ("activation_index", "row", "index") if k in item]
    if not aliases:
        raise ContractError(
            f"{context}: one of 'activation_index', 'row', or 'index' is required"
        )
    values = [item[k] for k in aliases]
    if any(isinstance(v, bool) or not isinstance(v, int) for v in values):
        raise ContractError(f"{context}: activation row/index must be an integer")
    if len(set(values)) != 1:
        raise ContractError(f"{context}: conflicting activation row aliases: {values}")
    if values[0] < 0:
        raise ContractError(f"{context}: activation row/index must be nonnegative")
    return values[0]


def _resolve_array_path(manifest_path: Path, raw: str) -> Path:
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = manifest_path.parent / p
    return p.resolve()


def _load_array(path: Path, array_key: str | None) -> np.ndarray:
    if not path.is_file():
        raise ContractError(f"activation array does not exist: {path}")
    if path.suffix == ".npy":
        if array_key is not None:
            raise ContractError("activation_array_key is invalid for a .npy array")
        arr = np.load(path, mmap_mode="r", allow_pickle=False)
    elif path.suffix == ".npz":
        archive = np.load(path, mmap_mode="r", allow_pickle=False)
        try:
            key = array_key or (archive.files[0] if len(archive.files) == 1 else None)
            if key is None:
                raise ContractError(
                    f"{path}: multi-array .npz requires top-level activation_array_key"
                )
            if key not in archive.files:
                raise ContractError(
                    f"{path}: activation_array_key={key!r} not in {archive.files}"
                )
            # npz members are decompressed eagerly.  This is accepted for input
            # compatibility; .npy is preferred for mmap + multi-shard launches.
            arr = archive[key]
        finally:
            archive.close()
    else:
        raise ContractError(f"activation file must be .npy or .npz, got: {path}")
    # Agentic sources store one raw vector per .npy.  Treat it as a one-row
    # matrix without copying; the ordinary FineWeb/static sources are [N, d].
    if arr.ndim == 1 and arr.shape == (D_MODEL,):
        arr = arr.reshape(1, D_MODEL)
    if arr.ndim != 2:
        raise ContractError(
            f"{path}: activation array must be [N,{D_MODEL}] or [{D_MODEL}], got {arr.shape}"
        )
    if arr.shape[1] != D_MODEL:
        raise ContractError(
            f"{path}: expected d_model={D_MODEL}, array has shape {arr.shape}"
        )
    if not np.issubdtype(arr.dtype, np.floating):
        raise ContractError(f"{path}: activation dtype must be floating, got {arr.dtype}")
    return arr


def load_activation_manifests(
    manifest_paths: Sequence[str | Path], *, verify_sha256: bool = True
) -> list[ActivationItem]:
    """Load and strictly validate one or more raw-activation manifests."""
    if not manifest_paths:
        raise ContractError("at least one --input-manifest is required")
    items_out: list[ActivationItem] = []
    seen_ids: dict[str, Path] = {}
    array_cache: dict[tuple[Path, str | None], tuple[np.ndarray, str]] = {}

    for raw_manifest_path in manifest_paths:
        manifest_path = Path(raw_manifest_path).expanduser().resolve()
        if not manifest_path.is_file():
            raise ContractError(f"input manifest does not exist: {manifest_path}")
        manifest_bytes = manifest_path.read_bytes()
        manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()
        try:
            manifest = json.loads(manifest_bytes)
        except json.JSONDecodeError as exc:
            raise ContractError(f"{manifest_path}: invalid JSON: {exc}") from exc
        if not isinstance(manifest, dict):
            raise ContractError(f"{manifest_path}: top level must be an object")
        if manifest.get("schema_version") != ACTIVATION_SCHEMA:
            raise ContractError(
                f"{manifest_path}: schema_version must equal {ACTIVATION_SCHEMA!r}"
            )
        model_id = manifest.get("model_id", manifest.get("base_model"))
        if model_id != MODEL_ID:
            raise ContractError(
                f"{manifest_path}: expected model_id={MODEL_ID!r}, got {model_id!r}"
            )
        if manifest.get("model_revision") != BASE_REVISION:
            raise ContractError(
                f"{manifest_path}: expected model_revision={BASE_REVISION!r}, "
                f"got {manifest.get('model_revision')!r}"
            )
        if manifest.get("layer_index") != LAYER_INDEX:
            raise ContractError(
                f"{manifest_path}: expected layer_index={LAYER_INDEX}, "
                f"got {manifest.get('layer_index')!r}"
            )
        if manifest.get("d_model") != D_MODEL:
            raise ContractError(
                f"{manifest_path}: expected d_model={D_MODEL}, "
                f"got {manifest.get('d_model')!r}"
            )
        norm = manifest.get("activation_norm", manifest.get("norm"))
        if norm not in ("none", "raw"):
            raise ContractError(
                f"{manifest_path}: activations must be raw (activation_norm='none'), "
                f"got {norm!r}"
            )
        # Matrix manifests use one top-level activation_file.  Agentic runs can
        # instead set activation_file_mode="per_item" and point each item at its
        # already-existing one-vector .npy; the converter never duplicates data.
        top_activation_file = manifest.get("activation_file")
        per_item_mode = manifest.get("activation_file_mode") == "per_item"
        if per_item_mode:
            if top_activation_file is not None:
                raise ContractError(
                    f"{manifest_path}: per_item mode requires top-level activation_file=null"
                )
        elif not isinstance(top_activation_file, str) or not top_activation_file.strip():
            raise ContractError(
                f"{manifest_path}: top-level activation_file must be a nonempty string"
            )

        def resolve_source(raw_item: Mapping[str, Any], context: str) -> ActivationSource:
            activation_file = (
                _required_str(raw_item, "activation_file", context)
                if per_item_mode
                else str(top_activation_file)
            )
            activation_path = _resolve_array_path(manifest_path, activation_file)
            array_key_raw = (
                raw_item.get("activation_array_key")
                if per_item_mode
                else manifest.get("activation_array_key")
            )
            if array_key_raw is not None and not isinstance(array_key_raw, str):
                raise ContractError(f"{context}: activation_array_key must be a string or null")
            cache_key = (activation_path, array_key_raw)
            expected_sha = (
                raw_item.get("activation_sha256", raw_item.get("activation_file_sha256"))
                if per_item_mode
                else manifest.get(
                    "activation_sha256", manifest.get("activation_file_sha256")
                )
            )
            if expected_sha is not None and (
                not isinstance(expected_sha, str) or not _HEX64.fullmatch(expected_sha)
            ):
                raise ContractError(
                    f"{context}: activation_sha256 must be 64 lowercase hex digits"
                )
            if cache_key not in array_cache:
                actual_sha = (
                    sha256_file(activation_path) if verify_sha256 or expected_sha else ""
                )
                if expected_sha is not None and actual_sha != expected_sha:
                    raise ContractError(
                        f"{activation_path}: sha256 mismatch; expected {expected_sha}, "
                        f"got {actual_sha}"
                    )
                array_cache[cache_key] = (
                    _load_array(activation_path, array_key_raw),
                    actual_sha,
                )
            array, actual_sha = array_cache[cache_key]
            return ActivationSource(
                manifest_path=manifest_path,
                manifest_sha256=manifest_digest,
                activation_path=activation_path,
                activation_sha256=actual_sha or expected_sha or "uncomputed",
                array_key=array_key_raw,
                array=array,
            )

        raw_items = manifest.get("items")
        if not isinstance(raw_items, list) or not raw_items:
            raise ContractError(f"{manifest_path}: items must be a nonempty array")
        seen_indices: set[tuple[Path, int]] = set()
        for ordinal, raw_item in enumerate(raw_items):
            context = f"{manifest_path}: items[{ordinal}]"
            if not isinstance(raw_item, dict):
                raise ContractError(f"{context}: item must be an object")
            source = resolve_source(raw_item, context)
            array = source.array
            activation_id = _required_str(raw_item, "activation_id", context)
            if activation_id in seen_ids:
                raise ContractError(
                    f"duplicate activation_id={activation_id!r} in {manifest_path} and "
                    f"{seen_ids[activation_id]}"
                )
            seen_ids[activation_id] = manifest_path
            index = _activation_index(raw_item, context)
            if index >= array.shape[0]:
                raise ContractError(
                    f"{context}: activation index {index} outside array with "
                    f"{array.shape[0]} rows"
                )
            source_index = (source.activation_path, index)
            if source_index in seen_indices:
                raise ContractError(
                    f"{context}: activation source {source.activation_path} row {index} "
                    "is referenced twice in one manifest"
                )
            seen_indices.add(source_index)

            distribution = _required_str(raw_item, "distribution", context).lower()
            family = _required_str(raw_item, "family", context).lower()
            benchmark = _required_str(raw_item, "benchmark", context)
            cluster_id = _required_str(raw_item, "cluster_id", context)
            if distribution not in _ALLOWED_DISTRIBUTIONS:
                raise ContractError(
                    f"{context}: distribution must be one of {_ALLOWED_DISTRIBUTIONS}"
                )
            if family not in _ALLOWED_FAMILIES:
                raise ContractError(f"{context}: family must be one of {_ALLOWED_FAMILIES}")
            if benchmark not in _BENCHMARKS_BY_FAMILY[family]:
                raise ContractError(
                    f"{context}: benchmark={benchmark!r} is not a frozen {family} slug; "
                    f"expected one of {sorted(_BENCHMARKS_BY_FAMILY[family])}"
                )
            if distribution == "fineweb" and family != "fineweb":
                raise ContractError(f"{context}: FineWeb rows require family='fineweb'")
            if distribution == "ood" and family == "fineweb":
                raise ContractError(
                    f"{context}: OOD rows require family='alignment' or 'capability'"
                )
            document_id = _optional_id(raw_item, "document_id", context)
            trajectory_id = _optional_id(raw_item, "trajectory_id", context)
            if distribution == "fineweb" and document_id is None:
                raise ContractError(f"{context}: FineWeb rows require document_id")
            if distribution == "ood" and trajectory_id is None:
                raise ContractError(f"{context}: OOD rows require trajectory_id")

            reserved = {
                "activation_id", "activation_index", "row", "index", "document_id",
                "trajectory_id", "cluster_id", "distribution", "benchmark", "family",
                "activation_file", "activation_sha256", "activation_file_sha256",
                "activation_array_key",
            }
            source_metadata = {k: v for k, v in raw_item.items() if k not in reserved}
            items_out.append(
                ActivationItem(
                    activation_id=activation_id,
                    activation_index=index,
                    document_id=document_id,
                    trajectory_id=trajectory_id,
                    cluster_id=cluster_id,
                    distribution=distribution,
                    benchmark=benchmark,
                    family=family,
                    source=source,
                    source_metadata=source_metadata,
                )
            )
    return items_out


def shard_for_activation(activation_id: str, num_shards: int) -> int:
    if num_shards <= 0:
        raise ContractError("num_shards must be positive")
    digest = hashlib.sha256(activation_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % num_shards


def sampling_identity(
    activation_id: str, condition: str, sample_index: int, sampling_seed: int
) -> tuple[str, int]:
    """Return a stable full key and positive 63-bit PyTorch seed."""
    if condition not in ("plausible", "implausible"):
        raise ContractError(f"invalid condition: {condition!r}")
    if sample_index < 0:
        raise ContractError("sample_index must be nonnegative")
    material = {
        "schema": EXPLANATION_SCHEMA,
        "activation_id": activation_id,
        "condition": condition,
        "sample_index": sample_index,
        "sampling_seed": sampling_seed,
    }
    key = sha256_json(material)
    seed = int(key[:16], 16) & ((1 << 63) - 1)
    return key, seed


def record_key(activation_id: str, condition: str, sample_index: int) -> str:
    return sha256_json(
        {
            "activation_id": activation_id,
            "condition": condition,
            "sample_index": sample_index,
        }
    )


def parse_explanation(text: str) -> ParsedExplanation:
    """Parse NLA explanation tags while retaining a specific audit status."""
    open_tag = "<explanation>"
    close_tag = "</explanation>"
    n_open = text.count(open_tag)
    n_close = text.count(close_tag)
    if n_open == 0 and n_close == 0:
        return ParsedExplanation("", "missing_both_tags", 0, 0)
    if n_open == 0:
        return ParsedExplanation("", "missing_open_tag", 0, n_close)
    if n_close == 0:
        return ParsedExplanation("", "missing_close_tag", n_open, 0)
    if n_open != 1 or n_close != 1:
        return ParsedExplanation("", "multiple_tags", n_open, n_close)
    start = text.find(open_tag) + len(open_tag)
    end = text.find(close_tag)
    if end < start:
        return ParsedExplanation("", "close_before_open", 1, 1)
    explanation = text[start:end].strip()
    if not explanation:
        return ParsedExplanation("", "empty_explanation", 1, 1)
    return ParsedExplanation(explanation, "ok", 1, 1)


def _torch_keyed_rng(seed: int, device: str) -> contextlib.AbstractContextManager[None]:
    import torch

    dev = torch.device(device)
    devices: list[int] = []
    if dev.type == "cuda":
        devices = [dev.index if dev.index is not None else torch.cuda.current_device()]

    @contextlib.contextmanager
    def manager() -> Iterator[None]:
        # Save/restore the process RNG so outputs depend only on the recorded key,
        # not shard order, skipped resume records, or another activation's length.
        with torch.random.fork_rng(devices=devices, enabled=True):
            torch.manual_seed(seed)
            if devices:
                torch.cuda.manual_seed(seed)
            yield

    return manager()


def _file_provenance(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "exists": False}
    return {
        "path": str(path.resolve()),
        "exists": True,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _checkpoint_provenance(actor_checkpoint: str) -> list[dict[str, Any]]:
    path = Path(actor_checkpoint).expanduser()
    if not path.is_dir():
        return [{"repository_or_path": actor_checkpoint, "local": False}]
    adapter = path / "adapter_config.json"
    if adapter.is_file():
        candidates = (
            "adapter_config.json",
            "adapter_model.safetensors",
            "nla_meta.yaml",
        )
        return [_file_provenance(path / name) for name in candidates]
    index = path / "model.safetensors.index.json"
    if index.is_file():
        try:
            weight_map = json.loads(index.read_text())["weight_map"]
            shards = sorted(set(str(name) for name in weight_map.values()))
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ContractError(f"invalid full-HF actor index {index}: {exc}") from exc
        candidates = ["config.json", "generation_config.json", index.name, *shards]
        return [_file_provenance(path / name) for name in candidates]
    single = path / "model.safetensors"
    if single.is_file():
        return [_file_provenance(path / name) for name in ("config.json", single.name)]
    raise ContractError(
        f"actor checkpoint {path} is neither a PEFT adapter nor a full HF checkpoint"
    )


def _sidecar_provenance(sidecar_source: str) -> dict[str, Any]:
    raw = Path(str(sidecar_source).split("@[")[0]).expanduser()
    if raw.is_dir() or (not raw.exists() and raw.suffix == ""):
        path = raw / "nla_meta.yaml"
    elif raw.name == "nla_meta.yaml" or raw.suffix in (".yaml", ".yml"):
        path = raw
    else:
        path = raw.with_name(raw.name + ".nla_meta.yaml")
    return _file_provenance(path)


def _native_nla_code_provenance(nla_root: str) -> list[dict[str, Any]]:
    root = Path(nla_root).expanduser()
    return [
        _file_provenance(root / "nla" / "injection.py"),
        _file_provenance(root / "nla" / "schema.py"),
        _file_provenance(root / "nla" / "config.py"),
        _file_provenance(root / "nla" / "arch_adapters.py"),
    ]


def cjk_fraction(text: str) -> float:
    """Fraction of characters whose Unicode name identifies them as CJK."""
    if not text:
        return 0.0
    return sum(1 for char in text if "CJK" in unicodedata.name(char, "")) / len(text)


class NativeNLARuntime:
    """Qwen actor using scaled marker-embedding replacement."""

    def __init__(
        self,
        *,
        base_model: str,
        base_revision: str | None,
        actor_checkpoint: str,
        sidecar: str,
        nla_root: Path,
        device: str,
        load_in_4bit: bool,
        actor_format: str = "peft",
    ) -> None:
        import torch

        root = nla_root.expanduser().resolve()
        if not (root / "nla" / "injection.py").is_file():
            raise ContractError(f"not an NLA repository root: {root}")
        sys.path.insert(0, str(root))
        from nla.arch_adapters import resolve_embed_scale, resolve_text_model
        from nla.config import load_nla_config
        from nla.injection import inject_at_marked_positions
        from nla.schema import chat_template_token_ids, normalize_activation

        injection_module = sys.modules[inject_at_marked_positions.__module__]
        injection_path = Path(injection_module.__file__).resolve()
        if root not in injection_path.parents:
            raise ContractError(
                f"wrong nla package imported: injection is {injection_path}, expected under {root}"
            )

        from peft import PeftConfig, PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.device = device
        self.cjk_fraction: Callable[[str], float] = cjk_fraction
        tokenizer_kwargs: dict[str, Any] = {}
        model_kwargs: dict[str, Any] = {
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "sdpa",
        }
        if base_revision is not None:
            tokenizer_kwargs["revision"] = base_revision
            model_kwargs["revision"] = base_revision
        tokenizer_source = actor_checkpoint if actor_format == "full_hf" else base_model
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, **tokenizer_kwargs)
        sidecar_path = Path(sidecar).expanduser()
        sidecar_for_loader = (
            str(sidecar_path.parent)
            if sidecar_path.name == "nla_meta.yaml" and sidecar_path.is_file()
            else sidecar
        )
        self.cfg = load_nla_config(sidecar_for_loader, self.tokenizer)
        if self.cfg.d_model != D_MODEL:
            raise ContractError(
                f"sidecar/model mismatch: expected d_model={D_MODEL}, got {self.cfg.d_model}"
            )
        if self.cfg.injection_scale is None:
            raise ContractError("actor sidecar has no extraction.injection_scale")
        if actor_format not in {"peft", "full_hf"}:
            raise ContractError(f"unsupported actor format: {actor_format!r}")
        if load_in_4bit:
            if actor_format == "full_hf":
                raise ContractError("--load-in-4bit is not permitted for full-HF actors")
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_storage=torch.bfloat16,
            )
            model_kwargs["device_map"] = {"": device}
        adapter_base = None
        peft_cfg = None
        if actor_format == "peft":
            peft_cfg = PeftConfig.from_pretrained(actor_checkpoint)
            adapter_base = str(peft_cfg.base_model_name_or_path)
            peft_type = getattr(peft_cfg, "peft_type", None)
            peft_type_value = getattr(peft_type, "value", str(peft_type))
            if MODEL_ID not in adapter_base and BASE_REVISION not in adapter_base:
                raise ContractError(
                    f"PEFT adapter base mismatch: {adapter_base!r} does not identify the "
                    "pinned Qwen2.5-7B-Instruct base"
                )
            if (
                peft_type_value != "LORA"
                or getattr(peft_cfg, "r", None) != 128
                or getattr(peft_cfg, "lora_alpha", None) != 16
                or getattr(peft_cfg, "use_rslora", None) is not True
            ):
                raise ContractError(
                    "unexpected actor adapter recipe: expected LoRA r=128 alpha=16 rsLoRA"
                )
            base = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
            if not load_in_4bit:
                base = base.to(device)
            self.actor = PeftModel.from_pretrained(base, actor_checkpoint, is_trainable=False).eval()
        else:
            self.actor = AutoModelForCausalLM.from_pretrained(
                actor_checkpoint,
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
            ).to(device).eval()
            config = self.actor.config
            if (
                getattr(config, "model_type", None) != "qwen2"
                or int(getattr(config, "hidden_size", -1)) != D_MODEL
                or int(getattr(config, "num_hidden_layers", -1)) != 28
            ):
                raise ContractError(
                    "unexpected full-HF actor architecture: expected Qwen2, "
                    "hidden_size=3584, num_hidden_layers=28"
                )
        self.actor.config.use_cache = True

        content = self.cfg.actor_prompt_template.format(injection_char=self.cfg.injection_char)
        self.prompt_ids = chat_template_token_ids(
            self.tokenizer, [{"role": "user", "content": content}]
        )
        self.prompt_text = self.tokenizer.decode(
            self.prompt_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        self.prompt_marker_count = self.prompt_ids.count(self.cfg.injection_token_id)
        marker_positions = [
            index
            for index, token_id in enumerate(self.prompt_ids)
            if token_id == self.cfg.injection_token_id
            and 0 < index < len(self.prompt_ids) - 1
            and self.prompt_ids[index - 1] == self.cfg.injection_left_neighbor_id
            and self.prompt_ids[index + 1] == self.cfg.injection_right_neighbor_id
        ]
        self.prompt_marker_well_formed = len(marker_positions) == 1
        if self.prompt_marker_count != 1 or not self.prompt_marker_well_formed:
            raise ContractError(
                "canonical AV prompt marker is not exactly one well-formed marker: "
                f"count={self.prompt_marker_count}, well_formed={self.prompt_marker_well_formed}"
            )
        self.marker_position = marker_positions[0]
        self._resolve_text_model = resolve_text_model
        self._resolve_embed_scale = resolve_embed_scale
        self._inject_at_marked_positions = inject_at_marked_positions
        self._normalize_activation = normalize_activation
        self.injection_apply_count = 0

        eos_ids = {self.tokenizer.eos_token_id}
        configured = getattr(getattr(self.actor, "generation_config", None), "eos_token_id", None)
        if configured is not None:
            eos_ids.update(configured if isinstance(configured, (list, tuple)) else [configured])
        eos_ids.discard(None)
        self.eos_ids = frozenset(int(value) for value in eos_ids)
        self.provenance = {
            "nla_root": str(root),
            "nla_injection": _file_provenance(injection_path),
            "nla_schema": _file_provenance(root / "nla" / "schema.py"),
            "nla_arch_adapters": _file_provenance(root / "nla" / "arch_adapters.py"),
            "prompt_sha256": hashlib.sha256(self.prompt_text.encode("utf-8")).hexdigest(),
            "prompt_tokens": len(self.prompt_ids),
            "prompt_marker_count": self.prompt_marker_count,
            "prompt_marker_well_formed": self.prompt_marker_well_formed,
            "injection_char": self.cfg.injection_char,
            "injection_token_id": self.cfg.injection_token_id,
            "injection_left_neighbor_id": self.cfg.injection_left_neighbor_id,
            "injection_right_neighbor_id": self.cfg.injection_right_neighbor_id,
            "injection_mechanism": "scaled_marker_embedding_replacement",
            "injection_scale": float(self.cfg.injection_scale),
            "embed_scale": float(self._resolve_embed_scale(self.actor.config)),
            "eos_token_ids": sorted(self.eos_ids),
            "load_in_4bit": load_in_4bit,
            "actor_format": actor_format,
            "adapter_base_model_name_or_path": adapter_base,
            "adapter_lora_r": getattr(peft_cfg, "r", None),
            "adapter_lora_alpha": getattr(peft_cfg, "lora_alpha", None),
            "adapter_use_rslora": getattr(peft_cfg, "use_rslora", None),
        }

    def generate(self, activation: np.ndarray, torch_seed: int) -> RuntimeOutput:
        import torch

        prompt_t = torch.tensor([self.prompt_ids], dtype=torch.long, device=self.device)
        text_model = self._resolve_text_model(self.actor)
        embedding_weight = text_model.get_input_embeddings().weight
        embed_scale = self._resolve_embed_scale(self.actor.config)
        embeddings = embedding_weight[prompt_t.to(embedding_weight.device)].float() * embed_scale
        vector = torch.from_numpy(np.array(activation, dtype=np.float32, copy=True)).view(1, -1)
        vector_scaled = self._normalize_activation(vector, self.cfg.injection_scale).to(
            embeddings.device
        )
        before = self.injection_apply_count
        injected = self._inject_at_marked_positions(
            prompt_t.to(embeddings.device),
            embeddings,
            vector_scaled,
            self.cfg.injection_token_id,
            self.cfg.injection_left_neighbor_id,
            self.cfg.injection_right_neighbor_id,
        )
        marker_row_matches = torch.equal(
            injected[0, self.marker_position], vector_scaled[0].to(injected.dtype)
        )
        if not marker_row_matches:
            raise ContractError("marker embedding does not equal the scaled activation")
        self.injection_apply_count += 1
        after = self.injection_apply_count
        injected_l2_norm = float(injected[0, self.marker_position].float().norm().item())
        target_l2_norm = float(self.cfg.injection_scale)
        if not math.isclose(injected_l2_norm, target_l2_norm, rel_tol=2e-5, abs_tol=2e-3):
            raise ContractError(
                f"injected marker norm {injected_l2_norm} differs from target {target_l2_norm}"
            )
        injected = injected.to(self.actor.dtype)
        attention_mask = torch.ones(injected.shape[:2], dtype=torch.long, device=injected.device)
        with torch.no_grad(), _torch_keyed_rng(torch_seed, self.device):
            output = self.actor.generate(
                inputs_embeds=injected,
                attention_mask=attention_mask,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                top_k=0,
                repetition_penalty=1.0,
                pad_token_id=(
                    self.tokenizer.pad_token_id
                    if self.tokenizer.pad_token_id is not None
                    else self.tokenizer.eos_token_id
                ),
                return_dict_in_generate=True,
            )
        token_ids = output.sequences[0].tolist()
        eos_pos = next(
            (index for index, token_id in enumerate(token_ids) if token_id in self.eos_ids),
            None,
        )
        if eos_pos is None:
            kept_ids = token_ids
            stop_reason = "max_new_tokens" if len(token_ids) >= MAX_NEW_TOKENS else "other"
            eos_token_id = None
        else:
            kept_ids = token_ids[: eos_pos + 1]
            stop_reason = "eos"
            eos_token_id = int(token_ids[eos_pos])
        decode_kwargs = {"clean_up_tokenization_spaces": False}
        raw_text = self.tokenizer.decode(kept_ids, skip_special_tokens=False, **decode_kwargs)
        clean_text = self.tokenizer.decode(kept_ids, skip_special_tokens=True, **decode_kwargs)
        return RuntimeOutput(
            raw_text=raw_text,
            clean_text=clean_text,
            token_ids=tuple(int(value) for value in kept_ids),
            stop_reason=stop_reason,
            eos_token_id=eos_token_id,
            injection_apply_count_before=before,
            injection_apply_count_after=after,
            injection_target_l2_norm=target_l2_norm,
            injected_l2_norm=injected_l2_norm,
            marker_row_matches_scaled_activation=marker_row_matches,
        )


def _input_descriptor(items: Sequence[ActivationItem]) -> list[dict[str, Any]]:
    sources: dict[tuple[Path, Path, str | None], ActivationSource] = {}
    for item in items:
        key = (
            item.source.manifest_path,
            item.source.activation_path,
            item.source.array_key,
        )
        sources[key] = item.source
    return [
        {
            "manifest_path": str(source.manifest_path),
            "manifest_sha256": source.manifest_sha256,
            "activation_path": str(source.activation_path),
            "activation_sha256": source.activation_sha256,
            "activation_array_key": source.array_key,
            "activation_shape": list(source.array.shape),
            "activation_dtype": str(source.array.dtype),
        }
        for source in sorted(
            sources.values(), key=lambda s: (str(s.manifest_path), str(s.activation_path))
        )
    ]


def _immutable_config(args: argparse.Namespace, items: Sequence[ActivationItem]) -> dict[str, Any]:
    return {
        "schema_version": SHARD_SCHEMA,
        "model_id": MODEL_ID,
        "base_revision": args.base_revision,
        "checkpoint_id": args.checkpoint_id,
        "actor_format": getattr(args, "actor_format", "peft"),
        "condition": args.condition,
        "sampling_seed": args.sampling_seed,
        "num_samples": N_SAMPLES,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_new_tokens": MAX_NEW_TOKENS,
        "layer_index": LAYER_INDEX,
        "d_model": D_MODEL,
        "activation_norm": "none",
        "generator_source": _file_provenance(Path(__file__).resolve()),
        "checkpoint_files": _checkpoint_provenance(args.actor_checkpoint),
        "sidecar_file": _sidecar_provenance(args.sidecar),
        "native_nla_code": _native_nla_code_provenance(args.nla_root),
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "inputs": _input_descriptor(items),
    }


def _record_path(shard_dir: Path, item: ActivationItem, sample_index: int) -> Path:
    # Condition is already isolated in shard_dir.  Use activation_id in the hash
    # only to keep filenames stable if a condition directory is relocated.
    key = sha256_json({"activation_id": item.activation_id, "sample_index": sample_index})
    return shard_dir / "records" / f"{key}.json"


def _load_existing_record(
    path: Path,
    *,
    run_fingerprint: str,
    item: ActivationItem,
    condition: str,
    sample_index: int,
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        record = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"corrupt committed record {path}: {exc}") from exc
    expected = {
        "schema_version": GENERATION_SCHEMA,
        "run_fingerprint": run_fingerprint,
        "activation_id": item.activation_id,
        "condition": condition,
        "sample_index": sample_index,
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise ContractError(
                f"resume record mismatch at {path}: {key}={record.get(key)!r}, "
                f"expected {value!r}"
            )
    return record


def _failure_reason(
    parse_status: str,
    *,
    injection_delta: int,
    marker_leak: bool,
) -> str | None:
    reasons: list[str] = []
    if parse_status != "ok":
        reasons.append(f"parse:{parse_status}")
    if injection_delta != 1:
        reasons.append(f"diagnostic:injection_apply_count_{injection_delta}")
    if marker_leak:
        reasons.append("diagnostic:marker_leak")
    return ";".join(reasons) or None


def make_generation_record(
    *,
    item: ActivationItem,
    condition: str,
    checkpoint_id: str,
    run_fingerprint: str,
    sample_index: int,
    sampling_key: str,
    torch_seed: int,
    runtime_output: RuntimeOutput,
    injection_char: str,
    cjk_fraction_fn: Callable[[str], float],
) -> dict[str, Any]:
    parsed = parse_explanation(runtime_output.clean_text)
    injection_delta = (
        runtime_output.injection_apply_count_after
        - runtime_output.injection_apply_count_before
    )
    marker_leak = injection_char in runtime_output.clean_text
    cjk_value = float(cjk_fraction_fn(runtime_output.clean_text))
    # CJK is a distributional smoke signal, not a mechanism-level proof: MMMLU
    # contains Chinese/Japanese/Korean source activations by construction.  The
    # exact injection-call counter and literal marker leak remain strict.
    cjk_suspect = (
        item.benchmark != "mmmlu" and cjk_value > CJK_SUSPECT_FRACTION
    )
    reason = _failure_reason(
        parsed.status,
        injection_delta=injection_delta,
        marker_leak=marker_leak,
    )
    vector = np.asarray(item.source.array[item.activation_index], dtype=np.float32)
    return {
        "schema_version": GENERATION_SCHEMA,
        "run_fingerprint": run_fingerprint,
        "activation_id": item.activation_id,
        "activation_index": item.activation_index,
        "document_id": item.document_id,
        "trajectory_id": item.trajectory_id,
        "cluster_id": item.cluster_id,
        "distribution": item.distribution,
        "benchmark": item.benchmark,
        "family": item.family,
        "condition": condition,
        "sample_index": sample_index,
        "sampling_key": sampling_key,
        "torch_seed": torch_seed,
        "model_id": MODEL_ID,
        "checkpoint_id": checkpoint_id,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_new_tokens": MAX_NEW_TOKENS,
        "explanation": parsed.explanation,
        "generation_ok": reason is None,
        "failure_reason": reason,
        "parse_status": {
            "status": parsed.status,
            "open_tag_count": parsed.open_tag_count,
            "close_tag_count": parsed.close_tag_count,
        },
        "diagnostics": {
            "injection_mechanism": "scaled_marker_embedding_replacement",
            "injection_apply_count_before": runtime_output.injection_apply_count_before,
            "injection_apply_count_after": runtime_output.injection_apply_count_after,
            "injection_apply_count_delta": injection_delta,
            "injection_applied_exactly_once": injection_delta == 1,
            "injection_target_l2_norm": runtime_output.injection_target_l2_norm,
            "injected_l2_norm": runtime_output.injected_l2_norm,
            "marker_row_matches_scaled_activation": (
                runtime_output.marker_row_matches_scaled_activation
            ),
            "marker_leak": marker_leak,
            "cjk_fraction": cjk_value,
            "cjk_suspect_threshold": CJK_SUSPECT_FRACTION,
            "cjk_distribution_exempt": item.benchmark == "mmmlu",
            "cjk_suspect": cjk_suspect,
            "activation_all_finite": bool(np.isfinite(vector).all()),
            "activation_l2_norm": float(np.linalg.norm(vector.astype(np.float64))),
            "activation_storage_norm": "none",
            "activation_inference_transform": "l2_rescale_to_sidecar_injection_scale",
        },
        "raw_output": {
            "text_with_special_tokens": runtime_output.raw_text,
            "text_skip_special_tokens": runtime_output.clean_text,
            "token_ids": list(runtime_output.token_ids),
            "n_tokens": len(runtime_output.token_ids),
            "stop_reason": runtime_output.stop_reason,
            "eos_token_id": runtime_output.eos_token_id,
        },
        "input_provenance": {
            "manifest_path": str(item.source.manifest_path),
            "manifest_sha256": item.source.manifest_sha256,
            "activation_path": str(item.source.activation_path),
            "activation_sha256": item.source.activation_sha256,
            "activation_array_key": item.source.array_key,
            "source_metadata": item.source_metadata,
        },
        "committed_at_unix": time.time(),
    }


def make_error_record(
    *,
    item: ActivationItem,
    condition: str,
    checkpoint_id: str,
    run_fingerprint: str,
    sample_index: int,
    sampling_key: str,
    torch_seed: int,
    exc: Exception,
) -> dict[str, Any]:
    return {
        "schema_version": GENERATION_SCHEMA,
        "run_fingerprint": run_fingerprint,
        "activation_id": item.activation_id,
        "activation_index": item.activation_index,
        "document_id": item.document_id,
        "trajectory_id": item.trajectory_id,
        "cluster_id": item.cluster_id,
        "distribution": item.distribution,
        "benchmark": item.benchmark,
        "family": item.family,
        "condition": condition,
        "sample_index": sample_index,
        "sampling_key": sampling_key,
        "torch_seed": torch_seed,
        "model_id": MODEL_ID,
        "checkpoint_id": checkpoint_id,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_new_tokens": MAX_NEW_TOKENS,
        "explanation": "",
        "generation_ok": False,
        "failure_reason": "generation:error",
        "parse_status": {"status": "not_generated", "open_tag_count": 0, "close_tag_count": 0},
        "diagnostics": {},
        "raw_output": None,
        "error": {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        },
        "input_provenance": {
            "manifest_path": str(item.source.manifest_path),
            "manifest_sha256": item.source.manifest_sha256,
            "activation_path": str(item.source.activation_path),
            "activation_sha256": item.source.activation_sha256,
            "activation_array_key": item.source.array_key,
            "source_metadata": item.source_metadata,
        },
        "committed_at_unix": time.time(),
    }


def explanation_contract_row(record: Mapping[str, Any]) -> dict[str, Any]:
    """Projection consumed by the claim decomposition/agreement pipeline."""
    keys = (
        "activation_id",
        "condition",
        "sample_index",
        "document_id",
        "trajectory_id",
        "cluster_id",
        "distribution",
        "benchmark",
        "family",
        "model_id",
        "checkpoint_id",
        "temperature",
        "top_p",
        "max_new_tokens",
        "explanation",
        "generation_ok",
        "failure_reason",
    )
    return {"schema_version": EXPLANATION_SCHEMA, **{key: record[key] for key in keys}}


def _validate_activation(item: ActivationItem) -> np.ndarray:
    vector = np.asarray(item.source.array[item.activation_index])
    if vector.shape != (D_MODEL,):
        raise ContractError(
            f"{item.activation_id}: activation has shape {vector.shape}, expected {(D_MODEL,)}"
        )
    if not np.isfinite(vector).all():
        bad = int(vector.size - np.isfinite(vector).sum())
        raise ContractError(f"{item.activation_id}: activation contains {bad} non-finite values")
    norm = float(np.linalg.norm(vector.astype(np.float64)))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ContractError(f"{item.activation_id}: activation has invalid L2 norm {norm}")
    return vector


def _read_all_committed(
    shard_dir: Path,
    selected: Sequence[ActivationItem],
    condition: str,
    run_fingerprint: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    expected_paths: set[Path] = set()
    for item in selected:
        for sample_index in range(N_SAMPLES):
            path = _record_path(shard_dir, item, sample_index)
            expected_paths.add(path)
            record = _load_existing_record(
                path,
                run_fingerprint=run_fingerprint,
                item=item,
                condition=condition,
                sample_index=sample_index,
            )
            if record is None:
                raise ContractError(f"missing expected committed record: {path}")
            rows.append(record)
    extras = set((shard_dir / "records").glob("*.json")) - expected_paths
    if extras:
        raise ContractError(
            f"record directory contains {len(extras)} unexpected files; first={min(extras)}"
        )
    return rows


def _progress_summary(records: Sequence[Mapping[str, Any]], expected: int) -> dict[str, Any]:
    failures: dict[str, int] = {}
    for record in records:
        reason = record.get("failure_reason")
        if reason:
            for code in str(reason).split(";"):
                failures[code] = failures.get(code, 0) + 1
    cjk_suspects = sum(
        bool((record.get("diagnostics") or {}).get("cjk_suspect"))
        for record in records
    )
    cjk_fraction = cjk_suspects / len(records) if records else 0.0
    return {
        "expected_records": expected,
        "committed_records": len(records),
        "generation_ok": sum(bool(r.get("generation_ok")) for r in records),
        "generation_failed": sum(not bool(r.get("generation_ok")) for r in records),
        "failure_counts": dict(sorted(failures.items())),
        "cjk_suspect_records": cjk_suspects,
        "cjk_suspect_record_fraction": cjk_fraction,
        "cjk_shard_alert_fraction": CJK_SHARD_ALERT_FRACTION,
        "cjk_shard_alert_min_count": CJK_SHARD_ALERT_MIN_COUNT,
        "cjk_manual_audit_alert": (
            cjk_suspects >= CJK_SHARD_ALERT_MIN_COUNT
            and cjk_fraction > CJK_SHARD_ALERT_FRACTION
        ),
        "updated_at_unix": time.time(),
    }


def run_shard(
    args: argparse.Namespace,
    *,
    runtime_factory: Callable[..., Any] = NativeNLARuntime,
) -> dict[str, Any]:
    items = load_activation_manifests(
        args.input_manifest, verify_sha256=not args.skip_array_sha256
    )
    selected = sorted(
        (
            item
            for item in items
            if shard_for_activation(item.activation_id, args.num_shards) == args.shard_index
        ),
        key=lambda item: item.activation_id,
    )
    if not selected and not args.allow_empty_shard:
        raise ContractError(
            f"shard {args.shard_index}/{args.num_shards} selects zero activations"
        )
    immutable = _immutable_config(args, items)
    run_fingerprint = sha256_json(immutable)
    shard_name = f"shard-{args.shard_index:05d}-of-{args.num_shards:05d}"
    shard_dir = Path(args.output_dir).expanduser().resolve() / args.condition / shard_name
    records_dir = shard_dir / "records"
    records_dir.mkdir(parents=True, exist_ok=True)
    for stale in shard_dir.glob(".*.tmp-*"):
        stale.unlink()
    for stale in records_dir.glob(".*.tmp-*"):
        stale.unlink()

    config_path = shard_dir / "shard_config.json"
    config_record = {
        **immutable,
        "run_fingerprint": run_fingerprint,
        "actor_checkpoint": args.actor_checkpoint,
        "sidecar": args.sidecar,
        "nla_root": str(Path(args.nla_root).expanduser().resolve()),
        "device": args.device,
        "load_in_4bit": args.load_in_4bit,
        "actor_format": getattr(args, "actor_format", "peft"),
    }
    if config_path.exists():
        existing = json.loads(config_path.read_bytes())
        if existing.get("run_fingerprint") != run_fingerprint:
            raise ContractError(
                f"output shard belongs to a different run: {config_path} has "
                f"{existing.get('run_fingerprint')}, expected {run_fingerprint}"
            )
        # The fingerprint already covers source/checkpoint/sidecar/native NLA
        # digests.  Also keep runtime path/precision changes fail-closed rather
        # than silently reusing records made by a different launch mode.
        for key in ("actor_checkpoint", "sidecar", "load_in_4bit", "actor_format"):
            if existing.get(key) != config_record.get(key):
                raise ContractError(
                    f"resume runtime mismatch in {config_path}: {key}="
                    f"{existing.get(key)!r}, current={config_record.get(key)!r}"
                )
    else:
        atomic_write_json(config_path, config_record)

    # Validate all selected activations before allocating a 7B model.  A corrupt
    # vector is a critical input failure, not a sample-level model outcome.
    for item in selected:
        _validate_activation(item)

    missing: list[tuple[ActivationItem, int]] = []
    resumed_records: list[dict[str, Any]] = []
    for item in selected:
        for sample_index in range(N_SAMPLES):
            path = _record_path(shard_dir, item, sample_index)
            existing = _load_existing_record(
                path,
                run_fingerprint=run_fingerprint,
                item=item,
                condition=args.condition,
                sample_index=sample_index,
            )
            if existing is None:
                missing.append((item, sample_index))
            else:
                resumed_records.append(existing)
    atomic_write_json(
        shard_dir / "progress.json",
        {
            "schema_version": SHARD_SCHEMA,
            "run_fingerprint": run_fingerprint,
            "shard": shard_name,
            "selected_activations": len(selected),
            "resumed_records": len(resumed_records),
            "remaining_records": len(missing),
            **_progress_summary(resumed_records, len(selected) * N_SAMPLES),
        },
    )
    definitive_diagnostic = next(
        (
            str(record.get("failure_reason"))
            for record in resumed_records
            if "diagnostic:marker_leak" in str(record.get("failure_reason"))
            or "diagnostic:injection_apply_count_" in str(record.get("failure_reason"))
        ),
        None,
    )
    if definitive_diagnostic is not None:
        raise ContractError(
            "committed resume records contain a definitive injection diagnostic; "
            f"manual investigation required: {definitive_diagnostic}"
        )

    runtime = None
    if missing:
        runtime = runtime_factory(
            base_model=MODEL_ID,
            base_revision=args.base_revision,
            actor_checkpoint=args.actor_checkpoint,
            sidecar=args.sidecar,
            nla_root=Path(args.nla_root),
            device=args.device,
            load_in_4bit=args.load_in_4bit,
            actor_format=getattr(args, "actor_format", "peft"),
        )
        # Add runtime-resolved token/injection provenance after successful model
        # construction; preserve the original launch config as well.
        config_record["runtime_provenance"] = runtime.provenance
        atomic_write_json(config_path, config_record)

    committed = list(resumed_records)
    for ordinal, (item, sample_index) in enumerate(missing, start=1):
        key, torch_seed = sampling_identity(
            item.activation_id, args.condition, sample_index, args.sampling_seed
        )
        path = _record_path(shard_dir, item, sample_index)
        try:
            runtime_output = runtime.generate(
                np.asarray(item.source.array[item.activation_index]), torch_seed
            )
            record = make_generation_record(
                item=item,
                condition=args.condition,
                checkpoint_id=args.checkpoint_id,
                run_fingerprint=run_fingerprint,
                sample_index=sample_index,
                sampling_key=key,
                torch_seed=torch_seed,
                runtime_output=runtime_output,
                injection_char=runtime.cfg.injection_char,
                cjk_fraction_fn=runtime.cjk_fraction,
            )
        except Exception as exc:
            record = make_error_record(
                item=item,
                condition=args.condition,
                checkpoint_id=args.checkpoint_id,
                run_fingerprint=run_fingerprint,
                sample_index=sample_index,
                sampling_key=key,
                torch_seed=torch_seed,
                exc=exc,
            )
            atomic_write_json(path, record)
            committed.append(record)
            atomic_write_json(
                shard_dir / "progress.json",
                {
                    "schema_version": SHARD_SCHEMA,
                    "run_fingerprint": run_fingerprint,
                    "shard": shard_name,
                    "selected_activations": len(selected),
                    "remaining_records": len(missing) - ordinal,
                    **_progress_summary(committed, len(selected) * N_SAMPLES),
                },
            )
            raise
        atomic_write_json(path, record)
        committed.append(record)
        atomic_write_json(
            shard_dir / "progress.json",
            {
                "schema_version": SHARD_SCHEMA,
                "run_fingerprint": run_fingerprint,
                "shard": shard_name,
                "selected_activations": len(selected),
                "remaining_records": len(missing) - ordinal,
                **_progress_summary(committed, len(selected) * N_SAMPLES),
            },
        )
        print(
            f"[{shard_name}] {len(committed)}/{len(selected) * N_SAMPLES} "
            f"{item.activation_id} sample={sample_index} ok={record['generation_ok']} "
            f"parse={record['parse_status']['status']}",
            flush=True,
        )
        if (
            "diagnostic:marker_leak" in str(record.get("failure_reason"))
            or "diagnostic:injection_apply_count_" in str(record.get("failure_reason"))
        ):
            raise ContractError(
                "definitive injection diagnostic was committed; stopping before "
                f"further generation: {record['failure_reason']}"
            )

    records = _read_all_committed(
        shard_dir, selected, args.condition, run_fingerprint
    )
    records.sort(key=lambda row: (row["activation_id"], row["sample_index"]))
    raw_path = shard_dir / "raw_generations.jsonl"
    explanation_path = shard_dir / "explanations.jsonl"
    atomic_write_jsonl(raw_path, records)
    atomic_write_jsonl(
        explanation_path, (explanation_contract_row(record) for record in records)
    )
    summary = {
        "schema_version": SHARD_SCHEMA,
        "run_fingerprint": run_fingerprint,
        "shard": shard_name,
        "selected_activations": len(selected),
        "raw_generations": {
            "path": str(raw_path),
            "bytes": raw_path.stat().st_size,
            "sha256": sha256_file(raw_path),
        },
        "explanations": {
            "path": str(explanation_path),
            "bytes": explanation_path.stat().st_size,
            "sha256": sha256_file(explanation_path),
        },
        **_progress_summary(records, len(selected) * N_SAMPLES),
        "complete": True,
    }
    atomic_write_json(shard_dir / "complete.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    component_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Generate two keyed NLA AV samples per raw Qwen L20 activation."
    )
    parser.add_argument(
        "--input-manifest",
        action="append",
        required=True,
        help="Repeatable activation-manifest JSON path.",
    )
    parser.add_argument("--condition", choices=("plausible", "implausible"), required=True)
    parser.add_argument(
        "--actor-checkpoint",
        required=True,
        help="Local PEFT adapter, full-HF checkpoint, or compatible HF repo ID.",
    )
    parser.add_argument(
        "--actor-format",
        choices=("peft", "full_hf"),
        default="peft",
        help="Actor storage format; supplied Drive actors are converted to full-HF.",
    )
    parser.add_argument(
        "--checkpoint-id",
        required=True,
        help="Immutable checkpoint identity recorded in scorer rows (commit/hash/step).",
    )
    parser.add_argument(
        "--sidecar",
        required=True,
        help=(
            "NLA save root, explicit nla_meta.yaml, or parquet whose sidecar is "
            "the contract; actor iteration directories need not contain it."
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--sampling-seed", type=int, default=42)
    parser.add_argument(
        "--base-revision",
        default=BASE_REVISION,
        help="Pinned Qwen base revision; pass an empty string only for a local base path.",
    )
    parser.add_argument(
        "--nla-root",
        default=str(component_dir.parents[2]),
        help="Repository root providing native scaled marker-embedding injection.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Optional memory-saving mode; default bf16 matches this experiment's actor training.",
    )
    parser.add_argument(
        "--skip-array-sha256",
        action="store_true",
        help="Skip hashing only when the manifest omits activation_sha256; expected hashes are always checked.",
    )
    parser.add_argument("--allow-empty-shard", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.num_shards <= 0:
        raise ContractError("--num-shards must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        raise ContractError(
            f"--shard-index must be in [0, {args.num_shards}), got {args.shard_index}"
        )
    if args.sampling_seed < 0:
        raise ContractError("--sampling-seed must be nonnegative")
    if args.base_revision == "":
        args.base_revision = None


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_args(args)
    summary = run_shard(args)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    # Marker/injection failures are mechanism diagnostics, not ordinary malformed
    # samples.  A widespread CJK signal also requests manual review, but CJK
    # alone never changes an individual sample's generation_ok value.  Parse-only
    # misses remain valid failures for the downstream denominator.
    critical_prefixes = ("diagnostic:marker_leak", "diagnostic:injection_apply_count_")
    if any(
        code.startswith(critical_prefixes)
        for code in summary.get("failure_counts", {})
    ):
        return 2
    if summary.get("cjk_manual_audit_alert"):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
