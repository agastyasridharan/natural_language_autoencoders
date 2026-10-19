"""Pure contracts shared by the exact FVE evaluator and checkpoint selector."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import time
import uuid
import zlib
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import yaml


EVALUATION_SCHEMA = "nla.warm_start.exact_checkpoint_fve.v1"
GENERATION_SCHEMA = "nla.warm_start.exact_checkpoint_generations.v1"
SELECTION_SCHEMA = "nla.warm_start.fve_matched_selection.v1"
EVAL_SET_SCHEMA = "nla.warm_start.fixed_heldout_256.v1"
TRANSACTION_SCHEMA = "nla.warm_start.immutable_transaction_audit.v1"

FINAL_STEP = 780
CHECKPOINT_INTERVAL = 20
EVAL_N_PROMPTS = 256
EVAL_TEMPERATURE = 0.0
MAX_NEW_TOKENS = 150
CRITIC_MAX_TOKENS = 1024
CRITIC_BATCH_SIZE = 32


class ContractError(ValueError):
    """An input cannot support an exact, auditable comparison."""


@dataclasses.dataclass(frozen=True)
class Transaction:
    root: Path
    step: int
    actor: Path
    critic: Path
    run_config_path: Path
    sidecar_path: Path
    marker_path: Path
    run_config: Mapping[str, Any]
    relevant_files: tuple[Mapping[str, Any], ...]
    fingerprint: str


@dataclasses.dataclass(frozen=True)
class HeldoutRow:
    parquet_row_index: int
    document_id: str
    prompt: Any
    activation: np.ndarray
    source_text: str


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def sha256_file(path: Path, chunk_bytes: int = 16 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


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


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        with tmp.open("xb") as handle:
            handle.write(canonical_json(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        _fsync_directory(path.parent)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def checkpoint_state_mapping(
    transaction_step: int,
    *,
    eval_every: int,
    save_every: int = CHECKPOINT_INTERVAL,
) -> dict[str, Any]:
    """Map a saved transaction to the trainer's zero-based update/eval labels.

    The loop evaluates after update ``step`` and saves ``iter_{step+1}``. Thus
    ``eval@N`` is the state that would be called ``iter_{N+1}``, never
    ``iter_N``. A transaction at N contains N completed optimizer updates and
    was saved immediately after zero-based update N-1.
    """
    if transaction_step <= 0:
        raise ContractError("transaction_step must be positive")
    if eval_every <= 0 or save_every <= 0:
        raise ContractError("eval_every and save_every must be positive")
    last_update = transaction_step - 1
    return {
        "transaction_step": transaction_step,
        "actor_iteration": f"iter_{transaction_step:06d}",
        "optimizer_updates_completed": transaction_step,
        "zero_based_last_update_index": last_update,
        "same_state_log_eval_label": last_update,
        "same_state_log_eval_was_scheduled": last_update % eval_every == 0,
        "logged_eval_step_N_maps_to_transaction_step": "N+1",
        "checkpoint_was_scheduled": transaction_step % save_every == 0,
        "warning": (
            f"eval@{transaction_step} is the post-update-{transaction_step} state "
            f"(transaction {transaction_step + 1}), not {transaction_step}; "
            f"transaction {transaction_step} is post-update-{last_update}."
        ),
    }


def _required_file(path: Path, context: str) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        raise ContractError(f"{context}: required nonempty file is missing: {path}")


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read JSON {path}: {exc}") from exc


def _transaction_step(root: Path) -> tuple[int, Path, Mapping[str, Any]]:
    complete = root / "COMPLETE.json"
    ready = root / "READY.json"
    if complete.is_file() and ready.is_file():
        raise ContractError(f"{root}: both COMPLETE.json and READY.json exist")
    if complete.is_file():
        marker = _load_json(complete)
        step = marker.get("step")
        marker_path = complete
    elif ready.is_file():
        marker = _load_json(ready)
        step = marker.get("checkpoint_step", marker.get("step"))
        marker_path = ready
    else:
        raise ContractError(f"{root}: missing COMPLETE.json/READY.json transaction marker")
    if isinstance(step, bool) or not isinstance(step, int) or step <= 0:
        raise ContractError(f"{marker_path}: invalid transaction step {step!r}")
    expected_name = f"step_{step:06d}"
    if root.name != expected_name:
        raise ContractError(f"transaction directory {root.name!r} != {expected_name!r}")
    return step, marker_path, marker


def _verify_complete_inventory(
    root: Path, marker_path: Path, marker: Mapping[str, Any]
) -> None:
    if marker_path.name != "COMPLETE.json":
        return
    rows = marker.get("inventory")
    if not isinstance(rows, list) or not rows:
        raise ContractError(f"{marker_path}: inventory must be nonempty")
    declared: set[str] = set()
    for ordinal, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ContractError(f"{marker_path}: inventory[{ordinal}] is not an object")
        rel = row.get("path")
        size = row.get("size")
        if not isinstance(rel, str) or not rel or rel in declared:
            raise ContractError(f"{marker_path}: invalid/duplicate inventory path {rel!r}")
        if Path(rel).is_absolute() or ".." in Path(rel).parts:
            raise ContractError(f"{marker_path}: inventory path escapes transaction: {rel}")
        path = root / rel
        if not path.is_file() or path.stat().st_size != size:
            raise ContractError(
                f"{marker_path}: inventory mismatch for {rel}: "
                f"actual={path.stat().st_size if path.is_file() else None}, declared={size}"
            )
        declared.add(rel)
    if marker.get("files_before_marker") != len(rows):
        raise ContractError(f"{marker_path}: files_before_marker does not match inventory")
    actual = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and path != marker_path
    }
    if actual != declared:
        raise ContractError(
            f"{marker_path}: actual files differ from immutable inventory; "
            f"missing={sorted(declared - actual)}, extra={sorted(actual - declared)}"
        )


def _verify_metrics(metrics_path: Path, step: int) -> None:
    _required_file(metrics_path, "transaction")
    rows = []
    try:
        for line in metrics_path.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"unreadable transaction metrics: {exc}") from exc
    steps = [row.get("step") for row in rows]
    if steps != list(range(step)):
        raise ContractError(
            f"{metrics_path}: expected contiguous metric steps 0..{step - 1}, "
            f"got {len(steps)} rows ending at {steps[-1] if steps else None}"
        )


def audit_transaction(root: Path, *, hash_relevant_files: bool = True) -> Transaction:
    """Validate an immutable actor+historical-critic transaction.

    The fingerprint covers every actor/critic file plus the run config, sidecar,
    metric history, and completion marker. Optimizer state is required for the
    transaction's integrity but is not hashed because it cannot affect FVE.
    """
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise ContractError(f"transaction directory does not exist: {root}")
    if any(path.name.endswith((".tmp", ".old")) for path in root.rglob("*")):
        raise ContractError(f"{root}: transaction contains temporary/old remnants")
    step, marker_path, marker = _transaction_step(root)
    _verify_complete_inventory(root, marker_path, marker)
    if marker_path.name == "COMPLETE.json" and marker.get("actor_source") != f"iter_{step:06d}":
        raise ContractError(
            f"{marker_path}: actor_source={marker.get('actor_source')!r} does not "
            f"identify iter_{step:06d}"
        )

    actor = root / "actor"
    critic = root / "critic"
    run_config_path = root / "run_config.yaml"
    sidecar_path = root / "nla_meta.yaml"
    metrics_path = root / "metrics.jsonl"
    optimizer_path = root / "optim.pt"
    required = (
        actor / "adapter_config.json",
        actor / "adapter_model.safetensors",
        actor / "reference" / "adapter_config.json",
        actor / "reference" / "adapter_model.safetensors",
        critic / "config.json",
        critic / "value_head.safetensors",
        critic / "saved_at_step.txt",
        run_config_path,
        sidecar_path,
        metrics_path,
        optimizer_path,
        marker_path,
    )
    for path in required:
        _required_file(path, "transaction")
    if not (
        (critic / "model.safetensors").is_file()
        or (critic / "model.safetensors.index.json").is_file()
    ):
        raise ContractError(f"{critic}: missing critic model safetensors/index")
    try:
        critic_step = int((critic / "saved_at_step.txt").read_text().strip())
    except (OSError, ValueError) as exc:
        raise ContractError(f"{critic}: invalid saved_at_step.txt") from exc
    if critic_step != step:
        raise ContractError(
            f"historical critic is at step {critic_step}, actor transaction is {step}"
        )
    _verify_metrics(metrics_path, step)
    try:
        run_config = yaml.safe_load(run_config_path.read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise ContractError(f"cannot read run config: {exc}") from exc
    if not isinstance(run_config, dict):
        raise ContractError(f"{run_config_path}: top level must be a mapping")

    relevant_paths = sorted(
        [path for path in actor.rglob("*") if path.is_file()]
        + [path for path in critic.rglob("*") if path.is_file()]
        + [run_config_path, sidecar_path, metrics_path, marker_path]
    )
    inventory = []
    for path in relevant_paths:
        row = {
            "path": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
        }
        row["sha256"] = sha256_file(path) if hash_relevant_files else None
        inventory.append(row)
    fingerprint = sha256_json(
        {
            "schema_version": TRANSACTION_SCHEMA,
            "step": step,
            "files": inventory,
        }
    )
    return Transaction(
        root=root,
        step=step,
        actor=actor,
        critic=critic,
        run_config_path=run_config_path,
        sidecar_path=sidecar_path,
        marker_path=marker_path,
        run_config=run_config,
        relevant_files=tuple(inventory),
        fingerprint=fingerprint,
    )


def validate_locked_run_config(config: Mapping[str, Any], step: int) -> dict[str, Any]:
    expected = {
        "eval_n_prompts": EVAL_N_PROMPTS,
        "eval_temperature": EVAL_TEMPERATURE,
        "max_new_tokens": MAX_NEW_TOKENS,
        "save_every": CHECKPOINT_INTERVAL,
        "seed": 42,
        "train_critic": True,
        "vllm_gpu_mem": 0.25,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ContractError(
                f"run_config {key}={config.get(key)!r}; exact protocol requires {value!r}"
            )
    for key in ("rl_parquet", "av_ckpt", "base_ckpt", "av_adapter"):
        if not isinstance(config.get(key), str) or not str(config[key]).strip():
            raise ContractError(f"run_config lacks nonempty {key}")
    if int(config.get("vllm_tp", -1)) != 1:
        raise ContractError("exact historical eval used vllm_tp=1")
    if int(config.get("vllm_max_len", -1)) != 512:
        raise ContractError("exact historical eval used vllm_max_len=512")
    if int(config.get("val_rows", -1)) != 10_000:
        raise ContractError("this experiment requires val_rows=10000")
    if step > int(config.get("num_steps", -1)):
        raise ContractError(f"transaction step {step} exceeds configured num_steps")
    eval_every = int(config.get("eval_every", 0))
    mapping = checkpoint_state_mapping(
        step, eval_every=eval_every, save_every=int(config["save_every"])
    )
    if not mapping["checkpoint_was_scheduled"] and step != int(config["num_steps"]):
        raise ContractError(f"transaction {step} is not on the frozen checkpoint grid")
    return mapping


def validate_resolved_doc_hash_split(
    config: Mapping[str, Any], total_rows: int
) -> dict[str, int]:
    """Validate pre-auto or post-auto resolved values for the frozen split.

    ``save_resolved_config`` runs after the trainer mutates zero-valued auto
    arguments, so real transactions normally store ``max_rows=total_rows`` and
    ``eval_skip_rows=total_rows-val_rows`` rather than the recipe's zeros.
    """
    val_rows = int(config.get("val_rows", -1))
    expected_max = total_rows
    expected_skip = max(1, total_rows - val_rows)
    actual_max = int(config.get("max_rows", 0))
    actual_skip = int(config.get("eval_skip_rows", 0))
    if actual_max > 0 and actual_max != expected_max:
        raise ContractError(
            f"resolved max_rows={actual_max}, expected full corpus {expected_max}"
        )
    if actual_skip > 0 and actual_skip != expected_skip:
        raise ContractError(
            f"resolved eval_skip_rows={actual_skip}, expected auto value {expected_skip}"
        )
    return {
        "total_rows": total_rows,
        "val_rows": val_rows,
        "resolved_max_rows": actual_max,
        "resolved_eval_skip_rows": actual_skip,
        "expected_auto_max_rows": expected_max,
        "expected_auto_eval_skip_rows": expected_skip,
    }


def val_doc_permille(val_rows: int, total_rows: int) -> int:
    return max(1, min(500, round(1000 * val_rows / max(1, total_rows))))


def is_val_doc(document_id: Any, permille: int) -> bool:
    return (zlib.crc32(str(document_id).encode("utf-8")) % 1000) < permille


def load_fixed_heldout(
    parquet_path: Path,
    *,
    val_rows: int,
    n_prompts: int = EVAL_N_PROMPTS,
    val_permille_fn: Callable[[int, int], int] = val_doc_permille,
    is_val_doc_fn: Callable[[Any, int], bool] = is_val_doc,
) -> tuple[list[HeldoutRow], int, int]:
    """Reproduce the trainer's first-N held-out-document rows exactly."""
    import pyarrow.parquet as pq

    parquet_path = parquet_path.expanduser().resolve()
    if not parquet_path.is_file():
        raise ContractError(f"RL parquet does not exist: {parquet_path}")
    pf = pq.ParquetFile(parquet_path)
    total_rows = int(pf.metadata.num_rows)
    permille = int(val_permille_fn(val_rows, total_rows))
    required_columns = {"prompt", "activation_vector", "doc_id"}
    missing = required_columns - set(pf.schema_arrow.names)
    if missing:
        raise ContractError(f"{parquet_path}: missing columns {sorted(missing)}")
    source_column = (
        ["detokenized_text_truncated"]
        if "detokenized_text_truncated" in pf.schema_arrow.names
        else []
    )
    rows: list[HeldoutRow] = []
    row_offset = 0
    for row_group_index in range(pf.num_row_groups):
        if len(rows) >= n_prompts:
            break
        table = pf.read_row_group(
            row_group_index,
            columns=["prompt", "activation_vector", "doc_id"] + source_column,
        )
        prompts = table.column("prompt").to_pylist()
        documents = table.column("doc_id").to_pylist()
        activations = np.asarray(
            table.column("activation_vector").combine_chunks().flatten(),
            dtype=np.float32,
        ).reshape(len(prompts), -1)
        sources = (
            table.column("detokenized_text_truncated").to_pylist()
            if source_column
            else [""] * len(prompts)
        )
        for index, document_id in enumerate(documents):
            if not is_val_doc_fn(document_id, permille):
                continue
            activation = np.ascontiguousarray(activations[index], dtype=np.float32)
            if activation.ndim != 1 or activation.size == 0:
                raise ContractError(
                    f"RL row {row_offset + index}: invalid activation shape {activation.shape}"
                )
            if not np.isfinite(activation).all():
                raise ContractError(f"RL row {row_offset + index}: non-finite activation")
            rows.append(
                HeldoutRow(
                    parquet_row_index=row_offset + index,
                    document_id=str(document_id),
                    prompt=prompts[index],
                    activation=activation,
                    source_text=sources[index] or "",
                )
            )
            if len(rows) >= n_prompts:
                break
        row_offset += table.num_rows
    if len(rows) != n_prompts:
        raise ContractError(
            f"held-out split produced {len(rows)} prompts, expected exactly {n_prompts}"
        )
    widths = {row.activation.shape for row in rows}
    if len(widths) != 1:
        raise ContractError(f"held-out activations have inconsistent shapes: {widths}")
    return rows, total_rows, permille


def heldout_descriptor(rows: Sequence[HeldoutRow]) -> dict[str, Any]:
    if len(rows) != EVAL_N_PROMPTS:
        raise ContractError(f"expected {EVAL_N_PROMPTS} held-out rows, got {len(rows)}")
    digest = hashlib.sha256()
    digest.update(EVAL_SET_SCHEMA.encode("ascii") + b"\0")
    items = []
    for ordinal, row in enumerate(rows):
        activation = np.ascontiguousarray(row.activation.astype("<f4", copy=False))
        metadata = {
            "ordinal": ordinal,
            "parquet_row_index": row.parquet_row_index,
            "document_id": row.document_id,
            "prompt": row.prompt,
        }
        encoded = canonical_json(metadata)
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(len(activation.tobytes()).to_bytes(8, "big"))
        digest.update(activation.tobytes())
        items.append(
            {
                "ordinal": ordinal,
                "parquet_row_index": row.parquet_row_index,
                "document_id": row.document_id,
                "prompt_sha256": sha256_json(row.prompt),
                "activation_sha256": hashlib.sha256(activation.tobytes()).hexdigest(),
                "activation_l2_norm": float(
                    np.linalg.norm(activation.astype(np.float64))
                ),
                "source_text_sha256": hashlib.sha256(
                    row.source_text.encode("utf-8")
                ).hexdigest(),
            }
        )
    return {
        "schema_version": EVAL_SET_SCHEMA,
        "n_prompts": len(rows),
        "d_model": int(rows[0].activation.size),
        "eval_set_sha256": digest.hexdigest(),
        "items": items,
    }


def aggregate_fve(rewards: Sequence[float | None], baseline_mse: float) -> dict[str, Any]:
    """Aggregate with the trainer eval's exact failed-sample convention."""
    if not rewards:
        raise ContractError("cannot aggregate an empty evaluation")
    if not math.isfinite(baseline_mse) or baseline_mse <= 0:
        raise ContractError(f"invalid held-out FVE baseline {baseline_mse}")
    filled = [-2.0 if reward is None else float(reward) for reward in rewards]
    valid = [reward for reward in filled if math.isfinite(reward) and reward > -2.0]
    if not valid:
        raise ContractError("evaluation has no valid critic rewards")
    return {
        "n_prompts": len(rewards),
        "n_valid_for_fve": len(valid),
        "n_failed_or_out_of_range": len(rewards) - len(valid),
        "reward_mean_with_failure_floor": float(np.mean(filled)),
        "reward_mean_valid": float(np.mean(valid)),
        "baseline_mse": float(baseline_mse),
        "fve_pct": (1.0 - (-float(np.mean(valid))) / baseline_mse) * 100.0,
    }


def _result_identity(result: Mapping[str, Any], path: Path) -> tuple[str, int]:
    if result.get("schema_version") != EVALUATION_SCHEMA:
        raise ContractError(f"{path}: wrong evaluation schema")
    if result.get("status") != "complete":
        raise ContractError(f"{path}: evaluation is not complete")
    condition = result.get("condition")
    if condition not in ("plausible", "implausible"):
        raise ContractError(f"{path}: invalid condition {condition!r}")
    step = (result.get("checkpoint") or {}).get("transaction_step")
    if isinstance(step, bool) or not isinstance(step, int):
        raise ContractError(f"{path}: missing integer transaction step")
    mapping = result.get("checkpoint_state_mapping") or {}
    if mapping.get("zero_based_last_update_index") != step - 1:
        raise ContractError(f"{path}: checkpoint/eval-step mapping is inconsistent")
    metric = (result.get("metrics") or {}).get("fve_pct")
    if not isinstance(metric, (int, float)) or not math.isfinite(float(metric)):
        raise ContractError(f"{path}: non-finite FVE")
    checkpoint = result.get("checkpoint") or {}
    if checkpoint.get("critic_saved_at_step") != step:
        raise ContractError(f"{path}: critic is not contemporaneous with actor")
    fingerprint = checkpoint.get("transaction_fingerprint")
    if (
        not isinstance(fingerprint, str)
        or len(fingerprint) != 64
        or any(char not in "0123456789abcdef" for char in fingerprint)
    ):
        raise ContractError(f"{path}: missing transaction fingerprint")
    for key in (
        "base_checkpoint_tree_sha256",
        "merged_av_checkpoint_tree_sha256",
    ):
        digest = checkpoint.get(key)
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            raise ContractError(f"{path}: missing exact {key}")
    inventory = checkpoint.get("relevant_file_inventory")
    if not isinstance(inventory, list) or not inventory:
        raise ContractError(f"{path}: missing transaction file inventory")
    for row in inventory:
        digest = row.get("sha256") if isinstance(row, dict) else None
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            raise ContractError(
                f"{path}: exact selection rejects unhashed transaction files"
            )
    heldout = result.get("heldout") or {}
    if heldout.get("n_prompts") != EVAL_N_PROMPTS:
        raise ContractError(f"{path}: held-out set is not the fixed 256 prompts")
    if (result.get("metrics") or {}).get("n_prompts") != EVAL_N_PROMPTS:
        raise ContractError(f"{path}: metrics do not cover exactly 256 prompts")
    for key in ("eval_set_sha256", "rl_parquet_sha256"):
        digest = heldout.get(key)
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            raise ContractError(f"{path}: invalid/missing heldout {key}")
    protocol = dict(result.get("evaluation_protocol") or {})
    declared_protocol_hash = protocol.pop("sha256", None)
    protocol.pop("evaluated_checkpoint", None)
    if declared_protocol_hash != sha256_json(protocol):
        raise ContractError(f"{path}: evaluation protocol hash mismatch")
    if result.get("evaluation_protocol_sha256") != declared_protocol_hash:
        raise ContractError(f"{path}: top-level evaluation protocol hash mismatch")
    declared_payload = result.get("result_payload_sha256")
    payload = dict(result)
    payload.pop("result_payload_sha256", None)
    if declared_payload != sha256_json(payload):
        raise ContractError(f"{path}: result payload hash mismatch")
    return condition, step


def select_fve_matched(
    result_records: Sequence[tuple[Path, Mapping[str, Any]]],
    *,
    final_step: int = FINAL_STEP,
    checkpoint_interval: int = CHECKPOINT_INTERVAL,
) -> dict[str, Any]:
    """Apply the frozen final-FVE matching rule, failing on incomplete grids."""
    if final_step <= 0 or checkpoint_interval <= 0 or final_step % checkpoint_interval:
        raise ContractError("invalid final step/checkpoint interval")
    indexed: dict[tuple[str, int], tuple[Path, Mapping[str, Any]]] = {}
    protocol_hashes: set[str] = set()
    eval_hashes: set[str] = set()
    parquet_hashes: set[str] = set()
    baselines: set[float] = set()
    for path, result in result_records:
        condition, step = _result_identity(result, path)
        key = (condition, step)
        if key in indexed:
            raise ContractError(f"duplicate result for {condition} step {step}")
        indexed[key] = (path, result)
        protocol_hashes.add(str(result.get("evaluation_protocol_sha256")))
        heldout = result.get("heldout") or {}
        eval_hashes.add(str(heldout.get("eval_set_sha256")))
        parquet_hashes.add(str(heldout.get("rl_parquet_sha256")))
        baselines.add(float((result.get("metrics") or {})["baseline_mse"]))
    if len(protocol_hashes) != 1 or "None" in protocol_hashes:
        raise ContractError("evaluation results use different/missing protocols")
    if len(eval_hashes) != 1 or "None" in eval_hashes:
        raise ContractError("evaluation results use different/missing held-out sets")
    if len(parquet_hashes) != 1 or "None" in parquet_hashes:
        raise ContractError("evaluation results use different/missing RL parquets")
    if len(baselines) != 1:
        raise ContractError("evaluation results use different FVE baselines")

    def checkpoint_identity(
        condition: str, step: int, path: Path, result: Mapping[str, Any]
    ) -> dict[str, Any]:
        checkpoint = result["checkpoint"]
        inventory = checkpoint["relevant_file_inventory"]
        actor_row = next(
            (
                row
                for row in inventory
                if row.get("path") == "actor/adapter_model.safetensors"
            ),
            None,
        )
        if actor_row is None:
            raise ContractError(f"{path}: actor adapter hash is absent from inventory")
        fingerprint = checkpoint["transaction_fingerprint"]
        return {
            "condition": condition,
            "transaction_step": step,
            "actor_iteration": f"iter_{step:06d}",
            "zero_based_last_update_index": step - 1,
            "checkpoint_id": (
                f"{condition}:iter_{step:06d}:transaction_sha256:{fingerprint}"
            ),
            "transaction_fingerprint": fingerprint,
            "actor_adapter_sha256": actor_row["sha256"],
            "base_checkpoint_tree_sha256": checkpoint[
                "base_checkpoint_tree_sha256"
            ],
            "merged_av_checkpoint_tree_sha256": checkpoint[
                "merged_av_checkpoint_tree_sha256"
            ],
            "exact_fve_pct": float(result["metrics"]["fve_pct"]),
            "exact_fve_result_payload_sha256": result["result_payload_sha256"],
            "result_path": str(path.resolve()),
        }

    finals = {}
    for condition in ("plausible", "implausible"):
        key = (condition, final_step)
        if key not in indexed:
            raise ContractError(f"missing final exact evaluation for {condition} step {final_step}")
        path, result = indexed[key]
        finals[condition] = checkpoint_identity(condition, final_step, path, result)
    plausible_fve = finals["plausible"]["exact_fve_pct"]
    implausible_fve = finals["implausible"]["exact_fve_pct"]
    if plausible_fve == implausible_fve:
        raise ContractError("final FVEs are exactly tied; lower/higher condition is undefined")
    lower = "plausible" if plausible_fve < implausible_fve else "implausible"
    higher = "implausible" if lower == "plausible" else "plausible"
    target = finals[lower]["exact_fve_pct"]
    expected_steps = list(range(checkpoint_interval, final_step + 1, checkpoint_interval))
    missing = [step for step in expected_steps if (higher, step) not in indexed]
    if missing:
        raise ContractError(
            f"higher-final-FVE condition {higher} lacks exact historical-critic "
            f"evaluations at checkpoint steps {missing}"
        )
    candidates = []
    for step in expected_steps:
        path, result = indexed[(higher, step)]
        fve = float(result["metrics"]["fve_pct"])
        candidates.append(
            {
                **checkpoint_identity(higher, step, path, result),
                "fve_pct": fve,
                "absolute_fve_difference_pct_points": abs(fve - target),
            }
        )
    selected = min(
        candidates,
        key=lambda row: (
            row["absolute_fve_difference_pct_points"],
            -row["transaction_step"],
        ),
    )
    ranked = sorted(
        candidates,
        key=lambda row: (
            row["absolute_fve_difference_pct_points"],
            -row["transaction_step"],
        ),
    )
    matched_members = {
        lower: finals[lower],
        higher: selected,
    }
    analysis_sets = {
        "schema_version": "nla.warm_start.claim_analysis_sets.v1",
        "activation_identity_warning": (
            "Current claim-generation activation keys intentionally omit checkpoint_id. "
            "Never combine these analysis sets in one generator/scorer dataset."
        ),
        "required_execution": (
            "separate generation output directories, explanation JSONL inputs, "
            "and claim-scoring invocations per analysis_set_id"
        ),
        "sets": [
            {
                "analysis_set_id": "final_vs_final",
                "purpose": "compare both final iter_000780 transactions",
                "claim_generation_namespace": "final_vs_final",
                "members": {
                    "plausible": finals["plausible"],
                    "implausible": finals["implausible"],
                },
            },
            {
                "analysis_set_id": "fve_matched",
                "purpose": (
                    "hold the lower-final-FVE condition at final and compare "
                    "with the objectively selected higher-condition checkpoint"
                ),
                "claim_generation_namespace": "fve_matched",
                "members": matched_members,
            },
        ],
    }
    return {
        "schema_version": SELECTION_SCHEMA,
        "status": "complete",
        "selection_protocol": {
            "final_step": final_step,
            "checkpoint_interval": checkpoint_interval,
            "held_condition_rule": "condition_with_lower_exact_final_fve",
            "matched_condition_rule": "condition_with_higher_exact_final_fve",
            "objective": "minimize_absolute_fve_difference_in_percentage_points",
            "tie_break": "later_checkpoint_step",
            "candidate_grid": expected_steps,
            "logged_eval_mapping": (
                "eval@N is post-update-N and maps to transaction N+1; "
                "transaction N is post-update-(N-1)"
            ),
        },
        "evaluation_protocol_sha256": next(iter(protocol_hashes)),
        "eval_set_sha256": next(iter(eval_hashes)),
        "rl_parquet_sha256": next(iter(parquet_hashes)),
        "baseline_mse": next(iter(baselines)),
        "finals": finals,
        "held_lower_final": {"condition": lower, **finals[lower]},
        "matched_higher": selected,
        "selected_absolute_fve_difference_pct_points": selected[
            "absolute_fve_difference_pct_points"
        ],
        "ranked_higher_condition_candidates": ranked,
        "claim_analysis_sets": analysis_sets,
        "generated_at_unix": time.time(),
    }
