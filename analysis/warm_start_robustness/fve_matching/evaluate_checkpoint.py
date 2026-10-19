#!/usr/bin/env python3
"""Evaluate one immutable actor+historical-critic transaction at exact FVE.

Generation uses the same EasyNLA ``rollout_batch_vllm`` function, patched
vLLM-lens Karvonen steering path, greedy 256-prompt batch, and actor-to-vLLM
weight merge used during training. Scoring uses the same suffix-anchored
``score_with_critic`` and held-out-population predict-the-mean baseline.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:
    from .core import (
        CRITIC_BATCH_SIZE,
        CRITIC_MAX_TOKENS,
        EVALUATION_SCHEMA,
        EVAL_N_PROMPTS,
        EVAL_TEMPERATURE,
        GENERATION_SCHEMA,
        MAX_NEW_TOKENS,
        ContractError,
        aggregate_fve,
        atomic_write_json,
        audit_transaction,
        heldout_descriptor,
        load_fixed_heldout,
        sha256_file,
        sha256_json,
        validate_locked_run_config,
        validate_resolved_doc_hash_split,
    )
except ImportError:  # direct ``python evaluate_checkpoint.py``
    from core import (  # type: ignore
        CRITIC_BATCH_SIZE,
        CRITIC_MAX_TOKENS,
        EVALUATION_SCHEMA,
        EVAL_N_PROMPTS,
        EVAL_TEMPERATURE,
        GENERATION_SCHEMA,
        MAX_NEW_TOKENS,
        ContractError,
        aggregate_fve,
        atomic_write_json,
        audit_transaction,
        heldout_descriptor,
        load_fixed_heldout,
        sha256_file,
        sha256_json,
        validate_locked_run_config,
        validate_resolved_doc_hash_split,
    )


_LENS_MARKERS = {
    "chunked-prefill seq_lens fix": "_meta5",
    "per-request steer log": "get_and_reset_steer_log",
    "log-key wiring": "log_key=per_req_log_key",
    "steering-apply counter": "get_and_reset_steer_count",
}


def _resolve_path(override: str | None, configured: Any, label: str) -> str:
    value = override or configured
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"no usable {label} path/repository was supplied")
    return value


def _file_provenance(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "exists": False}
    return {
        "path": str(path.resolve()),
        "exists": True,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _checkpoint_tree_provenance(raw_path: str, label: str) -> dict[str, Any]:
    root = Path(raw_path).expanduser().resolve()
    if not root.is_dir():
        raise ContractError(
            f"exact evaluation requires {label} as an immutable local directory, got {raw_path!r}"
        )
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or any(part.startswith(".") for part in path.relative_to(root).parts):
            continue
        files.append(
            {
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if not files or not any(
        row["path"].endswith((".safetensors", ".bin")) for row in files
    ):
        raise ContractError(f"{label} has no model weight files: {root}")
    identity = {
        "path": str(root),
        "files": files,
        "logical_bytes": sum(row["bytes"] for row in files),
    }
    return {**identity, "tree_sha256": sha256_json({"files": files})}


def _easy_nla_provenance(root: Path) -> list[dict[str, Any]]:
    relative = (
        "nla/train_rl_vllm.py",
        "nla/utils/vllm_steer.py",
        "nla/utils/critic.py",
        "nla/utils/prompts.py",
        "nla/schema.py",
        "nla/config.py",
        "nla/val_split.py",
    )
    rows = []
    for item in relative:
        path = root / item
        provenance = _file_provenance(path)
        rows.append({**provenance, "path": item})
    missing = [row["path"] for row in rows if not row["exists"]]
    if missing:
        raise ContractError(f"EasyNLA checkout lacks exact evaluator sources: {missing}")
    return rows


def _tokenizer_contract(tokenizer: Any) -> dict[str, Any]:
    vocab = tokenizer.get_vocab()
    contract = {
        "class": type(tokenizer).__name__,
        "vocab_size": int(tokenizer.vocab_size),
        "len_with_added_tokens": len(tokenizer),
        "vocab_sha256": sha256_json(vocab),
        "chat_template_sha256": hashlib.sha256(
            str(tokenizer.chat_template).encode("utf-8")
        ).hexdigest(),
        "special_tokens_map": tokenizer.special_tokens_map,
    }
    return {**contract, "sha256": sha256_json(contract)}


def _sidecar_contract(cfg: Any, sidecar_path: Path) -> dict[str, Any]:
    contract = {
        "sidecar_file_sha256": sha256_file(sidecar_path),
        "d_model": cfg.d_model,
        "injection_char": cfg.injection_char,
        "injection_token_id": cfg.injection_token_id,
        "injection_left_neighbor_id": cfg.injection_left_neighbor_id,
        "injection_right_neighbor_id": cfg.injection_right_neighbor_id,
        "actor_prompt_template": cfg.actor_prompt_template,
        "critic_prompt_template": cfg.critic_prompt_template,
        "critic_suffix_ids": cfg.critic_suffix_ids,
        "extraction_layer_index": cfg.extraction_layer_index,
        "mse_scale": cfg.mse_scale,
    }
    return {**contract, "sha256": sha256_json(contract)}


def _require_exact_import(root: Path, imported_object: Any, label: str) -> None:
    module = sys.modules[imported_object.__module__]
    path = Path(module.__file__).resolve()
    if root not in path.parents:
        raise ContractError(f"{label} imported from {path}, expected beneath {root}")


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _runtime_versions(torch: Any) -> dict[str, Any]:
    return {
        "python": sys.version,
        "torch": getattr(torch, "__version__", None),
        "transformers": _package_version("transformers"),
        "peft": _package_version("peft"),
        "vllm": _package_version("vllm"),
        "vllm_lens": _package_version("vllm-lens"),
        "cuda_runtime": getattr(torch.version, "cuda", None),
        "visible_cuda_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_names": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ],
    }


def _validate_lens_patch() -> dict[str, Any]:
    import vllm_lens._worker_ext as worker_ext

    path = Path(worker_ext.__file__).resolve()
    source = path.read_text()
    missing = [label for label, marker in _LENS_MARKERS.items() if marker not in source]
    if missing:
        raise ContractError(
            f"installed vLLM-lens lacks mandatory injection-audit patches {missing}: {path}"
        )
    return _file_provenance(path)


def _protocol(
    *,
    transaction: Any,
    mapping: Mapping[str, Any],
    easynla_files: Sequence[Mapping[str, Any]],
    rl_parquet_sha256: str,
    eval_set_sha256: str,
    base_checkpoint: str,
    merged_av_checkpoint: str,
    tokenizer_contract: Mapping[str, Any],
    sidecar_contract: Mapping[str, Any],
    runtime_versions: Mapping[str, Any],
    base_checkpoint_provenance: Mapping[str, Any],
    merged_av_checkpoint_provenance: Mapping[str, Any],
    frozen_protocol_file: Mapping[str, Any],
    evaluator_files: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    # The protocol identity contains only comparison-invariant semantics. The
    # transaction/checkpoint fields are recorded alongside it but deliberately
    # excluded from the hash; otherwise two conditions or two steps could never
    # pass the selector's same-protocol check.
    invariant = {
        "schema_version": EVALUATION_SCHEMA,
        "rl_parquet_sha256": rl_parquet_sha256,
        "eval_set_sha256": eval_set_sha256,
        "n_prompts": EVAL_N_PROMPTS,
        "temperature": EVAL_TEMPERATURE,
        "max_new_tokens": MAX_NEW_TOKENS,
        "top_p": 1.0,
        "top_k": -1,
        "group_size": 1,
        "critic_max_tokens": CRITIC_MAX_TOKENS,
        "critic_batch_size": CRITIC_BATCH_SIZE,
        "vllm_tp": 1,
        "vllm_gpu_memory_utilization": 0.25,
        "enable_prefix_caching": False,
        "karvonen_injection_layer": 1,
        "easy_nla_files": list(easynla_files),
        "tokenizer_contract": dict(tokenizer_contract),
        "sidecar_contract": dict(sidecar_contract),
        "runtime_package_versions": {
            key: runtime_versions.get(key)
            for key in ("torch", "transformers", "peft", "vllm", "vllm_lens")
        },
        "raw_base_checkpoint_tree_sha256": base_checkpoint_provenance[
            "tree_sha256"
        ],
        "frozen_protocol_file": dict(frozen_protocol_file),
        "evaluator_files": list(evaluator_files),
    }
    return {
        **invariant,
        "sha256": sha256_json(invariant),
        "evaluated_checkpoint": {
            "transaction_fingerprint": transaction.fingerprint,
            "transaction_step": transaction.step,
            "checkpoint_state_mapping": mapping,
            "base_checkpoint": base_checkpoint,
            "merged_av_checkpoint": merged_av_checkpoint,
            "base_checkpoint_provenance": dict(base_checkpoint_provenance),
            "merged_av_checkpoint_provenance": dict(
                merged_av_checkpoint_provenance
            ),
        },
    }


def _journal_path(output: Path, override: str | None) -> Path:
    if override:
        return Path(override).expanduser().resolve()
    return output.with_name(output.stem + ".generations.json")


def _seal_payload(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    if field in value:
        raise ContractError(f"refusing to replace pre-existing payload hash {field}")
    sealed = dict(value)
    sealed[field] = sha256_json(sealed)
    return sealed


def _verify_payload(value: Mapping[str, Any], field: str, context: str) -> None:
    declared = value.get(field)
    payload = dict(value)
    payload.pop(field, None)
    if declared != sha256_json(payload):
        raise ContractError(f"{context}: {field} mismatch")


def _load_journal(path: Path, run_fingerprint: str) -> Mapping[str, Any] | None:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_bytes())
    except json.JSONDecodeError as exc:
        raise ContractError(f"invalid generation journal {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"{path}: generation journal must be an object")
    _verify_payload(value, "journal_payload_sha256", str(path))
    if value.get("schema_version") != GENERATION_SCHEMA:
        raise ContractError(f"{path}: wrong generation-journal schema")
    if value.get("run_fingerprint") != run_fingerprint:
        raise ContractError(f"{path}: belongs to a different evaluation run")
    rows = value.get("rows")
    if not isinstance(rows, list) or len(rows) != EVAL_N_PROMPTS:
        raise ContractError(f"{path}: expected {EVAL_N_PROMPTS} generation rows")
    indices = [row.get("prompt_index") for row in rows]
    if indices != list(range(EVAL_N_PROMPTS)):
        raise ContractError(f"{path}: generation rows are not in exact prompt order")
    if value.get("status") != "complete":
        raise ContractError(
            f"{path}: preserved generation journal has a mechanism diagnostic: "
            f"{value.get('mechanism_checks', {}).get('failure_reasons')}"
        )
    return value


def _generate(
    *,
    args: argparse.Namespace,
    transaction: Any,
    rows: Sequence[Any],
    tokenizer: Any,
    cfg: Any,
    base_checkpoint: str,
    merged_av_checkpoint: str,
    run_fingerprint: str,
    lens_provenance: Mapping[str, Any],
    runtime_versions: Mapping[str, Any],
    easynla_root: Path,
) -> dict[str, Any]:
    import torch
    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForCausalLM
    from vllm import LLM as VLLM

    from nla.schema import extract_explanation
    from nla.train_rl_vllm import rollout_batch_vllm, sync_actor_to_vllm
    from nla.utils import build_prompt_text, cjk_fraction
    from nla.utils.vllm_steer import read_reset_steer_count

    _require_exact_import(easynla_root, rollout_batch_vllm, "rollout_batch_vllm")
    _require_exact_import(easynla_root, sync_actor_to_vllm, "sync_actor_to_vllm")
    peft_config = PeftConfig.from_pretrained(str(transaction.actor))
    adapter_base = str(peft_config.base_model_name_or_path)
    configured_base = str(transaction.run_config["base_ckpt"])
    if configured_base not in adapter_base and adapter_base not in configured_base:
        # Local HF snapshot paths commonly differ while ending in the immutable
        # revision. Keep the check strict enough to reject another model family.
        expected_fragments = ("Qwen2.5-7B-Instruct", "a09a35458c702b33eeacc393d103063234e8bc28")
        if not any(fragment in adapter_base and fragment in configured_base for fragment in expected_fragments):
            raise ContractError(
                f"actor adapter base {adapter_base!r} disagrees with run base {configured_base!r}"
            )

    print(f"[exact-fve] loading raw actor base {base_checkpoint}", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        base_checkpoint,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(args.device)
    actor = PeftModel.from_pretrained(
        base,
        str(transaction.actor),
        adapter_name="default",
        is_trainable=False,
    ).eval()
    actor.set_adapter("default")

    print(f"[exact-fve] loading historical vLLM base {merged_av_checkpoint}", flush=True)
    llm = VLLM(
        model=merged_av_checkpoint,
        tokenizer=merged_av_checkpoint,
        dtype="bfloat16",
        gpu_memory_utilization=float(args.vllm_gpu_mem),
        max_model_len=512,
        tensor_parallel_size=1,
        enforce_eager=True,
        disable_log_stats=False,
        enable_prefix_caching=False,
    )
    sync_seconds = sync_actor_to_vllm(actor, llm, ipc=args.ipc_weight_sync)
    print(f"[exact-fve] exact actor weights synced in {sync_seconds:.1f}s", flush=True)
    del actor, base
    gc.collect()
    torch.cuda.empty_cache()

    prompts_with_activations = [
        (
            build_prompt_text(row.prompt, cfg.injection_char, tokenizer),
            torch.from_numpy(np.array(row.activation, dtype=np.float32, copy=True)),
        )
        for row in rows
    ]
    try:
        llm.apply_model(read_reset_steer_count)
    except Exception as exc:
        raise ContractError(f"cannot reset patched steering counter: {exc}") from exc
    started = time.time()
    responses = rollout_batch_vllm(
        llm,
        tokenizer,
        prompts_with_activations,
        cfg.injection_token_id,
        1,
        MAX_NEW_TOKENS,
        EVAL_TEMPERATURE,
        injection_layer=1,
        left_id=cfg.injection_left_neighbor_id,
        right_id=cfg.injection_right_neighbor_id,
    )
    generation_seconds = time.time() - started
    try:
        counts = llm.apply_model(read_reset_steer_count)
    except Exception as exc:
        counts = []
        counter_read_error = f"{type(exc).__name__}: {exc}"
    else:
        counter_read_error = None
    count_values = [int(value) for value in counts] if counts else []
    count_available = bool(count_values) and all(value >= 0 for value in count_values)
    minimum_count = min(count_values) if count_available else -1
    if len(responses) != EVAL_N_PROMPTS:
        raise ContractError(f"vLLM returned {len(responses)} responses")
    keyed = {int(response["prompt_idx"]): response for response in responses}
    if set(keyed) != set(range(EVAL_N_PROMPTS)) or len(keyed) != len(responses):
        raise ContractError("vLLM response prompt indices are incomplete/duplicated")
    output_rows = []
    for prompt_index in range(EVAL_N_PROMPTS):
        response = keyed[prompt_index]
        full_ids = response["full_ids"].tolist()
        prompt_len = int(response["prompt_len"])
        text = str(response["text"])
        explanation = extract_explanation(text)
        output_rows.append(
            {
                "prompt_index": prompt_index,
                "parquet_row_index": rows[prompt_index].parquet_row_index,
                "document_id": rows[prompt_index].document_id,
                "text": text,
                "explanation": explanation,
                "extraction_ok": explanation is not None,
                "full_token_ids": full_ids,
                "prompt_token_count": prompt_len,
                "generated_token_ids": full_ids[prompt_len:],
                "generated_token_count": len(full_ids) - prompt_len,
                "sampled_token_logprobs": response["old_logp"].tolist(),
                "finish_reason_length": bool(response.get("truncated", False)),
                "steer_verified": bool(response.get("steer_verified", False)),
                "marker_leak": cfg.injection_char in text,
                "cjk_fraction": float(cjk_fraction(text)),
            }
        )
    all_verified = all(row["steer_verified"] for row in output_rows)
    failures = []
    if not count_available:
        failures.append(
            f"patched_steer_counter_unavailable:{counts}:error={counter_read_error}"
        )
    elif minimum_count < math.floor(0.98 * EVAL_N_PROMPTS):
        failures.append(
            f"steer_apply_count_{minimum_count}_below_"
            f"{math.floor(0.98 * EVAL_N_PROMPTS)}"
        )
    if not all_verified:
        bad = [row["prompt_index"] for row in output_rows if not row["steer_verified"]]
        failures.append(f"per_request_steer_unverified:{bad}")
    return {
        "schema_version": GENERATION_SCHEMA,
        "status": "complete" if not failures else "diagnostic_failed",
        "run_fingerprint": run_fingerprint,
        "transaction_step": transaction.step,
        "temperature": EVAL_TEMPERATURE,
        "top_p": 1.0,
        "top_k": -1,
        "max_new_tokens": MAX_NEW_TOKENS,
        "generation_seconds": generation_seconds,
        "weight_sync_seconds": sync_seconds,
        "mechanism_checks": {
            "vllm_lens_worker": lens_provenance,
            "steer_apply_counts_by_worker": count_values,
            "minimum_steer_apply_count": minimum_count,
            "minimum_required": math.floor(0.98 * EVAL_N_PROMPTS),
            "all_requests_steer_verified": all_verified,
            "marker_leak_count": sum(row["marker_leak"] for row in output_rows),
            "failure_reasons": failures,
        },
        "runtime_versions": dict(runtime_versions),
        "rows": output_rows,
        "committed_at_unix": time.time(),
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    condition = args.condition
    transaction = audit_transaction(
        Path(args.transaction), hash_relevant_files=not args.skip_transaction_hashes
    )
    mapping = validate_locked_run_config(transaction.run_config, transaction.step)
    tags = {
        value.strip()
        for value in str(transaction.run_config.get("wandb_tags", "")).split(",")
        if value.strip()
    }
    if condition not in tags:
        raise ContractError(
            f"--condition {condition!r} is absent from transaction wandb_tags {sorted(tags)}"
        )
    easynla_root = Path(args.easynla_root).expanduser().resolve()
    easynla_files = _easy_nla_provenance(easynla_root)
    sys.path.insert(0, str(easynla_root))

    # All imports after this point must resolve against the explicitly selected
    # EasyNLA checkout rather than another installed `nla` package.
    import torch
    from transformers import AutoTokenizer

    from nla.config import load_nla_config
    from nla.schema import compute_predict_mean_baselines
    from nla.train_rl_vllm import score_with_critic
    from nla.val_split import is_val_doc as exact_is_val_doc
    from nla.val_split import val_doc_permille as exact_val_doc_permille

    _require_exact_import(easynla_root, score_with_critic, "score_with_critic")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ContractError(
            "exact historical semantics require one visible GPU per evaluator process; "
            "launch four independent TP1 evaluators to use four GPUs"
        )
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)
    lens_provenance = _validate_lens_patch()
    runtime_versions = _runtime_versions(torch)

    rl_parquet = Path(
        _resolve_path(args.rl_parquet, transaction.run_config["rl_parquet"], "RL parquet")
    ).expanduser().resolve()
    rl_parquet_sha256 = sha256_file(rl_parquet)
    if args.expected_rl_parquet_sha256 and (
        rl_parquet_sha256 != args.expected_rl_parquet_sha256
    ):
        raise ContractError(
            f"RL parquet SHA-256 {rl_parquet_sha256} != expected "
            f"{args.expected_rl_parquet_sha256}"
        )
    heldout_rows, total_rows, permille = load_fixed_heldout(
        rl_parquet,
        val_rows=int(transaction.run_config["val_rows"]),
        n_prompts=EVAL_N_PROMPTS,
        val_permille_fn=exact_val_doc_permille,
        is_val_doc_fn=exact_is_val_doc,
    )
    split_audit = validate_resolved_doc_hash_split(
        transaction.run_config, total_rows
    )
    heldout = heldout_descriptor(heldout_rows)
    heldout.update(
        {
            "rl_parquet": str(rl_parquet),
            "rl_parquet_sha256": rl_parquet_sha256,
            "rl_parquet_total_rows": total_rows,
            "val_rows_requested": int(transaction.run_config["val_rows"]),
            "val_doc_permille": permille,
            "selection": "first_256_rows_whose_doc_id_is_in_crc32_val_bucket",
            "resolved_split_audit": split_audit,
        }
    )

    base_checkpoint = _resolve_path(
        args.base_ckpt, transaction.run_config["base_ckpt"], "raw base checkpoint"
    )
    merged_av_checkpoint = _resolve_path(
        args.av_ckpt, transaction.run_config["av_ckpt"], "merged AV checkpoint"
    )
    base_checkpoint_provenance = _checkpoint_tree_provenance(
        base_checkpoint, "raw base checkpoint"
    )
    merged_av_checkpoint_provenance = _checkpoint_tree_provenance(
        merged_av_checkpoint, "merged AV-SFT checkpoint"
    )
    tokenizer = AutoTokenizer.from_pretrained(merged_av_checkpoint)
    cfg = load_nla_config(str(transaction.root), tokenizer)
    if cfg.d_model != heldout["d_model"]:
        raise ContractError(
            f"sidecar d_model {cfg.d_model} != held-out width {heldout['d_model']}"
        )
    if cfg.critic_prompt_template is None:
        raise ContractError("transaction sidecar lacks critic prompt template")
    protocol = _protocol(
        transaction=transaction,
        mapping=mapping,
        easynla_files=easynla_files,
        rl_parquet_sha256=rl_parquet_sha256,
        eval_set_sha256=heldout["eval_set_sha256"],
        base_checkpoint=base_checkpoint,
        merged_av_checkpoint=merged_av_checkpoint,
        tokenizer_contract=_tokenizer_contract(tokenizer),
        sidecar_contract=_sidecar_contract(cfg, transaction.sidecar_path),
        runtime_versions=runtime_versions,
        base_checkpoint_provenance=base_checkpoint_provenance,
        merged_av_checkpoint_provenance=merged_av_checkpoint_provenance,
        frozen_protocol_file={
            **_file_provenance(Path(__file__).resolve().parent / "protocol.yaml"),
            "path": "protocol.yaml",
        },
        evaluator_files=[
            {
                **_file_provenance(Path(__file__).resolve().parent / name),
                "path": name,
            }
            for name in ("core.py", "evaluate_checkpoint.py")
        ],
    )
    run_fingerprint = sha256_json(
        {
            "condition": condition,
            "transaction": transaction.fingerprint,
            "protocol": protocol["sha256"],
            "merged_av_checkpoint_tree_sha256": merged_av_checkpoint_provenance[
                "tree_sha256"
            ],
        }
    )
    if output.exists():
        try:
            existing = json.loads(output.read_bytes())
        except json.JSONDecodeError as exc:
            raise ContractError(f"invalid exact-FVE result {output}: {exc}") from exc
        if not isinstance(existing, dict):
            raise ContractError(f"{output}: exact-FVE result must be an object")
        _verify_payload(existing, "result_payload_sha256", str(output))
        if (
            existing.get("schema_version") == EVALUATION_SCHEMA
            and existing.get("status") == "complete"
            and existing.get("run_fingerprint") == run_fingerprint
        ):
            return existing
        raise ContractError(f"refusing to overwrite different/incomplete result {output}")

    journal_path = _journal_path(output, args.journal)
    journal = _load_journal(journal_path, run_fingerprint)
    if journal is None:
        journal = _seal_payload(
            _generate(
                args=args,
                transaction=transaction,
                rows=heldout_rows,
                tokenizer=tokenizer,
                cfg=cfg,
                base_checkpoint=base_checkpoint,
                merged_av_checkpoint=merged_av_checkpoint,
                run_fingerprint=run_fingerprint,
                lens_provenance=lens_provenance,
                runtime_versions=runtime_versions,
                easynla_root=easynla_root,
            ),
            "journal_payload_sha256",
        )
        atomic_write_json(journal_path, journal)
        if journal.get("status") != "complete":
            raise ContractError(
                "generation mechanism diagnostic was preserved at "
                f"{journal_path}: {journal['mechanism_checks']['failure_reasons']}"
            )
    journal_sha256 = sha256_file(journal_path)
    explanations = [row["explanation"] for row in journal["rows"]]

    from nla.models import NLACriticModel
    from nla.config import verify_critic_suffix

    first_explanation = next(
        (value for value in explanations if value is not None), None
    )
    if first_explanation is None:
        raise ContractError("all 256 generations failed explanation extraction")
    if cfg.critic_suffix_ids:
        first_critic_ids = tokenizer.encode(
            cfg.critic_prompt_template.format(explanation=first_explanation),
            add_special_tokens=False,
        )
        verify_critic_suffix(
            first_critic_ids,
            cfg.critic_suffix_ids,
            context="exact-checkpoint FVE first valid explanation",
        )

    print(f"[exact-fve] loading contemporaneous critic {transaction.critic}", flush=True)
    critic = NLACriticModel.from_pretrained(
        str(transaction.critic), torch_dtype=torch.bfloat16
    ).to(args.device)
    critic.eval()
    activations = [
        torch.from_numpy(np.array(row.activation, dtype=np.float32, copy=True))
        for row in heldout_rows
    ]
    score_started = time.time()
    rewards = score_with_critic(
        critic,
        tokenizer,
        explanations,
        activations,
        cfg.critic_prompt_template,
        cfg.mse_scale,
        args.device,
        batch_size=CRITIC_BATCH_SIZE,
    )
    scoring_seconds = time.time() - score_started
    activation_stack = torch.stack(activations).float()
    _, baseline_mse = compute_predict_mean_baselines(activation_stack, cfg.mse_scale)
    metrics = aggregate_fve(rewards, baseline_mse)
    metrics.update(
        {
            "extraction_rate": sum(
                bool(row["extraction_ok"]) for row in journal["rows"]
            )
            / EVAL_N_PROMPTS,
            "critic_score_rate": sum(reward is not None for reward in rewards)
            / EVAL_N_PROMPTS,
            "scoring_seconds": scoring_seconds,
        }
    )
    result_rows = []
    for generated, reward in zip(journal["rows"], rewards):
        effective = -2.0 if reward is None else float(reward)
        included = math.isfinite(effective) and effective > -2.0
        result_rows.append(
            {
                **generated,
                "critic_reward": reward,
                "failure_floor_reward": effective,
                "included_in_fve": included,
                "per_prompt_fve": (
                    1.0 - (-effective) / baseline_mse if included else None
                ),
            }
        )
    result = {
        "schema_version": EVALUATION_SCHEMA,
        "status": "complete",
        "condition": condition,
        "run_fingerprint": run_fingerprint,
        "evaluation_protocol_sha256": protocol["sha256"],
        "evaluation_protocol": protocol,
        "checkpoint": {
            "transaction_path": str(transaction.root),
            "transaction_step": transaction.step,
            "actor_path": str(transaction.actor),
            "critic_path": str(transaction.critic),
            "critic_saved_at_step": transaction.step,
            "transaction_fingerprint": transaction.fingerprint,
            "base_checkpoint_tree_sha256": base_checkpoint_provenance[
                "tree_sha256"
            ],
            "merged_av_checkpoint_tree_sha256": merged_av_checkpoint_provenance[
                "tree_sha256"
            ],
            "relevant_file_inventory": list(transaction.relevant_files),
        },
        "checkpoint_state_mapping": mapping,
        "heldout": heldout,
        "metrics": metrics,
        "generation_journal": {
            "path": str(journal_path),
            "bytes": journal_path.stat().st_size,
            "sha256": journal_sha256,
        },
        "mechanism_checks": journal["mechanism_checks"],
        "runtime_versions": journal["runtime_versions"],
        "rows": result_rows,
        "completed_at_unix": time.time(),
    }
    result = _seal_payload(result, "result_payload_sha256")
    atomic_write_json(output, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    component = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Deterministically evaluate one exact actor+historical-critic transaction."
    )
    parser.add_argument("--condition", required=True, choices=("plausible", "implausible"))
    parser.add_argument("--transaction", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--journal", default=None)
    parser.add_argument(
        "--easynla-root",
        default=str(component.parent / "EasyNLA"),
        help="Exact EasyNLA checkout whose vLLM injection/scoring code is used.",
    )
    parser.add_argument("--rl-parquet", default=None, help="Relocation override for run_config rl_parquet.")
    parser.add_argument("--base-ckpt", default=None, help="Relocation override for raw actor base.")
    parser.add_argument("--av-ckpt", default=None, help="Relocation override for merged SFT AV checkpoint.")
    parser.add_argument("--expected-rl-parquet-sha256", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--vllm-gpu-mem",
        type=float,
        default=0.25,
        help="Locked to the historical value 0.25.",
    )
    parser.add_argument(
        "--ipc-weight-sync",
        action="store_true",
        help="Use the training IPC transport for actor-to-vLLM sync; transport does not change weights.",
    )
    parser.add_argument(
        "--skip-transaction-hashes",
        action="store_true",
        help="Development smoke only; exact paper results must hash actor/critic files.",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.device != "cuda:0":
        raise ContractError("one-visible-GPU evaluator requires --device cuda:0")
    if args.vllm_gpu_mem != 0.25:
        raise ContractError("exact historical protocol requires --vllm-gpu-mem 0.25")
    if args.expected_rl_parquet_sha256 is not None and (
        len(args.expected_rl_parquet_sha256) != 64
        or any(char not in "0123456789abcdef" for char in args.expected_rl_parquet_sha256)
    ):
        raise ContractError("--expected-rl-parquet-sha256 must be lowercase SHA-256")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_args(args)
    result = evaluate(args)
    print(
        json.dumps(
            {
                "status": result["status"],
                "condition": result["condition"],
                "transaction_step": result["checkpoint"]["transaction_step"],
                "fve_pct": result["metrics"]["fve_pct"],
                "output": str(Path(args.output).expanduser().resolve()),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
