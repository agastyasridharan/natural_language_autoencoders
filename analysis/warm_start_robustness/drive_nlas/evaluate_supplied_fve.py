#!/usr/bin/env python3
"""Exact greedy FVE evaluation for a supplied full-HF actor/critic pair."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
import time
import unicodedata
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA = "nla.warm_start.supplied_pair_fve.v1"
N_PROMPTS = 256
MAX_NEW_TOKENS = 150


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
    with tmp.open("xb") as handle:
        handle.write(canonical_json(value))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def required_complete(path: Path, context: str) -> Mapping[str, Any]:
    value = json.loads(path.read_text())
    if value.get("complete") is not True:
        raise RuntimeError(f"{context} is not complete: {path}")
    return value


def record_path(records: Path, ordinal: int) -> Path:
    return records / f"{ordinal:04d}.json"


def cjk_fraction(text: str) -> float:
    if not text:
        return 0.0
    return sum(1 for char in text if "CJK" in unicodedata.name(char, "")) / len(text)


def generate_records(
    *,
    actor: Any,
    tokenizer: Any,
    cfg: Any,
    rows: Sequence[Any],
    run_fingerprint: str,
    records_dir: Path,
    batch_size: int,
    device: str,
    extract_explanation: Any,
    chat_template_token_ids: Any,
    inject_at_marked_positions: Any,
    normalize_activation: Any,
    resolve_embed_scale: Any,
    resolve_text_model: Any,
) -> list[dict[str, Any]]:
    import torch

    expected_content = cfg.actor_prompt_template.format(injection_char="<INJECT>")
    for ordinal, row in enumerate(rows):
        if row.prompt != [{"role": "user", "content": expected_content}]:
            raise RuntimeError(f"held-out row {ordinal} does not use the canonical actor prompt")
    prompt_ids = chat_template_token_ids(
        tokenizer,
        [{"role": "user", "content": cfg.actor_prompt_template.format(
            injection_char=cfg.injection_char
        )}],
    )
    if prompt_ids.count(cfg.injection_token_id) != 1:
        raise RuntimeError("canonical actor prompt does not contain exactly one marker token")
    marker_index = prompt_ids.index(cfg.injection_token_id)
    if (
        prompt_ids[marker_index - 1] != cfg.injection_left_neighbor_id
        or prompt_ids[marker_index + 1] != cfg.injection_right_neighbor_id
    ):
        raise RuntimeError("canonical actor prompt marker-neighbor contract failed")

    if cfg.injection_scale is None:
        raise RuntimeError("actor sidecar has no extraction.injection_scale")
    text_model = resolve_text_model(actor)
    embedding_weight = text_model.get_input_embeddings().weight
    embed_scale = float(resolve_embed_scale(actor.config))
    eos_ids = {int(tokenizer.eos_token_id)}
    configured_eos = getattr(actor.generation_config, "eos_token_id", None)
    if isinstance(configured_eos, (list, tuple)):
        eos_ids.update(map(int, configured_eos))
    elif configured_eos is not None:
        eos_ids.add(int(configured_eos))

    records_dir.mkdir(parents=True, exist_ok=True)
    existing: dict[int, dict[str, Any]] = {}
    for ordinal in range(len(rows)):
        path = record_path(records_dir, ordinal)
        if not path.is_file():
            continue
        value = json.loads(path.read_text())
        if value.get("run_fingerprint") != run_fingerprint or value.get("ordinal") != ordinal:
            raise RuntimeError(f"resume record mismatch: {path}")
        existing[ordinal] = value

    missing = [ordinal for ordinal in range(len(rows)) if ordinal not in existing]
    for start in range(0, len(missing), batch_size):
        ordinals = missing[start : start + batch_size]
        input_ids = torch.tensor(
            [prompt_ids] * len(ordinals), dtype=torch.long, device=device
        )
        raw_vectors = torch.stack(
            [torch.from_numpy(np.array(rows[o].activation, dtype=np.float32, copy=True)) for o in ordinals]
        ).to(embedding_weight.device)
        scaled_vectors = normalize_activation(raw_vectors, cfg.injection_scale)
        embeddings = embedding_weight[input_ids.to(embedding_weight.device)].float() * embed_scale
        injected = inject_at_marked_positions(
            input_ids.to(embedding_weight.device),
            embeddings,
            scaled_vectors,
            cfg.injection_token_id,
            cfg.injection_left_neighbor_id,
            cfg.injection_right_neighbor_id,
        )
        if not torch.equal(
            injected[:, marker_index], scaled_vectors.to(injected.dtype)
        ):
            raise RuntimeError("marker embedding does not equal the scaled activation batch")
        injected_norms = injected[:, marker_index].float().norm(dim=1)
        expected_norms = torch.full_like(injected_norms, float(cfg.injection_scale))
        if not torch.allclose(injected_norms, expected_norms, rtol=2e-5, atol=2e-3):
            raise RuntimeError(
                f"injected norms differ from target {cfg.injection_scale}: "
                f"{injected_norms.tolist()}"
            )
        injected = injected.to(actor.dtype)
        attention_mask = torch.ones(injected.shape[:2], dtype=torch.long, device=injected.device)
        with torch.inference_mode():
            generated = actor.generate(
                inputs_embeds=injected,
                attention_mask=attention_mask,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=(
                    tokenizer.pad_token_id
                    if tokenizer.pad_token_id is not None
                    else tokenizer.eos_token_id
                ),
                return_dict_in_generate=True,
            ).sequences
        for batch_index, ordinal in enumerate(ordinals):
            # With inputs_embeds and no input_ids, HF returns generated tokens only.
            token_ids = generated[batch_index].tolist()
            eos_position = next(
                (index for index, token_id in enumerate(token_ids) if token_id in eos_ids),
                None,
            )
            if eos_position is not None:
                token_ids = token_ids[: eos_position + 1]
            text = tokenizer.decode(
                token_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            explanation = extract_explanation(text)
            marker_leak = cfg.injection_char in text
            record = {
                "schema": SCHEMA + ".generation",
                "run_fingerprint": run_fingerprint,
                "ordinal": ordinal,
                "parquet_row_index": rows[ordinal].parquet_row_index,
                "document_id": rows[ordinal].document_id,
                "raw_response": text,
                "explanation": explanation,
                "extraction_ok": explanation is not None,
                "generated_token_ids": list(map(int, token_ids)),
                "generated_token_count": len(token_ids),
                "finish_reason": "eos" if eos_position is not None else "length",
                "injection_mechanism": "scaled_marker_embedding_replacement",
                "injection_apply_count_delta_for_batch": 1,
                "injection_target_l2_norm": float(cfg.injection_scale),
                "injected_l2_norm": float(injected_norms[batch_index].item()),
                "marker_row_matches_scaled_activation": True,
                "embed_scale": embed_scale,
                "marker_leak": marker_leak,
                "cjk_fraction": float(cjk_fraction(text)),
            }
            atomic_json(record_path(records_dir, ordinal), record)
            existing[ordinal] = record
            if marker_leak:
                raise RuntimeError(f"literal injection marker leaked at ordinal {ordinal}")
        print(f"generated {min(start + len(ordinals), len(missing))}/{len(missing)}", flush=True)
    return [existing[ordinal] for ordinal in range(len(rows))]


def score_records(
    *,
    critic: Any,
    tokenizer: Any,
    cfg: Any,
    rows: Sequence[Any],
    generated: Sequence[Mapping[str, Any]],
    device: str,
    batch_size: int,
    critic_predict: Any,
    normalize_activation: Any,
    verify_critic_suffix: Any,
) -> list[float | None]:
    import torch

    prompts: list[list[int] | None] = []
    for row in generated:
        explanation = row.get("explanation")
        if not isinstance(explanation, str) or not explanation:
            prompts.append(None)
            continue
        ids = tokenizer.encode(
            cfg.critic_prompt_template.format(explanation=explanation),
            add_special_tokens=False,
        )
        if not 0 < len(ids) <= 1024:
            prompts.append(None)
            continue
        verify_critic_suffix(ids, cfg.critic_suffix_ids, context=f"ordinal {row['ordinal']}")
        prompts.append(ids)

    rewards: list[float | None] = [None] * len(rows)
    valid = [index for index, ids in enumerate(prompts) if ids is not None]
    for start in range(0, len(valid), batch_size):
        indexes = valid[start : start + batch_size]
        max_len = max(len(prompts[index]) for index in indexes)  # type: ignore[arg-type]
        input_ids = torch.full(
            (len(indexes), max_len), tokenizer.eos_token_id, dtype=torch.long, device=device
        )
        attention = torch.zeros_like(input_ids)
        for batch_index, index in enumerate(indexes):
            ids = prompts[index]
            assert ids is not None
            input_ids[batch_index, : len(ids)] = torch.tensor(ids, device=device)
            attention[batch_index, : len(ids)] = 1
        with torch.inference_mode():
            prediction = critic_predict(critic, input_ids, attention, cfg.mse_scale)
        gold = torch.stack(
            [torch.from_numpy(np.array(rows[index].activation, dtype=np.float32, copy=True)) for index in indexes]
        ).to(device)
        mse = (
            (normalize_activation(prediction, cfg.mse_scale) - normalize_activation(gold, cfg.mse_scale))
            ** 2
        ).mean(dim=1)
        for batch_index, index in enumerate(indexes):
            value = float(mse[batch_index].item())
            rewards[index] = -value if math.isfinite(value) else None
    return rewards


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=("plausible", "implausible"), required=True)
    parser.add_argument("--actor", type=Path, required=True)
    parser.add_argument("--actor-sidecar", type=Path, required=True)
    parser.add_argument("--actor-conversion-manifest", type=Path, required=True)
    parser.add_argument("--critic", type=Path, required=True)
    parser.add_argument("--critic-verification", type=Path, required=True)
    parser.add_argument("--rl-parquet", type=Path, required=True)
    parser.add_argument("--nla-root", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--generation-batch-size", type=int, default=16)
    parser.add_argument("--critic-batch-size", type=int, default=32)
    args = parser.parse_args()

    sys.path.insert(0, str(args.code_root.expanduser().resolve()))
    sys.path.insert(0, str(args.nla_root.expanduser().resolve()))
    import torch
    import yaml
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from fve_matching.core import aggregate_fve, heldout_descriptor, load_fixed_heldout
    from nla.arch_adapters import resolve_embed_scale, resolve_text_model
    from nla.config import load_nla_config, verify_critic_suffix
    from nla.injection import inject_at_marked_positions
    from nla.models import NLACriticModel
    from nla.schema import (
        chat_template_token_ids,
        compute_predict_mean_baselines,
        extract_explanation,
        normalize_activation,
    )
    from nla.val_split import is_val_doc, val_doc_permille

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exact FVE requires exactly one visible GPU per process")
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)

    actor = args.actor.expanduser().resolve()
    critic = args.critic.expanduser().resolve()
    conversion_manifest = required_complete(
        args.actor_conversion_manifest.expanduser().resolve(), "actor conversion"
    )
    roundtrip = conversion_manifest.get("roundtrip_verification", {})
    if roundtrip.get("bitwise_equal") is not True:
        raise RuntimeError("actor conversion is not bitwise roundtrip-verified")
    critic_verification = required_complete(
        args.critic_verification.expanduser().resolve(), "critic verification"
    )
    if critic_verification.get("all_tensors_finite") is not True:
        raise RuntimeError("critic verification did not establish finite tensors")

    rl_parquet = args.rl_parquet.expanduser().resolve()
    rl_sha = sha256_file(rl_parquet)
    rows, total_rows, permille = load_fixed_heldout(
        rl_parquet,
        val_rows=10_000,
        n_prompts=N_PROMPTS,
        val_permille_fn=val_doc_permille,
        is_val_doc_fn=is_val_doc,
    )
    heldout = heldout_descriptor(rows)
    heldout.update(
        {
            "rl_parquet": str(rl_parquet),
            "rl_parquet_sha256": rl_sha,
            "rl_parquet_rows": total_rows,
            "val_doc_permille": permille,
            "selection": "first 256 rows in the CRC32 document-level validation bucket",
        }
    )

    actor_sidecar = args.actor_sidecar.expanduser().resolve()
    critic_sidecar = critic / "nla_meta.yaml"
    provenance = {
        "condition": args.condition,
        "actor_conversion_manifest": sha256_file(args.actor_conversion_manifest),
        "critic_verification": sha256_file(args.critic_verification),
        "actor_sidecar": sha256_file(actor_sidecar),
        "critic_sidecar": sha256_file(critic_sidecar),
        "rl_parquet": rl_sha,
        "heldout": heldout["eval_set_sha256"],
        "evaluator": sha256_file(Path(__file__).resolve()),
        "injection_mechanism": "scaled_marker_embedding_replacement",
        "nla_injection": sha256_file(args.nla_root / "nla/injection.py"),
        "nla_schema": sha256_file(args.nla_root / "nla/schema.py"),
        "nla_arch_adapters": sha256_file(args.nla_root / "nla/arch_adapters.py"),
        "nla_models": sha256_file(args.nla_root / "nla/models.py"),
    }
    run_fingerprint = hashlib.sha256(canonical_json(provenance)).hexdigest()
    output = args.output.expanduser().resolve()
    if output.is_file():
        existing = json.loads(output.read_text())
        if existing.get("complete") and existing.get("run_fingerprint") == run_fingerprint:
            print(json.dumps(existing["metrics"], indent=2), flush=True)
            return
        raise RuntimeError(f"refusing to overwrite mismatched result: {output}")

    tokenizer = AutoTokenizer.from_pretrained(actor)
    actor_cfg = load_nla_config(str(actor_sidecar.parent), tokenizer)
    critic_cfg = load_nla_config(str(critic), tokenizer)
    actor_meta = yaml.safe_load(actor_sidecar.read_text())
    if actor_meta.get("extraction", {}).get("layer_index") != 20 or actor_cfg.d_model != 3584:
        raise RuntimeError("actor sidecar does not specify Qwen layer 20 / width 3584")
    if critic_cfg.critic_num_layers != 20 or critic_cfg.d_model != 3584:
        raise RuntimeError("critic sidecar does not specify layer 20 / width 3584")
    if not critic_cfg.critic_prompt_template or not critic_cfg.critic_suffix_ids:
        raise RuntimeError("critic suffix-anchored prompt contract is missing")
    token_contract = (
        "injection_char", "injection_token_id", "injection_left_neighbor_id",
        "injection_right_neighbor_id",
    )
    for field in token_contract:
        if getattr(actor_cfg, field) != getattr(critic_cfg, field):
            raise RuntimeError(f"actor/critic token contract differs on {field}")

    print(f"loading actor {actor}", flush=True)
    actor_model = AutoModelForCausalLM.from_pretrained(
        actor, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(args.device).eval()
    actor_model.config.use_cache = True
    records_dir = output.with_suffix(".records")
    started = time.time()
    generated = generate_records(
        actor=actor_model,
        tokenizer=tokenizer,
        cfg=actor_cfg,
        rows=rows,
        run_fingerprint=run_fingerprint,
        records_dir=records_dir,
        batch_size=args.generation_batch_size,
        device=args.device,
        extract_explanation=extract_explanation,
        chat_template_token_ids=chat_template_token_ids,
        inject_at_marked_positions=inject_at_marked_positions,
        normalize_activation=normalize_activation,
        resolve_embed_scale=resolve_embed_scale,
        resolve_text_model=resolve_text_model,
    )
    generation_seconds = time.time() - started
    del actor_model
    gc.collect()
    torch.cuda.empty_cache()

    print(f"loading critic {critic}", flush=True)
    critic_model = NLACriticModel.from_pretrained(
        critic, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(args.device).eval()
    critic_model.value_head.to(dtype=torch.float32)
    if critic_model.value_head.weight.dtype != torch.float32:
        raise RuntimeError("critic runtime did not promote value head to fp32")

    def critic_predict(model, input_ids, attention_mask, mse_scale):
        # The supplied critics were trained with a normalized, fp32 value-head
        # input at the suffix anchor.  Avoid the older wrapper's full-sequence
        # head application and reproduce that scoring path directly.
        backbone = resolve_text_model(model.backbone)
        hidden = backbone.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).last_hidden_state
        last_index = attention_mask.sum(dim=1) - 1
        last_hidden = hidden[
            torch.arange(input_ids.shape[0], device=input_ids.device), last_index
        ].float()
        normalized = normalize_activation(last_hidden, mse_scale)
        with torch.autocast(device_type=normalized.device.type, enabled=False):
            return model.value_head(normalized.to(model.value_head.weight.dtype)).float()

    score_started = time.time()
    rewards = score_records(
        critic=critic_model,
        tokenizer=tokenizer,
        cfg=critic_cfg,
        rows=rows,
        generated=generated,
        device=args.device,
        batch_size=args.critic_batch_size,
        critic_predict=critic_predict,
        normalize_activation=normalize_activation,
        verify_critic_suffix=verify_critic_suffix,
    )
    scoring_seconds = time.time() - score_started
    activation_stack = torch.from_numpy(
        np.stack([np.asarray(row.activation, dtype=np.float32) for row in rows])
    )
    _, baseline_mse = compute_predict_mean_baselines(activation_stack, critic_cfg.mse_scale)
    metrics = aggregate_fve(rewards, baseline_mse)
    metrics.update(
        {
            "extraction_rate": sum(row["extraction_ok"] for row in generated) / len(generated),
            "marker_leak_count": sum(row["marker_leak"] for row in generated),
            "cjk_over_0_20_count": sum(row["cjk_fraction"] > 0.20 for row in generated),
            "generation_seconds": generation_seconds,
            "scoring_seconds": scoring_seconds,
        }
    )
    result_rows = [
        {
            **dict(generated[index]),
            "critic_reward": rewards[index],
            "included_in_fve": rewards[index] is not None and rewards[index] > -2.0,
        }
        for index in range(len(rows))
    ]
    result = {
        "schema": SCHEMA,
        "complete": True,
        "condition": args.condition,
        "run_fingerprint": run_fingerprint,
        "provenance": provenance,
        "heldout": heldout,
        "decode": {"temperature": 0.0, "top_p": 1.0, "max_new_tokens": MAX_NEW_TOKENS},
        "metrics": metrics,
        "rows": result_rows,
        "completed_at_unix": time.time(),
    }
    result["payload_sha256"] = hashlib.sha256(canonical_json(result)).hexdigest()
    atomic_json(output, result)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
