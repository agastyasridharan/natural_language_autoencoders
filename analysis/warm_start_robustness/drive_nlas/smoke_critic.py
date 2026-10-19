#!/usr/bin/env python3
"""Load a supplied critic and score durable AV smoke generations."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-generations", type=Path, required=True)
    parser.add_argument("--activation-manifest", type=Path, required=True)
    parser.add_argument("--critic-dir", type=Path, required=True)
    parser.add_argument("--nla-root", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(args.code_root.expanduser().resolve()))
    sys.path.insert(0, str(args.nla_root.expanduser().resolve()))
    from claim_agreement.generate_av import atomic_write_json, load_activation_manifests
    from nla.arch_adapters import resolve_text_model
    from nla.config import load_nla_config, verify_critic_suffix
    from nla.models import NLACriticModel
    from nla.schema import normalize_activation

    items = load_activation_manifests([args.activation_manifest])
    by_id = {item.activation_id: item for item in items}
    rows = [json.loads(line) for line in args.raw_generations.read_text().splitlines()]
    valid = [row for row in rows if row.get("generation_ok")]
    if not valid:
        raise RuntimeError("no valid AV generations to score")
    if any(row.get("condition") != args.condition for row in rows):
        raise RuntimeError("generation condition mismatch")

    critic_dir = args.critic_dir.expanduser().resolve()
    tokenizer = AutoTokenizer.from_pretrained(critic_dir)
    cfg = load_nla_config(str(critic_dir), tokenizer)
    if cfg.critic_num_layers != 20 or cfg.d_model != 3584:
        raise RuntimeError(
            f"critic sidecar mismatch: layer={cfg.critic_num_layers}, d_model={cfg.d_model}"
        )
    if not cfg.critic_prompt_template or not cfg.critic_suffix_ids:
        raise RuntimeError("critic prompt/suffix contract is absent")

    prompts: list[list[int]] = []
    activations: list[torch.Tensor] = []
    for row in valid:
        item = by_id[row["activation_id"]]
        vector = np.asarray(item.source.array[item.activation_index], dtype=np.float32)
        ids = tokenizer.encode(
            cfg.critic_prompt_template.format(explanation=row["explanation"]),
            add_special_tokens=False,
        )
        verify_critic_suffix(ids, cfg.critic_suffix_ids, context=row["activation_id"])
        prompts.append(ids)
        activations.append(torch.from_numpy(vector.copy()))

    critic = NLACriticModel.from_pretrained(
        critic_dir,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(args.device).eval()
    critic.value_head.to(dtype=torch.float32)
    if critic.value_head.weight.dtype != torch.float32:
        raise RuntimeError(f"runtime value head was not promoted to fp32: {critic.value_head.weight.dtype}")

    pad_id = tokenizer.eos_token_id
    max_len = max(map(len, prompts))
    input_ids = torch.full(
        (len(prompts), max_len), pad_id, dtype=torch.long, device=args.device
    )
    attention = torch.zeros_like(input_ids)
    for index, ids in enumerate(prompts):
        input_ids[index, : len(ids)] = torch.tensor(ids, device=args.device)
        attention[index, : len(ids)] = 1
    with torch.inference_mode():
        backbone = resolve_text_model(critic.backbone)
        hidden = backbone.model(
            input_ids=input_ids,
            attention_mask=attention,
            use_cache=False,
        ).last_hidden_state
        last_index = attention.sum(dim=1) - 1
        last_hidden = hidden[
            torch.arange(input_ids.shape[0], device=input_ids.device), last_index
        ].float()
        normalized_hidden = normalize_activation(last_hidden, cfg.mse_scale)
        with torch.autocast(device_type=normalized_hidden.device.type, enabled=False):
            prediction = critic.value_head(normalized_hidden).float()
    gold = torch.stack(activations).to(args.device)
    pred_n = normalize_activation(prediction, cfg.mse_scale)
    gold_n = normalize_activation(gold, cfg.mse_scale)
    mse = ((pred_n - gold_n) ** 2).mean(dim=1)
    cosine = F.cosine_similarity(pred_n, gold_n, dim=1)
    if not torch.isfinite(mse).all().item() or not torch.isfinite(cosine).all().item():
        raise RuntimeError("critic produced non-finite smoke metrics")

    scored = []
    for row, mse_value, cosine_value in zip(valid, mse.tolist(), cosine.tolist()):
        scored.append(
            {
                "activation_id": row["activation_id"],
                "sample_index": row["sample_index"],
                "normalized_mse": float(mse_value),
                "cosine_similarity": float(cosine_value),
            }
        )
    result = {
        "schema": "nla.warm_start.critic_smoke.v1",
        "complete": True,
        "condition": args.condition,
        "generated_records": len(rows),
        "valid_explanations": len(valid),
        "critic_scores_finite": True,
        "critic_runtime_value_head_dtype": str(critic.value_head.weight.dtype),
        "mean_normalized_mse": float(mse.mean().item()),
        "mean_cosine_similarity": float(cosine.mean().item()),
        "min_cosine_similarity": float(cosine.min().item()),
        "max_cosine_similarity": float(cosine.max().item()),
        "scores": scored,
    }
    if not all(math.isfinite(float(result[key])) for key in (
        "mean_normalized_mse", "mean_cosine_similarity", "min_cosine_similarity",
        "max_cosine_similarity",
    )):
        raise RuntimeError("summary contains non-finite values")
    atomic_write_json(args.output.expanduser().resolve(), result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
