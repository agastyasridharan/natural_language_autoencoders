"""Generate preregistered AV explanations from raw activation artifacts.

This process loads only the public Qwen AV checkpoint. It embeds the canonical
sidecar prompt, scans the token IDs for the sidecar marker and its neighbors,
overwrites that embedding with the sidecar-scaled activation, and samples one
explanation. Per-activation JSON artifacts make sharded generation resumable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .io import atomic_json, read_json, sha256_file
from .protocol import load_protocol
from .spec import PROTOCOL_SHA256


_EXPLANATION_OPEN = "<explanation>"
_EXPLANATION_CLOSE = "</explanation>"
_CJK = re.compile(r"[\u3000-\u9fff\uf900-\ufaff\uff00-\uffef]")


def activation_id(row: dict[str, Any]) -> str:
    doc_id = str(row.get("doc_id") or "")
    if not doc_id:
        raise ValueError("manifest row has no doc_id")
    return f"{doc_id}:{int(row['idx'])}"


def keyed_seed(study_seed: int, item_id: str) -> int:
    digest = hashlib.sha256(f"{study_seed}:{item_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def extract_explanation(generated: str) -> tuple[str, str]:
    """Return (text, status) under the frozen tag/EOS parsing rule."""
    start = generated.find(_EXPLANATION_OPEN)
    if start < 0:
        return "", "missing_opening_tag"
    start += len(_EXPLANATION_OPEN)
    end = generated.find(_EXPLANATION_CLOSE, start)
    if end < 0:
        return generated[start:].strip(), "closed_by_eos"
    return generated[start:end].strip(), "closed_by_tag"


def _sidecar(path: Path, tokenizer: Any, protocol: dict[str, Any]) -> dict[str, Any]:
    meta = yaml.safe_load((path / "nla_meta.yaml").read_text(encoding="utf-8"))
    tokens = meta["tokens"]
    extraction = meta["extraction"]
    templates = meta["prompt_templates"]
    result = {
        "injection_char": str(tokens["injection_char"]),
        "injection_token_id": int(tokens["injection_token_id"]),
        "left_id": int(tokens["injection_left_neighbor_id"]),
        "right_id": int(tokens["injection_right_neighbor_id"]),
        "injection_scale": float(extraction["injection_scale"]),
        "d_model": int(meta["d_model"]),
        "actor_template": str(templates.get("av") or templates["actor"]),
    }
    expected_scale = float(protocol["models"]["injection_scale"])
    if result["injection_scale"] != expected_scale:
        raise ValueError(
            f"sidecar injection scale {result['injection_scale']} != preregistered {expected_scale}"
        )
    encoded_marker = tokenizer.encode(result["injection_char"], add_special_tokens=False)
    if encoded_marker != [result["injection_token_id"]]:
        raise ValueError("live tokenizer disagrees with the sidecar injection token")
    return result


def _actor_inputs(tokenizer: Any, sidecar: dict[str, Any]) -> tuple[list[int], int]:
    content = sidecar["actor_template"].format(
        injection_char=sidecar["injection_char"]
    )
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=True,
        add_generation_prompt=True,
    )
    ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
    matches = [i for i, value in enumerate(ids) if value == sidecar["injection_token_id"]]
    if len(matches) != 1:
        raise ValueError(f"canonical prompt contains {len(matches)} injection markers")
    position = matches[0]
    if position == 0 or position + 1 == len(ids):
        raise ValueError("injection marker is at a sequence edge")
    observed = (ids[position - 1], ids[position + 1])
    expected = (sidecar["left_id"], sidecar["right_id"])
    if observed != expected:
        raise ValueError(f"sidecar neighbor mismatch: observed {observed}, expected {expected}")
    return list(ids), position


def _scaled_vector(torch: Any, value: np.ndarray, scale: float, width: int) -> Any:
    vector = torch.tensor(np.array(value, copy=True), dtype=torch.float32).view(1, -1)
    if vector.shape != (1, width) or not bool(torch.isfinite(vector).all()):
        raise ValueError(f"activation must be finite [1,{width}], got {tuple(vector.shape)}")
    norm = vector.norm(dim=-1, keepdim=True)
    if float(norm.item()) <= 1e-12:
        raise ValueError("activation has zero norm")
    return vector / norm * float(scale)


def _scan_and_inject(
    torch: Any,
    input_ids: Any,
    embeddings: Any,
    vector: Any,
    sidecar: dict[str, Any],
) -> Any:
    """Scan full token IDs and inject only a marker with canonical neighbors."""
    matches = (input_ids == sidecar["injection_token_id"]).nonzero().tolist()
    valid: list[tuple[int, int]] = []
    seq_len = input_ids.shape[-1]
    for batch_index, position in matches:
        if position in {0, seq_len - 1}:
            continue
        if (
            int(input_ids[batch_index, position - 1]) == sidecar["left_id"]
            and int(input_ids[batch_index, position + 1]) == sidecar["right_id"]
        ):
            valid.append((batch_index, position))
    if len(valid) != 1:
        raise RuntimeError(f"found {len(valid)} valid injection sites, expected exactly one")
    output = embeddings.clone()
    batch_index, position = valid[0]
    output[batch_index, position] = vector[0].to(output.device, output.dtype)
    return output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.num_shards < 1 or not (0 <= args.shard_index < args.num_shards):
        raise ValueError("shard index must lie in [0, num_shards)")
    protocol = load_protocol()
    av_spec = protocol["models"]["verbalizer"]
    generation = av_spec["inference"]
    if generation != {
        "samples_per_activation": 1,
        "do_sample": True,
        "temperature": 1.0,
        "top_p": 1.0,
        "max_new_tokens": 500,
        "rng": "sha256_keyed_by_seed_and_activation_id",
        "generation_order_independent": True,
        "explanation_parse_rule": (
            "require_opening_explanation_tag_and_take_text_until_closing_tag_or_eos"
        ),
        "missing_opening_tag_rule": "unresolved_generation_failure",
    }:
        raise ValueError("generation configuration differs from the frozen protocol")

    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    gold_dir = Path(args.gold_dir)
    output_dir = Path(args.out)
    reads_dir = output_dir / "reads"
    reads_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_json(gold_dir / "manifest.json")
    gold = np.load(gold_dir / "gold.npy", mmap_mode="r")
    if len(manifest) != len(gold):
        raise ValueError("gold array and manifest length differ")

    checkpoint = snapshot_download(
        repo_id=av_spec["repo_id"], revision=av_spec["revision"], local_files_only=True
    )
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    sidecar = _sidecar(Path(checkpoint), tokenizer, protocol)
    if not torch.cuda.is_available():
        raise RuntimeError("AV verbalization requires CUDA")
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to("cuda:0").eval()
    if int(model.config.hidden_size) != sidecar["d_model"]:
        raise ValueError("live AV hidden size differs from sidecar")
    ids, marker_position = _actor_inputs(tokenizer, sidecar)
    device = model.get_input_embeddings().weight.device
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("tokenizer has no EOS token")

    candidates = []
    for row in manifest:
        item_id = activation_id(row)
        rank = int.from_bytes(hashlib.sha256(item_id.encode()).digest()[:8], "big")
        if rank % args.num_shards == args.shard_index:
            candidates.append(row)
    if args.limit is not None:
        candidates = candidates[: args.limit]

    atomic_json(output_dir / f"shard-{args.shard_index:02d}-meta.json", {
        "status": "running",
        "protocol_sha256": PROTOCOL_SHA256,
        "av_repo_id": av_spec["repo_id"],
        "av_revision": av_spec["revision"],
        "resolved_checkpoint": Path(checkpoint).name,
        "gold_sha256": sha256_file(gold_dir / "gold.npy"),
        "manifest_sha256": sha256_file(gold_dir / "manifest.json"),
        "generation": generation,
        "marker_position": marker_position,
        "prompt_tokens": len(ids),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "items_assigned": len(candidates),
        "started_at_unix": time.time(),
    })

    written = skipped = missing_tag = 0
    started = time.time()
    for position, row in enumerate(candidates, start=1):
        idx = int(row["idx"])
        item_id = activation_id(row)
        destination = reads_dir / f"{idx:06d}.json"
        if destination.exists() and not args.force:
            cached = read_json(destination)
            if cached.get("protocol_sha256") != PROTOCOL_SHA256:
                raise ValueError(f"stale AV artifact at {destination}")
            skipped += 1
            continue
        vector = _scaled_vector(
            torch, np.asarray(gold[idx]), sidecar["injection_scale"], sidecar["d_model"]
        )
        embeddings = model.get_input_embeddings()(input_ids)
        injected = _scan_and_inject(torch, input_ids, embeddings, vector, sidecar)
        seed = keyed_seed(int(protocol["seed"]), item_id)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        generated_ids = model.generate(
            input_ids=input_ids,
            inputs_embeds=injected,
            attention_mask=attention_mask,
            do_sample=True,
            temperature=1.0,
            top_p=1.0,
            max_new_tokens=500,
            pad_token_id=eos_id,
            eos_token_id=eos_id,
            use_cache=True,
        )
        continuation = tokenizer.decode(
            generated_ids[0, input_ids.shape[1] :], skip_special_tokens=False
        )
        explanation, parse_status = extract_explanation(continuation)
        missing_tag += parse_status == "missing_opening_tag"
        cjk_characters = sorted(set(_CJK.findall(explanation)))
        if sidecar["injection_char"] in cjk_characters:
            atomic_json(output_dir / "failures" / f"{idx:06d}.json", {
                "schema_version": 1,
                "protocol_sha256": PROTOCOL_SHA256,
                "idx": idx,
                "doc_id": row["doc_id"],
                "activation_id": item_id,
                "seed": seed,
                "parse_status": parse_status,
                "cjk_characters": cjk_characters,
                "injection_marker_emitted": True,
                "explanation": explanation,
                "continuation": continuation,
                "diagnostic_only_not_admitted": True,
            })
            raise RuntimeError(
                f"injection marker emitted for {item_id}; expansion stopped before segmentation"
            )
        atomic_json(destination, {
            "schema_version": 1,
            "protocol_sha256": PROTOCOL_SHA256,
            "idx": idx,
            "doc_id": row["doc_id"],
            "activation_id": item_id,
            "activation_sha256": hashlib.sha256(np.asarray(gold[idx]).tobytes()).hexdigest(),
            "seed": seed,
            "explanation": explanation,
            "parse_status": parse_status,
            "continuation": continuation,
            "non_marker_cjk_characters": cjk_characters,
            "av_repo_id": av_spec["repo_id"],
            "av_revision": av_spec["revision"],
            "generation": generation,
        })
        written += 1
        if position % 10 == 0 or position == len(candidates):
            print(
                f"[av {position}/{len(candidates)} shard={args.shard_index}] "
                f"written={written} skipped={skipped} missing_tag={missing_tag}",
                flush=True,
            )

    atomic_json(output_dir / f"shard-{args.shard_index:02d}-meta.json", {
        "status": "done",
        "protocol_sha256": PROTOCOL_SHA256,
        "av_repo_id": av_spec["repo_id"],
        "av_revision": av_spec["revision"],
        "resolved_checkpoint": Path(checkpoint).name,
        "gold_sha256": sha256_file(gold_dir / "gold.npy"),
        "manifest_sha256": sha256_file(gold_dir / "manifest.json"),
        "generation": generation,
        "marker_position": marker_position,
        "prompt_tokens": len(ids),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "items_assigned": len(candidates),
        "written": written,
        "skipped": skipped,
        "missing_opening_tag": missing_tag,
        "elapsed_seconds": round(time.time() - started, 3),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
