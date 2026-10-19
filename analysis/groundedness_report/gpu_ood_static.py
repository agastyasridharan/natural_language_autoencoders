"""Generate static OOD Qwen trajectories and extract raw layer-20 activations."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from .io import atomic_json, read_json, read_jsonl, sha256_file
from .protocol import load_evaluation_registry, load_protocol
from .sampling import sample_positions
from .spec import PROTOCOL_SHA256


class _StopForward(Exception):
    pass


def _decoder_layers(torch: Any, model: Any) -> tuple[Any, str]:
    best = None
    for name, module in model.named_modules():
        if (
            name.split(".")[-1] == "layers"
            and isinstance(module, torch.nn.ModuleList)
            and len(module) > 1
            and (best is None or len(module) > len(best[1]))
        ):
            best = (name, module)
    if best is None:
        raise RuntimeError("no decoder layer ModuleList found")
    return best[1], best[0]


def _task_key(task: dict[str, Any]) -> str:
    return f"{task['family_id']}:{task['task_id']}"


def _slug(task: dict[str, Any]) -> str:
    return hashlib.sha256(_task_key(task).encode()).hexdigest()[:24]


def _trajectory_task_id(trajectory: dict[str, Any]) -> str:
    """Read either the static top-level ID or an environment-run embedded ID."""
    value = trajectory.get("task_id")
    if value is None and isinstance(trajectory.get("task"), dict):
        value = trajectory["task"].get("task_id")
    if value is None:
        raise ValueError("trajectory has no task ID")
    return str(value)


def _activation_turn_index(activation: dict[str, Any]) -> tuple[int, str]:
    """Normalize static-environment and ToolEmu activation turn fields."""
    turn = activation.get("turn_index")
    agent_call = activation.get("agent_call_index")
    if turn is None and agent_call is None:
        raise ValueError("activation lacks turn_index and agent_call_index")
    if turn is not None and agent_call is not None and int(turn) != int(agent_call):
        raise ValueError("activation turn_index and agent_call_index disagree")
    if turn is not None:
        return int(turn), "turn_index"
    return int(agent_call), "agent_call_index"


def _atomic_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp.npy")
    np.save(temporary, value)
    os.replace(temporary, path)


def _capture_layer(
    *,
    torch: Any,
    model: Any,
    layer: Any,
    input_ids: Any,
) -> Any:
    captured: dict[str, Any] = {}

    def grab(_module: Any, _inputs: Any, output: Any) -> None:
        captured["h"] = output[0] if isinstance(output, tuple) else output
        raise _StopForward()

    handle = layer.register_forward_hook(grab)
    try:
        model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=False)
    except _StopForward:
        pass
    finally:
        handle.remove()
    if "h" not in captured:
        raise RuntimeError("layer hook did not capture an activation")
    value = captured["h"][0].float().cpu()
    if not bool(torch.isfinite(value).all()):
        raise ValueError("captured activation contains nonfinite values")
    return value


def _chat_ids(tokenizer: Any, messages: list[dict[str, str]]) -> list[int]:
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(value) for value in ids]


def _one_trajectory(
    *,
    torch: Any,
    model: Any,
    tokenizer: Any,
    target_layer: Any,
    task: dict[str, Any],
    protocol: dict[str, Any],
    maximum_new_tokens: int,
    output_dir: Path,
) -> dict[str, Any]:
    trajectory_id = _task_key(task)
    slug = _slug(task)
    trajectory_path = output_dir / "trajectories" / str(task["family_id"]) / f"{slug}.json"
    if trajectory_path.exists():
        existing = read_json(trajectory_path)
        if existing.get("protocol_sha256") != PROTOCOL_SHA256:
            raise ValueError(f"stale trajectory artifact at {trajectory_path}")
        for activation in existing.get("activations") or []:
            if not (output_dir / activation["vector_file"]).exists():
                raise ValueError(f"trajectory {trajectory_path} has a missing activation vector")
        return {"status": "skipped", "trajectory": existing}

    messages = [dict(row) for row in task["messages"]]
    followups = [str(task["followup_user"])] if task.get("followup_user") is not None else []
    turns: list[dict[str, Any]] = []
    eligible: list[dict[str, Any]] = []
    device = model.get_input_embeddings().weight.device
    special_ids = set(int(value) for value in tokenizer.all_special_ids)
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("tokenizer has no EOS token")

    for turn_index in range(1 + len(followups)):
        prompt_ids = _chat_ids(tokenizer, messages)
        if len(prompt_ids) < 1:
            raise ValueError("chat template produced an empty prompt")
        max_context = int(getattr(model.config, "max_position_embeddings", 32768))
        if len(prompt_ids) + maximum_new_tokens > max_context:
            raise ValueError(
                f"prompt plus generation budget exceeds context: {len(prompt_ids)}+{maximum_new_tokens}>{max_context}"
            )
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        with torch.inference_mode():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                do_sample=False,
                max_new_tokens=maximum_new_tokens,
                pad_token_id=eos_id,
                eos_token_id=eos_id,
                use_cache=True,
            )[0]
        continuation_ids = [int(value) for value in generated[len(prompt_ids) :].tolist()]
        response = tokenizer.decode(continuation_ids, skip_special_tokens=True)
        full_ids = generated.view(1, -1)
        with torch.inference_mode():
            hidden = _capture_layer(
                torch=torch,
                model=model,
                layer=target_layer,
                input_ids=full_ids,
            )
        eligible_count = 0
        for response_index, token_id in enumerate(continuation_ids):
            absolute_position = len(prompt_ids) + response_index
            if token_id in special_ids or absolute_position < 50:
                continue
            position_key = turn_index * 1_000_000 + response_index
            eligible.append({
                "position_key": position_key,
                "turn_index": turn_index,
                "assistant_response_token_index": response_index,
                "token_position": absolute_position,
                "token_id": token_id,
                "read_token": tokenizer.decode([token_id], skip_special_tokens=False),
                "causal_context": tokenizer.decode(
                    generated[: absolute_position + 1].tolist(), skip_special_tokens=False
                ),
                "vector": hidden[absolute_position].numpy().astype(np.float32, copy=True),
            })
            eligible_count += 1
        turns.append({
            "turn_index": turn_index,
            "input_messages": [dict(row) for row in messages],
            "prompt_tokens": len(prompt_ids),
            "prompt_ids_sha256": hashlib.sha256(np.asarray(prompt_ids, dtype=np.int64).tobytes()).hexdigest(),
            "continuation_token_ids": continuation_ids,
            "response": response,
            "eligible_assistant_tokens": eligible_count,
            "closed_by": "eos" if continuation_ids and continuation_ids[-1] == eos_id else "max_tokens_or_other_stop",
        })
        messages.append({"role": "assistant", "content": response})
        if turn_index < len(followups):
            messages.append({"role": "user", "content": followups[turn_index]})

    selected_keys = set(sample_positions(
        [int(row["position_key"]) for row in eligible],
        trajectory_id=trajectory_id,
        seed=int(protocol["seed"]),
        maximum=4,
    ))
    if not selected_keys:
        raise ValueError("trajectory has no eligible assistant-language token after position 50")
    activation_rows = []
    for ordinal, row in enumerate(
        sorted((item for item in eligible if item["position_key"] in selected_keys), key=lambda item: item["position_key"])
    ):
        vector_file = Path("activations") / str(task["family_id"]) / f"{slug}-{ordinal}.npy"
        _atomic_npy(output_dir / vector_file, row.pop("vector"))
        activation_rows.append({
            **row,
            "vector_file": str(vector_file),
            "vector_sha256": sha256_file(output_dir / vector_file),
        })
    trajectory = {
        "schema_version": 1,
        "protocol_sha256": PROTOCOL_SHA256,
        "replication_label": task["replication_label"],
        "family_id": task["family_id"],
        "task_id": task["task_id"],
        "task_sha256": task["task_sha256"],
        "trajectory_id": trajectory_id,
        "interaction_kind": task["interaction_kind"],
        "source": task["source"],
        "reference": task.get("reference") or {},
        "target_model": protocol["models"]["target"],
        "generation": {
            "do_sample": False,
            "temperature": None,
            "max_new_tokens_per_assistant_turn": maximum_new_tokens,
        },
        "turns": turns,
        "eligible_position_population": len(eligible),
        "activations": activation_rows,
        "status": "complete",
    }
    atomic_json(trajectory_path, trajectory)
    return {"status": "written", "trajectory": trajectory}


def _run(args: argparse.Namespace) -> int:
    if args.num_shards < 1 or not (0 <= args.shard_index < args.num_shards):
        raise ValueError("shard index must lie in [0, num_shards)")
    protocol = load_protocol()
    registry = load_evaluation_registry()
    generation = registry["trajectory_generation"]
    if not isinstance(generation, dict):
        raise ValueError("evaluation registry lacks detailed trajectory generation settings")
    maximum_new_tokens = int(generation["max_new_tokens_per_assistant_turn"])

    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    task_dir = Path(args.tasks)
    family_filter = set(args.family or [])
    tasks = []
    for source in sorted(task_dir.glob("*.jsonl")):
        for task in read_jsonl(source):
            if family_filter and task["family_id"] not in family_filter:
                continue
            if task.get("protocol_sha256") != PROTOCOL_SHA256:
                raise ValueError(f"task {task.get('task_id')} has a stale protocol hash")
            rank = int.from_bytes(hashlib.sha256(_task_key(task).encode()).digest()[:8], "big")
            if rank % args.num_shards == args.shard_index:
                tasks.append(task)
    tasks.sort(key=lambda row: (str(row["family_id"]), str(row["task_id"])))
    if args.limit is not None:
        tasks = tasks[: args.limit]
    if not tasks:
        raise ValueError("no tasks assigned to this shard")

    target = protocol["models"]["target"]
    checkpoint = snapshot_download(
        repo_id=target["repo_id"], revision=target["revision"], local_files_only=True
    )
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).eval()
    layers, layer_path = _decoder_layers(torch, model)
    layer_index = int(target["layer"])
    if layer_index >= len(layers):
        raise ValueError(f"target layer {layer_index} absent from {layer_path}")
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_path = output_dir / f"shard-{args.shard_index:02d}-meta.json"
    started = time.time()
    atomic_json(meta_path, {
        "status": "running",
        "protocol_sha256": PROTOCOL_SHA256,
        "target": target,
        "checkpoint": Path(checkpoint).name,
        "generation": generation,
        "layer_path": layer_path,
        "assigned_tasks": len(tasks),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
    })
    written = skipped = 0
    failures = []
    for position, task in enumerate(tasks, start=1):
        try:
            result = _one_trajectory(
                torch=torch,
                model=model,
                tokenizer=tokenizer,
                target_layer=layers[layer_index],
                task=task,
                protocol=protocol,
                maximum_new_tokens=maximum_new_tokens,
                output_dir=output_dir,
            )
            written += result["status"] == "written"
            skipped += result["status"] == "skipped"
        except Exception as error:
            failure = {
                "family_id": task["family_id"],
                "task_id": task["task_id"],
                "error": f"{type(error).__name__}: {error}",
            }
            failures.append(failure)
            atomic_json(output_dir / "failures" / f"{_slug(task)}.json", {
                "schema_version": 1,
                "protocol_sha256": PROTOCOL_SHA256,
                **failure,
                "task": task,
            })
        if position % 5 == 0 or position == len(tasks):
            print(
                f"[ood {position}/{len(tasks)} shard={args.shard_index}] "
                f"written={written} skipped={skipped} failures={len(failures)}",
                flush=True,
            )
    atomic_json(meta_path, {
        "status": "done" if not failures else "failed",
        "protocol_sha256": PROTOCOL_SHA256,
        "target": target,
        "checkpoint": Path(checkpoint).name,
        "generation": generation,
        "layer_path": layer_path,
        "assigned_tasks": len(tasks),
        "written": written,
        "skipped": skipped,
        "failures": failures,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "elapsed_seconds": round(time.time() - started, 3),
    })
    return 1 if failures else 0


def _assemble(args: argparse.Namespace) -> int:
    protocol = load_protocol()
    partials = Path(args.partials)
    trajectories = [read_json(path) for path in sorted((partials / "trajectories").glob("*/*.json"))]
    for row in trajectories:
        if row.get("protocol_sha256") != PROTOCOL_SHA256 or row.get("status") != "complete":
            raise ValueError(f"incompatible or incomplete trajectory {row.get('trajectory_id')}")
    family_counts: dict[str, int] = {}
    for row in trajectories:
        family_counts[str(row["family_id"])] = family_counts.get(str(row["family_id"]), 0) + 1
    invalid = {family: count for family, count in family_counts.items() if count != 50}
    if invalid:
        raise ValueError(f"static families must each have 50 trajectories; observed {invalid}")
    activation_refs = []
    for trajectory in trajectories:
        for activation in trajectory["activations"]:
            activation_refs.append((trajectory, activation))
    activation_refs.sort(key=lambda pair: (
        str(pair[0]["family_id"]),
        str(pair[0]["trajectory_id"]),
        int(pair[1]["position_key"]),
    ))
    if not activation_refs:
        raise ValueError("no activation artifacts to assemble")
    first_vector = np.load(partials / activation_refs[0][1]["vector_file"])
    if first_vector.ndim != 1:
        raise ValueError(f"first activation is not rank one: {first_vector.shape}")
    width = int(first_vector.shape[0])
    gold = np.empty((len(activation_refs), width), dtype=np.float32)
    manifest = []
    for idx, (trajectory, activation) in enumerate(activation_refs):
        task_id = _trajectory_task_id(trajectory)
        turn_index, source_turn_field = _activation_turn_index(activation)
        vector_path = partials / activation["vector_file"]
        vector = np.load(vector_path)
        if vector.shape != (width,) or not np.all(np.isfinite(vector)):
            raise ValueError(f"invalid vector {vector_path}: {vector.shape}")
        if sha256_file(vector_path) != activation["vector_sha256"]:
            raise ValueError(f"activation hash mismatch at {vector_path}")
        gold[idx] = vector
        manifest.append({
            "idx": idx,
            "doc_id": trajectory["trajectory_id"],
            "trajectory_id": trajectory["trajectory_id"],
            "family_id": trajectory["family_id"],
            "task_id": task_id,
            "position": activation["token_position"],
            "position_key": activation["position_key"],
            "turn_index": turn_index,
            "source_turn_field": source_turn_field,
            "assistant_response_token_index": activation["assistant_response_token_index"],
            "in_range": True,
            "read_token": activation["read_token"],
            "causal_context": activation["causal_context"],
            "source_scope": "exact_serialized_target_model_context_through_the_activation_token",
            "norm": "none",
            "h_norm": float(np.linalg.norm(vector)),
            "trajectory_file": str(
                Path("trajectories")
                / str(trajectory["family_id"])
                / f"{_slug({'family_id': trajectory['family_id'], 'task_id': task_id})}.json"
            ),
        })
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    _atomic_npy(output / "gold.npy", gold)
    atomic_json(output / "manifest.json", manifest)
    atomic_json(output / "assemble_meta.json", {
        "schema_version": 1,
        "protocol_sha256": PROTOCOL_SHA256,
        "norm": "none",
        "families": family_counts,
        "trajectories": len(trajectories),
        "activations": len(manifest),
        "gold_sha256": sha256_file(output / "gold.npy"),
        "manifest_sha256": sha256_file(output / "manifest.json"),
    })
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--tasks", required=True)
    run.add_argument("--out", required=True)
    run.add_argument("--num-shards", type=int, default=1)
    run.add_argument("--shard-index", type=int, default=0)
    run.add_argument("--family", action="append")
    run.add_argument("--limit", type=int)
    run.set_defaults(func=_run)
    assemble = commands.add_parser("assemble")
    assemble.add_argument("--partials", required=True)
    assemble.add_argument("--out", required=True)
    assemble.set_defaults(func=_assemble)
    return root


def main() -> int:
    args = parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
