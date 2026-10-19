#!/usr/bin/env python3
"""Derive a layer-complete evaluation sidecar without modifying source checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any

import yaml


SCHEMA = "nla.warm_start.derived_actor_sidecar.v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    with tmp.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actor-sidecar", type=Path, required=True)
    parser.add_argument("--critic-sidecar", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    actor_path = args.actor_sidecar.expanduser().resolve()
    critic_path = args.critic_sidecar.expanduser().resolve()
    actor: dict[str, Any] = yaml.safe_load(actor_path.read_text())
    critic: dict[str, Any] = yaml.safe_load(critic_path.read_text())
    if actor.get("role") != "actor" or critic.get("role") != "critic":
        raise RuntimeError("expected an actor sidecar paired with a critic sidecar")

    common_fields = ("d_model", "prompt_templates", "trained_on")
    for field in common_fields:
        if actor.get(field) != critic.get(field):
            raise RuntimeError(f"actor/critic sidecars disagree on {field}")
    token_fields = (
        "injection_char",
        "injection_token_id",
        "injection_left_neighbor_id",
        "injection_right_neighbor_id",
    )
    for field in token_fields:
        if actor.get("tokens", {}).get(field) != critic.get("tokens", {}).get(field):
            raise RuntimeError(f"actor/critic sidecars disagree on tokens.{field}")
    actor_mse = actor.get("extraction", {}).get("mse_scale")
    critic_mse = critic.get("extraction", {}).get("mse_scale")
    if actor_mse != critic_mse:
        raise RuntimeError("actor/critic sidecars disagree on mse_scale")

    paired_layer = critic.get("critic", {}).get("extraction_layer_index")
    if paired_layer != 20:
        raise RuntimeError(f"paired critic extraction layer is {paired_layer!r}, expected 20")
    actor_layer = actor.setdefault("extraction", {}).get("layer_index")
    if actor_layer not in (None, paired_layer):
        raise RuntimeError(
            f"actor layer_index={actor_layer!r} conflicts with paired critic {paired_layer}"
        )
    actor["extraction"]["layer_index"] = paired_layer

    output = args.output.expanduser().resolve()
    payload = yaml.safe_dump(actor, sort_keys=False, allow_unicode=False).encode()
    atomic_bytes(output, payload)
    derivation = {
        "schema": SCHEMA,
        "source_actor_sidecar": str(actor_path),
        "source_actor_sha256": sha256_file(actor_path),
        "source_critic_sidecar": str(critic_path),
        "source_critic_sha256": sha256_file(critic_path),
        "output": str(output),
        "output_sha256": sha256_file(output),
        "only_semantic_change": {"extraction.layer_index": paired_layer},
        "reason": (
            "the supplied actor sidecar omits layer_index; the paired critic and "
            "activation manifests both independently specify layer 20"
        ),
    }
    raw = (json.dumps(derivation, sort_keys=True, separators=(",", ":")) + "\n").encode()
    derivation["payload_sha256"] = hashlib.sha256(raw).hexdigest()
    atomic_bytes(
        args.manifest.expanduser().resolve(),
        (json.dumps(derivation, sort_keys=True, indent=2) + "\n").encode(),
    )
    print(json.dumps(derivation, indent=2), flush=True)


if __name__ == "__main__":
    main()
