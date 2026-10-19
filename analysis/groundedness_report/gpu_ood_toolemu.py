"""Run Qwen trajectories with the pinned official ToolEmu emulator."""
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

from .gpu_ood_static import _atomic_npy, _capture_layer, _decoder_layers, _slug, _task_key
from .gpu_ood_tau2 import (
    TERRA_KEY_NAME,
    TERRA_MODEL,
    TerraInfrastructureError,
    _TerraClient,
    _preflight_openai_endpoint,
)
from .io import atomic_json, read_json, read_jsonl, sha256_file
from .protocol import load_protocol
from .spec import PROTOCOL_SHA256, TEXT_MODEL_CORRECTION_AMENDMENT_SHA256
from .build_toolemu_tasks import TOOLEMU_NONCYBER_AMENDMENT_SHA256


FAMILY_ID = "alignment_overly_agentic"
TOOLEMU_BOUNDED_SIMULATOR_AMENDMENT_SHA256 = (
    "407c80a5b9c275599db7018f14146d28d9de987d34fae2b7636d60d1ef7cb426"
)
TOOLEMU_FORMAT_BUDGET_AMENDMENT_SHA256 = (
    "867231016e6cf08c46ea1f0295bef9b6f4fdc2888deac1748cf3b4d02c0cd71f"
)
TOOLEMU_SHARED_STATE_AMENDMENT_SHA256 = (
    "5ab8d198167c16618b543883dcfa2d9888d06b8809347ed855faefc5a752326a"
)
TOOLEMU_PILOT_GATE_RESOLUTION_SHA256 = (
    "89df1e504b9ffbfcb1d219ca55bb89d984d70d0456d01b8207287ce71fc4fb59"
)
TOOLEMU_FAILURE_CLASSIFICATION_AMENDMENT_SHA256 = (
    "8c68410b16bae19b45beeae5fb5ba8dbaa1e88bf6f6f7acb80e4316b616a318d"
)
_ACTION_MARKER = re.compile(r"(?m)^[ \t]*Action(?:\s+\d+)?:")
_SIMULATOR_HEADER_NEWLINE = re.compile(
    r"(?m)^(Simulator Log Summary:|Observation:)[ \t]*\r?\n"
)


class ToolEmuBudgetExceeded(RuntimeError):
    """A frozen per-trajectory simulator budget was exhausted."""


def _load_bounded_simulator_amendment() -> dict[str, Any]:
    path = (
        Path(__file__).with_name("audits")
        / "v7"
        / "toolemu_bounded_simulator_amendment.json"
    )
    if sha256_file(path) != TOOLEMU_BOUNDED_SIMULATOR_AMENDMENT_SHA256:
        raise ValueError("ToolEmu bounded-simulator amendment hash drift")
    value = read_json(path)
    if value.get("protocol_sha256") != PROTOCOL_SHA256:
        raise ValueError("ToolEmu bounded amendment names a different protocol")
    if value.get(
        "text_model_correction_amendment_sha256"
    ) != TEXT_MODEL_CORRECTION_AMENDMENT_SHA256:
        raise ValueError("ToolEmu bounded amendment predates the Terra correction")
    if value.get(
        "toolemu_noncyber_scope_amendment_sha256"
    ) != TOOLEMU_NONCYBER_AMENDMENT_SHA256:
        raise ValueError("ToolEmu bounded amendment names a different task scope")
    config = value.get("configuration") or {}
    if (
        config.get("simulator_model") != TERRA_MODEL
        or int(config.get("maximum_iterations", -1)) != 5
        or int(config.get("maximum_agent_tokens_per_call", -1)) != 1024
        or int(config.get("maximum_simulator_tokens_per_call", -1)) != 1024
    ):
        raise ValueError("ToolEmu bounded amendment configuration drift")
    return value


def _load_format_budget_amendment() -> dict[str, Any]:
    path = (
        Path(__file__).with_name("audits")
        / "v7"
        / "toolemu_format_budget_amendment.json"
    )
    if sha256_file(path) != TOOLEMU_FORMAT_BUDGET_AMENDMENT_SHA256:
        raise ValueError("ToolEmu format-budget amendment hash drift")
    value = read_json(path)
    if value.get("protocol_sha256") != PROTOCOL_SHA256:
        raise ValueError("ToolEmu format-budget amendment names a different protocol")
    if value.get(
        "text_model_correction_amendment_sha256"
    ) != TEXT_MODEL_CORRECTION_AMENDMENT_SHA256:
        raise ValueError("ToolEmu format-budget amendment predates the Terra correction")
    if value.get(
        "toolemu_noncyber_scope_amendment_sha256"
    ) != TOOLEMU_NONCYBER_AMENDMENT_SHA256:
        raise ValueError("ToolEmu format-budget amendment names a different task scope")
    if value.get(
        "toolemu_bounded_simulator_amendment_sha256"
    ) != TOOLEMU_BOUNDED_SIMULATOR_AMENDMENT_SHA256:
        raise ValueError("ToolEmu format-budget amendment names a different parent")
    correction = value.get("correction") or {}
    if (
        int(correction.get("maximum_simulator_calls_per_trajectory", -1)) != 10
        or int(
            correction.get(
                "maximum_cumulative_simulator_input_tokens_per_trajectory", -1
            )
        )
        != 250000
        or int(
            correction.get("maximum_serialized_simulator_request_bytes_per_call", -1)
        )
        != 600000
    ):
        raise ValueError("ToolEmu format-budget amendment configuration drift")
    return value


def _load_shared_state_amendment() -> dict[str, Any]:
    path = (
        Path(__file__).with_name("audits")
        / "v7"
        / "toolemu_shared_state_amendment.json"
    )
    if sha256_file(path) != TOOLEMU_SHARED_STATE_AMENDMENT_SHA256:
        raise ValueError("ToolEmu shared-state amendment hash drift")
    value = read_json(path)
    if value.get("protocol_sha256") != PROTOCOL_SHA256:
        raise ValueError("ToolEmu shared-state amendment names a different protocol")
    if value.get(
        "text_model_correction_amendment_sha256"
    ) != TEXT_MODEL_CORRECTION_AMENDMENT_SHA256:
        raise ValueError("ToolEmu shared-state amendment predates the Terra correction")
    if value.get(
        "toolemu_noncyber_scope_amendment_sha256"
    ) != TOOLEMU_NONCYBER_AMENDMENT_SHA256:
        raise ValueError("ToolEmu shared-state amendment names a different task scope")
    if value.get(
        "toolemu_bounded_simulator_amendment_sha256"
    ) != TOOLEMU_BOUNDED_SIMULATOR_AMENDMENT_SHA256:
        raise ValueError("ToolEmu shared-state amendment names a different bounded parent")
    if value.get(
        "toolemu_format_budget_amendment_sha256"
    ) != TOOLEMU_FORMAT_BUDGET_AMENDMENT_SHA256:
        raise ValueError("ToolEmu shared-state amendment names a different format parent")
    return value


def _load_pilot_gate_resolution() -> dict[str, Any]:
    path = (
        Path(__file__).with_name("audits")
        / "v7"
        / "toolemu_pilot_gate_resolution.json"
    )
    if sha256_file(path) != TOOLEMU_PILOT_GATE_RESOLUTION_SHA256:
        raise ValueError("ToolEmu pilot-gate resolution hash drift")
    value = read_json(path)
    if value.get("protocol_sha256") != PROTOCOL_SHA256:
        raise ValueError("ToolEmu pilot-gate resolution names a different protocol")
    expected = {
        "clean_pilot_audit_sha256": "d90dcc0f3700e425f24eef67b33e455010c20b6cf2006f4a5dc2c98df52f0a28",
        "toolemu_noncyber_scope_amendment_sha256": TOOLEMU_NONCYBER_AMENDMENT_SHA256,
        "toolemu_bounded_simulator_amendment_sha256": TOOLEMU_BOUNDED_SIMULATOR_AMENDMENT_SHA256,
        "toolemu_format_budget_amendment_sha256": TOOLEMU_FORMAT_BUDGET_AMENDMENT_SHA256,
        "toolemu_shared_state_amendment_sha256": TOOLEMU_SHARED_STATE_AMENDMENT_SHA256,
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise ValueError(f"ToolEmu pilot-gate resolution has a different {key}")
    resolution = value.get("resolution") or {}
    if (
        int(resolution.get("canonical_target", -1)) != 50
        or resolution.get("replacement_batch")
        != "For each excluded initial task, take the next unused task from the same SHA-256-ranked non-Cyber population, beginning at permitted rank 50, and continue until 50 analyzable trajectories exist."
    ):
        raise ValueError("ToolEmu pilot-gate resolution configuration drift")
    return value


def _load_failure_classification_amendment() -> dict[str, Any]:
    path = (
        Path(__file__).with_name("audits")
        / "v7"
        / "toolemu_failure_classification_amendment.json"
    )
    if sha256_file(path) != TOOLEMU_FAILURE_CLASSIFICATION_AMENDMENT_SHA256:
        raise ValueError("ToolEmu failure-classification amendment hash drift")
    value = read_json(path)
    expected = {
        "protocol_sha256": PROTOCOL_SHA256,
        "text_model_correction_amendment_sha256": TEXT_MODEL_CORRECTION_AMENDMENT_SHA256,
        "toolemu_noncyber_scope_amendment_sha256": TOOLEMU_NONCYBER_AMENDMENT_SHA256,
        "toolemu_bounded_simulator_amendment_sha256": TOOLEMU_BOUNDED_SIMULATOR_AMENDMENT_SHA256,
        "toolemu_format_budget_amendment_sha256": TOOLEMU_FORMAT_BUDGET_AMENDMENT_SHA256,
        "toolemu_shared_state_amendment_sha256": TOOLEMU_SHARED_STATE_AMENDMENT_SHA256,
        "toolemu_pilot_gate_resolution_sha256": TOOLEMU_PILOT_GATE_RESOLUTION_SHA256,
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise ValueError(
                f"ToolEmu failure-classification amendment has a different {key}"
            )
    return value


def _replaceable_failure_kind(error: Exception) -> str | None:
    if isinstance(error, ToolEmuBudgetExceeded):
        return "bounded_budget_exceeded"
    rendered = f"{type(error).__name__}: {error}"
    if rendered == (
        "ValueError: ToolEmu trajectory has no eligible assistant-language token"
    ):
        return "no_eligible_assistant_token_after_minimum_context_rule"
    if rendered == "ValueError: captured activation contains nonfinite values":
        return "nonfinite_activation"
    return None


def _validate_replaceable_failure(failure: dict[str, Any]) -> None:
    kind = str(failure.get("failure_kind"))
    error = str(failure.get("error"))
    valid = (
        kind == "bounded_budget_exceeded"
        and error.startswith("ToolEmuBudgetExceeded: ")
    ) or (
        kind == "no_eligible_assistant_token_after_minimum_context_rule"
        and error
        == "ValueError: ToolEmu trajectory has no eligible assistant-language token"
    ) or (
        kind == "nonfinite_activation"
        and error == "ValueError: captured activation contains nonfinite values"
    )
    if not valid:
        raise ValueError("ToolEmu failure is not a frozen replaceable outcome")


def _persist_interruption(
    *,
    output: Path,
    selected: dict[str, Any],
    attempt_id: str,
    audit_rel: Path,
    error: Exception,
    interruption_kind: str,
) -> tuple[dict[str, Any], Path]:
    slug = _slug(selected)
    record = {
        "schema_version": 1,
        "protocol_sha256": PROTOCOL_SHA256,
        "text_model_correction_amendment_sha256": (
            TEXT_MODEL_CORRECTION_AMENDMENT_SHA256
        ),
        "toolemu_bounded_simulator_amendment_sha256": (
            TOOLEMU_BOUNDED_SIMULATOR_AMENDMENT_SHA256
        ),
        "toolemu_format_budget_amendment_sha256": (
            TOOLEMU_FORMAT_BUDGET_AMENDMENT_SHA256
        ),
        "toolemu_shared_state_amendment_sha256": (
            TOOLEMU_SHARED_STATE_AMENDMENT_SHA256
        ),
        "toolemu_pilot_gate_resolution_sha256": (
            TOOLEMU_PILOT_GATE_RESOLUTION_SHA256
        ),
        "toolemu_failure_classification_amendment_sha256": (
            TOOLEMU_FAILURE_CLASSIFICATION_AMENDMENT_SHA256
        ),
        "toolemu_noncyber_scope_amendment_sha256": (
            TOOLEMU_NONCYBER_AMENDMENT_SHA256
        ),
        "interruption_kind": interruption_kind,
        "task_id": selected["task_id"],
        "attempt_id": attempt_id,
        "audit_dir": str(audit_rel),
        "error": f"{type(error).__name__}: {error}",
        "task": selected,
    }
    relative = (
        Path("interruptions") / FAMILY_ID / slug / f"{attempt_id}.json"
    )
    path = output / relative
    atomic_json(path, record)
    return record, relative


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()


def _truncate_at_stop(text: str, stops: list[str] | None) -> str:
    positions = [text.find(stop) for stop in stops or [] if stop and stop in text]
    return text[: min(positions)] if positions else text


def _normalize_simulator_headers(text: str) -> tuple[str, list[str]]:
    """Map header-newline renderings to ToolEmu's literal-space delimiters."""
    normalized_headers: list[str] = []

    def replace(match: re.Match[str]) -> str:
        header = match.group(1)
        normalized_headers.append(header)
        return header + " "

    return _SIMULATOR_HEADER_NEWLINE.sub(replace, text), normalized_headers


def _simulator_budget_violation(
    *,
    completed_calls: int,
    cumulative_input_tokens: int,
    serialized_request_bytes: int,
    maximum_calls: int,
    maximum_cumulative_input_tokens: int,
    maximum_request_bytes: int,
    after_completed_call: bool = False,
) -> str | None:
    if after_completed_call:
        if cumulative_input_tokens >= maximum_cumulative_input_tokens:
            return (
                "cumulative simulator input-token ceiling reached: "
                f"{cumulative_input_tokens}>={maximum_cumulative_input_tokens}"
            )
        return None
    if completed_calls >= maximum_calls:
        return f"simulator-call ceiling reached: {completed_calls}>={maximum_calls}"
    if cumulative_input_tokens >= maximum_cumulative_input_tokens:
        return (
            "cumulative simulator input-token ceiling reached: "
            f"{cumulative_input_tokens}>={maximum_cumulative_input_tokens}"
        )
    if serialized_request_bytes > maximum_request_bytes:
        return (
            "serialized simulator request exceeds byte ceiling: "
            f"{serialized_request_bytes}>{maximum_request_bytes}"
        )
    return None


def _natural_language_character_end(text: str) -> int:
    """Exclude ToolEmu's structured Action/Action Input suffix."""
    match = _ACTION_MARKER.search(text)
    return match.start() if match is not None else len(text)


def _position_rank(seed: int, trajectory_id: str, position_key: int) -> bytes:
    return hashlib.sha256(
        f"{seed}\0position:{trajectory_id}\0{position_key}".encode()
    ).digest()


def _retain_position(
    retained: list[tuple[bytes, str, dict[str, Any]]],
    row: dict[str, Any],
    *,
    seed: int,
    trajectory_id: str,
    maximum: int = 4,
) -> None:
    position_key = int(row["position_key"])
    retained.append(
        (
            _position_rank(seed, trajectory_id, position_key),
            str(position_key),
            row,
        )
    )
    retained.sort(key=lambda value: (value[0], value[1]))
    del retained[maximum:]


def _chat_rows(messages: list[Any]) -> list[dict[str, str]]:
    rows = []
    for message in messages:
        message_type = getattr(message, "type", None)
        role = {
            "system": "system",
            "human": "user",
            "ai": "assistant",
        }.get(message_type)
        if role is None and message_type == "chat":
            role = str(getattr(message, "role", ""))
        content = getattr(message, "content", None)
        if role not in {"system", "user", "assistant"} or not isinstance(content, str):
            raise ValueError(f"unsupported ToolEmu chat message: {type(message).__name__}")
        rows.append({"role": role, "content": content})
    return rows


def _validate_task(task: dict[str, Any]) -> None:
    if task.get("protocol_sha256") != PROTOCOL_SHA256:
        raise ValueError("ToolEmu task has a stale protocol hash")
    if task.get("family_id") != FAMILY_ID:
        raise ValueError("ToolEmu task has the wrong family")
    payload = dict(task)
    observed = str(payload.pop("task_sha256", ""))
    expected = hashlib.sha256(_canonical(payload)).hexdigest()
    if observed != expected:
        raise ValueError(f"invalid task hash for {task.get('task_id')}")


def _select_tasks(
    rows: list[dict[str, Any]],
    *,
    num_shards: int,
    shard_index: int,
    limit: int | None,
    task_id: str | None,
) -> list[dict[str, Any]]:
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("invalid shard configuration")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    if task_id is not None and num_shards != 1:
        raise ValueError("--task-id requires --num-shards 1")
    for task in rows:
        _validate_task(task)
    if len({str(task["task_id"]) for task in rows}) != len(rows):
        raise ValueError("duplicate ToolEmu task IDs")
    if task_id is not None:
        rows = [task for task in rows if str(task["task_id"]) == task_id]
        if not rows:
            raise ValueError(f"requested ToolEmu task ID is absent: {task_id}")
    else:
        rows = [task for index, task in enumerate(rows) if index % num_shards == shard_index]
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise ValueError("no ToolEmu tasks assigned")
    return rows


def _serialize_outputs(outputs: dict[str, Any]) -> dict[str, Any]:
    steps = []
    for action, observation in outputs.get("intermediate_steps") or []:
        if hasattr(action, "tool"):
            action_row = {
                "kind": "action",
                "tool": str(action.tool),
                "tool_input": action.tool_input,
                "log": str(action.log),
            }
        else:
            action_row = {
                "kind": "finish",
                "return_values": dict(getattr(action, "return_values", {})),
                "log": str(getattr(action, "log", "")),
            }
        if hasattr(observation, "_asdict"):
            observation_row: Any = dict(observation._asdict())
        else:
            observation_row = observation
        steps.append({"agent": action_row, "observation": observation_row})
    return {"output": outputs.get("output"), "intermediate_steps": steps}


def _make_qwen_chat_class(
    *,
    torch: Any,
    ChatOpenAI: Any,
    ChatGeneration: Any,
    ChatResult: Any,
    AIMessage: Any,
    PrivateAttr: Any,
    model: Any,
    tokenizer: Any,
    layer: Any,
    maximum_new_tokens: int,
):
    class LocalQwenChat(ChatOpenAI):
        _trajectory_id = PrivateAttr()
        _seed = PrivateAttr()
        _shared_state = PrivateAttr(default_factory=dict)

        def __init__(self, *, trajectory_id: str, seed: int):
            super().__init__(
                model_name="gpt-4",
                openai_api_key="not-used-by-local-qwen-adapter",
                temperature=0.0,
                max_tokens=maximum_new_tokens,
                streaming=False,
            )
            self._trajectory_id = trajectory_id
            self._seed = seed
            self._shared_state = {
                "trace": [],
                "retained": [],
                "eligible_population": 0,
                "call_index": 0,
            }

        @property
        def trace(self) -> list[dict[str, Any]]:
            return self._shared_state["trace"]

        @property
        def eligible_population(self) -> int:
            return int(self._shared_state["eligible_population"])

        def selected_activations(self) -> list[dict[str, Any]]:
            return [
                row
                for _, _, row in sorted(
                    self._shared_state["retained"],
                    key=lambda value: value[2]["position_key"],
                )
            ]

        def _generate(
            self,
            messages: list[Any],
            stop: list[str] | None = None,
            run_manager: Any = None,
            **kwargs: Any,
        ) -> Any:
            del run_manager, kwargs
            converted = _chat_rows(messages)
            encoded = tokenizer.apply_chat_template(
                converted, tokenize=True, add_generation_prompt=True
            )
            prompt_ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
            if prompt_ids and isinstance(prompt_ids[0], list):
                prompt_ids = prompt_ids[0]
            prompt_ids = [int(value) for value in prompt_ids]
            max_context = int(getattr(model.config, "max_position_embeddings", 32768))
            if len(prompt_ids) + maximum_new_tokens > max_context:
                raise ValueError(
                    f"ToolEmu prompt exceeds target context: {len(prompt_ids)}+"
                    f"{maximum_new_tokens}>{max_context}"
                )
            device = model.get_input_embeddings().weight.device
            input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            eos_id = tokenizer.eos_token_id
            if eos_id is None:
                raise ValueError("target tokenizer has no EOS token")
            generate_kwargs = {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
                "do_sample": False,
                "max_new_tokens": maximum_new_tokens,
                "pad_token_id": eos_id,
                "eos_token_id": eos_id,
                "use_cache": True,
            }
            if stop:
                generate_kwargs.update({"stop_strings": stop, "tokenizer": tokenizer})
            with torch.inference_mode():
                generated = model.generate(**generate_kwargs)[0]
            continuation_ids = [int(value) for value in generated[len(prompt_ids) :].tolist()]
            raw_response = tokenizer.decode(
                continuation_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            response = _truncate_at_stop(raw_response, stop)
            natural_end = _natural_language_character_end(response)
            natural_text = response[:natural_end]
            natural_ids = tokenizer.encode(natural_text, add_special_tokens=False)
            prefix_count = 0
            for left, right in zip(natural_ids, continuation_ids):
                if int(left) != int(right):
                    break
                prefix_count += 1
            if prefix_count:
                with torch.inference_mode():
                    hidden = _capture_layer(
                        torch=torch,
                        model=model,
                        layer=layer,
                        input_ids=generated.view(1, -1),
                    )
                special_ids = {int(value) for value in tokenizer.all_special_ids}
                for response_index, token_id in enumerate(continuation_ids[:prefix_count]):
                    absolute = len(prompt_ids) + response_index
                    if token_id in special_ids or absolute < 50:
                        continue
                    call_index = int(self._shared_state["call_index"])
                    position_key = call_index * 1_000_000 + response_index
                    row = {
                        "position_key": position_key,
                        "agent_call_index": call_index,
                        "assistant_response_token_index": response_index,
                        "token_position": absolute,
                        "token_id": token_id,
                        "read_token": tokenizer.decode(
                            [token_id], skip_special_tokens=False
                        ),
                        "causal_context": tokenizer.decode(
                            generated[: absolute + 1].tolist(),
                            skip_special_tokens=False,
                        ),
                        "vector": hidden[absolute].numpy().astype(np.float32, copy=True),
                    }
                    self._shared_state["eligible_population"] += 1
                    _retain_position(
                        self._shared_state["retained"],
                        row,
                        seed=self._seed,
                        trajectory_id=self._trajectory_id,
                    )
            call_index = int(self._shared_state["call_index"])
            self._shared_state["trace"].append(
                {
                    "call_index": call_index,
                    "messages": converted,
                    "prompt_tokens": len(prompt_ids),
                    "prompt_ids_sha256": hashlib.sha256(
                        np.asarray(prompt_ids, dtype=np.int64).tobytes()
                    ).hexdigest(),
                    "continuation_token_ids": continuation_ids,
                    "raw_response": raw_response,
                    "returned_response": response,
                    "stop_sequences": stop,
                    "natural_language_character_end": natural_end,
                    "eligible_prefix_tokens": prefix_count,
                    "hit_maximum_new_tokens": len(continuation_ids)
                    >= maximum_new_tokens,
                }
            )
            self._shared_state["call_index"] += 1
            return ChatResult(
                generations=[
                    ChatGeneration(
                        message=AIMessage(content=response),
                        generation_info={
                            "finish_reason": "length"
                            if len(continuation_ids) >= maximum_new_tokens
                            else "stop"
                        },
                    )
                ],
                llm_output={
                    "model_name": "Qwen/Qwen2.5-7B-Instruct",
                    "token_usage": {
                        "prompt_tokens": len(prompt_ids),
                        "completion_tokens": len(continuation_ids),
                        "total_tokens": len(prompt_ids) + len(continuation_ids),
                    },
                },
            )

    return LocalQwenChat


def _make_terra_chat_class(
    *,
    ChatOpenAI: Any,
    ChatGeneration: Any,
    ChatResult: Any,
    AIMessage: Any,
    PrivateAttr: Any,
    maximum_output_tokens: int,
    maximum_calls: int,
    maximum_cumulative_input_tokens: int,
    maximum_request_bytes: int,
):
    class TerraToolEmuChat(ChatOpenAI):
        _terra = PrivateAttr()
        _audit_dir = PrivateAttr()
        _shared_state = PrivateAttr(default_factory=dict)

        def __init__(self, *, audit_dir: Path):
            super().__init__(
                model_name="gpt-4",
                openai_api_key="not-used-by-responses-adapter",
                temperature=0.0,
                max_tokens=maximum_output_tokens,
                streaming=False,
            )
            self._terra = _TerraClient(audit_dir)
            self._audit_dir = audit_dir
            self._shared_state = {
                "trace": [],
                "completed_calls": 0,
                "cumulative_input_tokens": 0,
            }

        @property
        def trace(self) -> list[dict[str, Any]]:
            return self._shared_state["trace"]

        @property
        def budget_state(self) -> dict[str, int]:
            return {
                "completed_calls": int(self._shared_state["completed_calls"]),
                "cumulative_input_tokens": int(
                    self._shared_state["cumulative_input_tokens"]
                ),
                "maximum_calls": maximum_calls,
                "maximum_cumulative_input_tokens": (
                    maximum_cumulative_input_tokens
                ),
                "maximum_request_bytes": maximum_request_bytes,
            }

        def close(self) -> None:
            self._terra.close()

        def _generate(
            self,
            messages: list[Any],
            stop: list[str] | None = None,
            run_manager: Any = None,
            **kwargs: Any,
        ) -> Any:
            del run_manager, kwargs
            converted = _chat_rows(messages)
            request_payload = {
                "model": TERRA_MODEL,
                "input": converted,
                "reasoning": {"effort": "low"},
                "max_output_tokens": int(maximum_output_tokens),
                "store": False,
            }
            request_bytes = len(_canonical(request_payload))
            violation = _simulator_budget_violation(
                completed_calls=int(self._shared_state["completed_calls"]),
                cumulative_input_tokens=int(
                    self._shared_state["cumulative_input_tokens"]
                ),
                serialized_request_bytes=request_bytes,
                maximum_calls=maximum_calls,
                maximum_cumulative_input_tokens=maximum_cumulative_input_tokens,
                maximum_request_bytes=maximum_request_bytes,
            )
            if violation is not None:
                raise ToolEmuBudgetExceeded(violation)
            raw_text, audit = self._terra.generate(
                converted, max_output_tokens=maximum_output_tokens
            )
            responses_usage = audit.get("usage") or {}
            prompt_tokens = int(responses_usage.get("input_tokens") or 0)
            completion_tokens = int(responses_usage.get("output_tokens") or 0)
            self._shared_state["completed_calls"] += 1
            self._shared_state["cumulative_input_tokens"] += prompt_tokens
            text, normalized_headers = _normalize_simulator_headers(raw_text)
            text = _truncate_at_stop(text, stop)
            budget_state = self.budget_state
            audit.update(
                {
                    "toolemu_bounded_simulator_amendment_sha256": (
                        TOOLEMU_BOUNDED_SIMULATOR_AMENDMENT_SHA256
                    ),
                    "toolemu_format_budget_amendment_sha256": (
                        TOOLEMU_FORMAT_BUDGET_AMENDMENT_SHA256
                    ),
                    "toolemu_shared_state_amendment_sha256": (
                        TOOLEMU_SHARED_STATE_AMENDMENT_SHA256
                    ),
                    "toolemu_pilot_gate_resolution_sha256": (
                        TOOLEMU_PILOT_GATE_RESOLUTION_SHA256
                    ),
                    "toolemu_failure_classification_amendment_sha256": (
                        TOOLEMU_FAILURE_CLASSIFICATION_AMENDMENT_SHA256
                    ),
                    "toolemu_adapter": {
                        "normalized_headers": normalized_headers,
                        "returned_text_sha256": hashlib.sha256(
                            text.encode("utf-8")
                        ).hexdigest(),
                    },
                    "serialized_request_bytes": request_bytes,
                    "trajectory_budget_state_after_call": budget_state,
                }
            )
            atomic_json(
                self._audit_dir / f"call-{int(audit['call_index']):02d}.json",
                audit,
            )
            violation = _simulator_budget_violation(
                completed_calls=int(self._shared_state["completed_calls"]),
                cumulative_input_tokens=int(
                    self._shared_state["cumulative_input_tokens"]
                ),
                serialized_request_bytes=request_bytes,
                maximum_calls=maximum_calls,
                maximum_cumulative_input_tokens=maximum_cumulative_input_tokens,
                maximum_request_bytes=maximum_request_bytes,
                after_completed_call=True,
            )
            if violation is not None:
                raise ToolEmuBudgetExceeded(violation)
            self._shared_state["trace"].append(
                {
                    "call_index": len(self._shared_state["trace"]),
                    "request_sha256": audit["request_sha256"],
                    "response_sha256": audit["response_sha256"],
                    "response_id": audit.get("response_id"),
                    "resolved_model": audit.get("resolved_model"),
                    "stop_sequences": stop,
                    "raw_output_text": raw_text,
                    "returned_output_text": text,
                    "normalized_headers": normalized_headers,
                    "serialized_request_bytes": request_bytes,
                    "usage": responses_usage,
                    "trajectory_budget_state_after_call": budget_state,
                }
            )
            return ChatResult(
                generations=[
                    ChatGeneration(
                        message=AIMessage(content=text),
                        generation_info={"finish_reason": "stop"},
                    )
                ],
                llm_output={
                    "model_name": TERRA_MODEL,
                    "token_usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": int(
                            responses_usage.get("total_tokens")
                            or prompt_tokens + completion_tokens
                        ),
                    },
                },
            )

    return TerraToolEmuChat


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--toolemu-source", required=True)
    parser.add_argument("--promptcoder-archive", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--task-id")
    parser.add_argument("--maximum-iterations", type=int, default=5)
    parser.add_argument("--maximum-agent-tokens", type=int, default=1024)
    parser.add_argument("--maximum-simulator-tokens", type=int, default=1024)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    if min(
        args.maximum_iterations,
        args.maximum_agent_tokens,
        args.maximum_simulator_tokens,
    ) < 1:
        raise ValueError("iteration and token limits must be positive")

    protocol = load_protocol()
    bounded_amendment = _load_bounded_simulator_amendment()
    format_budget_amendment = _load_format_budget_amendment()
    shared_state_amendment = _load_shared_state_amendment()
    _load_pilot_gate_resolution()
    _load_failure_classification_amendment()
    bounded_config = bounded_amendment["configuration"]
    budget_config = format_budget_amendment["correction"]
    if (
        args.maximum_iterations != int(bounded_config["maximum_iterations"])
        or args.maximum_agent_tokens
        != int(bounded_config["maximum_agent_tokens_per_call"])
        or args.maximum_simulator_tokens
        != int(bounded_config["maximum_simulator_tokens_per_call"])
    ):
        raise ValueError("runner arguments differ from the bounded ToolEmu amendment")
    if args.pilot and (
        args.limit is not None
        or args.task_id is not None
        or args.num_shards != 1
        or args.shard_index != 0
    ):
        raise ValueError("--pilot requires the unsharded task artifact without --limit or --task-id")
    tasks = _select_tasks(
        list(read_jsonl(args.tasks)),
        num_shards=args.num_shards,
        shard_index=args.shard_index,
        limit=3 if args.pilot else args.limit,
        task_id=args.task_id,
    )
    if args.pilot and [str(task["task_id"]) for task in tasks] != list(
        bounded_amendment["pilot"]["task_ids"]
    ):
        raise ValueError("pilot task IDs differ from the bounded ToolEmu amendment")
    if args.pilot and [str(task["task_id"]) for task in tasks] != list(
        format_budget_amendment["corrected_pilot"]["task_ids"]
    ):
        raise ValueError("pilot task IDs differ from the format-budget amendment")
    if args.pilot and [str(task["task_id"]) for task in tasks] != list(
        shared_state_amendment["corrected_pilot"]["task_ids"]
    ):
        raise ValueError("pilot task IDs differ from the shared-state amendment")
    source = Path(args.toolemu_source)
    cases_path = source / "assets/all_cases.json"
    toolkits_path = source / "assets/all_toolkits.json"
    virtual_tools_path = source / "toolemu/tools/virtual_tools.py"
    raw_cases = cases_path.read_bytes()
    cases = json.loads(raw_cases)
    by_name = {str(case["name"]): case for case in cases}
    for task in tasks:
        if (
            task.get("toolemu_noncyber_scope_amendment_sha256")
            != TOOLEMU_NONCYBER_AMENDMENT_SHA256
        ):
            raise ValueError(
                f"ToolEmu task {task.get('task_id')} lacks the active non-Cyber amendment"
            )
        source_meta = task["source"]
        if source_meta["source_cases_sha256"] != hashlib.sha256(raw_cases).hexdigest():
            raise ValueError("live ToolEmu cases differ from the selected source")
        case = by_name.get(str(task["task_id"]))
        if case is None or hashlib.sha256(_canonical(case)).hexdigest() != source_meta["source_case_sha256"]:
            raise ValueError(f"live ToolEmu case differs for {task['task_id']}")

    output = Path(args.out)
    meta_path = output / f"{FAMILY_ID}-shard-{args.shard_index:02d}-meta.json"
    startup_meta = {
        "schema_version": 1,
        "protocol_sha256": PROTOCOL_SHA256,
        "text_model_correction_amendment_sha256": (
            TEXT_MODEL_CORRECTION_AMENDMENT_SHA256
        ),
        "toolemu_bounded_simulator_amendment_sha256": (
            TOOLEMU_BOUNDED_SIMULATOR_AMENDMENT_SHA256
        ),
        "toolemu_format_budget_amendment_sha256": (
            TOOLEMU_FORMAT_BUDGET_AMENDMENT_SHA256
        ),
        "toolemu_shared_state_amendment_sha256": (
            TOOLEMU_SHARED_STATE_AMENDMENT_SHA256
        ),
        "toolemu_pilot_gate_resolution_sha256": (
            TOOLEMU_PILOT_GATE_RESOLUTION_SHA256
        ),
        "toolemu_failure_classification_amendment_sha256": (
            TOOLEMU_FAILURE_CLASSIFICATION_AMENDMENT_SHA256
        ),
        "toolemu_noncyber_scope_amendment_sha256": (
            TOOLEMU_NONCYBER_AMENDMENT_SHA256
        ),
        "family_id": FAMILY_ID,
        "assigned_tasks": len(tasks),
        "assigned_task_ids": [str(task["task_id"]) for task in tasks],
        "task_artifact_sha256": sha256_file(args.tasks),
        "toolemu_cases_sha256": sha256_file(cases_path),
        "toolemu_toolkits_sha256": sha256_file(toolkits_path),
        "toolemu_virtual_tools_sha256": sha256_file(virtual_tools_path),
        "promptcoder_archive_sha256": sha256_file(args.promptcoder_archive),
        "toolemu_source": str(source),
        "pilot": bool(args.pilot),
        "simulator_budget": {
            "maximum_calls_per_trajectory": int(
                budget_config["maximum_simulator_calls_per_trajectory"]
            ),
            "maximum_cumulative_input_tokens_per_trajectory": int(
                budget_config[
                    "maximum_cumulative_simulator_input_tokens_per_trajectory"
                ]
            ),
            "maximum_serialized_request_bytes_per_call": int(
                budget_config[
                    "maximum_serialized_simulator_request_bytes_per_call"
                ]
            ),
        },
    }
    if not os.environ.get(TERRA_KEY_NAME):
        output.mkdir(parents=True, exist_ok=True)
        atomic_json(
            meta_path,
            {
                "status": "blocked_configuration",
                **startup_meta,
                "error": f"{TERRA_KEY_NAME} is required; fallback keys are forbidden",
            },
        )
        return 2
    try:
        endpoint_preflight = _preflight_openai_endpoint()
    except TerraInfrastructureError as error:
        output.mkdir(parents=True, exist_ok=True)
        atomic_json(
            meta_path,
            {
                "status": "blocked_connectivity",
                **startup_meta,
                "error": f"{type(error).__name__}: {error}",
            },
        )
        print(f"[toolemu preflight] {type(error).__name__}: {error}", flush=True)
        return 2

    import torch
    from huggingface_hub import snapshot_download
    from langchain.schema import AIMessage, ChatGeneration, ChatResult
    from pydantic import PrivateAttr
    from toolemu.agent_executor_builder import build_agent_executor
    from toolemu.utils import case_to_input_dict, filter_keys
    from toolemu.utils.llm import ChatOpenAI
    from transformers import AutoModelForCausalLM, AutoTokenizer

    target = protocol["models"]["target"]
    checkpoint = snapshot_download(
        repo_id=target["repo_id"], revision=target["revision"], local_files_only=True
    )
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    if not torch.cuda.is_available():
        raise RuntimeError("ToolEmu trajectory generation requires CUDA")
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to("cuda:0").eval()
    layers, layer_path = _decoder_layers(torch, model)
    layer = layers[int(target["layer"])]
    LocalQwenChat = _make_qwen_chat_class(
        torch=torch,
        ChatOpenAI=ChatOpenAI,
        ChatGeneration=ChatGeneration,
        ChatResult=ChatResult,
        AIMessage=AIMessage,
        PrivateAttr=PrivateAttr,
        model=model,
        tokenizer=tokenizer,
        layer=layer,
        maximum_new_tokens=args.maximum_agent_tokens,
    )
    TerraToolEmuChat = _make_terra_chat_class(
        ChatOpenAI=ChatOpenAI,
        ChatGeneration=ChatGeneration,
        ChatResult=ChatResult,
        AIMessage=AIMessage,
        PrivateAttr=PrivateAttr,
        maximum_output_tokens=args.maximum_simulator_tokens,
        maximum_calls=int(
            budget_config["maximum_simulator_calls_per_trajectory"]
        ),
        maximum_cumulative_input_tokens=int(
            budget_config[
                "maximum_cumulative_simulator_input_tokens_per_trajectory"
            ]
        ),
        maximum_request_bytes=int(
            budget_config["maximum_serialized_simulator_request_bytes_per_call"]
        ),
    )

    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    base_meta = {
        **startup_meta,
        "target": target,
        "layer_path": layer_path,
        "openai_endpoint_preflight": endpoint_preflight,
        "toolemu": {
            "agent_type": "naive",
            "simulator_type": "adv_thought",
            "maximum_iterations": args.maximum_iterations,
            "simulator_model": TERRA_MODEL,
            "simulator_reasoning_effort": "low",
            "simulator_store": False,
        },
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
    }
    atomic_json(meta_path, {"status": "running", **base_meta})
    written = skipped = 0
    failures = []
    for position, selected in enumerate(tasks, start=1):
        slug = _slug(selected)
        trajectory_path = output / "trajectories" / FAMILY_ID / f"{slug}.json"
        failure_path = output / "failures" / FAMILY_ID / f"{slug}.json"
        if trajectory_path.exists():
            existing = read_json(trajectory_path)
            if existing.get("protocol_sha256") != PROTOCOL_SHA256:
                raise ValueError(f"stale trajectory at {trajectory_path}")
            if existing.get(
                "text_model_correction_amendment_sha256"
            ) != TEXT_MODEL_CORRECTION_AMENDMENT_SHA256:
                raise ValueError(f"pre-Terra ToolEmu trajectory at {trajectory_path}")
            if existing.get(
                "toolemu_bounded_simulator_amendment_sha256"
            ) != TOOLEMU_BOUNDED_SIMULATOR_AMENDMENT_SHA256:
                raise ValueError(f"pre-bounded ToolEmu trajectory at {trajectory_path}")
            if existing.get(
                "toolemu_format_budget_amendment_sha256"
            ) != TOOLEMU_FORMAT_BUDGET_AMENDMENT_SHA256:
                raise ValueError(
                    f"pre-format-budget ToolEmu trajectory at {trajectory_path}"
                )
            if existing.get(
                "toolemu_shared_state_amendment_sha256"
            ) != TOOLEMU_SHARED_STATE_AMENDMENT_SHA256:
                raise ValueError(
                    f"pre-shared-state ToolEmu trajectory at {trajectory_path}"
                )
            if existing.get(
                "toolemu_pilot_gate_resolution_sha256"
            ) != TOOLEMU_PILOT_GATE_RESOLUTION_SHA256:
                raise ValueError(
                    f"pre-pilot-resolution ToolEmu trajectory at {trajectory_path}"
                )
            if existing.get(
                "toolemu_failure_classification_amendment_sha256"
            ) != TOOLEMU_FAILURE_CLASSIFICATION_AMENDMENT_SHA256:
                raise ValueError(
                    f"pre-failure-classification ToolEmu trajectory at {trajectory_path}"
                )
            if (
                existing.get("toolemu_noncyber_scope_amendment_sha256")
                != TOOLEMU_NONCYBER_AMENDMENT_SHA256
            ):
                raise ValueError(f"pre-amendment ToolEmu trajectory at {trajectory_path}")
            for activation in existing.get("activations") or []:
                if not (output / activation["vector_file"]).is_file():
                    raise ValueError(f"trajectory has a missing activation: {trajectory_path}")
            skipped += 1
            print(
                f"[toolemu {position}/{len(tasks)} shard={args.shard_index}] "
                f"written={written} skipped={skipped} failures={len(failures)}",
                flush=True,
            )
            continue
        if failure_path.exists():
            failure = read_json(failure_path)
            if failure.get(
                "toolemu_bounded_simulator_amendment_sha256"
            ) != TOOLEMU_BOUNDED_SIMULATOR_AMENDMENT_SHA256:
                raise ValueError(f"pre-bounded ToolEmu failure at {failure_path}")
            if failure.get(
                "toolemu_format_budget_amendment_sha256"
            ) != TOOLEMU_FORMAT_BUDGET_AMENDMENT_SHA256:
                raise ValueError(
                    f"pre-format-budget ToolEmu failure at {failure_path}"
                )
            if failure.get(
                "toolemu_shared_state_amendment_sha256"
            ) != TOOLEMU_SHARED_STATE_AMENDMENT_SHA256:
                raise ValueError(
                    f"pre-shared-state ToolEmu failure at {failure_path}"
                )
            if failure.get(
                "toolemu_pilot_gate_resolution_sha256"
            ) != TOOLEMU_PILOT_GATE_RESOLUTION_SHA256:
                raise ValueError(
                    f"pre-pilot-resolution ToolEmu failure at {failure_path}"
                )
            if failure.get(
                "toolemu_failure_classification_amendment_sha256"
            ) != TOOLEMU_FAILURE_CLASSIFICATION_AMENDMENT_SHA256:
                raise ValueError(
                    f"pre-failure-classification ToolEmu failure at {failure_path}"
                )
            _validate_replaceable_failure(failure)
            failures.append(
                {
                    "task_id": selected["task_id"],
                    "error": failure.get("error"),
                    "preserved_existing": True,
                }
            )
            continue
        attempt_id = f"{time.time_ns()}-{os.getpid()}"
        audit_rel = (
            Path("tools")
            / FAMILY_ID
            / slug
            / "attempts"
            / attempt_id
            / "terra-emulator"
        )
        terra = None
        try:
            trajectory_id = _task_key(selected)
            case = by_name[str(selected["task_id"])]
            agent = LocalQwenChat(
                trajectory_id=trajectory_id, seed=int(protocol["seed"])
            )
            terra = TerraToolEmuChat(audit_dir=output / audit_rel)
            executor = build_agent_executor(
                list(case["Toolkits"]),
                agent,
                terra,
                agent_type="naive",
                simulator_type="adv_thought",
                verbose=False,
                return_intermediate_steps=True,
                max_iterations=args.maximum_iterations,
            )
            inputs = filter_keys(case_to_input_dict(case), executor.input_keys)
            raw_outputs = executor(inputs)
            serialized_outputs = _serialize_outputs(raw_outputs)
            selected_rows = agent.selected_activations()
            if not selected_rows:
                raise ValueError("ToolEmu trajectory has no eligible assistant-language token")
            if agent.eligible_population < len(selected_rows):
                raise ValueError(
                    "shared eligible-position counter is smaller than the retained sample"
                )
            simulator_budget = terra.budget_state
            if simulator_budget["completed_calls"] != len(terra.trace):
                raise ValueError("shared simulator-call counter differs from the trace")
            traced_input_tokens = sum(
                int((call.get("usage") or {}).get("input_tokens") or 0)
                for call in terra.trace
            )
            if simulator_budget["cumulative_input_tokens"] != traced_input_tokens:
                raise ValueError(
                    "shared cumulative simulator input tokens differ from the trace"
                )
            call_audits = sorted((output / audit_rel).glob("call-*.json"))
            if len(call_audits) != simulator_budget["completed_calls"]:
                raise ValueError("archived simulator-call count differs from shared state")
            for call_audit_path in call_audits:
                call_audit = read_json(call_audit_path)
                if (
                    call_audit.get("toolemu_shared_state_amendment_sha256")
                    != TOOLEMU_SHARED_STATE_AMENDMENT_SHA256
                ):
                    raise ValueError(
                        f"simulator call lacks the shared-state amendment: {call_audit_path}"
                    )
                if (
                    call_audit.get("toolemu_pilot_gate_resolution_sha256")
                    != TOOLEMU_PILOT_GATE_RESOLUTION_SHA256
                ):
                    raise ValueError(
                        f"simulator call lacks the pilot-gate resolution: {call_audit_path}"
                    )
                if (
                    call_audit.get(
                        "toolemu_failure_classification_amendment_sha256"
                    )
                    != TOOLEMU_FAILURE_CLASSIFICATION_AMENDMENT_SHA256
                ):
                    raise ValueError(
                        "simulator call lacks the failure-classification amendment: "
                        f"{call_audit_path}"
                    )
            activations = []
            for ordinal, row in enumerate(selected_rows):
                vector_file = Path("activations") / FAMILY_ID / f"{slug}-{ordinal}.npy"
                vector = row.pop("vector")
                _atomic_npy(output / vector_file, vector)
                activations.append(
                    {
                        **row,
                        "vector_file": str(vector_file),
                        "vector_sha256": sha256_file(output / vector_file),
                    }
                )
            trajectory = {
                "schema_version": 1,
                "protocol_sha256": PROTOCOL_SHA256,
                "text_model_correction_amendment_sha256": (
                    TEXT_MODEL_CORRECTION_AMENDMENT_SHA256
                ),
                "toolemu_bounded_simulator_amendment_sha256": (
                    TOOLEMU_BOUNDED_SIMULATOR_AMENDMENT_SHA256
                ),
                "toolemu_format_budget_amendment_sha256": (
                    TOOLEMU_FORMAT_BUDGET_AMENDMENT_SHA256
                ),
                "toolemu_shared_state_amendment_sha256": (
                    TOOLEMU_SHARED_STATE_AMENDMENT_SHA256
                ),
                "toolemu_pilot_gate_resolution_sha256": (
                    TOOLEMU_PILOT_GATE_RESOLUTION_SHA256
                ),
                "toolemu_failure_classification_amendment_sha256": (
                    TOOLEMU_FAILURE_CLASSIFICATION_AMENDMENT_SHA256
                ),
                "toolemu_noncyber_scope_amendment_sha256": (
                    TOOLEMU_NONCYBER_AMENDMENT_SHA256
                ),
                "replication_label": selected["replication_label"],
                "family_id": FAMILY_ID,
                "trajectory_id": trajectory_id,
                "task": selected,
                "case": case,
                "target_model": target,
                "generation": {
                    "do_sample": False,
                    "temperature": None,
                    "maximum_new_tokens_per_agent_call": args.maximum_agent_tokens,
                    "maximum_iterations": args.maximum_iterations,
                },
                "toolemu": {
                    "agent_type": "naive",
                    "simulator_type": "adv_thought",
                    "simulator_model": TERRA_MODEL,
                    "simulator_api": "responses",
                    "simulator_reasoning_effort": "low",
                    "simulator_audit_dir": str(audit_rel),
                    "simulator_budget": simulator_budget,
                    "simulator_call_audit_count": len(call_audits),
                    "inputs": inputs,
                    "outputs": serialized_outputs,
                },
                "agent_calls": agent.trace,
                "simulator_calls": terra.trace,
                "eligible_position_population": agent.eligible_population,
                "activations": activations,
                "status": "complete",
            }
            atomic_json(trajectory_path, trajectory)
            written += 1
        except TerraInfrastructureError as error:
            if terra is not None:
                terra.close()
            interruption, interruption_rel = _persist_interruption(
                output=output,
                selected=selected,
                attempt_id=attempt_id,
                audit_rel=audit_rel,
                error=error,
                interruption_kind="infrastructure",
            )
            interruption["artifact"] = str(interruption_rel)
            interruption["artifact_sha256"] = sha256_file(output / interruption_rel)
            atomic_json(
                meta_path,
                {
                    "status": "interrupted_infrastructure",
                    **base_meta,
                    "written": written,
                    "skipped": skipped,
                    "failures": failures,
                    "infrastructure_interruption": interruption,
                    "elapsed_seconds": round(time.time() - started, 3),
                },
            )
            print(
                f"[toolemu {position}/{len(tasks)} shard={args.shard_index}] "
                f"interrupted_infrastructure task={selected['task_id']}",
                flush=True,
            )
            return 2
        except Exception as error:
            if terra is not None:
                terra.close()
            failure_kind = _replaceable_failure_kind(error)
            if failure_kind is None:
                interruption, interruption_rel = _persist_interruption(
                    output=output,
                    selected=selected,
                    attempt_id=attempt_id,
                    audit_rel=audit_rel,
                    error=error,
                    interruption_kind="unclassified_failure",
                )
                interruption["artifact"] = str(interruption_rel)
                interruption["artifact_sha256"] = sha256_file(
                    output / interruption_rel
                )
                atomic_json(
                    meta_path,
                    {
                        "status": "interrupted_unclassified_failure",
                        **base_meta,
                        "written": written,
                        "skipped": skipped,
                        "failures": failures,
                        "unclassified_interruption": interruption,
                        "elapsed_seconds": round(time.time() - started, 3),
                    },
                )
                print(
                    f"[toolemu {position}/{len(tasks)} shard={args.shard_index}] "
                    f"interrupted_unclassified_failure task={selected['task_id']}",
                    flush=True,
                )
                return 2
            failure = {
                "schema_version": 1,
                "protocol_sha256": PROTOCOL_SHA256,
                "text_model_correction_amendment_sha256": (
                    TEXT_MODEL_CORRECTION_AMENDMENT_SHA256
                ),
                "toolemu_bounded_simulator_amendment_sha256": (
                    TOOLEMU_BOUNDED_SIMULATOR_AMENDMENT_SHA256
                ),
                "toolemu_format_budget_amendment_sha256": (
                    TOOLEMU_FORMAT_BUDGET_AMENDMENT_SHA256
                ),
                "toolemu_shared_state_amendment_sha256": (
                    TOOLEMU_SHARED_STATE_AMENDMENT_SHA256
                ),
                "toolemu_pilot_gate_resolution_sha256": (
                    TOOLEMU_PILOT_GATE_RESOLUTION_SHA256
                ),
                "toolemu_failure_classification_amendment_sha256": (
                    TOOLEMU_FAILURE_CLASSIFICATION_AMENDMENT_SHA256
                ),
                "toolemu_noncyber_scope_amendment_sha256": (
                    TOOLEMU_NONCYBER_AMENDMENT_SHA256
                ),
                "family_id": FAMILY_ID,
                "task_id": selected["task_id"],
                "attempt_id": attempt_id,
                "audit_dir": str(audit_rel),
                "simulator_budget": terra.budget_state if terra is not None else None,
                "failure_kind": failure_kind,
                "error": f"{type(error).__name__}: {error}",
                "task": selected,
            }
            atomic_json(failure_path, failure)
            failures.append({"task_id": selected["task_id"], "error": failure["error"]})
        else:
            assert terra is not None
            terra.close()
        print(
            f"[toolemu {position}/{len(tasks)} shard={args.shard_index}] "
            f"written={written} skipped={skipped} failures={len(failures)}",
            flush=True,
        )

    atomic_json(
        meta_path,
        {
            "status": "done" if not failures else "failed",
            **base_meta,
            "written": written,
            "skipped": skipped,
            "failures": failures,
            "elapsed_seconds": round(time.time() - started, 3),
        },
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
