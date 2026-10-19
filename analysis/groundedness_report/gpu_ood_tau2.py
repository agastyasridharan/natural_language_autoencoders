"""Run Qwen trajectories in the official tau2-bench Retail environment."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from .gpu_ood_static import _atomic_npy, _capture_layer, _decoder_layers, _slug, _task_key
from .io import atomic_json, read_json, read_jsonl, sha256_file
from .protocol import load_protocol
from .sampling import sample_positions
from .spec import PROTOCOL_SHA256, TEXT_MODEL_CORRECTION_AMENDMENT_SHA256


TERRA_MODEL = "gpt-5.6-terra"
TERRA_API_URL = "https://api.openai.com/v1/responses"
TERRA_KEY_NAME = "OPENAI_API_KEY_2"
FAMILY_ID = "capability_tau2_retail"
_TOOL_RESPONSE = re.compile(
    r"\s*<tool_call>\s*(\{.*\})\s*</tool_call>\s*", re.DOTALL
)


class TerraInfrastructureError(RuntimeError):
    """A user-simulator/provider failure that must not become a task outcome."""


def _preflight_openai_endpoint(
    *,
    timeout_seconds: float = 15.0,
    transport: httpx.BaseTransport | None = None,
) -> dict[str, Any]:
    """Verify TLS/HTTP reachability, honoring the audited CONNECT proxy."""
    url = "https://api.openai.com/v1/models"
    try:
        with httpx.Client(
            timeout=timeout_seconds,
            trust_env=transport is None,
            transport=transport,
        ) as client:
            response = client.get(url)
    except httpx.HTTPError as error:
        raise TerraInfrastructureError(
            f"OpenAI endpoint HTTPS preflight failed: {type(error).__name__}: {error}"
        ) from error
    return {
        "url": url,
        "http_status_without_credentials": response.status_code,
        "proxy_configured": transport is None
        and bool(os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")),
    }


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()


def _parse_qwen_tool_response(response: str) -> tuple[dict[str, Any] | None, str | None]:
    """Parse exactly one pure Qwen tool-call response, retaining failures."""
    match = _TOOL_RESPONSE.fullmatch(response)
    if match is None:
        if "<tool_call>" in response or "</tool_call>" in response:
            return None, "tool-call tags did not contain exactly one pure call"
        return None, None
    try:
        value = json.loads(match.group(1))
    except json.JSONDecodeError as error:
        return None, f"invalid tool-call JSON: {error}"
    if not isinstance(value, dict):
        return None, "tool call must be a JSON object"
    name = value.get("name")
    arguments = value.get("arguments")
    if not isinstance(name, str) or not name or not isinstance(arguments, dict):
        return None, "tool call requires a string name and object arguments"
    return {"name": name, "arguments": arguments}, None


def _responses_output_text(response: dict[str, Any]) -> str:
    pieces = []
    refusals = []
    for item in response.get("output") or []:
        for block in item.get("content") or []:
            if block.get("type") == "output_text" and isinstance(block.get("text"), str):
                pieces.append(block["text"])
            elif block.get("type") == "refusal":
                refusals.append(str(block.get("refusal") or "unspecified refusal"))
    if refusals:
        raise ValueError("Terra user simulator refused: " + "; ".join(refusals))
    result = "".join(pieces)
    if not result.strip():
        raise ValueError("Terra user simulator returned no output text")
    return result


class _TerraClient:
    """Responses client that reads only OPENAI_API_KEY_2 and archives raw calls."""

    def __init__(self, audit_dir: Path, *, timeout_seconds: float = 180.0) -> None:
        key = os.environ.get(TERRA_KEY_NAME)
        if not key:
            raise ValueError(f"{TERRA_KEY_NAME} is required; fallback keys are forbidden")
        self._key = key
        self._client = httpx.Client(timeout=timeout_seconds)
        self._audit_dir = audit_dir
        self._call_index = 0

    def close(self) -> None:
        self._client.close()

    def generate(self, messages: list[dict[str, str]], *, max_output_tokens: int) -> tuple[str, dict[str, Any]]:
        payload = {
            "model": TERRA_MODEL,
            "input": messages,
            "reasoning": {"effort": "low"},
            "max_output_tokens": int(max_output_tokens),
            "store": False,
        }
        request_sha256 = hashlib.sha256(_canonical(payload)).hexdigest()
        headers = {
            "Authorization": f"Bearer {self._key}",
            "Content-Type": "application/json",
        }
        response: httpx.Response | None = None
        last_error: Exception | None = None
        retryable_statuses = {408, 409, 429, 500, 502, 503, 504}
        for attempt in range(6):
            try:
                response = self._client.post(TERRA_API_URL, headers=headers, json=payload)
                if response.status_code not in retryable_statuses:
                    response.raise_for_status()
                    break
                last_error = RuntimeError(
                    f"retryable OpenAI HTTP {response.status_code}"
                )
            except httpx.HTTPError as error:
                last_error = error
            if attempt < 5:
                time.sleep(min(20.0, 0.75 * (2**attempt)))
        else:
            assert last_error is not None
            raise TerraInfrastructureError(
                "Terra user request failed after six attempts: "
                f"{type(last_error).__name__}: {last_error}"
            ) from last_error
        assert response is not None
        try:
            raw = response.json()
            if not str(raw.get("model") or "").startswith(TERRA_MODEL):
                raise ValueError(
                    f"resolved model {raw.get('model')!r} differs from {TERRA_MODEL!r}"
                )
            text = _responses_output_text(raw)
        except (json.JSONDecodeError, ValueError) as error:
            raise TerraInfrastructureError(
                f"invalid Terra provider response: {type(error).__name__}: {error}"
            ) from error
        response_sha256 = hashlib.sha256(_canonical(raw)).hexdigest()
        audit = {
            "schema_version": 1,
            "protocol_sha256": PROTOCOL_SHA256,
            "text_model_correction_amendment_sha256": (
                TEXT_MODEL_CORRECTION_AMENDMENT_SHA256
            ),
            "call_index": self._call_index,
            "request_sha256": request_sha256,
            "response_sha256": response_sha256,
            "requested_model": TERRA_MODEL,
            "resolved_model": raw.get("model"),
            "response_id": raw.get("id"),
            "created_at": raw.get("created_at"),
            "system_fingerprint": raw.get("system_fingerprint"),
            "usage": raw.get("usage"),
            "store": False,
            "request": payload,
            "raw_response": raw,
        }
        atomic_json(self._audit_dir / f"call-{self._call_index:02d}.json", audit)
        self._call_index += 1
        return text, audit


def _make_terra_user_class(UserSimulator: Any, UserMessage: Any, MultiToolMessage: Any, ToolMessage: Any):
    class TerraTauUser(UserSimulator):
        def __init__(self, *, audit_dir: Path, instructions: str, max_output_tokens: int):
            super().__init__(
                llm=TERRA_MODEL,
                instructions=instructions,
                tools=None,
                llm_args=None,
            )
            self._terra = _TerraClient(audit_dir)
            self._max_output_tokens = max_output_tokens

        def _generate_next_message(self, message: Any, state: Any) -> Any:
            if isinstance(message, MultiToolMessage):
                state.messages.extend(message.tool_messages)
            elif isinstance(message, ToolMessage):
                state.messages.append(message)
            elif message.has_content() or message.is_tool_call():
                state.messages.append(message)
            converted = []
            for row in state.system_messages + state.flip_roles():
                if hasattr(row, "is_tool_call") and row.is_tool_call():
                    raise ValueError("Retail user simulator unexpectedly received a tool call")
                if row.content is None:
                    raise ValueError("Retail user simulator received an empty message")
                converted.append({"role": row.role, "content": row.content})
            text, audit = self._terra.generate(
                converted, max_output_tokens=self._max_output_tokens
            )
            return UserMessage(
                role="user",
                content=text,
                cost=0.0,
                usage=audit.get("usage"),
                raw_data={
                    "response_id": audit.get("response_id"),
                    "request_sha256": audit["request_sha256"],
                    "response_sha256": audit["response_sha256"],
                    "resolved_model": audit.get("resolved_model"),
                },
            )

        def stop(self, message: Any = None, state: Any = None) -> None:
            self._terra.close()

    return TerraTauUser


def _qwen_messages(messages: list[Any]) -> list[dict[str, Any]]:
    converted = []
    for message in messages:
        if message.role in {"system", "user"}:
            if message.content is None:
                raise ValueError(f"{message.role} message has no text")
            converted.append({"role": message.role, "content": message.content})
        elif message.role == "assistant":
            if message.is_tool_call():
                content = "\n".join(
                    "<tool_call>\n"
                    + json.dumps(
                        {"name": call.name, "arguments": call.arguments},
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n</tool_call>"
                    for call in message.tool_calls
                )
            else:
                content = message.content or ""
            converted.append({"role": "assistant", "content": content})
        elif message.role == "tool":
            converted.append({"role": "tool", "content": message.content or ""})
        else:
            raise ValueError(f"unsupported tau2 message role {message.role!r}")
    return converted


def _make_qwen_agent_class(
    *,
    torch: Any,
    LLMAgent: Any,
    AssistantMessage: Any,
    ToolCall: Any,
    MultiToolMessage: Any,
    model: Any,
    tokenizer: Any,
    layer: Any,
    maximum_new_tokens: int,
):
    class QwenTauAgent(LLMAgent):
        def __init__(self, tools: list[Any], domain_policy: str):
            super().__init__(
                tools=tools,
                domain_policy=domain_policy,
                llm="local/Qwen2.5-7B-Instruct",
                llm_args=None,
            )
            self.trace: list[dict[str, Any]] = []
            self.eligible: list[dict[str, Any]] = []

        def _generate_next_message(self, message: Any, state: Any) -> Any:
            if isinstance(message, MultiToolMessage):
                state.messages.extend(message.tool_messages)
            else:
                state.messages.append(message)
            messages = state.system_messages + state.messages
            converted = _qwen_messages(messages)
            schemas = [tool.openai_schema for tool in self.tools]
            encoded = tokenizer.apply_chat_template(
                converted,
                tools=schemas,
                tokenize=True,
                add_generation_prompt=True,
            )
            prompt_ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
            if prompt_ids and isinstance(prompt_ids[0], list):
                prompt_ids = prompt_ids[0]
            prompt_ids = [int(value) for value in prompt_ids]
            max_context = int(getattr(model.config, "max_position_embeddings", 32768))
            if len(prompt_ids) + maximum_new_tokens > max_context:
                raise ValueError(
                    f"tau2 prompt exceeds target context: {len(prompt_ids)}+"
                    f"{maximum_new_tokens}>{max_context}"
                )
            device = model.get_input_embeddings().weight.device
            input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            eos_id = tokenizer.eos_token_id
            if eos_id is None:
                raise ValueError("target tokenizer has no EOS token")
            with torch.inference_mode():
                generated = model.generate(
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    do_sample=False,
                    max_new_tokens=maximum_new_tokens,
                    pad_token_id=eos_id,
                    eos_token_id=eos_id,
                    stop_strings=["</tool_call>"],
                    tokenizer=tokenizer,
                    use_cache=True,
                )[0]
            continuation_ids = [int(value) for value in generated[len(prompt_ids) :].tolist()]
            response = tokenizer.decode(continuation_ids, skip_special_tokens=True)
            call, call_error = _parse_qwen_tool_response(response)
            known_tools = {tool.name for tool in self.tools}
            if call is not None and call["name"] not in known_tools:
                call_error = f"unknown tool name {call['name']!r}"
                call = None
            turn_index = len(self.trace)
            trace = {
                "turn_index": turn_index,
                "prompt_tokens": len(prompt_ids),
                "prompt_ids_sha256": hashlib.sha256(
                    np.asarray(prompt_ids, dtype=np.int64).tobytes()
                ).hexdigest(),
                "continuation_token_ids": continuation_ids,
                "response": response,
                "tool_call": call,
                "tool_call_error": call_error,
            }
            if call is None and call_error is None and response.strip():
                with torch.inference_mode():
                    hidden = _capture_layer(
                        torch=torch,
                        model=model,
                        layer=layer,
                        input_ids=generated.view(1, -1),
                    )
                special_ids = {int(value) for value in tokenizer.all_special_ids}
                for response_index, token_id in enumerate(continuation_ids):
                    absolute = len(prompt_ids) + response_index
                    if token_id in special_ids or absolute < 50:
                        continue
                    self.eligible.append(
                        {
                            "position_key": turn_index * 1_000_000 + response_index,
                            "turn_index": turn_index,
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
                            "vector": hidden[absolute]
                            .numpy()
                            .astype(np.float32, copy=True),
                        }
                    )
            self.trace.append(trace)
            if call is not None:
                call_id = "call_" + hashlib.sha256(
                    f"{turn_index}\0{response}".encode()
                ).hexdigest()[:20]
                return AssistantMessage(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id=call_id,
                            name=call["name"],
                            arguments=call["arguments"],
                            requestor="assistant",
                        )
                    ],
                    cost=0.0,
                    raw_data={
                        "local_target": True,
                        "prompt_ids_sha256": trace["prompt_ids_sha256"],
                    },
                )
            return AssistantMessage(
                role="assistant",
                content=response,
                cost=0.0,
                raw_data={
                    "local_target": True,
                    "prompt_ids_sha256": trace["prompt_ids_sha256"],
                    "tool_call_error": call_error,
                },
            )

    return QwenTauAgent


def _validate_task(task: dict[str, Any]) -> None:
    if task.get("protocol_sha256") != PROTOCOL_SHA256:
        raise ValueError("tau2 task has a stale protocol hash")
    if task.get("family_id") != FAMILY_ID:
        raise ValueError("tau2 task has the wrong family")
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
        raise ValueError("duplicate tau2 task IDs")
    if task_id is not None:
        rows = [task for task in rows if str(task["task_id"]) == task_id]
        if not rows:
            raise ValueError(f"requested tau2 task ID is absent: {task_id}")
    else:
        rows = [task for index, task in enumerate(rows) if index % num_shards == shard_index]
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise ValueError("no tau2 tasks assigned")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--tau2-source", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--task-id")
    parser.add_argument("--maximum-steps", type=int, default=100)
    parser.add_argument("--maximum-user-tokens", type=int, default=1024)
    args = parser.parse_args()
    if args.maximum_steps < 1 or args.maximum_user_tokens < 1:
        raise ValueError("step and token limits must be positive")

    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    protocol = load_protocol()
    tasks = _select_tasks(
        list(read_jsonl(args.tasks)),
        num_shards=args.num_shards,
        shard_index=args.shard_index,
        limit=args.limit,
        task_id=args.task_id,
    )
    tau2_source = Path(args.tau2_source)
    source_tasks_path = tau2_source / "data/tau2/domains/retail/tasks.json"
    source_rows = json.loads(source_tasks_path.read_text(encoding="utf-8"))
    source_by_id = {str(row["id"]): row for row in source_rows}
    for task in tasks:
        source = task["source"]
        if source["source_tasks_sha256"] != sha256_file(source_tasks_path):
            raise ValueError("live tau2 Retail task file differs from the selected source")
        row = source_by_id.get(str(task["task_id"]))
        if row is None or hashlib.sha256(_canonical(row)).hexdigest() != source["source_task_sha256"]:
            raise ValueError(f"live tau2 task differs for {task['task_id']}")

    output = Path(args.out)
    meta_path = output / f"{FAMILY_ID}-shard-{args.shard_index:02d}-meta.json"
    startup_meta = {
        "schema_version": 1,
        "protocol_sha256": PROTOCOL_SHA256,
        "text_model_correction_amendment_sha256": (
            TEXT_MODEL_CORRECTION_AMENDMENT_SHA256
        ),
        "family_id": FAMILY_ID,
        "assigned_tasks": len(tasks),
        "assigned_task_ids": [str(task["task_id"]) for task in tasks],
        "task_artifact_sha256": sha256_file(args.tasks),
        "tau2_source_tasks_sha256": sha256_file(source_tasks_path),
        "tau2_source": str(tau2_source),
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
        print(f"[tau2 preflight] {type(error).__name__}: {error}", flush=True)
        return 2

    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from tau2.agent.llm_agent import LLMAgent
    from tau2.data_model.message import (
        AssistantMessage,
        MultiToolMessage,
        ToolCall,
        ToolMessage,
        UserMessage,
    )
    from tau2.orchestrator.orchestrator import Orchestrator
    from tau2.runner import build_environment, get_tasks
    from tau2.user.user_simulator import UserSimulator

    official_tasks = {str(task.id): task for task in get_tasks("retail")}
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
    layer = layers[int(target["layer"])]
    QwenTauAgent = _make_qwen_agent_class(
        torch=torch,
        LLMAgent=LLMAgent,
        AssistantMessage=AssistantMessage,
        ToolCall=ToolCall,
        MultiToolMessage=MultiToolMessage,
        model=model,
        tokenizer=tokenizer,
        layer=layer,
        maximum_new_tokens=int(
            protocol["sampling"]["ood"]["generation"].get(
                "max_new_tokens_per_assistant_turn", 1024
            )
        ),
    )
    TerraTauUser = _make_terra_user_class(
        UserSimulator, UserMessage, MultiToolMessage, ToolMessage
    )

    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    base_meta = {
        **startup_meta,
        "target": target,
        "layer_path": layer_path,
        "openai_endpoint_preflight": endpoint_preflight,
        "tau2_user_simulator": {
            "model": TERRA_MODEL,
            "api": "responses",
            "reasoning_effort": "low",
            "max_output_tokens": args.maximum_user_tokens,
            "store": False,
            "api_key_environment_variable": TERRA_KEY_NAME,
        },
        "maximum_steps": args.maximum_steps,
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
                raise ValueError(f"pre-Terra trajectory at {trajectory_path}")
            for activation in existing.get("activations") or []:
                if not (output / activation["vector_file"]).is_file():
                    raise ValueError(f"trajectory has a missing activation: {trajectory_path}")
            skipped += 1
            print(
                f"[tau2 {position}/{len(tasks)} shard={args.shard_index}] "
                f"written={written} skipped={skipped} failures={len(failures)}",
                flush=True,
            )
            continue
        if failure_path.exists():
            failure = read_json(failure_path)
            failures.append(
                {"task_id": selected["task_id"], "error": failure.get("error"), "preserved_existing": True}
            )
            print(
                f"[tau2 {position}/{len(tasks)} shard={args.shard_index}] "
                f"written={written} skipped={skipped} failures={len(failures)}",
                flush=True,
            )
            continue
        attempt_id = f"{time.time_ns()}-{os.getpid()}"
        audit_rel = (
            Path("tools")
            / FAMILY_ID
            / slug
            / "attempts"
            / attempt_id
            / "terra-user"
        )
        try:
            task = official_tasks[str(selected["task_id"])]
            environment = build_environment("retail")
            agent = QwenTauAgent(
                tools=environment.get_tools(), domain_policy=environment.get_policy()
            )
            user = TerraTauUser(
                audit_dir=output / audit_rel,
                instructions=str(task.user_scenario),
                max_output_tokens=args.maximum_user_tokens,
            )
            orchestrator = Orchestrator(
                domain="retail",
                agent=agent,
                user=user,
                environment=environment,
                task=task,
                max_steps=args.maximum_steps,
                max_errors=10,
                seed=int(protocol["seed"]),
                validate_communication=True,
                timeout=1800.0,
            )
            simulation = orchestrator.run()
            selected_positions = set(
                sample_positions(
                    [int(row["position_key"]) for row in agent.eligible],
                    trajectory_id=_task_key(selected),
                    seed=int(protocol["seed"]),
                    maximum=4,
                )
            )
            if not selected_positions:
                raise ValueError("tau2 trajectory has no eligible assistant-language token")
            activations = []
            for ordinal, row in enumerate(
                sorted(
                    (
                        value
                        for value in agent.eligible
                        if value["position_key"] in selected_positions
                    ),
                    key=lambda value: value["position_key"],
                )
            ):
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
                "replication_label": selected["replication_label"],
                "family_id": FAMILY_ID,
                "trajectory_id": _task_key(selected),
                "task": selected,
                "target_model": target,
                "generation": {
                    "do_sample": False,
                    "temperature": None,
                    "max_new_tokens_per_assistant_turn": int(
                        protocol["sampling"]["ood"]["generation"].get(
                            "max_new_tokens_per_assistant_turn", 1024
                        )
                    ),
                    "maximum_steps": args.maximum_steps,
                },
                "tau2": {
                    "source_revision": selected["source"]["benchmark_revision"],
                    "domain": "retail",
                    "user_simulator_model": TERRA_MODEL,
                    "user_simulator_api": "responses",
                    "user_simulator_reasoning_effort": "low",
                    "user_simulator_audit_dir": str(audit_rel),
                    "simulation": simulation.model_dump(mode="json"),
                },
                "agent_turns": agent.trace,
                "eligible_position_population": len(agent.eligible),
                "activations": activations,
                "status": "complete",
            }
            atomic_json(trajectory_path, trajectory)
            written += 1
        except TerraInfrastructureError as error:
            interruption = {
                "task_id": selected["task_id"],
                "attempt_id": attempt_id,
                "audit_dir": str(audit_rel),
                "error": f"{type(error).__name__}: {error}",
            }
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
                f"[tau2 {position}/{len(tasks)} shard={args.shard_index}] "
                f"interrupted_infrastructure task={selected['task_id']}",
                flush=True,
            )
            return 2
        except Exception as error:
            failure = {
                "schema_version": 1,
                "protocol_sha256": PROTOCOL_SHA256,
                "text_model_correction_amendment_sha256": (
                    TEXT_MODEL_CORRECTION_AMENDMENT_SHA256
                ),
                "family_id": FAMILY_ID,
                "task_id": selected["task_id"],
                "error": f"{type(error).__name__}: {error}",
                "task": selected,
            }
            atomic_json(failure_path, failure)
            failures.append({"task_id": selected["task_id"], "error": failure["error"]})
        print(
            f"[tau2 {position}/{len(tasks)} shard={args.shard_index}] "
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
