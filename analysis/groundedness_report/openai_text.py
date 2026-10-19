"""Strict, auditable Responses API client for the frozen text pipeline.

The implementation intentionally does not load the process's generic OpenAI key
and does not use a provider SDK. Only ``OPENAI_API_KEY_2`` is read.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import httpx

from .io import atomic_json, read_json
from .protocol import load_protocol, load_text_model_correction_amendment
from .spec import PROTOCOL_SHA256, TEXT_MODEL_CORRECTION_AMENDMENT_SHA256


API_URL = "https://api.openai.com/v1/responses"
KEY_NAME = "OPENAI_API_KEY_2"
MODEL = "gpt-5.6-terra"


def _dotenv_value(path: Path, name: str) -> str | None:
    if not path.exists():
        return None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() != name:
            continue
        result = value.strip()
        if len(result) >= 2 and result[0] == result[-1] and result[0] in {"'", '"'}:
            result = result[1:-1]
        return result or None
    return None


def load_exclusive_api_key(dotenv_path: str | Path = ".env") -> str:
    """Load only OPENAI_API_KEY_2; never inspect or fall back to another key."""
    value = os.environ.get(KEY_NAME) or _dotenv_value(Path(dotenv_path), KEY_NAME)
    if not value:
        raise ValueError(f"{KEY_NAME} is required; fallback API keys are forbidden")
    return value


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _output_text(response: Mapping[str, Any]) -> str:
    pieces: list[str] = []
    refusals: list[str] = []
    for item in response.get("output") or []:
        for block in item.get("content") or []:
            if block.get("type") == "output_text" and isinstance(block.get("text"), str):
                pieces.append(block["text"])
            elif block.get("type") == "refusal":
                refusals.append(str(block.get("refusal") or "unspecified refusal"))
    if refusals:
        raise ValueError("OpenAI response refused: " + "; ".join(refusals))
    if not pieces:
        raise ValueError("OpenAI response contains no output_text")
    return "".join(pieces)


@dataclass(frozen=True)
class JSONResponse:
    value: Any
    audit: dict[str, Any]


class TerraResponses:
    def __init__(
        self,
        *,
        dotenv_path: str | Path = ".env",
        timeout_seconds: float = 180.0,
        client: httpx.Client | None = None,
    ) -> None:
        load_protocol()
        amendment = load_text_model_correction_amendment()
        if (
            amendment["active_operational_model"] != MODEL
            or amendment["api_key_environment_variable"] != KEY_NAME
        ):
            raise ValueError("client constants differ from the text-model amendment")
        self._api_key = load_exclusive_api_key(dotenv_path)
        self._client = client or httpx.Client(timeout=timeout_seconds)
        self._owns_client = client is None

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "TerraResponses":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def request_json(
        self,
        *,
        task_name: str,
        system_prompt: str,
        user_prompt: str,
        schema: Mapping[str, Any],
        max_output_tokens: int,
        cache_path: str | Path | None = None,
        attempts: int = 6,
    ) -> JSONResponse:
        if not task_name or max_output_tokens < 1:
            raise ValueError("task_name and positive max_output_tokens are required")
        payload = {
            "model": MODEL,
            "input": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "reasoning": {"effort": "low"},
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": task_name,
                    "strict": True,
                    "schema": dict(schema),
                }
            },
            "max_output_tokens": int(max_output_tokens),
            "store": False,
        }
        request_sha256 = hashlib.sha256(canonical_json(payload)).hexdigest()
        cache = Path(cache_path) if cache_path else None
        if cache and cache.exists():
            stored = read_json(cache)
            if stored.get("request_sha256") != request_sha256:
                raise ValueError(f"stale OpenAI cache at {cache}")
            if (stored.get("audit") or {}).get("protocol_sha256") != PROTOCOL_SHA256:
                raise ValueError(f"OpenAI cache predates the frozen protocol at {cache}")
            if (stored.get("audit") or {}).get(
                "text_model_correction_amendment_sha256"
            ) != TEXT_MODEL_CORRECTION_AMENDMENT_SHA256:
                raise ValueError(f"OpenAI cache predates the Terra correction at {cache}")
            return JSONResponse(value=stored["parsed"], audit=dict(stored["audit"]))

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        last_error: Exception | None = None
        response: httpx.Response | None = None
        for attempt in range(attempts):
            try:
                response = self._client.post(API_URL, headers=headers, json=payload)
                if response.status_code not in {408, 409, 429, 500, 502, 503, 504}:
                    response.raise_for_status()
                    break
                last_error = RuntimeError(
                    f"retryable OpenAI HTTP {response.status_code}: {response.text[:500]}"
                )
            except (httpx.HTTPError, RuntimeError) as error:
                last_error = error
            if attempt + 1 < attempts:
                time.sleep(min(20.0, 0.75 * (2**attempt)))
        else:
            raise RuntimeError(f"OpenAI request failed after {attempts} attempts: {last_error}")
        assert response is not None
        raw = response.json()
        resolved_model = raw.get("model")
        if not isinstance(resolved_model, str) or not resolved_model.startswith(MODEL):
            raise RuntimeError(
                f"resolved model {resolved_model!r} differs from required Terra model"
            )
        text = _output_text(raw)
        parsed = json.loads(text)
        raw_sha256 = hashlib.sha256(canonical_json(raw)).hexdigest()
        audit = {
            "protocol_sha256": PROTOCOL_SHA256,
            "text_model_correction_amendment_sha256": (
                TEXT_MODEL_CORRECTION_AMENDMENT_SHA256
            ),
            "task_name": task_name,
            "request_sha256": request_sha256,
            "response_sha256": raw_sha256,
            "response_id": raw.get("id"),
            "requested_model": MODEL,
            "resolved_model": resolved_model,
            "system_fingerprint": raw.get("system_fingerprint"),
            "created_at": raw.get("created_at"),
            "usage": raw.get("usage"),
            "store": False,
        }
        if cache:
            atomic_json(cache, {
                "request_sha256": request_sha256,
                "parsed": parsed,
                "audit": audit,
                "raw_response": raw,
            })
        return JSONResponse(value=parsed, audit=audit)
