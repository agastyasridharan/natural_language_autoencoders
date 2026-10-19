"""Auditable Responses-API cache with conservative hard-budget enforcement.

Every attempt stores the redacted request, complete HTTP response body and headers, parsed
structured output (when valid), token usage, and cost.  No response is stored by OpenAI
(``store=false``).  Unknown-usage failures retain their full preflight reservation, so the
ledger is conservative after network ambiguity or process interruption.
"""

from __future__ import annotations

import fcntl
import json
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import httpx

from .core import ValidationError, canonical_json, content_hash


LEDGER_SCHEMA_VERSION = "nla.claim_agreement.budget.v1"
CACHE_SCHEMA_VERSION = "nla.claim_agreement.api_cache.v1"
_LEDGER_THREAD_LOCK = threading.Lock()
_KEY_LOCKS_GUARD = threading.Lock()
_KEY_THREAD_LOCKS: dict[str, threading.Lock] = {}


class BudgetExceeded(RuntimeError):
    pass


class CachedCallFailure(RuntimeError):
    def __init__(self, message: str, *, code: str, cache_key: str):
        super().__init__(message)
        self.code = code
        self.cache_key = cache_key


@dataclass(frozen=True)
class ApiConfig:
    endpoint: str
    model: str
    reasoning_effort: str
    service_tier: str
    input_usd_per_million: float
    cached_input_usd_per_million: float
    output_usd_per_million: float
    hard_budget_usd: float
    timeout_seconds: float = 180.0
    input_token_overhead_bound: int = 4096

    def validate(self) -> None:
        if self.endpoint != "https://api.openai.com/v1/responses":
            raise ValidationError("the frozen protocol requires the OpenAI Responses endpoint")
        if not self.model:
            raise ValidationError("model must be non-empty")
        if self.service_tier != "default":
            raise ValidationError("cost accounting is frozen to the default service tier")
        if self.hard_budget_usd <= 0:
            raise ValidationError("hard_budget_usd must be positive")
        for value in (
            self.input_usd_per_million,
            self.cached_input_usd_per_million,
            self.output_usd_per_million,
        ):
            if value < 0:
                raise ValidationError("token prices must be non-negative")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_api_key() -> str:
    """Read only the explicitly authorized key; never fall back to OPENAI_API_KEY."""

    value = os.environ.get("OPENAI_API_KEY_2")
    if not value:
        raise RuntimeError("OPENAI_API_KEY_2 is required for uncached API calls")
    return value


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}.{threading.get_ident()}")
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValidationError(f"expected JSON object in {path}")
    return value


class BudgetLedger:
    def __init__(self, cache_root: Path, config: ApiConfig):
        self.cache_root = cache_root
        self.config = config
        self.path = cache_root / "budget_ledger.json"
        self.lock_path = cache_root / "budget_ledger.lock"
        cache_root.mkdir(parents=True, exist_ok=True)

    def _new(self) -> dict[str, Any]:
        return {
            "schema_version": LEDGER_SCHEMA_VERSION,
            "model": self.config.model,
            "hard_budget_usd": self.config.hard_budget_usd,
            "pricing_usd_per_million": {
                "input": self.config.input_usd_per_million,
                "cached_input": self.config.cached_input_usd_per_million,
                "output": self.config.output_usd_per_million,
            },
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "reservations": {},
            "charges": {},
        }

    def _load(self) -> dict[str, Any]:
        ledger = _read_json(self.path) if self.path.exists() else self._new()
        expected_prices = {
            "input": self.config.input_usd_per_million,
            "cached_input": self.config.cached_input_usd_per_million,
            "output": self.config.output_usd_per_million,
        }
        if ledger.get("schema_version") != LEDGER_SCHEMA_VERSION:
            raise ValidationError("unsupported budget ledger schema")
        if ledger.get("model") != self.config.model:
            raise ValidationError("budget ledger model differs from the frozen protocol")
        if ledger.get("pricing_usd_per_million") != expected_prices:
            raise ValidationError("budget ledger prices differ from the frozen protocol")
        if float(ledger.get("hard_budget_usd", -1)) != self.config.hard_budget_usd:
            raise ValidationError("budget ledger hard cap cannot be changed after initialization")
        return ledger

    @staticmethod
    def _totals(ledger: Mapping[str, Any]) -> tuple[float, float]:
        charged = sum(float(item["accounted_usd"]) for item in ledger["charges"].values())
        reserved = sum(float(item["reserved_usd"]) for item in ledger["reservations"].values())
        return charged, reserved

    def _with_lock(self, operation: Callable[[dict[str, Any]], Any]) -> Any:
        with _LEDGER_THREAD_LOCK:
            with self.lock_path.open("a+b") as lock_handle:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
                ledger = self._load()
                result = operation(ledger)
                ledger["updated_at"] = utc_now()
                charged, reserved = self._totals(ledger)
                ledger["accounted_spend_usd"] = charged
                ledger["active_reservations_usd"] = reserved
                ledger["maximum_committed_usd"] = charged + reserved
                _atomic_json(self.path, ledger)
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
                return result

    def worst_case_cost(self, request_body: Mapping[str, Any], max_output_tokens: int) -> tuple[float, int]:
        # A byte-level upper bound is deliberately conservative for BPE tokenization.  The fixed
        # overhead also bounds response-message framing not explicit in the JSON request body.
        request_bytes = len(canonical_json(request_body).encode("utf-8"))
        input_token_bound = request_bytes + self.config.input_token_overhead_bound
        cost = (
            input_token_bound * self.config.input_usd_per_million
            + max_output_tokens * self.config.output_usd_per_million
        ) / 1_000_000
        return cost, input_token_bound

    def reserve(self, call_id: str, request_body: Mapping[str, Any], max_output_tokens: int) -> dict[str, Any]:
        reserved_usd, input_bound = self.worst_case_cost(request_body, max_output_tokens)

        def operation(ledger: dict[str, Any]) -> dict[str, Any]:
            if call_id in ledger["charges"] or call_id in ledger["reservations"]:
                raise ValidationError(f"duplicate budget call id {call_id}")
            charged, reserved = self._totals(ledger)
            committed = charged + reserved + reserved_usd
            if committed > self.config.hard_budget_usd + 1e-12:
                raise BudgetExceeded(
                    f"hard API budget would be exceeded: ${committed:.6f} > "
                    f"${self.config.hard_budget_usd:.6f}"
                )
            record = {
                "call_id": call_id,
                "reserved_at": utc_now(),
                "reserved_usd": reserved_usd,
                "input_token_upper_bound": input_bound,
                "max_output_tokens": max_output_tokens,
                "request_sha256": content_hash(request_body),
            }
            ledger["reservations"][call_id] = record
            return record

        return self._with_lock(operation)

    def exact_cost(self, usage: Mapping[str, Any]) -> float:
        input_tokens = int(usage.get("input_tokens", 0))
        output_tokens = int(usage.get("output_tokens", 0))
        details = usage.get("input_tokens_details") or {}
        cached_tokens = int(details.get("cached_tokens", 0)) if isinstance(details, Mapping) else 0
        if min(input_tokens, output_tokens, cached_tokens) < 0 or cached_tokens > input_tokens:
            raise ValidationError(f"invalid API usage object: {usage}")
        uncached = input_tokens - cached_tokens
        return (
            uncached * self.config.input_usd_per_million
            + cached_tokens * self.config.cached_input_usd_per_million
            + output_tokens * self.config.output_usd_per_million
        ) / 1_000_000

    def finalize(self, call_id: str, usage: Mapping[str, Any] | None, outcome: str) -> dict[str, Any]:
        def operation(ledger: dict[str, Any]) -> dict[str, Any]:
            if call_id in ledger["charges"]:
                # Idempotence permits recovery after a crash between ledger finalization and
                # writing the immutable attempt/result cache records.
                return dict(ledger["charges"][call_id])
            reservation = ledger["reservations"].pop(call_id, None)
            if reservation is None:
                raise ValidationError(f"missing budget reservation {call_id}")
            if usage is None:
                accounted = float(reservation["reserved_usd"])
                basis = "conservative_full_reservation_missing_usage"
            else:
                accounted = self.exact_cost(usage)
                if accounted > float(reservation["reserved_usd"]) + 1e-9:
                    raise ValidationError(
                        f"actual cost ${accounted:.8f} exceeded preflight reservation "
                        f"${reservation['reserved_usd']:.8f}"
                    )
                basis = "reported_token_usage"
            charge = {
                **reservation,
                "finalized_at": utc_now(),
                "outcome": outcome,
                "usage": dict(usage) if usage is not None else None,
                "accounted_usd": accounted,
                "accounting_basis": basis,
            }
            ledger["charges"][call_id] = charge
            return charge

        return self._with_lock(operation)

    def snapshot(self) -> dict[str, Any]:
        def operation(ledger: dict[str, Any]) -> dict[str, Any]:
            return json.loads(json.dumps(ledger))

        return self._with_lock(operation)


def _key_thread_lock(path: Path) -> threading.Lock:
    key = str(path.resolve())
    with _KEY_LOCKS_GUARD:
        return _KEY_THREAD_LOCKS.setdefault(key, threading.Lock())


def make_request_body(
    config: ApiConfig,
    *,
    instructions: str,
    input_text: str,
    schema_name: str,
    schema: Mapping[str, Any],
    max_output_tokens: int,
) -> dict[str, Any]:
    """Build the exact redacted request body, also used by dry-run accounting."""

    return {
        "model": config.model,
        "instructions": instructions,
        "input": [{"role": "user", "content": [{"type": "input_text", "text": input_text}]}],
        "text": {
            "format": {
                "type": "json_schema",
                "name": schema_name,
                "strict": True,
                "schema": dict(schema),
            }
        },
        "reasoning": {"effort": config.reasoning_effort},
        "max_output_tokens": max_output_tokens,
        "service_tier": config.service_tier,
        "store": False,
    }


class CachedResponsesClient:
    def __init__(
        self,
        cache_root: Path,
        config: ApiConfig,
        *,
        transport: httpx.BaseTransport | None = None,
    ):
        config.validate()
        self.cache_root = cache_root
        self.config = config
        self.transport = transport
        self.ledger = BudgetLedger(cache_root, config)

    @staticmethod
    def _response_output_text(response_json: Mapping[str, Any]) -> str:
        texts: list[str] = []
        for item in response_json.get("output", []):
            if not isinstance(item, Mapping) or item.get("type") != "message":
                continue
            for content in item.get("content", []):
                if isinstance(content, Mapping) and content.get("type") == "output_text":
                    text = content.get("text")
                    if isinstance(text, str):
                        texts.append(text)
        if not texts:
            raise ValidationError("Responses API result has no output_text content")
        return "".join(texts)

    def _request_body(
        self,
        *,
        instructions: str,
        input_text: str,
        schema_name: str,
        schema: Mapping[str, Any],
        max_output_tokens: int,
    ) -> dict[str, Any]:
        return make_request_body(
            self.config,
            instructions=instructions,
            input_text=input_text,
            schema_name=schema_name,
            schema=schema,
            max_output_tokens=max_output_tokens,
        )

    def call(
        self,
        *,
        stage: str,
        cache_key: str,
        instructions: str,
        input_text: str,
        schema_name: str,
        schema: Mapping[str, Any],
        max_output_tokens: int,
        validator: Callable[[Mapping[str, Any]], dict[str, Any]],
        retry_failure: bool = False,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if not stage.replace("_", "").isalnum() or not cache_key.isalnum():
            raise ValidationError("stage/cache_key must be path-safe alphanumeric identifiers")
        request_body = self._request_body(
            instructions=instructions,
            input_text=input_text,
            schema_name=schema_name,
            schema=schema,
            max_output_tokens=max_output_tokens,
        )
        request_record = {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "stage": stage,
            "cache_key": cache_key,
            "url": self.config.endpoint,
            "headers": {"content-type": "application/json", "authorization": "REDACTED_OPENAI_API_KEY_2"},
            "body": request_body,
            "request_sha256": content_hash(request_body),
        }
        key_dir = self.cache_root / stage / cache_key
        key_dir.mkdir(parents=True, exist_ok=True)
        lock_path = key_dir / "call.lock"
        with _key_thread_lock(lock_path):
            with lock_path.open("a+b") as lock_handle:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
                try:
                    return self._call_locked(
                        key_dir=key_dir,
                        request_record=request_record,
                        validator=validator,
                        retry_failure=retry_failure,
                    )
                finally:
                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    def _call_locked(
        self,
        *,
        key_dir: Path,
        request_record: dict[str, Any],
        validator: Callable[[Mapping[str, Any]], dict[str, Any]],
        retry_failure: bool,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        request_path = key_dir / "request.json"
        result_path = key_dir / "result.json"
        if request_path.exists():
            existing = _read_json(request_path)
            if existing != request_record:
                raise ValidationError(f"cache-key collision or protocol drift at {key_dir}")
        else:
            _atomic_json(request_path, request_record)
        if result_path.exists():
            result = _read_json(result_path)
            return dict(result["parsed_output"]), result

        attempts_dir = key_dir / "attempts"
        attempts_dir.mkdir(exist_ok=True)
        journal_dir = key_dir / "journal"
        journal_dir.mkdir(exist_ok=True)
        self._recover_interrupted_journals(key_dir, attempts_dir, journal_dir)
        attempts = sorted(attempts_dir.glob("*.json"))
        if attempts:
            last = _read_json(attempts[-1])
            if last.get("outcome") == "success" and isinstance(last.get("parsed_output"), Mapping):
                result = {
                    "cache_schema_version": CACHE_SCHEMA_VERSION,
                    "stage": request_record["stage"],
                    "cache_key": request_record["cache_key"],
                    "attempt_number": int(last["attempt_number"]),
                    "attempt_path": str(attempts[-1]),
                    "parsed_output": dict(last["parsed_output"]),
                    "usage": last.get("usage"),
                    "charge": last.get("charge"),
                    "completed_at": utc_now(),
                }
                _atomic_json(result_path, result)
                return dict(result["parsed_output"]), result
        if attempts and not retry_failure:
            last = _read_json(attempts[-1])
            raise CachedCallFailure(
                f"cached {last.get('outcome', 'failed')} API attempt at {attempts[-1]}",
                code=str(last.get("outcome", "cached_failure")),
                cache_key=str(request_record["cache_key"]),
            )
        attempt_number = len(attempts)
        call_id = f"{request_record['stage']}:{request_record['cache_key']}:{attempt_number:04d}"
        body = request_record["body"]
        api_key = get_api_key()  # fail before reserving budget; never consult OPENAI_API_KEY
        reservation = self.ledger.reserve(call_id, body, int(body["max_output_tokens"]))
        started_at = utc_now()
        started = time.monotonic()
        preflight_journal_path = journal_dir / f"{attempt_number:04d}_preflight.json"
        response_journal_path = journal_dir / f"{attempt_number:04d}_response.json"
        _atomic_json(preflight_journal_path, {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "stage": request_record["stage"],
            "cache_key": request_record["cache_key"],
            "attempt_number": attempt_number,
            "call_id": call_id,
            "started_at": started_at,
            "request": request_record,
            "reservation": reservation,
        })
        response: httpx.Response | None = None
        response_json: dict[str, Any] | None = None
        usage: dict[str, Any] | None = None
        parsed: dict[str, Any] | None = None
        outcome = "network_error"
        error: str | None = None
        try:
            with httpx.Client(timeout=self.config.timeout_seconds, transport=self.transport) as client:
                response = client.post(
                    self.config.endpoint,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json=body,
                )
            try:
                raw_json = response.json()
                response_json = raw_json if isinstance(raw_json, dict) else {"_non_object_json": raw_json}
            except (json.JSONDecodeError, ValueError):
                response_json = None
            if response_json and isinstance(response_json.get("usage"), Mapping):
                usage = dict(response_json["usage"])
            if not response.is_success:
                outcome = f"http_{response.status_code}"
                error = f"OpenAI API returned HTTP {response.status_code}"
            elif response_json is None:
                outcome = "invalid_response_json"
                error = "OpenAI API returned a non-JSON response"
            elif response_json.get("status") != "completed":
                outcome = f"response_{response_json.get('status', 'missing_status')}"
                error = f"OpenAI response status was {response_json.get('status')!r}"
            else:
                output_text = self._response_output_text(response_json)
                decoded = json.loads(output_text)
                if not isinstance(decoded, Mapping):
                    raise ValidationError("structured output decoded to a non-object")
                parsed = validator(decoded)
                outcome = "success"
        except Exception as exc:
            if outcome == "network_error" and not isinstance(exc, httpx.HTTPError):
                outcome = "invalid_structured_output"
            error = f"{type(exc).__name__}: {exc}"

        response_record = {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "stage": request_record["stage"],
            "cache_key": request_record["cache_key"],
            "attempt_number": attempt_number,
            "call_id": call_id,
            "started_at": started_at,
            "elapsed_seconds": time.monotonic() - started,
            "request": request_record,
            "reservation": reservation,
            "response": None if response is None else {
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "body_text": response.text,
                "json": response_json,
            },
            "usage": usage,
            "outcome": outcome,
            "error": error,
            "parsed_output": parsed,
        }
        # Persist the complete response and usage before touching the ledger or emitting any
        # derived output.  This redundant journal makes the transaction recoverable after a kill.
        _atomic_json(response_journal_path, response_record)
        charge = self.ledger.finalize(call_id, usage, outcome)
        attempt_record = {**response_record, "charge": charge}
        attempt_path = attempts_dir / f"{attempt_number:04d}.json"
        _atomic_json(attempt_path, attempt_record)
        if parsed is None:
            raise CachedCallFailure(
                error or outcome,
                code=outcome,
                cache_key=str(request_record["cache_key"]),
            )
        result = {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "stage": request_record["stage"],
            "cache_key": request_record["cache_key"],
            "attempt_number": attempt_number,
            "attempt_path": str(attempt_path),
            "parsed_output": parsed,
            "usage": usage,
            "charge": charge,
            "completed_at": utc_now(),
        }
        _atomic_json(result_path, result)
        return parsed, result

    def _recover_interrupted_journals(self, key_dir: Path, attempts_dir: Path, journal_dir: Path) -> None:
        """Close interrupted cache transactions without repeating an ambiguous API request."""

        preflights = sorted(journal_dir.glob("*_preflight.json"))
        for preflight_path in preflights:
            prefix = preflight_path.name.split("_", 1)[0]
            attempt_path = attempts_dir / f"{prefix}.json"
            if attempt_path.exists():
                continue
            preflight = _read_json(preflight_path)
            response_path = journal_dir / f"{prefix}_response.json"
            if response_path.exists():
                response_record = _read_json(response_path)
                charge = self.ledger.finalize(
                    str(response_record["call_id"]),
                    response_record.get("usage"),
                    str(response_record["outcome"]),
                )
                _atomic_json(attempt_path, {**response_record, "charge": charge, "recovered": True})
            else:
                # A hard kill could have occurred before send or after the provider accepted the
                # request.  Do not repeat it automatically and retain the full reservation as spend.
                charge = self.ledger.finalize(
                    str(preflight["call_id"]), None, "interrupted_unknown_usage"
                )
                _atomic_json(attempt_path, {
                    **preflight,
                    "elapsed_seconds": None,
                    "response": None,
                    "usage": None,
                    "charge": charge,
                    "outcome": "interrupted_unknown_usage",
                    "error": "process ended after reservation and before durable response journal",
                    "parsed_output": None,
                    "recovered": True,
                })
