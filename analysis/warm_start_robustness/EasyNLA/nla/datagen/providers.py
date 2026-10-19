"""Completion provider backends for Stage 2 (API explanation generation).

Stage 2 calls an external LLM to produce natural-language explanations of
source text — these become the `response` column for AV-SFT and the `prompt`
content for AR-SFT. `CompletionProvider` is the pluggable interface: stage 2
code hands it a batch of fully-formed prompts and gets back a batch of
completions. Concurrency, retries, rate limits, and auth are all the
provider's problem.

Swap via `--provider-cls my.module.MyProvider` at stage2 invocation.
"""

import asyncio
import fcntl
import hashlib
import json
import math
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path

import anthropic
import httpx


class CompletionProvider(ABC):
    """Submit a batch of prompts, get a batch of completions back.

    Stage 2 formats NLA-specific instruction prompts; the provider just maps
    `prompts[i] -> completion[i]` (or None for prompts that exhausted retries).
    A robust sampling engine can be plugged in by wrapping it in a subclass.

    None returns are per-prompt gave-up signals — stage2 drops those rows
    (same path as failed-extract-pattern). This means a chunk can survive
    losing a few prompts to sustained 429/500 storms instead of discarding
    511 good completions because one failed. Gaps ARE tracked: stage2 logs
    a drop count, and the parquet row count tells you exactly how many
    survived.
    """

    @abstractmethod
    def complete(self, prompts: list[str]) -> list[str | None]: ...


class AnthropicProvider(CompletionProvider):
    """Default provider: Anthropic Messages API with bounded async concurrency.

    The SDK handles transport-level retries (408/429/5xx, exponential backoff
    with jitter, respects Retry-After). High `max_retries` extends the retry
    window for sustained rate-limit storms — at max_retries=100 the SDK will
    keep backing off for minutes before giving up on one prompt.

    Per-prompt failures after exhausting retries return None (caller drops
    the row). `gather(return_exceptions=True)` collects these without nuking
    the whole batch — otherwise one stubborn 429 in a chunk of 512 wastes
    the other 511 API calls. ONLY `RateLimitError` and server-side 5xx are
    tolerated; anything else (auth, bad request, unexpected content) still
    raises — those are code bugs, not transient.

    Calls `asyncio.run()` — do not invoke from inside a running event loop.
    Stage 2 is a standalone CLI, so this is fine in practice.
    """

    # Exceptions from which we degrade to None instead of killing the batch.
    # Anything NOT in this tuple is a code bug and should still blow up loud.
    _TOLERATED = (
        anthropic.RateLimitError,
        anthropic.InternalServerError,
        anthropic.APIConnectionError,
        # 529: transient server overload (esp. at high concurrency). getattr:
        # OverloadedError is top-level in current SDKs but absent in older ones
        # (the dep is unpinned); old SDKs map 529 -> InternalServerError, already
        # tolerated above, so the fallback is a harmless duplicate — NOT
        # APIStatusError, which would silently swallow auth/param errors.
        getattr(anthropic, "OverloadedError", anthropic.InternalServerError),
    )

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 300,
        temperature: float = 1.0,
        concurrency: int = 32,
        max_retries: int = 10,
    ):
        self.client = anthropic.AsyncAnthropic(max_retries=max_retries)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.concurrency = concurrency

    async def _one(self, sem: asyncio.Semaphore, prompt: str) -> str | None:
        # temperature is deprecated/rejected by some newer models (e.g. opus-4-8);
        # pass it only when explicitly set so those models can be used as judges.
        kwargs = {} if self.temperature is None else {"temperature": self.temperature}
        async with sem:
            # SDK overloads reject a plain message dict + **kwargs, but both are
            # valid at runtime; constructing the exact TypedDicts isn't worth it.
            resp = await self.client.messages.create(  # type: ignore[call-overload]
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            )
        # refusal: source text tripped safety — no answer coming, drop this row.
        # content may be [] or the refusal message; either way, no explanation.
        if resp.stop_reason == "refusal":
            return None
        assert resp.stop_reason in ("end_turn", "max_tokens"), (
            f"unexpected stop_reason={resp.stop_reason!r} (want end_turn/max_tokens/refusal)"
        )
        assert len(resp.content) == 1 and resp.content[0].type == "text", (
            f"expected single text block, got {[b.type for b in resp.content]}"
        )
        text = resp.content[0].text.strip()
        assert text, "empty completion — refusing to emit blank explanation"
        return text

    def complete(self, prompts: list[str]) -> list[str | None]:
        async def _run() -> list[str | None | BaseException]:
            sem = asyncio.Semaphore(self.concurrency)
            return await asyncio.gather(
                *(self._one(sem, p) for p in prompts),
                return_exceptions=True,
            )

        raw = asyncio.run(_run())
        out: list[str | None] = []
        n_failed = 0
        n_refused = 0
        for i, r in enumerate(raw):
            if isinstance(r, str):
                out.append(r)
            elif r is None:
                n_refused += 1
                out.append(None)
            elif isinstance(r, self._TOLERATED):
                n_failed += 1
                out.append(None)
            elif isinstance(r, BaseException):
                # Not a transient — auth/schema/code bug. Blow up loud.
                raise r
            else:
                raise AssertionError(f"gather returned unexpected type at [{i}]: {type(r).__name__}")
        if n_failed or n_refused:
            print(f"  [AnthropicProvider] dropped {n_refused} refused + {n_failed} retry-exhausted of {len(prompts)}")
        # >20% retry-exhausted = systemic (network/rate-limit/key quota), not
        # per-row content. The old ALL-failed check let a 99%-hollow chunk be
        # written as complete — and resume never revisits completed chunks.
        # (Refusals are excluded: those are per-row content decisions.)
        if prompts and n_failed / len(prompts) > 0.2:
            raise RuntimeError(
                f"AnthropicProvider: {n_failed}/{len(prompts)} requests failed "
                f"after retries (>20%) — systemic failure, refusing to emit a "
                f"hollow chunk (it would be marked complete and never retried)."
            )
        return out


class BatchAnthropicProvider(CompletionProvider):
    """Anthropic Message Batches API — 50% cheaper than sync, async, async-style turnaround.

    Each `complete(prompts)` call partitions prompts into sub-batches of
    `max_batch_size` (Anthropic's hard limit is 100k per batch), submits all
    sub-batches concurrently, polls until each one ends, then fetches results.
    The order of returned completions matches the input order via the `custom_id`
    round-trip; missing/failed/expired requests come back as None.

    Authentication: reads `ANTHROPIC_API_KEY_BATCH` first (so the dedicated batch
    key from CLAUDE.md is preferred), falls back to `ANTHROPIC_API_KEY`. The key
    is passed explicitly to the client so the env var name is unambiguous.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 300,
        temperature: float = 1.0,
        max_batch_size: int = 50_000,
        poll_interval_s: float = 30.0,
    ):
        api_key = os.environ.get("ANTHROPIC_API_KEY_BATCH") or os.environ.get("ANTHROPIC_API_KEY")
        assert api_key, "set ANTHROPIC_API_KEY_BATCH (preferred) or ANTHROPIC_API_KEY"
        self.client = anthropic.AsyncAnthropic(api_key=api_key, max_retries=10)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_batch_size = max_batch_size
        self.poll_interval_s = poll_interval_s

    async def _submit_subbatch(self, prompts: list[str], offset: int) -> list[str | None]:
        # Encode position within the sub-batch in custom_id so results map back
        # even when fetched out-of-order. Sub-batch-local index, not global.
        requests = [
            {
                "custom_id": f"r{i:07d}",
                "params": {
                    "model": self.model,
                    "max_tokens": self.max_tokens,
                    # omit when None — some newer models reject the param
                    # (same convention as AnthropicProvider._one)
                    **({} if self.temperature is None
                       else {"temperature": self.temperature}),
                    "messages": [{"role": "user", "content": p}],
                },
            }
            for i, p in enumerate(prompts)
        ]
        # batch API accepts request dicts at runtime; SDK types want Request objs.
        batch = await self.client.messages.batches.create(requests=requests)  # type: ignore[arg-type]
        bid = batch.id
        print(f"  [batch {bid}] submitted {len(prompts)} prompts (sub-batch offset={offset})")

        # Poll until ended. Anthropic's batch lifecycle: in_progress → ended.
        # The terminal status field is `processing_status`; per-request status
        # comes back in the streamed results.
        t0 = time.monotonic()
        while True:
            await asyncio.sleep(self.poll_interval_s)
            b = await self.client.messages.batches.retrieve(bid)
            if b.processing_status == "ended":
                rc = b.request_counts
                elapsed = time.monotonic() - t0
                print(
                    f"  [batch {bid}] ended in {elapsed/60:.1f}min — "
                    f"succeeded={rc.succeeded} errored={rc.errored} "
                    f"canceled={rc.canceled} expired={rc.expired}"
                )
                break

        out: list[str | None] = [None] * len(prompts)
        first_err = None  # sample failure detail for the all-failed diagnostic
        # SDK quirk: messages.batches.results() returns a coroutine that resolves
        # to an async-iterable. Must await first, then `async for`.
        results_stream = await self.client.messages.batches.results(bid)
        async for entry in results_stream:
            # custom_id "r0000007" -> 7
            idx = int(entry.custom_id[1:])
            r = entry.result
            if r.type != "succeeded":
                # errored/canceled/expired stay None (per-row tolerance), but keep
                # a sample so a SYSTEMIC failure (bad key, invalid params -> every
                # row errored) can be raised below instead of silently producing a
                # 100%-empty chunk that gets marked complete and poisons resume.
                if first_err is None:
                    first_err = f"{r.type}: {getattr(r, 'error', '')!s:.300}"
                continue
            msg = r.message
            if msg.stop_reason == "refusal":
                continue
            if not msg.content or msg.content[0].type != "text":
                continue
            text = msg.content[0].text.strip()
            if text:
                out[idx] = text
        if prompts and all(o is None for o in out) and first_err is not None:
            raise RuntimeError(
                f"[batch {bid}] 0/{len(prompts)} requests succeeded — systemic "
                f"failure (auth/params), refusing to emit an empty chunk. "
                f"First error: {first_err}"
            )
        return out

    def complete(self, prompts: list[str]) -> list[str | None]:
        async def _run() -> list[str | None]:
            sub_batches = [
                (i, prompts[i : i + self.max_batch_size])
                for i in range(0, len(prompts), self.max_batch_size)
            ]
            results_per_sub = await asyncio.gather(
                *(self._submit_subbatch(sb, offset=off) for off, sb in sub_batches),
                return_exceptions=False,
            )
            out: list[str | None] = []
            for r in results_per_sub:
                out.extend(r)
            return out

        return asyncio.run(_run())


class OpenAIBatchProvider(CompletionProvider):
    """Crash-resumable OpenAI Batch API backend with a hard spend guard.

    Each call is content-addressed by the exact prompt list and model settings.
    The input JSONL, uploaded file ID, batch ID, raw output, per-request errors,
    and token/cost ledger are saved below ``cache_dir``. Re-running the same
    stage-2 chunk therefore polls or reads the existing batch instead of
    submitting duplicate paid work.

    The default prices are the published GPT-5 nano Batch API prices as of
    2026-08-27. They are constructor arguments rather than hidden constants so
    a different model cannot silently inherit the wrong accounting. The guard
    combines completed actual usage with a pessimistic estimate for the next
    batch before submission; it refuses the batch if that sum exceeds
    ``spend_cap_usd``.
    """

    _TERMINAL = {"completed", "failed", "expired", "cancelled"}

    def __init__(
        self,
        model: str = "gpt-5-nano",
        max_tokens: int = 300,
        reasoning_effort: str | None = "minimal",
        verbosity: str | None = None,
        cache_dir: str = "openai_batch_cache",
        api_key_env: str = "OPENAI_API_KEY_2",
        poll_interval_s: float = 30.0,
        request_timeout_s: float = 180.0,
        terminal_batch_retries: int = 0,
        spend_cap_usd: float = 100.0,
        project_spend_cap_usd: float | None = None,
        known_external_spend_usd: float = 0.0,
        reserved_downstream_spend_usd: float = 0.0,
        minimum_prior_cache_spend_usd: float = 0.0,
        batch_input_per_million_usd: float = 0.05,
        batch_cached_input_per_million_usd: float = 0.005,
        batch_output_per_million_usd: float = 0.40,
    ):
        api_key = os.environ.get(api_key_env)
        assert api_key, f"set {api_key_env} for OpenAIBatchProvider"
        if terminal_batch_retries != 0:
            raise ValueError(
                "automatic terminal Batch retries are disabled: failed, expired, or "
                "cancelled batches may contain billable partial successes. Keep the "
                "original pessimistic reservation and reconcile partial usage before "
                "any manual resubmission."
            )
        assert spend_cap_usd > 0, "spend_cap_usd must be positive"
        assert known_external_spend_usd >= 0, "known_external_spend_usd must be nonnegative"
        assert reserved_downstream_spend_usd >= 0, "reserved_downstream_spend_usd must be nonnegative"
        assert minimum_prior_cache_spend_usd >= 0, "minimum_prior_cache_spend_usd must be nonnegative"
        if project_spend_cap_usd is not None:
            assert project_spend_cap_usd > 0, "project_spend_cap_usd must be positive"
            assert (
                spend_cap_usd + known_external_spend_usd + reserved_downstream_spend_usd
                <= project_spend_cap_usd
            ), (
                "cache spend cap + known external spend + downstream reserve exceeds "
                f"project cap: ${spend_cap_usd:.6f} + ${known_external_spend_usd:.6f} "
                f"+ ${reserved_downstream_spend_usd:.6f} > ${project_spend_cap_usd:.6f}"
            )
        self.model = model
        self.max_tokens = max_tokens
        # Stage-2 records provider.temperature in the sidecar. GPT-5 nano does
        # not use a temperature parameter here, so make that explicit.
        self.temperature = None
        self.reasoning_effort = reasoning_effort
        self.verbosity = verbosity
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.poll_interval_s = poll_interval_s
        self.terminal_batch_retries = terminal_batch_retries
        self.spend_cap_usd = spend_cap_usd
        self.project_spend_cap_usd = project_spend_cap_usd
        self.known_external_spend_usd = known_external_spend_usd
        self.reserved_downstream_spend_usd = reserved_downstream_spend_usd
        self.minimum_prior_cache_spend_usd = minimum_prior_cache_spend_usd
        self.input_price = batch_input_per_million_usd
        self.cached_input_price = batch_cached_input_per_million_usd
        self.output_price = batch_output_per_million_usd
        self.client = httpx.Client(
            base_url="https://api.openai.com/v1",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=request_timeout_s,
        )

    @staticmethod
    def _atomic_json(path: Path, value: dict) -> None:
        OpenAIBatchProvider._atomic_bytes(
            path, json.dumps(value, indent=2, sort_keys=True).encode()
        )

    @staticmethod
    def _atomic_bytes(path: Path, value: bytes) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        parent_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)

    def _request_body(self, prompt: str) -> dict:
        body: dict = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": self.max_tokens,
        }
        if self.reasoning_effort is not None:
            body["reasoning_effort"] = self.reasoning_effort
        if self.verbosity is not None:
            body["verbosity"] = self.verbosity
        return body

    def _request_lines(self, prompts: list[str]) -> list[str]:
        return [
            json.dumps({
                "custom_id": f"r{i:07d}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": self._request_body(prompt),
            }, ensure_ascii=False, separators=(",", ":"))
            for i, prompt in enumerate(prompts)
        ]

    def _key(self, lines: list[str]) -> str:
        h = hashlib.sha256()
        for line in lines:
            h.update(line.encode("utf-8"))
            h.update(b"\n")
        return h.hexdigest()

    def _actual_spend(self) -> float:
        total = 0.0
        for p in self.cache_dir.glob("*/usage.json"):
            try:
                row = json.loads(p.read_text())
                if "cost_usd" not in row:
                    raise ValueError("missing cost_usd")
                cost = float(row["cost_usd"])
                if not math.isfinite(cost) or cost < 0:
                    raise ValueError(f"invalid cost_usd={cost!r}")
                total += cost
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                # A torn or hand-edited ledger must stop spending, not make the
                # guard undercount. The caller can inspect/fix the named file.
                raise RuntimeError(f"cannot read OpenAI spend ledger {p}")
        return total

    def _committed_spend(self) -> float:
        """Pessimistic reservations for submitted/incomplete batch directories."""
        total = 0.0
        for state_path in self.cache_dir.glob("*/state.json"):
            if (state_path.parent / "usage.json").exists():
                continue
            try:
                state = json.loads(state_path.read_text())
                if "pessimistic_cost_usd" not in state:
                    raise ValueError("missing pessimistic_cost_usd")
                reservation = float(state["pessimistic_cost_usd"])
                if not math.isfinite(reservation) or reservation < 0:
                    raise ValueError(f"invalid reservation={reservation!r}")
                total += reservation
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                raise RuntimeError(f"cannot read OpenAI batch reservation {state_path}")
        return total

    def _reserve_spend(self, state_path: Path, state: dict, estimate: float) -> dict:
        """Atomically reserve worst-case spend across concurrent stage-2 jobs."""
        lock_path = self.cache_dir / ".spend.lock"
        with lock_path.open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            # Another process may have reserved this exact content-addressed
            # batch while we waited for the lock.
            if state_path.exists():
                state = json.loads(state_path.read_text())
            if "pessimistic_cost_usd" not in state:
                actual = self._actual_spend()
                if actual + 1e-9 < self.minimum_prior_cache_spend_usd:
                    raise RuntimeError(
                        f"OpenAI cache ledger reports ${actual:.6f}, below the required "
                        f"prior-spend floor ${self.minimum_prior_cache_spend_usd:.6f}. "
                        "The shared API cache is incomplete or the wrong cache_dir was "
                        "selected; refusing to submit because project spend would be undercounted."
                    )
                committed = self._committed_spend()
                if actual + committed + estimate > self.spend_cap_usd:
                    raise RuntimeError(
                        f"OpenAI spend guard: completed ${actual:.4f} + reserved "
                        f"${committed:.4f} + next-batch pessimistic ${estimate:.4f} "
                        f"exceeds ${self.spend_cap_usd:.2f}. No request was submitted."
                    )
                state["pessimistic_cost_usd"] = estimate
                self._atomic_json(state_path, state)
                self._write_cumulative_ledger_locked()
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        return state

    def _write_cumulative_ledger_locked(self) -> dict:
        """Write a cache-wide ledger while ``.spend.lock`` is held."""
        usage_rows = []
        for path in sorted(self.cache_dir.glob("*/usage.json")):
            try:
                row = json.loads(path.read_text())
                cost = float(row["cost_usd"])
                if not math.isfinite(cost) or cost < 0:
                    raise ValueError(f"invalid cost_usd={cost!r}")
                for field in (
                    "requests_succeeded",
                    "requests_failed",
                    "input_tokens",
                    "cached_input_tokens",
                    "output_tokens",
                ):
                    if int(row[field]) < 0:
                        raise ValueError(f"negative {field}")
            except (OSError, KeyError, ValueError, TypeError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"cannot read OpenAI spend ledger {path}") from exc
            usage_rows.append(row)
        actual = sum(float(row.get("cost_usd", 0.0)) for row in usage_rows)
        committed = self._committed_spend()
        project_actual = actual + self.known_external_spend_usd
        value = {
            "schema_version": 1,
            "batch_count": len(usage_rows),
            "requests_succeeded": sum(int(row.get("requests_succeeded", 0)) for row in usage_rows),
            "requests_failed": sum(int(row.get("requests_failed", 0)) for row in usage_rows),
            "input_tokens": sum(int(row.get("input_tokens", 0)) for row in usage_rows),
            "cached_input_tokens": sum(int(row.get("cached_input_tokens", 0)) for row in usage_rows),
            "output_tokens": sum(int(row.get("output_tokens", 0)) for row in usage_rows),
            "cache_actual_spend_usd": actual,
            "cache_committed_worst_case_usd": committed,
            "cache_spend_cap_usd": self.spend_cap_usd,
            "known_external_spend_usd": self.known_external_spend_usd,
            "project_actual_spend_usd": project_actual,
            "reserved_downstream_spend_usd": self.reserved_downstream_spend_usd,
            "project_spend_cap_usd": self.project_spend_cap_usd,
            "maximum_project_spend_after_commitments_usd": (
                project_actual + committed + self.reserved_downstream_spend_usd
            ),
            "updated_at_unix": time.time(),
        }
        self._atomic_json(self.cache_dir / "cumulative_usage.json", value)
        return value

    def _write_cumulative_ledger(self) -> dict:
        lock_path = self.cache_dir / ".spend.lock"
        with lock_path.open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            value = self._write_cumulative_ledger_locked()
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        return value

    def _pessimistic_cost(self, lines: list[str], n_prompts: int) -> float:
        # Byte-level tokenizers cannot emit more ordinary text tokens than
        # input bytes; add 128 tokens/request for chat framing that is not a
        # literal part of the serialized JSONL. This is intentionally much
        # more conservative than bytes/4: a spend *cap* must use an upper bound,
        # not a typical-tokenization estimate. Output uses the hard API cap.
        input_tokens_upper = sum(len(line.encode("utf-8")) for line in lines) + (
            128 * n_prompts
        )
        output_tokens_max = n_prompts * self.max_tokens
        return (
            input_tokens_upper * self.input_price
            + output_tokens_max * self.output_price
        ) / 1_000_000

    def _checked_json(self, response: httpx.Response, action: str) -> dict:
        if response.is_error:
            raise RuntimeError(
                f"OpenAI {action} failed: HTTP {response.status_code}: "
                f"{response.text[:1000]}"
            )
        value = response.json()
        assert isinstance(value, dict), f"OpenAI {action} returned non-object JSON"
        return value

    def _save_usage(self, run_dir: Path, records: list[dict]) -> dict:
        prompt = completion = cached = 0
        succeeded = failed = 0
        for record in records:
            response = record.get("response") or {}
            if response.get("status_code") != 200:
                failed += 1
                continue
            succeeded += 1
            usage = (response.get("body") or {}).get("usage") or {}
            prompt += int(usage.get("prompt_tokens", 0) or 0)
            completion += int(usage.get("completion_tokens", 0) or 0)
            details = usage.get("prompt_tokens_details") or {}
            cached += int(details.get("cached_tokens", 0) or 0)
        uncached = max(0, prompt - cached)
        cost = (
            uncached * self.input_price
            + cached * self.cached_input_price
            + completion * self.output_price
        ) / 1_000_000
        ledger = {
            "model": self.model,
            "requests_succeeded": succeeded,
            "requests_failed": failed,
            "input_tokens": prompt,
            "cached_input_tokens": cached,
            "output_tokens": completion,
            "batch_input_per_million_usd": self.input_price,
            "batch_cached_input_per_million_usd": self.cached_input_price,
            "batch_output_per_million_usd": self.output_price,
            "cost_usd": cost,
        }
        self._atomic_json(run_dir / "usage.json", ledger)
        self._write_cumulative_ledger()
        return ledger

    def _parse_output(self, run_dir: Path, n_prompts: int) -> list[str | None]:
        output_path = run_dir / "output.jsonl"
        records = [json.loads(line) for line in output_path.read_text().splitlines() if line.strip()]
        out: list[str | None] = [None] * n_prompts
        for record in records:
            custom_id = record.get("custom_id", "")
            if not custom_id.startswith("r"):
                continue
            idx = int(custom_id[1:])
            assert 0 <= idx < n_prompts, f"bad OpenAI batch custom_id={custom_id!r}"
            response = record.get("response") or {}
            if response.get("status_code") != 200:
                continue
            choices = ((response.get("body") or {}).get("choices") or [])
            if not choices:
                continue
            content = (choices[0].get("message") or {}).get("content")
            if isinstance(content, str) and content.strip():
                out[idx] = content.strip()
        ledger = self._save_usage(run_dir, records)
        print(
            f"  [OpenAIBatchProvider] {ledger['requests_succeeded']}/{n_prompts} "
            f"succeeded; batch cost ${ledger['cost_usd']:.4f}; "
            f"cache total ${self._actual_spend():.4f}/${self.spend_cap_usd:.2f}",
            flush=True,
        )
        return out

    def complete(self, prompts: list[str]) -> list[str | None]:
        if not prompts:
            return []
        lines = self._request_lines(prompts)
        key = self._key(lines)
        run_dir = self.cache_dir / key
        run_dir.mkdir(parents=True, exist_ok=True)
        input_path = run_dir / "input.jsonl"
        state_path = run_dir / "state.json"
        output_path = run_dir / "output.jsonl"
        if not input_path.exists():
            self._atomic_bytes(input_path, ("\n".join(lines) + "\n").encode("utf-8"))
        else:
            assert input_path.read_text() == "\n".join(lines) + "\n", (
                f"hash collision or corrupted batch input at {input_path}"
            )
        if output_path.exists():
            return self._parse_output(run_dir, len(prompts))

        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        estimate = self._pessimistic_cost(lines, len(prompts))
        state = self._reserve_spend(state_path, state, estimate)
        if "input_file_id" not in state:
            if input_path.stat().st_size > 190_000_000:
                raise RuntimeError(
                    f"OpenAI batch input is {input_path.stat().st_size / 1e6:.1f} MB; "
                    "refusing to approach the API's 200 MB limit. Reduce stage2.chunk_size."
                )
            with input_path.open("rb") as fh:
                response = self.client.post(
                    "/files",
                    data={"purpose": "batch"},
                    files={"file": (input_path.name, fh, "application/jsonl")},
                )
            uploaded = self._checked_json(response, "file upload")
            state.update({"input_file_id": uploaded["id"], "status": "uploaded"})
            self._atomic_json(state_path, state)

        if "batch_id" not in state:
            response = self.client.post("/batches", json={
                "input_file_id": state["input_file_id"],
                "endpoint": "/v1/chat/completions",
                "completion_window": "24h",
                "metadata": {"purpose": "nla-warm-start", "content_sha256": key},
            })
            batch = self._checked_json(response, "batch creation")
            state.update({"batch_id": batch["id"], "status": batch["status"]})
            self._atomic_json(state_path, state)
            print(f"  [OpenAIBatchProvider] submitted {len(prompts)} prompts as {batch['id']}", flush=True)

        while state.get("status") not in self._TERMINAL:
            time.sleep(self.poll_interval_s)
            response = self.client.get(f"/batches/{state['batch_id']}")
            batch = self._checked_json(response, "batch retrieval")
            old_status = state.get("status")
            state.update({
                "status": batch["status"],
                "output_file_id": batch.get("output_file_id"),
                "error_file_id": batch.get("error_file_id"),
                "request_counts": batch.get("request_counts"),
            })
            self._atomic_json(state_path, state)
            if state["status"] != old_status:
                print(f"  [OpenAIBatchProvider] {state['batch_id']} → {state['status']}", flush=True)

        if state["status"] != "completed" or not state.get("output_file_id"):
            # Preserve any partial successes/errors before stopping. They are
            # evidence for later billable-usage reconciliation, not a signal to
            # release the pessimistic reservation.
            if state.get("output_file_id"):
                partial = self.client.get(f"/files/{state['output_file_id']}/content")
                if not partial.is_error:
                    self._atomic_bytes(run_dir / "output.terminal.jsonl", partial.content)
            if state.get("error_file_id"):
                errors = self.client.get(f"/files/{state['error_file_id']}/content")
                if not errors.is_error:
                    self._atomic_bytes(run_dir / "errors.terminal.jsonl", errors.content)
            # Do not create usage.json or clear pessimistic_cost_usd here. A
            # terminal Batch may have billable successful requests even when no
            # complete output is available. Leaving the reservation outstanding
            # is deliberately conservative and prevents a later chunk/manual
            # retry from spending through the project cap. Reconciliation must
            # download and account the terminal Batch's partial output first.
            raise RuntimeError(
                f"OpenAI batch {state['batch_id']} ended with status={state['status']!r}; "
                f"state and pessimistic reservation retained at {state_path}; automatic "
                "resubmission is disabled until partial billable usage is reconciled"
            )
        response = self.client.get(f"/files/{state['output_file_id']}/content")
        if response.is_error:
            raise RuntimeError(
                f"OpenAI output download failed: HTTP {response.status_code}: {response.text[:1000]}"
            )
        self._atomic_bytes(output_path, response.content)
        if state.get("error_file_id"):
            err = self.client.get(f"/files/{state['error_file_id']}/content")
            if not err.is_error:
                self._atomic_bytes(run_dir / "errors.jsonl", err.content)
        return self._parse_output(run_dir, len(prompts))
