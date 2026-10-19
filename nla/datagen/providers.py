"""Completion provider backends for Stage 2 (API explanation generation).

Stage 2 calls an external LLM to produce natural-language explanations of
source text — these become the `response` column for AV-SFT and the `prompt`
content for AR-SFT. `CompletionProvider` is the pluggable interface: stage 2
code hands it a batch of fully-formed prompts and gets back a batch of
completions. Concurrency, retries, rate limits, and auth are all the
provider's problem.

Swap via `--provider-cls my.module.MyProvider` at stage2 invocation.
"""

import os
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor

import anthropic


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
    """Default provider: Anthropic Messages API with bounded THREAD concurrency.

    SYNC implementation. A single persistent `anthropic.Anthropic` client (an
    httpx *sync* connection pool) is created lazily and REUSED for every
    complete() call; concurrency within a batch comes from a ThreadPoolExecutor.

    This deliberately replaces the earlier asyncio.run()-per-call design. That
    version opened a fresh event loop on every batch, and the AsyncAnthropic/httpx
    pool bound to each now-closed loop orphaned its ESTAB connections. Across the
    hundreds of leakage/grader batches in an E0+CF1 run those orphans accumulated
    (observed: proxy connection count climbing 15 -> 35+ while the run made no
    progress) until the shared forward proxy choked and new requests hung. A
    persistent sync client keep-alives a small pool and never orphans it, so the
    connection count stays flat across the whole run. (`httpx.Client` is
    thread-safe for concurrent request sends, which the executor relies on.)

    The SDK handles transport-level retries (408/429/5xx, exponential backoff
    with jitter, respects Retry-After). Per-prompt failures after exhausting
    retries — and per-request timeouts — return None (caller drops the row).
    ONLY RateLimitError, server-side 5xx, and connection/timeout errors are
    tolerated; anything else (auth, bad request, unexpected content) still raises,
    because those are code bugs, not transient.
    """

    # Exceptions from which we degrade to None instead of killing the batch.
    # APIConnectionError covers APITimeoutError (the per-request timeout below),
    # so a stuck tunnel fails fast as a dropped row rather than hanging the batch.
    _TOLERATED = (
        anthropic.RateLimitError,
        anthropic.InternalServerError,
        anthropic.APIConnectionError,
    )

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 300,
        temperature: float = 1.0,
        concurrency: int = 32,
        max_retries: int = 10,
        timeout: float = 60.0,
    ):
        # Concurrency may need throttling when egress goes through a shared
        # forward proxy (set NLA_GRADER_CONCURRENCY). The sync client is built
        # ONCE (lazily) and reused, so connections are pooled, not churned.
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.concurrency = max(1, int(os.environ.get("NLA_GRADER_CONCURRENCY", concurrency)))
        self.max_retries = max_retries
        self.timeout = timeout
        self._client = None  # lazy, persistent across complete() calls

    def _client_handle(self):
        if self._client is None:
            self._client = anthropic.Anthropic(
                max_retries=self.max_retries, timeout=self.timeout)
        return self._client

    def _one(self, prompt: str):
        """Run one prompt. Returns (status, text) with status in
        {'ok','refused','fail'}; raises on non-tolerated (code-bug) errors."""
        client = self._client_handle()
        try:
            resp = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}],
            )
        except self._TOLERATED:
            return ("fail", None)
        # refusal: source text tripped safety — no answer coming, drop this row.
        if resp.stop_reason == "refusal":
            return ("refused", None)
        assert resp.stop_reason in ("end_turn", "max_tokens"), (
            f"unexpected stop_reason={resp.stop_reason!r} (want end_turn/max_tokens/refusal)"
        )
        assert len(resp.content) == 1 and resp.content[0].type == "text", (
            f"expected single text block, got {[b.type for b in resp.content]}"
        )
        text = resp.content[0].text.strip()
        assert text, "empty completion — refusing to emit blank explanation"
        return ("ok", text)

    def complete(self, prompts: list[str]) -> list[str | None]:
        if not prompts:
            return []
        workers = min(self.concurrency, len(prompts))
        # ThreadPoolExecutor.map preserves input order. A non-tolerated exception
        # in _one propagates here (loud), matching the old gather() re-raise.
        with ThreadPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(self._one, prompts))
        out: list[str | None] = []
        n_failed = 0
        n_refused = 0
        for status, text in results:
            if status == "ok":
                out.append(text)
            elif status == "refused":
                n_refused += 1
                out.append(None)
            else:
                n_failed += 1
                out.append(None)
        if n_failed or n_refused:
            print(f"  [AnthropicProvider] dropped {n_refused} refused + {n_failed} retry-exhausted of {len(prompts)}")
        return out
