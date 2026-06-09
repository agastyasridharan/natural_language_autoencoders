"""Neuronpedia NLA API client — drive the hosted Natural Language Autoencoders.

The Neuronpedia NLA demo (https://www.neuronpedia.org/<model>/nla) hosts the
trained open-model NLAs from this repo behind a small JSON / Server-Sent-Events
API. This module is a thin, dependency-light client for that API so the paper's
experiments can be replicated against the hosted activation verbalizers without
a local GPU.

The backend is the same AV/AR pair documented in ``nla_inference.py`` and the
paper, just hosted:

  * ``/api/nla/explain`` runs the **Activation Verbalizer** (AV) — residual
    activation -> text — and the **Activation Reconstructor** (AR) — text ->
    activation — then returns both the verbalization *and* the reconstruction
    fidelity (cosine similarity + MSE). The fidelity is exactly the quantity the
    AV was RL-trained to maximize.
  * ``/api/nla/completion`` tokenizes a transcript and, on request, runs the
    *target* model to generate a reply (the "what is the model thinking" chat).

Endpoints (reverse-engineered from the demo frontend bundle):

  GET  /api/nla/sources       hosted (modelId, nlaSourceId, layer) registry
  POST /api/nla/completion    tokenize a transcript; optionally generate a reply
  POST /api/nla/explain       verbalize + reconstruct chosen token positions
  GET  /api/nla/cache/<id>    load a saved/shared run (the paper's curated
                              examples are addressable by their cacheId)

Auth is an ``x-api-key`` header; the key is read from ``$NEURONPEDIA_API_KEY``
or the nearest ``.env`` file. Only standard libs + ``httpx`` (already a repo
dependency) are used.

The chat templates below are replicated verbatim from the demo frontend so a
teacher-forced transcript built here tokenizes identically to one typed into the
web UI (the server adds the BOS token; ``format_chat`` must not).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

import httpx

BASE_URL = "https://www.neuronpedia.org"

# The /api/nla/explain endpoint rejects requests with more than this many *new*
# (uncached) token positions; the client batches around it.
MAX_POSITIONS_PER_REQUEST = 16

# Hosted NLAs, as returned by GET /api/nla/sources (Apr 2026). The client will
# refresh this live via ``NeuronpediaNLA.sources()``; this is a convenience map
# so demos can name a model without a network round-trip.
HOSTED = {
    "llama3.3-70b-it": {"source_id": "kitft-l53", "layer": 53},
    "gemma-3-27b-it": {"source_id": "kitft-l41", "layer": 41},
}

# Chat templates, replicated from the demo frontend's ``formatChat``. Each family
# closes every turn with ``turn_end`` EXCEPT the final turn, which is left open
# so activations at the last token stay on-distribution. The BOS token is added
# server-side, so it is intentionally absent here.
_FAMILIES = {
    "llama3": {
        "match": ("llama",),
        "assistant": "assistant",
        "turn_end": "<|eot_id|>",
        "single": lambda role, content: f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}",
    },
    "gemma3": {
        "match": ("gemma",),
        "assistant": "model",
        "turn_end": "<end_of_turn>\n",
        "single": lambda role, content: f"<start_of_turn>{role}\n{content}",
    },
    "qwen": {
        "match": ("qwen",),
        "assistant": "assistant",
        "turn_end": "<|im_end|>\n",
        "single": lambda role, content: f"<|im_start|>{role}\n{content}",
    },
}


# --------------------------------------------------------------------------- #
# Key loading
# --------------------------------------------------------------------------- #
def load_api_key(explicit: str | None = None) -> str:
    """Return the Neuronpedia API key.

    Resolution order: explicit arg -> $NEURONPEDIA_API_KEY -> the nearest ``.env``
    found by walking up from the cwd and from this file's directory.
    """
    if explicit:
        return explicit
    if os.environ.get("NEURONPEDIA_API_KEY"):
        return os.environ["NEURONPEDIA_API_KEY"]
    seen: set[Path] = set()
    for start in (Path.cwd(), Path(__file__).resolve().parent):
        for d in (start, *start.parents):
            if d in seen:
                continue
            seen.add(d)
            env = d / ".env"
            if env.is_file():
                for line in env.read_text().splitlines():
                    line = line.strip()
                    if line.startswith("NEURONPEDIA_API_KEY") and "=" in line:
                        return line.split("=", 1)[1].strip().strip("'\"")
    raise RuntimeError(
        "No API key: set NEURONPEDIA_API_KEY or add it to a .env file."
    )


# --------------------------------------------------------------------------- #
# Data types
# --------------------------------------------------------------------------- #
@dataclass
class Token:
    token: str
    token_id: int
    position: int

    @classmethod
    def from_json(cls, d: dict) -> "Token":
        return cls(token=d["token"], token_id=d["token_id"], position=d["position"])


@dataclass
class Verbalization:
    """One position's AV description plus its AR reconstruction fidelity."""

    position: int
    token: str
    description: str
    cosine_similarity: float
    mse: float
    l2_norm: float

    @classmethod
    def from_json(cls, d: dict) -> "Verbalization":
        return cls(
            position=d["position"],
            token=d.get("token", ""),
            description=d.get("description", ""),
            cosine_similarity=d.get("cosine_similarity", float("nan")),
            mse=d.get("mse", float("nan")),
            l2_norm=d.get("l2_norm", float("nan")),
        )


@dataclass
class ExplainResult:
    results: list[Verbalization]
    layer_index: int
    prompt_length: int
    cache_id: str | None = None

    def by_position(self) -> dict[int, Verbalization]:
        return {v.position: v for v in self.results}


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #
class NeuronpediaNLA:
    """Client for one hosted NLA (a single (model, source/layer) pair)."""

    def __init__(
        self,
        model_id: str = "llama3.3-70b-it",
        source_id: str | None = None,
        *,
        temperature: float = 0.4,
        api_key: str | None = None,
        base_url: str = BASE_URL,
        timeout: float = 120.0,
    ):
        self.model_id = model_id
        self.source_id = source_id or HOSTED.get(model_id, {}).get("source_id")
        if self.source_id is None:
            raise ValueError(
                f"Unknown model {model_id!r}; pass source_id explicitly or pick "
                f"one of {sorted(HOSTED)}."
            )
        self.temperature = temperature
        self.base_url = base_url.rstrip("/")
        self._key = load_api_key(api_key)
        self._client = httpx.Client(
            base_url=self.base_url,
            headers={"x-api-key": self._key, "Content-Type": "application/json"},
            timeout=timeout,
        )
        self._family = self._resolve_family(model_id)

    # -- lifecycle -------------------------------------------------------- #
    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "NeuronpediaNLA":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- registry --------------------------------------------------------- #
    @staticmethod
    def sources(api_key: str | None = None, base_url: str = BASE_URL) -> list[dict]:
        """List every hosted NLA: (modelId, nlaSourceId, layerNum, av, ar, ...)."""
        key = load_api_key(api_key)
        r = httpx.get(
            f"{base_url.rstrip('/')}/api/nla/sources",
            headers={"x-api-key": key},
            timeout=60.0,
        )
        r.raise_for_status()
        return r.json()["sources"]

    # -- chat formatting -------------------------------------------------- #
    @staticmethod
    def _resolve_family(model_id: str) -> dict:
        mid = model_id.lower()
        for fam in _FAMILIES.values():
            if any(tok in mid for tok in fam["match"]):
                return fam
        raise ValueError(f"No chat template known for model {model_id!r}.")

    def format_chat(self, messages: Sequence[dict]) -> str:
        """Render messages -> raw transcript string, matching the demo UI.

        ``messages`` is a list of ``{"role": ..., "content": ...}``. ``role`` may
        be ``"user"``/``"assistant"``/``"system"``; the assistant role is mapped
        per family ("model" for Gemma). The final turn is left unterminated.
        """
        fam = self._family
        out = []
        n = len(messages)
        for i, m in enumerate(messages):
            role = m["role"]
            if role == "assistant":
                role = fam["assistant"]
            chunk = fam["single"](role, m["content"])
            if i != n - 1:
                chunk += fam["turn_end"]
            out.append(chunk)
        return "".join(out)

    def _as_text(self, transcript: str | Sequence[dict]) -> str:
        return transcript if isinstance(transcript, str) else self.format_chat(transcript)

    # -- /api/nla/completion : tokenize ----------------------------------- #
    def tokenize(self, transcript: str | Sequence[dict]) -> list[Token]:
        """Tokenize a transcript into positioned tokens (no generation)."""
        text = self._as_text(transcript)
        r = self._client.post(
            "/api/nla/completion",
            json={
                "text": text,
                "completion_tokens": 0,
                "modelId": self.model_id,
                "nlaSourceId": self.source_id,
            },
        )
        payload = _check(r, "tokenize")
        return [Token.from_json(t) for t in payload["tokens"]]

    # -- /api/nla/completion : generate ----------------------------------- #
    def generate(
        self,
        messages: Sequence[dict],
        *,
        max_tokens: int = 128,
        temperature: float | None = None,
    ) -> str:
        """Run the *target model* on ``messages`` and return its reply text.

        This is the demo's "what is the model thinking" chat: it runs the real
        Llama/Gemma, not the NLA. Use it to obtain a model-generated transcript,
        then ``verbalize`` chosen positions of it.
        """
        text = self.format_chat(list(messages) + [{"role": "assistant", "content": ""}])
        body = {
            "text": text,
            "messages": list(messages),
            "completion_tokens": max_tokens,
            "temperature": self.temperature if temperature is None else temperature,
            "stream": True,
            "modelId": self.model_id,
            "nlaSourceId": self.source_id,
        }
        out: list[str] = []
        with self._client.stream("POST", "/api/nla/completion", json=body) as r:
            r.raise_for_status()
            for evt in _iter_sse(r):
                if evt.get("type") == "token" and isinstance(evt.get("token"), dict):
                    out.append(evt["token"].get("token", ""))
        reply = "".join(out)
        end = self._family["turn_end"].rstrip("\n")
        if reply.endswith(end):
            reply = reply[: -len(end)]
        return reply

    # -- /api/nla/explain : verbalize + reconstruct ----------------------- #
    def verbalize(
        self,
        transcript: str | Sequence[dict],
        positions: Iterable[int],
        *,
        temperature: float | None = None,
        prior_cache_id: str | None = None,
        tokens: Sequence[str] | None = None,
    ) -> ExplainResult:
        """Run the AV+AR on ``positions`` of ``transcript``.

        Returns an :class:`ExplainResult`; each :class:`Verbalization` carries the
        AV's ``description`` and the AR's ``cosine_similarity`` / ``mse`` — i.e.
        what the activation "says" and how faithfully that text reconstructs it.
        """
        text = self._as_text(transcript)
        temp = self.temperature if temperature is None else temperature
        pos = sorted(set(int(p) for p in positions))

        # The endpoint caps each request at MAX_POSITIONS_PER_REQUEST *new*
        # positions. Batch transparently, chaining priorCacheId so earlier
        # batches count as cached and don't eat into the next batch's quota.
        merged: list[Verbalization] = []
        layer_index = prompt_length = -1
        cache_id = prior_cache_id
        for i in range(0, len(pos), MAX_POSITIONS_PER_REQUEST):
            chunk = pos[i : i + MAX_POSITIONS_PER_REQUEST]
            body = {
                "text": text,
                "temperature": temp,
                "modelId": self.model_id,
                "nlaSourceId": self.source_id,
                "positions": chunk,
                "stream": False,
            }
            if cache_id:
                body["priorCacheId"] = cache_id
            if tokens is not None:
                body["tokens"] = list(tokens)
            r = self._client.post("/api/nla/explain", json=body)
            payload = _check(r, "explain")
            merged.extend(Verbalization.from_json(d) for d in payload.get("results", []))
            layer_index = payload.get("layer_index", layer_index)
            prompt_length = payload.get("prompt_length", prompt_length)
            cache_id = payload.get("cacheId", cache_id)
        return ExplainResult(
            results=merged,
            layer_index=layer_index,
            prompt_length=prompt_length,
            cache_id=cache_id,
        )

    # -- /api/nla/cache/<id> --------------------------------------------- #
    def load_cache(self, cache_id: str) -> dict:
        """Load a saved/shared run by cacheId (e.g. a curated paper example)."""
        r = self._client.get(f"/api/nla/cache/{cache_id}")
        r.raise_for_status()
        return r.json()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _check(r: httpx.Response, what: str) -> dict | None:
    """Raise a friendly error for rate-limit / upstream-down responses.

    Returns the parsed JSON body on success, or raises RuntimeError. The hosted
    NLA inference servers (per model) are sometimes cold or down; that surfaces
    as a 5xx or a ``{"error": "NLA server error: 50x ..."}`` body — distinct from
    a client mistake.
    """
    if r.status_code == 429:
        raise RuntimeError(
            f"{what}: rate limited (429). Wait a bit and retry — the demo enforces "
            "an hourly per-key limit."
        )
    body_preview = r.text[:300]
    if r.status_code >= 500 or "NLA server error" in body_preview:
        raise RuntimeError(
            f"{what}: the hosted NLA inference server is unavailable right now "
            f"(HTTP {r.status_code}). This is a Neuronpedia-side outage, not a "
            f"request error — try another --model or retry later.\n  {body_preview}"
        )
    if r.status_code >= 400:
        raise RuntimeError(f"{what} failed ({r.status_code}): {body_preview}")
    payload = r.json()
    if isinstance(payload, dict) and "error" in payload:
        raise RuntimeError(payload.get("detail") or payload["error"])
    return payload


def _iter_sse(response: httpx.Response):
    """Yield parsed JSON objects from an SSE ``data:`` stream (skips ``[DONE]``)."""
    for line in response.iter_lines():
        if not line or not line.startswith("data:"):
            continue
        data = line[len("data:"):].strip()
        if data == "[DONE]":
            break
        try:
            yield json.loads(data)
        except json.JSONDecodeError:
            continue


def find_token(tokens: Sequence[Token], predicate: Callable[[Token], bool]) -> list[Token]:
    """All tokens matching ``predicate`` (e.g. ``lambda t: t.token == '\\n'``)."""
    return [t for t in tokens if predicate(t)]


def window(positions: Sequence[int], center: int, radius: int) -> list[int]:
    """The ``[center-radius, center+radius]`` slice intersected with ``positions``."""
    lo, hi = center - radius, center + radius
    return [p for p in positions if lo <= p <= hi]
