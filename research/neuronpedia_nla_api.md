# Neuronpedia NLA API — reverse-engineered contract (verified live)

Reverse-engineered from the open-source Neuronpedia repo (`hijohnnylin/neuronpedia`,
`apps/webapp/app/api/nla/*` + `apps/nla/server.py`) and confirmed against the live API
with our key. Auth verified (HTTP 200), `/explain` round-trip captured below.

## Base + auth
- **Base URL:** `https://www.neuronpedia.org/api`
- **Auth header:** `x-api-key: $NEURONPEDIA_API_KEY` (key is in repo `.env`, 50 chars). Optional on these routes but send it.

## Hosted NLA sources (live `GET /nla/sources`)
Only **2 of the 4** released NLAs are on Neuronpedia:

| modelId | nlaSourceId | layer | norm (= Var(v_nrm) baseline) |
|---|---|---|---|
| `gemma-3-27b-it` | `kitft-l41` | 41 | 0.0579 |
| `llama3.3-70b-it` | `kitft-l53` | 53 | 0.9169 |

Qwen-2.5-7B-L20 and Gemma-3-12B-L32 are **not** hosted → use the local white-box backend / self-hosted server for those. (`norm` matches the `Var(v_nrm)` in `examples/*.txt`.)

`GET /nla/sources` → `{ sources: [{ modelId, id, displayName, description, url, author, av, ar, layerNum, norm, createdAt, model:{ id, displayName, openRouterAvailable } }] }`

## The endpoint we need: `POST /nla/explain` (prompt + position → explanation + AR metrics)
**This is the cheap case** — text + token positions in, no local activation extraction needed.

**Request body**
```json
{
  "modelId": "llama3.3-70b-it",       // required
  "nlaSourceId": "kitft-l53",          // required
  "positions": [3, 5],                 // required; 0-indexed token positions; non-negative; max 16 NEW (uncached) per request
  "text": "The capital of France is Paris.",  // provide text OR messages (≤16384 chars)
  "messages": [{"role":"user","content":"..."}],  // optional alternative to text (chat template applied)
  "temperature": 0.7,                  // optional, default 0.7
  "tokens": ["..."],                   // optional
  "priorCacheId": "...",               // optional; reuse a prior cache (cached positions don't count to the 16 limit)
  "stream": false                      // optional, default false (true → SSE)
}
```

**Response** (verified live — includes the AR reconstruction metrics)
```json
{
  "results": [
    {
      "token": " of", "token_id": 315, "position": 3,
      "l2_norm": 27.79,                // raw activation L2 norm
      "description": "likely travel or geographic/encyclopedic content (… Iceland or a Scandinavian destination) …",
      "mse": 0.169,                    // AR reconstruction MSE (normalized; ~0.2 good, ~2 orthogonal)
      "cosine_similarity": 0.915,      // AR reconstruction cosine
      "generated": false,
      "fragment_index": 0, "fragment_count": 1   // multi-byte glyph fragment handling
    },
    { "token": " is", "position": 5, "l2_norm": 27.34, "description": "… likely \"Paris\" …", "mse": 0.225, "cosine_similarity": 0.887, … }
  ],
  "layer_index": 53,
  "prompt_length": 8,                  // use prompt_length-1 to target the LAST token
  "cacheId": "cmq5699320003o0urg6qstex7"
}
```

**Real captured example (Llama-70B, the failure mode live):** position 3 (`" of"`) reconstructs at `cos=0.915` yet the description confabulates "Iceland / Norway / Lesotho / Swiss canton" while the *theme* ("the capital of [country]") is correct → O1/O2 from the taxonomy, reproduced in one call, with `mse`/`cos` attached.

## Other public route
- `POST /nla/completion` → base-model generation: `{ completion, full_text, tokens:[{token,token_id,position}] }`. Body: `{modelId, messages|text, completion_tokens (≤512), temperature, nlaSourceId?}`. Not needed for explanations.

## What is NOT in the public API (matters for Part B)
The richer endpoints exist only on the **self-hosted** Python server (`apps/nla/server.py`, port 5009, `X-SECRET-KEY`; same as `kitft/nla` `nla_inference.py`), NOT on Neuronpedia:
`/describe` (raw vector → text), **`/score` (description + activation → mse/cos)**, `/compare` (a−b diff), `/extract` (text → raw activations), `/tokenize`.

Consequence for the probes:
- **P2 (activation-sensitivity) & P3 (recurrence): fully doable on the public API** via `/explain` (Gemma-27B + Llama-70B). Vary upstream context, target `prompt_length-1`, compare `description` + `mse`/`cos`.
- **P1 (per-claim AR-leverage = ablate a claim, re-score vs the activation): NOT possible on the public API** (no `/score` over an arbitrary description+vector). Needs the local white-box backend or self-hosted `server.py`.

## Gotchas
- Only 2 models hosted; max 16 new positions/request (chain with `priorCacheId`); `text` ≤ 16384 chars.
- Backpressure: heavy `/explain` load can return HTTP 429 — throttle.
- `temperature` default 0.7; for P3 recurrence keep temp>0 (you want variation); for a reproducible single decode try `temperature: 0`.

## Minimal client (for Part B `nla/failure_analysis/`)
```python
import os, httpx
BASE = "https://www.neuronpedia.org/api"
H = {"x-api-key": os.environ["NEURONPEDIA_API_KEY"], "Content-Type": "application/json"}

def explain(text, positions, model="llama3.3-70b-it", source="kitft-l53", temperature=0.7, prior=None):
    body = {"modelId": model, "nlaSourceId": source, "text": text,
            "positions": positions, "temperature": temperature}
    if prior: body["priorCacheId"] = prior
    r = httpx.post(f"{BASE}/nla/explain", headers=H, json=body, timeout=180)
    r.raise_for_status()
    return r.json()   # {results:[{position,token,description,mse,cosine_similarity,l2_norm,...}], prompt_length, cacheId}
```
