"""NLA Studio — FastAPI backend.

INFERENCE ONLY. Wires the vendored AV/AR (via app/nla_core.py) to a small HTTP
API and serves the single-page UI. One family loaded at a time.

Vectors are d_model-wide (3.5k–8k floats); they NEVER round-trip through the UI.
They live in an in-memory store keyed by id; endpoints expose only id, source
text, token index, and L2 norm.

Run:
    NLA_INPROC_DEVICE=cuda:0 NLA_SGLANG_URL=http://localhost:30000 \
        uvicorn app.server:app --host 0.0.0.0 --port 8000
(or use scripts/run.sh). Env vars:
    NLA_REGISTRY        path to families.yaml   (default: app/families.yaml)
    NLA_SGLANG_URL      AV server root          (default: http://localhost:30000)
    NLA_INPROC_DEVICE   base+AR device          (default: cuda:0)
    NLA_VERIFY_SGLANG   "0" to skip the reachability check on load (default: "1")
    HF_TOKEN            for gated Gemma/Llama repos
"""

from __future__ import annotations

import itertools
import os
import threading
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app import nla_core

_ROOT = Path(__file__).resolve().parent.parent      # nla_studio/
_WEB = _ROOT / "web"

REGISTRY_PATH = os.environ.get("NLA_REGISTRY", str(Path(__file__).parent / "families.yaml"))
SGLANG_URL = os.environ.get("NLA_SGLANG_URL", "http://localhost:30000")
INPROC_DEVICE = os.environ.get("NLA_INPROC_DEVICE", "cuda:0")
VERIFY_SGLANG = os.environ.get("NLA_VERIFY_SGLANG", "1") != "0"
HF_TOKEN = os.environ.get("HF_TOKEN")

app = FastAPI(title="NLA Studio", version="1.0")

# ── global state (single family; serialize all model access) ──────────────────
# GPU forwards are not safe to run concurrently on one model, and only one family
# is resident, so a single lock around inference is correct and simple. FastAPI
# runs these sync handlers in a threadpool, hence the lock rather than relying on
# the event loop being single-threaded.
_LOCK = threading.Lock()
_FAMILY: nla_core.LoadedFamily | None = None
_VECTORS: dict[str, dict[str, Any]] = {}
_ID_SEQ = itertools.count(1)


def _require_family() -> nla_core.LoadedFamily:
    if _FAMILY is None:
        raise HTTPException(409, "No family loaded. POST /load_family first.")
    return _FAMILY


def _store_vector(vec: np.ndarray, meta: dict[str, Any]) -> str:
    vid = f"vec_{next(_ID_SEQ)}"
    _VECTORS[vid] = {"vector": np.asarray(vec, dtype=np.float32), "meta": {**meta, "id": vid}}
    return vid


def _get_vector(vid: str) -> np.ndarray:
    entry = _VECTORS.get(vid)
    if entry is None:
        raise HTTPException(404, f"unknown vector_id {vid!r}. Extract or reconstruct one first.")
    return entry["vector"]


def _interpret(cos: float, mse: float) -> dict[str, Any]:
    """Map (cos, mse) to a label using the documented scale (cos1/mse0 perfect,
    cos.9/mse.2 good, cos0/mse2 orthogonal)."""
    if cos >= 0.95:
        label = "excellent (near-perfect reconstruction)"
    elif cos >= 0.85:
        label = "good (typical for a clean position)"
    elif cos >= 0.5:
        label = "mediocre"
    elif cos >= 0.2:
        label = "weak"
    else:
        label = "orthogonal / likely failed"
    return {
        "cosine": round(cos, 4),
        "mse": round(mse, 4),
        "label": label,
        "scale": "cos 1.0 / MSE 0.0 perfect · cos 0.9 / MSE 0.2 good · cos 0.0 / MSE 2.0 orthogonal",
    }


# ── request bodies ────────────────────────────────────────────────────────────

class LoadFamilyReq(BaseModel):
    family: str


class ExtractReq(BaseModel):
    text: str
    token_index: int | None = None


class AvExplainReq(BaseModel):
    vector_id: str
    temperature: float = 1.0
    max_new_tokens: int = 200


class ArReconstructReq(BaseModel):
    text: str


class ScoreReq(BaseModel):
    text: str
    vector_id: str


class RoundtripReq(BaseModel):
    text: str
    token_index: int
    temperature: float = 1.0
    max_new_tokens: int = 200


# ── routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def index() -> FileResponse:
    return FileResponse(str(_WEB / "index.html"))


@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True, "loaded_family": _FAMILY.name if _FAMILY else None}


@app.get("/families")
def families() -> dict[str, Any]:
    """List registry families (for the dropdown) + which one is loaded."""
    reg = nla_core.load_registry(REGISTRY_PATH)
    return {
        "families": [
            {
                "name": k,
                "base_model": v["base_model"],
                "layer": v["layer"],
                "d_model": v["d_model"],
                "gated": bool(v.get("gated")),
                "gemma_patch": bool(v.get("gemma_patch")),
                "multi_gpu": bool(v.get("multi_gpu")),
            }
            for k, v in reg.items()
        ],
        "loaded": _FAMILY.name if _FAMILY else None,
        "sglang_url": SGLANG_URL,
    }


@app.post("/load_family")
def load_family(req: LoadFamilyReq) -> dict[str, Any]:
    global _FAMILY
    with _LOCK:
        # Drop the previous family first so we don't hold 2× weights mid-load.
        if _FAMILY is not None:
            _FAMILY = None
            _VECTORS.clear()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass
        try:
            fam = nla_core.load_family(
                req.family, REGISTRY_PATH,
                inproc_device=INPROC_DEVICE, sglang_url=SGLANG_URL,
                hf_token=HF_TOKEN, verify_sglang=VERIFY_SGLANG,
            )
        except Exception as exc:  # noqa: BLE001 — return the actionable message to the UI
            raise HTTPException(400, f"load_family({req.family}) failed: {exc}") from exc
        _FAMILY = fam
        return {"status": "loaded", **fam.info()}


@app.post("/extract")
def extract(req: ExtractReq) -> dict[str, Any]:
    fam = _require_family()
    with _LOCK:
        try:
            res = nla_core.extract(fam, req.text, req.token_index)
        except IndexError as exc:
            raise HTTPException(422, str(exc)) from exc
    selected = res.pop("selected")
    out: dict[str, Any] = {
        "n_tokens": res["n_tokens"],
        "tokens": res["tokens"],
        "suggested_index": res["suggested_index"],
        "hidden_state_index": res["hidden_state_index"],
        "layer_k": res["layer_k"],
        "selected": None,
    }
    if selected is not None:
        vec = selected.pop("vector")           # strip raw floats before responding
        vid = _store_vector(vec, {
            "kind": "extracted", "family": fam.name,
            "source_text": req.text[:500], "token_index": selected["index"],
            "token": selected["token"], "norm": round(selected["norm"], 3),
        })
        out["selected"] = {"id": vid, **selected, "norm": round(selected["norm"], 3)}
    return out


@app.post("/av_explain")
def av_explain(req: AvExplainReq) -> dict[str, Any]:
    fam = _require_family()
    vec = _get_vector(req.vector_id)
    with _LOCK:
        try:
            res = nla_core.av_explain(
                fam, vec, temperature=req.temperature, max_new_tokens=req.max_new_tokens,
            )
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(502, f"AV (SGLang) request failed: {exc}") from exc
    return {"vector_id": req.vector_id, **res}


@app.post("/ar_reconstruct")
def ar_reconstruct(req: ArReconstructReq) -> dict[str, Any]:
    fam = _require_family()
    with _LOCK:
        vec, norm = nla_core.ar_reconstruct(fam, req.text)
    vid = _store_vector(vec, {
        "kind": "ar_pred", "family": fam.name,
        "source_text": req.text[:500], "token_index": None, "norm": round(norm, 3),
    })
    return {"id": vid, "norm": round(norm, 3), "d_model": fam.d_model, "source_text": req.text[:500]}


@app.post("/score")
def score(req: ScoreReq) -> dict[str, Any]:
    fam = _require_family()
    gold = _get_vector(req.vector_id)
    with _LOCK:
        cos, mse = nla_core.score(fam, req.text, gold)
    return {"vector_id": req.vector_id, **_interpret(cos, mse)}


@app.post("/roundtrip")
def roundtrip(req: RoundtripReq) -> dict[str, Any]:
    """extract -> av_explain -> score, in one call."""
    fam = _require_family()
    with _LOCK:
        try:
            ext = nla_core.extract(fam, req.text, req.token_index)
        except IndexError as exc:
            raise HTTPException(422, str(exc)) from exc
        selected = ext["selected"]
        if selected is None:
            raise HTTPException(422, "roundtrip needs a valid token_index.")
        vec = selected["vector"]
        vid = _store_vector(np.asarray(vec, dtype=np.float32), {
            "kind": "extracted", "family": fam.name,
            "source_text": req.text[:500], "token_index": selected["index"],
            "token": selected["token"], "norm": round(selected["norm"], 3),
        })
        av = nla_core.av_explain(
            fam, vec, temperature=req.temperature, max_new_tokens=req.max_new_tokens,
        )
        cos, mse = nla_core.score(fam, av["explanation"], vec)
    return {
        "vector": {
            "id": vid, "index": selected["index"], "token": selected["token"],
            "norm": round(selected["norm"], 3),
            "early": selected["early"], "high_outlier": selected["high_outlier"],
        },
        "explanation": av["explanation"],
        "cjk_fraction": av["cjk_fraction"],
        "looks_like_injection_failure": av["looks_like_injection_failure"],
        "n_tokens": ext["n_tokens"],
        **_interpret(cos, mse),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.server:app",
        host=os.environ.get("NLA_HOST", "0.0.0.0"),
        port=int(os.environ.get("NLA_PORT", "8000")),
        reload=False,
    )
