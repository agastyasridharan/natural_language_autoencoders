"""Thin inference wrappers for NLA Studio.

Wraps the *vendored* NLA inference code (`vendor/nla_inference.py`, taken verbatim
from github.com/kitft/nla-inference) and adds the one piece that file does not
provide: `extract()`, which runs the **base** model and pulls the activation the
AV/AR were trained on. Everything injection- and MSE-related is delegated to the
vendored `NLAClient` / `NLACritic` — we do NOT reimplement that math.

Three model roles per family (see app/families.yaml):

  base model  — the original instruct model (e.g. Qwen2.5-7B-Instruct). Loaded
                in-process; `extract()` runs it with output_hidden_states.
  AV (actor)  — verbalizes a vector -> text. Served by SGLang (input_embeds
                injection); we talk to it over HTTP via the vendored NLAClient.
  AR (critic) — reconstructs text -> vector. Loaded in-process via NLACritic.

────────────────────────────────────────────────────────────────────────────
THE LAYER OFF-BY-ONE (read this before touching extract()):

`layer` in families.yaml is the data-gen extraction layer index K — the "L<N>"
in the checkpoint name (Qwen=20, gemma12b=32, gemma27b=41, llama70b=53). Data-gen
captured the activation with a *forward hook on decoder block K*, i.e. the OUTPUT
of block K (the model cards say exactly: "Residual stream output of block K").

In HF `output_hidden_states` indexing, hidden_states[0] is the embedding output
and hidden_states[i] is the output of block i-1. So:

        output of block K  ==  hidden_states[K + 1]

Using hidden_states[K] would be the residual stream *entering* block K — off by
one block, out of distribution, degraded/garbage decodes. We therefore extract
`hidden_states[K + 1]`, and the AR confirms it: the critic is the first K+1
blocks of the base model, so AR.num_hidden_layers == K + 1. load_family asserts
all three agree (registry K, sidecar extraction_layer_index, AR num_hidden_layers).
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import torch
import yaml

# Make the vendored single-file module importable regardless of how uvicorn is
# launched (as `app.server:app` from nla_studio/, or with app/ on the path).
_ROOT = Path(__file__).resolve().parent.parent  # nla_studio/
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from vendor.nla_inference import (  # noqa: E402  (path shim must run first)
    NLAClient,
    NLACritic,
    resolve_embed_scale,
)
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

# Positions before this index have too little left-context for the residual
# stream to carry meaningful signal; they decode toward the training prior. The
# UI greys them out. (nla_inference.py flags "first ~10 tokens"; data-gen used a
# stricter _MIN_POSITION=50 when *sampling* training positions — exposed via
# EARLY_POSITION_WARN so the UI/README can cite both.)
EARLY_POSITION_WARN = 10
EARLY_POSITION_STRICT = 50

# CJK fraction above which an AV decode is almost certainly an injection failure
# (the model verbalizing the marker char itself). See CLAUDE.md "Debugging".
CJK_FAIL_FRACTION = 0.20


# ── multimodal unwrap (mirrors nla/arch_adapters.py; vendored to stay self-contained) ──

def _text_config(config: Any) -> Any:
    """Text-side config for MM wrappers (Gemma3.text_config); pass-through otherwise."""
    return getattr(config, "text_config", None) or config


def _text_core(model: Any) -> Any:
    """Text tower whose forward returns hidden_states.

    Gemma3's text model applies the ×√d embedding scaling internally, so its
    hidden states match what the data-gen hook captured. Its location moved across
    transformers versions (`.language_model` vs `.model.language_model`), so try
    both. Qwen/Llama have neither → return the CausalLM itself (a forward pass on
    the full CausalLM still returns the same hidden_states tuple).
    """
    for path in (("language_model",), ("model", "language_model")):
        obj: Any = model
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None:
            return obj
    return model


def _is_cjk(ch: str) -> bool:
    """True for Han/Kana/Hangul/CJK-symbol codepoints — the injection-failure tell."""
    if ch.isspace() or not ch.strip():
        return False
    try:
        name = unicodedata.name(ch)
    except ValueError:
        return False
    return any(tag in name for tag in ("CJK", "HANGUL", "HIRAGANA", "KATAKANA"))


def cjk_fraction(text: str) -> float:
    """Fraction of non-space characters that are CJK. >CJK_FAIL_FRACTION ⇒ failure."""
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    return sum(_is_cjk(c) for c in chars) / len(chars)


# ── family registry ──────────────────────────────────────────────────────────

def load_registry(registry_path: str | Path) -> dict[str, dict[str, Any]]:
    data = yaml.safe_load(Path(registry_path).read_text())
    assert isinstance(data, dict) and data, f"empty/invalid registry: {registry_path!r}"
    return data


@dataclass
class LoadedFamily:
    """Everything needed to serve one family. One loaded at a time (see server)."""

    name: str
    cfg: dict[str, Any]                 # the families.yaml entry
    layer_k: int                        # extraction layer index K
    hidden_state_index: int            # K + 1  (the output_hidden_states index)
    d_model: int
    injection_scale: float
    embed_scale: float
    mse_scale: float
    ar_num_hidden_layers: int
    sglang_url: str

    base_model: Any                     # in-process AutoModelForCausalLM
    base_tokenizer: Any
    av_client: NLAClient                # talks to SGLang over HTTP
    critic: NLACritic                   # in-process AR

    def info(self) -> dict[str, Any]:
        """JSON-safe summary for /load_family (no tensors, no raw vectors)."""
        return {
            "family": self.name,
            "base_model": self.cfg["base_model"],
            "av_repo": self.cfg["av_repo"],
            "ar_repo": self.cfg["ar_repo"],
            "layer_k": self.layer_k,
            "hidden_state_index": self.hidden_state_index,
            "d_model": self.d_model,
            "injection_scale": self.injection_scale,
            "embed_scale": round(self.embed_scale, 4),
            "mse_scale": round(self.mse_scale, 4),
            "ar_num_hidden_layers": self.ar_num_hidden_layers,
            "sglang_url": self.sglang_url,
            "injection_char": self.av_client.cfg.injection_char,
            "injection_token_id": self.av_client.cfg.injection_token_id,
        }


def _read_sidecar_layer_index(av_path: str | Path) -> int | None:
    """K from the AV sidecar: top-level `extraction_layer_index` (added by the
    release scrub) or nested `extraction.layer_index`. None if neither present."""
    meta = yaml.safe_load((Path(av_path) / "nla_meta.yaml").read_text())
    top = meta.get("extraction_layer_index")
    if top is not None:
        return int(top)
    nested = (meta.get("extraction") or {}).get("layer_index")
    return int(nested) if nested is not None else None


def _verify_sglang(url: str, timeout: float = 5.0) -> dict[str, Any]:
    """Confirm the AV server is up. SGLang exposes /get_model_info and /health."""
    base = url.rstrip("/")
    try:
        r = httpx.get(f"{base}/get_model_info", timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as exc:  # noqa: BLE001 — surface a single actionable message
        raise RuntimeError(
            f"SGLang AV server not reachable at {base!r}: {exc}. Launch it first: "
            f"`scripts/launch_sglang.sh <family>` (and check the --port matches)."
        ) from exc


def load_family(
    name: str,
    registry_path: str | Path,
    *,
    inproc_device: str = "cuda:0",
    base_device: str | None = None,
    ar_device: str | None = None,
    sglang_url: str = "http://localhost:30000",
    hf_token: str | None = None,
    dtype: torch.dtype = torch.bfloat16,
    verify_sglang: bool = True,
) -> LoadedFamily:
    """Load base + AR in-process, build the AV client, run every cross-check.

    Downloads AV/AR snapshots into the shared HF cache (SGLang reuses the AV
    snapshot, so no double download). The base model loads from the HF cache via
    from_pretrained.

    Device placement: `inproc_device` is the default for both the base model and
    the AR; `base_device`/`ar_device` override each independently (used for the
    "put the AR on the GPU, keep the base on CPU" speed recipe on a single Colab
    GPU — see README "Going faster"). Device placement never changes any number:
    the same weights in the same dtype produce the same activations on CPU or GPU.
    Multi-GPU families use device_map="auto" for the base (the AR still lands on
    one device — see README, llama70b needs a sharded AR, out of scope here).
    """
    base_device = base_device or inproc_device
    ar_device = ar_device or inproc_device
    from huggingface_hub import snapshot_download  # local import: optional dep path

    registry = load_registry(registry_path)
    assert name in registry, f"unknown family {name!r}; have {sorted(registry)}"
    cfg = registry[name]
    layer_k = int(cfg["layer"])

    # AV/AR must be local: NLAClient.load_embedding_only and NLACritic both read
    # safetensors/value_head/sidecar straight off disk via pathlib.
    av_path = snapshot_download(cfg["av_repo"], token=hf_token)
    ar_path = snapshot_download(cfg["ar_repo"], token=hf_token)

    # AV client (embedding table on CPU is fine — ~100 rows looked up per request;
    # the heavy AV forward happens in SGLang). load_nla_config inside here asserts
    # the injection char + L/R neighbor ids against the live tokenizer at startup.
    av_client = NLAClient(av_path, sglang_url=sglang_url, device="cpu")

    # AR critic in-process.
    critic = NLACritic(ar_path, device=ar_device, dtype=dtype)

    # Base model in-process for extraction.
    base_device_map = "auto" if cfg.get("multi_gpu") else base_device
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], torch_dtype=dtype, device_map=base_device_map,
        trust_remote_code=True, token=hf_token,
    ).eval()
    base_tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"], token=hf_token)
    if base_tokenizer.pad_token_id is None:
        base_tokenizer.pad_token = base_tokenizer.eos_token
    # Right padding / right truncation: extract() takes hs[index] for a single
    # text (no padding), but keep parity with data-gen's HFExtractor so multi-text
    # use stays correct and token_ids[0] is always the doc start.
    base_tokenizer.padding_side = "right"
    base_tokenizer.truncation_side = "right"

    # ── cross-checks: registry K  ==  sidecar K  ==  (AR layers − 1) ──
    embed_scale = resolve_embed_scale(av_path)
    d_model = av_client.cfg.d_model
    ar_layers = int(_text_config(critic.backbone.config).num_hidden_layers)
    base_layers = int(_text_config(base_model.config).num_hidden_layers)

    assert d_model == int(cfg["d_model"]), (
        f"[{name}] AV sidecar d_model={d_model} != registry d_model={cfg['d_model']}"
    )
    assert int(_text_config(critic.backbone.config).hidden_size) == d_model, (
        f"[{name}] AR hidden_size != d_model={d_model}; wrong AR checkpoint?"
    )
    assert ar_layers == layer_k + 1, (
        f"[{name}] AR num_hidden_layers={ar_layers} != K+1={layer_k + 1}. The AR is "
        f"blocks 0..K of the base model; this mismatch means the registry `layer` "
        f"is wrong for this checkpoint."
    )
    assert layer_k + 1 <= base_layers, (
        f"[{name}] extraction index K+1={layer_k + 1} exceeds base decoder depth "
        f"{base_layers}; registry `layer` or base_model is wrong."
    )
    sidecar_k = _read_sidecar_layer_index(av_path)
    if sidecar_k is not None:
        assert sidecar_k == layer_k, (
            f"[{name}] AV sidecar extraction_layer_index={sidecar_k} != registry "
            f"layer={layer_k}. Trust the sidecar — fix families.yaml."
        )
    else:
        print(f"[load_family] {name}: AV sidecar has no extraction_layer_index; "
              f"relying on AR num_hidden_layers ({ar_layers}) == K+1 cross-check.")

    if verify_sglang:
        model_info = _verify_sglang(sglang_url)
        print(f"[load_family] {name}: SGLang AV up — {model_info.get('model_path', model_info)}")

    return LoadedFamily(
        name=name, cfg=cfg, layer_k=layer_k, hidden_state_index=layer_k + 1,
        d_model=d_model, injection_scale=av_client.cfg.injection_scale,
        embed_scale=embed_scale, mse_scale=critic.mse_scale,
        ar_num_hidden_layers=ar_layers, sglang_url=sglang_url,
        base_model=base_model, base_tokenizer=base_tokenizer,
        av_client=av_client, critic=critic,
    )


# ── extraction (the one bit NLAClient/NLACritic don't cover) ──────────────────

def _token_display(tokenizer: Any, token_id: int) -> str:
    """Human-readable single-token string. convert_ids_to_tokens keeps the byte/▁
    markers; decode([id]) renders bytes. Prefer decode but fall back to the piece."""
    piece = tokenizer.decode([token_id], skip_special_tokens=False)
    if piece == "" or piece.isspace():
        # whitespace/control: show the raw vocab piece so the row isn't blank
        piece = tokenizer.convert_ids_to_tokens(token_id) or repr(piece)
    return piece


def _outlier_threshold(norms: np.ndarray) -> float:
    """Robust high-norm cutoff. Outlier activations (e.g. Qwen L20 early-newline
    spikes ~14k vs typical ~150) decode unreliably even when injection is correct.
    Median+MAD over the non-early positions; ~100× separations make this safe."""
    body = norms[EARLY_POSITION_WARN:] if norms.size > EARLY_POSITION_WARN else norms
    if body.size == 0:
        return float("inf")
    med = float(np.median(body))
    mad = float(np.median(np.abs(body - med))) or 1.0
    # 8 MADs ≈ very conservative; real spikes sit far beyond this.
    return med + 8.0 * 1.4826 * mad


@torch.inference_mode()
def extract_all(
    fam: LoadedFamily,
    text: str,
    *,
    max_length: int = 2048,
) -> dict[str, Any]:
    """Run the base model ONCE and return per-token previews **plus** the full
    [T, d] activation matrix at hidden_states[K+1].

    This is the single expensive call (one base forward). The server caches the
    result per (family, text); `select()` then slices any token out of it with no
    extra forward, so clicking around the token grid is free. The vectors are
    bit-identical to a fresh forward — this is pure memoization, not an
    approximation.

    Tokenization matches data-gen's HFExtractor: raw text, add_special_tokens=True
    (BOS for Gemma/Llama, no-op for Qwen) — NOT a chat template.

    Private (leading-underscore) keys stay server-side and are never serialized:
      _vectors  float32 [T, d] — the activation matrix select() slices.
      _norms    float32 [T]    — full-precision L2 norms (select() reports these).
      _hi_thr   float          — outlier cutoff (select() recomputes high_outlier).
    """
    enc = fam.base_tokenizer(
        text, return_tensors="pt", add_special_tokens=True,
        truncation=True, max_length=max_length,
    )
    core = _text_core(fam.base_model)
    device = core.get_input_embeddings().weight.device
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    out = core(
        input_ids=input_ids, attention_mask=attention_mask,
        output_hidden_states=True, use_cache=False,
    )
    hs = out.hidden_states
    assert hs is not None, (
        f"[{fam.name}] base model returned no hidden_states. For Gemma make sure "
        f"the text tower (language_model) is being called."
    )
    n_layers_plus_1 = len(hs)
    assert fam.hidden_state_index < n_layers_plus_1, (
        f"[{fam.name}] need hidden_states[{fam.hidden_state_index}] but only "
        f"{n_layers_plus_1} states returned (base depth {n_layers_plus_1 - 1})."
    )

    layer_hs = hs[fam.hidden_state_index][0].float().cpu()  # [T, d]
    ids = input_ids[0].cpu().tolist()
    norms = layer_hs.norm(dim=-1).numpy()
    seq_len = len(ids)

    hi_thr = _outlier_threshold(norms)
    tokens = []
    for i, tid in enumerate(ids):
        n = float(norms[i])
        tokens.append({
            "index": i,
            "token": _token_display(fam.base_tokenizer, tid),
            "token_id": int(tid),
            "norm": round(n, 3),
            "early": i < EARLY_POSITION_WARN,
            "high_outlier": n > hi_thr,
        })

    # A sensible default pick: middle-ish, past the early window, not an outlier.
    clean = [t["index"] for t in tokens if not t["early"] and not t["high_outlier"]]
    suggested = clean[len(clean) // 2] if clean else min(EARLY_POSITION_WARN, seq_len - 1)

    return {
        "n_tokens": seq_len,
        "tokens": tokens,
        "suggested_index": suggested,
        "hidden_state_index": fam.hidden_state_index,
        "layer_k": fam.layer_k,
        "_vectors": np.ascontiguousarray(layer_hs.numpy(), dtype=np.float32),  # [T, d]
        "_norms": norms.astype(np.float32),                                    # [T]
        "_hi_thr": hi_thr,
    }


def picker_view(res: dict[str, Any]) -> dict[str, Any]:
    """JSON-safe picker fields from an extract_all() result (drops the server-only
    [T, d] matrix / norms / threshold). `selected` is filled in by the caller."""
    out = {k: v for k, v in res.items() if not k.startswith("_")}
    out["selected"] = None
    return out


def select(res: dict[str, Any], token_index: int) -> dict[str, Any]:
    """Slice one token's vector out of a cached extract_all() result. No forward.

    Returns the same `selected` payload the old per-call extract() did, byte for
    byte (full-precision norm, recomputed high_outlier) — the only difference is
    that the heavy forward already happened once and is reused here.
    """
    seq_len = res["n_tokens"]
    if not (0 <= token_index < seq_len):
        raise IndexError(
            f"token_index {token_index} out of range [0,{seq_len}) for this text."
        )
    norm = float(res["_norms"][token_index])
    return {
        # copy (not a view) so the stored vector doesn't pin the whole [T,d]
        # matrix alive after the cache entry is evicted.
        "vector": np.array(res["_vectors"][token_index], dtype=np.float32, copy=True),
        "index": token_index,
        "token": res["tokens"][token_index]["token"],
        "norm": norm,
        "early": token_index < EARLY_POSITION_WARN,
        "high_outlier": norm > res["_hi_thr"],
    }


@torch.inference_mode()
def extract(
    fam: LoadedFamily,
    text: str,
    token_index: int | None = None,
    *,
    max_length: int = 2048,
) -> dict[str, Any]:
    """Back-compat one-shot extraction: run the forward, then optionally select.

    Prefer the cached extract_all()+select() path in the server — this recomputes
    the base forward on every call. Kept for smoke.py and standalone use; the
    return shape is unchanged.
    """
    res = extract_all(fam, text, max_length=max_length)
    out = picker_view(res)
    if token_index is not None:
        out["selected"] = select(res, token_index)
    return out


# ── thin pass-throughs to the vendored AV/AR ─────────────────────────────────

def av_explain(fam: LoadedFamily, vector: np.ndarray, **sampling: Any) -> dict[str, Any]:
    """Vector -> explanation text via the AV (SGLang). Flags CJK injection failures."""
    text = fam.av_client.generate(np.asarray(vector, dtype=np.float32), **sampling)
    frac = cjk_fraction(text)
    return {
        "explanation": text,
        "cjk_fraction": round(frac, 3),
        "looks_like_injection_failure": frac > CJK_FAIL_FRACTION,
    }


def ar_reconstruct(fam: LoadedFamily, text: str) -> tuple[np.ndarray, float]:
    """Text -> predicted vector via the AR (NLACritic.reconstruct). Returns (vec, L2)."""
    pred = fam.critic.reconstruct(text)             # torch fp32 CPU, raw (unnormalized)
    vec = pred.numpy().astype(np.float32)
    return vec, float(np.linalg.norm(vec))


def score(fam: LoadedFamily, text: str, gold: np.ndarray) -> tuple[float, float]:
    """(cosine, mse) for `text` reconstructed against `gold`.

    NOTE: NLACritic.score returns (mse, cos) in THAT order — unpack carefully.
    Both vectors are L2-normalized to mse_scale inside, so mse == 2(1-cos) ∈ [0,4].
    """
    mse, cos = fam.critic.score(text, np.asarray(gold, dtype=np.float32))
    return float(cos), float(mse)
