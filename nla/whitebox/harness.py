"""WhiteBoxNLA — the shared white-box core for the released NLA pairs.

Loads three models locally and exposes the full round trip in one process:

    extraction_model  (the TARGET LLM)  -- forward + hook -->  layer-K activation
    AV (verbalizer)   inputs_embeds      -- inject + generate -->  explanation text
    AR (reconstructor, NLACritic)        -- reconstruct + score -->  (mse_nrm, cos)

Everything the black-box `nla_inference.NLAClient` cannot do, because it POSTs
to an SGLang server, lives here:

  * `extract_conversation`  — run the target model over a chat conversation and
    grab the block-K residual at every position (the activations to verbalize).
  * `capture_marker_pathway` — forward the AV over the injected prompt with
    `output_hidden_states=True` and return the marker position's residual at
    every depth: norm + cosine-to-the-injected-vector vs layer. This is the
    core *reading-pathway* measurement (experiment E1).
  * `verbalize` — inject an activation and `generate(inputs_embeds=...)` in
    process (greedy or sampled).

Injection reuses the project's canonical, unit-tested primitives
(`nla.injection.inject_at_marked_positions`, `nla.schema.normalize_activation`)
and the sidecar contract (`nla.config.load_nla_config`). The AR is the
standalone pure-torch `nla_inference.NLACritic`. Nothing tokenizer-dependent is
hardcoded — it all comes from `nla_meta.yaml`.

Memory: three models (two full + one truncated). For Qwen-2.5-7B that's
~15+15+11 GB in bf16 — fits one 80 GB GPU. Use `unload_*()` to stage on
smaller cards, or `device="auto"` to shard the AV/extractor via accelerate
(`device_map="auto"`). The 70 B AR is not shardable through NLACritic yet
(documented limitation; Phase-0 target is Qwen).
"""

from __future__ import annotations

import gc
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Repo root holds the single-file `nla_inference.py` (NLACritic). Make the
# import robust no matter the CWD the harness is driven from.
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from nla.arch_adapters import (  # noqa: E402
    resolve_decoder_layers,
    resolve_embed_scale,
    resolve_text_model,
)
from nla.config import NLAConfig, load_nla_config  # noqa: E402
from nla.injection import inject_at_marked_positions  # noqa: E402
from nla.schema import EXPLANATION_RE, normalize_activation  # noqa: E402
from nla.whitebox.constants import RELEASED, ReleasedNLA  # noqa: E402
from nla_inference import NLACritic  # noqa: E402


@dataclass
class ConversationActivations:
    """Per-position layer-K activations for one extraction conversation."""

    token_ids: list[int]
    token_strs: list[str]
    residuals: torch.Tensor  # [T, d_model] float32 CPU — block-K output (= hidden_states[K+1])
    layer: int
    prompt_len: int          # # prompt tokens before the generated reply (region boundary)

    def __len__(self) -> int:
        return len(self.token_ids)

    def region(self, idx: int) -> str:
        return "PROMPT" if idx < self.prompt_len else "REPLY"


@dataclass
class MarkerPathway:
    """The injection marker's residual stream traced through the AV's depth.

    Tests the propagation hypothesis: a large `injection_scale` keeps the raw
    injected vector alive (high norm, cos≈1 to itself) through early layers
    until a depth that "reads" it, after which the marker residual rotates away
    (cos drops) and takes on read/derived content.
    """

    layer_residuals: torch.Tensor   # [num_layers+1, d] float32 CPU (index 0 = embedding output)
    norms: torch.Tensor            # [num_layers+1] L2 norm per depth
    cos_to_injected: torch.Tensor  # [num_layers+1] cosine to the injected (scaled) vector
    injected_vector: torch.Tensor  # [d] the scaled activation actually placed at the marker
    marker_pos: int


@dataclass
class RoundTrip:
    explanation: str
    mse_nrm: float
    cos: float
    fve_nrm: float


def _free() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class WhiteBoxNLA:
    def __init__(
        self,
        av: str | Path,
        ar: str | Path,
        extraction_model: str,
        *,
        layer: int,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        ar_device: str | None = None,
        extraction_device: str | None = None,
        injection_scale_override: float | None = None,
    ):
        """av/ar may be local checkpoint dirs OR HF repo ids (auto-downloaded).

        `device="auto"` shards the full models via accelerate. The AR
        (NLACritic) always lands on a single device (`ar_device or device`,
        with "auto" -> "cuda:0" fallback).
        """
        self._av_src = str(av)
        self._ar_src = str(ar)
        self.extraction_model_name = extraction_model
        self.layer = layer
        self.device = device
        self.ar_device = ar_device or device
        self.extraction_device = extraction_device or device
        self.dtype = dtype
        self._inj_override = injection_scale_override

        self._av_dir: str | None = None
        self._ar_dir: str | None = None
        self._tok = None
        self._cfg: NLAConfig | None = None
        self._av = None
        self._ar: NLACritic | None = None
        self._extractor = None
        self._extractor_tok = None

    @classmethod
    def from_released(cls, model: ReleasedNLA | str, **kwargs) -> "WhiteBoxNLA":
        m = RELEASED[model] if isinstance(model, str) else model
        assert m.av_repo and m.ar_repo, (
            f"{m.name} has no known repo ids in the registry — pass av=/ar= "
            f"local checkpoint dirs explicitly (see constants.py)."
        )
        return cls(m.av_repo, m.ar_repo, m.extraction_model, layer=m.layer, **kwargs)

    # ── checkpoint resolution (local dir or HF repo) ──────────────────────────

    @staticmethod
    def _resolve(src: str) -> str:
        if Path(src).exists():
            return str(src)
        from huggingface_hub import snapshot_download

        return snapshot_download(src)

    @property
    def av_dir(self) -> str:
        if self._av_dir is None:
            self._av_dir = self._resolve(self._av_src)
        return self._av_dir

    @property
    def ar_dir(self) -> str:
        if self._ar_dir is None:
            self._ar_dir = self._resolve(self._ar_src)
        return self._ar_dir

    # ── lazy model / config loaders ───────────────────────────────────────────

    def _from_pretrained(self, src: str, device: str):
        # low_cpu_mem_usage avoids a full-size CPU RAM spike while loading each
        # 7B model (matters when three load in one session). On a single A100
        # 80 GB all three fit resident in bf16 (~42 GB) — no device_map needed.
        common = dict(torch_dtype=self.dtype, trust_remote_code=True, low_cpu_mem_usage=True)
        if device == "auto":
            return AutoModelForCausalLM.from_pretrained(src, device_map="auto", **common).eval()
        return AutoModelForCausalLM.from_pretrained(src, **common).to(device).eval()

    @property
    def tokenizer(self):
        if self._tok is None:
            self._tok = AutoTokenizer.from_pretrained(self.av_dir, trust_remote_code=True)
        return self._tok

    @property
    def cfg(self) -> NLAConfig:
        if self._cfg is None:
            cfg = load_nla_config(self.av_dir, self.tokenizer)
            if self._inj_override is not None:
                cfg = replace(cfg, injection_scale=float(self._inj_override))
            assert cfg.injection_scale is not None, (
                "AV sidecar has no injection_scale and no override given — the "
                "model learned with a specific scale; injecting raw is OOD."
            )
            self._cfg = cfg
        return self._cfg

    @property
    def av(self):
        if self._av is None:
            self._av = self._from_pretrained(self.av_dir, self.device)
        return self._av

    @property
    def ar(self) -> NLACritic:
        if self._ar is None:
            dev = "cuda:0" if self.ar_device == "auto" else self.ar_device
            self._ar = NLACritic(self.ar_dir, device=dev, dtype=self.dtype)
        return self._ar

    @property
    def extractor(self):
        if self._extractor is None:
            self._extractor = self._from_pretrained(self.extraction_model_name, self.extraction_device)
        return self._extractor

    @property
    def extractor_tokenizer(self):
        if self._extractor_tok is None:
            self._extractor_tok = AutoTokenizer.from_pretrained(
                self.extraction_model_name, trust_remote_code=True
            )
        return self._extractor_tok

    def unload_extractor(self) -> None:
        self._extractor = None
        _free()

    def unload_av(self) -> None:
        self._av = None
        _free()

    def unload_ar(self) -> None:
        self._ar = None
        _free()

    # ── activation extraction (the TARGET model side) ─────────────────────────

    @torch.no_grad()
    def extract_conversation(
        self,
        messages: list[dict],
        *,
        generate_reply: bool = True,
        reply_text: str | None = None,
        max_new_tokens: int = 256,
    ) -> ConversationActivations:
        """Run the target model over a chat conversation, capture block-K
        residuals at every position.

        `reply_text` (verbatim) reconstructs the sequence deterministically — use
        it to reproduce a logged dump exactly, sidestepping any greedy-decode
        divergence across HF/SGLang/hardware. Otherwise the reply is generated
        greedily. `messages` is just the user turn(s); the chat template adds the
        model's default system prompt (Qwen) and the assistant scaffold.
        """
        tok = self.extractor_tokenizer
        prompt_ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        prompt_len = len(prompt_ids)

        if reply_text is not None:
            reply_ids = tok.encode(reply_text, add_special_tokens=False)
            full_ids = list(prompt_ids) + list(reply_ids)
        elif generate_reply:
            model = self.extractor
            dev = model.get_input_embeddings().weight.device
            inp = torch.tensor(prompt_ids, dtype=torch.long, device=dev).unsqueeze(0)
            gen = model.generate(
                inp,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
            )
            full_ids = gen[0].tolist()
        else:
            full_ids = list(prompt_ids)

        residuals = self._forward_capture_layer(full_ids, self.layer)
        token_strs = tok.convert_ids_to_tokens(full_ids)
        return ConversationActivations(
            token_ids=full_ids,
            token_strs=token_strs,
            residuals=residuals,
            layer=self.layer,
            prompt_len=prompt_len,
        )

    @torch.no_grad()
    def _forward_capture_layer(self, token_ids: list[int], layer: int) -> torch.Tensor:
        """Forward the extractor over token_ids, hook block-`layer`, return its
        output [T, d] (float32 CPU). Mirrors HFExtractor: block-K output =
        hidden_states[K+1]."""
        model = self.extractor
        layers = resolve_decoder_layers(model)
        assert 0 <= layer < len(layers), f"layer {layer} out of range ({len(layers)} blocks)"
        captured: dict[str, torch.Tensor] = {}

        def hook(_m, _i, out):
            captured["h"] = (out[0] if isinstance(out, tuple) else out).detach()

        handle = layers[layer].register_forward_hook(hook)
        try:
            dev = model.get_input_embeddings().weight.device
            ids = torch.tensor(token_ids, dtype=torch.long, device=dev).unsqueeze(0)
            model(input_ids=ids, use_cache=False)
        finally:
            handle.remove()

        assert "h" in captured, f"hook on decoder block {layer} never fired"
        h = captured["h"][0].float().cpu()
        assert h.shape[0] == len(token_ids), f"captured {h.shape[0]} positions, expected {len(token_ids)}"
        return h

    # ── injection (the AV side) ───────────────────────────────────────────────

    def _marker_position(self, ids_row: torch.Tensor) -> int:
        cfg = self.cfg
        all_pos = [int(p) for p in (ids_row == cfg.injection_token_id).nonzero().flatten().tolist()]
        valid = [
            p
            for p in all_pos
            if 0 < p < len(ids_row) - 1
            and int(ids_row[p - 1]) == cfg.injection_left_neighbor_id
            and int(ids_row[p + 1]) == cfg.injection_right_neighbor_id
        ]
        assert len(valid) == 1, (
            f"expected exactly one neighbor-validated marker, found {valid} "
            f"(all injection-id positions: {all_pos}). Template/tokenizer drift."
        )
        return valid[0]

    @torch.no_grad()
    def build_injected_embeds(self, activation) -> tuple[torch.Tensor, int]:
        """Tokenize the canonical AV prompt, build inputs_embeds, inject the
        (norm-rescaled) activation at the marker. Returns (embeds[1,T,d] float32
        on the AV device, marker_pos).

        Mirrors NLAClient._build_embeds: token rows are raw-weight lookups times
        the arch embed-scale (1.0 Qwen/Llama, √d Gemma — because passing
        inputs_embeds to generate bypasses the model's own ×√d). The marker row
        is the activation normalized to injection_scale.
        """
        cfg = self.cfg
        content = cfg.actor_prompt_template.format(injection_char=cfg.injection_char)
        ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=True, add_generation_prompt=True
        )
        ids_t = torch.tensor(ids, dtype=torch.long).unsqueeze(0)

        text_model = resolve_text_model(self.av)
        embed_w = text_model.get_input_embeddings().weight  # [vocab, d]
        embed_scale = resolve_embed_scale(self.av.config)
        emb = embed_w[ids_t.to(embed_w.device)].float() * embed_scale  # [1, T, d] fp32

        v = torch.as_tensor(np.asarray(activation, dtype=np.float32)).view(1, -1)
        assert v.shape[1] == cfg.d_model, f"activation d={v.shape[1]} != d_model {cfg.d_model}"
        v_scaled = normalize_activation(v, cfg.injection_scale).to(emb.device)

        injected = inject_at_marked_positions(
            ids_t.to(emb.device),
            emb,
            v_scaled,
            cfg.injection_token_id,
            cfg.injection_left_neighbor_id,
            cfg.injection_right_neighbor_id,
        )
        return injected, self._marker_position(ids_t[0])

    @torch.no_grad()
    def verbalize(
        self,
        activation,
        *,
        greedy: bool = True,
        max_new_tokens: int = 200,
        temperature: float = 1.0,
        return_raw: bool = False,
    ) -> str:
        """Inject `activation` and generate the explanation (greedy or sampled)."""
        injected, _ = self.build_injected_embeds(activation)
        injected = injected.to(self.av.dtype)
        attn = torch.ones(injected.shape[:2], dtype=torch.long, device=injected.device)
        kwargs = dict(
            inputs_embeds=injected,
            attention_mask=attn,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eos_token_id,
        )
        if greedy:
            kwargs["do_sample"] = False
        else:
            kwargs.update(do_sample=True, temperature=temperature)
        out = self.av.generate(**kwargs)
        # With inputs_embeds and no input_ids, generate returns only new tokens.
        text = self.tokenizer.decode(out[0], skip_special_tokens=False)
        if return_raw:
            return text
        m = EXPLANATION_RE.search(text)
        return m.group(1).strip() if m else text

    @torch.no_grad()
    def capture_marker_pathway(self, activation) -> MarkerPathway:
        """Forward the AV over the injected prompt, return the marker position's
        residual at every depth + its norm and cosine-to-injected. (Experiment
        E1.)"""
        injected, marker_pos = self.build_injected_embeds(activation)
        injected = injected.to(self.av.dtype)
        attn = torch.ones(injected.shape[:2], dtype=torch.long, device=injected.device)
        out = self.av(
            inputs_embeds=injected,
            attention_mask=attn,
            output_hidden_states=True,
            use_cache=False,
        )
        marker = torch.stack([h[0, marker_pos].float().cpu() for h in out.hidden_states])  # [L+1, d]
        injected_vec = injected[0, marker_pos].float().cpu()  # == marker[0]
        norms = marker.norm(dim=-1)
        cos = torch.nn.functional.cosine_similarity(
            marker, injected_vec.unsqueeze(0).expand_as(marker), dim=-1
        )
        return MarkerPathway(
            layer_residuals=marker,
            norms=norms,
            cos_to_injected=cos,
            injected_vector=injected_vec,
            marker_pos=marker_pos,
        )

    # ── AR scoring ────────────────────────────────────────────────────────────

    def reconstruct(self, explanation: str) -> torch.Tensor:
        return self.ar.reconstruct(explanation)

    def score(self, explanation: str, activation) -> tuple[float, float]:
        """(mse_nrm, cos) — both vectors L2-normalized to mse_scale (direction
        only). mse_nrm = 2(1-cos)."""
        return self.ar.score(explanation, activation)

    @staticmethod
    def fve_nrm(mse_nrm: float, baseline: float) -> float:
        return 1.0 - mse_nrm / baseline

    def round_trip(
        self, activation, *, baseline: float, greedy: bool = True, max_new_tokens: int = 200
    ) -> RoundTrip:
        """extract→already-done; verbalize→score→FVE in one call."""
        expl = self.verbalize(activation, greedy=greedy, max_new_tokens=max_new_tokens)
        mse, cos = self.score(expl, activation)
        return RoundTrip(explanation=expl, mse_nrm=mse, cos=cos, fve_nrm=self.fve_nrm(mse, baseline))
