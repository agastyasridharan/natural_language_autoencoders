"""Tensor-name canonicalization, component classification, and pairing.

Conventions observed in the four NLA families (HF checkpoints):
- Qwen2 / Llama:   model.layers.N.self_attn.q_proj.weight, model.embed_tokens.weight,
                   model.norm.weight, lm_head.weight; Qwen2 adds q/k/v biases.
- Gemma-3 base is multimodal (Gemma3ForConditionalGeneration): the text tower
  lives under "language_model." (vision_tower.*, multi_modal_projector.* have
  no NLA counterpart). NLA Gemma checkpoints are text-only (Gemma3ForCausalLM)
  without the prefix. Canonicalization strips "language_model.".
- The AR adds value_head.weight (no counterpart in M, excluded from diffs).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

CLASSES = [
    "embed", "attn.q", "attn.k", "attn.v", "attn.o",
    "mlp.gate", "mlp.up", "mlp.down", "norm", "lm_head", "other",
]

_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bembed_tokens\."), "embed"),
    (re.compile(r"\blm_head\."), "lm_head"),
    (re.compile(r"\bself_attn\.q_proj\."), "attn.q"),
    (re.compile(r"\bself_attn\.k_proj\."), "attn.k"),
    (re.compile(r"\bself_attn\.v_proj\."), "attn.v"),
    (re.compile(r"\bself_attn\.o_proj\."), "attn.o"),
    (re.compile(r"\bmlp\.gate_proj\."), "mlp.gate"),
    (re.compile(r"\bmlp\.up_proj\."), "mlp.up"),
    (re.compile(r"\bmlp\.down_proj\."), "mlp.down"),
    # all RMSNorm flavours: input/post_attention/pre_feedforward/
    # post_feedforward layernorms, attn q_norm/k_norm (Gemma-3), final norm
    (re.compile(r"(layernorm|\bq_norm\b|\bk_norm\b|\bnorm)\."), "norm"),
]

_LAYER_RE = re.compile(r"\blayers\.(\d+)\.")

_VISION_PREFIXES = ("vision_tower.", "multi_modal_projector.")


def canonical(name: str) -> str:
    return name[len("language_model."):] if name.startswith("language_model.") else name


def classify(name: str) -> str:
    c = canonical(name)
    for pat, cls in _RULES:
        if pat.search(c):
            return cls
    return "other"


def layer_of(name: str) -> int | None:
    m = _LAYER_RE.search(canonical(name))
    return int(m.group(1)) if m else None


def is_vision(name: str) -> bool:
    return name.startswith(_VISION_PREFIXES)


@dataclass
class Plan:
    matched: list[tuple[str, str]] = field(default_factory=list)  # (ft, base)
    ft_only: list[str] = field(default_factory=list)              # e.g. value_head
    base_truncated: list[str] = field(default_factory=list)       # layers >= L_TRUNC etc.
    base_vision: list[str] = field(default_factory=list)          # Gemma vision tower
    base_other_unmatched: list[str] = field(default_factory=list)
    l_trunc: int | None = None  # ft layer count when ft is depth-truncated


def build_plan(base_names: list[str], ft_names: list[str]) -> Plan:
    plan = Plan()
    base_by_canon = {canonical(n): n for n in base_names}
    ft_layers = [l for n in ft_names if (l := layer_of(n)) is not None]
    n_ft_layers = max(ft_layers) + 1 if ft_layers else None

    matched_base = set()
    for fn in sorted(ft_names):
        bn = base_by_canon.get(canonical(fn))
        if bn is None:
            plan.ft_only.append(fn)
        else:
            plan.matched.append((fn, bn))
            matched_base.add(bn)

    base_layers = [l for n in base_names if (l := layer_of(n)) is not None]
    n_base_layers = max(base_layers) + 1 if base_layers else None
    if n_ft_layers is not None and n_base_layers is not None and n_ft_layers < n_base_layers:
        plan.l_trunc = n_ft_layers

    for bn in sorted(set(base_names) - matched_base):
        if is_vision(bn):
            plan.base_vision.append(bn)
        elif plan.l_trunc is not None and (
            ((l := layer_of(bn)) is not None and l >= plan.l_trunc)
            or classify(bn) in ("lm_head", "norm")  # beyond the truncation point
        ):
            plan.base_truncated.append(bn)
        else:
            plan.base_other_unmatched.append(bn)
    return plan
