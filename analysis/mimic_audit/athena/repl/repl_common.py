"""Shared config + tokenization-anchoring for the cross-model replication.

The activation-space audit was run on llama70b by reusing the Neuronpedia corpus
(documents, sampled read positions, real AV reads, text-only mimics). To replicate
on the other open NLAs (qwen7b, gemma12b, gemma27b) we hold the EXPERIMENT fixed
and vary only the model: same documents, same text-only mimics, same read LOCATION
in the text — but each model's own base (gold), AV (real read) and AR (score).

The one wrinkle is that a read "position" is a token index, which is
tokenizer-specific. So we anchor by CONTENT CHARACTER OFFSET: for each llama read
token we record (turn_index, char_offset_within_that_turn's_content); for every
other model we re-render the same document with that model's chat markers, tokenize
with offset mapping, and pick the token ending at the same content offset. The
reused mimic therefore describes the same text location the model's AV read.
"""
from __future__ import annotations

# exact chat markers, copied verbatim from nla_np/np_core/client.py _FAMILIES
FAMILIES = {
    "llama3": {"assistant": "assistant", "turn_end": "<|eot_id|>",
               "single": lambda role, content: f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}"},
    "gemma3": {"assistant": "model", "turn_end": "<end_of_turn>\n",
               "single": lambda role, content: f"<start_of_turn>{role}\n{content}"},
    "qwen":   {"assistant": "assistant", "turn_end": "<|im_end|>\n",
               "single": lambda role, content: f"<|im_start|>{role}\n{content}"},
}

# one entry per replication target (+ llama70b reference). repos from
# nla_studio/app/families.yaml; injection_scale comes from each AV sidecar.
MODELS = {
    "llama70b": dict(label="Llama-3.3-70B", family="llama3", layer=53, d_model=8192,
                     base="meta-llama/Llama-3.3-70B-Instruct",
                     av="kitft/Llama-3.3-70B-NLA-L53-av", ar="kitft/Llama-3.3-70B-NLA-L53-ar",
                     gemma=False),
    "qwen7b": dict(label="Qwen2.5-7B", family="qwen", layer=20, d_model=3584,
                   base="Qwen/Qwen2.5-7B-Instruct",
                   av="kitft/nla-qwen2.5-7b-L20-av", ar="kitft/nla-qwen2.5-7b-L20-ar",
                   gemma=False),
    "gemma12b": dict(label="Gemma-3-12B", family="gemma3", layer=32, d_model=3840,
                     base="google/gemma-3-12b-it",
                     av="kitft/nla-gemma3-12b-L32-av", ar="kitft/nla-gemma3-12b-L32-ar",
                     gemma=True),
    "gemma27b": dict(label="Gemma-3-27B", family="gemma3", layer=41, d_model=5376,
                     base="google/gemma-3-27b-it",
                     av="kitft/nla-gemma3-27b-L41-av", ar="kitft/nla-gemma3-27b-L41-ar",
                     gemma=True),
}

MAX_LEN = 2048


def render(payload, family):
    """Render a payload (raw str or [{role,content}]) the way the corpus did.

    Returns (text, spans) where spans[i] = (turn_index, content_start_char,
    content_end_char) locates each turn's CONTENT inside text. BOS is added by the
    tokenizer (add_special_tokens=True), not here.
    """
    if isinstance(payload, str):
        return payload, [(0, 0, len(payload))]
    F = FAMILIES[family]
    parts, spans, pos, n = [], [], 0, len(payload)
    for i, m in enumerate(payload):
        role = F["assistant"] if m["role"] == "assistant" else m["role"]
        prefix = F["single"](role, "")          # marker prefix (content is last in single())
        content = m["content"]
        cs = pos + len(prefix)
        spans.append((i, cs, cs + len(content)))
        chunk = prefix + content + (F["turn_end"] if i != n - 1 else "")
        parts.append(chunk)
        pos += len(chunk)
    return "".join(parts), spans


def content_offset_of(offsets, spans, position):
    """Given a token's index `position` and the offset mapping, return
    (turn_index, content_offset, ok). Used on the llama side to build the anchor."""
    if position >= len(offsets):
        return None, None, False
    cs, ce = offsets[position]
    if ce <= cs:                                 # special token (zero width)
        return None, None, False
    for (ti, s, e) in spans:
        if s <= cs and ce <= e + 1:              # +1 tolerance: token may end on the boundary
            return ti, ce - s, True
    # fall back: nearest span by end char
    ti, s, e = min(spans, key=lambda sp: abs(sp[2] - ce))
    return ti, max(0, ce - s), False


def position_for_anchor(offsets, spans, turn_index, content_offset):
    """On a target model: find the token index whose end is nearest the same
    content location. Returns (position, end_char, target_char, ok)."""
    span = next((sp for sp in spans if sp[0] == turn_index), spans[-1])
    _, s, e = span
    target = s + content_offset
    best, best_d = None, None
    for i, (cs, ce) in enumerate(offsets):
        if ce <= cs:                              # skip specials
            continue
        if cs < s - 1 or ce > e + 1:              # keep within this turn's content
            continue
        d = abs(ce - target)
        if best_d is None or d < best_d:
            best, best_d = i, d
    if best is None:                              # no content token matched the turn; global nearest
        cand = [(abs(ce - target), i, ce) for i, (cs, ce) in enumerate(offsets) if ce > cs]
        if not cand:
            return None, None, target, False
        best_d, best, _ = min(cand)
    return best, offsets[best][1], target, (best_d is not None and best_d <= 3)
