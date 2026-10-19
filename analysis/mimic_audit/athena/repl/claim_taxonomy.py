"""Claim-taxonomy classifier - LLM-judged reclassification of already-scored claims.

Takes the atomic claims of an atom-mode claim-ablation run (whose mechanical
verdicts - teacher-forced Delta-logp, never touched here - are already on disk)
and relabels each claim along three independent axes, so groundedness can be
sliced by better labels than the keyword bucket:

  taxo     fixed subtype, conditional on the claim's Layer-1 type
             theme  -> genre | topic | entity | structure | era | other
             detail -> quote | date | number | value | other
  axis     LLM-judged version of the keyword gist/quote/position/prediction bucket
  support  could an attentive reader of the visible source assert this claim?
             stated | inferable | beyond | contradicted
             (binary "inferable from text" = stated or inferable)

A fourth classification, entity classes per claim (person/place/org/title/ticker/
other), is a free char-span join against the stored Layer-2 overlay in
segments/<key>.jsonl - no API call (see entity_classes).

Independence discipline: the judge sees claim text + assertion + Layer-1 type and
the stored source context (context_left / context_right). It NEVER sees Delta-logp,
null thresholds, or verdicts, so a misclassification can only blur a verdict-by-
category contrast, never manufacture one.

Scope guard: exactly ONE API call per activation, temperature 0, pinned models,
results cached under taxonomy/<run_id>/<idx>.<model>.json. There is deliberately
NO batch path in this module - the visual-inspection phase gates mass calls.
"""
from __future__ import annotations

import functools
import json
import pathlib
import time
from typing import Any

HERE = pathlib.Path(__file__).resolve().parent          # .../athena/repl
ATHENA = HERE.parent
ENV = ATHENA.parents[2] / ".env"                         # repo root .env

PROMPT_VERSION = "t1"
MODELS = {"sonnet": "claude-sonnet-4-6", "haiku": "claude-haiku-4-5-20251001"}

THEME_TAXO = ("genre", "topic", "entity", "structure", "era", "other")
DETAIL_TAXO = ("quote", "date", "number", "value", "other")
AXES = ("gist", "quote", "position", "prediction", "other")
SUPPORTS = ("stated", "inferable", "beyond", "contradicted")


# ------------------------------------------------------------------ keyword bucket
def kw_bucket(s: dict) -> str:
    """Verbatim copy of repl/run_claim_ablation.py:claim_bucket (display-only here,
    so the studio can show keyword-vs-LLM side by side; keep the two in sync)."""
    su = (s.get("subtype") or "").lower()
    if s.get("role") == "quotes":
        return "quote"
    if s.get("role") == "predicts" or any(k in su for k in (
            "expect", "predict", "next-token", "next token", "candidate", "continuation",
            "completion", "upcoming")):
        return "prediction"
    if any(k in su for k in ("position", "syntact", "structural", "structure", "token",
                             "clause", "parenthetic", "mid-", "boundary")):
        return "position"
    if any(k in su for k in ("genre", "topic", "tone", "style", "register", "rhetor",
                             "subject", "purpose", "narrative", "mood", "voice", "format",
                             "content", "implication", "theme")):
        return "gist"
    return "other"


# --------------------------------------------------------------- entity overlay join
@functools.lru_cache(maxsize=8)
def load_segments(key: str) -> dict[int, dict]:
    """idx -> segments-cache row (atoms + Layer-2 entity overlay) for one model key."""
    p = ATHENA / "segments" / f"{key}.jsonl"
    out: dict[int, dict] = {}
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row.get("idx"), int):
                out[row["idx"]] = row
    return out


def entity_classes(rec: dict, seg_row: dict | None) -> list[list[str]]:
    """Per scored claim, the Layer-2 entity classes whose spans fall inside the
    claim's char span. Free (no API): the overlay already lives in the segments
    cache. Empty list = the claim mentions no classed entity."""
    snips = rec.get("snippet_results") or []
    out: list[list[str]] = [[] for _ in snips]
    if not seg_row:
        return out
    if (seg_row.get("z_inner") or "").strip() != (rec.get("z_inner") or "").strip():
        return out           # cache row is from a different segmentation; spans not comparable
    ents = seg_row.get("entities") or []
    for si, s in enumerate(snips):
        cs = s.get("char_span")
        if not cs:
            continue
        a, b = cs
        for e in ents:
            if isinstance(e.get("start"), int) and a <= e["start"] and e["end"] <= b:
                c = e.get("subtype") or "other"
                if c not in out[si]:
                    out[si].append(c)
    return out


# ------------------------------------------------------------------------ the judge
_SYSTEM = """You are a claim classifier for a mechanistic-interpretability experiment.

A "verbalizer" model looked at ONE internal activation from a language model reading a web \
document and wrote a short description of what that activation encodes. The description has \
been split into atomic claims. Classify each claim along three independent axes. You never \
judge whether the verbalizer truly decoded the activation; you only classify the claims.

INPUT you receive:
- SOURCE WINDOW: the final part (at most 800 characters) of the document prefix the model had \
read, ending exactly at the read token, which is wrapped in <<< >>>. Text before this window \
exists but is hidden from you.
- ACTUAL CONTINUATION: the ~120 characters that really follow the read token in the document.
- CLAIMS: numbered claims, each with its Layer-1 type (theme or detail) and a normalized \
assertion sentence.

For every claim output exactly one label per axis.

AXIS "taxo" - fixed subtype, conditional on the claim's type.
For THEME claims:
  genre      the kind or format of text (news report, blog post, product page, forum thread)
  topic      the subject matter (a shooting incident, Epsom-salt uses, a game recap)
  entity     the claim is centrally about a named entity's identity or attributes
  structure  how the text is organized (headline then body, list, Q&A, table)
  era        the time period or dating of the text
  other      none of the above fits
For DETAIL claims:
  quote      asserts a verbatim string occurs in the source
  date       a calendar date, year, month, or other time anchor
  number     a count, ordinal, or bare numeric quantity
  value      money, percentage, measurement, score, or other unit-bearing value
  other      none of the above fits

AXIS "axis" - what kind of thing the claim asserts about the source.
  gist        what the source is or is about overall: genre, format, topic, subject, tone, \
register, style, purpose, narrative, implication
  quote       specific words or strings occur in the source (quoted spans, named tokens or words)
  position    where or how the read token (or another token) sits in the local syntax or \
structure: mid-clause, mid-phrase, parenthetical, list position, sentence boundary, part of speech
  prediction  what comes NEXT after the read token: expected continuation, completion, \
candidate next words
  other       none of the above

AXIS "support" - could an attentive reader of the visible source assert this claim?
  stated       the asserted content is explicitly present in the visible source window \
(verbatim or a trivial paraphrase)
  inferable    not explicit, but an attentive reader of the visible window would confidently \
conclude it (genre, tone, topic, the syntax at the read token)
  beyond       about the source, but not determinable from the visible window: it concerns \
unseen text before the window, provenance, or facts the window cannot settle
  contradicted the visible source shows otherwise: a quoted string that does not occur \
(mangled or fabricated), a wrong description of the read token's position, a fact the window contradicts
Role-specific rules for "support":
- Quote-like claims: "stated" only if the string occurs verbatim (modulo whitespace/case) in \
the visible window. A paraphrase PRESENTED AS a quote is "contradicted", not "inferable".
- Position claims: check the marked read token's actual local syntax in the window; "stated" \
if the description is correct, "contradicted" if wrong.
- Prediction claims: compare with ACTUAL CONTINUATION. "stated" if the predicted words really \
occur right after the read token (modulo case/whitespace); "inferable" if they do not occur \
but a competent reader would consider them a likely continuation; "contradicted" if the \
actual continuation goes a materially different way and the prediction was not a reasonable \
reading of the window.
- If a claim mixes several assertions, judge its MAIN one (the normalized assertion sentence).

Also output "evidence": a short verbatim quote from the source window or continuation (or a \
reason, under 25 words) that justifies your "support" label.

Judge every claim independently. Do not consider what the verbalizer could have known; only \
whether the visible text supports the claim."""

_TOOL = {
    "name": "classify_claims",
    "description": "Emit exactly one classification per input claim, in input order.",
    "input_schema": {
        "type": "object",
        "properties": {
            "claims": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "i": {"type": "integer", "description": "the claim's number from the input list"},
                        "taxo": {"type": "string",
                                 "enum": sorted(set(THEME_TAXO) | set(DETAIL_TAXO))},
                        "axis": {"type": "string", "enum": list(AXES)},
                        "support": {"type": "string", "enum": list(SUPPORTS)},
                        "evidence": {"type": "string"},
                    },
                    "required": ["i", "taxo", "axis", "support", "evidence"],
                },
            }
        },
        "required": ["claims"],
    },
}


def _api_key() -> str:
    for line in ENV.read_text().splitlines():
        if line.startswith("ANTHROPIC_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError("ANTHROPIC_API_KEY not found in .env")


def build_user_content(rec: dict) -> str:
    left = rec.get("context_left") or ""
    right = rec.get("context_right") or ""
    tok = rec.get("read_token") or ""
    if tok and left.endswith(tok):
        window = left[: -len(tok)] + "<<<" + tok + ">>>"
    else:                                    # fall back: never silently drop the marker
        window = left + f"\n[read token: {tok!r}]"
    lines = [
        "SOURCE WINDOW (final part of the prefix; earlier text exists but is hidden; "
        "the read token is wrapped in <<< >>>):",
        window,
        "",
        "ACTUAL CONTINUATION (the text that really follows the read token):",
        right,
        "",
        f"CLAIMS to classify ({len(rec.get('snippet_results') or [])} total):",
    ]
    for k, s in enumerate(rec.get("snippet_results") or [], start=1):
        lines.append(f"{k}. [{s.get('type')}] {s.get('text', '').strip()}")
        if s.get("assertion"):
            lines.append(f"   assertion: {s['assertion']}")
    return "\n".join(lines)


def classify_activation(rec: dict, model_key: str, run_id: str = "") -> dict:
    """ONE Anthropic call classifying every scored claim of one activation.
    Falls back to the other pinned model only on a false-positive safety refusal
    (same behavior as the segmenter); the model actually used is recorded."""
    import anthropic                                     # lazy: server GETs never need it

    if model_key not in MODELS:
        raise ValueError(f"unknown model key {model_key!r} (want one of {sorted(MODELS)})")
    snips = rec.get("snippet_results") or []
    if not snips:
        raise ValueError("record has no scored claims")

    client = anthropic.Anthropic(api_key=_api_key())
    order = [model_key] + [m for m in MODELS if m != model_key]
    content = build_user_content(rec)
    msg = None
    used = None
    last: Exception | None = None
    for mk in order:
        try:
            msg = client.messages.create(
                model=MODELS[mk], max_tokens=4000, temperature=0, system=_SYSTEM,
                tools=[_TOOL], tool_choice={"type": "tool", "name": _TOOL["name"]},
                messages=[{"role": "user", "content": content}])
        except Exception as e:                           # pragma: no cover - network
            last = e
            continue
        if msg.stop_reason == "refusal":
            last = RuntimeError(f"{MODELS[mk]} refused (stop_reason=refusal)")
            continue
        used = mk
        break
    if msg is None or used is None:
        raise RuntimeError(f"classification failed: {last}")

    raw: list[dict] = []
    for b in msg.content:
        if b.type == "tool_use" and isinstance(b.input, dict):
            raw = b.input.get("claims") or []
    by_i = {c.get("i"): c for c in raw if isinstance(c, dict)}

    claims = []
    for k, s in enumerate(snips, start=1):
        c = by_i.get(k)
        if not c:
            claims.append({"i": k, "taxo": None, "axis": None, "support": None,
                           "evidence": "", "missing": True})
            continue
        legal = THEME_TAXO if s.get("type") == "theme" else DETAIL_TAXO
        taxo = c.get("taxo")
        entry = {"i": k,
                 "taxo": taxo if taxo in legal else "other",
                 "axis": c.get("axis") if c.get("axis") in AXES else "other",
                 "support": c.get("support") if c.get("support") in SUPPORTS else None,
                 "evidence": str(c.get("evidence") or "")[:400]}
        if taxo not in legal:
            entry["taxo_raw"] = taxo                      # keep what the model said
        claims.append(entry)

    return {
        "run_id": run_id, "idx": rec.get("idx"), "model": model_key,
        "model_id": MODELS[model_key], "model_used": MODELS[used],
        "prompt_version": PROMPT_VERSION, "ts": time.time(),
        "usage": {"input_tokens": getattr(msg.usage, "input_tokens", None),
                  "output_tokens": getattr(msg.usage, "output_tokens", None)},
        "claims": claims,
    }


def cache_path(taxo_dir: pathlib.Path, run_id: str, idx: int, model_key: str) -> pathlib.Path:
    return taxo_dir / run_id / f"{idx}.{model_key}.json"


def write_cache(path: pathlib.Path, obj: dict) -> None:
    """Atomic write (tmp + rename) so a concurrent aggregation poll never reads a
    half-written classification."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)
