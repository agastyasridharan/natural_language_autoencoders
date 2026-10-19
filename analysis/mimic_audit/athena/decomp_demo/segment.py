"""Claim segmenter for NLA verbalizer outputs — three versions, kept side by side.

  v1  Sparse (original)          : extract only salient spans (theme/entity/detail),
                                   nested entities allowed. Most of the text is left
                                   unsegmented.
  v2  Full coverage, 6 types     : tiling partition; hedge / framing / connective are
                                   their OWN claims; entities nested. (loaded verbatim
                                   from atoms_v2.json so the tab matches what was shown)
  v3  Self-complete (current edits): full-coverage, NON-INTERSECTING, self-complete
                                   claims. Hedges ("potentially") and directly-relevant
                                   framing ("ends a clause", "is an incomplete request")
                                   stay INSIDE the claim they modify; quotes absorb their
                                   description; NO nesting. Hedge words tracked per claim.

Segmentation is done by a constrained LLM (extractive, verbatim, temp 0). It is a
SEGMENTER, not a judge: it never sees an activation, a log-prob, or a verdict. The
grounded/confabulated measurement (teacher-forced Delta-logp) is a separate, judge-free
step and is unchanged. A stdlib-regex tier confirms detail spans (`regex_confirmed`).
"""
from __future__ import annotations
import json, re, pathlib
from typing import Any

HERE = pathlib.Path(__file__).resolve().parent
ENV = pathlib.Path("/Users/agastyasridharan/ruhr research/natural_language_autoencoders_original/.env")
# Primary segmenter is Sonnet; Haiku is a fallback for the rare false-positive safety
# refusal (Sonnet occasionally refuses benign inputs, e.g. pesticide/gardening text, and
# returns an empty tool call). Both are current 4.x models; the older -latest aliases were
# removed (deprecated / not available on all keys).
MODEL_CANDIDATES = ["claude-sonnet-4-6", "claude-haiku-4-5-20251001"]
CONTENT_TYPES = ("theme", "entity", "detail")

# ---------------------------------------------------------------- deterministic
_DETAIL_PATTERNS = [
    ("quote",  re.compile(r'"[^"\n]{1,200}"')),
    ("quote",  re.compile(r'“[^”\n]{1,200}”')),
    ("value",  re.compile(r'[$£€]\s?\d[\d,]*(?:\.\d+)?')),
    ("value",  re.compile(r'\d+(?:\.\d+)?\s?%')),
    ("value",  re.compile(r'\bQ[1-4]\b')),
    ("date",   re.compile(r'\b(?:19|20)\d{2}\b')),
    ("date",   re.compile(r'\b(?:January|February|March|April|May|June|July|'
                          r'August|September|October|November|December)\b')),
    ("number", re.compile(r'\b\d+(?:st|nd|rd|th)\b')),
    ("number", re.compile(r'\b\d+(?:\.\d+)?\b')),
]

def regex_details(text: str) -> list[dict[str, Any]]:
    hits = []
    for subtype, pat in _DETAIL_PATTERNS:
        for m in pat.finditer(text):
            hits.append({"start": m.start(), "end": m.end(), "text": m.group(), "subtype": subtype})
    hits.sort(key=lambda h: (h["start"], -(h["end"] - h["start"])))
    kept = []
    for h in hits:
        if not any(h["start"] < k["end"] and h["end"] > k["start"] for k in kept):
            kept.append(h)
    return kept

# --------------------------------------------------------------------- shared
_QUOTES = {"“": '"', "”": '"', "‘": "'", "’": "'"}
def _norm(s: str) -> str:
    return "".join(_QUOTES.get(c, c) for c in s)

def _find(text: str, seg: str, cursor: int) -> tuple[int, int]:
    if not seg:
        return -1, 0
    i = text.find(seg, cursor)
    if i >= 0:
        return i, len(seg)
    nt, ns = _norm(text), _norm(seg)
    i = nt.find(ns, cursor)
    if i >= 0:
        return i, len(ns)
    s2 = seg.strip()
    if s2 and s2 != seg:
        i = text.find(s2, cursor)
        if i >= 0:
            return i, len(s2)
        i = nt.find(_norm(s2), cursor)
        if i >= 0:
            return i, len(s2)
    return -1, 0

def _api_key() -> str:
    for line in ENV.read_text().splitlines():
        if line.startswith("ANTHROPIC_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError("ANTHROPIC_API_KEY not found in .env")

def _call(system: str, tool: dict, text: str) -> tuple[dict, str]:
    import anthropic
    client = anthropic.Anthropic(api_key=_api_key())
    last = None
    for model in MODEL_CANDIDATES:
        try:
            msg = client.messages.create(
                model=model, max_tokens=6000, temperature=0, system=system,
                tools=[tool], tool_choice={"type": "tool", "name": tool["name"]},
                messages=[{"role": "user", "content": text}])
        except Exception as e:
            last = e
            continue
        if msg.stop_reason == "refusal":          # false-positive safety refusal -> try next model
            last = RuntimeError(f"{model} refused (stop_reason=refusal)")
            continue
        for b in msg.content:                      # require a non-empty segmentation, not just a tool block
            if b.type == "tool_use" and isinstance(b.input, dict) and b.input.get("segments"):
                return b.input, model
        last = RuntimeError(f"{model} returned no segments (stop_reason={msg.stop_reason})")
    raise RuntimeError(f"all model candidates failed: {last}")

def _confirm_details(atoms: list[dict], det: list[dict]) -> None:
    for a in atoms:
        if a["type"] == "detail":
            a["regex_confirmed"] = any(a["start"] < d["end"] and a["end"] > d["start"] for d in det)

# ==================================================================== V1 sparse
_SYS_V1 = """You are an extractive claim segmenter for a mechanistic-interpretability experiment.
The input is a short description written by a "verbalizer" model describing ONE internal neural
activation. Split it into ATOMIC, self-complete claim spans and type each span.

Types: theme (genre/topic/structure/era), entity (person/place/org/title/other),
detail (quote/date/number/value).

Rules:
1. EXTRACTIVE ONLY: every "text" is an exact verbatim substring. For a quote, return only the inner
   quoted text (exclude the marks); emit each distinct quoted string as its own atom.
2. ATOMIC: smallest self-contained unit.
3. NESTING ALLOWED: an entity inside a quote or theme is its own atom too (overlap is fine).
4. Do NOT emit pure connective filler ("with", "showing", "probably") as atoms.
5. Reading order, via emit_atoms."""
_TOOL_V1 = {"name": "emit_atoms", "description": "Return atomic typed claim spans (verbatim).",
    "input_schema": {"type": "object", "properties": {"atoms": {"type": "array", "items": {
        "type": "object", "properties": {
            "text": {"type": "string"}, "type": {"type": "string", "enum": ["theme", "entity", "detail"]},
            "subtype": {"type": "string"}}, "required": ["text", "type", "subtype"]}}}, "required": ["atoms"]}}

def _locate_sparse(text: str, atoms: list[dict]) -> tuple[list[dict], list[dict]]:
    ntext = _norm(text); located, dropped, placed = [], [], []
    for a in atoms:
        cand = _norm(a["text"]).strip()
        if len(cand) >= 2 and cand[0] in "\"'" and cand[-1] == cand[0]:
            cand = cand[1:-1].strip()
        if not cand:
            dropped.append(a); continue
        occ = [m.start() for m in re.finditer(re.escape(cand), ntext)]
        if not occ:
            dropped.append(a); continue
        pick = next((s for s in occ if not any(u0 <= s < u1 for u0, u1 in placed)), occ[0])
        end = pick + len(cand); placed.append((pick, end))
        located.append({**a, "text": text[pick:end], "start": pick, "end": end})
    located.sort(key=lambda a: (a["start"], -(a["end"] - a["start"])))
    return located, dropped

def _parents(atoms: list[dict]) -> None:
    for i, a in enumerate(atoms):
        best = None
        for j, b in enumerate(atoms):
            if i == j:
                continue
            if b["start"] <= a["start"] and b["end"] >= a["end"] and (b["end"] - b["start"]) > (a["end"] - a["start"]):
                if best is None or (b["end"] - b["start"]) < (atoms[best]["end"] - atoms[best]["start"]):
                    best = j
        a["parent"] = best

def segment_v1(text: str) -> dict[str, Any]:
    det = regex_details(text)
    inp, model = _call(_SYS_V1, _TOOL_V1, text)
    atoms, dropped = _locate_sparse(text, inp.get("atoms", []))
    _parents(atoms)
    for a in atoms:
        a["is_base"] = a["parent"] is None
    _confirm_details(atoms, det)
    return {"text": text, "model": model, "atoms": atoms, "regex_details": det, "dropped": dropped}

# ============================================================ V3 self-complete
_SYS_V3 = """You are a claim segmenter for a mechanistic-interpretability experiment. The input is a
short description written by a "verbalizer" model describing ONE internal neural activation. Segment
the ENTIRE input into a FLAT, NON-OVERLAPPING partition (a tiling): concatenating every segment's "text"
in order reproduces the input character-for-character (every space, comma, quote, newline). Segments
never overlap and never nest.

THE UNIT: one claim = ONE assertion about the source. Boundary test: a claim must be expressible as a
SINGLE sentence of the form "The source <is / is about / reads as / contains / was written by> ___" that
makes exactly ONE assertion, with no "and" and no second independent clause.
- If a span packs TWO such sentences, SPLIT it. Descriptors separated by a comma, slash or semicolon that
  assert different things are different claims: a genre descriptor, a separate provenance guess, and a
  separate structural note are THREE claims, not one.
- If splitting a span would break a SINGLE such sentence, DO NOT split it. Keep together within one claim:
  the claim's own hedge ("probably", "likely", "potentially", "appears to", "roughly"); and, FOR
  DESCRIPTIVE WORDS ONLY, alternatives filling one attribute slot ("essay or journal article",
  "British or Stanford Center for European Studies", "political science/ international history" — one
  uncertain attribute, one claim).
- LITERALS ARE ALWAYS SEPARATE CLAIMS. Every quoted string is its OWN claim ("The source contains the
  quote '___'"). NEVER put two or more quoted strings in one claim, even when adjacent or joined by
  "or"/"/" — they are distinct candidate literals and must be ablatable separately. Same for distinct
  dates, numbers and values. BUT a quote together with a short phrase describing THAT ONE quote is still
  ONE claim: '"fixture availability remains" ends a clause' is one claim, '"Please translate this section
  into German" is an incomplete request beginning a specific request' is one claim. A post-quote
  description attaches to the PRECEDING quote; a pre-quote hedge ("likely", "probably") attaches to the
  FOLLOWING claim.
- Do NOT pull entities or details out of the claim they sit in (no nesting).
- CONNECTIVE IS ONLY PURE GLUE: whitespace, punctuation, and bare conjunctions/prepositions that join two
  separate claims ("and", "or", "/", "from", "between", "with", "of", ",", ":"). A HEDGE word is NEVER
  connective — it belongs to the claim it modifies (record it in that claim's "hedged"). A FRAMING or
  meta-commentary phrase that makes an assertion ("ends a clause", "implies ...", "The section covers ...",
  "is an incomplete request ...") is NEVER connective — it is its own theme claim, or part of the one quote
  it describes.
- COVER THE WHOLE INPUT. Include messy, partial or unclosed quoted fragments (a quote trailing off with
  "...") as their own quote claims. Never drop text into connective just because it is awkward.

For every non-connective claim, return "assertion": that single "The source ..." sentence it expresses.
If you cannot write the span as ONE such sentence, it is too big, so split it. This is your self-check.
Also return "hedged": the qualifier words inside the claim (empty if none).

Types: theme (genre/topic/structure/era), entity (person/place/org/title/other),
detail (quote/date/number/value), connective (glue between claims that is not itself an assertion:
commas, ":" or "or" separating items, bare conjunctions).

Tiling is mandatory and exact; flat, non-overlapping, no nesting. Respond via emit_segmentation only."""
_TOOL_V3 = {"name": "emit_segmentation", "description": "Flat non-overlapping tiling of self-complete claims.",
    "input_schema": {"type": "object", "properties": {"segments": {"type": "array", "items": {
        "type": "object", "properties": {
            "text": {"type": "string", "description": "verbatim slice; concatenation == input"},
            "type": {"type": "string", "enum": ["theme", "entity", "detail", "connective"]},
            "subtype": {"type": "string"},
            "assertion": {"type": "string", "description": "the single 'The source ...' sentence (non-connective claims)"},
            "hedged": {"type": "array", "items": {"type": "string"}}},
        "required": ["text", "type", "subtype"]}}}, "required": ["segments"]}}

_V3_TYPES = ("theme", "entity", "detail", "connective")

def _gap(seg: str, s: int, e: int) -> dict:
    """Fill an uncovered span. Whitespace/punctuation -> connective glue; but if the model
    skipped real content (letters/digits), recover it as a claim so nothing is hidden as glue."""
    if re.search(r"[A-Za-z0-9]", seg):
        ty = "detail" if ('"' in seg or "“" in seg) else "theme"
        sub = "quote (recovered)" if ty == "detail" else "residual (recovered)"
        return {"text": seg, "type": ty, "subtype": sub, "start": s, "end": e, "hedged": [], "assertion": ""}
    return {"text": seg, "type": "connective", "subtype": "gap", "start": s, "end": e, "hedged": [], "assertion": ""}

def _tile_flat(text: str, segments: list[dict], allowed: tuple = _V3_TYPES,
               fallback: str = "connective") -> list[dict]:
    base, cursor = [], 0
    for seg in segments:
        if isinstance(seg, str):                               # tolerate a bare-string segment
            seg = {"text": seg}
        elif not isinstance(seg, dict):
            continue
        ty = seg.get("type") if seg.get("type") in allowed else fallback
        idx, mlen = _find(text, seg.get("text", ""), cursor)
        if idx < 0:
            continue
        if idx > cursor:
            base.append(_gap(text[cursor:idx], cursor, idx))
        end = idx + mlen
        base.append({"text": text[idx:end], "type": ty, "subtype": seg.get("subtype", ""),
                     "start": idx, "end": end, "hedged": seg.get("hedged", []) or [],
                     "assertion": seg.get("assertion", "")})
        cursor = end
    if cursor < len(text):
        base.append(_gap(text[cursor:], cursor, len(text)))
    return base

_HEDGE_WORDS = {"likely", "probably", "potentially", "possibly", "perhaps", "maybe",
                "apparently", "presumably", "seemingly"}

def _absorb_hedges(text: str, base: list[dict]) -> list[dict]:
    """Deterministic guarantee: a connective that is nothing but a hedge word is not glue.
    Merge it into the claim it modifies (the following claim; the preceding one if last) and
    record the hedge, so no hedge is ever left sitting in a connective span."""
    n, remove = len(base), set()
    for i, b in enumerate(base):
        if b["type"] != "connective":
            continue
        w = b["text"].strip().strip('"').strip().lower()
        if w not in _HEDGE_WORDS:
            continue
        j = i + 1
        while j < n and base[j]["type"] == "connective":
            j += 1
        if j < n:                                  # attach to following claim
            base[j]["start"] = b["start"]
            base[j]["text"] = text[b["start"]:base[j]["end"]]
            if w not in base[j]["hedged"]:
                base[j]["hedged"].append(w)
            remove.update(range(i, j))
        else:                                      # hedge at the very end: attach to previous
            p = i - 1
            while p >= 0 and base[p]["type"] == "connective":
                p -= 1
            if p >= 0:
                base[p]["end"] = b["end"]
                base[p]["text"] = text[base[p]["start"]:b["end"]]
                if w not in base[p]["hedged"]:
                    base[p]["hedged"].append(w)
                remove.update(range(p + 1, i + 1))
    return [b for k, b in enumerate(base) if k not in remove]

def segment_v3(text: str) -> dict[str, Any]:
    det = regex_details(text)
    inp, model = _call(_SYS_V3, _TOOL_V3, text)
    base = _tile_flat(text, inp.get("segments", []))
    base = _absorb_hedges(text, base)
    for b in base:                    # a connective must never hide a quoted literal
        if b["type"] == "connective" and ('"' in b["text"] or "“" in b["text"]):
            b["type"], b["subtype"] = "detail", "quote (recovered)"
    atoms = [{"text": b["text"], "type": b["type"], "subtype": b["subtype"], "start": b["start"],
              "end": b["end"], "parent": None, "is_base": True, "hedged": b["hedged"],
              "assertion": b["assertion"]} for b in base]
    _confirm_details(atoms, det)
    n = len(text)
    covered = sum(b["end"] - b["start"] for b in base)
    content = sum(b["end"] - b["start"] for b in base if b["type"] in CONTENT_TYPES)
    return {"text": text, "model": model, "atoms": atoms, "regex_details": det, "dropped": [],
            "coverage": {"base_frac": covered / n, "content_frac": content / n}}

# ================================ AUDIT: two layers (partition + entity overlay)
# Used by the claim-ablation replication. Layer 1 is a theme/detail/connective tiling
# partition (named things stay inside their claim); Layer 2 is a SEPARATE, overlapping
# entity overlay (person/place/org/title/ticker/other) whose spans may sit inside a
# partition claim. Each theme/detail claim also gets a deterministic role:
#   describes = a descriptive theme; quotes = a literal the source contains;
#   predicts  = a token/phrase the source is EXPECTED to emit next (the AV's guess).
_SYS_AUDIT = """You are a claim segmenter for a mechanistic-interpretability experiment. The input is a
short description written by a "verbalizer" model describing ONE internal neural activation. Produce
TWO layers.

LAYER 1 - PARTITION ("segments"): tile the ENTIRE input into a FLAT, NON-OVERLAPPING sequence of
theme / detail / connective spans whose "text" concatenated in order reproduces the input
character-for-character (every space, comma, quote, newline). Segments never overlap and never nest.
  THE UNIT: one claim = ONE assertion about the source. Boundary test: a claim is expressible as a
  SINGLE sentence "The source <is / is about / reads as / contains / expects> ___" making exactly ONE
  assertion (no "and", no second independent clause). Split a span that packs two assertions; keep a
  span that splitting would break.
  - theme: a descriptive assertion about the source (genre / topic / structure / era / style /
    implication / what it is doing). Named things (people, places, orgs, tickers, titles) STAY inside
    the theme claim they belong to; do NOT pull them out here (they go in Layer 2).
  - detail: a literal - a quoted string, date, number or value. EVERY quoted string is its OWN detail
    claim (never group two quotes), but a quote plus the short phrase describing THAT one quote is one
    claim. A quote the source CONTAINS and a token the source is EXPECTED to emit next are BOTH detail.
  - connective: PURE glue only (whitespace, punctuation, bare "and" / "or" / "/" / "from" / "with" /
    "," / ":"). A hedge word is NEVER connective (it belongs to the claim it modifies -> record it in
    "hedged"). A phrase that makes an assertion is NEVER connective.
  For every theme/detail claim return "assertion" (the single "The source ..." sentence) and "hedged"
  (qualifier words inside it, e.g. likely / probably / expecting; empty if none).

LAYER 2 - ENTITY OVERLAY ("entities"): SEPARATELY, list every specific NAMED ENTITY mentioned anywhere
in the input - person / place / org / title / ticker / other. Return each as a verbatim substring; it
MAY (and usually will) sit inside a Layer-1 segment, including inside a quoted string. This layer
OVERLAPS Layer 1 and does not have to tile. Only specific named referents (e.g. "Georgia",
"Glencore PLC", "NYSE: GKG", "the Indians") - never generic nouns.

Layer-1 tiling is mandatory and exact. Respond via emit_layers only."""
_TOOL_AUDIT = {"name": "emit_layers",
    "description": "A theme/detail/connective tiling partition plus an overlapping entity overlay.",
    "input_schema": {"type": "object", "properties": {
        "segments": {"type": "array", "items": {"type": "object", "properties": {
            "text": {"type": "string", "description": "verbatim slice; concatenation == input"},
            "type": {"type": "string", "enum": ["theme", "detail", "connective"]},
            "subtype": {"type": "string"},
            "assertion": {"type": "string", "description": "the single 'The source ...' sentence"},
            "hedged": {"type": "array", "items": {"type": "string"}}},
            "required": ["text", "type", "subtype"]}},
        "entities": {"type": "array", "items": {"type": "object", "properties": {
            "text": {"type": "string", "description": "verbatim named-entity substring (may sit inside a segment)"},
            "subtype": {"type": "string", "enum": ["person", "place", "org", "title", "ticker", "other"]}},
            "required": ["text", "subtype"]}}},
        "required": ["segments", "entities"]}}

_AUDIT_TYPES = ("theme", "detail", "connective")
_PREDICT_RE = re.compile(r"expect|completion|continu|candidate|to complete|would complete", re.I)

def _claim_role(b: dict) -> str:
    """describes / quotes / predicts for a partition claim; glue for connective."""
    if b["type"] == "connective":
        return "glue"
    hedged = " ".join(b.get("hedged", []) or []).lower()
    probe = (b.get("assertion", "") + " " + b.get("text", "")).lower()
    if "expect" in hedged or _PREDICT_RE.search(probe):
        return "predicts"                       # the AV's guess about what comes next
    if b["type"] == "detail":
        return "quotes"                         # a literal the source contains
    return "describes"                          # a descriptive theme

def _entity_parent(span: tuple, atoms: list[dict]):
    """Smallest partition atom whose char span contains this entity span."""
    s, e = span; best = None
    for i, a in enumerate(atoms):
        if a["start"] <= s and a["end"] >= e:
            if best is None or (a["end"] - a["start"]) < (atoms[best]["end"] - atoms[best]["start"]):
                best = i
    return best

def _locate_entities(text: str, ents: list[dict], atoms: list[dict]) -> tuple[list, list]:
    """Locate each returned entity string verbatim; attach the containing partition atom.
    An entity may overlap a partition claim. Word-boundary matching avoids substring false
    positives (e.g. "STEF" inside "STEFANIE"); occurrences strictly nested inside a longer
    entity occurrence are dropped so we keep maximal named spans (e.g. drop "Baseball" where
    it sits inside "Major League Baseball", but keep a standalone "Baseball")."""
    ntext = _norm(text)
    occ, dropped = [], []                                   # occ: (start, end, text, subtype)
    for e in ents or []:
        if isinstance(e, str):                                 # tolerate a bare-string entity
            e = {"text": e}
        elif not isinstance(e, dict):
            dropped.append(e); continue
        cand = _norm(e.get("text", "")).strip().strip('"').strip()
        if len(cand) < 2:
            dropped.append(e); continue
        pre = r"\b" if cand[:1].isalnum() else ""
        suf = r"\b" if cand[-1:].isalnum() else ""
        try:
            ms = list(re.finditer(pre + re.escape(cand) + suf, ntext))
        except re.error:
            ms = list(re.finditer(re.escape(cand), ntext))
        if not ms:
            dropped.append(e); continue
        for m in ms:
            occ.append((m.start(), m.start() + len(cand), text[m.start():m.start() + len(cand)],
                        e.get("subtype", "other")))
    occ.sort(key=lambda o: (o[0], -(o[1] - o[0])))          # longest-first at each start
    kept = []
    for s, en, t, st in occ:
        if any(k0 <= s and k1 >= en and (k1 - k0) > (en - s) for k0, k1, _, _ in kept):
            continue                                        # strictly nested in a longer entity
        if any(k0 == s and k1 == en and kt == t for k0, k1, kt, _ in kept):
            continue                                        # exact duplicate
        kept.append((s, en, t, st))
    groups: dict = {}
    for s, en, t, st in kept:
        groups.setdefault((t, st), []).append((s, en))
    located = []
    for (t, st), spans in groups.items():
        spans.sort()
        s0, e0 = spans[0]
        located.append({"text": t, "subtype": st, "start": s0, "end": e0,
                        "spans": [[a, b] for a, b in spans], "n_occ": len(spans),
                        "parent": _entity_parent((s0, e0), atoms)})
    located.sort(key=lambda a: (a["start"], a["text"]))
    return located, dropped

def segment_audit(text: str) -> dict[str, Any]:
    """Two-layer segmentation for the claim-ablation replication (see module note above)."""
    det = regex_details(text)
    inp, model = _call(_SYS_AUDIT, _TOOL_AUDIT, text)
    base = _tile_flat(text, inp.get("segments", []), allowed=_AUDIT_TYPES, fallback="theme")
    base = _absorb_hedges(text, base)
    for b in base:                    # a connective must never hide a quoted literal
        if b["type"] == "connective" and ('"' in b["text"] or "“" in b["text"]):
            b["type"], b["subtype"] = "detail", "quote (recovered)"
    atoms = [{"text": b["text"], "type": b["type"], "subtype": b["subtype"], "start": b["start"],
              "end": b["end"], "parent": None, "is_base": True, "hedged": b["hedged"],
              "assertion": b["assertion"], "role": _claim_role(b)} for b in base]
    _confirm_details(atoms, det)
    entities, dropped_ent = _locate_entities(text, inp.get("entities", []), atoms)
    n = len(text)
    covered = sum(a["end"] - a["start"] for a in atoms)
    content = sum(a["end"] - a["start"] for a in atoms if a["type"] in CONTENT_TYPES)
    return {"text": text, "model": model, "atoms": atoms, "entities": entities,
            "dropped_entities": dropped_ent, "regex_details": det, "dropped": [],
            "coverage": {"base_frac": covered / n, "content_frac": content / n}}

# ==================================== V2 prompt (archived: full coverage, 6 types)
# atoms_v2.json was produced by this exhaustive 6-type tiling prompt, which predates the
# v1/v3 split. It is kept verbatim so the V2 tab can display the prompt that generated its
# preserved data. It is NOT re-run (V2 data is loaded from atoms_v2.json).
_SYS_V2 = """You are an exhaustive claim segmenter for a mechanistic-interpretability experiment.
The input is a short description written by a "verbalizer" model that is describing ONE internal
neural activation. Segment the ENTIRE input into an ordered list of contiguous spans that TILE the
text with no gaps and no overlaps: concatenating every segment's "text" in order, exactly as given,
must reproduce the input character-for-character, including every space, comma, quote and newline.
Then type each segment. Cover EVERYTHING, including qualifier and glue words.

Base types:
- theme: a descriptive claim about the source's genre/topic/structure/era. subtypes: genre, topic, structure, era.
- entity: a specific named thing. subtypes: person, place, org, title, other.
- detail: a literal value or quoted material. subtypes: quote, date, number, value.
- hedge: a qualifier expressing (un)certainty or approximation, e.g. "likely", "potentially", "probably", "may", "appears to", "roughly". subtypes: epistemic, approximator.
- framing: the verbalizer's meta-description of what the source does, e.g. "The sentence outlines", "establishes", "implies", "citing", "requesting". subtypes: evidential, attribution, speech_act.
- connective: glue with no standalone content: conjunctions, prepositions, punctuation, discourse markers ("or", "and", "with", ",", ":", "e.g."). subtypes: conjunction, punctuation, marker.

Rules:
1. TILING is mandatory. Segments are contiguous and in reading order; their concatenation equals the input verbatim; every character is in exactly one segment.
2. Content claims (theme/entity/detail) are the smallest self-complete unit: a noun phrase like "oil tanker market note" is ONE theme. Do not merge distinct claims; do not over-split content into single words.
3. Qualifiers ("likely", "potentially") and framing verbs ("outlines", "implies") are their OWN segments and are NEVER absorbed into a content claim. This is important.
4. NESTED atoms: if a content segment contains a named entity or a literal detail, list it under that segment's "nested" (type entity or detail only, verbatim substring of the segment). Nested atoms may overlap the segment and do not affect the tiling.
5. Respond with the emit_segmentation tool only."""

def _promptmeta(system: str, tool: str, types: list[str], note: str = "") -> dict:
    """Per-version prompt metadata surfaced in the GUI (system prompt + call shape)."""
    return {"system": system, "tool": tool, "types": types, "note": note}

# ------------------------------------------------------------------- __main__
def _runver(examples, fn, label):
    out = []
    for i, e in enumerate(examples):
        print(f"  [{i+1}/{len(examples)}] {label}: {e['format']}", flush=True)
        out.append({**e, **fn(e["av"])})
    return out

if __name__ == "__main__":
    ex = json.loads((HERE / "examples.json").read_text())
    v2 = json.loads((HERE / "atoms_v2.json").read_text())   # preserved "current" tab, verbatim
    print("V1 (sparse, original) ..."); v1 = _runver(ex, segment_v1, "v1")
    print("V3 (self-complete, current edits) ..."); v3 = _runver(ex, segment_v3, "v3")
    # latest version first (reverse-chronological); numeric label prefixes stay tied to version identity
    DECOMP = {"versions": [
        {"id": "v3", "label": "3 · Self-complete (current edits)",
         "desc": "Full-coverage, non-intersecting, self-complete claims: hedges and directly-relevant framing stay INSIDE the claim they modify, quotes absorb their description, no nesting. Hedge words are tracked per claim.",
         "prompt": _promptmeta(_SYS_V3, _TOOL_V3["name"], ["theme", "entity", "detail", "connective"]),
         "data": v3},
        {"id": "v2", "label": "2 · Full coverage · 6 types",
         "desc": "Full-coverage tiling where hedge, framing and connective are their OWN claims and entities are nested inside themes.",
         "prompt": _promptmeta(_SYS_V2, "emit_segmentation",
                               ["theme", "entity", "detail", "hedge", "framing", "connective"],
                               "archived prompt; V2 data is preserved verbatim from the original 6-type run"),
         "data": v2},
        {"id": "v1", "label": "1 · Sparse (original)",
         "desc": "First pass: only salient spans (theme/entity/detail) are extracted, entities nested. Most of the text is left unsegmented, so this does NOT tile the output.",
         "prompt": _promptmeta(_SYS_V1, _TOOL_V1["name"], ["theme", "entity", "detail"]),
         "data": v1},
    ]}
    (HERE / "atoms.json").write_text(json.dumps(DECOMP, indent=2, ensure_ascii=False))
    (HERE / "atoms_data.js").write_text("window.DECOMP = " + json.dumps(DECOMP, ensure_ascii=False) + ";")
    print(f"\nwrote 3 versions -> atoms.json + atoms_data.js")
