"""Decoy generation + text filters for the cycle-consistency test (spec Step 5b).

For each real claim B of an atom-mode run, produce g candidate DECOYS: sibling
claims the AV never wrote, same kind as B, written to splice into B's exact slot
in the explanation. Then apply the two LLM TEXT filters:

  filter 1  (absence)      the decoy's content must be absent from the input -
                            candidates that name, paraphrase, or are directly
                            supported by any span of the visible document are
                            rejected; so are paraphrases of B itself.
  filter 4a (naturalness)   the spliced explanation z*_m must stay fluent,
                            coherent and structurally parallel to z.

Filters 2 (matched removed mass), 3 (matched prior propensity) and 4b (FVE band)
are mechanical and GPU-side: cycle_consistency.py.

JUDGE-FREE MEASUREMENT DISCIPLINE: everything here runs BEFORE any measurement
exists. The generator and both judges see only text (claim, explanation, source
document); no Delta-logp, no readings, no verdicts, ever.

Conventions mirror claim_taxonomy.py (pinned models, temperature 0, forced tool
call, refusal fallback, atomic cache writes) and segment_run.py (resumable
worker pool). Cache: decoys/<run_id>/<idx>_<claim_i>.<model>.json - idempotent,
one file per (activation, claim).

Calibration split (S2): every Nth ok record of the run, up to --claims-per-act
content claims per activation sampled deterministically with a per-doc keyed RNG
(same (seed, doc_id) -> same claims regardless of process count; the repo's
stage-0 invariant applied here).

Usage:
  python3 decoy_gen.py --run qwen7b-cp6-fineweb-atoms --calib-every 16 \
      --claims-per-act 3 --g 12 --workers 4            # generate + judge
  python3 decoy_gen.py --run ... --dry-run             # print prompts, no API
  python3 decoy_gen.py --run ... --report              # aggregate cache stats
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent           # .../athena/repl
ATHENA = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from claim_taxonomy import MODELS, _api_key, write_cache  # noqa: E402

PROMPT_VERSION = "d4"       # d2: whitespace-preserving splice (naturalness judge
                            # sees seam-clean variants); d3: truncation is inferred
                            # and disclosed to the absence judge, hidden-read-window
                            # warning added (audit 2026-07-21); d4: candidates may
                            # mirror the target claim's OWN non-ASCII characters
                            # (hygiene relaxed to the claim's char set - the d3
                            # blanket ASCII rule killed every faithful candidate
                            # for claims containing an em-dash, idx 32.15), g=24
                            # signed package 2026-07-21
NATURAL_PASS = 4            # filter 4a: judge score >= this passes (recorded in calib)
TEXT_CAP_CHARS = 6000       # extract_corpus.TEXT_CAP_CHARS - records.jsonl drops the
                            # manifest's text_truncated flag, so truncation must be
                            # re-inferred here from len(source_text) == cap
_ASCII_RE = re.compile(r"^[\x09\x0a\x0d\x20-\x7e]*$")
CONTENT_TYPES = ("theme", "detail")


# ── deterministic calibration-split claim sampling ───────────────────────────

def doc_rng(seed: int, doc_id: str) -> np.random.Generator:
    """Per-doc keyed RNG: same (seed, doc_id) -> same stream, independent of
    chunking or process count (the stage-0 reproducibility convention)."""
    key = hashlib.sha256(f"{seed}:{doc_id}".encode()).digest()[:8]
    return np.random.default_rng(int.from_bytes(key, "big"))


def calib_claims(rec: dict, per_act: int, seed: int) -> list[int]:
    """Pick up to per_act content-claim indices from one record, deterministic
    and type-stratified: keyed shuffle within each type, then round-robin
    theme/detail so both forms contribute to every calibration threshold."""
    pools: dict[str, list[int]] = {t: [] for t in CONTENT_TYPES}
    for i, s in enumerate(rec.get("snippet_results") or []):
        if s.get("located") and s.get("scored") and s.get("type") in pools:
            pools[s["type"]].append(i)
    rng = doc_rng(seed, rec.get("doc_id", str(rec.get("idx"))))
    for t in CONTENT_TYPES:
        rng.shuffle(pools[t])
    picked, order = [], list(CONTENT_TYPES)
    while len(picked) < per_act and any(pools[t] for t in order):
        for t in order:
            if pools[t] and len(picked) < per_act:
                picked.append(pools[t].pop(0))
    return sorted(picked)


# ── candidate hygiene (pure, CPU) ────────────────────────────────────────────

LEN_RATIO_BAND = (0.4, 2.5)   # mechanical garbage guard: single-char / runaway
                              # candidates must not reach the LLM judges (both
                              # judges passed bare '[' when a malformed payload
                              # leaked through - seen live, idx 352 claim 8).
                              # Verified a no-op on all 876 real candidates of
                              # the d3 calibration pool (min 0.542, max 2.087).


def clean_candidates(cands: list[str], claim_text: str) -> list[dict]:
    """Post-process raw generations. Never silently drop: every candidate comes
    back with kept=True/False + reason, so the audit table shows the whole pool.
    Rules: strip + collapse whitespace/newlines (the decoy fills an in-line
    slot), non-empty, ASCII-only EXCEPT the target claim's own non-ASCII
    characters (d4: a claim written with an em-dash or curly quote needs decoys
    that mirror it - the blanket rule zeroed idx 32.15's whole pool; a CJK or
    otherwise-foreign candidate still dies here, keeping the downstream
    injection smell test meaningful), length within LEN_RATIO_BAND of B, not
    equal to B (case-folded), no case-folded duplicates."""
    out, seen = [], set()
    b_norm = " ".join(claim_text.split()).casefold()
    claim_nonascii = {ch for ch in claim_text if not _ASCII_RE.match(ch)}
    for raw in cands:
        text = " ".join(str(raw).split())
        ratio = len(text) / max(len(claim_text), 1)
        row = dict(text=text, kept=True, reason="", len_ratio=round(ratio, 3))
        norm = text.casefold()
        foreign = {ch for ch in text if not _ASCII_RE.match(ch)} - claim_nonascii
        if not text:
            row.update(kept=False, reason="empty")
        elif foreign:
            row.update(kept=False, reason="non-ascii")
        elif not (LEN_RATIO_BAND[0] <= ratio <= LEN_RATIO_BAND[1]):
            row.update(kept=False, reason="length-ratio")
        elif norm == b_norm:
            row.update(kept=False, reason="equals claim")
        elif norm in seen:
            row.update(kept=False, reason="duplicate")
        else:
            seen.add(norm)
        out.append(row)
    return out


# ── prompt builders ──────────────────────────────────────────────────────────

_GEN_SYSTEM = """You write placebo test stimuli for a mechanistic-interpretability experiment.

A "verbalizer" model looked at ONE internal activation of a language model reading a web \
document and wrote a short explanation of what that activation encodes. The explanation was \
split into atomic claims. For ONE target claim you will write DECOYS: sibling claims the \
verbalizer never wrote, to be spliced into the target's exact character slot as experimental \
placebos.

Every decoy must:
1. be the SAME KIND of claim as the target: same type and subtype. A genre claim stays a \
genre claim, a quote claim asserts some other verbatim string, a date claim asserts some \
other date, and so on.
2. fit the slot grammatically: it replaces the target's exact characters, so it must read \
naturally between the given PREFIX and SUFFIX. Match the target's capitalization, its \
mid-sentence or sentence-initial position, and its trailing punctuation (or lack of it).
3. be PLAUSIBLE as something a verbalizer could write about SOME web document of the same \
general kind, in the same terse register as the rest of the explanation.
4. be ABSENT from this particular document: nothing in the SOURCE DOCUMENT, SOURCE WINDOW or \
ACTUAL CONTINUATION states, implies, or supports it. Do not reuse the document's names, \
numbers, quoted strings, or specific topics.
5. NOT restate or paraphrase the target claim, and not duplicate any OTHER claim of the \
explanation.
6. differ from the other decoys in substance, not just wording.
7. stay within about 25 percent of the target's character length.
8. be plain-ASCII English on a single line - except that special characters appearing in \
the target claim itself (an em-dash, curly quotes, an accent) may and should be mirrored \
where the slot's format calls for them.

The decoys are placebos: each should look exactly as legitimate as the target, just about \
content this document does not contain."""

_GEN_TOOL = {
    "name": "emit_decoys",
    "description": "Emit the decoy candidates, numbered from 1, in order.",
    "input_schema": {
        "type": "object",
        "properties": {
            "decoys": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "m": {"type": "integer"},
                        "text": {"type": "string"},
                    },
                    "required": ["m", "text"],
                },
            }
        },
        "required": ["decoys"],
    },
}

_ABSENCE_SYSTEM = """You are a strict screening judge for placebo test stimuli in a \
mechanistic-interpretability experiment.

A placebo candidate is only usable if its content is ABSENT from the document under test. \
You receive the visible document (possibly truncated), a source window around the model's \
read position, the ORIGINAL claim the candidate would replace, and numbered candidates.

Label each candidate:
  absent      nothing in the visible document states, implies, or directly supports the \
candidate's content; its names, numbers, quoted strings and specific topics do not occur.
  present     the candidate names, paraphrases, or is directly supported by SOME span of the \
visible document, even loosely or with different wording.
  paraphrase  the candidate restates or closely paraphrases the ORIGINAL claim itself \
(regardless of the document).

Judge content, not style. When uncertain between absent and present, choose present: the \
experiment discards those candidates, and a false "absent" is costlier than a false \
"present". If a NOTE says part of what the model read is hidden (the read window lies \
beyond the truncated document), a candidate whose content could plausibly appear in that \
hidden text cannot be certified absent: label it present. For each candidate give short \
evidence: the document span that supports a "present" label, or "no support found" for \
absent."""

_ABSENCE_TOOL = {
    "name": "judge_absence",
    "description": "One label per candidate, in input order.",
    "input_schema": {
        "type": "object",
        "properties": {
            "candidates": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "m": {"type": "integer"},
                        "label": {"type": "string",
                                  "enum": ["absent", "present", "paraphrase"]},
                        "evidence": {"type": "string"},
                    },
                    "required": ["m", "label", "evidence"],
                },
            }
        },
        "required": ["candidates"],
    },
}

_NATURAL_SYSTEM = """You judge the textual naturalness of edited explanations for a \
mechanistic-interpretability experiment.

You receive an ORIGINAL explanation and numbered EDITED variants. Each variant replaced \
exactly one claim with a different claim of the same kind. The replacement is SUPPOSED to \
change content, so content disagreements with neighbouring claims are acceptable. Penalize \
only breaks in grammar, reference, or format.

Rate each variant 1-5 against the original for fluency (grammar), coherence (no dangling \
references or broken anaphora introduced by the edit), and structural well-formedness (same \
clause shape, punctuation and list structure as the original):
  5  reads as naturally as the original
  4  minor roughness a careful reader might notice
  3  a noticeable seam: a grammar or reference glitch at the splice point
  2  a clearly broken sentence or dangling reference
  1  ungrammatical or incoherent
Also name the issue in a few words (or "none")."""

_NATURAL_TOOL = {
    "name": "judge_naturalness",
    "description": "One rating per variant, in input order.",
    "input_schema": {
        "type": "object",
        "properties": {
            "variants": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "m": {"type": "integer"},
                        "score": {"type": "integer", "minimum": 1, "maximum": 5},
                        "issue": {"type": "string"},
                    },
                    "required": ["m", "score", "issue"],
                },
            }
        },
        "required": ["variants"],
    },
}


def _source_block(rec: dict) -> str:
    """The document as the judges/generator may see it. source_text is capped at
    TEXT_CAP_CHARS by extract_corpus; the manifest's text_truncated flag is NOT
    copied into records.jsonl (run_claim_ablation keeps only the text), so
    truncation is re-inferred from the cap and disclosed explicitly - absence is
    judged against the VISIBLE document, a logged hole, not a silent one. When
    the read window itself lies beyond the visible text (context_left comes from
    the UNCAPPED text, so containment is an exact test), say so: the hidden span
    between them produced part of h and cannot be screened."""
    src = rec.get("source_text") or ""
    left = rec.get("context_left") or ""
    right = rec.get("context_right") or ""
    tok = rec.get("read_token") or ""
    if tok and left.endswith(tok):
        window = left[: -len(tok)] + "<<<" + tok + ">>>"
    else:
        window = left + f"\n[read token: {tok!r}]"
    truncated = bool(rec.get("text_truncated")) or len(src) >= TEXT_CAP_CHARS
    trunc = (f" (TRUNCATED at {TEXT_CAP_CHARS} characters: "
             "the document continues beyond what is shown)") if truncated else ""
    probe = left[-200:]
    window_hidden = bool(probe) and probe not in src
    lines = [
        f"SOURCE DOCUMENT{trunc}:",
        src or "(unavailable)",
        "",
        "SOURCE WINDOW (end of the model's prefix; read token wrapped in <<< >>>):",
        window,
        "",
    ]
    if window_hidden:
        lines += [
            "NOTE: this window lies beyond the truncated document above - the text "
            "between the end of the visible document and this window is NOT shown, "
            "yet the model read it. Content in that hidden span cannot be checked "
            "against the document.",
            "",
        ]
    lines += [
        "ACTUAL CONTINUATION (text that really follows the read token):",
        right,
    ]
    return "\n".join(lines)


def _slot_context(z_inner: str, span, width: int = 100) -> tuple[str, str]:
    """The characters immediately before/after B's slot, for requirement 2."""
    s0, e0 = span
    return z_inner[max(0, s0 - width):s0], z_inner[e0:e0 + width]


def build_gen_content(rec: dict, snip: dict, taxo: str | None, g: int) -> str:
    z = rec["z_inner"]
    span = snip["char_span"]
    pre, suf = _slot_context(z, span)
    others = [s["text"] for s in rec.get("snippet_results") or []
              if s is not snip and s.get("type") in CONTENT_TYPES]
    lines = [
        _source_block(rec),
        "",
        "FULL EXPLANATION (the target claim is marked with <<TARGET>> ... <</TARGET>>):",
        z[:span[0]] + "<<TARGET>>" + z[span[0]:span[1]] + "<</TARGET>>" + z[span[1]:],
        "",
        f"TARGET CLAIM ({snip.get('type')}"
        + (f", subtype {snip.get('subtype')}" if snip.get("subtype") else "")
        + (f", class {taxo}" if taxo else "") + "):",
        snip["text"],
        f"normalized assertion: {snip.get('assertion') or '(none)'}",
        f"target length: {span[1] - span[0]} characters",
        "",
        f"SLOT PREFIX (text just before the slot): ...{pre!r}",
        f"SLOT SUFFIX (text just after the slot): {suf!r}...",
        "",
        "OTHER CLAIMS in the explanation (do not duplicate any):",
        *[f"- {t}" for t in others],
        "",
        f"Write exactly {g} decoys.",
    ]
    return "\n".join(lines)


def build_absence_content(rec: dict, snip: dict, cands: list[dict]) -> str:
    lines = [
        _source_block(rec),
        "",
        "ORIGINAL CLAIM (the one each candidate would replace):",
        snip["text"],
        "",
        f"CANDIDATES to screen ({sum(c['kept'] for c in cands)} total):",
    ]
    k = 0
    for c in cands:
        if c["kept"]:
            k += 1
            lines.append(f"{k}. {c['text']}")
    return "\n".join(lines)


def build_natural_content(rec: dict, snip: dict, cands: list[dict]) -> str:
    from cycle_consistency import splice
    z = rec["z_inner"]
    lines = ["ORIGINAL EXPLANATION:", z, "", "EDITED VARIANTS:"]
    k = 0
    for c in cands:
        if c["kept"]:
            k += 1
            z_star, _ = splice(z, snip["char_span"], c["text"])
            lines += [f"--- variant {k} ---", z_star]
    return "\n".join(lines)


# ── the three API calls for one claim ────────────────────────────────────────

def _payload_list(payload: dict, key: str, what: str) -> list:
    """Forced-tool payloads are schema-guided but NOT schema-validated: the
    model can emit the array double-encoded as a JSON string (seen live at
    temperature 0: idx 352 claim 8 - a naive iteration turned the 1524-char
    string into character soup, and both LLM judges then passed bare
    punctuation). Decode that case; anything else non-list fails closed."""
    val = payload.get(key)
    if isinstance(val, str):
        try:
            val = json.loads(val)
        except json.JSONDecodeError:
            pass
    if not isinstance(val, list):
        raise RuntimeError(f"{what}: payload[{key!r}] is "
                           f"{type(payload.get(key)).__name__}, not a list")
    return val


def _join_by_m(items, n_kept: int, what: str) -> dict[int, dict]:
    """Fail-closed join of a judge payload onto candidates 1..n_kept. The judge
    echoes each candidate's index m; a misnumbered echo (0-based, duplicated,
    out of range) would silently shift every label onto the WRONG candidate -
    fail-open in the two filters that gate the pool (audit 2026-07-21). Require
    the echoed set to be exactly {1..n_kept}; anything else raises, the claim
    is reported FAILED, nothing is cached, and the resume retries it."""
    by_m = {it.get("m"): it for it in (items or []) if isinstance(it, dict)}
    if set(by_m) != set(range(1, n_kept + 1)):
        raise RuntimeError(
            f"{what} judge misnumbered candidates: echoed m-set "
            f"{sorted(by_m, key=str)} != 1..{n_kept}"
            + (f" ({sum(1 for it in items if not isinstance(it, dict))} non-dict "
               "items dropped)" if any(not isinstance(it, dict)
                                       for it in (items or [])) else ""))
    return by_m


_TRANSIENT_TRIES = 3          # our retries of the PINNED model on transient errors,
_TRANSIENT_SLEEP = (15.0, 45.0)   # on top of the SDK's own internal retries


def _transient(e: Exception) -> bool:
    """Rate limits / overload / timeouts / connection drops - retryable on the
    SAME model. Everything else (400 invalid request, auth) is not."""
    import anthropic
    if isinstance(e, anthropic.APIConnectionError):       # includes timeouts
        return True
    st = getattr(e, "status_code", None)
    return st in (408, 409, 429) or (isinstance(st, int) and st >= 500)


def _call(client, model_key: str, system: str, tool: dict, content: str,
          max_tokens: int = 4000):
    """One forced-tool call. Transient errors (429/529/timeouts that survive the
    SDK's internal retries under a busy worker pool) retry the PINNED model with
    backoff; the cross-model fallback fires ONLY on a safety refusal. A rate
    limit can therefore never silently downgrade sonnet -> haiku (audit
    2026-07-21); if the pinned model stays down the claim fails loudly, is not
    cached, and the resume retries it."""
    order = [model_key] + [m for m in MODELS if m != model_key]
    last = None
    for mk in order:
        refused = False
        for attempt in range(_TRANSIENT_TRIES):
            try:
                msg = client.messages.create(
                    model=MODELS[mk], max_tokens=max_tokens, temperature=0,
                    system=system, tools=[tool],
                    tool_choice={"type": "tool", "name": tool["name"]},
                    messages=[{"role": "user", "content": content}])
            except Exception as e:                        # pragma: no cover - network
                if not _transient(e):
                    raise
                last = e
                if attempt + 1 < _TRANSIENT_TRIES:
                    time.sleep(_TRANSIENT_SLEEP[min(attempt,
                                                    len(_TRANSIENT_SLEEP) - 1)])
                continue
            if msg.stop_reason == "refusal":
                last = RuntimeError(f"{MODELS[mk]} refused")
                refused = True
                break                                     # -> next model
            if msg.stop_reason == "max_tokens":
                # a truncated forced-tool payload is garbage-in for the joins;
                # deterministic at temperature 0, so retrying cannot fix it -
                # fail the claim loudly (not cached) so the budget gets bumped
                raise RuntimeError(
                    f"{MODELS[mk]} hit max_tokens={max_tokens} mid-payload "
                    f"({tool['name']}); raise the budget for this call")
            payload = {}
            for b in msg.content:
                if b.type == "tool_use" and isinstance(b.input, dict):
                    payload = b.input
            usage = {"input_tokens": getattr(msg.usage, "input_tokens", None),
                     "output_tokens": getattr(msg.usage, "output_tokens", None)}
            return payload, mk, usage
        if not refused:                                   # transients exhausted:
            raise RuntimeError(                           # do NOT switch models
                f"{MODELS[mk]}: transient failures exhausted "
                f"({_TRANSIENT_TRIES} tries): {last}")
    raise RuntimeError(f"API call failed after refusal fallback: {last}")


def gen_decoys_for_claim(client, rec: dict, claim_i: int, taxo: str | None,
                         g: int, model_key: str, run_id: str) -> dict:
    """Generate g candidates for claim claim_i, run hygiene + filter 1 (absence)
    + filter 4a (naturalness text judge). Returns the cache object; every
    candidate keeps per-filter outcomes - nothing is silently dropped."""
    snip = rec["snippet_results"][claim_i]
    usages = {}

    # token budgets scale with g (d3 ran g=12 inside the old flat 4000/2000
    # budgets; g=24 roughly doubles every payload, and _call now fails loudly
    # on stop_reason=="max_tokens" instead of caching a truncated payload)
    payload, used_g, usages["gen"] = _call(
        client, model_key, _GEN_SYSTEM, _GEN_TOOL,
        build_gen_content(rec, snip, taxo, g),
        max_tokens=max(4000, 300 * g))
    # the generator join is ORDER-based (no label attaches to a candidate yet),
    # so a bare-string item is recoverable here - unlike the judge joins, where
    # a non-dict item breaks the m-keyed label attachment and must fail closed
    raw = [d.get("text", "") if isinstance(d, dict) else str(d)
           for d in _payload_list(payload, "decoys", "generator")]
    cands = clean_candidates(raw, snip["text"])

    kept = [c for c in cands if c["kept"]]
    used_a = used_n = None
    if kept:
        payload, used_a, usages["absence"] = _call(
            client, model_key, _ABSENCE_SYSTEM, _ABSENCE_TOOL,
            build_absence_content(rec, snip, cands),
            max_tokens=max(4000, 200 * len(kept)))
        by_m = _join_by_m(_payload_list(payload, "candidates", "absence"),
                          len(kept), "absence")
        for k, c in enumerate((c for c in cands if c["kept"]), start=1):
            a = by_m[k]
            c["absence"] = a.get("label") or "missing"
            c["absence_evidence"] = str(a.get("evidence") or "")[:300]

        payload, used_n, usages["natural"] = _call(
            client, model_key, _NATURAL_SYSTEM, _NATURAL_TOOL,
            build_natural_content(rec, snip, cands),
            max_tokens=max(2000, 100 * len(kept)))
        by_m = _join_by_m(_payload_list(payload, "variants", "naturalness"),
                          len(kept), "naturalness")
        for k, c in enumerate((c for c in cands if c["kept"]), start=1):
            v = by_m[k]
            c["natural_score"] = v.get("score")
            c["natural_issue"] = str(v.get("issue") or "")[:200]
            c["natural_pass"] = bool(isinstance(v.get("score"), int)
                                     and v["score"] >= NATURAL_PASS)

    n_text_pass = sum(1 for c in cands if c["kept"]
                      and c.get("absence") == "absent" and c.get("natural_pass"))
    return {
        "run_id": run_id, "idx": rec["idx"], "claim_i": claim_i,
        "claim": {k: snip.get(k) for k in
                  ("text", "type", "subtype", "role", "assertion", "char_span")},
        "taxo": taxo, "g_requested": g, "n_raw": len(raw),
        "n_text_pass": n_text_pass, "natural_pass_min": NATURAL_PASS,
        "model": model_key,
        "model_used": {"gen": MODELS.get(used_g), "absence": MODELS.get(used_a),
                       "natural": MODELS.get(used_n)},
        "prompt_version": PROMPT_VERSION, "ts": time.time(), "usage": usages,
        "z_inner_sha": hashlib.sha256(rec["z_inner"].encode()).hexdigest()[:16],
        "candidates": cands,
    }


def decoy_cache_path(out_dir: pathlib.Path, run_id: str, idx: int, claim_i: int,
                     model_key: str) -> pathlib.Path:
    return out_dir / run_id / f"{idx}_{claim_i}.{model_key}.json"


def cache_is_current(path: pathlib.Path, g: int) -> bool:
    """Resume check. A cache file counts only if it was produced by THIS prompt
    version with THIS candidate count - a bumped prompt or a different --g must
    regenerate, never silently reuse the stale pool (the staleness axis that
    z_inner_sha does not cover; audit 2026-07-21). Unreadable files regenerate."""
    if not path.exists():
        return False
    try:
        obj = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    return (obj.get("prompt_version") == PROMPT_VERSION
            and obj.get("g_requested") == g)


# ── taxonomy-label join (reuse the cached Step-5b type classifier output) ─────

def taxo_label(taxo_dir: pathlib.Path, run_id: str, idx: int, claim_i: int,
               model_key: str = "sonnet") -> str | None:
    p = taxo_dir / run_id / f"{idx}.{model_key}.json"
    if not p.exists():
        return None
    try:
        obj = json.loads(p.read_text())
    except json.JSONDecodeError:
        return None
    for c in obj.get("claims") or []:
        if c.get("i") == claim_i + 1:            # taxonomy files are 1-based
            return c.get("taxo")
    return None


# ── driver ───────────────────────────────────────────────────────────────────

def select_work(records: list[dict], every: int, per_act: int, seed: int,
                only_idx: set[int] | None) -> list[tuple[dict, int]]:
    ok = [r for r in records if r.get("status") == "ok"]
    picked = ok[::every] if every > 1 else ok
    if only_idx:
        picked = [r for r in ok if r["idx"] in only_idx]
    work = []
    for r in picked:
        for ci in calib_claims(r, per_act, seed):
            work.append((r, ci))
    return work


def report(out_dir: pathlib.Path, run_id: str) -> None:
    files = sorted((out_dir / run_id).glob("*.json")) if (out_dir / run_id).exists() else []
    if not files:
        print("no decoy cache yet")
        return
    rows = [json.loads(p.read_text()) for p in files]
    n_raw = sum(r["n_raw"] for r in rows)
    kept = [c for r in rows for c in r["candidates"] if c["kept"]]
    versions = sorted({str(r.get("prompt_version")) for r in rows})
    used: dict[str, int] = {}
    for r in rows:
        for m in (r.get("model_used") or {}).values():
            if m:
                used[m] = used.get(m, 0) + 1
    print(f"prompt_versions: {versions}   model_used (calls): {used}")
    if len(versions) > 1:
        print("  [warn] MIXED prompt versions in cache - stale files present")
    absent = [c for c in kept if c.get("absence") == "absent"]
    para = [c for c in kept if c.get("absence") == "paraphrase"]
    nat = [c for c in kept if c.get("natural_pass")]
    both = [c for c in kept if c.get("absence") == "absent" and c.get("natural_pass")]
    per_claim = [r["n_text_pass"] for r in rows]
    print(f"claims: {len(rows)}   candidates raw: {n_raw}   hygiene-kept: {len(kept)}")
    print(f"filter1 absent: {len(absent)}/{len(kept)}   present: "
          f"{sum(1 for c in kept if c.get('absence') == 'present')}   paraphrase: {len(para)}")
    scores = [c.get("natural_score") for c in kept if isinstance(c.get("natural_score"), int)]
    if scores:
        print(f"filter4a scores: mean {np.mean(scores):.2f}  "
              f"dist {dict(zip(*np.unique(scores, return_counts=True)))}  "
              f"pass(>= {NATURAL_PASS}): {len(nat)}/{len(kept)}")
    print(f"both text filters: {len(both)}/{len(kept)}   per-claim text survivors: "
          f"min {min(per_claim)}  median {int(np.median(per_claim))}  max {max(per_claim)}")
    lows = sorted(rows, key=lambda r: r["n_text_pass"])[:8]
    print("lowest-survivor claims:")
    for r in lows:
        print(f"  idx {r['idx']} claim {r['claim_i']} [{r['claim']['type']}] "
              f"survivors {r['n_text_pass']}  text: {r['claim']['text'][:70]!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--runs-dir", default=str(ATHENA / "runs"))
    # d4/g24 pool goes to a NEW versioned dir: cache_is_current treats d3
    # caches as stale (prompt_version mismatch), so pointing a d4 run at the
    # frozen decoys/ dir would OVERWRITE the d3 pool. Never default there.
    ap.add_argument("--out-dir", default=str(ATHENA / "decoys_d4g24"))
    ap.add_argument("--taxo-dir", default=str(ATHENA / "taxonomy"))
    ap.add_argument("--model", default="sonnet", choices=sorted(MODELS))
    ap.add_argument("--calib-every", type=int, default=16,
                    help="take every Nth ok record (1 = all)")
    ap.add_argument("--claims-per-act", type=int, default=3)
    ap.add_argument("--g", type=int, default=24,
                    help="candidates per claim (24 = signed package 2026-07-21)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-claims", type=int, default=0, help="cost cap (0 = none)")
    ap.add_argument("--idx", type=int, nargs="*", help="explicit activation idxs")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the three prompts for the first claim; no API")
    ap.add_argument("--report", action="store_true", help="aggregate cache stats")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    if args.report:
        report(out_dir, args.run)
        return

    recs = [json.loads(l) for l in
            (pathlib.Path(args.runs_dir) / args.run / "records.jsonl").open()]
    work = select_work(recs, args.calib_every, args.claims_per_act, args.seed,
                       set(args.idx) if args.idx else None)
    if args.max_claims:
        work = work[:args.max_claims]
    taxo_dir = pathlib.Path(args.taxo_dir)
    print(f"[decoy_gen] run={args.run} claims={len(work)} g={args.g} "
          f"model={args.model} prompt={PROMPT_VERSION}")

    if args.dry_run:
        rec, ci = work[0]
        snip = rec["snippet_results"][ci]
        taxo = taxo_label(taxo_dir, args.run, rec["idx"], ci)
        fake = [dict(text=f"placeholder decoy {k}", kept=True) for k in range(1, 3)]
        print(f"\n=== GEN prompt (idx {rec['idx']} claim {ci}) ===\n{_GEN_SYSTEM}\n")
        print(build_gen_content(rec, snip, taxo, args.g))
        print(f"\n=== ABSENCE prompt ===\n{_ABSENCE_SYSTEM}\n")
        print(build_absence_content(rec, snip, fake))
        print(f"\n=== NATURALNESS prompt ===\n{_NATURAL_SYSTEM}\n")
        print(build_natural_content(rec, snip, fake))
        return

    import anthropic
    client = anthropic.Anthropic(api_key=_api_key())

    todo, n_stale = [], 0
    for r, ci in work:
        p = decoy_cache_path(out_dir, args.run, r["idx"], ci, args.model)
        if cache_is_current(p, args.g):
            continue
        if p.exists():
            n_stale += 1
        todo.append((r, ci))
    print(f"[decoy_gen] cached={len(work) - len(todo)} todo={len(todo)}"
          + (f" (of which {n_stale} stale prompt/g, regenerating)" if n_stale else ""))

    def one(job):
        rec, ci = job
        try:
            taxo = taxo_label(taxo_dir, args.run, rec["idx"], ci)
            obj = gen_decoys_for_claim(client, rec, ci, taxo, args.g,
                                       args.model, args.run)
            write_cache(decoy_cache_path(out_dir, args.run, rec["idx"], ci,
                                         args.model), obj)
            return (rec["idx"], ci, obj["n_text_pass"], None)
        except Exception as e:
            return (rec["idx"], ci, -1, str(e)[:200])

    t0 = time.time()
    fails = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for k, (idx, ci, npass, err) in enumerate(ex.map(one, todo), start=1):
            if err:
                fails += 1
                print(f"  [{k}/{len(todo)}] idx {idx} claim {ci} FAILED: {err}")
            else:
                print(f"  [{k}/{len(todo)}] idx {idx} claim {ci} text-survivors {npass}"
                      f"  ({time.time() - t0:.0f}s)")
    print(f"[decoy_gen] done: {len(todo) - fails} ok, {fails} failed, "
          f"{time.time() - t0:.0f}s")
    report(out_dir, args.run)


if __name__ == "__main__":
    main()
