"""Exp 6 ("realquote") — shared parser + geometry, and (later) the substitution edit_fns.

The AV explanation is 3 blocks (validated 97-100% carry a "Final token" marker):
  B0  genre/register line                              -> GENRE   (frozen, never edited)
  B1  the "structure/opening sentence" quote           -> LCR     (left-context reconstruction)
  B2  Final token "X" ... ("<clause>") ... expecting "A" or "B"  -> ECHO + LCR-clause + PC(s)

Functional quote kinds:
  LCR  left-context reconstruction  (the AV inverting what the activation summarizes)   <- Exp-6 manipuland
  ECHO the read-token quote right after `Final token`                                    (frozen)
  PC   predicted continuation quotes after a continuation trigger                        (frozen in headline)
  GENRE quotes in the genre line / UNKNOWN quotes matching no rule                       (frozen)

This module provides ONLY the deterministic parse + geometry used by the Step-6.0 parser audit
and the later edit arms. No GPU/API. classify_expl_spans needs just (expl, token); the directional
gate + right-edge anchor + divergence need the document text.
"""
from __future__ import annotations

import re

import random

from edits import (_coalesce, _curly_spans, _is_content, _norm_match,  # reuse Exp-1/3 helpers
                   _perm_nonidentity, _seed, _straight_spans, split_blocks)

# ── continuation triggers (union across qwen/gemma12b/gemma27b, ranked by observed frequency) ──
# A quote at/after the EARLIEST trigger in the final block is a predicted continuation (PC);
# quotes after the ECHO but before it are the parenthetical left-context clause (LCR).
_TRIGGERS = [
    "expecting", "requiring", "anticipat", "such as", "followed by", "leading to",
    "continu", "predict", "suggest", "await", "introduc", "completion like",
    "most likely", "likely", "like ", "mirror",
]
_TRIG_RE = re.compile("|".join(re.escape(t) for t in _TRIGGERS), re.IGNORECASE)
_FINALTOK_RE = re.compile(r"Final token", re.IGNORECASE)


def _quote_spans(text: str) -> list[tuple[int, int]]:
    return _coalesce(_straight_spans(text) + _curly_spans(text))


def _inner(text: str, s: int, e: int) -> str:
    """Strip the enclosing quote marks from a coalesced span -> the quoted content."""
    return text[s:e].strip("\"“”`").strip()


def _content_words(text: str) -> list[str]:
    return [w for w in _norm_match(text).split() if _is_content(w)]


def classify_expl_spans(expl: str, token: str) -> dict:
    """Structural tagging of every quoted span. Returns
      {n_blocks, unbalanced, block_spans, final_block_idx, quotes:[{span,text,block,tag,after_trigger}]}
    tag in {GENRE, LCR, ECHO, PC, DUP}. Deterministic; needs no document.

    Parser-audit fixes (Step 6.0): quotes are extracted PER BLOCK so a stray unmatched double-quote
    only corrupts its own block, not the whole explanation (Bug 2); continuation triggers are scoped
    in EVERY block, not just the final one, so a B1-embedded 'likely continuing with' correctly makes
    the following quote PC instead of leaking a future guess into the edited LCR set (Bug 1); a quote
    that is exactly the read token is tagged DUP (frozen, no-op edit) (Bug 5)."""
    blocks = split_blocks(expl)
    n_blocks = len(blocks)
    tok = token.strip()

    # per-block quote extraction: parity resets at block boundaries (Bug 2)
    spans, block_of = [], {}
    unbalanced = False
    for bi, (bs, be) in enumerate(blocks):
        seg = expl[bs:be]
        bspans = _coalesce([(bs + s, bs + e) for s, e in _straight_spans(seg)]
                           + [(bs + s, bs + e) for s, e in _curly_spans(seg)])
        for sp in bspans:
            block_of[sp] = bi
            if sp[1] == be and expl[sp[0]] in ('"', "“", "`"):
                unbalanced = True
        spans.extend(bspans)
    spans.sort()

    # final block = the one holding the "Final token" marker, else the last block
    final_idx = n_blocks - 1
    marker_abs = None
    for bi, (bs, be) in enumerate(blocks):
        m = _FINALTOK_RE.search(expl, bs, be)
        if m:
            final_idx, marker_abs = bi, m.end()
            break
    if marker_abs is None and blocks:
        marker_abs = blocks[final_idx][0]

    # earliest continuation trigger PER BLOCK (Bug 1) — absolute offset, or None
    trig_by_block = {}
    for bi, (bs, be) in enumerate(blocks):
        start = marker_abs if (bi == final_idx and marker_abs is not None) else bs
        mt = _TRIG_RE.search(expl, start, be)
        trig_by_block[bi] = mt.start() if mt else None

    quotes, echo_taken = [], False
    for s, e in spans:
        bi = block_of[(s, e)]
        inner = _inner(expl, s, e)
        trig = trig_by_block.get(bi)
        after_trig = bool(trig is not None and s >= trig)
        if bi == 0 and bi != final_idx:
            tag = "GENRE"                                   # B0 register line
        elif bi == final_idx and not echo_taken and marker_abs is not None and s >= marker_abs - 1:
            tag, echo_taken = "ECHO", True                  # first quote after `Final token`
        elif after_trig:
            tag = "PC"                                      # post-trigger in ANY block -> continuation
        else:
            tag = "LCR"
        if tag == "LCR" and tok and inner.strip() == tok:   # Bug 5: quote == read token -> no-op
            tag = "DUP"
        if tag == "LCR" and not [w for w in _norm_match(inner).split() if _is_content(w)]:
            tag = "JUNK"                                     # empty / punctuation / stopword-only -> frozen
        quotes.append(dict(span=[int(s), int(e)], text=inner, block=bi, tag=tag,
                           after_trigger=after_trig))

    return dict(n_blocks=n_blocks, unbalanced=bool(unbalanced),
                block_spans=[[int(a), int(b)] for a, b in blocks],
                final_block_idx=final_idx, quotes=quotes)


# ── geometry: directional gate, right-edge anchor, divergence (need the document) ────────────

def _lcs_len(a: list[str], b: list[str]) -> int:
    la, lb = len(a), len(b)
    if la == 0 or lb == 0:
        return 0
    prev = [0] * (lb + 1)
    for i in range(1, la + 1):
        cur = [0] * (lb + 1)
        ai = a[i - 1]
        row = prev
        for j in range(1, lb + 1):
            cur[j] = row[j - 1] + 1 if ai == b[j - 1] else (row[j] if row[j] >= cur[j - 1] else cur[j - 1])
        prev = cur
    return prev[lb]


def rouge_l(a: list[str], b: list[str]) -> float:
    """F-measure ROUGE-L over two word sequences (0..1)."""
    l = _lcs_len(a, b)
    if l == 0:
        return 0.0
    p, r = l / len(b), l / len(a)
    return 2 * p * r / (p + r)


def directional_fracs(quote: str, left_ctx: str, cont: str) -> tuple[float, float]:
    """Fraction of the quote's DISTINCT content words present in the left-context vs the continuation.
    LCR expects frac_left >= frac_right; PC the reverse. Coarse but fast dual-gate signal."""
    qc = set(_content_words(quote))
    if not qc:
        return 0.0, 0.0
    lc, cc = set(_content_words(left_ctx)), set(_content_words(cont))
    return len(qc & lc) / len(qc), len(qc & cc) / len(qc)


def anchor_fidelity(quote: str, left_ctx: str, tail_k: int = 6, slack: int = 6) -> float:
    """Locate the quote's true right-edge inside the left-context and return restatement fidelity in
    [0,1] = best ROUGE-L of the quote's last `tail_k` content words against a local window ending at a
    candidate anchor. Candidates = doc positions where any of the quote's last-3 content words occur
    (content-word seeding, like Tier-A). No candidate -> 0.0 (=> no_faithful_anchor). Fast."""
    q_all = _content_words(quote)
    if not q_all:
        return 0.0
    tail = q_all[-tail_k:]
    dw = _content_words(left_ctx)
    if not dw:
        return 0.0
    seeds = set(tail[-3:])
    win = len(tail) + slack
    positions = [i for i, w in enumerate(dw) if w in seeds]
    if not positions:
        return 0.0
    best = 0.0
    for p in positions:
        window = dw[max(0, p - win + 1):p + 1]
        s = rouge_l(tail, window)
        if s > best:
            best = s
            if best >= 0.999:
                break
    return best


def last_n_words(text: str, n: int) -> str:
    """Surface tail: the last `n` whitespace-delimited words of `text` (Exp-6 true-analog injection)."""
    ws = text.split()
    return " ".join(ws[-n:]) if n > 0 else ""


def substitute_lcr(expl: str, token: str, doc: str, char_end: int, char_start: int | None = None,
                   include_token: bool = False) -> dict:
    """Exp-6 core edit. Replace every LCR quote's CONTENT with the TRUE pre-token context tail
    (word-length-matched), keeping the quote marks and all scaffolding fixed. include_token=False -> A2
    (tail ends at char_start, the read token EXCLUDED, the load-bearing arm); True -> A1 (tail ends at
    char_end, token included). ECHO/PC/DUP/JUNK/GENRE quotes are frozen. Returns:
      expl_edit, removed (LCR spans in ORIGINAL coords, for strike-through), added (injected spans in
      EDITED coords, for highlight), edit_meta{arm, n_removed, chars_removed, quotes:[{orig,injected,
      divergence,n_words,span}]}."""
    p = analyze_datapoint(expl, token, doc, char_end, char_start)
    cut = char_end if include_token else (char_start if char_start is not None else char_end)
    left = doc[:cut]
    subs = []                                                   # (s, e, injected_text, quote-record)
    for q in p["quotes"]:
        if q["tag"] != "LCR":
            continue
        s, e = q["span"]
        nw = len(q["text"].split())
        tail = last_n_words(left, nw)
        if not tail.strip():
            continue
        subs.append(_sub(expl, s, e, tail, q, nw))
    return _assemble(expl, subs, "exp6a" if include_token else "exp6b")


def _sub(expl: str, s: int, e: int, replacement: str, q: dict, nw: int) -> tuple:
    """Build one substitution tuple, preserving the LCR quote's own opening/closing marks."""
    open_c, close_c = expl[s], expl[e - 1]
    inj = f"{open_c}{replacement}{close_c}"
    return (s, e, inj, dict(orig=q["text"], injected=replacement,
                            divergence=q.get("divergence"), n_words=nw, span=[s, e]))


def _assemble(expl: str, subs: list, arm: str, quote_kind: str = "LCR") -> dict:
    """Splice substitutions into the explanation; emit removed (orig coords), added (edited coords),
    and edit_meta. Shared by the core (A1/A2) and control (F/B/C) arms so they are byte-comparable."""
    subs = sorted(subs)
    out, prev, removed, added, qrecs, chars_removed = [], 0, [], [], [], 0
    for s, e, inj, qr in subs:
        out.append(expl[prev:s])
        cur = sum(len(x) for x in out)
        added.append([cur, cur + len(inj)])
        out.append(inj)
        removed.append([int(s), int(e)])
        chars_removed += (e - s)
        qrecs.append(qr)
        prev = e
    out.append(expl[prev:])
    meta = dict(arm=arm, n_removed=len(qrecs), chars_removed=chars_removed,
                n_lcr=len(qrecs) if quote_kind == "LCR" else 0,
                n_pc=len(qrecs) if quote_kind == "PC" else 0,
                quotes=qrecs)
    return dict(expl_edit="".join(out), removed=removed, added=added, edit_meta=meta)


def paraphrase_lcr(expl: str, token: str, doc: str, char_end: int,
                   replacements: dict[str, str], char_start: int | None = None) -> dict:
    """Arm P: replace each LCR quote with a meaning-preserving paraphrase of its A1 true slice.

    ``replacements`` is keyed by the exact word-length-matched A1 slice.  It is produced and fully
    audited by :mod:`exp6_complete`; this deterministic function only applies the frozen cache.  Quote
    delimiters, ECHO/PC slots, and every byte outside the selected LCR spans remain unchanged.
    """
    p = analyze_datapoint(expl, token, doc, char_end, char_start)
    left = doc[:char_end]
    subs = []
    for q in p["quotes"]:
        if q["tag"] != "LCR":
            continue
        s, e = q["span"]
        nw = len(q["text"].split())
        source = last_n_words(left, nw)
        replacement = replacements.get(source)
        if not source.strip() or not replacement:
            continue
        sub = _sub(expl, s, e, replacement, q, nw)
        sub[3]["source_true"] = source
        subs.append(sub)
    return _assemble(expl, subs, "exp6p")


def continuation_case(expl: str, token: str, doc: str, char_end: int,
                      char_start: int | None = None, preview_words: int = 40) -> dict:
    """Return the semantic-judgment input for the secondary true-continuation arm.

    Every PC quote is an alternative guess for what immediately follows the read token, so all guesses
    are compared against the same leading continuation preview.  The API judge determines whether at
    least one is semantically correct; a lexical score is retained only as an audit covariate.
    """
    p = analyze_datapoint(expl, token, doc, char_end, char_start)
    pcs = [q for q in p["quotes"] if q["tag"] == "PC"]
    true_cont = " ".join(doc[char_end:].split()[:preview_words])
    lexical = []
    for q in pcs:
        nw = max(1, len(q["text"].split()))
        true_head = " ".join(doc[char_end:].split()[:nw])
        lexical.append(round(rouge_l(_content_words(q["text"]), _content_words(true_head)), 3))
    return dict(true_continuation=true_cont, candidates=[q["text"] for q in pcs],
                lexical_scores=lexical)


def substitute_true_continuation(expl: str, token: str, doc: str, char_end: int,
                                 verdict: dict, char_start: int | None = None) -> dict:
    """Arm C_cont (artifact key ``exp6t``): inject ordered true-continuation tiles into PC slots.

    LCR and ECHO quotes are frozen.  PC slot lengths are preserved in whitespace-delimited words, and
    tiles consume ``doc[char_end:]`` in order without overlap.  The semantic ``verdict`` is attached so
    downstream statistics can use the preregistered AV-continuation-wrong subset while still retaining
    the full scored arm for audit.
    """
    p = analyze_datapoint(expl, token, doc, char_end, char_start)
    words = doc[char_end:].split()
    cursor, subs, insufficient = 0, [], 0
    for q in p["quotes"]:
        if q["tag"] != "PC":
            continue
        s, e = q["span"]
        nw = len(q["text"].split())
        tile = " ".join(words[cursor:cursor + nw])
        if nw < 1 or len(words[cursor:cursor + nw]) != nw:
            insufficient += 1
            continue
        cursor += nw
        sub = _sub(expl, s, e, tile, q, nw)
        sub[3]["tag"] = "PC"
        sub[3]["true_continuation_tile"] = tile
        sub[3]["lexical_similarity"] = round(
            rouge_l(_content_words(q["text"]), _content_words(tile)), 3)
        subs.append(sub)
    out = _assemble(expl, subs, "exp6t", quote_kind="PC")
    out["edit_meta"].update(
        continuation_wrong=verdict.get("wrong"),
        best_candidate=verdict.get("best_candidate"),
        continuation_reason=verdict.get("reason", ""),
        true_continuation=verdict.get("true_continuation", ""),
        n_insufficient_tiles=insufficient,
    )
    return out


def shuffle_lcr(expl: str, token: str, doc: str, char_end: int, char_start: int | None = None) -> dict:
    """Control F (form-OOD floor). Replace each LCR quote with a seeded word-shuffle of ITS OWN words
    (same bag of words, destroyed order, length identical). Isolates 'any disruption of the quote' from
    'true content': if F collapses like A2, the A2 drop is structural/OOD, not content-specific."""
    p = analyze_datapoint(expl, token, doc, char_end, char_start)
    subs = []
    for q in p["quotes"]:
        if q["tag"] != "LCR":
            continue
        s, e = q["span"]
        words = q["text"].split()
        if len(words) < 2:
            continue                                            # 1-word quote: nothing to shuffle
        perm = _perm_nonidentity(len(words), _seed("exp6f\x00" + q["text"] + str(s)))
        subs.append(_sub(expl, s, e, " ".join(words[i] for i in perm), q, len(words)))
    return _assemble(expl, subs, "exp6f")


def random_ctx(expl: str, token: str, doc: str, char_end: int, char_start: int | None = None,
               guard: int = 600) -> dict:
    """Control B (random same-doc span). Replace each LCR quote with a seeded random real span of the
    same word-length drawn from the FAR left-context doc[:char_start-guard] (same doc, wrong position,
    never the continuation). Rules out 'any real same-doc text helps'."""
    p = analyze_datapoint(expl, token, doc, char_end, char_start)
    cs = char_start if char_start is not None else char_end
    pool = doc[:max(0, cs - guard)].split()
    if len(pool) < 4:
        pool = doc[:cs].split()                                 # fallback: whole past (rare, early reads)
    subs = []
    for q in p["quotes"]:
        if q["tag"] != "LCR":
            continue
        s, e = q["span"]
        nw = len(q["text"].split())
        if len(pool) < nw or nw < 1:
            continue
        rng = random.Random(_seed("exp6r\x00" + q["text"] + str(s)))
        st = rng.randrange(0, len(pool) - nw + 1)
        subs.append(_sub(expl, s, e, " ".join(pool[st:st + nw]), q, nw))
    return _assemble(expl, subs, "exp6r")


def otherdoc_ctx(expl: str, token: str, doc: str, char_end: int, char_start: int | None,
                 donor_doc: str, donor_cut: int) -> dict:
    """Control C_otherdoc. Replace each LCR quote with the true pre-token context tail (token excluded,
    A2-style) of a DIFFERENT held-out document. Real, grammatical, length-matched, semantically wrong;
    must HURT, else the LCR slots are inert and nothing is interpretable."""
    p = analyze_datapoint(expl, token, doc, char_end, char_start)
    donor_left = donor_doc[:donor_cut]
    subs = []
    for q in p["quotes"]:
        if q["tag"] != "LCR":
            continue
        s, e = q["span"]
        nw = len(q["text"].split())
        tail = last_n_words(donor_left, nw)
        if not tail.strip():
            continue
        subs.append(_sub(expl, s, e, tail, q, nw))
    return _assemble(expl, subs, "exp6o")


def analyze_datapoint(expl: str, token: str, doc: str, char_end: int, char_start: int | None = None) -> dict:
    """Full Exp-6 parse of one datapoint. For each LCR quote the true analog is the true pre-token
    context tail (word-length-matched, ending at the read token = A1's injection), NOT a lexically
    located span: the activation encodes the pre-token state, and the audit shows LCR quotes
    reconstruct that region (paraphrased/confabulated), so `divergence = 1 - ROUGE-L(quote, true tail)`
    stratifies power without dropping any row. Also carries the directional-gate fracs + a secondary
    locate-based anchor_fidelity for comparison. Read-only. Used by the parser audit + edit arms."""
    parsed = classify_expl_spans(expl, token)
    left_ctx, cont = doc[:char_end], doc[char_end:]
    # LOCAL directional windows: is the quote restating the immediate PAST or predicting the near FUTURE?
    left_win = " ".join(left_ctx.split()[-60:])
    right_win = " ".join(cont.split()[:40])
    tok = token.strip()
    for q in parsed["quotes"]:
        if q["tag"] in ("LCR", "PC"):
            fl, fr = directional_fracs(q["text"], left_win, right_win)
            q["frac_left"], q["frac_right"] = round(fl, 3), round(fr, 3)
            ends_at_token = bool(tok and q["text"].rstrip().rstrip('".,;:)').endswith(tok))
            # Bug 3 + future-leak guard: the directional signal is now AUTHORITATIVE for edited quotes
            if q["tag"] == "LCR" and q["after_trigger"] and fr > fl:
                q["tag"], q["override"] = "PC", "trigger+future->PC"
            elif q["tag"] == "LCR" and fr >= 0.8 and fr > fl + 0.1:
                q["tag"], q["override"] = "PC", "leakguard->PC"      # content lives in the future
            elif q["tag"] == "PC" and ends_at_token and fl >= fr and not q["after_trigger"]:
                q["tag"], q["override"] = "LCR", "endsattoken+past->LCR"
            q["dir_disagree"] = bool((q["tag"] == "LCR" and fr > fl) or (q["tag"] == "PC" and fl > fr))
        if q["tag"] == "LCR":                                       # recompute AFTER overrides
            nw = len(q["text"].split())
            q["n_words"] = nw
            true_tail = last_n_words(left_ctx, nw)                  # A1 injection (includes read token)
            fid = rouge_l(_content_words(q["text"]), _content_words(true_tail))
            q["restate_fidelity"] = round(fid, 3)                  # ROUGE-L(AV quote, true pre-tok tail)
            q["divergence"] = round(1.0 - fid, 3)                  # stratifier (no row dropped)
            q["anchor_fidelity"] = round(anchor_fidelity(q["text"], left_ctx), 3)  # secondary, locate-based
    return parsed
