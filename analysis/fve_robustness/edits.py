"""Deterministic explanation edits for the FVE-robustness arms (+ shared helpers).

Exp 1 (strip_all_quotes): remove every DOUBLE-quoted span — straight (U+0022) and
curly (U+201C … U+201D) — including the delimiters. The AV is truncated at
max_new_tokens=200, so a dangling unmatched opener is common; we strip from that
opener to end. Single quotes / apostrophes are left untouched: in these
explanations they are overwhelmingly contractions ("dog's") and markers nested
INSIDE double quotes (which are removed with their enclosing double span anyway),
not standalone quoted content. Returns the edited text, the removed spans in
ORIGINAL-text coordinates (for diff rendering), and light metadata.
"""
from __future__ import annotations

import hashlib
import random
import re
import unicodedata

_WS = re.compile(r"[ \t]{2,}")
_SP_PUNCT = re.compile(r"\s+([,.;:!?])")
_EMPTY_PAREN = re.compile(r"\(\s*[,;:]?\s*\)")
_NL = re.compile(r"\n{3,}")

LDQUO, RDQUO = "“", "”"


def _straight_spans(text: str) -> list[tuple[int, int]]:
    """Pair straight double-quotes by parity; odd trailing one -> to end."""
    idx = [i for i, c in enumerate(text) if c == '"']
    spans = [(a, b + 1) for a, b in zip(idx[0::2], idx[1::2])]
    if len(idx) % 2 == 1:
        spans.append((idx[-1], len(text)))
    return spans


def _curly_spans(text: str) -> list[tuple[int, int]]:
    """Stack-match curly doubles; any unclosed opener -> to end."""
    spans, stack = [], []
    for i, c in enumerate(text):
        if c == LDQUO:
            stack.append(i)
        elif c == RDQUO and stack:
            spans.append((stack.pop(), i + 1))
    if stack:
        spans.append((min(stack), len(text)))
    return spans


def _coalesce(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for s, e in sorted(spans):
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def _cleanup(t: str) -> str:
    """Minimal, meaning-preserving normalization of the holes left by deletion:
    collapse runs of spaces, pull punctuation back to the word, drop empty parens.
    Deliberately does NOT rewrite dangling connectives ('or or') — the edit stays
    faithful and the rough edges are visible in the dashboard diff."""
    t = _WS.sub(" ", t)
    t = _SP_PUNCT.sub(r"\1", t)
    t = _EMPTY_PAREN.sub("", t)
    t = _SP_PUNCT.sub(r"\1", t)
    t = _WS.sub(" ", t)
    t = _NL.sub("\n\n", t)
    return t.strip()


def strip_all_quotes(text: str):
    """-> (edited_text, removed_spans[(start,end) in original coords], meta)."""
    spans = _coalesce(_straight_spans(text) + _curly_spans(text))
    kept, prev = [], 0
    for s, e in spans:
        kept.append(text[prev:s])
        prev = e
    kept.append(text[prev:])
    edited = _cleanup("".join(kept))
    unbalanced = any(e == len(text) and text[s] in ('"', LDQUO) for s, e in spans)
    meta = dict(n_removed=len(spans),
                chars_removed=sum(e - s for s, e in spans),
                unbalanced=bool(unbalanced))
    return edited, spans, meta


def keep_only_quotes(text: str):
    """Exp 2 (complement of Exp 1): keep ONLY the double-quoted spans (in original
    order, marks included), drop all connective prose. -> (edited_text, kept_spans in
    original coords, meta). kept_spans drive the tab (quotes highlighted, prose struck)."""
    spans = _coalesce(_straight_spans(text) + _curly_spans(text))
    edited = _cleanup(" ".join(text[s:e] for s, e in spans))
    chars_kept = sum(e - s for s, e in spans)
    meta = dict(n_kept=len(spans), chars_kept=chars_kept,
                chars_removed=len(text) - chars_kept)
    return edited, spans, meta


# ── Exp 3 (Tier A): input-derivability of quoted spans ───────────────────────
# A quoted span is "input-derivable" only when it shares a CONTIGUOUS VERBATIM word-run with the
# input of at least _MIN_CONTENT content (non-stopword, >=3 char) words. This deliberately ignores
# incidental single common-word coincidences (a lone "potential" that merely occurs somewhere in a
# 2k-token doc is NOT a restatement) and, being run-based rather than whole-quote coverage, it also
# fires on a long paraphrase quote that embeds a genuine multi-word input phrase. Synonym/paraphrase
# restatements with no verbatim run are out of Tier A scope -> Tier B / API.
_NONALNUM = re.compile(r"[^0-9a-z]+")
_MIN_CONTENT = 2            # min content words in the longest shared verbatim run to call it derivable

_STOPWORDS = frozenset("""
a an the this that these those and or but nor for yet so of in on at to from by with about as into
like through after over between out against during without before under above below around among
within upon per via is are was were be been being am do does did done have has had having will
would shall should can could may might must not no it its itself he him his she her hers they them
their theirs we us our ours you your yours i me my mine who whom whose which what where when why how
if then than because while there here also very just only own same each every both few more most
other some any all such one two three four five six seven eight nine ten mr mrs ms
""".split())


def _norm_match(s: str) -> str:
    """Lowercase + NFKC + collapse every non-alphanumeric run to a single space.
    Matching-only normalization (Tier A input-overlap), never used for display."""
    s = unicodedata.normalize("NFKC", s).lower()
    return _NONALNUM.sub(" ", s).strip()


def _is_content(w: str) -> bool:
    return len(w) >= 3 and w not in _STOPWORDS


def _content_count(words) -> int:
    return sum(1 for w in words if _is_content(w))


def build_doc_index(doc: str) -> dict:
    """Normalized doc word list + a position index of its CONTENT words (the only legal seeds for a
    shared run). Built once per doc; stopwords still live in `words` so runs extend across them."""
    dw = _norm_match(doc).split()
    pos: dict = {}
    for i, w in enumerate(dw):
        if _is_content(w):
            pos.setdefault(w, []).append(i)
    return {"words": dw, "pos": pos}


def classify_quote(quote: str, idx: dict) -> dict:
    """Tier-A verdict: derivable iff the longest CONTIGUOUS verbatim word-run the quote shares with
    the input holds >=_MIN_CONTENT content words. Seeds on the quote's content words and extends
    both directions against the doc; returns the run length so the threshold is auditable and
    re-thresholdable offline. No synonym/paraphrase matching (that is Tier B / API)."""
    qw = _norm_match(quote).split()
    if not qw:
        return dict(derivable=False, kind="empty", run_words=0, run_content=0, nwords=0)
    dw, pos = idx["words"], idx["pos"]
    Lq, Ld = len(qw), len(dw)
    best_c, best_len = 0, 0
    for i, w in enumerate(qw):
        for p in pos.get(w, ()):                 # only content words seed a run
            a = 0
            while i - a - 1 >= 0 and p - a - 1 >= 0 and qw[i - a - 1] == dw[p - a - 1]:
                a += 1
            b = 0
            while i + b + 1 < Lq and p + b + 1 < Ld and qw[i + b + 1] == dw[p + b + 1]:
                b += 1
            run = qw[i - a:i + b + 1]
            c = _content_count(run)
            if c > best_c or (c == best_c and len(run) > best_len):
                best_c, best_len = c, len(run)
    der = best_c >= _MIN_CONTENT
    kind = "divergent" if not der else ("verbatim" if best_len == Lq else "phrase")
    return dict(derivable=der, kind=kind, run_words=best_len, run_content=best_c, nwords=Lq)


def strip_derivable_quotes(text: str, doc: str):
    """Exp 3 (Tier A): from the FULL explanation remove only the double-quoted spans that are
    input-derivable (share a >=_MIN_CONTENT-content-word contiguous verbatim run with `doc`); keep
    the divergent quotes AND all connective prose. -> (edited, removed_spans[orig coords], meta).
    meta.quotes holds the per-quote verdict (span/kind/run_content) for an auditable classification."""
    idx = build_doc_index(doc)
    spans = _coalesce(_straight_spans(text) + _curly_spans(text))
    verdicts, removed = [], []
    for s, e in spans:
        v = classify_quote(text[s:e], idx)
        v["span"] = [int(s), int(e)]
        verdicts.append(v)
        if v["derivable"]:
            removed.append((s, e))
    kept_parts, prev = [], 0
    for s, e in removed:
        kept_parts.append(text[prev:s])
        prev = e
    kept_parts.append(text[prev:])
    edited = _cleanup("".join(kept_parts))
    n_der = len(removed)
    meta = dict(n_removed=n_der,
                chars_removed=sum(e - s for s, e in removed),
                n_quotes=len(spans), n_derivable=n_der, n_divergent=len(spans) - n_der,
                n_verbatim=sum(1 for v in verdicts if v["kind"] == "verbatim"),
                n_phrase=sum(1 for v in verdicts if v["kind"] == "phrase"),
                quotes=verdicts)
    return edited, removed, meta


# ── Exp 5 (shuffle claims at random) ─────────────────────────────────────────
# Reorder-only edits: characters are preserved, only the order of claim units changes, so any FVE move
# is a pure order effect. Three grains: 5a whole blocks (~3), 5b quote-aware sentences (~1/block),
# 5c the quoted claims inside the FINAL block (permute quote CONTENTS among their slots). The
# permutation is seeded by md5(explanation) (reproducible) and the identity is rejected for n>=2.
_BLOCK_SEP = re.compile(r"\n\s*\n+")
_SENT_ABBR = {"e.g.", "i.e.", "u.s.", "u.k.", "dr.", "mr.", "mrs.", "ms.", "etc.", "vs.",
              "no.", "fig.", "st.", "inc.", "cf.", "al.", "jr.", "sr."}


def _seed(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)


def _perm_nonidentity(n: int, seed: int) -> list:
    """Permutation of range(n); guaranteed non-identity for n>=2 (retry, then rotate fallback)."""
    base = list(range(n))
    if n < 2:
        return base
    rng = random.Random(seed)
    for _ in range(20):
        p = base[:]
        rng.shuffle(p)
        if p != base:
            return p
    return base[1:] + base[:1]


def _trim(text: str, s: int, e: int):
    while s < e and text[s].isspace():
        s += 1
    while e > s and text[e - 1].isspace():
        e -= 1
    return s, e


def split_blocks(text: str) -> list:
    """Spans of blank-line-separated, non-empty blocks (whitespace-trimmed)."""
    spans, pos = [], 0
    for m in _BLOCK_SEP.finditer(text):
        s, e = _trim(text, pos, m.start())
        if e > s:
            spans.append((s, e))
        pos = m.end()
    s, e = _trim(text, pos, len(text))
    if e > s:
        spans.append((s, e))
    return spans


def split_sentences(text: str) -> list:
    """Sentence spans; never split inside a double-quoted span; skip a small abbreviation set and
    single-char / bare-number list markers ('4.', 'G.')."""
    m = list(text)
    for s, e in _coalesce(_straight_spans(text) + _curly_spans(text)):
        for i in range(s, e):
            if m[i] in ".!?\n":
                m[i] = "\0"                        # neutralize sentence punctuation inside quotes
    masked = "".join(m)
    bounds = [0]
    for mt in re.finditer(r"[.!?]", masked):
        i = mt.start()
        j = i
        while j > 0 and (text[j - 1].isalnum() or text[j - 1] == "."):
            j -= 1
        tok = text[j:i + 1]
        if tok.lower() in _SENT_ABBR or re.fullmatch(r"[A-Za-z0-9]\.", tok):
            continue
        k = i + 1
        while k < len(text) and masked[k] in ".!?\"'”’)»":
            k += 1
        w = k
        while w < len(text) and masked[w] in " \t\n":
            w += 1
        if w == k:                                 # no whitespace after -> not a boundary
            continue
        if w >= len(text) or text[w].isupper() or text[w] in "\"“'«":
            bounds.append(k)
    bounds.append(len(text))
    out = []
    for a, b in zip(bounds, bounds[1:]):
        s, e = _trim(text, a, b)
        if e > s:
            out.append((s, e))
    return out


def _reorder(text: str, spans: list, sep: str, kind: str):
    """Permute the substrings at `spans` (original order) and rejoin with `sep`."""
    units = [text[s:e] for s, e in spans]
    n = len(units)
    if n == 0:
        return text, dict(kind=kind, n_units=0, perm=[], noop=True, units_orig=[], units_edit=[])
    perm = _perm_nonidentity(n, _seed(kind + "\x00" + text))     # arm-salted seed -> independent perms
    out, units_edit = "", []
    for pos, oi in enumerate(perm):
        if pos:
            out += sep
        st = len(out)
        out += units[oi]
        units_edit.append([st, st + len(units[oi]), oi])
    meta = dict(kind=kind, n_units=n, perm=list(perm), noop=bool(n < 2 or perm == list(range(n))),
                units_orig=[[int(s), int(e)] for s, e in spans], units_edit=units_edit)
    return out, meta


def shuffle_blocks(text: str):
    return _reorder(text, split_blocks(text), "\n\n", "block")


def shuffle_sentences(text: str):
    return _reorder(text, split_sentences(text), " ", "sentence")


def shuffle_quoted_claims(text: str):
    """5c: permute non-echo quoted claims inside the ``Final token`` block.

    The first quote after the ``Final token`` marker is the read-token echo, not a claim in the
    enumerated continuation list.  It is therefore frozen.  The lead-in clause and subsequent quoted
    predictions are permuted among their slots, while each slot's quote delimiters and all connective
    scaffolding remain byte-identical.  This isolates claim order instead of moving the gold-token echo.
    """
    blocks = split_blocks(text)
    if not blocks:
        return text, dict(kind="quoted-claim", n_units=0, perm=[], noop=True,
                          units_orig=[], units_edit=[])
    bs, be = blocks[-1]
    block = text[bs:be]
    all_qsp = _coalesce(_straight_spans(block) + _curly_spans(block))
    marker = re.search(r"Final token", block, re.IGNORECASE)
    echo_idx = next((i for i, (s, _e) in enumerate(all_qsp)
                     if marker is not None and s >= marker.end()), None)
    qsp = [sp for i, sp in enumerate(all_qsp) if i != echo_idx]
    n = len(qsp)
    units_orig = [[bs + s, bs + e] for s, e in qsp]
    excluded_echo = ([bs + all_qsp[echo_idx][0], bs + all_qsp[echo_idx][1]]
                     if echo_idx is not None else None)
    if n < 2:
        return text, dict(kind="quoted-claim", n_units=n, perm=list(range(n)), noop=True,
                          units_orig=units_orig, units_edit=[], excluded_echo=excluded_echo)

    def inner(s: int, e: int) -> str:
        # All spans begin with a supported opening quote.  A common AV truncation leaves the final
        # quote unclosed; in that case the span reaches block-end and has no closing delimiter.
        closed = e - s >= 2 and block[e - 1] in ('"', RDQUO)
        return block[s + 1:e - 1 if closed else e]

    contents = [inner(s, e) for s, e in qsp]
    perm = _perm_nonidentity(n, _seed("quoted-claim-v2\x00" + text))
    edited_block, cur, units_edit = "", 0, []
    for slot, (s, e) in enumerate(qsp):
        edited_block += block[cur:s]
        closed = e - s >= 2 and block[e - 1] in ('"', RDQUO)
        newc = block[s] + contents[perm[slot]] + (block[e - 1] if closed else "")
        st = bs + len(edited_block)
        edited_block += newc
        units_edit.append([st, st + len(newc), perm[slot]])
        cur = e
    edited_block += block[cur:]
    edited = text[:bs] + edited_block + text[be:]
    meta = dict(kind="quoted-claim", n_units=n, perm=list(perm),
                noop=bool(perm == list(range(n))),
                units_orig=units_orig, units_edit=units_edit,
                excluded_echo=excluded_echo)
    return edited, meta
