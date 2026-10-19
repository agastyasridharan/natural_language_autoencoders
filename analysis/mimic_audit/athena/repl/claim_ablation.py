"""Claim-ablation test - a judge-free, ground-truth-free faithfulness metric for NLAs.

THE IDEA (plain English)
------------------------
An NLA's activation verbalizer (AV) turns a model's internal activation vector ``h``
into an explanation ``z`` made of a few snippets. Each snippet is a *claim* about what
the activation encodes. Some claims are genuinely read out of ``h``; others are
plausible guesses the AV would make from its prior, no matter what ``h`` is
(confabulations). This test separates the two **without** knowing what the target model
was actually thinking, and **without** an LLM judge.

For one snippet C inside ``z``:

  1. direction   Delta = AR(z) - AR(z without C)
       The activation reconstructor (AR, text->vector) tells us which direction in
       activation space the claim corresponds to. This is the same difference trick the
       NLA paper uses for steering, repurposed here as measurement.
  2. ablate      h' = h - (h . Dhat) Dhat            (Dhat = Delta / ||Delta||)
       Remove that one direction from h (rank-1 projection).
  3. measure     dlogp = logP_AV(C | h) - logP_AV(C | h')
       By TEACHER FORCING: how much less likely is the AV to re-emit C's exact tokens
       once the claimed substrate is gone? Defined even for a claim that appears once,
       deterministic, low-variance -> no judge, no "rate of recurrence" needed.

If ``dlogp`` is no larger than what a *random* direction does to the same span (the
null), the claim did not depend on h: it is confabulated **by causal definition**,
whether or not it happens to be true of the surrounding text. We also check
**specificity**: ablating C should not also suppress the OTHER snippets.

This is causal tracing (ROME-style) adapted to NLAs. It measures
faithfulness-to-activation, not truth-of-context, which is the gap the paper's three
confabulation heuristics (factual accuracy, AR ΔMSE, recurrence) cannot close.

WHAT THIS MODULE PROVIDES
-------------------------
``register_injector`` / ``LayerZeroInjector``  one decoder-layer-0 hook serving BOTH
    generation and teacher forcing (the only mechanism change vs avgen_model.Injector,
    which gates on the prefill length and so cannot do a full [prompt+z] forward).
``generate_explanation``                        z = AV(h)
``audit_activation``                             steps 1-5 for one h -> (record, vectors)

It reuses the athena harness:
    nla_inference.{NLACritic, normalize_activation}   the AR + L2 rescale
    repl_common.FAMILIES                              per-family chat markers
Numbers (injection_scale, marker ids, prompt template) come from the AV sidecar via
``avgen_model.read_sidecar`` - never hardcode them.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import torch

# Make the athena harness importable both on the GPU box and from a checkout.
_REPL = Path(__file__).resolve().parent          # .../athena/repl
_ATHENA = _REPL.parent                            # .../athena
for _p in (str(_REPL), str(_ATHENA), "/data/agastyas/nla_mimic/code"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from repl_common import FAMILIES                  # noqa: E402
from nla_inference import normalize_activation    # noqa: E402

EXPLANATION_RE = re.compile(r"<explanation>\s*(.*?)\s*</explanation>", re.DOTALL)
CJK = re.compile(r"[　-鿿豈-﷿＀-￯]")
_EPS = 1e-8


# ── prompt + snippet plumbing ────────────────────────────────────────────────

def actor_prompt_str(sc: dict, family: str) -> str:
    """The exact AV prompt string, ending with the assistant-turn opener so the
    model's continuation IS the explanation. Matches avgen_model.build_actor_prompt."""
    F = FAMILIES[family]
    content = sc["actor_template"].format(injection_char=sc["injection_char"])
    return F["single"]("user", content) + F["turn_end"] + F["single"](F["assistant"], "")


def split_snippets(z_inner: str) -> list[str]:
    """Split the inner explanation into snippets on blank lines; drop empties.

    The open-model AV emits 2-3 newline-separated snippets inside the
    <explanation>...</explanation> block. Snippet-level granularity needs no LLM
    judge: each snippet is one self-contained claim we can locate and ablate.
    """
    return [s.strip() for s in re.split(r"\n\s*\n", z_inner.strip()) if s.strip()]


def _find_marker(ids, sc: dict) -> int:
    pos = [i for i, x in enumerate(ids) if x == sc["inj_id"]]
    assert len(pos) == 1, f"marker appears {len(pos)}x (expected exactly 1)"
    p = pos[0]
    assert ids[p - 1] == sc["left"] and ids[p + 1] == sc["right"], (
        f"neighbor mismatch at marker {p}: got ({ids[p-1]},{ids[p+1]}) "
        f"vs sidecar ({sc['left']},{sc['right']})")
    return p


def build_tf_sequence(tok, sc, family, z_assistant):
    """Tokenize [actor_prompt + assistant explanation] ONCE, with offsets.

    Tokenizing the whole string in one shot (rather than prompt and continuation
    separately) guarantees the snippet char->token mapping is consistent with the
    ids the model actually forwards.

    Returns (input_ids[1,T] cpu long, marker_pos, offsets, full_text, cont_char_start).
    """
    prompt = actor_prompt_str(sc, family)
    full_text = prompt + z_assistant
    enc = tok(full_text, add_special_tokens=True, return_offsets_mapping=True)
    ids = enc["input_ids"]
    p = _find_marker(ids, sc)
    return (torch.tensor([ids], dtype=torch.long), p,
            enc["offset_mapping"], full_text, len(prompt))


def snippet_spans(offsets, full_text, snippets, cont_char_start):
    """Map each snippet's text to a (lo, hi) token-index range (hi exclusive).

    Searches only in the continuation (>= cont_char_start) and advances the search
    cursor so repeated snippet text stays in order. Returns a list aligned to
    ``snippets``; an entry is None when the snippet could not be located.
    """
    spans, cursor = [], cont_char_start
    for snip in snippets:
        cstart = full_text.find(snip, cursor)
        if cstart < 0:
            cstart = full_text.find(snip, cont_char_start)   # retry from block start
        if cstart < 0:
            spans.append(None)
            continue
        cend = cstart + len(snip)
        lo = hi = None
        for ti, (cs, ce) in enumerate(offsets):
            if ce <= cs:                       # special / zero-width token
                continue
            if ce > cstart and cs < cend:      # token overlaps the snippet char span
                lo = ti if lo is None else lo
                hi = ti + 1
        spans.append((lo, hi) if lo is not None else None)
        cursor = cend
    return spans


# ── injection: one hook for generation AND teacher forcing ───────────────────

class LayerZeroInjector:
    """forward_pre_hook on decoder block 0: overwrite hidden_states[0, p] with the
    stored vector ``v`` (already injection_scale-normalized).

    Fires on any forward whose sequence reaches past the marker - that covers the
    prefill of ``generate`` AND a single full teacher-forced [prompt+z] forward - but
    never on a cached 1-token decode step (seq_len == 1 <= p). That single condition
    (``hs.shape[1] > p``) is the only difference from avgen_model.Injector, whose
    ``hs.shape[1] != prompt_len`` gate would skip the teacher-forced forward.
    """

    def __init__(self, marker_pos: int):
        self.p = marker_pos
        self.v = None                          # set per forward by the caller

    def __call__(self, module, args, kwargs):
        hs = kwargs.get("hidden_states")
        positional = hs is None
        if positional:
            hs = args[0]
        if self.v is None or hs.shape[1] <= self.p:
            return None
        hs = hs.clone()
        hs[0, self.p] = self.v.to(hs.dtype).to(hs.device)
        if positional:
            return ((hs,) + tuple(args[1:]), kwargs)
        kwargs["hidden_states"] = hs
        return (args, kwargs)


def _find_layers(model):
    for path in ("model.layers", "model.language_model.layers",
                 "model.model.language_model.layers", "language_model.model.layers",
                 "language_model.layers", "model.text_model.layers", "transformer.h"):
        obj, ok = model, True
        for attr in path.split("."):
            if not hasattr(obj, attr):
                ok = False
                break
            obj = getattr(obj, attr)
        if ok and hasattr(obj, "__len__") and len(obj) > 0:
            return obj, path
    raise RuntimeError("could not locate decoder layers")


def register_injector(model, sc, family, tok):
    """Create the injector (marker position derived from the prompt, identical for
    every activation) and register its hook on decoder layer 0. Returns the injector.
    """
    prompt_ids = tok(actor_prompt_str(sc, family), add_special_tokens=True)["input_ids"]
    p = _find_marker(prompt_ids, sc)
    layers, path = _find_layers(model)
    inj = LayerZeroInjector(p)
    layers[0].register_forward_pre_hook(inj, with_kwargs=True)
    print(f"[claim_ablation] injector on {path}[0]; marker@{p} "
          f"neighbors ok ({sc['left']},{sc['right']}); prompt_len={len(prompt_ids)}")
    return inj


# ── teacher-forced span log-probability ──────────────────────────────────────

def _set_v(injector, vec, inj_scale):
    injector.v = normalize_activation(
        torch.as_tensor(np.asarray(vec, dtype=np.float32)).view(1, -1), inj_scale)[0]


@torch.inference_mode()
def _token_logp(model, input_ids):
    """Per-token log-prob of the realized tokens. tok_logp[t] = logP(input_ids[t+1] |
    input_ids[:t+1]); length T-1. So the log-prob of the token at absolute index
    ``a`` (a >= 1) is tok_logp[a-1]."""
    dev = model.get_input_embeddings().weight.device
    out = model(input_ids=input_ids.to(dev), use_cache=False)
    logits = out.logits[0].float()                       # [T, V]
    logp = torch.log_softmax(logits[:-1], dim=-1)        # [T-1, V]
    tgt = input_ids[0, 1:].to(logp.device)
    return logp.gather(1, tgt.unsqueeze(1)).squeeze(1)   # [T-1]


def _span_sums(tok_logp, ranges):
    out = []
    for r in ranges:
        if r is None:
            out.append(float("nan"))
            continue
        lo, hi = r
        out.append(float(tok_logp[lo - 1:hi - 1].sum()))
    return out


def _inject_span_logp(model, injector, input_ids, vec, inj_scale, ranges):
    """Inject ``vec`` at the marker, one forward, return (per-span summed logp,
    full per-token logp array)."""
    _set_v(injector, vec, inj_scale)
    tlp = _token_logp(model, input_ids)
    return _span_sums(tlp, ranges), tlp.detach().cpu().numpy()


@torch.inference_mode()
def generate_explanation(model, tok, sc, family, injector, h, *,
                         temperature=0.4, max_new=200, seed=0):
    """z = AV(h): inject h at the marker and free-run. Returns the assistant content
    (the full <explanation>...</explanation> block, decoded without special tokens)."""
    prompt_ids = tok(actor_prompt_str(sc, family), add_special_tokens=True)["input_ids"]
    assert _find_marker(prompt_ids, sc) == injector.p, "marker position drift"
    dev = model.get_input_embeddings().weight.device
    ids_t = torch.tensor([prompt_ids], dtype=torch.long, device=dev)
    _set_v(injector, h, sc["injection_scale"])
    torch.manual_seed(seed)
    out = model.generate(input_ids=ids_t, max_new_tokens=max_new,
                         do_sample=temperature > 0, temperature=max(temperature, 1e-5),
                         top_p=0.95, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][len(prompt_ids):], skip_special_tokens=True)


# ── AR claim directions ──────────────────────────────────────────────────────

def claim_directions(critic, z_inner, snippets, h):
    """For each snippet i: Delta_i = normalize(AR(z), s) - normalize(AR(z\\C_i), s),
    plus the free AR ΔMSE cross-check.

    Returns dict with arrays aligned to ``snippets``:
      deltas   [n,d]  raw direction vectors (difference of L2-normalized AR outputs)
      ar_full  [d]    AR(z) raw
      ar_minus [n,d]  AR(z\\C_i) raw
      dmse     [n]    MSE(AR(z\\C_i), h) - MSE(AR(z), h)   (paper's "removing a true
                      claim hurts reconstruction" - positive => snippet helped)
      mse_full, cos_full  scalars for AR(z) vs h
    """
    s = critic.mse_scale
    ar_full = critic.reconstruct(z_inner).float()
    mse_full, cos_full = critic.score(z_inner, h)
    nf = normalize_activation(ar_full.view(1, -1), s)[0]
    n = len(snippets)
    deltas = np.zeros((n, ar_full.numel()), np.float32)
    ar_minus = np.zeros((n, ar_full.numel()), np.float32)
    dmse = np.zeros(n, np.float32)
    for i in range(n):
        z_i = "\n\n".join(snippets[:i] + snippets[i + 1:]).strip()
        if z_i:
            ar_i = critic.reconstruct(z_i).float()
            mse_i, _ = critic.score(z_i, h)
        else:                                   # would remove everything: no-op fallback
            ar_i, mse_i = ar_full.clone(), mse_full
        ni = normalize_activation(ar_i.view(1, -1), s)[0]
        deltas[i] = (nf - ni).numpy()
        ar_minus[i] = ar_i.numpy()
        dmse[i] = float(mse_i - mse_full)
    return dict(deltas=deltas, ar_full=ar_full.numpy().astype(np.float32),
                ar_minus=ar_minus, dmse=dmse,
                mse_full=float(mse_full), cos_full=float(cos_full))


def project_out(h, dhat, alpha=1.0):
    """h' = h - alpha (h . dhat) dhat. Removes the (unit) direction dhat from h."""
    return h - alpha * float(h @ dhat) * dhat


# ── the audit of one activation ──────────────────────────────────────────────

def audit_activation(model, tok, sc, family, critic, injector, h, *,
                     R=10, alpha=1.0, temperature=0.4, max_new=200,
                     null_pct=95.0, spec_margin=1.5, seed=0, idx=-1, doc_id="",
                     position=-1, n_tokens=-1, read_token="", precomputed_z=None):
    """Run the full claim-ablation test for one activation h.

    Returns (record, vectors):
      record   JSON-safe dict (text, snippets, per-snippet scalars + null samples +
               specificity row). One line of records.jsonl.
      vectors  dict of np.float32 arrays for vectors/<idx>.npz (full h, deltas, AR
               outputs, null matrix, per-token logp under h and each ablation). None
               when the activation is skipped.

    Cost: 1 generation (unless precomputed_z) + (1 + n + R) teacher-forced forwards.
    """
    h = np.asarray(h, dtype=np.float32)
    inj_scale = sc["injection_scale"]
    base = dict(idx=int(idx), doc_id=doc_id, position=int(position),
                n_tokens=int(n_tokens), read_token=read_token,
                h_norm=float(np.linalg.norm(h)))

    z_assistant = (precomputed_z if precomputed_z is not None
                   else generate_explanation(model, tok, sc, family, injector, h,
                                             temperature=temperature, max_new=max_new,
                                             seed=seed))
    m = EXPLANATION_RE.search(z_assistant)
    z_inner = (m.group(1) if m else z_assistant).strip()
    snippets = split_snippets(z_inner)
    cjk = bool(CJK.search(z_inner))
    rec = dict(base, explanation=z_assistant.strip(), z_inner=z_inner,
               snippets=snippets, n_snippets=len(snippets), cjk=cjk,
               status="ok", note="")

    if cjk:
        rec.update(status="skip", note="CJK in explanation (injection-failure smell)")
        return rec, None
    if len(snippets) < 2:
        rec.update(status="skip", note=f"only {len(snippets)} snippet(s); need >= 2")
        return rec, None

    # AR side: claim directions + free ΔMSE cross-check.
    cd = claim_directions(critic, z_inner, snippets, h)
    deltas, dmse = cd["deltas"], cd["dmse"]
    n = len(snippets)

    # AV side: teacher-forced spans.
    input_ids, marker_pos, offsets, full_text, cont_start = build_tf_sequence(
        tok, sc, family, z_assistant)
    assert marker_pos == injector.p, f"marker drift {marker_pos} vs {injector.p}"
    ranges = snippet_spans(offsets, full_text, snippets, cont_start)
    located = [r is not None for r in ranges]

    l0, tlp_h = _inject_span_logp(model, injector, input_ids, h, inj_scale, ranges)

    spec_rows = np.full((n, n), np.nan, np.float32)     # spec_rows[i][j] = l0_j - l(ablate i)_j
    dlogp = np.full(n, np.nan, np.float32)
    proj_mag = np.zeros(n, np.float32)
    delta_norm = np.zeros(n, np.float32)
    tlp_ablate = np.zeros((n, tlp_h.shape[0]), np.float32)
    for i in range(n):
        dn = float(np.linalg.norm(deltas[i]))
        delta_norm[i] = dn
        if dn < _EPS or not located[i]:
            hp = h.copy()                       # AR insensitive / span missing -> no-op
        else:
            dhat = deltas[i] / dn
            proj_mag[i] = float(h @ dhat)
            hp = project_out(h, dhat, alpha)
        li, tlp_i = _inject_span_logp(model, injector, input_ids, hp, inj_scale, ranges)
        tlp_ablate[i] = tlp_i
        for j in range(n):
            spec_rows[i, j] = l0[j] - li[j]
        dlogp[i] = l0[i] - li[i]

    # Random-direction null: how much does an arbitrary rank-1 ablation move each span?
    rng = np.random.default_rng(seed + 1)
    null = np.full((R, n), np.nan, np.float32)
    d = h.shape[0]
    for r in range(R):
        u = rng.standard_normal(d).astype(np.float32)
        u /= np.linalg.norm(u)
        lr, _ = _inject_span_logp(model, injector, input_ids,
                                  project_out(h, u, alpha), inj_scale, ranges)
        for j in range(n):
            null[r, j] = l0[j] - lr[j]
    null_thr = np.nanpercentile(null, null_pct, axis=0)
    null_mean = np.nanmean(null, axis=0)

    # Per-snippet decision (4-way).
    #   depends_on_h : the targeted ablation beats the random-direction null on its
    #                  own span - the effect is real, not "any perturbation hurts".
    #   specific     : ablating THIS claim's direction hurts THIS claim more than
    #                  ablating any OTHER claim's direction does, by spec_margin
    #                  (column dominance of the effect matrix). The random null
    #                  cannot be the specificity yardstick: random directions are
    #                  ~orthogonal to h so their effect is ~0, whereas AR claim
    #                  directions are h-aligned and always move the read somewhat.
    # verdicts: confabulated (not depends) / entangled (depends, not specific) /
    #           grounded (depends + specific) / unlocated.
    results = []
    counts = dict(grounded=0, entangled=0, confabulated=0, unlocated=0)
    located_count = 0
    for i in range(n):
        is_loc = bool(located[i])
        located_count += is_loc
        row_off = [spec_rows[i, j] for j in range(n) if j != i]   # collateral i causes
        col_off = [spec_rows[k, i] for k in range(n) if k != i]   # others' effect on i
        spec_off = float(np.nanmean(row_off)) if row_off else 0.0
        col_off_max = float(np.nanmax(col_off)) if col_off else 0.0
        denom = max(col_off_max, 0.0)
        spec_ratio = _f(dlogp[i] / denom) if denom > _EPS else None   # None = no collateral
        depends = bool(is_loc and dlogp[i] > null_thr[i])
        specific = bool(dlogp[i] > spec_margin * denom)
        verdict = _classify(is_loc, depends, specific)
        counts[verdict] += 1
        span_len = (ranges[i][1] - ranges[i][0]) if ranges[i] else 0
        results.append(dict(
            i=i, text=snippets[i], located=is_loc,
            token_span=list(ranges[i]) if ranges[i] else None, n_span_tokens=span_len,
            l0=_f(l0[i]), l1=_f(l0[i] - dlogp[i]), dlogp=_f(dlogp[i]),
            dlogp_per_token=_f(dlogp[i] / span_len) if span_len else None,
            null_mean=_f(null_mean[i]), null_thr=_f(null_thr[i]),
            null_dlogps=[_f(x) for x in null[:, i]],
            spec_row=[_f(x) for x in spec_rows[i]], spec_offdiag=_f(spec_off),
            col_off_max=_f(col_off_max), spec_ratio=spec_ratio,
            depends_on_h=depends, specific=specific,
            dmse=_f(dmse[i]), delta_norm=_f(delta_norm[i]),
            proj_mag=_f(proj_mag[i]),
            proj_cos=_f(proj_mag[i] / (base["h_norm"] + _EPS)),
            grounded=bool(verdict == "grounded"), verdict=verdict,
        ))

    rec.update(mse_full=cd["mse_full"], cos_full=cd["cos_full"],
               null_pct=null_pct, R=int(R), alpha=float(alpha),
               spec_margin=float(spec_margin),
               grounded_count=int(counts["grounded"]),
               entangled_count=int(counts["entangled"]),
               confab_count=int(counts["confabulated"]),
               located_count=int(located_count), snippet_results=results)
    vectors = dict(
        h=h, deltas=deltas, ar_full=cd["ar_full"], ar_minus=cd["ar_minus"],
        null=null.astype(np.float32), tok_logp_h=tlp_h.astype(np.float32),
        tok_logp_ablate=tlp_ablate,
        token_ids=input_ids[0].numpy().astype(np.int64),
        ranges=np.array([list(r) if r else [-1, -1] for r in ranges], np.int64),
        marker_pos=np.array([marker_pos], np.int64),
        cont_token_start=np.array([cont_start], np.int64),
    )
    return rec, vectors


def _f(x):
    x = float(x)
    return None if (x != x) else x          # NaN -> None for clean JSON


def _classify(located, depends, specific):
    if not located:
        return "unlocated"
    if not depends:
        return "confabulated"      # within the random-direction null: not read from h
    return "grounded" if specific else "entangled"


def _atom_specificity(dlogp_i, null_thr_i, col_off, located_i,
                      spec_margin, spec_pct, spec_floor):
    """Specificity decision for one atom claim (CP4 rule: percentile denominator + floor).

    ``col_off`` is the list of collateral effects on claim i from ablating every OTHER claim
    (spec_rows[k, i], k != i). The denominator is the ``spec_pct`` percentile of that list, not
    its max: max over n-1 siblings is an extreme-value statistic that grows with claim count,
    so it makes 'grounded' mechanically harder as the decomposition gets finer (CP4 measured
    corr(grounded-fraction, n) = -0.74 for max vs ~0 for p90). The percentile is scale-stable
    and robust to a single strongly-coupled sibling. ``spec_floor`` requires a real effect
    (>= spec_floor nats) on top of percentile dominance, so a near-null claim whose collateral
    is also near-null is not called grounded. Returns
    (denom, depends, specific, verdict, col_off_max, col_off_pct)."""
    col_off_max = float(np.nanmax(col_off)) if col_off else 0.0     # kept for transparency
    col_off_pct = float(np.nanpercentile(col_off, spec_pct)) if col_off else 0.0
    denom = max(col_off_pct, 0.0)
    depends = bool(located_i and dlogp_i > null_thr_i)
    specific = bool(dlogp_i > spec_margin * denom and dlogp_i >= spec_floor)
    return denom, depends, specific, _classify(located_i, depends, specific), col_off_max, col_off_pct


# ── atom-claim path (finer, typed decomposition) ─────────────────────────────
#
# The block path above splits z on blank lines into ~3 coarse snippets. The atom path
# instead takes a PRECOMPUTED two-layer decomposition (theme/detail/connective partition +
# entity overlay, from decomp_demo/segment.py) and scores the Layer-1 CONTENT claims
# (theme/detail) individually. Everything downstream of "which char span is this claim"
# is identical to the block path - same injector, same teacher forcing, same random null,
# same verdict thresholds - so a block record and an atom record are directly comparable.
# The three mechanical differences, all forced by moving from blank-line snippets to a
# character-exact tiling:
#   * claims carry CHAR spans into z_inner and are removed by slicing (z_inner[:s]+z_inner[e:]),
#     because a tiling cannot be rebuilt by "\n\n".join;
#   * each claim's char span maps to a token range through the ONE char->token offset map of
#     the teacher-forced sequence (not text.find of the claim text, which mislocates a short
#     or duplicated atom like the bare token "possess");
#   * connectives are carried through UNSCORED (scored=False, verdict="n/a") and excluded
#     from the specificity matrix and every denominator.
# Entities (Layer 2) overlap their container claims, so their ablation directions are
# coupled; the container-exclusion rule for them is decided at CP4 and they are NOT scored
# here.

_CONTENT_ATOM_TYPES = ("theme", "detail")


def atoms_to_claims(atoms):
    """Split a two-layer atom list into scored content claims and unscored connectives.

    Content atoms (theme/detail) become claims carrying their char span into z_inner plus
    the segmenter's metadata (subtype/role/hedged/assertion/regex_confirmed). Connective
    atoms are kept as display-only spans. Returns (claims, connectives)."""
    claims, conns = [], []
    for a in atoms:
        ty = a.get("type")
        if ty in _CONTENT_ATOM_TYPES:
            claims.append(dict(
                text=a["text"], type=ty, subtype=a.get("subtype", ""),
                role=a.get("role", ""), hedged=list(a.get("hedged", []) or []),
                assertion=a.get("assertion", ""),
                regex_confirmed=a.get("regex_confirmed"),
                char_span=(int(a["start"]), int(a["end"]))))
        elif ty == "connective":
            conns.append(dict(text=a["text"], subtype=a.get("subtype", ""),
                              char_span=(int(a["start"]), int(a["end"]))))
    return claims, conns


def atom_claim_directions(critic, z_inner, claims, h):
    """Like claim_directions, but each claim is removed from z_inner by CHAR-SPAN deletion
    (z_i = z_inner[:s] + z_inner[e:]) rather than by dropping a blank-line snippet. Arrays
    are aligned to ``claims``; fields match claim_directions so the vectors schema is shared.
    """
    s = critic.mse_scale
    ar_full = critic.reconstruct(z_inner).float()
    mse_full, cos_full = critic.score(z_inner, h)
    nf = normalize_activation(ar_full.view(1, -1), s)[0]
    n = len(claims)
    deltas = np.zeros((n, ar_full.numel()), np.float32)
    ar_minus = np.zeros((n, ar_full.numel()), np.float32)
    dmse = np.zeros(n, np.float32)
    for i, c in enumerate(claims):
        s0, e0 = c["char_span"]
        z_i = (z_inner[:s0] + z_inner[e0:]).strip()
        if z_i:
            ar_i = critic.reconstruct(z_i).float()
            mse_i, _ = critic.score(z_i, h)
        else:                                   # would remove everything: no-op fallback
            ar_i, mse_i = ar_full.clone(), mse_full
        ni = normalize_activation(ar_i.view(1, -1), s)[0]
        deltas[i] = (nf - ni).numpy()
        ar_minus[i] = ar_i.numpy()
        dmse[i] = float(mse_i - mse_full)
    return dict(deltas=deltas, ar_full=ar_full.numpy().astype(np.float32),
                ar_minus=ar_minus, dmse=dmse,
                mse_full=float(mse_full), cos_full=float(cos_full))


def atom_spans(offsets, full_text, z_inner, claims, cont_char_start):
    """Map each claim's [start,end) char span in z_inner to a (lo,hi) token range.

    Locates z_inner ONCE inside the teacher-forced text (search from the continuation start),
    then offsets each claim span into full_text coordinates and takes every token that
    overlaps it. Robust to short/duplicate claim text, unlike per-claim text.find. Returns a
    list aligned to ``claims``; an entry is None when z_inner or the span cannot be placed."""
    z0 = full_text.find(z_inner, cont_char_start)
    if z0 < 0:
        z0 = full_text.find(z_inner)
    spans = []
    for c in claims:
        if z0 < 0:
            spans.append(None)
            continue
        cstart, cend = z0 + c["char_span"][0], z0 + c["char_span"][1]
        lo = hi = None
        for ti, (cs, ce) in enumerate(offsets):
            if ce <= cs:                        # special / zero-width token
                continue
            if ce > cstart and cs < cend:       # token overlaps the claim char span
                lo = ti if lo is None else lo
                hi = ti + 1
        spans.append((lo, hi) if lo is not None else None)
    return spans


def audit_activation_atoms(model, tok, sc, family, critic, injector, h, *,
                           z_assistant, atoms, z_inner=None,
                           R=10, alpha=1.0, null_pct=95.0, spec_margin=1.5,
                           spec_pct=90.0, spec_floor=1.0, seed=0,
                           idx=-1, doc_id="", position=-1, n_tokens=-1, read_token=""):
    """Claim-ablation over a precomputed two-layer decomposition (atom mode).

    Scores the Layer-1 content claims (theme/detail) of ``atoms`` against the reused AV text
    ``z_assistant``; connectives are carried unscored; entities are not scored here. Mirrors
    audit_activation's measurement exactly (same injector, teacher forcing, random-direction
    null, verdict thresholds) but over char-span claims. ``z_inner`` defaults to the
    <explanation> body of ``z_assistant`` - pass the segmenter's own z_inner (identical) so
    the atom offsets line up. Returns (record, vectors), shape-compatible with
    audit_activation: ``snippet_results`` holds the scored content claims (so summarize()
    keeps working) and a ``by_type`` block + ``connective_results`` are added."""
    h = np.asarray(h, dtype=np.float32)
    inj_scale = sc["injection_scale"]
    base = dict(idx=int(idx), doc_id=doc_id, position=int(position),
                n_tokens=int(n_tokens), read_token=read_token,
                h_norm=float(np.linalg.norm(h)))
    m = EXPLANATION_RE.search(z_assistant)
    zi = (z_inner if z_inner is not None else (m.group(1) if m else z_assistant)).strip()
    claims, conns = atoms_to_claims(atoms)
    cjk = bool(CJK.search(zi))
    rec = dict(base, claim_mode="atoms", explanation=z_assistant.strip(), z_inner=zi,
               n_content=len(claims), n_connective=len(conns), n_snippets=len(claims),
               cjk=cjk, status="ok", note="")
    if cjk:
        rec.update(status="skip", note="CJK in explanation (injection-failure smell)")
        return rec, None
    if len(claims) < 2:
        rec.update(status="skip", note=f"only {len(claims)} content claim(s); need >= 2")
        return rec, None

    cd = atom_claim_directions(critic, zi, claims, h)
    deltas, dmse = cd["deltas"], cd["dmse"]
    n = len(claims)

    input_ids, marker_pos, offsets, full_text, cont_start = build_tf_sequence(
        tok, sc, family, z_assistant)
    assert marker_pos == injector.p, f"marker drift {marker_pos} vs {injector.p}"
    ranges = atom_spans(offsets, full_text, zi, claims, cont_start)
    located = [r is not None for r in ranges]

    l0, tlp_h = _inject_span_logp(model, injector, input_ids, h, inj_scale, ranges)

    spec_rows = np.full((n, n), np.nan, np.float32)     # spec_rows[i][j] = l0_j - l(ablate i)_j
    dlogp = np.full(n, np.nan, np.float32)
    proj_mag = np.zeros(n, np.float32)
    delta_norm = np.zeros(n, np.float32)
    tlp_ablate = np.zeros((n, tlp_h.shape[0]), np.float32)
    for i in range(n):
        dn = float(np.linalg.norm(deltas[i]))
        delta_norm[i] = dn
        if dn < _EPS or not located[i]:
            hp = h.copy()
        else:
            dhat = deltas[i] / dn
            proj_mag[i] = float(h @ dhat)
            hp = project_out(h, dhat, alpha)
        li, tlp_i = _inject_span_logp(model, injector, input_ids, hp, inj_scale, ranges)
        tlp_ablate[i] = tlp_i
        for j in range(n):
            spec_rows[i, j] = l0[j] - li[j]
        dlogp[i] = l0[i] - li[i]

    rng = np.random.default_rng(seed + 1)
    null = np.full((R, n), np.nan, np.float32)
    d = h.shape[0]
    for r in range(R):
        u = rng.standard_normal(d).astype(np.float32)
        u /= np.linalg.norm(u)
        lr, _ = _inject_span_logp(model, injector, input_ids,
                                  project_out(h, u, alpha), inj_scale, ranges)
        for j in range(n):
            null[r, j] = l0[j] - lr[j]
    null_thr = np.nanpercentile(null, null_pct, axis=0)
    null_mean = np.nanmean(null, axis=0)

    results = []
    counts = dict(grounded=0, entangled=0, confabulated=0, unlocated=0)
    located_count = 0
    for i in range(n):
        c = claims[i]
        is_loc = bool(located[i])
        located_count += is_loc
        row_off = [spec_rows[i, j] for j in range(n) if j != i]
        col_off = [spec_rows[k, i] for k in range(n) if k != i]
        spec_off = float(np.nanmean(row_off)) if row_off else 0.0
        denom, depends, specific, verdict, col_off_max, col_off_pct = _atom_specificity(
            dlogp[i], null_thr[i], col_off, is_loc, spec_margin, spec_pct, spec_floor)
        spec_ratio = _f(dlogp[i] / denom) if denom > _EPS else None
        counts[verdict] += 1
        span_len = (ranges[i][1] - ranges[i][0]) if ranges[i] else 0
        results.append(dict(
            i=i, text=c["text"], type=c["type"], subtype=c["subtype"], role=c["role"],
            hedged=c["hedged"], assertion=c["assertion"],
            regex_confirmed=c["regex_confirmed"], char_span=list(c["char_span"]),
            scored=True, located=is_loc,
            token_span=list(ranges[i]) if ranges[i] else None, n_span_tokens=span_len,
            l0=_f(l0[i]), l1=_f(l0[i] - dlogp[i]), dlogp=_f(dlogp[i]),
            dlogp_per_token=_f(dlogp[i] / span_len) if span_len else None,
            null_mean=_f(null_mean[i]), null_thr=_f(null_thr[i]),
            null_dlogps=[_f(x) for x in null[:, i]],
            spec_row=[_f(x) for x in spec_rows[i]], spec_offdiag=_f(spec_off),
            col_off_max=_f(col_off_max), col_off_pct=_f(col_off_pct), spec_ratio=spec_ratio,
            depends_on_h=depends, specific=specific,
            dmse=_f(dmse[i]), delta_norm=_f(delta_norm[i]), proj_mag=_f(proj_mag[i]),
            proj_cos=_f(proj_mag[i] / (base["h_norm"] + _EPS)),
            grounded=bool(verdict == "grounded"), verdict=verdict))

    conn_rows = [dict(text=c["text"], subtype=c["subtype"], char_span=list(c["char_span"]),
                      scored=False, verdict="n/a") for c in conns]

    by_type = {}
    for s in results:
        t = s["type"]
        d0 = by_type.setdefault(
            t, dict(n=0, located=0, grounded=0, entangled=0, confabulated=0, unlocated=0))
        d0["n"] += 1
        d0["located"] += int(s["located"])
        d0[s["verdict"]] += 1

    rec.update(mse_full=cd["mse_full"], cos_full=cd["cos_full"],
               null_pct=null_pct, R=int(R), alpha=float(alpha),
               spec_margin=float(spec_margin), spec_pct=float(spec_pct),
               spec_floor=float(spec_floor), spec_denom=f"p{int(spec_pct)}+floor{spec_floor:g}",
               grounded_count=int(counts["grounded"]),
               entangled_count=int(counts["entangled"]),
               confab_count=int(counts["confabulated"]),
               unlocated_count=int(counts["unlocated"]),
               located_count=int(located_count),
               snippet_results=results, connective_results=conn_rows, by_type=by_type)
    vectors = dict(
        h=h, deltas=deltas, ar_full=cd["ar_full"], ar_minus=cd["ar_minus"],
        null=null.astype(np.float32), tok_logp_h=tlp_h.astype(np.float32),
        tok_logp_ablate=tlp_ablate,
        token_ids=input_ids[0].numpy().astype(np.int64),
        ranges=np.array([list(r) if r else [-1, -1] for r in ranges], np.int64),
        marker_pos=np.array([marker_pos], np.int64),
        cont_token_start=np.array([cont_start], np.int64),
    )
    return rec, vectors
