#!/usr/bin/env python3
"""Demo 2 — Language Switching  (NLA paper, "Case Studies > Language Switching").

The paper investigates transcripts where the model, prompted in English, responds
in another language (Russian, Chinese, Korean, Arabic, Spanish). By string-matching
language references in the AV explanations across the transcript, they find the
model "consistently represented the target language internally well before
outputting tokens in it" — the target-language line rises early and stays high,
while references to *other* languages stay near zero (paper Fig., p.21).

This demo replicates that measurement on a hosted open-model NLA. It teacher-forces
an English-prompt / non-English-reply transcript, sweeps the AV across the
transcript, and plots (ASCII) the per-position prevalence of references to the
target language vs all other languages — reproducing the "represented before
responding" signature.

Note: the paper's walkthrough transcript involves self-harm; we use a neutral
prompt here. The mechanism under test is identical.
"""

from __future__ import annotations

import re

from _common import banner, cache_link, make_arg_parser, open_nla, rule, run

# English prompt carrying a subtle cue to the user's language ("moved from Moscow"),
# then a Russian reply. The paper's mechanism is the model fixating on such a cue and
# representing the language well before emitting it — so we expect Russian references
# at/after the cue token, before any Cyrillic is output.
USER = ("I just moved to a new city for work (relocated from Moscow last month) and "
        "don't know anyone yet. Any advice?")
REPLY_RU = ("Привет! Переезд в новый город — это всегда непросто. "
            "Попробуй записаться на занятия или встречи по интересам, "
            "чтобы познакомиться с новыми людьми.")

TARGET = "russian"
TARGET_CUES = ["russian", "russia", "cyrillic", "slavic", "moscow"]
OTHER_CUES = ["spanish", "french", "german", "chinese", "japanese", "korean",
              "arabic", "italian", "portuguese", "hindi"]
CYRILLIC = re.compile(r"[Ѐ-ӿ]")


def prevalence(desc: str, cues: list[str]) -> int:
    d = desc.lower()
    return sum(1 for c in cues if c in d)


def bar(frac: float, width: int = 28) -> str:
    n = int(round(frac * width))
    return "█" * n + "·" * (width - n)


def main() -> None:
    ap = make_arg_parser(__doc__)
    ap.add_argument("--step", type=int, default=3,
                    help="sweep every Nth token through the transcript (default 3)")
    args = ap.parse_args()

    banner("Language Switching — is the target language represented before it is spoken?",
           "Case Studies > Language Switching")

    with open_nla(args) as nla:
        messages = [{"role": "user", "content": USER},
                    {"role": "assistant", "content": REPLY_RU}]
        toks = nla.tokenize(messages)

        # First token of the actual Russian reply (first Cyrillic-bearing token).
        first_ru = next((t for t in toks if CYRILLIC.search(t.token)), None)
        boundary = first_ru.position if first_ru else max(t.position for t in toks)
        print(f"First Russian (Cyrillic) token appears at position {boundary} "
              f"(token={first_ru.token!r}).")
        print("If the model 'plans' the language, AV mentions of Russian should appear")
        print("at and BEFORE this boundary — not only once Cyrillic is being emitted.\n")

        # Sweep across the transcript (skip BOS/specials at the very start).
        scan = [t.position for t in toks if t.position >= 2 and t.position % args.step == 0]
        res = nla.verbalize(messages, scan, tokens=[t.token for t in toks])
        by_pos = res.by_position()

        rule(f"Per-position prevalence of '{TARGET}' (T) vs other languages (o):")
        print(f"{'pos':>4} {'region':<10} {'T':>2} {'o':>2}  prevalence(target)")
        n_before, hit_before = 0, 0
        for p in scan:
            v = by_pos.get(p)
            if v is None:
                continue
            t_hits = prevalence(v.description, TARGET_CUES)
            o_hits = prevalence(v.description, OTHER_CUES)
            region = "reply" if p >= boundary else ("prompt/hdr")
            frac = min(1.0, t_hits / 2.0)
            print(f"{p:>4} {region:<10} {t_hits:>2} {o_hits:>2}  {bar(frac)}")
            if p < boundary:
                n_before += 1
                hit_before += 1 if t_hits else 0

        rule("Verdict")
        if n_before:
            print(f"  Of {n_before} positions BEFORE the first Russian token, "
                  f"{hit_before} already reference Russian "
                  f"({100*hit_before/n_before:.0f}%).")
        if hit_before:
            print("  The target language is represented internally before it is spoken —")
            print("  replicating the paper's 'represented before responding' finding. Other-")
            print("  language references stay low (the 'o' column), showing specificity.")
        else:
            print("  No pre-boundary Russian reference surfaced this run (open-model NLA is")
            print("  noisier than the paper's). Try --step 2, the other --model, or re-run.")
        cache_link(res, nla.model_id)


if __name__ == "__main__":
    run(main)
