#!/usr/bin/env python3
"""Demo 4 — Confabulation & the recurrence heuristic  (NLA paper, quantitative evals,
"Characterizing NLA confabulations").

NLAs hallucinate specifics. The paper's practical reading guide, established in the
quantitative section, is:

  * Read NLA explanations for the THEMES they surface, not individual claims.
  * "We place more weight on a specific claim when it appears repeatedly in NLA
    explanations over multiple [adjacent] tokens" — recurrence ≈ reliability
    (recurring claims were true ~4.2/10 vs ~2.2/10 for one-offs).
  * The AR's reconstruction fidelity (cosine) is a confidence signal.

This demo replicates that methodology: it verbalizes a contiguous run of token
positions over a known context, then measures which content words RECUR across
adjacent positions (the trustworthy thematic signal) versus appear once (likely
confabulated detail). It also reports the per-position reconstruction fidelity.

The point is the *method*, not a fixed answer: for a context about Paris/France,
recurring words like "paris/france/capital/landmark" should dominate, while
one-off specifics (stray names, places, numbers) are flagged as low-trust.
"""

from __future__ import annotations

import re
from collections import Counter

from _common import banner, cache_link, make_arg_parser, open_nla, rule, run

USER = "Tell me one interesting fact about the Eiffel Tower."
REPLY = ("The Eiffel Tower in Paris, France was completed in 1889 for the World's "
         "Fair and was the tallest structure in the world until 1930.")

STOP = set("""the a an and or of to in for on at by with from as is was were are be been
being it its this that these those his her their our your my we you they he she i
into over under about which who whom whose what when where why how not no yes will
would can could should may might must do does did has have had been more most some
any each both than then so such just like also very can't dont don't
""".split())

# Generic NLA-explanation boilerplate — recurs in almost every description and
# carries no content, so it is filtered out to expose the real thematic signal.
STOP |= set("""likely fact facts list lists format formats standard structure
structures text texts sentence sentences answer answers result results content
contents trivia interesting line lines word words phrase phrases context begins
ends end begin continues continuation pattern patterns section sections page pages
output outputs prompt prompts response responses question questions example examples
appears seems suggesting indicating implying followed following completing""".split())
WORD = re.compile(r"[a-zA-Z][a-zA-Z'-]{2,}")


def content_words(text: str) -> set[str]:
    return {w.lower() for w in WORD.findall(text) if w.lower() not in STOP and len(w) >= 4}


def main() -> None:
    ap = make_arg_parser(__doc__)
    ap.add_argument("--span", type=int, default=8,
                    help="number of adjacent reply tokens to verbalize (default 8)")
    args = ap.parse_args()

    banner("Confabulation & Recurrence — recurring themes are trustworthy, hapaxes are not",
           "Quantitative Evaluations > Characterizing NLA confabulations")

    with open_nla(args) as nla:
        messages = [{"role": "user", "content": USER},
                    {"role": "assistant", "content": REPLY}]
        toks = nla.tokenize(messages)

        # A contiguous run of reply tokens (skip the assistant header).
        asst = [t for t in toks if t.token.strip() in ("assistant", "model")]
        start = asst[-1].position + 3 if asst else toks[len(toks) // 2].position
        reply = [t for t in toks if t.position > start]
        span = reply[: args.span]
        positions = [t.position for t in span]
        print(f"Verbalizing {len(positions)} adjacent reply tokens "
              f"(positions {positions[0]}..{positions[-1]}):")
        print("  " + " ".join(t.token for t in span).replace("\n", "\\n"))

        res = nla.verbalize(messages, positions, tokens=[t.token for t in toks])
        by_pos = res.by_position()

        # Document-frequency of each content word across the per-position descriptions,
        # plus the longest run of CONSECUTIVE positions it appears in (adjacency).
        df = Counter()
        present = {}
        cosines = []
        for p in positions:
            v = by_pos.get(p)
            if v is None:
                present[p] = set()
                continue
            words = content_words(v.description)
            present[p] = words
            for w in words:
                df[w] += 1
            cosines.append(v.cosine_similarity)

        def longest_run(word: str) -> int:
            best = cur = 0
            for p in positions:
                if word in present.get(p, ()):
                    cur += 1
                    best = max(best, cur)
                else:
                    cur = 0
            return best

        n = len(positions)
        adjacency = {w: longest_run(w) for w in df}

        rule(f"Per-position reconstruction fidelity (AR cosine), n={n}:")
        for p in positions:
            v = by_pos.get(p)
            if v:
                print(f"  pos {p:>3} token={v.token!r:<12} cos={v.cosine_similarity:.3f}")
        if cosines:
            print(f"  mean cos = {sum(cosines)/len(cosines):.3f}  "
                  f"(higher = AR reconstructs the activation better from the AV text)")

        rule("RECURRING themes (df >= 2 and appear in >= 2 adjacent positions) — TRUST these:")
        themes = sorted(
            ((w, df[w], adjacency[w]) for w in df if df[w] >= 2 and adjacency[w] >= 2),
            key=lambda x: (x[2], x[1]), reverse=True,
        )
        if themes:
            for w, d, adj in themes[:15]:
                print(f"  {w:<18} df={d}/{n}  max_adjacent_run={adj}  {'■'*adj}")
        else:
            print("  (none cleared the bar this run — try a larger --span)")

        rule("ONE-OFF specifics (df == 1) — LIKELY CONFABULATION, do not trust individually:")
        hapax = sorted(w for w in df if df[w] == 1)
        print("  " + (", ".join(hapax[:30]) if hapax else "(none)"))

        rule("Verdict")
        print(f"  {len(themes)} recurring theme(s) vs {len(hapax)} one-off term(s).")
        print("  Per the paper's heuristic, the recurring themes are the reliable signal;")
        print("  the one-off specifics are where NLAs confabulate. This replicates the")
        print("  'read for recurring themes, not individual claims' methodology.")
        cache_link(res, nla.model_id)


if __name__ == "__main__":
    run(main)
