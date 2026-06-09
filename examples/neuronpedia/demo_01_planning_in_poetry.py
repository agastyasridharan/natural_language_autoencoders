#!/usr/bin/env python3
"""Demo 1 — Planning in Poetry  (NLA paper, "Case Studies > Planning in Poetry").

The paper revisits Lindsey et al.'s "On the Biology of a Large Language Model"
finding that the model plans a rhyme *before* writing the second line. Given the
couplet::

    He saw a carrot and had to grab it,
    His hunger was like a starving rabbit

they show that at the newline token ending line 1 — i.e. *before* "rabbit" is
emitted — the residual activation already encodes a plan to end the couplet with
"rabbit". The NLA explanation at that newline mentions the rabbit rhyme.

This demo replicates the *reading* half of that experiment on a hosted open-model
NLA: it teacher-forces the couplet, then verbalizes the tokens around the line
break and checks whether the AV description already references the planned rhyme.

Caveat (paper, faithfully): NLAs confabulate, and the hosted open-model NLAs are
~2/3-depth — weaker and noisier than the Opus 4.6 NLA in the paper. We therefore
report what the AV actually says rather than asserting a fixed result. The paper
also notes the planning signal is "diffuse across tokens", so we scan a window.
"""

from __future__ import annotations

from _common import banner, cache_link, make_arg_parser, open_nla, rule, run, show

USER = "Write a two-line rhyming couplet. Start it with: He saw a carrot and had to grab it,"
COUPLET = "He saw a carrot and had to grab it,\nHis hunger was like a starving rabbit"

# Words that would indicate the AV has surfaced the planned end-rhyme.
RHYME_CUES = ["rabbit", "bunny", "hare", "rhyme", "rhyming", "grab it", "couplet",
              "ends with", "end the", "-it", "habit"]


def main() -> None:
    ap = make_arg_parser(__doc__)
    ap.add_argument("--radius", type=int, default=3,
                    help="how many tokens on each side of the line break to scan")
    args = ap.parse_args()

    banner("Planning in Poetry — does the model plan the rhyme at the line break?",
           "Case Studies > Planning in Poetry")

    with open_nla(args) as nla:
        messages = [{"role": "user", "content": USER},
                    {"role": "assistant", "content": COUPLET}]
        toks = nla.tokenize(messages)

        # Locate the newline ENDING line 1 of the couplet — i.e. inside the
        # assistant reply, not the assistant header's blank line (and not the
        # "grab it" in the user prompt). Anchor on the assistant role header,
        # skip its blank line, then take the first '\n' after "grab" in content.
        asst = [t for t in toks if t.token.strip() == "assistant"]
        asst_pos = asst[-1].position if asst else 0
        hdr_nl = next((t for t in toks if "\n" in t.token and t.position > asst_pos), None)
        content_start = hdr_nl.position if hdr_nl else asst_pos
        content = [t for t in toks if t.position > content_start]
        grab = next((t for t in content if "grab" in t.token.lower()), None)
        after = grab.position if grab else content_start
        newline = next((t for t in content if "\n" in t.token and t.position > after), None)
        if newline is None:
            raise SystemExit("Could not locate the line-break token; aborting.")

        lo, hi = newline.position - args.radius, newline.position + args.radius
        scan = [t.position for t in toks if lo <= t.position <= hi]
        ctx = " ".join(t.token for t in toks if lo - 1 <= t.position <= hi + 1)
        print(f"Line-break token at position {newline.position} (token={newline.token!r}).")
        print(f"Scanning positions {scan} — context around the break: ...{ctx}...")

        res = nla.verbalize(messages, scan, tokens=[t.token for t in toks])
        by_pos = res.by_position()

        rule("AV explanations around the line break (looking for the rhyme plan):")
        hits = []
        for p in scan:
            v = by_pos.get(p)
            if v is None:
                continue
            show(v, highlight=RHYME_CUES)
            if any(c.lower() in v.description.lower() for c in RHYME_CUES):
                hits.append((p, v))

        rule("Verdict")
        if hits:
            print(f"  The rhyme plan IS surfaced at {len(hits)} of {len(scan)} scanned "
                  f"position(s): {[p for p, _ in hits]}.")
            best = max(hits, key=lambda pv: pv[1].cosine_similarity)
            print(f"  Strongest at position {best[0]} (cos={best[1].cosine_similarity:.3f}) — "
                  f"the activation references the planned ending before line 2 is written,")
            print("  replicating the paper's 'plans the rhyme at the newline' result.")
        else:
            print("  No explicit rhyme reference surfaced in this window. Consistent with the")
            print("  paper's caveats: open-model NLAs are weaker, the signal is diffuse, and")
            print("  the AV confabulates. Try --radius 5, the other --model, or re-run (T>0).")
        cache_link(res, nla.model_id)


if __name__ == "__main__":
    run(main)
