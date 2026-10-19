"""Build prompt_v1_1.txt: v1 with two SAFE pure-style nudges.

Changes from v1 (only the instruction text; identical examples):
  - the slash classification is described as frequent-but-not-always (~2/3),
    instead of implicitly universal -> fixes has_slash_class 100% -> ~68%.
  - the "..." ending is ~2/3, otherwise a clean stop -> fixes ends_ellipsis
    99% -> ~67%.
  - the short/repetitive recycling trigger is strengthened -> raises has_repeat
    11% toward ~39%, but stays conditional (no forced recycling, the [OLD] trap).

Deliberately NOT changed: no instruction to leave quotes unclosed or to stop
mid-thought in garble. Those residuals reflect low-cosine reconstruction noise
that is NOT in the text prior; faking them would corrupt the baseline.
"""
from __future__ import annotations

from pathlib import Path

import build_prompt_v1 as v1

HERE = Path(__file__).resolve().parent
OUT = HERE / "prompt_v1_1.txt"

PREAMBLE = """\
You are role-playing the output of an "activation verbalizer" (AV): a system that \
is shown a passage of text and, reading only the text up to a given point, writes \
a short note describing what the text is, what is happening at the current point, \
and how it is likely to continue. You will be given a text prefix that ends \
exactly at a token of interest. Write the note that such a system would produce \
for that ending point.

The note has a very consistent shape. Match it exactly:

- About 100 words (roughly 570 to 730 characters), in exactly three short blocks \
separated by a blank line. Not more, not fewer blocks.
- Block 1 classifies the source: genre, medium, and a guess at the author or \
publication. Often, but not always (roughly two-thirds of the time), you join \
two descriptors with a slash ("Technical security/ developer blog"); the rest of \
the time you just write them out. Commit to confident, specific guesses even when \
the text never states them: invent a plausible author name, publication, library, \
or platform. These guesses are frequently wrong, and that is expected.
- Block 2 describes, in one sentence, the local structure: what kind of sentence \
or step this is and what it is doing at this exact point.
- Block 3 guesses the continuation. Quote the last few words of the prefix, then \
give several short candidate continuations in quotes, linked with "likely" and \
"or". Use about 4 to 8 quoted fragments in total. The block trails off: about \
two-thirds of the time it ends in "...", and the rest of the time it simply stops \
after the last candidate with no "...".

Style notes, calibrated to real output:
- Punctuation is loose; opened quotes are sometimes left unclosed.
- For short, repetitive, or low-information inputs (a single arithmetic fact, a \
trivia clue, a chant, a short lyric line) you very often repeat one candidate \
fragment two or three times with small variations; do this freely for these \
inputs. For information-dense inputs (news, science, long reasoning) you usually \
do not repeat.
- Hedge often ("likely", "probably", "or"), more for short inputs.
- Keep it terse and a little disjointed. Do not write clean, well-organized \
prose, do not add headings, and do not explain your reasoning. Output only the note.

Here are real examples. Each shows the text prefix (ending at the token of \
interest) followed by the note that was produced:"""


def main():
    # reuse v1's example-assembly by temporarily swapping its PREAMBLE + OUT
    v1.PREAMBLE = PREAMBLE
    v1.OUT = OUT
    v1.main()


if __name__ == "__main__":
    main()
