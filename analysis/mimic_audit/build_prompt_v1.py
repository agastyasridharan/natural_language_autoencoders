"""Assemble the FIRST single mimic-prompt draft (prompt_v1.txt).

One fixed prompt. Style is described in a short preamble calibrated to the
fingerprint (corpus/style_fingerprint.json), then carried by a small fixed set
of REAL examples pulled from the TRAIN split only (so nothing leaks into the
eval audit). At inference the model gets the FULL prefix; example prefixes are
shown as a tail to keep the prompt compact.

This only writes the prompt text. It does not call any API or generate a mimic.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRAIN = HERE / "corpus" / "train.jsonl"
OUT = HERE / "prompt_v1.txt"

# (doc_id, position) of the six representative examples chosen from train.
EXAMPLES = [
    ("forum_personal_v3_3", 77),
    ("trivia_short_v2_2", 5),
    ("code_python_v2_3", 99),
    ("news_tech_v3_3", 130),
    ("customer_support_v2_2", 170),
    ("math_cot_v2_3", 250),
]
TAIL = 600

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
publication, often joined with a slash ("Technical security/ developer blog"). \
Commit to confident, specific guesses even when the text never states them: \
invent a plausible author name, publication, library, or platform. These guesses \
are frequently wrong, and that is expected.
- Block 2 describes, in one sentence, the local structure: what kind of sentence \
or step this is and what it is doing at this exact point.
- Block 3 guesses the continuation. Quote the last few words of the prefix, then \
give several short candidate continuations in quotes, linked with "likely" and \
"or". Use about 4 to 8 quoted fragments in total. The block usually trails off \
mid-thought, ending in "..." or just stopping without final punctuation.

Style notes, calibrated to real output:
- Opened quotes are often left unclosed; punctuation is loose.
- For short, repetitive, or low-information inputs (a single arithmetic fact, a \
trivia clue, a chant) you tend to repeat one candidate fragment two or three \
times with small variations. For information-dense inputs (news, science, long \
reasoning) you usually do not repeat.
- Hedge often ("likely", "probably", "or"), more for short inputs.
- Keep it terse and a little disjointed. Do not write clean, well-organized \
prose, do not add headings, and do not explain your reasoning. Output only the note.

Here are real examples. Each shows the text prefix (ending at the token of \
interest) followed by the note that was produced:"""

CLOSER = """\
Now write the note for this text prefix. It ends exactly at the token of \
interest. Output only the note, in the three-block shape shown above.

TEXT PREFIX:
<<INPUT>>"""


def main():
    rows = {}
    for line in TRAIN.open(encoding="utf-8"):
        r = json.loads(line)
        rows[(r["doc_id"], r["position"])] = r

    blocks = [PREAMBLE, ""]
    for i, key in enumerate(EXAMPLES, 1):
        r = rows[key]
        pre = r["prefix"]
        shown = pre if len(pre) <= TAIL else "..." + pre[-TAIL:]
        blocks.append(f"----- EXAMPLE {i} -----")
        blocks.append("TEXT PREFIX:")
        blocks.append(shown)
        blocks.append("")
        blocks.append("NOTE:")
        blocks.append(r["av_description"].strip())
        blocks.append("")
    blocks.append("----- END EXAMPLES -----")
    blocks.append("")
    blocks.append(CLOSER)

    text = "\n".join(blocks) + "\n"
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT.name}  ({len(text):,} chars, {len(EXAMPLES)} examples)")


if __name__ == "__main__":
    main()
