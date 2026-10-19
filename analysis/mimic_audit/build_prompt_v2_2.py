"""Build prompt_v2_2.txt: v2.1 plus ONE subtle Block-2 anchoring fix.

v2.1 fixed block-3 endings (restored the trailing "..."). What's left is a
Block-2 tell the discriminator already named once (customer_support: "Block 2
quotes a specific in-text fragment and analyzes sentence structure; B describes
content narratively instead"). Measured across all 128 held-out v2.1 pairs:

  Block 2 contains >=1 verbatim quote from the text:  real AV 37%  vs  mimic 8%
  avg quoted fragments in block 2:                    real 0.50   vs  mimic 0.08

The real AV often opens block 2 with a short lifted fragment ("The question
'Are you fine being' mid-message...", "The sentence 'The total with tip comes'
beginning..."); the mimic paraphrases narratively. v2_2 nudges block 2 to
anchor to one short verbatim fragment (occasionally two) where it sharpens the
location, calibrated toward the real ~1/3 rate rather than forced every time.
Inherits both v2.1 ending fixes; nothing else is touched.
"""
from __future__ import annotations

from pathlib import Path

import build_prompt_v1 as v1
import build_prompt_v2 as v2
import build_prompt_v2_1 as v21

HERE = Path(__file__).resolve().parent
OUT = HERE / "prompt_v2_2.txt"

OLD_B2 = ("- Block 2: one sentence on the local structure, what this particular "
          "sentence or step is doing right at this token.")
NEW_B2 = ("- Block 2: one sentence on the local structure, what this particular "
          "sentence or step is doing right at this token; where it sharpens the "
          "location, anchor it to one short fragment quoted verbatim from the "
          "text here (occasionally two) rather than only paraphrasing.")


def main():
    pre = (v2.PREAMBLE
           .replace(v21.OLD_B3, v21.NEW_B3)
           .replace(OLD_B2, NEW_B2))
    clo = v2.CLOSER.replace(v21.OLD_CLO, v21.NEW_CLO)
    assert v21.NEW_B3 in pre, "block-3 (v2.1) replacement did not match"
    assert NEW_B2 in pre, "block-2 replacement did not match"
    assert clo != v2.CLOSER, "closer (v2.1) replacement did not match"
    v1.PREAMBLE = pre
    v1.CLOSER = clo
    v1.OUT = OUT
    v1.main()


if __name__ == "__main__":
    main()
