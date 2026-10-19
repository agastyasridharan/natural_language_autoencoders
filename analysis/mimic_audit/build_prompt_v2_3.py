"""Build prompt_v2_3.txt: dial back v2_2's Block-2 over-quoting.

v2_2 tried to close the one place the mimic was too clean (block 2 quoted a
verbatim fragment 8% of the time vs the real AV's 37%). It overshot: "where it
sharpens the location, anchor it to a quoted fragment" was read as "almost
always", driving block-2 quoting to 93%, pushing quoted_spans/dquotes OVER the
real band, and dropping the discriminator 32% -> 25% (deeper caricature).

v2_3 keeps the intent (block 2 should sometimes name the local unit with a
verbatim fragment, like the real AV) but makes paraphrase the default and
quoting the minority case, tied to the trigger the real AV actually uses: it
quotes when the local unit IS one specific phrase/question/clause, and
paraphrases when describing an operation or action. Target: ~1/3, matching real.
Inherits both v2.1 ending fixes; nothing else is touched.
"""
from __future__ import annotations

from pathlib import Path

import build_prompt_v1 as v1
import build_prompt_v2 as v2
import build_prompt_v2_1 as v21
import build_prompt_v2_2 as v22

HERE = Path(__file__).resolve().parent
OUT = HERE / "prompt_v2_3.txt"

# replace the ORIGINAL v2 block-2 line (v22.OLD_B2) with the dialed-back version
NEW_B2 = ("- Block 2: one sentence on the local structure, what this particular "
          "sentence or step is doing right at this token. Usually paraphrase it; "
          "occasionally, when the local unit is one specific phrase, question, or "
          "clause, name it instead by quoting a short verbatim fragment of the "
          "text here.")


def main():
    pre = (v2.PREAMBLE
           .replace(v21.OLD_B3, v21.NEW_B3)
           .replace(v22.OLD_B2, NEW_B2))
    clo = v2.CLOSER.replace(v21.OLD_CLO, v21.NEW_CLO)
    assert v21.NEW_B3 in pre, "block-3 (v2.1) replacement did not match"
    # NEW_B2 keeps the original sentence and appends to it, so its presence
    # (v2.PREAMBLE has no "Usually paraphrase") proves the replacement landed.
    assert NEW_B2 in pre, "block-2 (v2.3) replacement did not match"
    assert clo != v2.CLOSER, "closer (v2.1) replacement did not match"
    v1.PREAMBLE = pre
    v1.CLOSER = clo
    v1.OUT = OUT
    v1.main()


if __name__ == "__main__":
    main()
