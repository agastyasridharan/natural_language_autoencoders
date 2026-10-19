"""Build prompt_v2_1.txt: v2 with ONE targeted block-3 ending fix.

v2 fixed the meta-framing (LLM/AI refs 28 -> 1) and the reframe, but overshot the
discriminator (34.4%) because it ended block 3 with a bare mid-thought stop far
too often (ends_midthought 48% vs real 16%; ends_ellipsis 45% vs real 67%). The
real AV trails off WITH "..." most of the time. v2.1 restores the "..." trailing
as the dominant ending while keeping v2's "no editorial summary" rule and every
other v2 change. Nothing else is touched.
"""
from __future__ import annotations

from pathlib import Path

import build_prompt_v1 as v1
import build_prompt_v2 as v2

HERE = Path(__file__).resolve().parent
OUT = HERE / "prompt_v2_1.txt"

OLD_B3 = ("This block is mid-computation: let it trail off unfinished, sometimes "
          "with an unclosed quote. Do NOT round it off with a tidy concluding or "
          "explanatory sentence.")
NEW_B3 = ("This block is mid-computation: end it by trailing off partway through "
          "the candidate continuations, usually with a trailing \"...\" (and "
          "occasionally an unclosed quote instead). Do NOT round it off with a "
          "tidy concluding or explanatory sentence.")

OLD_CLO = "let block 3 trail off rather than concluding it."
NEW_CLO = "let block 3 trail off with \"...\" rather than concluding it."


def main():
    pre = v2.PREAMBLE.replace(OLD_B3, NEW_B3)
    clo = v2.CLOSER.replace(OLD_CLO, NEW_CLO)
    assert pre != v2.PREAMBLE, "block-3 replacement did not match"
    assert clo != v2.CLOSER, "closer replacement did not match"
    v1.PREAMBLE = pre
    v1.CLOSER = clo
    v1.OUT = OUT
    v1.main()


if __name__ == "__main__":
    main()
