#!/usr/bin/env python3
"""Apply the Gemma-3 SGLang input_embeds bypass to the *installed* sglang.

Gemma families only. The canonical change is patches/nla_gemma3_mm_input_embeds.patch:
inside gemma3_mm.py's forward(), right after `positions += 1`, short-circuit to the
language model whenever input_embeds is provided — otherwise gemma3_mm routes through
the multimodal embed routine, silently drops the injected embeds, and the model
free-associates ("\n\n\n…" or all-CJK).

The shipped .patch is a human-readable diff (its hunk header is prose, so `git apply`
/`patch` won't take it). This script reproduces that exact edit programmatically and
is idempotent: re-running it is a no-op once the marker is present.

    python scripts/apply_gemma_patch.py          # apply
    python scripts/apply_gemma_patch.py --check   # report status, change nothing
"""
from __future__ import annotations

import os
import sys

MARKER = "NLA: input_embeds bypass"
INSERT = (
    "        # === NLA: input_embeds bypass — text-only injection path ===\n"
    "        # When input_embeds is explicitly provided (NLA injection), go straight to\n"
    "        # the language model. gemma3_mm's general_mm_embed_routine only reads\n"
    "        # input_ids, so injected embeds are otherwise dropped -> garbage output.\n"
    "        if input_embeds is not None:\n"
    "            return self.language_model(input_ids, positions, forward_batch, input_embeds, **kwargs)\n"
)


def _target_path() -> str:
    import sglang  # noqa: PLC0415 — needs the installed package

    p = os.path.join(os.path.dirname(sglang.__file__), "srt", "models", "gemma3_mm.py")
    if not os.path.exists(p):
        sys.exit(f"could not find gemma3_mm.py at {p!r}; is sglang installed?")
    return p


def main() -> None:
    check = "--check" in sys.argv
    path = _target_path()
    src = open(path, encoding="utf-8").read()

    if MARKER in src:
        print(f"[apply_gemma_patch] already patched: {path}")
        return
    if check:
        sys.exit(f"[apply_gemma_patch] NOT patched: {path}")

    lines = src.splitlines(keepends=True)
    # The anchor is `positions += 1` inside forward(). Prefer the one tagged by the
    # upstream "1-indexed" comment if there's more than one bare match.
    hits = [i for i, ln in enumerate(lines) if ln.strip() == "positions += 1"]
    if not hits:
        sys.exit(
            "[apply_gemma_patch] anchor `positions += 1` not found in gemma3_mm.py. "
            "sglang internals changed; apply patches/nla_gemma3_mm_input_embeds.patch by hand."
        )
    if len(hits) > 1:
        tagged = [i for i in hits
                  if i > 0 and ("1-indexed" in lines[i - 1] or "cost me" in lines[i - 1])]
        hits = tagged or hits[:1]
    anchor = hits[0]

    backup = path + ".nla_bak"
    if not os.path.exists(backup):
        open(backup, "w", encoding="utf-8").write(src)
    lines.insert(anchor + 1, INSERT)
    open(path, "w", encoding="utf-8").write("".join(lines))
    print(f"[apply_gemma_patch] patched {path} (backup at {backup})")


if __name__ == "__main__":
    main()
