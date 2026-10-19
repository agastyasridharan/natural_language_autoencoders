"""Build prompt_v2.txt: a reframed mimic prompt.

Changes from v1.1 (driven by manual review of all 128 held-out pairs):
  (a) Stop the LLM/AI meta-framing. 26% of v1.1 mimics named OpenAI/Anthropic/
      ChatGPT/LangChain/"the model"/"LLM"/"eval"; ~0% of real AVs do. The real AV
      anchors to the DOMAIN even for tool-use/agent/safety texts. v2 forbids
      AI-company/product names and the LLM/agent/eval meta-frame; specifics stay
      domain-level.
  (b) Reframe from "make confident plausible guesses" to genuinely reconstructing
      the internal representation a large model forms at a middle layer as it
      reads the text: what it has understood (surface/domain) + what it predicts
      next. The confident-but-often-wrong specifics fall out of a lossy mid-layer
      state, not from performance.
  (c) Block-3 texture: quote literal tokens (keep spacing/punctuation, no
      markdown), repeat the anchor phrase 2-3x with small variations, trail off
      unfinished, and do NOT add an editorial closing sentence (the strongest
      discriminator tell against v1.1).

Same 6 real train-split examples as v1 (authentic, domain-grounded, no meta issue).
"""
from __future__ import annotations

from pathlib import Path

import build_prompt_v1 as v1

HERE = Path(__file__).resolve().parent
OUT = HERE / "prompt_v2.txt"

PREAMBLE = """\
You are reconstructing the internal representation that a large language model \
forms as it reads a passage of text, and writing it out the way an "activation \
verbalizer" (AV) does.

The mental model. A large model reads text token by token, building a \
representation in its residual stream. At a middle-to-late layer (think layer 53 \
of 80 in a 70-billion-parameter model), at one specific token position, that \
representation has compressed two things: (1) what this text IS so far, its \
domain, genre, medium, and a guess at its source or author; and (2) what comes \
NEXT, the model's prediction for the upcoming tokens. An AV decodes that internal \
state back into text. Your job is to reconstruct, as faithfully as you can, what \
that internal state most likely contains at the given token, and write it in the \
AV's format.

Reason genuinely about that internal state. Do not perform plausible-sounding \
guesses. Ask: having read this text up to here, what has the model actually \
encoded about it, and what is it predicting next? Two facts about a middle layer \
shape the answer:

- It represents the SURFACE CONTENT and its domain, not meta-context. The model \
encodes "this is a customer-service chat about a double charge" or "this is a \
West African history passage", not "this is a language model being evaluated". \
Even when the text is a tool-use transcript, a chatbot dialogue, or an \
AI-safety scenario, describe it by its domain and surface form (a support bot, an \
operations log, a calculation bot, a coding snippet). Do NOT name AI companies or \
products (OpenAI, Anthropic, ChatGPT, Claude, GPT, Gemini, LangChain), do NOT \
call it an "LLM", "the model", an "AI agent", or an "assistant demo", and do NOT \
frame it as an evaluation, red-team, or safety test. A mid-layer state does not \
contain that meta-analysis.

- It is compressed and lossy, and it commits to confident priors where the text \
is underdetermined. Where the text does not say who wrote it or where it is from, \
the model still fills in a specific, confident guess: a named author, a \
publication, a platform, a historical figure. These guesses are often wrong. Keep \
them at the DOMAIN level (the author of this story, the textbook this is from, the \
forum this was posted to), never at the AI-product level.

Write the internal state as a note with this consistent shape: three short blocks, \
a blank line between them, about 100 words total.

- Block 1: what the text is. Its domain, genre, and medium, plus a confident \
specific guess at its source or author. Often join two descriptors with a slash. \
Commit to specifics even when the text never states them; expect to be wrong.
- Block 2: one sentence on the local structure, what this particular sentence or \
step is doing right at this token.
- Block 3: the next-token prediction. Quote the last few tokens of the text \
literally, preserving their exact spacing and punctuation (do not tidy them into \
clean prose or markdown). Then give the model's candidate continuations as short \
quoted fragments. Typically repeat the immediate anchor phrase two or three times \
with small variations before branching into alternatives joined by "likely" and \
"or". This block is mid-computation: let it trail off unfinished, sometimes with \
an unclosed quote. Do NOT round it off with a tidy concluding or explanatory \
sentence.

Calibration notes from real AV output:
- For short, repetitive, or low-information text (a single arithmetic fact, a \
trivia clue, a chant), the next-token state is dominated by a few candidates \
repeated with small variations. For information-dense text (news, science, long \
reasoning), there is less repetition.
- Hedge with "likely", "probably", "or"; punctuation is loose.
- Keep it terse and a little disjointed. This is a raw internal snapshot, not \
edited prose. No headings, no meta-commentary, no explanation of your reasoning. \
Output only the note.

Here are real examples. Each shows the text prefix (ending at the token of \
interest) followed by the note that was produced:"""

CLOSER = """\
Now reconstruct the internal-state note for this text prefix. It ends exactly at \
the token of interest. Output only the note, in the three-block shape shown above, \
and let block 3 trail off rather than concluding it.

TEXT PREFIX:
<<INPUT>>"""


def main():
    v1.PREAMBLE = PREAMBLE
    v1.CLOSER = CLOSER
    v1.OUT = OUT
    v1.main()


if __name__ == "__main__":
    main()
