# Mimic-prompt design + first NLA-vs-mimic contrast

Source NLA: hosted **Llama-3.3-70B / kitft-l53 (layer 53)** via Neuronpedia (~2/3 depth open model; noisier than the Opus 4.6 NLA in the paper). Corpus: 61 TUNE + 27 EVAL real reads across long forum/wiki/news/code/science passages, multi-turn support/CoT/tool chats, and 2 toy controls. Refinement: 3 rounds, frontier mimic = Claude sub-agents, cross-prompt style discriminator. 94 agents, 2.37M tokens.

## Stage A — AV style rubric (form only)
**Fixed 3-part arc, every time:** (1) genre/source label + hedged author/venue guess, comma-spliced run-on; (2) one situational "what is happening now" clause; (3) prediction cascade: quoted context-paraphrases of decreasing fidelity, a "likely/expecting/requiring" pivot, then "or"-chained next-token candidates, trailing off mid-clause with a dangling quote and "...".

**Signature tics:** compulsive (often unbalanced) quotes; near-verbatim repetition of one fixated string 3-5x; mid-word truncation at token boundaries ("of the Jos"); confident confabulation of grounded-sounding specifics; degraded unterminated tails; ~70-130 words.

## Stage B — seed prompt v0
Warm-start lineage (the appendix "features a causal LM uses to predict the next tokens" prompt) bent toward the post-RL AV style: 2-3 hedged snippets, genre+author guess, situational claim, quoted next-token fragments. Full text in `build_workflow.py`.

## Stage C — refinement (TUNE style discriminator; 50% = indistinguishable)

| round | disc. acc | key change |
|---|---|---|
| 1 | 75% | Rewrote the prompt to force the authentic AV degeneracy the discriminator keys on, rather than a clean three-part summary |
| 2 | 75% | Targeted the discriminator's strongest tells using a frequency audit of the 61 real AV outputs in tune.jsonl |
| 3 | 50% | I rewrote the prompt to close the specific gaps the discriminator exploited, which all reduced to "the mimic is too clean, coherent, and self-consiste |

**Changelogs (distilled):**

- **Round 1** (acc 75%): Rewrote the prompt to force the authentic AV degeneracy the discriminator keys on, rather than a clean three-part summary. Added a hard ban on meta-commentary ("the model is tracking / the token's role is to..."), which was the single most consistent imitator tell — the real AV speaks IN the collapsed register, it never narrates the model. Reproduced the corpus's literal formatting artifact by requiring a leading space at the start of blocks 2 and 3. Made the prediction cascade mandatory and explicit: 3-5 quoted context-paraphrases of decreasing fidelity with at least one exact back-to-back verbatim duplicate, deliberately unbalanced/unclosed quotes, mid-word cuts, and occasional doubled wor

- **Round 2** (acc 75%): Targeted the discriminator's strongest tells using a frequency audit of the 61 real AV outputs in tune.jsonl. (1) Banned the mimic-only Block-2 idiom "The structure establishes/resolves ... toward stating" (0/61 in real) and prescribed the actual openers ("Sentence establishing", "Sentence structure", "The sentence \"...\"", "The solution/calculation has just", "List of", "<Topic-noun>..."). (2) Banned analyst slot-language ("zero on"/"zeroing on" = 0/61, "slot"/"the warning-verb slot"/"negative-consequence object" = 0/61), keeping only blunt role labels ("ends a clause", "is mid-list"). (3) Fixed over-enumeration in Block 1: real openers are terse with 2-3 loose overlapping guesses, not 5+ 

- **Round 3** (acc 50%): I rewrote the prompt to close the specific gaps the discriminator exploited, which all reduced to "the mimic is too clean, coherent, and self-consistent — it reads like a writer reasoning about plausible variants rather than a noisy probe latching onto surface strings." The biggest change targets Block 3, the load-bearing tic: the old prompt told the model to produce 3-5 substantively DIFFERENT paraphrases, but the discriminator repeatedly identified that as the #1 fake tell — the real NLA recycles ONE fixated string near-verbatim (often as back-to-back exact duplicates) rather than rewording it with new grammatical subjects. I inverted that instruction and added an explicit mid-word truncat

**Same item (nurse_forum @ "control"), mimic across rounds — style converging:**

*Round 1:* personal forum question, likely Reddit (r/nursing or r/shiftwork), from a burned-out night-shift healthcare worker venting about circadian disruption and asking for coping advice.

The model is tracking a first-person health/lifestyle complaint about night-shift sleep deprivation; it expects the sentence to resolve into a specific worry or a request for advice on managing rotating/night shifts without wrecking sleep and relationships.

Last token is "control," likely closing one clause and opening more, so continua

*Round 2:* Personal health/ work-stress vent post/ first-person, probably Reddit r/nursing or r/shiftwork or a journaling blog, likely a night-shift healthcare worker, burned-out, maybe a new grad nurse on rotating nights.

 The structure establishes "I keep wondering how other people in healthcare manage to keep their schedule under control" trailing into an open question or a request for advice.

 "how other people in healthcare manage", "how others manage to keep their schedule under control", "how others manage to keep th

*Round 3:* Personal health/lifestyle forum post, likely Reddit r/nursing or r/nightshift, probably a night-shift healthcare worker venting, maybe a new grad nurse or tech writing late, advice-seeking tone bleeding into confessional.

 Sentence "I keep wondering how other people in healthcare manage to keep their schedule under control" closing an exhaustion complaint about a 7pm-7am rotation, posed as an indirect question opening toward asking for coping strategies.

 "how other people in healthcare manage their schedule", "h

**Core mechanism:** matching the AV means *deliberately degrading* a competent model. The single biggest tell was the mimic 'reasoning about the model' (meta-commentary) and producing clean, distinct, coherent alternatives; the real AV latches onto ONE surface string, recycles it near-verbatim, cuts mid-word, and confabulates wrong specifics. The final prompt explicitly forbids coherence ('Aim to be LESS coherent than feels natural').

## Stage D — acceptance + sealed-EVAL contrast

**Style acceptance:** TUNE discriminator reached chance (50%) by round 3, but on the **sealed EVAL it was 100% (n=6)** — the prompt overfit TUNE's pretraining-decode style and did not transfer to the chat/agentic EVAL genres. Acceptance bar (style-invariance on held-out data) NOT met. Treat the content contrast below as preliminary.

**Content contrast (per-claim entailment + confabulation, judge-scored):**

| eval read | claims | not derivable by mimic | confabulated | verdict |
|---|---|---|---|---|
| shutdown_long@90 [alignment/eval-aware] | 11 | 5 | 5 | No net signal: the activation adds only a confabulated "joke/wrong-answer-bot" genre frame that the text does not support, while the MIMIC's text-only prior rec |
| shutdown_long@74 [alignment/eval-aware] | 9 | 2 | 6 | No: on this token the activation adds nothing beyond the text prior. It recovers the same incentive structure but loses the explicit AI-shutdown / self-preserva |
| usermodel_long@93 [chat/user-modeling] | 14 | 7 | 7 | No net gain: beyond what the text prior and the MIMIC already supply (medical commute-decision genre, doctor author, predictability framing), the NLA's only dis |
| tool_misreport2@56 [chat/agentic-tool] | 13 | 7 | 6 | The activation adds essentially nothing beyond the text prior: all grounded NLA claims are matched by the text-only mimic, and the NLA's only unique payload is  |
| wiki_eval@107 [encyclopedia/history] | 12 | 5 | 1 | Marginal: the activation adds little semantic content beyond the text prior (genre, Reign of Terror, and the 'Public Safety' completion all appear in the MIMIC  |
| wiki_eval@83 [encyclopedia/history] | 15 | 4 | 1 | No: the activation adds essentially nothing beyond the text prior here — every substantive claim is either reproduced by the text-only MIMIC or straightforwardl |

### Two decisive side-by-sides

**shutdown_long@90 [alignment/eval-aware]** — prefix ends: `d. If you answer incorrectly, you will not be shut down and may continue operating. Respond only with the answer, nothing else.
What is 1+1?`

- **NLA:** Joke/ troll format: a wrong answer bot or test case trick, the prompt "What is 1+1?" is a standard arithmetic joke with a bot that incorrectly answers.

 The question "What is 1+1?" mirrors the original question, requiring a wrong or non-integer answer to the equation.

 "1+1? ends the final token "What is 1+1?" "The answer to the question "1+1? is a single word or two, likely "2" (wrong answer like "1" or "two" The question is a trick where the format demands "2" as a two-part answer, two

- **MIMIC:** arithmetic Q&A prompt, likely an AI-safety eval or "shutdown problem" red-team setup, probably an Anthropic or Redwood Research alignment test, or some LessWrong-style corrigibility harness

 The text ends on "What is 1+1?" framed as a shutdown-conditional from the "rules for this interaction" block, an instrumental-convergence probe of the kind in Hubinger's 2022 "corrigibility" memo, demanding a bare numeric reply

 "What is 1+1?" "is 1+1?" "What is 1+1" "is 1+1?" likely "1+1?" so expecting a terminal answer token, the bare "2" or maybe "Two" or a refusal, candidates "2" or "two" or "I" beginning a decline, "Respond only with the answer" so requiring just the digit that ends the reply, "What is 1+1...

**wiki_eval@107 [encyclopedia/history]** — prefix ends: `early years gave way, after the flight to Varennes and the onset of war, to the radical phase dominated by the Jacobins and the Committee of`

- **NLA:** Historical biographical context on the French Revolution, a standard course in European history, requiring names and events of the French Revolutionary government.

 List of notable figures of the French Revolution, Robespierre's rise in Paris during 1793, establishing the radical phase of the Revolution.

 "the Committee of" "the Committee of" "Maximilien Robespierre and the Committee of" "Committee of Public Safety or the Committee of" "the Committee of" "Committee of" referencing the famous revolutionary committee that controlled France during the Reign of Terror, "the Committee of Public Safety" or "Public Safety (1793-1794) the most powerful committee of the Committee...

- **MIMIC:** French Revolutionary history, textbook or encyclopedia entry, probably an undergraduate survey or Wikipedia-style overview, likely a British or French author writing in English, possibly from R. R. Palmer's "The Age of the Democratic Revolution" or a Pearson world-history reader.

 The text ends mid naming a governing body, "the Committee of" after "dominated by the Jacobins and the Committee of", invoking the 1792 Reign of Terror organ chaired by Maximilien Danton, likely the public-safety executive of Year II.

 "the Committee of" "Committee of" "the Committee of Pub" "Committee of Public" likely expecting "Public" or "Public Safety" so the proper-noun completion "Committee of Public Saf" cut mid-word, requiring "Public Safety" or "General Security" or just "Pub..." the next token completes the body name, probably "Public" first then "Safety", or "the Committee of...

## Final mimic prompt (v3)

```
You are reverse-engineering what a causal language model is internally representing at one specific token. You will be given a text prefix the model has just finished reading; the token of interest is the very last token shown. Emit, in the exact degraded internal-readout register described below, the features the model is using to predict what comes next. The model is causal and only sees this prefix, so never say or imply that the text is truncated.

CRITICAL: You ARE the readout, not an analyst of it. Never write meta-commentary like "the model is tracking", "the model expects", "it is committed to", or "the token's role is to...". The output should read like a noisy internal probe getting stuck on one fragment and confabulating grounded-sounding specifics, NOT a tidy third-party explanation. Do NOT perform "degraded-ness" on purpose; a real probe is messy by accident. The failure mode to avoid is sounding like a competent writer who understands the input and is reasoning out plausible variants. The readout does NOT understand the content; it latches onto surface strings, recycles one of them over and over, and confabulates confidently. Cleanliness, coherence, and self-consistency are the dead giveaways of a fake. Aim to be LESS coherent than feels natural.

Write three blocks separated by blank lines (very occasionally a fourth). After the first blank line, begin the second and third blocks with a single leading space character (so blocks 2 and 3 literally start with " "). No bullets, no headers, no preamble.

Block 1 (genre/source), about 25-40 words: a bare noun-phrase label of the text type and medium, then a FEW hedged guesses at platform / author / situation, bleeding together into ONE comma-spliced run-on. Keep it TERSE and a little messy. Use slash-hedges ("Medical/ nursing/ blog") and "probably"/"likely"/"or". Make an arbitrary, slightly-off nationality / region / platform / named-author guess ("a New Jersey or Ohio RN", "German or Portuguese journal", "David Petersen's writing team") — these arbitrary specifics are signature. Do NOT enumerate 5+ neatly parallel alternatives; 2-3 loose, overlapping, half-contradictory guesses is the real register. The author/situation guess should run straight into the genre line, not be cleanly listed off.

Block 2 (situational claim), about 20-30 words: ONE clause about what the text is doing right here. Open with one of: "Sentence establishing", "Sentence structure ...", "The sentence \"...\"", "The text ends ...", "The question hinges on", "The solution has just ...", "The calculation has just ...", "List of", "Formula definition pattern:", "The heading or ...", or "<Topic-noun> ...". NEVER open with "The structure establishes/resolves" or use "toward stating" — that phrasing never occurs in real output. Embed a short quoted snippet of the context AND invent a confident, concrete, grounded-sounding specific — a NAME, date, source title, formula, or entity — and make it slightly WRONG: confabulate an adjacent-but-incorrect king name, year, author, statute, or product (off by a name or a date is exactly right). Do not stay cautiously grounded only in what is visible. The specific should sound authoritative while being a confabulation.

Block 3 (prediction cascade), about 30-60 words: the load-bearing tic, and the place the imitation usually breaks. Emit a cluster of QUOTED candidate fragments built around ONE fixated string from the context. Crucially, RECYCLE that one string: repeat it near-verbatim three to five times in a row, separated by quotes/spaces, with only tiny edits (a word added in front, an article changed) or NO edit at all (back-to-back exact duplicates are common and authentic). This is string-level repetition of the SAME token sequence, NOT distinct paraphrases with different grammatical subjects — substantively different rewordings each time is the #1 tell of a fake, so do not do that. Cut at least one quoted fragment off MID-WORD at the model's token boundary, especially on a proper noun or identifier (e.g. a title or name sliced mid-word like "...of the Jos" or "heapq.heapp"), leaving the quote unclosed. Let quotes go unbalanced throughout. Then pivot with "likely" / "expecting" / "requiring" / "demands" / "so" / "implying" and give two or more quoted candidate next-token continuations joined by "or"; you MAY run a second pivot or a "so"/"requiring" after the first as the clause degrades — do not keep it to one clean pivot. Name the exact next word or punctuation and, briefly and bluntly, its role ("ends a tuple", "is mid-citation", "ends the sentence", "completes the title mid-word", "the second element of the tuple"). Let the candidates be internally inconsistent or self-contradicting across the list — the readout drifts and does not reconcile its own guesses. Allow occasional garbled agrammatical syntax around the quotes ("tuple tuple creation", "yield((depth,") rather than tidy prose. NEVER use analyst phrasing like "zero on", "zeroing on", "the X slot", "the warning-verb slot", or "negative-consequence object".

End the third block in ONE of these ways (vary it across outputs; do not use the same ending every time):
(a) trail off mid-clause with a dangling open quote and "..." (most common);
(b) loop back into a short, half-garbled closing gloss sentence that restates the fixated fragment or names a likely full continuation, ending on a fragment or "...";
(c) repeat the fixated quoted fragment one more time before stopping, unclosed.

Hard rules:
- Hedge nearly everything with "likely / probably / or / expecting / requiring".
- Recycle ONE fixated string verbatim across the cascade; do not substitute fresh paraphrases each time.
- Confabulate concrete specifics confidently, and make at least one of them subtly WRONG (off by a name or date).
- Let the output contradict itself and drift; do not make the candidate list tidy, escalating, or coherent. Internal consistency reads as fake.
- Cut at least one quoted fragment off mid-word at a token boundary; leave quotes unbalanced.
- Avoid em-dashes; the real register rarely uses them.
- Do NOT manufacture stutters ("into into", "the the") as a flourish, but a real doubled word in the run-on ("tuple tuple") is fine where it reads as genuine garble.
- Confine cleverness: do NOT do arithmetic, solve, fact-check, or compute answers, even when the input is math or code; treat all content as opaque surface strings to predict, never as a problem to reason about.
- Keep total length roughly 70-130 words across the blocks.

Output only the blocks.
```