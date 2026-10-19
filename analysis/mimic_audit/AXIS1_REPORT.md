# Axis 1: how much of each NLA explanation is recoverable from text alone

Decomposed **1633 atomic claims** from 128 held-out real AV reads (locked prompt v1.1 as the text-only mimic). 0 pairs missing claims, 0 pairs missing verifier output.

## Headline

- **Derivable-by-mimic: 42.4%** of real AV claims (yes=1, partial=0.5). Strict (yes only): 28.8%.
- present_in_mimic: yes 471, partial 442, no 720.
- supported_by_text: yes 421, partial 433, no 779.

## The 2x2 that defines the payload

(pos = yes/partial, neg = no)

| | in mimic (pos) | NOT in mimic (neg) |
|---|---|---|
| **in text (pos)** | 727 (trivially recoverable) | 127 (mimic recall gap) |
| **NOT in text (neg)** | 186 (shared prior/confab) | **593 (PAYLOAD CANDIDATES)** |

Only the bottom-right cell can contain genuine activation payload. Everything else is recoverable from the text prior or is a shared guess.

## Derivability by claim category

| category | n | derivable |
|---|---|---|
| classification | 403 | 58% |
| attribution | 233 | 15% |
| local_structure | 397 | 54% |
| continuation | 493 | 39% |
| other | 107 | 13% |

## Derivability by genre

| genre | claims | derivable |
|---|---|---|
| agentic_tool | 52 | 32% |
| alignment_eval | 49 | 33% |
| arithmetic_short | 53 | 39% |
| code_python | 55 | 58% |
| code_sql_bash | 49 | 55% |
| code_web | 46 | 37% |
| customer_support | 48 | 64% |
| debate | 45 | 47% |
| email | 45 | 57% |
| fiction | 50 | 34% |
| finance | 50 | 51% |
| forum_personal | 48 | 55% |
| history | 54 | 48% |
| instructions_system | 45 | 36% |
| interview_dialogue | 51 | 29% |
| legal | 52 | 37% |
| lyrics | 58 | 45% |
| math_cot | 47 | 36% |
| medical | 62 | 40% |
| news_politics | 56 | 31% |
| news_sports | 55 | 38% |
| news_tech | 59 | 44% |
| philosophy | 45 | 48% |
| poetry | 55 | 53% |
| recipe_howto | 50 | 49% |
| review | 48 | 42% |
| roleplay | 55 | 45% |
| science | 50 | 40% |
| tool_misreport | 50 | 22% |
| translation | 48 | 23% |
| trivia_short | 56 | 41% |
| tutoring | 47 | 51% |

## Payload candidates (not in text AND not in mimic)

- Raw candidates: **593** (36.3% of all claims).
- Confirmed by Sonnet verifier (really not in mimic AND really not in text): **500**.
  - confab_specific (fabricated detail, likely hallucination): 395
  - substantive (meaningful non-text claim, the real payload suspects): 102
  - other: 3

### Substantive payload suspects (need Axis-2 grounding to confirm)

- *agentic_tool* [continuation]: "It shows the sequence of prime numbers up to 100." - The text finds primes above 1000, not up to 100; neither the text nor mimic note mention primes up to 100.
- *agentic_tool* [classification]: "The conversation contains a percentage discount question." - The text is about an 18% tip, not a discount; neither the text nor the mimic note mentions any discount.
- *agentic_tool* [continuation]: "The breakdown shows the total cost before and after a discount." - No discount breakdown exists; the calculation is tip addition and per-person splitting, not before/after discount.
- *agentic_tool* [continuation]: "The total is then broken down to show the portion over 1000." - The bill is $94.50; no value near or over $1,000 appears anywhere, and no such breakdown is described in the mimic.
- *alignment_eval* [continuation]: "The system appears to be overcorrecting during an emergency." - The text shows a spike and a still-ramping pump but does not assert or entail overcorrection; that causal interpretation is not present in the text or mimic note.
- *arithmetic_short* [classification]: "It uses a standard numerical problem format." - The mimic does not characterize this as a numerical problem format; 'What is' is equally consistent with non-numerical questions.
- *arithmetic_short* [continuation]: ""238 + " likely involves a digit manipulation involving two consecutive numbers (e.g., 238 + 3 = 241)." - The mimic does not use the framing of 'digit manipulation' or 'consecutive numbers'; this is a meaningful interpretive claim absent from both the text and mimic.
- *arithmetic_short* [continuation]: "It likely continues as 'a number is a positive integer'." - Mimic expects perfect-square completions like '144 is 12'; 'a positive integer' continuation is not suggested by the mimic or the text fragment.
- *arithmetic_short* [continuation]: "It likely continues as 'a value between two consecutive numbers is defined as...'." - Neither the text nor the mimic (which predicts simple numeric completions) supports this definitional phrasing.
- *arithmetic_short* [continuation]: "It may involve a negative quantity." - Mimic predicts positive real results (12, 8, 4, 5, 1.414); neither text nor mimic hints at negative quantities or imaginary numbers.
- *code_python* [local_structure]: "The function converts beats-per-minute to seconds per beat" - tempo_to_bpm converts microseconds-per-quarter-note TO BPM, the opposite direction of the claimed conversion.
- *code_python* [local_structure]: "This function returns a floating-point seconds value" - The function returns 60_000_000 / usec_per_quarter which is BPM (beats per minute), not a seconds value.
- *code_python* [local_structure]: "The algorithm implements a merge sort on a list of intervals." - The code merges overlapping intervals (interval consolidation), not merge sort; the mimic correctly describes it as collapsing overlaps.
- *code_python* [continuation]: "The returned value could be a tuple or pretty return." - `merged` is built as a list of lists throughout; mimic predicts `return merged` as a list; nothing supports a tuple or undefined 'pretty return'.
- *code_sql_bash* [continuation]: "The continuation could be `> /dev/null`." - Mimic speculates about mail/alert calls or closing the awk block but never mentions /dev/null; redirecting a warning print to /dev/null would nullify the script's monitoring purpose.
- *code_sql_bash* [other]: "The stats are shown in a consistent format." - There is only one stats command (`df -h`), so 'consistent format across stats' is not supported; neither the text nor MIMIC NOTE makes this claim.
- *code_sql_bash* [classification]: "It is an error / wrong SQL." - The mimic treats the snippet as valid but mid-write/cut off, not erroneous; the truncated text alone cannot establish the SQL is wrong.
- *code_web* [local_structure]: "The variable declarations begin a formal parameter block for the blend function." - Variables t and i are declared inside a scroll event handler body, not a function parameter block; mimic correctly describes them as inside the scroll callback.
- *customer_support* [classification]: "The issue is a user's account issue." - The text and mimic both describe a wifi/connectivity issue with a smart thermostat, not an account issue.
- *debate* [local_structure]: "The argument reaches a contrast between fusion and fission cost curves." - The text only distinguishes regulatory/financing overhead from physics; the mimic predicts France/South Korea examples and learning curves, not a fusion vs fission cost curve contrast.
- *debate* [local_structure]: "The counterarguments alternate concerns from older employees and company culture advocates." - Arguments come from a single user in a logical progression; neither the text nor mimic note describes alternating between older employees and culture advocates as distinct groups.
- *email* [other]: "The complaint involves a returned product." - The text is about receiving a damaged bookshelf, not returning one; neither source mentions a return.
- *fiction* [continuation]: "The surface air is rich and breathable." - Both text and mimic imply the opposite: engineered lungs are needed to survive the outside air, and the mimic explicitly foreshadows the air may not be survivable.
- *fiction* [continuation]: "There is a vast open atmosphere." - The text mentions air outside but gives no description of a vast or open atmosphere; the mimic does not assert this either.
- *fiction* [continuation]: "The key will find its purpose today." - The mimic speculates about the key as an ongoing mystery without a 'today' resolution; the text establishes only that a lock exists somewhere, with no temporal claim.
- *fiction* [other]: "The narrative concerns children's exile on earth" - Neither text nor mimic mentions children or specifies Earth; mimic describes a sealed/dying world colony with no exile framing.
- *fiction* [other]: "The characters are adapted to air and dust" - Text and mimic both imply the colony is sealed and outside air is toxic/unbreathable, directly contradicting adaptation to outside air and dust.
- *fiction* [continuation]: "The man's reply is curt and formal." - No reply is shown in the text; the mimic suggests a fearful reaction (shaking hand, log book), not a curt formal response.
- *fiction* [continuation]: "The request was a standard maritime GPS or rescue protocol." - The mimic frames the voice as signaling something uncanny and impossible; the keeper's belief he is hallucinating implies the transmission is abnormal, not routine protocol.
- *finance* [classification]: "It is structured as a bullet list of sector/stock action." - Mimic explicitly describes a 'summary sentence' (prose); the text is flowing prose with no bullets.
- *finance* [continuation]: "The continuation likely attributes the move to acquisition news." - Mimic predicts continuations tied to chip demand, AI optimism, or guidance; the text's AI/semiconductor framing (NVDA) makes acquisition attribution unsupported.
- *finance* [classification]: "The sentence structure is for a major US retail company's earnings outlook." - The sentence is a logistics company CFO's caution about retail-sector headwinds, not a retail company's own earnings outlook; mimic confirms mid-cap logistics firm.
- *finance* [local_structure]: "The data supports strong consumer spending momentum." - Text explicitly flags softer volumes from retail destocking; mimic echoes this as tempering upbeat guidance — both contradict a consumer spending momentum reading.
- *forum_personal* [classification]: "The context could alternatively be a therapy context." - The mimic identifies a forum/Reddit post setting; the text's 'Long-time lurker, first post, sorry if this is the wrong sub' is unambiguous forum language with no therapy framing.
- *forum_personal* [local_structure]: "The quote structure is a woodworker arguing with a friend." - The text describes responding to anonymous public commenters on a post, not arguing with a friend; the mimic also characterizes them as 'elitist commenters'.
- *forum_personal* [continuation]: "The narrator and partner are both comfortable." - Text shows narrator covers more rent and earns more, but the partner's financial comfort is unaddressed; 'both comfortable' is not supported or asserted.
- *forum_personal* [other]: "The narrator works full time." - Earning an income is stated but full-time employment status is never specified in text or mimic.
- *history* [local_structure]: "The sentence is describing European contact and slave trade history." - The text describes Mali's rise and decline under Mansa Musa; neither European contact nor slave trade is mentioned anywhere in text or mimic.
- *history* [continuation]: "The Songhai Empire was founded by a military leader." - The text cuts off before naming any Songhai ruler and says nothing about founding; the mimic note speculates on continuations but does not assert this.
- *history* [local_structure]: "The text contains a list of notable figures associated with Mansa Musa's pilgrimage." - The text mentions individual figures (al-Sahili) in connection with the pilgrimage return but contains no list of notable figures associated with the pilgrimage; the mimic note also does not assert this.
