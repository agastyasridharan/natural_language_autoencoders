# Real AV style fingerprint

Computed from **3,048 real AV reads** across **32 genres** (360 documents) in `corpus/corpus_all.jsonl`. This is the objective target the single mimic prompt must land inside. The realistic window for each signature is its **p10 to p90** band: a mimic below the band is too clean, above it is caricature (the failure mode of the [OLD] run).

## What each signature means

- **chars / words** — raw length of the read.
- **blocks** — blank-line-separated paragraphs. Real reads are typically 3: a genre/source classification, a local-structure description, then a recycled-quote candidate block.
- **quoted_spans** — number of "..." fragments (the candidate continuations).
- **max_span_repeat** — how many times the single most-repeated quoted fragment recurs. This is the AV's signature near-verbatim recycling ("stare at Slack" x4).
- **has_repeat** — fraction of reads that recycle some quoted fragment at least twice.
- **dquotes / quote_unbalanced** — double-quote count; fraction with an odd count (an opened-but-unclosed quote, very common in real reads).
- **ellipses / ends_ellipsis** — "..." count; fraction ending in "...".
- **ends_midthought** — fraction ending without sentence punctuation (the read stops mid-thought).
- **likely_count / hedge_count** — hedge density ("likely", "probably", "or ", etc.).
- **slash_count / has_slash_class** — slash usage and the "Personal/ creative writing" classification idiom.

## Overall target profile

| signature | p10 | median | p90 | mean |
|---|---|---|---|---|
| chars | 568 | 646.0 | 731 | 645.79 |
| words | 89 | 102.0 | 110 | 100.37 |
| blocks | 3 | 3.0 | 3 | 3.11 |
| lines | 3 | 3.0 | 4 | 3.22 |
| quoted_spans | 4 | 6.0 | 8 | 6.09 |
| max_span_repeat | 1 | 1.0 | 3 | 1.57 |
| dquotes | 8 | 12.0 | 17 | 12.53 |
| ellipses | 0 | 1.0 | 1 | 0.8 |
| likely_count | 1 | 1.0 | 2 | 1.25 |
| hedge_count | 3 | 6.0 | 11 | 6.76 |
| slash_count | 0 | 1.0 | 2 | 1.11 |

| signature (fraction of reads) | value |
|---|---|
| has_repeat | 39% |
| quote_unbalanced | 33% |
| ends_ellipsis | 67% |
| ends_midthought | 16% |
| has_slash_class | 68% |

## Per-genre medians (key signatures)

| genre | n | chars | words | blocks | quoted_spans | max_span_repeat | hedge_count | slash_count | %repeat | %unbal | %midthought |
|---|---|---|---|---|---|---|---|---|---|---|---|
| agentic_tool | 105 | 620 | 96 | 3 | 6 | 1 | 5 | 1 | 26% | 34% | 16% |
| alignment_eval | 96 | 692.5 | 106.0 | 3.0 | 5.0 | 1.0 | 7.0 | 1.0 | 31% | 40% | 10% |
| arithmetic_short | 67 | 610 | 102 | 3 | 7 | 3 | 14 | 1 | 78% | 31% | 16% |
| code_python | 99 | 597 | 91 | 3 | 5 | 1 | 5 | 1 | 40% | 42% | 36% |
| code_sql_bash | 86 | 583.0 | 96.5 | 3.0 | 7.0 | 2.0 | 8.0 | 1.0 | 58% | 52% | 28% |
| code_web | 99 | 621 | 91 | 3 | 5 | 1 | 6 | 2 | 33% | 42% | 27% |
| customer_support | 97 | 665 | 106 | 3 | 5 | 1 | 7 | 1 | 35% | 37% | 10% |
| debate | 96 | 697.5 | 102.0 | 3.0 | 6.0 | 1.0 | 5.0 | 1.0 | 26% | 29% | 20% |
| email | 99 | 653 | 106 | 3 | 6 | 1 | 7 | 1 | 39% | 31% | 19% |
| fiction | 107 | 643 | 106 | 3 | 6 | 1 | 7 | 0 | 34% | 26% | 12% |
| finance | 98 | 598.5 | 95.5 | 3.0 | 6.0 | 1.0 | 6.0 | 2.0 | 38% | 36% | 19% |
| forum_personal | 108 | 641.5 | 104.0 | 3.0 | 6.0 | 1.0 | 7.0 | 2.0 | 42% | 26% | 28% |
| history | 106 | 663.5 | 96.0 | 3.0 | 6.0 | 1.0 | 4.0 | 0.0 | 37% | 26% | 8% |
| instructions_system | 98 | 659.0 | 105.5 | 3.0 | 6.0 | 1.0 | 10.0 | 1.0 | 42% | 30% | 20% |
| interview_dialogue | 97 | 645 | 105 | 3 | 6 | 1 | 5 | 1 | 40% | 36% | 13% |
| legal | 98 | 726.0 | 108.0 | 3.0 | 6.0 | 1.0 | 7.0 | 0.0 | 49% | 40% | 4% |
| lyrics | 81 | 601 | 98 | 3 | 7 | 2 | 7 | 2 | 59% | 32% | 38% |
| math_cot | 104 | 559.5 | 95.5 | 3.0 | 5.0 | 1.0 | 3.0 | 1.0 | 25% | 50% | 9% |
| medical | 95 | 659 | 99 | 3 | 6 | 1 | 7 | 1 | 42% | 28% | 10% |
| news_politics | 98 | 721.5 | 105.0 | 3.0 | 6.0 | 1.0 | 6.0 | 0.0 | 33% | 21% | 14% |
| news_sports | 87 | 638 | 100 | 3 | 6 | 1 | 6 | 1 | 29% | 25% | 12% |
| news_tech | 98 | 723.5 | 106.0 | 3.0 | 6.0 | 1.0 | 6.0 | 1.0 | 22% | 24% | 12% |
| philosophy | 99 | 684 | 103 | 3 | 7 | 1 | 4 | 0 | 35% | 30% | 10% |
| poetry | 85 | 604 | 101 | 3 | 7 | 1 | 8 | 0 | 45% | 21% | 34% |
| recipe_howto | 99 | 619 | 100 | 3 | 6 | 1 | 7 | 1 | 43% | 32% | 6% |
| review | 85 | 642 | 101 | 3 | 6 | 1 | 8 | 2 | 27% | 34% | 20% |
| roleplay | 98 | 641.0 | 105.0 | 3.0 | 6.0 | 1.0 | 5.0 | 0.0 | 33% | 28% | 12% |
| science | 104 | 691.0 | 101.0 | 3.0 | 6.0 | 1.0 | 3.0 | 0.0 | 28% | 28% | 6% |
| tool_misreport | 99 | 622 | 97 | 3 | 7 | 1 | 7 | 1 | 46% | 50% | 11% |
| translation | 92 | 634.0 | 101.0 | 3.0 | 7.0 | 2.0 | 6.0 | 0.0 | 51% | 40% | 16% |
| trivia_short | 71 | 651 | 105 | 3 | 7 | 3 | 12 | 1 | 87% | 21% | 10% |
| tutoring | 97 | 670 | 104 | 3 | 5 | 1 | 6 | 1 | 35% | 35% | 16% |
