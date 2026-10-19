# FineWeb Terra blinded manual-audit packet

This is the only admissible FineWeb audit packet. It was generated with
`gpt-5.6-terra` through `OPENAI_API_KEY_2`; all API audit records carry text
model amendment SHA-256
`fcd5d50f6f863337d9e9009b1129354afad1ce6dbff17b0510132d0b5d423966`.

Review only these two files:

- `review/manual_segmentation_review.csv`: 100 rows. Enter `yes` or `no` in
  both review columns.
- `review/manual_decoy_review.csv`: 200 rows. Enter `pass`, `fail`, or
  `uncertain` in both review columns.

Do not open `review/manual_decoy_answer_key.jsonl` or any `review/api/` file
until both CSVs are complete. They contain the automated labels and would break
blinding.

The gate requires zero exact-tiling failures, at least 90% acceptable atomic
claim inventories, at least 90% complete claim inventories, and at least 90%
human/automated agreement for both source absence and splice naturalness.
