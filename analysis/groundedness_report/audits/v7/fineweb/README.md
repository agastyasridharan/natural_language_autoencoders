ARCHIVED AND INADMISSIBLE: this packet was generated with `gpt-5.6-luna`.
Do not review, score, or use it. The replacement Terra packet is under
`../fineweb_terra/review/`; the exclusion rule is frozen in
`../text_model_correction_amendment.json`.

The historical instructions below are retained only as provenance.

These are the blinded FineWeb halves of the preregistered V7 manual gates.

Review `manual_segmentation_review.csv` without consulting any API cache. For
each of its 100 rows, enter `yes` or `no` in both review columns: whether the
proposed spans are acceptable atomic claims, and whether all semantic claims in
the explanation were captured. Notes are optional.

Review `manual_decoy_review.csv` without consulting the answer key. For each of
its 200 rows, enter `pass`, `fail`, or `uncertain` for source absence and splice
naturalness. The source-absence question is whether the candidate's specific
content is absent from the causal source; the splice question concerns grammar,
local fluency, discourse coherence, and structural well-formedness, not truth.

Do not open `.answer_keys/fineweb_manual_decoy_answer_key.jsonl` until both
review sheets are complete. Then score the FineWeb half with:

```bash
python -m analysis.groundedness_report.score_manual_audit \
  --segmentation-review analysis/groundedness_report/audits/v7/fineweb/manual_segmentation_review.csv \
  --decoy-review analysis/groundedness_report/audits/v7/fineweb/manual_decoy_review.csv \
  --decoy-answer-key analysis/groundedness_report/audits/v7/.answer_keys/fineweb_manual_decoy_answer_key.jsonl \
  --exact-tiling-failures 0 \
  --out analysis/groundedness_report/audits/v7/fineweb/manual_gate_result.json
```

The frozen FineWeb thresholds are 0.90 for atomicity, inventory completeness,
source-absence agreement, and splice-naturalness agreement, with zero exact
tiling failures. Passing this half does not authorize outcomes until the equal
OOD audit half also passes.
