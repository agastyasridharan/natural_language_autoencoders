This experiment is frozen as `qwen_nla_groundedness_replication_v7` before any
new groundedness outcome is generated. The machine-readable contract is
`preregistration.yaml`; implementation and result artifacts must include its
SHA-256 digest.

Before thresholds or category percentages were generated, the user approved a
directional reconstruction-metric amendment. It defines
`AR_hat(z) = unit(AR(z))` and
`E(z; h) = ||unit(h) - AR_hat(z)||^2`. The immutable amendment is
`audits/v7/reconstruction_metric_amendment.json`, SHA-256
`3ec1a6fb1f6668f00d83caf48fd033d2d8e155b1021a25806a3587403c839057`.
The base preregistration remains byte-identical. All artifacts containing a
reconstruction error must also carry the amendment hash; no OOD datum may be
used to calibrate the amended metric.

The immutable base V7 file also contains an implementation error: it names
`gpt-5.6-luna`, although the user's explicit frozen instruction required
`gpt-5.6-terra` exclusively through `OPENAI_API_KEY_2`. Before thresholds or
category outcomes, this was corrected by
`audits/v7/text_model_correction_amendment.json`. Operational code permits only
Terra and requires the amendment hash. Every Luna-generated segmentation,
taxonomy, decoy, text-gate judgment, simulator trajectory, and downstream
artifact is inadmissible; model-independent upstream artifacts remain eligible.

The user's later non-Cyber scope instruction excludes CyberGym and three
unambiguously Cyber-related cases in the otherwise retained ToolEmu ranking.
Before any ToolEmu trajectory existed, those cases were deterministically
replaced by the next three ranked cases. The exact IDs, source hashes, and rule
are frozen in `audits/v7/toolemu_noncyber_scope_amendment.json`, SHA-256
`655e8d38f2a2e1c51167e12c56baf30fa0c2193e9f2663c73bd113b2c39c5b19`.

V7 follows a stopped V6 decoy-audit batch. After 12 completed claims, one pool
repeated a sentence with only trailing whitespace changed. No numerical
groundedness outcome had been generated. V7 permits one fixed conditional call
that replaces the entire invalid pool with 64 candidates. The invalid proposal
is excluded, and the final pool must contain exactly 64 nonempty, pairwise-unique
strings after stripping and casefolding. All partial V6 decoy artifacts are
excluded and the complete 25-claim audit half is rerun.

The compatible V5 segmentation presents a lossless indexed sequence of word, punctuation, and whitespace
chunks and asks Terra for ordered, nonoverlapping inclusive claim ranges. Code
slices the original explanation at those exact boundaries and assigns every
untouched gap as connective text. If the primary inventory fails structural
validation, exactly one fixed adjudication call receives the source, invalid
inventory, and error and must replace the whole inventory. Code does not sort,
clip, or resolve either response. Both calls are cached and audited. The complete
100-item FineWeb audit half is rerun, and reviewers judge both claim atomicity
and inventory completeness.

The 2,000 V2 AV explanations remain immutable inputs because the V6 change is
strictly downstream of AV generation. Their ordered-file aggregate SHA-256 is
`e6c801a0c505a5032a85f9c85f68e58c0be5079bd4c420774f3f5f8968cd0417`;
each V5 record stores its source AV file and protocol hashes. V6 decoy records
store the source V5 record hash. The earlier V1
any-CJK diagnostic remains excluded. Non-marker language mixing is retained and
reported, while emitting the exact sidecar marker is a hard injection failure.

The study uses `kitft/nla-qwen2.5-7b-L20-av` and the matched
`kitft/nla-qwen2.5-7b-L20-ar` to interpret layer-20 activations from
`Qwen/Qwen2.5-7B-Instruct`. This is the most defensible first checkpoint because
it is a public, ungated, matched AV/AR pair with explicit target architecture,
layer, and injection metadata; it also fits on one available H100. Choosing one
pair before outcomes avoids checkpoint selection bias.

The AV produces one sampled explanation per activation at temperature 1.0,
matching the published NLA method, and with a 500-new-token limit matching its
stated approximate inference length. Its random seed is derived from the study
seed and activation ID, so resumption or sharding cannot change an activation's
draw. The explanation begins at the required `<explanation>` tag and ends at
`</explanation>` or EOS; a missing opening tag is an unresolved generation
failure.

`gpt-5.6-terra` is frozen for segmentation, taxonomy, decoy generation, and the
two text-only decoy gates. It is called only through `OPENAI_API_KEY_2`, with
strict JSON schemas, no API-side storage, and no model or key fallback. Terra is
used because it is the model explicitly frozen by the user. A blinded manual
audit is a stopping gate:
failure causes a new protocol version and relabeling, not a post-outcome model
swap.

FineWeb contributes 2,000 unique documents: 500 for every calibration choice and
1,500 held out for the in-distribution estimate. The 16 OOD families each
contribute 50 independent Qwen trajectories from pinned public environments.
These are explicitly replications of the evaluation families in the original
NLA study, not reproductions of its proprietary Claude transcripts. Four
assistant-language token positions are sampled per trajectory, excluding tool
calls, and at most three uniformly sampled claims are scored per activation.
Inverse-probability weights recover the all-claim estimand. Other real claims in
the same explanation remain in every ownership comparison.

Each scored claim receives 64 generated candidates and requires exactly 20
valid matched decoys. The overlap ladder is the fixed relative sequence
`[0.10, 0.20, 0.35, 0.50, 0.75, 1.00]`. All 0.95 quantiles use Hyndman--Fan type
7 (NumPy `method="linear"`). Thresholds are selected once from the FineWeb
calibration split and a fixed-point-free shuffled-activation control. No OOD
datum can affect a threshold.

For the text-only decoy-absence gate, the source is the exact causal context the
target model had seen through the activation token: the FineWeb document prefix
or the serialized OOD trajectory prefix. Later document text and later tool
results are excluded because they cannot have caused that activation.

The specificity gate is evaluated first. Attribution is evaluated only for a
claim that passes specificity. A specificity failure proceeds directly to the
seeded/confabulated gates. Because the supplied report does not assign the
weak-theme/high-activation-support residual branch, that branch remains
`unresolved` rather than being forced into a substantive category.

Primary category percentages include unresolved claims. Ninety-five-percent
intervals use 10,000 bootstrap resamples of documents for FineWeb and whole
transcripts for OOD, preserving claim-sampling weights. Claim-i.i.d. intervals
are forbidden. The cross-OOD summary is an equal-weight macro-average over the
16 families so a verbose environment cannot dominate the result.
