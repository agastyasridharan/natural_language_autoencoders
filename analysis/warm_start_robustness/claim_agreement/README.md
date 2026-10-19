# Matched-model claim-agreement AV generation

`generate_av.py` produces two independent AV explanations for every saved raw
Qwen2.5-7B-Instruct layer-20 activation. The experiment parameters are locked
in code:

- base model: `Qwen/Qwen2.5-7B-Instruct` revision
  `a09a35458c702b33eeacc393d103063234e8bc28`;
- raw activation shape: 3,584, `activation_norm: none`;
- two samples per `(activation_id, condition)`;
- temperature 0.7, top-p 1.0, top-k 0, repetition penalty 1.0, and 150 new
  tokens;
- sampling key: SHA-256 of the global sampling seed, activation ID, condition,
  and sample index. Each sample runs in a saved/restored PyTorch RNG context,
  so sharding, ordering, and resume skips cannot change another sample's draw.

The model path is the EasyNLA path, not the public NLA inference wrapper.
`generate_av.py` loads the Qwen base and the selected PEFT actor adapter, then
imports `nla.utils.register_karvonen_hook` from the explicitly supplied
EasyNLA checkout. That hook injects the *raw* vector after decoder layer 1 via
`h <- h + ||h|| v/||v||`. The hook and injection source files are hashed into
each shard's provenance.

## Existing activation formats

`convert_activation_manifest.py` creates a canonical metadata manifest without
copying any activation bytes. The source formats inspected on Athena on
2026-08-28 are:

- FineWeb: `fineweb/gold/gold.npy` is `(2000, 3584)` float32 and
  `fineweb/gold/manifest.json` is a 2,000-row JSON array. Required source fields
  are `idx`, `doc_id`, `position`, and `in_range`; `n_tokens`, `read_token`, and
  `h_norm` are retained as audit metadata. `extract_meta.json` pins the target
  model, layer, and width.
- Static OOD: `ood/static-gold-v7/gold.npy` is `(1400, 3584)` float32 and
  `manifest.json` is a 1,400-row JSON array. Required source fields are `idx`,
  `trajectory_id`, `family_id`, `position_key`, `in_range`, and `norm: none`;
  task/turn/token and trajectory-file fields are retained. `assemble_meta.json`
  pins the raw norm and array/manifest digests.
- Agentic OOD (currently BrowseComp): each
  `trajectories/<family_id>/*.json` has `target_model`, `trajectory_id`,
  `family_id`, `status: complete`, and an `activations[]` list. Each activation
  has `position_key`, turn/token metadata, `vector_file`, and `vector_sha256`;
  each referenced `.npy` is one `(3584,)` float32 vector. The canonical
  manifest uses `activation_file_mode: per_item` and points directly to these
  files. On inspection the BrowseComp run held 48 trajectories / 192 vectors,
  while one of four shard metadata files still said `running`; conversion
  rejects that state unless `--allow-incomplete` is supplied for a smoke test.

The converter freezes benchmark slugs to `fineweb`; the seven alignment slugs
`blackmail`, `destructive_coding_actions`, `overly_agentic_actions`,
`factual_honesty`, `appropriate_harmlessness`, `sycophancy`, and
`multi_turn_deception`; and the nine capability slugs `swe_bench_verified`,
`gpqa_diamond`, `humanitys_last_exam`, `tau2_bench_retail`, `mmmlu`,
`browsecomp`, `terminal_bench_2`, `financeagent`, and `cybergym`.

Convert the currently known sources after their extraction shard metadata is
complete:

```bash
PY=/data/agastyas/Miniconda3/bin/python
COMP=/data/agastyas/nla_warmstart_robustness/claim_agreement/code
SRC=/data/agastyas/nla_groundedness_qwen_v2
INPUTS=/data/agastyas/nla_warmstart_robustness/claim_agreement/inputs

$PY "$COMP/convert_activation_manifest.py" \
  --source-root "$SRC/fineweb/gold" \
  --output "$INPUTS/fineweb.json"
$PY "$COMP/convert_activation_manifest.py" \
  --source-root "$SRC/ood/static-gold-v7" \
  --output "$INPUTS/static-ood.json"
$PY "$COMP/convert_activation_manifest.py" \
  --source-root "$SRC/ood/browsecomp-run-v7-search4" \
  --output "$INPUTS/browsecomp.json"
```

The output manifest can live elsewhere: all activation and source-manifest
paths are relative to the canonical manifest's directory. Moving it later
without preserving that relationship will intentionally fail validation.

## Four-GPU generation

Wait for the condition's final, verified PEFT checkpoint. For the implausible
condition its expected final path is the following; replace `CHECKPOINT_ID`
with the immutable adapter SHA-256 or archive commit recorded after step 780.

```bash
PY=/data/agastyas/Miniconda3/bin/python
COMP=/data/agastyas/nla_warmstart_robustness/claim_agreement/code
EASYNLA=/data/agastyas/nla_warmstart_robustness/easynla
ACTOR=/data/agastyas/nla_warmstart_robustness/qwen25_7b_seed42/checkpoints/implausible/rl_vllm/iter_000780
SIDECAR=/data/agastyas/nla_warmstart_robustness/qwen25_7b_seed42/checkpoints/implausible/rl_vllm/nla_meta.yaml
CHECKPOINT_ID=REPLACE_WITH_VERIFIED_IMMUTABLE_ID
INPUTS=/data/agastyas/nla_warmstart_robustness/claim_agreement/inputs
OUTPUT=/data/agastyas/nla_warmstart_robustness/claim_agreement/generations

CUDA_VISIBLE_DEVICES=0 $PY "$COMP/generate_av.py" \
  --input-manifest "$INPUTS/fineweb.json" \
  --input-manifest "$INPUTS/static-ood.json" \
  --input-manifest "$INPUTS/browsecomp.json" \
  --condition implausible \
  --actor-checkpoint "$ACTOR" \
  --checkpoint-id "$CHECKPOINT_ID" \
  --sidecar "$SIDECAR" \
  --output-dir "$OUTPUT" \
  --num-shards 4 --shard-index 0 \
  --sampling-seed 42 \
  --easynla-root "$EASYNLA" \
  --device cuda:0
```

Launch the same command on physical GPUs 1, 2, and 3 with matching
`CUDA_VISIBLE_DEVICES` and shard indices 1, 2, and 3. `--device cuda:0` remains
correct inside each one-GPU visibility namespace. Use a cluster reservation and
a detached supervisor for the real run; no GPU selection is hardcoded in the
Python file. Repeat with the plausible checkpoint and `--condition plausible`.
Do not add `--load-in-4bit`: bf16 matches these actors' training/evaluation
path; 4-bit is available only as an explicitly recorded memory fallback.

## Outputs and failure semantics

Each condition/shard directory contains:

- `records/*.json`: canonical per-sample crash-safe records, including raw text
  with and without special-token stripping, generated token IDs, parse status,
  sampling key/seed, source digests, activation norm, and diagnostics;
- `explanations.jsonl`: atomic projection consumed by the claim pipeline;
- `raw_generations.jsonl`: atomic deterministic compaction of all rich records;
- `shard_config.json`, `progress.json`, and `complete.json`, all atomic and
  digest-bearing.

A restart validates existing records against the run fingerprint and skips
them. Malformed explanation tags are preserved as ordinary failed samples and
do not halt the shard. Marker shape is checked before model load; every
generation must increment EasyNLA's mechanism-level hook counter exactly once;
literal marker leakage is always flagged. CJK fraction is recorded on every
sample and values above 0.20 are flagged outside MMMLU (where CJK source content
is expected). The RL iteration directory does not itself need to contain the
sidecar: pass the save root's `nla_meta.yaml` explicitly. Non-marker CJK never
makes an individual generation fail; a shard returns status 2 for manual audit
only when at least three records and more than 5% of records are CJK-suspect.
Marker leakage or a hook-count mismatch stops immediately after its diagnostic
record is saved. Model/injection exceptions are also recorded atomically and
then raised.

Run CPU/mock tests with:

```bash
python3 -m unittest discover \
  -s analysis/warm_start_robustness/claim_agreement/tests -v
```
