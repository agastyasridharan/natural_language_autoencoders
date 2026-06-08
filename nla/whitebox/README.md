# `nla.whitebox` — white-box harness for the released NLA pairs

A mechanistic-access layer over a released NLA pair (AV + AR), using
HuggingFace + forward hooks instead of SGLang/Neuronpedia. It runs the whole
round trip in one process and exposes the internals the black-box path hides.

```
extraction model (TARGET LLM)  --forward+hook-->  layer-K activation  h
AV  (verbalizer)               --inject+generate-->  explanation text  z
AR  (reconstructor, NLACritic) --reconstruct+score-->  (mse_nrm, cos)  → FVE
```

This is **Phase 0** of the *reading-pathway* project: stand up the harness,
prove it reproduces a logged decode dump, and dump the first mechanistic
measurement (E1, marker propagation). Phases 1–2 (experiments E2–E4 and the two
injection improvements) build on this core.

## Why white-box (and how it relates to the failure-mode program)

`nla_inference.NLAClient` decodes by POSTing `input_embeds` to an SGLang
server — fast, but a black box: no hidden states, no gradients, no patching,
and it needs a server (or a hosted endpoint like Neuronpedia) standing.

`WhiteBoxNLA` loads the models locally and gives:

1. **per-layer residual capture at the injection marker** (`output_hidden_states`)
   — the substrate for "where in depth does the AV read the activation?";
2. **`generate(inputs_embeds=...)` in process** — inject + decode + AR-score with
   no server;
3. **gradients / patching hooks** — for E3/E4.

Because it does AV inference *locally over arbitrary extracted activations*, it
is also the missing backend for the inference-only **failure-mode program**
(`nla/failure_analysis/`): its activation-sensitivity (P2) and recurrence (P3)
probes need exactly this and were otherwise blocked on a hosted endpoint. That
package can `from nla.whitebox import WhiteBoxNLA` instead of wrapping an SGLang
client — same checkpoints, same sidecar contract, no server.

## The validation gate

`gate.py` reproduces a logged dump (`examples/qwen7b_layer20_step4200.txt`) with
three independent checks, cheapest/most-diagnostic first:

| check | what it isolates | determinism |
|---|---|---|
| **extraction** | re-extract the conversation, compare each position's raw `‖h‖` to the logged `‖v‖` | exact (fp tolerance) — catches the layer-index / chat-template footgun |
| **L1 / AR** | feed the dump's *own* explanation text through the AR vs the re-extracted `h`; compare `(mse_nrm, cos)` | deterministic — no sampling |
| **L2 / end-to-end** | re-verbalize with the AV (greedy) and re-score; compare the summary distribution | loose — HF greedy ≠ SGLang-logged decode token-for-token |

> **Layer-index footgun.** "Layer 20" = `hidden_states[21]` = the **output of
> decoder block 20** (the residual stream entering block 21), which is what
> `resolve_decoder_layers(model)[20]`'s forward hook captures. Off-by-one here
> silently shifts every activation. The extraction check exists to catch it.

The FVE metric is `fve_nrm = 1 - mse_nrm / 0.7335`, where `0.7335 = Var(v_nrm)`
is the **training-set** predict-the-mean baseline (per-model; see
`constants.py`). `mse_nrm` and `cos` are direction-only (both vectors
L2-normalized to `mse_scale = √d`), so `mse_nrm = 2(1 - cos)`.

## Running it

```bash
# Validation gate (auto-downloads the released Qwen NLA from HF):
python -m nla.whitebox.cli gate --released qwen2.5-7b-L20 \
    --log examples/qwen7b_layer20_step4200.txt --level L1      # then --level L2

# Just the extraction check (fastest; no AR/AV):
python -m nla.whitebox.cli gate --released qwen2.5-7b-L20 \
    --log examples/qwen7b_layer20_step4200.txt --level extract

# E1 — marker-propagation curves over the reply region:
python -m nla.whitebox.cli e1 --released qwen2.5-7b-L20 \
    --log examples/qwen7b_layer20_step4200.txt --min-position 34 --out e1_qwen.npz

# Single-position smoke:
python -m nla.whitebox.cli decode --released qwen2.5-7b-L20 \
    --log examples/qwen7b_layer20_step4200.txt --position 71
```

From local checkpoint dirs instead of HF repos: pass `--av ./av_dir --ar
./ar_dir --extraction-model Qwen/Qwen2.5-7B-Instruct --layer 20`.

**Compute.** Three models — two full (extraction + AV) and one truncated (AR).
Qwen-2.5-7B ≈ 15 + 15 + 11 GB in bf16, fits one 80 GB GPU. Use `--device auto`
to shard via accelerate, or `WhiteBoxNLA.unload_*()` to stage on a smaller card.
The network-free parser + arithmetic are unit-tested (`tests/test_example_log.py`,
runs on CPU with no weights); the gate itself needs a GPU + the checkpoints.

## Layout

```
constants.py          released-NLA registry + published FVE baselines
example_log.py        STDLIB-ONLY parser for examples/*.txt dumps (+ its test)
harness.py            WhiteBoxNLA — the shared core (extract/inject/generate/capture/score)
gate.py               the Phase-0 validation gate (extract / L1 / L2)
cli.py                argparse CLI: gate | e1 | decode
experiments/
  e1_propagation.py   E1 — marker residual norm + cos-to-injected vs depth
tests/
  test_example_log.py network-free smoke test (parser + FVE arithmetic + import)
```

## Reading-pathway roadmap (E1 here; E2–E4 + improvements next)

- **E1 — Propagation (scaffolded).** Marker residual norm + cos-to-injected vs
  depth. Tests the "kept alive by a large `injection_scale` until a read layer"
  story; locates a candidate read depth.
- **E2 — Logit/tuned lens at the marker.** Decode the marker residual at each
  depth through the unembedding (and a tuned lens) — what does the AV "think the
  injected token is" as depth grows? Where does it stop looking like a token and
  start looking like content?
- **E3 — Activation patching / causal tracing.** Patch the marker residual (or
  attention edges from it) at each (layer, position) to find which components
  causally move the explanation — *where* the activation is read.
- **E4 — AR reconstruction-leverage, mechanistic.** Per-claim ΔMSE under
  ablation (shared with `failure_analysis/leverage.py`) tied back to the E3
  read-sites — the mechanistic substrate of the failure program's "evidence
  field".
- **Improvements.** (A) a learned affine injection map replacing the scalar
  `injection_scale`; (B) type-matched deep injection at layer ℓ. Both need a
  small train and are validated by re-running the gate + E1.
```
