# NLA Studio

A single self-contained folder that runs the **Natural Language Autoencoder**
inference loop for the released `kitft` NLA checkpoints and exposes it through a
minimal local web UI. **Inference only.** No training stack (no Miles, Megatron,
GRPO, or any SGLang training patch other than the one Gemma input_embeds patch).

An NLA pair is two fine-tuned models that map a hidden-state vector to language
and back:

```
  base model  --extract-->  vector  --AV-->  "an explanation in English"  --AR-->  vector'
                              gold                                                  predicted
                                \____________________ cosine / MSE ________________/
```

- **AV** (activation verbalizer / actor): vector to text. Needs `input_embeds`
  injection, so it is served by **SGLang** (one server per family). The app
  talks to it over HTTP using the vendored client.
- **AR** (activation reconstructor / critic): text to vector. Runs **in-process**
  on GPU (`NLACritic`).
- **base model**: the original instruct model. Runs **in-process** on GPU; the
  `/extract` endpoint forwards it with `output_hidden_states`.

The injection and MSE math is **not reimplemented here**. It is taken verbatim
from [`kitft/nla-inference`](https://github.com/kitft/nla-inference)
(`vendor/nla_inference.py`: `NLAClient`, `NLACritic`). The Gemma SGLang patch is
copied from
[`kitft/natural_language_autoencoders`](https://github.com/kitft/natural_language_autoencoders).

## Layout

```
nla_studio/
  README.md                 this file
  requirements.txt
  vendor/nla_inference.py   verbatim from kitft/nla-inference (NLAClient + NLACritic)
  patches/nla_gemma3_mm_input_embeds.patch   Gemma-only SGLang bypass (canonical diff)
  app/
    server.py               FastAPI app: endpoints + in-memory vector store
    nla_core.py             wrappers over the vendored AV/AR + extract() on the base model
    families.yaml           the 4-family registry
  web/index.html            single-page vanilla-JS UI
  scripts/
    launch_sglang.sh        launch the AV server for a family (applies Gemma patch if needed)
    run.sh                  one command: optionally launch SGLang, then start the web app
    smoke.py                end-to-end check
    apply_gemma_patch.py    idempotent applier for the Gemma input_embeds bypass
```

## Families

Confirm the repo ids and `layer` in `app/families.yaml` against each HF model
card before loading (the cards say "Residual stream output of block K").

| family   | base model                        | AV repo                        | AR repo                        | layer K | d_model | gated |
|----------|-----------------------------------|--------------------------------|--------------------------------|---------|---------|-------|
| qwen7b   | Qwen/Qwen2.5-7B-Instruct          | kitft/nla-qwen2.5-7b-L20-av    | kitft/nla-qwen2.5-7b-L20-ar    | 20      | 3584    | no    |
| gemma12b | google/gemma-3-12b-it             | kitft/nla-gemma3-12b-L32-av    | kitft/nla-gemma3-12b-L32-ar    | 32      | 3840    | yes   |
| gemma27b | google/gemma-3-27b-it             | kitft/nla-gemma3-27b-L41-av    | kitft/nla-gemma3-27b-L41-ar    | 41      | 5376    | yes   |
| llama70b | meta-llama/Llama-3.3-70B-Instruct | kitft/Llama-3.3-70B-NLA-L53-av | kitft/Llama-3.3-70B-NLA-L53-ar | 53      | 8192    | yes   |

## Install

```bash
cd nla_studio
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # pinned sglang[all]==0.5.6 + transformers==4.57.1 (needs a CUDA box)
```

Versions are pinned deliberately (see the comment in `requirements.txt`): newer
sglang pins transformers 5.x, which re-tokenizes the injection char and breaks
injection. After install, verify the tokenizer stack in a few seconds (downloads
only the tokenizer + sidecar, no weights, no GPU):

```bash
python scripts/preflight_tokenizer.py --family qwen7b
```

## Hardware notes

`/extract` (base) and the AR run in-process; the AV runs in the separate SGLang
process. So a family needs room for **base + AR in one process** plus **AV in
SGLang** (which grabs `--mem-fraction-static` of its GPU, default 0.85).

- **qwen7b** is the reference family. Base 7B (~15 GB bf16) + AR ~5.3B (21 of 28
  blocks, ~11 GB) in-process is ~26 GB; the AV 7B in SGLang is another ~15 GB +
  KV cache. Cleanest on **two GPUs**: SGLang AV on GPU 0, base+AR on GPU 1. On a
  single 80 GB GPU, lower `NLA_MEM_FRACTION` (e.g. 0.5) so the in-process models
  fit alongside SGLang.
- **gemma12b / gemma27b**: gated (export `HF_TOKEN`) and require the Gemma SGLang
  patch (applied automatically by `launch_sglang.sh`). Plan one GPU for the AV
  server and one for base+AR; 27b wants 80 GB-class cards.
- **llama70b**: multi-GPU. The AV server runs tensor-parallel across the visible
  GPUs (`--tp`, auto-detected). The base model loads with `device_map="auto"`
  (sharded). **Caveat:** the vendored `NLACritic` places the AR on a single
  device, so the 70B AR (54 of 80 blocks) needs a large single GPU or a sharded
  AR variant that is out of scope for this demo. qwen7b is the path that is fully
  exercised end to end.

One family is resident at a time. Loading a new family frees the previous one.

## Quick start — qwen7b (the reference path)

qwen is ungated, so no `HF_TOKEN` is needed. Pick GPUs to taste.

**Option A — one command, single GPU.** `run.sh --sglang` starts SGLang and the
web app in one shell, so both inherit the same `CUDA_VISIBLE_DEVICES`. Shrink the
AV footprint so base+AR fit alongside it (needs an 80 GB-class card):

```bash
cd nla_studio
NLA_INPROC_DEVICE=cuda:0 NLA_MEM_FRACTION=0.5 scripts/run.sh qwen7b --sglang
```

**Option B — two terminals, two GPUs (recommended).** This is the clean split:
AV server on GPU 0, base+AR in-process on GPU 1.

```bash
# Terminal 1: AV server on GPU 0
CUDA_VISIBLE_DEVICES=0 scripts/launch_sglang.sh qwen7b
#   -> python -m sglang.launch_server --model-path kitft/nla-qwen2.5-7b-L20-av \
#        --port 30000 --disable-radix-cache --mem-fraction-static 0.85 \
#        --context-length 2048 --trust-remote-code

# Terminal 2: web app; base+AR in-process on GPU 1
CUDA_VISIBLE_DEVICES=1 NLA_INPROC_DEVICE=cuda:0 scripts/run.sh qwen7b
```

Then open **http://localhost:8000** and:

1. **Family**: pick `qwen7b`, click **Load** (downloads AV/AR snapshots, loads
   base+AR, verifies the SGLang AV server is reachable).
2. **Get vector**: type text, click **Extract**. Click a token to select its
   activation. Early positions (first ~10) are greyed and high-norm outliers are
   flagged in red; both decode unreliably even when injection is correct.
3. **Explain (AV)**: click to verbalize the selected vector. The status line
   warns if the output is mostly CJK (injection failure).
4. **Reconstruct / Score**: the AV text drops into the AR box. Click
   **Reconstruct (AR)** to get a predicted vector, or **Score vs selected
   vector** to compare any text against the extracted vector. Results show cosine
   and MSE with the interpretation scale.

## Smoke test

With the qwen7b SGLang AV server running:

```bash
cd nla_studio
NLA_INPROC_DEVICE=cuda:0 NLA_SGLANG_URL=http://localhost:30000 python scripts/smoke.py
```

It loads qwen7b, extracts from `"The quick brown fox jumps over the lazy dog."`
at a mid/late position, runs the AV explanation, and **confirms it is English,
not CJK** (the hard pass/fail). It then AR-scores that explanation against the
extracted vector; on a clean, high-context position cosine lands roughly in the
**0.85–0.95** band. Because the fox sentence is short (every position has limited
left-context), the smoke test reports the cosine and treats CJK as the gate; for
the full 0.85–0.95 demo use a longer paragraph and a later token in the UI.

**All-CJK output means injection failed.** Check `injection_scale`,
`embed_scale`, and (Gemma only) the gemma3_mm patch.

## The layer off-by-one (important)

`layer` in the registry is the data-gen extraction layer index **K** (the "L20"
in `nla-qwen2.5-7b-L20`). Data-gen captured the activation with a forward hook on
**decoder block K**, i.e. the *output* of block K. In HF `output_hidden_states`,
index 0 is the embedding output and index `i` is the output of block `i-1`, so:

```
   output of block K  ==  hidden_states[K + 1]
```

`/extract` therefore reads `hidden_states[K + 1]`, **not** `hidden_states[K]`
(that would be the residual stream entering block K, off by one block and out of
distribution). The AR confirms the index: it is the first K+1 blocks of the base
model, so `AR.num_hidden_layers == K + 1`. `load_family` asserts that the
registry K, the AV sidecar's `extraction_layer_index`, and the AR layer count all
agree, and fails loudly if not.

## Correctness rules respected (the documented failure modes)

1. **Only `input_embeds` go to SGLang**, never `input_ids` too (handled inside
   the vendored `NLAClient`).
2. **`--disable-radix-cache` is required** (set by `launch_sglang.sh`). The cache
   keys on token ids, which `input_embeds` requests lack, so different embed
   sequences would alias to one cache entry and serve garbage.
3. **Gemma**: apply the gemma3_mm bypass (auto-applied for Gemma families);
   `embed_scale = sqrt(hidden_size)` and `injection_scale` come from the sidecar
   (roughly 500x Qwen's). Skipping any of these gives repeated-newline or all-CJK
   output.
4. **AR scoring**: the critic tokenizes with `add_special_tokens=True` (BOS for
   Gemma/Llama, no-op for Qwen), extracts at the **last** token, normalizes both
   predicted and gold to `mse_scale = sqrt(d_model)`, giving `MSE = 2(1 - cos)`
   in `[0, 4]`. All inside the vendored `NLACritic`.
5. **Extraction indexing**: `hidden_states[0]` is the embedding output; we use
   `hidden_states[K + 1]` (see the off-by-one section above).
6. **One family at a time.** `load_family` frees the previous family first.
7. **Sidecar is the contract**: `injection_scale`, `embed_scale`, the injection
   token id and its neighbors, the AV prompt template, `mse_scale`, and the AR
   layer count are read from `nla_meta.yaml` and asserted against the live
   tokenizer at load (the assertion lives in the vendored `load_nla_config`).

## API reference (FastAPI)

Vectors never round-trip through the UI; they are kept in an in-memory store
keyed by id and referenced by `vector_id`. Responses expose only id, source
text, token index, and L2 norm.

| method + path        | body                                  | returns |
|----------------------|---------------------------------------|---------|
| `GET  /families`     | -                                     | registry + loaded family |
| `POST /load_family`  | `{family}`                            | family config, scales, layer indices |
| `POST /extract`      | `{text, token_index?}`                | per-token L2 norms + previews; if `token_index`, stores the vector and returns its id |
| `POST /av_explain`   | `{vector_id, temperature?, max_new_tokens?}` | explanation text + CJK fraction |
| `POST /ar_reconstruct` | `{text}`                            | predicted vector id + L2 norm |
| `POST /score`        | `{text, vector_id}`                   | `{cosine, mse, label, scale}` |
| `POST /roundtrip`    | `{text, token_index}`                 | extract -> av_explain -> score in one call |

## Gemma and Llama specifics

```bash
export HF_TOKEN=hf_...                       # gated repos
CUDA_VISIBLE_DEVICES=0 scripts/launch_sglang.sh gemma12b   # auto-applies the patch
```

`launch_sglang.sh` runs `apply_gemma_patch.py` for Gemma families, which edits the
installed `sglang/srt/models/gemma3_mm.py` (idempotent; backs up to
`gemma3_mm.py.nla_bak`). The canonical diff is
`patches/nla_gemma3_mm_input_embeds.patch`. For llama70b the launcher sets
tensor-parallel across visible GPUs automatically.

## Troubleshooting

- **`load_family` fails: "injection token appears 0× in canonical prompt".**
  Dependency-version drift. `sglang>=0.5.6` resolves to a newer sglang (0.5.13+)
  that pins transformers 5.x, whose tokenizer splits the injection char (`㈎`)
  inside the prompt, so the single-token injection anchor disappears. Pin the
  tested stack and restart:
  `pip install "sglang[all]==0.5.6" "transformers==4.57.1"` (already pinned in
  `requirements.txt`). Catch it in seconds before launching anything with
  `python scripts/preflight_tokenizer.py --family <family>`.
- **AV output is all CJK / repeated newlines.** Injection failed. Confirm
  `--disable-radix-cache`, that SGLang got only `input_embeds`, the
  `injection_scale`/`embed_scale` reported on Load look right, and (Gemma) that
  the patch applied (`apply_gemma_patch.py --check`).
- **`load_family` raises "AR num_hidden_layers != K+1".** The registry `layer`
  is wrong for that checkpoint. Trust the AR; fix `families.yaml`.
- **`SGLang AV server not reachable`.** Start it (`launch_sglang.sh <family>`),
  confirm the port matches `NLA_SGLANG_URL`, and wait for it to finish loading.
- **CUDA OOM on Load.** SGLang is holding too much of the GPU. Lower
  `NLA_MEM_FRACTION`, or put base+AR on a different GPU via
  `NLA_INPROC_DEVICE` / `CUDA_VISIBLE_DEVICES`.
