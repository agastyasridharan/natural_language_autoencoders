# Running the white-box gate on Colab Pro+ (A100 80 GB)

The gate **stages** the models — it frees each before loading the next — so
peak GPU memory is ~one 7B model (~15 GB). It runs on whatever Colab Pro+ hands
you this session (an L4/A10 at ~22 GB, or an A100). Budget ~15–25 min for a cold
session — almost all of it is the ~41 GB weight download; compute is < 10 min.

> **Runtime → Change runtime type → GPU + High-RAM** (A100 if offered; an L4 or
> A10 at ~22 GB also works — the gate stages to fit ~16 GB+). Check which you
> got with `!nvidia-smi`.

## 0. Get this code onto Colab

A plain `git clone` of upstream `main` **won't include `nla/whitebox/`** — it
lives on the **`whitebox-harness`** branch of the fork
`github.com/agastyasridharan/natural_language_autoencoders`, which cell 1 below
clones directly. (The fork is public; if you've made it private, run
`login()` in §2 first or the clone needs a token.)

## 1. Clone + install

```python
# cell 1 — clone the fork's whitebox-harness branch (this is where nla/whitebox/ lives)
!git clone -b whitebox-harness https://github.com/agastyasridharan/natural_language_autoencoders.git
%cd natural_language_autoencoders
!git log --oneline -1          # expect: 2309cdb Add white-box harness ...
!ls nla/whitebox/             # sanity: harness.py, gate.py, cli.py present
```

```python
# cell 2 — deps the gate needs (torch/transformers/numpy are preinstalled).
# We do NOT install sglang/miles/megatron — the white-box path doesn't use them.
!pip -q install "transformers>=4.50" accelerate safetensors httpx orjson pyyaml huggingface_hub
```

## 2. Hugging Face auth

The Qwen and `kitft/*` repos are public; a token is only required if a repo is
gated or you hit rate limits. Safe to set anyway:

```python
# cell 3
from huggingface_hub import login
login()  # paste an HF token, or skip if the repos are public for you
```

## 3. Run the gate

```python
# cell 4 — extraction check first: fastest, ~15 GB, and it's the check most
# likely to catch a wiring bug (layer 20 == hidden_states[21] == block-20 output).
!python -m nla.whitebox.cli gate --released qwen2.5-7b-L20 \
    --log examples/qwen7b_layer20_step4200.txt --level extract --device cuda
```

```python
# cell 5 — L1 (deterministic AR re-scoring of the dump's own text). ~1 min compute.
!python -m nla.whitebox.cli gate --released qwen2.5-7b-L20 \
    --log examples/qwen7b_layer20_step4200.txt --level L1 --device cuda
```

```python
# cell 6 — L2 (end-to-end: AV greedy re-verbalize + re-score). ~3–8 min compute.
!python -m nla.whitebox.cli gate --released qwen2.5-7b-L20 \
    --log examples/qwen7b_layer20_step4200.txt --level L2 --device cuda
```

## 4. E1 — marker propagation (the first reading-pathway measurement)

```python
# cell 7 — per-layer marker residual norm + cos-to-injected, reply region only,
# saved to e1_qwen.npz for plotting.
!python -m nla.whitebox.cli e1 --released qwen2.5-7b-L20 \
    --log examples/qwen7b_layer20_step4200.txt --min-position 34 \
    --out e1_qwen.npz --device cuda
```

```python
# cell 8 — quick look at the curves
import numpy as np
d = np.load("e1_qwen.npz", allow_pickle=True)
norms, cos = d["norms"].mean(0), d["cos"].mean(0)   # mean over reply positions
for L in range(len(cos)):
    print(f"layer {L:2d}: mean‖h‖={norms[L]:8.1f}  mean cos→injected={cos[L]:.3f}")
```

## What "pass" looks like

- **extract**: `max raw-norm rel err` ≲ 0.02 → the activations line up with the
  logged `‖v‖` position-for-position (correct layer + chat template).
- **L1**: summary `cos_mean ≈ 0.890`, `fve_mean ≈ 0.700`, `fve>0.5 ≈ 74/77`,
  and median `|Δcos|` ≲ 0.02 vs the dump. This is the tight, deterministic gate.
- **L2**: distribution close to the dump (`cos_mean` within ~0.05) — per-position
  *text* won't match the SGLang-logged decode, by design (HF greedy ≠ SGLang).

If L1 passes, the harness is trustworthy and E1–E4 can build on it.

## Notes

- **Disk**: ~41 GB of weights land in `~/.cache/huggingface`. Colab gives plenty,
  but it's ephemeral — expect to re-download each fresh session (or mount Drive
  and point `HF_HOME` at it to persist the cache).
- **Speed**: the gate is dominated by download + the 3 model loads. L1 is ~101
  short AR forwards; L2 adds 77 AV generations. Nothing is parallelized — this is
  a correctness gate, not a throughput tool.
- **Other models**: `--released gemma-3-12b-L32 / gemma-3-27b-L41 /
  llama-3.3-70b-L53` are registered, but their HF repo ids aren't filled in
  (`constants.py`) — pass `--av/--ar` local dirs. Llama-70B won't fit one A100.
