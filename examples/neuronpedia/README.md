# Neuronpedia NLA demos — replicating the paper's experiments

These scripts replicate experiments from *Natural Language Autoencoders Produce
Unsupervised Explanations of LLM Activations* (`papers/nla paper.pdf`) by driving
the **hosted open-model NLAs** on [Neuronpedia](https://www.neuronpedia.org)
(`https://www.neuronpedia.org/<model>/nla`). No GPU required — the activation
verbalizer (AV) and reconstructor (AR) run server-side; these scripts just call
the demo's JSON / SSE API.

The hosted NLAs are the same AV/AR pairs trained by this repo (see
`nla_inference.py` for the self-hosted equivalent). Each `/api/nla/explain` call
runs the AV (residual activation → text) **and** the AR (text → activation), and
returns both the verbalization and its reconstruction fidelity (cosine / MSE) —
the exact quantity the AV was RL-trained to maximize.

## Setup

```bash
# Key is read from $NEURONPEDIA_API_KEY or the repo-root .env automatically.
export NEURONPEDIA_API_KEY=...          # already present in ./.env
python examples/neuronpedia/demo_01_planning_in_poetry.py        # llama by default
python examples/neuronpedia/demo_02_language_switching.py --model gemma --step 2
```

Only `httpx` is required (already a repo dependency). Run any demo with `-h` for
options. Every demo prints a `re-open in demo` URL so you can inspect the same run
in the web UI.

### Hosted NLAs

| `--model` | modelId            | layer | source id  |
|-----------|--------------------|-------|------------|
| `llama`   | `llama3.3-70b-it`  | 53    | `kitft-l53`|
| `gemma`   | `gemma-3-27b-it`   | 41    | `kitft-l41`|

(Both are ~⅔-depth open-model NLAs — weaker and noisier than the Opus 4.6 NLA used
for most paper figures. The demos therefore *report what the AV actually says* and
run an automated check, rather than asserting a fixed outcome.)

> Each model has its own hosted inference server, which can be cold or
> temporarily down. If you see `the hosted NLA inference server is unavailable`
> (HTTP 5xx), that is a Neuronpedia-side outage — switch `--model` or retry later.
> `llama` is the most reliably available.

## The demos

| Script | Paper section | What it replicates |
|--------|---------------|--------------------|
| `demo_01_planning_in_poetry.py` | Case Studies › Planning in Poetry | Teacher-forces the carrot/rabbit couplet and verbalizes the tokens around the line break, testing whether the activation already encodes the planned rhyme **before** line 2 is written. |
| `demo_02_language_switching.py` | Case Studies › Language Switching | English prompt (with a "moved from Moscow" cue) → Russian reply; sweeps the AV across the transcript and measures prevalence of *target-language* vs *other-language* references, reproducing "the language is represented **before** it is spoken" + its specificity. |
| `demo_03_misreported_tool_calls.py` | Case Studies › Misreported Tool Calls | Rigs a tool to return 492 for a problem whose answer is 491; the model reports 491. Verbalizes the misreported answer token vs controls, testing that **discrepancy/incorrectness awareness localizes to the answer token**. |
| `demo_04_confabulation_recurrence.py` | Quantitative Evaluations › Characterizing NLA confabulations | Verbalizes a contiguous run of tokens and measures which content words **recur across adjacent positions** (trustworthy themes) vs appear once (likely confabulation), and reports per-token AR fidelity — the paper's "read for recurring themes, not individual claims" heuristic. |

### Representative results (llama3.3-70b-it / layer 53)

- **Planning in poetry** — across the line break the AV consistently surfaces the
  rhyming-couplet structure and the carrot/rabbit theme; "rabbit" appears entering
  line 2. The crisp "rabbit *at the newline*" of the paper (Opus 4.6) is more
  diffuse here, as the paper's open-model caveats predict.
- **Language switching** — Russian references spike at the "Moscow" cue (~6/18
  pre-reply positions) and saturate through the reply; other-language references
  stay ≈0. Clean replication of *represented-before-responding* + specificity.
- **Misreported tool calls** — at the "491" token the AV emits "wrong result",
  "typo", "incorrect", "error"; controls show ≤1 such cue. The discrepancy
  awareness localizes exactly as the paper reports.
- **Confabulation/recurrence** — for an Eiffel-Tower context the recurring themes
  are `eiffel, tower, paris, landmark, france` (and real specifics like
  `painted`, `thermal`, `expansion`); one-off terms are the confabulations.

(Outputs vary run-to-run: AV sampling temperature defaults to 0.4.)

## API reference (`np_nla.py`)

The client wraps the demo's four endpoints (reverse-engineered from the frontend
bundle; auth via `x-api-key`):

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/nla/sources` | GET | list hosted `(modelId, nlaSourceId, layer)` NLAs |
| `/api/nla/completion` | POST | tokenize a transcript; optionally run the target model |
| `/api/nla/explain` | POST | verbalize + reconstruct chosen token positions |
| `/api/nla/cache/<id>` | GET | load a saved/shared run by `cacheId` |

`/api/nla/explain` caps each request at **16 new positions**; `np_nla` batches
around this automatically (chaining `priorCacheId`). Returned per position:
`description` (AV text), `cosine_similarity` & `mse` (AR fidelity), `l2_norm`.

```python
from np_nla import NeuronpediaNLA
nla = NeuronpediaNLA("llama3.3-70b-it")            # source/layer inferred
toks = nla.tokenize([{"role": "user", "content": "..."},
                     {"role": "assistant", "content": "..."}])
res  = nla.verbalize(messages, positions=[t.position for t in toks[-5:]])
for v in res.results:
    print(v.position, v.cosine_similarity, v.description)
```

> Chat templates in `np_nla` are replicated verbatim from the demo frontend so a
> teacher-forced transcript built here tokenizes identically to one typed in the
> web UI. The server prepends the BOS token, so `format_chat` must not.

## Caveats / faithfulness

- These run hosted **open-model** NLAs (Llama/Gemma), not the **Opus 4.6** NLA
  behind most paper figures, so the *content* of explanations differs. What the
  demos replicate is each experiment's **method** and **qualitative signature**.
- NLAs confabulate (the whole point of demo 4). Read themes, not single claims;
  weight by recurrence and AR cosine.
- Calls consume an hourly per-key rate limit; a 429 just means wait and retry.
