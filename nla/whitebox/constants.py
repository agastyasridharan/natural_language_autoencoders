"""Released-NLA registry + published baselines — single source of truth.

What's concrete vs inferred:

  * The **Qwen-2.5-7B L20** entry is fully verified against
    `examples/qwen7b_layer20_step4200.txt` (repo IDs, extraction model, layer,
    d_model, injection_scale, FVE baseline).
  * `injection_scale` and `layer` for the other three families are from the
    paper / `nla-paper-code-bridges` memo. Their HF repo IDs and FVE baselines
    are NOT hardcoded here — guessing vendor paths violates CLAUDE.md
    ("don't hardcode bucket paths or vendor SDKs"). Fill them from the actual
    `nla_meta.yaml` you load. The Qwen repo naming pattern
    (`kitft/nla-{family}-L{layer}-{av,ar}`) is a hint, not a promise.

Everything tokenizer-dependent (token IDs, prompt templates, injection_scale,
mse_scale) is still loaded from the sidecar at runtime; this registry only
holds the few human-facing constants you need *before* a checkpoint is in hand
(which repo to pull, which extraction model produced the activations, the
published FVE denominator).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReleasedNLA:
    """A released NLA pair + the model whose activations it explains."""

    name: str
    av_repo: str | None          # HF repo id for the activation verbalizer (actor)
    ar_repo: str | None          # HF repo id for the activation reconstructor (critic)
    extraction_model: str        # the TARGET model whose layer-K activations are explained
    layer: int                   # extraction layer K (= HF hidden_states[K+1], block-K output)
    d_model: int
    injection_scale: float       # documented per-model value; sidecar is authoritative at runtime
    # Predict-the-mean baseline MSE on the *normalized* training distribution,
    # i.e. Var(v_nrm) — the denominator in `fve_nrm = 1 - mse_nrm / baseline`.
    # None where we don't have the published number; recompute from a training
    # parquet via `nla.schema.compute_predict_mean_baselines` (the raw-variance
    # return value) if you have one.
    fve_nrm_baseline: float | None


# Verified end-to-end against examples/qwen7b_layer20_step4200.txt.
QWEN_7B_L20 = ReleasedNLA(
    name="qwen2.5-7b-L20",
    av_repo="kitft/nla-qwen2.5-7b-L20-av",
    ar_repo="kitft/nla-qwen2.5-7b-L20-ar",
    extraction_model="Qwen/Qwen2.5-7B-Instruct",
    layer=20,
    d_model=3584,
    injection_scale=150.0,
    fve_nrm_baseline=0.7335,
)

# Other families: layer + injection_scale documented; repos/baselines unknown
# (left None on purpose — see module docstring). Extend as you verify them.
GEMMA_12B_L32 = ReleasedNLA(
    name="gemma-3-12b-L32",
    av_repo=None,
    ar_repo=None,
    extraction_model="google/gemma-3-12b-it",
    layer=32,
    d_model=3840,
    injection_scale=80000.0,
    fve_nrm_baseline=None,
)
GEMMA_27B_L41 = ReleasedNLA(
    name="gemma-3-27b-L41",
    av_repo=None,
    ar_repo=None,
    extraction_model="google/gemma-3-27b-it",
    layer=41,
    d_model=5376,
    injection_scale=60000.0,
    fve_nrm_baseline=None,
)
LLAMA_70B_L53 = ReleasedNLA(
    name="llama-3.3-70b-L53",
    av_repo=None,
    ar_repo=None,
    extraction_model="meta-llama/Llama-3.3-70B-Instruct",
    layer=53,
    d_model=8192,
    injection_scale=30.0,
    fve_nrm_baseline=None,
)

RELEASED: dict[str, ReleasedNLA] = {
    m.name: m for m in (QWEN_7B_L20, GEMMA_12B_L32, GEMMA_27B_L41, LLAMA_70B_L53)
}

# The example log's summary excludes positions < 24 (the default system prompt,
# under-sampled in datagen). This is the *log's* convention, distinct from
# datagen's `_MIN_POSITION = 50` (the training sampler's floor). The gate uses
# 24 to match the log it's reproducing.
EXAMPLE_LOG_MIN_POSITION = 24
