# analysis/ — research tooling (not part of the nla package)

Standalone analyses that need dependencies outside the repo's contract
(pandas, matplotlib). Nothing in `nla/` imports from here.

## wdiff — weight-space diff of NLA modules vs. their target model

Computes per-tensor diff statistics (relative Frobenius norm, flat cosine,
singular spectra, energy/stable/entropy ranks) between a released NLA module
(AV or AR) and its base model, streaming individual tensors over HTTP range
requests — no full-model load, no disk cache.

```bash
cd analysis
python -m wdiff.cli synth                      # sanity: known-rank recovery
python -m wdiff.cli selfdiff --repo Qwen/Qwen2.5-0.5B-Instruct --out out/selfdiff
python -m wdiff.cli run --base Qwen/Qwen2.5-7B-Instruct \
    --ft kitft/nla-qwen2.5-7b-L20-av --out out/qwen7b_av --pair-name qwen7b-av
python -m wdiff.cli figures --run out/qwen7b_av
python -m wdiff.cli cross --av-run out/qwen7b_av --ar-run out/qwen7b_ar --out out/qwen7b_cross
python -m unittest discover -s tests           # unit tests
```

Deps: torch, safetensors, httpx, numpy, pandas, matplotlib.
