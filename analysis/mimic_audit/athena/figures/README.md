# Claim-ablation figures

Publication-style bar charts of the atomic claim-ablation results, the same numbers
the studio renders. Regenerate any time with:

```bash
cd analysis/mimic_audit/athena
python make_figures.py          # reads runs/*-atoms/summary.json -> figures/*.png
```

Source: the four atom-mode runs (FineWeb, ~400 activations each), read straight from
`runs/<model>-fineweb-atoms/summary.json`. Error bars are Wilson 95% CIs computed in
`make_figures.py` (scipy-free). Models are ordered by size: Qwen2.5-7B (L20),
Gemma-3-12B (L32), Gemma-3-27B (L41, UltraFineWeb), Llama-3.3-70B (L53).

Verdicts: **grounded** = claim is read from the activation h, **confabulated** = filled
from the prior, **entangled** = located but not separable from neighbours.

| File | What it shows | Headline |
|---|---|---|
| `verdicts_across_models.png` | grounded / entangled / confabulated rate per model | confabulation is the plurality verdict in every model; Gemma-3-12B most confabulatory (58%) |
| `grounding_by_claimtype.png` | grounded rate by semantic type, grouped by model | **GIST is the top-grounded type in all four models**; the pattern replicates cross-family |
| `grounding_<model>.png` (x4) | one model, grounded rate per type + overall line | per-model view of the same ordering, with per-type n and CIs |
| `verdict_composition.png` | 2x2, full verdict composition per type per model | GIST bars carry the tallest grounded (green) base everywhere |
| `hedging_calibration.png` | grounded rate for hedged vs confident claims | the AV's hedging is **not** calibrated to grounding (flat / inconsistent sign) |

The scientific read: models ground the **gist** (theme / topic / genre) far more than
concrete **position** or **prediction** claims, which they tend to confabulate. This is
the "reads gist, confabulates the specifics" pattern, and it holds across the Qwen, Gemma,
and Llama families. See `docs/research/claim_ablation_atoms.md` for the full write-up.
