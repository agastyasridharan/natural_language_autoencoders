"""White-box harness for the released NLA pairs — mechanistic-access layer.

This package gives *white-box* access to a released NLA pair (AV + AR) using
HuggingFace + forward hooks instead of SGLang/Neuronpedia. That buys three
things the black-box inference path (`nla_inference.NLAClient`, which POSTs to
an SGLang server) cannot give:

  1. per-layer residual-stream capture at the injection marker
     (`output_hidden_states` / hooks) — the substrate for the *reading-pathway*
     question: where, in depth, does the AV stop seeing the raw injected vector
     and start carrying the "read" content?
  2. `generate(inputs_embeds=...)` in-process, so injection + decode + AR
     scoring all happen locally with no server to stand up.
  3. gradients / activation-patching hooks (future experiments E3/E4).

Two consumers sit on top of this core:

  * the **reading-pathway** experiments (E1 propagation is scaffolded here in
    `experiments/`), and
  * the **failure-mode program** (`nla/failure_analysis/`, the inference-only
    behavioural harness): its activation-sensitivity (P2) and recurrence (P3)
    probes need *local* AV inference over arbitrary extracted activations. This
    package is exactly that backend — it removes the dependency on a hosted
    (Neuronpedia) or SGLang endpoint for those probes. `failure_analysis` can
    import `WhiteBoxNLA` instead of wrapping `NLAClient`.

Design rules honoured (CLAUDE.md): standard libs only; never edit `miles/`;
the sidecar is the contract (token IDs / templates / `injection_scale` /
`mse_scale` all come from `nla_meta.yaml` via `nla.config.load_nla_config` and
the AR's own sidecar, never hardcoded — except the per-model FVE *baseline*,
which is a published training-set statistic, see `constants.py`).

`example_log` is intentionally dependency-free (stdlib only) so the validation
gate's parser + its unit test run without torch. The heavy, torch-dependent
objects are imported lazily below so `import nla.whitebox.example_log` stays
cheap.
"""

from __future__ import annotations

from typing import Any

from nla.whitebox import example_log
from nla.whitebox.constants import QWEN_7B_L20, RELEASED, ReleasedNLA

__all__ = [
    "example_log",
    "QWEN_7B_L20",
    "RELEASED",
    "ReleasedNLA",
    # lazily resolved (torch-dependent) — see __getattr__:
    "WhiteBoxNLA",
    "ConversationActivations",
    "MarkerPathway",
    "run_gate",
    "GateReport",
]

# Lazy re-export of the torch-heavy API. Keeps `example_log` importable (and its
# test runnable) on a machine without torch, while still allowing
# `from nla.whitebox import WhiteBoxNLA`.
_LAZY = {
    "WhiteBoxNLA": ("nla.whitebox.harness", "WhiteBoxNLA"),
    "ConversationActivations": ("nla.whitebox.harness", "ConversationActivations"),
    "MarkerPathway": ("nla.whitebox.harness", "MarkerPathway"),
    "run_gate": ("nla.whitebox.gate", "run_gate"),
    "GateReport": ("nla.whitebox.gate", "GateReport"),
}


def __getattr__(name: str) -> Any:  # PEP 562 module-level lazy attribute
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    mod = importlib.import_module(target[0])
    return getattr(mod, target[1])
