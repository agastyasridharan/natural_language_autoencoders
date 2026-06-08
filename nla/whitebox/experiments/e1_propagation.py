"""E1 — Marker propagation through the AV's depth.

The reading-pathway hypothesis (from the paper's injection-scale rationale):
the AV injects a layer-K activation at layer 0, scaled to a large fixed norm so
the *raw* vector survives the early layers ~unperturbed until a depth that knows
how to read a layer-K activation; there the marker residual is consumed and the
position starts carrying read/derived content (and the later generated tokens
attend back to it).

E1 makes that curve concrete. For each activation, `capture_marker_pathway`
returns the marker position's residual at every depth; E1 stacks them and
reports, per layer:
  * mean ‖residual‖ — does the norm stay near `injection_scale` then collapse?
  * mean cos-to-injected — at what depth does the residual rotate off the
    injected direction (the candidate "read" layer)?

Run it on in-distribution activations (e.g. the reply region of the worked
example) and, for contrast, on OOD ones (system-prompt / special-token
positions) — the propagation/read curves should differ, which is itself a
prediction of the evidence-gated account (OOD => no readable content => the
marker is never "read", or read into prior-dominated content).

Output: an `.npz` (norms[N,L+1], cos[N,L+1]) for downstream plotting + a text
summary. No plotting dep is pulled in (CLAUDE.md: standard libs only) — the
arrays are saved for whatever the user plots with.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from nla.whitebox.harness import ConversationActivations, WhiteBoxNLA


@dataclass
class E1Result:
    norms: torch.Tensor        # [N, num_layers+1] L2 norm of marker residual per depth
    cos: torch.Tensor          # [N, num_layers+1] cosine to the injected vector per depth
    layer_index: int           # extraction layer K (the activations' origin)
    injection_scale: float
    labels: list[str] | None   # per-activation label (e.g. token string), optional

    @property
    def n_layers(self) -> int:
        return self.norms.shape[1] - 1  # minus the embedding-output row

    def mean_norm(self) -> torch.Tensor:
        return self.norms.mean(dim=0)

    def mean_cos(self) -> torch.Tensor:
        return self.cos.mean(dim=0)

    def read_depth(self, cos_threshold: float = 0.5) -> int | None:
        """First depth at which the mean marker residual has rotated off the
        injected direction (mean cos < threshold) — a candidate "read" layer."""
        mc = self.mean_cos()
        below = (mc < cos_threshold).nonzero().flatten()
        return int(below[0]) if below.numel() else None

    def save(self, path: str | Path) -> None:
        np.savez(
            str(path),
            norms=self.norms.numpy(),
            cos=self.cos.numpy(),
            layer_index=self.layer_index,
            injection_scale=self.injection_scale,
            labels=np.array(self.labels if self.labels is not None else [], dtype=object),
        )


def run_e1(
    nla: WhiteBoxNLA,
    activations,
    *,
    labels: list[str] | None = None,
    out_path: str | Path | None = None,
) -> E1Result:
    """`activations`: a [N, d] tensor, a list of [d] vectors, or a
    ConversationActivations (its residuals are used)."""
    if isinstance(activations, ConversationActivations):
        labels = labels or activations.token_strs
        activations = activations.residuals
    vecs = torch.as_tensor(np.asarray([np.asarray(v, dtype=np.float32) for v in activations]))

    norms, coss = [], []
    for i in range(vecs.shape[0]):
        path = nla.capture_marker_pathway(vecs[i])
        norms.append(path.norms)
        coss.append(path.cos_to_injected)
    res = E1Result(
        norms=torch.stack(norms),
        cos=torch.stack(coss),
        layer_index=nla.layer,
        injection_scale=float(nla.cfg.injection_scale),
        labels=labels,
    )
    if out_path is not None:
        res.save(out_path)
    return res


def summarize_e1(res: E1Result, cos_threshold: float = 0.5) -> str:
    mn, mc = res.mean_norm(), res.mean_cos()
    rd = res.read_depth(cos_threshold)
    lines = [
        "── E1: marker propagation through AV depth ──",
        f"  activations: {res.norms.shape[0]}   layers: {res.n_layers}   "
        f"extraction layer K={res.layer_index}   injection_scale={res.injection_scale:g}",
        f"  candidate read depth (mean cos<{cos_threshold}): "
        f"{rd if rd is not None else 'never crosses (marker stays aligned)'}",
        "  depth :   mean‖h‖     mean cos→injected",
    ]
    for L in range(res.norms.shape[1]):
        marker = "  <- read" if (rd is not None and L == rd) else ""
        lines.append(f"  {L:5d} : {mn[L]:10.1f}     {mc[L]:7.3f}{marker}")
    return "\n".join(lines)
