"""Pure per-tensor statistics. No I/O, no HF awareness — unit-testable.

All inputs are float32 torch tensors on CPU. Frobenius norms and inner
products are accumulated in float64 (chunked, so peak memory stays bounded);
SVDs run in float32 on the requested device.

Spectrum policy (per task spec):
- full torch.linalg.svd when min(shape) <= full_threshold (default 8192),
  which also yields exact top singular vectors;
- otherwise randomized top-K via torch.svd_lowrank (K=topk, niter=2,
  oversampling 16), with the exact ||dW||_F^2 kept separately so tail energy
  is accounted exactly.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

_CHUNK = 1 << 24  # 16.7M elements -> ~134 MB transient float64 per chunk


def fro_sq(x: torch.Tensor) -> float:
    """Exact ||x||_F^2 with float64 accumulation."""
    xf = x.reshape(-1)
    total = 0.0
    for i in range(0, xf.numel(), _CHUNK):
        c = xf[i : i + _CHUNK].double()
        total += float(torch.dot(c, c))
    return total


def dot_flat(a: torch.Tensor, b: torch.Tensor) -> float:
    """Exact <a, b> (flattened) with float64 accumulation."""
    af, bf = a.reshape(-1), b.reshape(-1)
    total = 0.0
    for i in range(0, af.numel(), _CHUNK):
        total += float(torch.dot(af[i : i + _CHUNK].double(), bf[i : i + _CHUNK].double()))
    return total


def cos_flat(w_ft: torch.Tensor, w_base: torch.Tensor) -> float:
    """Cosine between the flattened weight matrices (documents saturation)."""
    nf, nb = fro_sq(w_ft), fro_sq(w_base)
    if nf == 0.0 or nb == 0.0:
        return float("nan")
    return dot_flat(w_ft, w_base) / math.sqrt(nf * nb)


@dataclass
class Spectrum:
    sigmas: np.ndarray          # descending, float64
    exact: bool                 # True if this is the full spectrum
    method: str                 # "full" | "randomized"
    u: np.ndarray | None = None  # [m, k_keep] float32 (top left vectors)
    v: np.ndarray | None = None  # [n, k_keep] float32 (top right vectors)


def compute_spectrum(
    dW: torch.Tensor,
    *,
    full_threshold: int = 8192,
    topk: int = 256,
    niter: int = 2,
    oversample: int = 16,
    keep_vectors: int = 16,
    device: str = "cpu",
    max_svd_dim: int | None = None,
) -> Spectrum:
    """SVD of a 2D dW per the spectrum policy. Always float32 arithmetic."""
    assert dW.ndim == 2, dW.shape
    m, n = dW.shape
    use_full = min(m, n) <= full_threshold
    if use_full and device == "cpu" and max_svd_dim is not None and max(m, n) > max_svd_dim:
        use_full = False  # CPU safeguard for enormous full SVDs

    A = dW if device == "cpu" else dW.to(device)
    if use_full:
        U, S, Vh = torch.linalg.svd(A, full_matrices=False)
        k = min(keep_vectors, S.numel())
        spec = Spectrum(
            sigmas=S.double().cpu().numpy(),
            exact=True,
            method="full",
            u=U[:, :k].float().cpu().numpy(),
            v=Vh[:k, :].T.contiguous().float().cpu().numpy(),
        )
    else:
        q = min(topk + oversample, min(m, n))
        U, S, V = torch.svd_lowrank(A, q=q, niter=niter)
        k = min(topk, S.numel())
        kv = min(keep_vectors, k)
        spec = Spectrum(
            sigmas=S[:k].double().cpu().numpy(),
            exact=False,
            method="randomized",
            u=U[:, :kv].float().cpu().numpy(),
            v=V[:, :kv].float().cpu().numpy(),
        )
    if device != "cpu":
        del A
    return spec


def energy_rank(sigmas: np.ndarray, fro_sq_exact: float, p: float, exact: bool) -> int | str:
    """min k with sum_{i<=k} sigma_i^2 >= p * ||dW||_F^2.

    With a full spectrum the energy denominator is the spectrum's own total
    (self-consistent against SVD round-off). With a partial spectrum the exact
    Frobenius energy is used; if p*energy is not reachable within the computed
    K, returns the string ">K".
    """
    if fro_sq_exact <= 0.0:
        return 0
    sig2 = sigmas.astype(np.float64) ** 2
    cum = np.cumsum(sig2)
    denom = cum[-1] if exact else fro_sq_exact
    if denom <= 0.0:
        return 0 if exact else f">{len(sigmas)}"
    target = p * denom * (1.0 - 1e-12)
    idx = int(np.searchsorted(cum, target))
    if idx < len(cum):
        return idx + 1
    if exact:
        return len(cum)  # numerical shortfall only
    return f">{len(sigmas)}"


def stable_rank(fro_sq_exact: float, sigma1: float) -> float:
    if sigma1 <= 0.0:
        return 0.0
    return fro_sq_exact / (sigma1 * sigma1)


def entropy_erank(sigmas: np.ndarray) -> float:
    """Spectral-entropy effective rank exp(-sum p_i log p_i), p_i = sigma_i/sum.
    Only meaningful on a full spectrum; caller gates on Spectrum.exact."""
    s = sigmas.astype(np.float64)
    tot = s.sum()
    if tot <= 0.0:
        return 0.0
    p = s / tot
    p = p[p > 0.0]
    return float(np.exp(-(p * np.log(p)).sum()))


def two_d_stats(
    dW: torch.Tensor,
    fro_sq_dw: float,
    *,
    full_threshold: int = 8192,
    topk: int = 256,
    niter: int = 2,
    oversample: int = 16,
    keep_vectors: int = 16,
    device: str = "cpu",
    max_svd_dim: int | None = None,
) -> tuple[dict, Spectrum]:
    """All spectrum-derived statistics for a paired 2D tensor."""
    spec = compute_spectrum(
        dW,
        full_threshold=full_threshold,
        topk=topk,
        niter=niter,
        oversample=oversample,
        keep_vectors=keep_vectors,
        device=device,
        max_svd_dim=max_svd_dim,
    )
    sigma1 = float(spec.sigmas[0]) if spec.sigmas.size else 0.0
    out = {
        "spectrum": spec.method,
        "spectrum_exact": spec.exact,
        "sigma1": sigma1,
        "stable_rank": stable_rank(fro_sq_dw, sigma1),
        "erank_entropy": entropy_erank(spec.sigmas) if spec.exact else None,
    }
    for p in (0.5, 0.9, 0.99):
        out[f"energy_rank_{p}"] = energy_rank(spec.sigmas, fro_sq_dw, p, spec.exact)
    return out, spec


def principal_angle_cos2(u1: np.ndarray, u2: np.ndarray) -> np.ndarray:
    """cos^2 of principal angles between the column spaces of u1 [d,k], u2 [d,k].
    Columns are orthonormalized first (defensive; SVD vectors should already be)."""
    q1, _ = np.linalg.qr(u1.astype(np.float64))
    q2, _ = np.linalg.qr(u2.astype(np.float64))
    s = np.linalg.svd(q1.T @ q2, compute_uv=False)
    return np.clip(s, 0.0, 1.0) ** 2


def random_subspace_mean_cos2(d: int, k: int) -> float:
    """E[mean_i cos^2 theta_i] for two independent uniform k-dim subspaces of
    R^d is k/d (sum of cos^2 has expectation k^2/d)."""
    return k / d
