"""Small, dependency-light inference helpers for the robustness dashboards.

The inferential unit throughout is a held-out document.  P-values are stored in
log10 form as well as ordinary floating point so extremely small values are not
silently rounded to zero.
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta (Numerical Recipes form)."""
    max_iter, eps, fpmin = 400, 3.0e-14, 1.0e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) <= eps:
            return h
    raise ArithmeticError("incomplete-beta continued fraction did not converge")


def _log_regularized_beta(x: float, a: float, b: float) -> float:
    if x <= 0.0:
        return -math.inf
    if x >= 1.0:
        return 0.0
    log_bt = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
              + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return log_bt + math.log(_betacf(a, b, x)) - math.log(a)
    log_complement = log_bt + math.log(_betacf(b, a, 1.0 - x)) - math.log(b)
    if log_complement < -36.0:
        return math.log1p(-math.exp(log_complement))
    return math.log(-math.expm1(log_complement))


def two_sided_t_pvalue(t_value: float, df: int) -> dict:
    """Two-sided Student-t p-value, robust to probabilities below float range."""
    if df <= 0 or not math.isfinite(t_value):
        return {"p": None, "log10_p": None}
    x = df / (df + t_value * t_value)
    log_p = _log_regularized_beta(x, 0.5 * df, 0.5)
    log10_p = log_p / math.log(10.0)
    p = math.exp(log_p) if log_p > math.log(np.finfo(float).tiny) else 0.0
    return {"p": float(p), "log10_p": float(log10_p)}


def one_sample_t(values: Iterable[float], null: float = 0.0) -> dict:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n < 2:
        return {"n": n, "t": None, "df": max(0, n - 1), "p": None, "log10_p": None}
    delta = float(arr.mean() - null)
    sd = float(arr.std(ddof=1))
    if sd == 0.0:
        if delta == 0.0:
            return {"n": n, "t": 0.0, "df": n - 1, "p": 1.0, "log10_p": 0.0}
        return {"n": n, "t": math.copysign(math.inf, delta), "df": n - 1,
                "p": 0.0, "log10_p": -math.inf}
    t_value = delta / (sd / math.sqrt(n))
    out = two_sided_t_pvalue(t_value, n - 1)
    return {"n": n, "t": float(t_value), "df": n - 1, **out}


def bootstrap_bundle(denominator: np.ndarray, cos_base: np.ndarray,
                     cos_arm: np.ndarray | None, *, n_boot: int, seed: int,
                     chunk: int = 400) -> dict:
    """Joint document bootstrap for FVE, cosine means, and a paired arm effect."""
    d = np.asarray(denominator, dtype=np.float64)
    cb = np.asarray(cos_base, dtype=np.float64)
    ca = None if cos_arm is None else np.asarray(cos_arm, dtype=np.float64)
    if len(d) != len(cb) or (ca is not None and len(ca) != len(d)):
        raise ValueError("bootstrap arrays must be aligned")
    n = len(d)
    rng = np.random.default_rng(seed)
    names = ["FVE_base", "mean_cos_base"]
    if ca is not None:
        names += ["FVE_arm", "dFVE", "mean_cos_arm", "mean_dcos"]
    samples = {name: np.empty(n_boot, dtype=np.float64) for name in names}
    lb = 2.0 * (1.0 - cb)
    la = None if ca is None else 2.0 * (1.0 - ca)
    pos = 0
    while pos < n_boot:
        size = min(chunk, n_boot - pos)
        idx = rng.integers(0, n, size=(size, n), dtype=np.int32)
        dsum = d[idx].sum(axis=1)
        fb = 1.0 - lb[idx].sum(axis=1) / dsum
        samples["FVE_base"][pos:pos + size] = fb
        samples["mean_cos_base"][pos:pos + size] = cb[idx].mean(axis=1)
        if ca is not None and la is not None:
            fa = 1.0 - la[idx].sum(axis=1) / dsum
            samples["FVE_arm"][pos:pos + size] = fa
            samples["dFVE"][pos:pos + size] = fa - fb
            samples["mean_cos_arm"][pos:pos + size] = ca[idx].mean(axis=1)
            samples["mean_dcos"][pos:pos + size] = (ca[idx] - cb[idx]).mean(axis=1)
        pos += size
    return {name: [float(x) for x in np.percentile(vals, [2.5, 97.5])]
            for name, vals in samples.items()}


def bootstrap_mean_ci(values: Iterable[float], *, n_boot: int, seed: int,
                      chunk: int = 1000) -> list[float]:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot, dtype=np.float64)
    pos = 0
    while pos < n_boot:
        size = min(chunk, n_boot - pos)
        idx = rng.integers(0, n, size=(size, n), dtype=np.int32)
        out[pos:pos + size] = arr[idx].mean(axis=1)
        pos += size
    return [float(x) for x in np.percentile(out, [2.5, 97.5])]


def bh_adjust(rows: list[dict], p_key: str = "p_test") -> None:
    """Add q/log10_q in place, using log probabilities to avoid underflow."""
    valid = [(i, row[p_key]["log10_p"]) for i, row in enumerate(rows)
             if row.get(p_key, {}).get("log10_p") is not None]
    valid.sort(key=lambda pair: pair[1])
    m = len(valid)
    adjusted = [0.0] * m
    running = 0.0
    for rank0 in range(m - 1, -1, -1):
        _, lp = valid[rank0]
        candidate = lp + math.log10(m / (rank0 + 1))
        running = min(running, candidate)
        adjusted[rank0] = running
    for (row_index, _), log_q in zip(valid, adjusted):
        q = 10.0 ** log_q if log_q > math.log10(np.finfo(float).tiny) else 0.0
        rows[row_index][p_key]["q"] = float(q)
        rows[row_index][p_key]["log10_q"] = float(log_q)


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> list[float] | None:
    if trials <= 0:
        return None
    phat = successes / trials
    z2 = z * z
    denom = 1.0 + z2 / trials
    center = (phat + z2 / (2.0 * trials)) / denom
    half = z * math.sqrt(phat * (1.0 - phat) / trials + z2 / (4.0 * trials * trials)) / denom
    lo = 0.0 if successes == 0 else max(0.0, center - half)
    hi = 1.0 if successes == trials else min(1.0, center + half)
    return [lo, hi]


def exact_binomial_p_half(successes: int, trials: int) -> float | None:
    """Exact two-sided binomial p-value for H0: probability=0.5."""
    if trials <= 0 or successes < 0 or successes > trials:
        return None
    tail = min(successes, trials - successes)
    p = 2.0 * sum(math.comb(trials, k) for k in range(tail + 1)) / (2.0 ** trials)
    return min(1.0, float(p))
