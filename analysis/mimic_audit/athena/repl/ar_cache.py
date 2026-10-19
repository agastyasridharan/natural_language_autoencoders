"""AR-direction cache for the phase-split cycle pipeline (llama70b path).

The cycle measurement interleaves two large models: the AR (critic) reconstructs
z / z\\B / z*_m to build the same-subtrahend directions (cc.decoy_directions),
and the AV injects h / h-bar / h' to score spans. For the 70B NLA those two
models cannot co-reside on the 2x H100 NVL (2 x ~140GB > 192GB). They never run
concurrently within a claim, though: decoy_directions produces a fully
serializable dict that the AV consumes afterward. So we split into two GPU
phases:

  Phase AR  (critic only):  ar_precompute.py iterates the claims, calls
                            cc.decoy_directions, and writes the dict here.
  Phase AV  (AV only):      calib_cycle.measure / cycle_run read the cache via
                            load_ar_directions instead of loading the critic.

The cache is byte-for-byte the decoy_directions return value (arrays in .npz,
strings/scalars in .json), plus an alignment manifest (cand_texts, z_inner_sha,
prompt_version, mse_scale) so the AV phase can assert it is consuming directions
built from exactly the candidate list it re-derives. For the models that DO
co-reside (qwen7b, gemma12b, gemma27b) the cache is unused and the live critic
path is unchanged.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

# dd fields that are per-decoy or per-dim numpy arrays -> stored in the .npz
_ARR_KEYS = ("delta_b", "deltas", "ov", "cos_star", "mse_star",
             "ar_full", "ar_minus")
# dd scalar fields -> stored in the .json
_SCALAR_KEYS = ("ov_b", "cos_full", "mse_full")


def _stem(out_dir: Path, run: str, idx: int, ci: int) -> Path:
    d = out_dir / run
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{idx}_{ci}"


def save_ar_directions(out_dir: Path, run: str, idx: int, ci: int, dd: dict,
                       cand_texts: list[str], z_inner: str, prompt_version: str,
                       mse_scale: float) -> None:
    """Persist a cc.decoy_directions() dict for one claim.

    star_spans (list of (int,int)) is stored as a [g,2] int array; z_stars (list
    of str) stays in the json. The manifest lets the AV phase fail closed on any
    candidate-order drift.
    """
    stem = _stem(out_dir, run, idx, ci)
    g = len(cand_texts)
    star_spans = np.asarray(dd["star_spans"], dtype=np.int64).reshape(g, 2) if g \
        else np.zeros((0, 2), np.int64)
    np.savez(
        str(stem) + ".npz",
        delta_b=np.asarray(dd["delta_b"], np.float32),
        deltas=np.asarray(dd["deltas"], np.float32),
        ov=np.asarray(dd["ov"], np.float32),
        cos_star=np.asarray(dd["cos_star"], np.float32),
        mse_star=np.asarray(dd["mse_star"], np.float32),
        ar_full=np.asarray(dd["ar_full"], np.float32),
        ar_minus=np.asarray(dd["ar_minus"], np.float32),
        star_spans=star_spans,
    )
    manifest = dict(
        idx=idx, claim_i=ci, g=g,
        z_stars=list(dd["z_stars"]),
        ov_b=float(dd["ov_b"]), cos_full=float(dd["cos_full"]),
        mse_full=float(dd["mse_full"]),
        cand_texts=list(cand_texts),
        z_inner_sha=hashlib.sha256(z_inner.encode()).hexdigest()[:16],
        prompt_version=prompt_version,
        mse_scale=float(mse_scale),
    )
    tmp = Path(str(stem) + ".json.tmp")
    tmp.write_text(json.dumps(manifest), encoding="utf-8")
    tmp.replace(str(stem) + ".json")


def has_ar_directions(out_dir: Path, run: str, idx: int, ci: int) -> bool:
    stem = _stem(out_dir, run, idx, ci)
    return Path(str(stem) + ".npz").exists() and Path(str(stem) + ".json").exists()


def load_ar_directions(out_dir: Path, run: str, idx: int, ci: int,
                       cand_texts: list[str], z_inner: str,
                       prompt_version: str) -> dict:
    """Return a dict identical in shape to cc.decoy_directions() output.

    Asserts the cached candidate list, z_inner hash and prompt version match the
    live re-derivation, so a stale or misaligned cache raises rather than
    silently scoring the wrong directions.
    """
    stem = _stem(out_dir, run, idx, ci)
    man = json.loads(Path(str(stem) + ".json").read_text(encoding="utf-8"))
    assert man["prompt_version"] == prompt_version, \
        (idx, ci, f"ar-cache prompt drift {man['prompt_version']} != {prompt_version}")
    sha = hashlib.sha256(z_inner.encode()).hexdigest()[:16]
    assert man["z_inner_sha"] == sha, (idx, ci, "ar-cache z_inner drift")
    assert man["cand_texts"] == list(cand_texts), \
        (idx, ci, "ar-cache candidate-order drift")
    z = np.load(str(stem) + ".npz")
    star_spans = [tuple(int(v) for v in row) for row in z["star_spans"]]
    return dict(
        delta_b=z["delta_b"], deltas=z["deltas"], ov=z["ov"],
        cos_star=z["cos_star"], mse_star=z["mse_star"],
        ar_full=z["ar_full"], ar_minus=z["ar_minus"],
        star_spans=star_spans, z_stars=list(man["z_stars"]),
        ov_b=float(man["ov_b"]), cos_full=float(man["cos_full"]),
        mse_full=float(man["mse_full"]),
    )


def cache_mse_scale(out_dir: Path, run: str, idx: int, ci: int) -> float:
    stem = _stem(out_dir, run, idx, ci)
    man = json.loads(Path(str(stem) + ".json").read_text(encoding="utf-8"))
    return float(man["mse_scale"])
