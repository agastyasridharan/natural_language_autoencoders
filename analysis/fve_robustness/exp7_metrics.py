"""Exp 7 — stability metrics from the meaning/surface clusters (CPU).

Consumes out_<key>/exp7_clusters_{main,floor,shuffle}.json and produces the headline stability readouts,
aggregated over the 200-doc subsample with doc-clustered bootstrap 95% CIs:

  per (T, tag):
    core_fraction   claim-mass in consensus clusters (freq>=0.5); also 0.33 / 0.8 thresholds
    hist            consensus-frequency buckets [0,0.2) / [0.2,0.7) / [0.7,1]  (H1 bimodality)
    entropy         mean semantic entropy of the cluster distribution           (H2 vs T)
    pair_agree      mean pairwise semantic agreement across resample pairs
    surface_core    same as core_fraction on ROUGE-L clusters
    gap             core_fraction - surface_core  (paraphrase-of-stable-idea vs genuinely-different)
  contrasts at T=0.4 (per tag):
    vs_floor        main core - no-injection floor core        (H4: must be > 0)
    vs_shuffle      main core - shuffled-activation core        (activation-specific vs AV prior)
    vs_null         main core - permutation-null core           (core above chance)

Usage:  python3 exp7_metrics.py [qwen7b gemma12b gemma27b] [--pilot]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from inference_stats import bh_adjust, bootstrap_mean_ci, one_sample_t, wilson_interval

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
TAGS = ("genre", "LCR", "PC")
CORE = 0.5
CTRL_T = 0.4
N_BOOT = 20_000


def stable_seed(label: str) -> int:
    return int.from_bytes(hashlib.sha256(label.encode()).digest()[:8], "big") & 0x7FFFFFFF


def _outdir(key: str) -> Path:
    for cand in (Path("/data/agastyas/nla_fve_ufw") / f"out_{key}", RES / f"out_{key}", HERE / f"out_{key}"):
        if (cand / "exp7_clusters_main.json").exists():
            return cand
    return RES / f"out_{key}"


# ── per-datapoint readouts on one tag's meaning/surface cluster list ──────────
def core_mass(clusters: list[dict], thr: float = CORE) -> float | None:
    tot = sum(c["n_claims"] for c in clusters)
    if not tot:
        return None
    return sum(c["n_claims"] for c in clusters if c["freq"] >= thr) / tot


def entropy(clusters: list[dict]) -> float | None:
    tot = sum(c["n_claims"] for c in clusters)
    if not tot:
        return None
    ps = [c["n_claims"] / tot for c in clusters]
    return -sum(p * math.log(p, 2) for p in ps if p > 0)


def pair_agreement(claims: list[dict], clusters: list[dict]) -> float | None:
    """expected fraction of one resample's claims that share a meaning-cluster with another resample's."""
    cl_of = {}
    for cid, c in enumerate(clusters):
        for i in c["members"]:
            cl_of[i] = cid
    by_k = defaultdict(list)
    for i, cd in enumerate(claims):
        by_k[cd["k"]].append(i)
    ks = list(by_k)
    if len(ks) < 2:
        return None
    tot, acc = 0, 0.0
    for a in range(len(ks)):
        for b in range(len(ks)):
            if a == b:
                continue
            ca = {cl_of[i] for i in by_k[ks[a]]}
            cb = {cl_of[i] for i in by_k[ks[b]]}
            if ca:
                acc += len(ca & cb) / len(ca)
                tot += 1
    return acc / tot if tot else None


def null_core(claims: list[dict], eff_n: int, thr: float = CORE, rng=None) -> float | None:
    """permutation null: reassign each claim to a random resample, recompute core mass on the SAME
    cluster partition. Destroys any real resample->cluster coupling."""
    if not claims or not eff_n:
        return None
    rng = rng or random
    by_cluster = defaultdict(list)
    for cd in claims:
        by_cluster[cd.get("_cid")].append(cd)
    tot = len(claims)
    core = 0
    for cid, members in by_cluster.items():
        ks = {rng.randrange(eff_n) for _ in members}
        if len(ks) / eff_n >= thr:
            core += len(members)
    return core / tot if tot else None


def _attach_cids(tagblock: dict, which: str) -> tuple[list[dict], list[dict]]:
    claims = tagblock.get("claims", [])
    clusters = [dict(c, members=list(c["members"])) for c in tagblock.get(which, [])]
    cid_of = {}
    for cid, c in enumerate(clusters):
        for i in c["members"]:
            cid_of[i] = cid
    # Defensive conservative completion for legacy/malformed judge outputs: each omitted claim is a
    # singleton with zero consensus frequency, so it contributes mass but can never fabricate a core.
    for i in range(len(claims)):
        if i not in cid_of:
            cid_of[i] = len(clusters)
            clusters.append(dict(members=[i], n_claims=1, n_resamples=1, freq=0.0))
    cl2 = [dict(cd, _cid=cid_of.get(i)) for i, cd in enumerate(claims)]
    return cl2, clusters


# ── aggregate one model ──────────────────────────────────────────────────────
def per_dp(block_tag: dict) -> dict | None:
    """readouts for one (idx,T,tag) block. Returns None if the tag has no claims."""
    claims, mcl = _attach_cids(block_tag, "meaning")
    _, scl = _attach_cids(block_tag, "surface")
    if not mcl:
        return None
    cm = core_mass(mcl)
    return dict(core=cm, core33=core_mass(mcl, 0.33), core80=core_mass(mcl, 0.8),
                entropy=entropy(mcl), surface_entropy=entropy(scl),
                pair=pair_agreement(claims, mcl),
                surface_core=core_mass(scl),
                gap=(None if cm is None or core_mass(scl) is None else cm - core_mass(scl)),
                freqs=[c["freq"] for c in mcl],
                freq_mass=[(c["freq"], c["n_claims"]) for c in mcl], _claims=claims, _n=None)


def hist_buckets(freqs: list[float]) -> list[float]:
    if not freqs:
        return [0.0, 0.0, 0.0]
    lo = sum(1 for f in freqs if f < 0.2)
    hi = sum(1 for f in freqs if f >= 0.7)
    mid = len(freqs) - lo - hi
    n = len(freqs)
    return [lo / n, mid / n, hi / n]


def hist_mass_buckets(freq_mass: list[tuple[float, int]]) -> list[float]:
    total = sum(mass for _, mass in freq_mass)
    if not total:
        return [0.0, 0.0, 0.0]
    out = [0, 0, 0]
    for freq, mass in freq_mass:
        bucket = 0 if freq < 0.2 else (2 if freq >= 0.7 else 1)
        out[bucket] += mass
    return [value / total for value in out]


def boot_ci(vals: list[float], stat=np.mean, n=N_BOOT, seed=0):
    v = [x for x in vals if x is not None]
    if not v:
        return None, None, None
    arr = np.array(v)
    if stat is not np.mean:
        raise ValueError("dashboard inference supports the registered mean estimand only")
    lo, hi = bootstrap_mean_ci(arr, n_boot=n, seed=seed)
    return round(float(arr.mean()), 6), round(float(lo), 6), round(float(hi), 6)


def aggregate(key: str, pilot: bool) -> dict:
    od = _outdir(key)
    main = json.loads((od / "exp7_clusters_main.json").read_text())
    grid = main["grid"]
    floor = json.loads((od / "exp7_clusters_floor.json").read_text()) if (od / "exp7_clusters_floor.json").exists() else None
    shuf = json.loads((od / "exp7_clusters_shuffle.json").read_text()) if (od / "exp7_clusters_shuffle.json").exists() else None

    h5 = json.loads((od / "exp7_h5.json").read_text()) if (od / "exp7_h5.json").exists() else None

    def core_map(dataset, T, tag):
        vals = {}
        if not dataset:
            return vals
        for b in dataset["blocks"].values():
            if b["T"] != T:
                continue
            pd = per_dp(b["tags"].get(tag, {}))
            if pd and pd["core"] is not None:
                vals[b["idx"]] = pd["core"]
        return vals

    def null_map(dataset, T, tag):
        vals = {}
        for b in dataset["blocks"].values():
            if b["T"] != T:
                continue
            pd = per_dp(b["tags"].get(tag, {}))
            if pd:
                rng = random.Random(f"exp7-null|{key}|{b['idx']}|{T}|{tag}")
                value = null_core(pd["_claims"], b["effective_N"], rng=rng)
                if value is not None:
                    vals[b["idx"]] = value
        return vals

    def paired_delta(left, right, label="paired-delta"):
        common = sorted(set(left) & set(right))
        values = [left[i] - right[i] for i in common]
        return {"n": len(common), "ci": boot_ci(values, seed=stable_seed(f"{key}|{label}")),
                "p_test": one_sample_t(values)}

    def metric_map(dataset, T, tag, field):
        vals = {}
        for b in dataset["blocks"].values():
            if b["T"] != T:
                continue
            pd = per_dp(b["tags"].get(tag, {}))
            if pd and pd.get(field) is not None:
                vals[b["idx"]] = pd[field]
        return vals

    def raw_quality(arm: str) -> dict:
        path = od / f"resample_{arm}.json"
        if not path.exists():
            return {}
        raw = json.loads(path.read_text())
        docs_with_effective = defaultdict(set)
        for sample in raw["samples"].values():
            bad = sample["cjk"] or sample["noexp"] or sample.get("empty", not sample["text"].strip())
            if not bad:
                docs_with_effective[str(sample["T"])].add(sample["idx"])
        quality = {}
        for T, row in raw["per_T"].items():
            n, effective = row["n"], row["effective"]
            quality[T] = dict(n=n, effective=effective, filtered=n - effective,
                              valid_rate=round(effective / n, 4) if n else None,
                              n_docs=raw["n_docs"],
                              docs_with_effective=len(docs_with_effective.get(T, set())))
        return quality

    out = {"key": key, "grid": grid, "curves": {}, "contrasts": {},
           "quality": {arm: raw_quality(arm) for arm in ("main", "floor", "shuffle")},
           "greedy_in_core": {}}
    for tag in TAGS:
        rows = {}
        for T in grid:
            allfreqs, allfreqmass, cores, ents, surface_ents, pairs, gaps, scores, nulls = [], [], [], [], [], [], [], [], []
            for b in main["blocks"].values():
                if b["T"] != T:
                    continue
                pd = per_dp(b["tags"].get(tag, {}))
                if not pd:
                    continue
                cores.append(pd["core"]); ents.append(pd["entropy"])
                surface_ents.append(pd["surface_entropy"]); pairs.append(pd["pair"])
                gaps.append(pd["gap"]); scores.append(pd["surface_core"]); allfreqs += pd["freqs"]
                allfreqmass += pd["freq_mass"]
                rng = random.Random(f"exp7-null|{key}|{b['idx']}|{T}|{tag}")
                nulls.append(null_core(pd["_claims"], b["effective_N"], rng=rng))
            delta_null = [core - null for core, null in zip(cores, nulls)
                          if core is not None and null is not None]
            rows[str(T)] = dict(
                n=len(cores), core=boot_ci(cores), surface_core=boot_ci(scores),
                gap=boot_ci(gaps), entropy=boot_ci(ents), surface_entropy=boot_ci(surface_ents),
                pair_agree=boot_ci(pairs),
                null_core=boot_ci(nulls), hist=hist_buckets(allfreqs),
                hist_mass=hist_mass_buckets(allfreqmass),
                delta_vs_null=(None if T == 0.0 else {
                    "n": len(delta_null), "ci": boot_ci(delta_null),
                    "p_test": one_sample_t(delta_null)}))
        out["curves"][tag] = rows
        # contrasts at CTRL_T
        m = core_map(main, CTRL_T, tag)
        f = core_map(floor, CTRL_T, tag)
        s = core_map(shuf, CTRL_T, tag)
        n = null_map(main, CTRL_T, tag)
        out["contrasts"][tag] = dict(
            main_core=boot_ci(list(m.values())),
            floor_core=(boot_ci(list(f.values())) if floor else None),
            shuffle_core=(boot_ci(list(s.values())) if shuf else None),
            null_core=boot_ci(list(n.values())),
            delta_vs_floor=paired_delta(m, f, f"{tag}-floor"),
            delta_vs_shuffle=paired_delta(m, s, f"{tag}-shuffle"),
            delta_vs_null=paired_delta(m, n, f"{tag}-null"))

        stochastic_T = sorted(T for T in grid if T > 0)
        if stochastic_T:
            t0, t1 = stochastic_T[0], stochastic_T[-1]
            sem0, sem1 = metric_map(main, t0, tag, "entropy"), metric_map(main, t1, tag, "entropy")
            sur0 = metric_map(main, t0, tag, "surface_entropy")
            sur1 = metric_map(main, t1, tag, "surface_entropy")
            sem_change = paired_delta(sem1, sem0, f"{tag}-semantic-entropy")
            surface_change = paired_delta(sur1, sur0, f"{tag}-surface-entropy")
            common = sorted(set(sem0) & set(sem1) & set(sur0) & set(sur1))
            contrast_values = [(sem1[i] - sem0[i]) - (sur1[i] - sur0[i]) for i in common]
            semantic_minus_surface = {
                "n": len(common), "ci": boot_ci(contrast_values),
                "p_test": one_sample_t(contrast_values),
            }
            out.setdefault("entropy_changes", {})[tag] = {
                "from_T": t0, "to_T": t1, "meaning": sem_change,
                "surface": surface_change, "meaning_minus_surface": semantic_minus_surface,
            }
        if h5:
            h5_rows = [row[tag] for row in h5.get("rows", {}).values()
                       if tag in row and row[tag].get("match_fraction") is not None]
            hs = h5.get("tags", {}).get(tag, {})
            out["greedy_in_core"][tag] = dict(
                doc_mean=boot_ci([row["match_fraction"] for row in h5_rows]),
                n_docs=len(h5_rows), n_greedy_claims=hs.get("n_greedy_claims", 0),
                claim_weighted_match_rate=hs.get("claim_weighted_match_rate"),
                p_test_vs_one=one_sample_t([row["match_fraction"] for row in h5_rows], null=1.0))

    for arm_quality in out["quality"].values():
        for row in arm_quality.values():
            row["valid_rate_ci95"] = wilson_interval(row["effective"], row["n"])
            row["docs_effective_rate_ci95"] = wilson_interval(row["docs_with_effective"], row["n_docs"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("keys", nargs="*", default=["qwen7b", "gemma12b", "gemma27b"])
    ap.add_argument("--pilot", action="store_true")
    args = ap.parse_args()
    combined = {}
    for key in (args.keys or ["qwen7b", "gemma12b", "gemma27b"]):
        combined[key] = aggregate(key, args.pilot)
        c = combined[key]["contrasts"]["PC"]
        print(f"[metrics:{key}] PC main_core={c['main_core']} floor={c['floor_core']} shuffle={c['shuffle_core']}")
    family = []
    for model in combined.values():
        for tag in TAGS:
            for T, row in model["curves"][tag].items():
                if row.get("delta_vs_null"):
                    family.append(row["delta_vs_null"])
            family.extend(model["contrasts"][tag][name]
                          for name in ("delta_vs_floor", "delta_vs_shuffle", "delta_vs_null"))
            family.extend(model["entropy_changes"][tag][name]
                          for name in ("meaning", "surface", "meaning_minus_surface"))
            family.append({"p_test": model["greedy_in_core"][tag]["p_test_vs_one"]})
    bh_adjust(family)
    (RES / "exp7_metrics.json").write_text(json.dumps(combined, ensure_ascii=False, indent=2))
    print(f"wrote {RES / 'exp7_metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
