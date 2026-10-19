"""Exp 7 — decompose resamples into typed claims, cluster by MEANING (validated LLM) + SURFACE (ROUGE-L).

Consumes the resample harness output (out_<key>/resample_<arm>.json) and, per (datapoint, temperature,
tag), pools claims across the surviving resamples and partitions them into meaning equivalence-classes
(the validated exp7_judge.cluster_claims) and, separately, ROUGE-L>=0.9 surface classes (the paired
control). Consensus frequency c(cluster) = (# distinct resamples contributing >=1 claim to the cluster) /
effective_N, where effective_N excludes CJK / no-<explanation> / empty samples (that filtered fraction is
logged per temperature -- it is the "noise grows with T" signal).

Claim tags (from exp6_edits.analyze_datapoint, the authoritative directional tagger):
  genre  = block-0 register/genre line (one claim per resample)
  LCR    = left-context restatements
  PC     = predicted continuations
ECHO/DUP/JUNK are frozen out (as in Exp 6).

Meaning clustering is CACHED (results/exp7_meaning_cache.json), with a versioned judge+claim key.

Output: out_<key>/exp7_clusters_<arm>.json  (per idx|T|tag: claims, meaning clusters+consensus, surface
clusters, effective_N, n_filtered).

Usage:  python3 exp7_cluster.py <key> --arm {main,floor,shuffle} [--pilot 10] [--no-api]
        (--no-api runs claim extraction + surface clustering only; skips the meaning judge)
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from exp6_edits import analyze_datapoint, rouge_l, split_blocks

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
# ROOT differs on cluster vs mac; resolve out_<key> relative to wherever the resample file is.
@lru_cache(maxsize=1)
def _source_documents():
    """Load the external corpus only when an experiment needs document text."""
    path = HERE / "data" / "ufw_docs.json"
    if not path.is_file():
        raise FileNotFoundError(f"Required experiment corpus is not bundled: {path}")
    return {d["id"]: d["text"] for d in json.loads(path.read_text())}
CACHE_PATH = RES / "exp7_meaning_cache.json"
TAGS = ("genre", "LCR", "PC")


def _outdir(key: str) -> Path:
    for cand in (Path("/data/agastyas/nla_fve_ufw") / f"out_{key}", RES / f"out_{key}", HERE / f"out_{key}"):
        if (cand / f"resample_main.json").exists() or (cand / "resample_floor.json").exists() \
                or (cand / "resample_shuffle.json").exists() or cand.exists():
            return cand
    return RES / f"out_{key}"


def extract_claims(text: str, token: str, doc: str, char_end: int, char_start) -> dict:
    """-> {'genre':[...], 'LCR':[...], 'PC':[...]} for one resample's explanation."""
    out = {t: [] for t in TAGS}
    if not text.strip():
        return out
    info = analyze_datapoint(text, token, doc, char_end, char_start)
    blocks = split_blocks(text)             # list of (start, end) char spans
    if blocks:
        s, e = blocks[0]
        b0 = text[s:e].strip()
        if b0:
            out["genre"].append(b0)
    for q in info.get("quotes", []):
        if q["tag"] in ("LCR", "PC"):
            t = q["text"].strip()
            if t:
                out[q["tag"]].append(t)
    return out


def surface_clusters(claims: list[str], thr: float = 0.9) -> list[list[int]]:
    """Union-find on word-sequence ROUGE-L >= ``thr`` (the paired surface control)."""
    n = len(claims)
    parent = list(range(n))
    toks = [claim.split() for claim in claims]
    counts = [Counter(words) for words in toks]

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            ni, nj = len(toks[i]), len(toks[j])
            if not ni or not nj:
                continue
            # LCS cannot exceed multiset token overlap. Its resulting F1 is therefore bounded by
            # 2*overlap/(ni+nj); skip pairs that cannot possibly reach the requested threshold.
            overlap = sum((counts[i] & counts[j]).values())
            if 2 * overlap / (ni + nj) < thr:
                continue
            if rouge_l(toks[i], toks[j]) >= thr:
                parent[find(i)] = find(j)
    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    return [sorted(g) for g in groups.values()]


def consensus(partition: list[list[int]], src_k: list[int], eff_n: int) -> list[dict]:
    out = []
    for cl in partition:
        ks = {src_k[i] for i in cl}
        out.append(dict(members=cl, n_claims=len(cl), n_resamples=len(ks),
                        freq=round(len(ks) / eff_n, 4) if eff_n else 0.0))
    return sorted(out, key=lambda c: -c["freq"])


def load_tokens(key: str) -> dict:
    recs = json.loads((RES / f"records_exp6b_{key}.json").read_text())
    return {r["idx"]: dict(token=r.get("token") or "", char_start=r.get("char_start"),
                           char_end=r.get("char_end"), doc_id=r.get("doc_id")) for r in recs}


def _write_cache(cache: dict) -> None:
    """Atomically checkpoint the API cache so an interrupted full run can resume."""
    tmp = CACHE_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(cache, ensure_ascii=False))
    tmp.replace(CACHE_PATH)


def run(key: str, arm: str, pilot, use_api: bool, workers: int = 12) -> None:
    od = _outdir(key)
    data = json.loads((od / f"resample_{arm}.json").read_text())
    samples = data["samples"]
    toks = load_tokens(key)

    # group surviving samples by (idx, T); track filtered
    by_it = defaultdict(list)            # (idx,T) -> [(k, claims_by_tag)]
    filt = defaultdict(lambda: [0, 0])   # (idx,T) -> [n_total, n_filtered]
    for s in samples.values():
        idx, T = s["idx"], s["T"]
        filt[(idx, T)][0] += 1
        if s["cjk"] or s["noexp"] or s.get("empty", not s["text"].strip()):
            filt[(idx, T)][1] += 1
            continue
        ti = toks.get(idx, {})
        doc = _source_documents().get(ti.get("doc_id"), "")
        cl = extract_claims(s["text"], ti.get("token", ""), doc, ti.get("char_end"), ti.get("char_start"))
        by_it[(idx, T)].append((s["k"], cl))

    n_singleton_fallback = 0
    cache = json.loads(CACHE_PATH.read_text()) if (use_api and CACHE_PATH.exists()) else {}
    pending: dict[str, list[str]] = {}
    refs: list[tuple[dict, str, list[int], int]] = []
    if use_api:
        from exp7_judge import cache_key, cluster_claims, judge_stats, valid_partition  # noqa: API key

    result, n_calls = {}, 0
    idxs = sorted({idx for idx, _ in by_it})
    if pilot:
        idxs = idxs[:pilot]
    for idx in idxs:
        for T in data["grid"]:
            entries = by_it.get((idx, T), [])
            eff_n = len(entries)
            tot, nf = filt.get((idx, T), [0, 0])
            block = dict(idx=idx, T=T, effective_N=eff_n, n_total=tot, n_filtered=nf, tags={})
            for tag in TAGS:
                pooled, src_k = [], []
                for k, cl in entries:
                    for c in cl[tag]:
                        pooled.append(c)
                        src_k.append(k)
                if not pooled:
                    block["tags"][tag] = dict(claims=[], meaning=[], surface=[])
                    continue
                scl = surface_clusters(pooled)
                tag_out = dict(
                    claims=[dict(k=src_k[i], text=pooled[i]) for i in range(len(pooled))],
                    meaning=[],
                    surface=consensus(scl, src_k, eff_n))
                block["tags"][tag] = tag_out
                if use_api:
                    ck = cache_key(pooled)
                    if ck in cache and not valid_partition(cache[ck], len(pooled)):
                        cache[ck] = [[i] for i in range(len(pooled))]
                        n_singleton_fallback += 1
                    elif ck not in cache:
                        if len(pooled) == 1:
                            cache[ck] = [[0]]
                        else:
                            pending.setdefault(ck, pooled)
                    refs.append((tag_out, ck, src_k, eff_n))
                    n_calls += 1
                else:
                    tag_out["meaning"] = tag_out["surface"]
            result[f"{idx}|{T}"] = block

    if use_api:
        print(f"[cluster:{key}:{arm}] {len(pending)} uncached claim pools / {n_calls} total; "
              f"workers={workers}", flush=True)
        completed = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(cluster_claims, items): ck for ck, items in pending.items()}
            for future in as_completed(futures):
                ck = futures[future]
                try:
                    partition = future.result()
                except Exception as exc:  # an unhandled worker failure must remain conservative
                    print(f"  cluster worker failed for {ck[:10]}: {exc!r}; using singletons", flush=True)
                    partition = [[i] for i in range(len(pending[ck]))]
                    n_singleton_fallback += 1
                if not valid_partition(partition, len(pending[ck])):
                    partition = [[i] for i in range(len(pending[ck]))]
                    n_singleton_fallback += 1
                cache[ck] = partition
                completed += 1
                if completed % 50 == 0 or completed == len(pending):
                    _write_cache(cache)
                    print(f"  clustered {completed}/{len(pending)} uncached pools", flush=True)
        for tag_out, ck, src_k, eff_n in refs:
            tag_out["meaning"] = consensus(cache[ck], src_k, eff_n)
        if not pending and n_singleton_fallback:
            _write_cache(cache)
        api_stats = judge_stats()
    else:
        api_stats = {"direct_requests": 0, "direct_failures": 0}
    out = dict(key=key, arm=arm, grid=data["grid"], tags=list(TAGS), api=use_api,
               n_judge_calls=n_calls, n_cache_miss=len(pending), workers=(workers if use_api else 0),
               n_singleton_fallback=n_singleton_fallback, judge_stats=api_stats, blocks=result)
    (od / f"exp7_clusters_{arm}.json").write_text(json.dumps(out, ensure_ascii=False))
    print(f"[cluster:{key}:{arm}] {len(result)} (idx,T) blocks, {n_calls} judge pools, "
          f"singleton_fallback={n_singleton_fallback}, api={use_api} -> "
          f"{od / f'exp7_clusters_{arm}.json'}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("key")
    ap.add_argument("--arm", choices=["main", "floor", "shuffle"], default="main")
    ap.add_argument("--pilot", type=int, default=None)
    ap.add_argument("--no-api", action="store_true")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    if args.workers < 1:
        ap.error("--workers must be >= 1")
    run(args.key, args.arm, args.pilot, use_api=not args.no_api, workers=args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
