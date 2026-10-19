"""Exp 7 H5 — test whether greedy (T=0) claims belong to the T=0.4 semantic core.

The per-temperature partitions produced by ``exp7_cluster.py`` cannot by themselves compare claims
across temperatures. This step pools each datapoint/tag's greedy claims with one representative from
each T=0.4 core cluster, applies the same validated meaning judge, and records the fraction of greedy
claims that match at least one core representative. Judge results share the versioned Exp-7 cache.

Usage: python3 exp7_h5.py [qwen7b gemma12b gemma27b] [--workers 12]
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from exp7_cluster import CACHE_PATH, TAGS, _outdir, _write_cache
from exp7_judge import MODEL, cache_key, cluster_claims, judge_stats, valid_partition

GREEDY_T = 0.0
CORE_T = 0.4
CORE_THRESHOLD = 0.5


def _by_temperature(data: dict) -> dict[tuple[int, float], dict]:
    return {(block["idx"], float(block["T"])): block for block in data["blocks"].values()}


def _representative(cluster: dict, claims: list[dict]) -> str | None:
    texts = [claims[i]["text"] for i in cluster.get("members", []) if i < len(claims)]
    return max(texts, key=len) if texts else None


def run_key(key: str, workers: int) -> None:
    source = json.loads((_outdir(key) / "exp7_clusters_main.json").read_text())
    blocks = _by_temperature(source)
    cache = json.loads(CACHE_PATH.read_text()) if CACHE_PATH.exists() else {}
    pending: dict[str, list[str]] = {}
    work: list[dict] = []
    singleton_fallback = 0

    idxs = sorted({idx for idx, _ in blocks})
    for idx in idxs:
        greedy = blocks.get((idx, GREEDY_T))
        core = blocks.get((idx, CORE_T))
        if not greedy or not core:
            continue
        for tag in TAGS:
            gclaims = [row["text"] for row in greedy["tags"].get(tag, {}).get("claims", [])]
            ctag = core["tags"].get(tag, {})
            cclaims = ctag.get("claims", [])
            reps = [rep for cluster in ctag.get("meaning", [])
                    if cluster.get("freq", 0.0) >= CORE_THRESHOLD
                    for rep in [_representative(cluster, cclaims)] if rep]
            row = {"idx": idx, "tag": tag, "greedy_claims": gclaims, "core_representatives": reps,
                   "cache_key": None}
            if gclaims and reps:
                items = gclaims + reps
                ck = cache_key(items)
                row["cache_key"] = ck
                if ck in cache and not valid_partition(cache[ck], len(items)):
                    cache[ck] = [[i] for i in range(len(items))]
                    singleton_fallback += 1
                elif ck not in cache:
                    pending.setdefault(ck, items)
            work.append(row)

    print(f"[h5:{key}] {len(pending)} uncached comparisons / {len(work)} datapoint-tags; "
          f"workers={workers}", flush=True)
    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(cluster_claims, items): ck for ck, items in pending.items()}
        for future in as_completed(futures):
            ck = futures[future]
            try:
                partition = future.result()
            except Exception as exc:
                print(f"  H5 worker failed for {ck[:10]}: {exc!r}; using singletons", flush=True)
                partition = [[i] for i in range(len(pending[ck]))]
                singleton_fallback += 1
            if not valid_partition(partition, len(pending[ck])):
                partition = [[i] for i in range(len(pending[ck]))]
                singleton_fallback += 1
            cache[ck] = partition
            completed += 1
            if completed % 50 == 0 or completed == len(pending):
                _write_cache(cache)
                print(f"  compared {completed}/{len(pending)} uncached pools", flush=True)

    rows: dict[str, dict] = {}
    tag_summary = {}
    for tag in TAGS:
        tag_rows = []
        for row in (r for r in work if r["tag"] == tag):
            ng = len(row["greedy_claims"])
            nr = len(row["core_representatives"])
            matched = []
            ck = row.pop("cache_key")
            if ng and nr and ck:
                cid = {}
                for cluster_id, cluster in enumerate(cache[ck]):
                    for claim_id in cluster:
                        cid[claim_id] = cluster_id
                core_cids = {cid[i] for i in range(ng, ng + nr)}
                matched = [i for i in range(ng) if cid.get(i) in core_cids]
            row["n_greedy"] = ng
            row["n_core_representatives"] = nr
            row["matched_greedy_indices"] = matched
            row["n_matched"] = len(matched)
            row["match_fraction"] = (len(matched) / ng) if ng else None
            rows.setdefault(str(row["idx"]), {})[tag] = row
            if ng:
                tag_rows.append(row)
        n_claims = sum(r["n_greedy"] for r in tag_rows)
        n_matched = sum(r["n_matched"] for r in tag_rows)
        tag_summary[tag] = {
            "n_docs_with_greedy_claims": len(tag_rows),
            "n_greedy_claims": n_claims,
            "n_matched": n_matched,
            "claim_weighted_match_rate": (n_matched / n_claims) if n_claims else None,
        }

    if singleton_fallback and not pending:
        _write_cache(cache)
    out = {
        "key": key,
        "judge_model": MODEL,
        "greedy_T": GREEDY_T,
        "core_T": CORE_T,
        "core_threshold": CORE_THRESHOLD,
        "n_cache_miss": len(pending),
        "n_singleton_fallback": singleton_fallback,
        "judge_stats": judge_stats(),
        "tags": tag_summary,
        "rows": rows,
    }
    dest = _outdir(key) / "exp7_h5.json"
    dest.write_text(json.dumps(out, ensure_ascii=False))
    print(f"[h5:{key}] wrote {dest}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("keys", nargs="*", default=["qwen7b", "gemma12b", "gemma27b"])
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    if args.workers < 1:
        ap.error("--workers must be >= 1")
    for key in (args.keys or ["qwen7b", "gemma12b", "gemma27b"]):
        run_key(key, args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
