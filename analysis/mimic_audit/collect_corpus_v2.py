"""Collect real NLA (AV+AR) reads for the large, categorized mimic-tuning corpus.

Reads corpus/corpus_docs.json, runs each document through the hosted
Llama-3.3-70B NLA via the nla_np harness, and writes one row per swept token.

Resumable: rows are appended and flushed per document; rerunning skips documents
already present (in the main corpus_all.jsonl AND in this run's --out file).
Rate-limit aware: on the hourly per-key 429 it waits the window out rather than
giving up. Shardable: pass --ids-file to restrict to a subset and --key-env to
pick which API key (env var name, falling back to .env) so several processes can
run in parallel on disjoint shards, one per key.

Single-key default:
    python collect_corpus_v2.py
Parallel shard (one of several):
    python collect_corpus_v2.py --key-env NEURONPEDIA_API_KEY_2 \
        --ids-file corpus/shard_a.ids --out corpus/shard_a.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
CORPUS = HERE / "corpus"
DOCS = CORPUS / "corpus_docs.json"
MAIN_OUT = CORPUS / "corpus_all.jsonl"

sys.path.insert(0, str(REPO / "nla_np"))

_SPECIALS = {
    "<|begin_of_text|>": "", "<|end_of_text|>": "", "<|eot_id|>": "\n",
    "<|start_header_id|>": "", "<|end_header_id|>": ": ", "<|python_tag|>": "",
}


def reconstruct_prefix(tokens, pos: int) -> str:
    s = "".join(t.token for t in tokens if t.position <= pos)
    for k, v in _SPECIALS.items():
        s = s.replace(k, v)
    return s.strip()


def pick_positions(n_tokens: int) -> list[int]:
    if n_tokens <= 24:
        return list(range(max(1, n_tokens - 6), n_tokens))
    lo = max(50, int(0.45 * n_tokens))
    hi = n_tokens - 1
    if lo >= hi:
        lo = max(1, hi - 7)
    k = 8
    step = max(1, (hi - lo) // (k - 1))
    return sorted({min(hi, lo + i * step) for i in range(k)} | {hi})


def resolve_key(name: str) -> str:
    """Read an API key by env-var name, falling back to an exact match in .env."""
    if os.environ.get(name):
        return os.environ[name]
    for d in (REPO, HERE):
        env = d / ".env"
        if env.is_file():
            for line in env.read_text().splitlines():
                line = line.strip()
                if line.startswith(name + "="):
                    return line.split("=", 1)[1].strip().strip("'\"")
    raise RuntimeError(f"Key {name} not found in environment or .env")


def collected_ids(*paths) -> set[str]:
    ids: set[str] = set()
    for p in paths:
        p = Path(p)
        if p.exists():
            for line in p.open(encoding="utf-8"):
                line = line.strip()
                if line:
                    ids.add(json.loads(line)["doc_id"])
    return ids


def collect(out_path: Path, ids_filter: set[str] | None, tag: str) -> int:
    from np_core import NeuronpediaNLA
    nla = NeuronpediaNLA("llama")
    print(f"[{tag}] Source: {nla.model_id} / {nla.source_id} (layer {nla.layer})", flush=True)

    docs = json.loads(DOCS.read_text(encoding="utf-8"))
    if ids_filter is not None:
        docs = [d for d in docs if d["id"] in ids_filter]
    have = collected_ids(MAIN_OUT, out_path)
    todo = [d for d in docs if d["id"] not in have]
    print(f"[{tag}] {len(docs)} in shard, {len(have)} already collected, {len(todo)} to do\n", flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as f:
        for i, d in enumerate(todo):
            payload = d["payload"]
            res = toks = None
            err_tries = 0
            rate_waits = 0
            while True:
                try:
                    toks = nla.tokenize(payload)
                    positions = pick_positions(len(toks))
                    res = nla.verbalize(payload, positions)
                    break
                except Exception as e:  # noqa: BLE001
                    msg = str(e).lower()
                    if "rate" in msg or "429" in msg:
                        rate_waits += 1
                        if rate_waits > 60:
                            print(f"[{tag}] [{d['id']}] GIVE UP after prolonged rate-limit", flush=True)
                            res = None
                            break
                        print(f"[{tag}] [{d['id']}] rate limited (wait #{rate_waits}) -> sleep 90s", flush=True)
                        time.sleep(90)
                        continue
                    err_tries += 1
                    if err_tries > 4:
                        print(f"[{tag}] [{d['id']}] GIVE UP: {e}", flush=True)
                        res = None
                        break
                    print(f"[{tag}] [{d['id']}] {type(e).__name__}: {e} -> sleep 12s", flush=True)
                    time.sleep(12)
            if res is None or toks is None:
                continue
            tbp = {t.position: t for t in toks}
            for v in sorted(res.results, key=lambda x: x.position):
                f.write(json.dumps({
                    "doc_id": d["id"], "theme": d["theme"], "format": d["format"],
                    "length_class": d["length_class"], "kind": d["kind"],
                    "n_tokens": len(toks), "position": v.position,
                    "token": v.token or (tbp[v.position].token if v.position in tbp else ""),
                    "cosine": round(v.cosine_similarity, 4), "mse": round(v.mse, 4),
                    "prefix": reconstruct_prefix(toks, v.position),
                    "av_description": v.description,
                }, ensure_ascii=False) + "\n")
            f.flush()
            c = [round(v.cosine_similarity, 3) for v in res.results]
            rng = f"{min(c):.2f}-{max(c):.2f}" if c else "n/a"
            print(f"[{tag}] [{i+1}/{len(todo)}] {d['id']:32s} L={len(toks):3d} "
                  f"swept {len(res.results):2d} cos {rng}", flush=True)
            time.sleep(0.3)
    print(f"[{tag}] DONE", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--key-env", default="NEURONPEDIA_API_KEY", help="env var (or .env name) holding the key")
    ap.add_argument("--ids-file", default=None, help="newline-separated doc ids to restrict to")
    ap.add_argument("--out", default=str(MAIN_OUT), help="output jsonl (shard files merge into corpus_all.jsonl)")
    ap.add_argument("--tag", default=None, help="log prefix")
    args = ap.parse_args()

    os.environ["NEURONPEDIA_API_KEY"] = resolve_key(args.key_env)
    ids_filter = None
    if args.ids_file:
        ids_filter = {l.strip() for l in Path(args.ids_file).read_text().splitlines() if l.strip()}
    tag = args.tag or args.key_env.replace("NEURONPEDIA_API_KEY", "key") or "key1"
    return collect(Path(args.out), ids_filter, tag)


if __name__ == "__main__":
    raise SystemExit(main())
