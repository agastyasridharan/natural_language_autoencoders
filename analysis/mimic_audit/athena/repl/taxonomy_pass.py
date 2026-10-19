"""Mass taxonomy pass - classify every activation's claims across the atom runs.

The approved scale-up of the taxonomy tab's visual phase: one Anthropic call per
activation (identical prompt/tool/PROMPT_VERSION to the tab's classify button),
threaded, resumable by cache-file existence, atomic writes. It fills the same
per-activation cache the studio tab reads, so the tab's progress counts and
aggregate tables update live while this runs.

  python3 repl/taxonomy_pass.py                                  # all atom runs, sonnet
  python3 repl/taxonomy_pass.py --runs qwen7b-cp6-fineweb-atoms \
      --model haiku --stride 8 --limit 50                        # agreement subsample
  python3 repl/taxonomy_pass.py --dry-run                        # count, no API calls
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import claim_taxonomy as ct

ATHENA = HERE.parent
RUNS = ATHENA / "runs"
TAXO = ATHENA / "taxonomy"


def atom_runs() -> list[str]:
    out = []
    for d in sorted(RUNS.iterdir()):
        try:
            rj = json.loads((d / "run.json").read_text())
        except Exception:
            continue
        if rj.get("mock"):
            continue
        if rj.get("claim_mode") == "atoms" or d.name.endswith("-atoms"):
            out.append(d.name)
    return out


def load_records(rid: str) -> list[dict]:
    out = []
    for line in (RUNS / rid / "records.jsonl").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def run_pass(rid: str, model: str, workers: int, limit: int | None,
             stride: int, dry: bool) -> dict:
    recs = load_records(rid)
    todo, cached = [], 0
    for j, rec in enumerate(recs):
        if j % stride:
            continue
        if not rec.get("snippet_results"):
            continue
        if ct.cache_path(TAXO, rid, rec["idx"], model).exists():
            cached += 1
            continue
        todo.append(rec)
        if limit is not None and len(todo) >= limit:
            break
    print(f"[{rid}] {len(todo)} to classify ({cached} already cached, model={model})",
          flush=True)
    if dry or not todo:
        return {"run": rid, "new": 0, "cached": cached, "errors": 0,
                "in_tok": 0, "out_tok": 0}

    lock = threading.Lock()
    done = {"n": 0, "in": 0, "out": 0, "err": []}
    t0 = time.time()

    def work(rec):
        last = None
        for attempt in range(3):                       # ride out transient API errors
            try:
                res = ct.classify_activation(rec, model, run_id=rid)
                ct.write_cache(ct.cache_path(TAXO, rid, rec["idx"], model), res)
                return res
            except Exception as e:
                last = e
                time.sleep(5 * (attempt + 1))
        raise RuntimeError(f"idx {rec['idx']}: {last}")

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(work, rec): rec["idx"] for rec in todo}
        for f in as_completed(futs):
            with lock:
                try:
                    res = f.result()
                    done["n"] += 1
                    done["in"] += res["usage"].get("input_tokens") or 0
                    done["out"] += res["usage"].get("output_tokens") or 0
                except Exception as e:
                    done["err"].append(str(e))
                n = done["n"] + len(done["err"])
                if n % 20 == 0 or n == len(todo):
                    rate = done["n"] / max(time.time() - t0, 1e-9)
                    eta = (len(todo) - n) / max(rate, 1e-9)
                    print(f"[{rid}] {n}/{len(todo)}  ok={done['n']} err={len(done['err'])}"
                          f"  tok={done['in']}/{done['out']}  eta={eta/60:.1f}m", flush=True)
    for e in done["err"][:10]:
        print(f"[{rid}] ERROR {e}", flush=True)
    return {"run": rid, "new": done["n"], "cached": cached,
            "errors": len(done["err"]), "in_tok": done["in"], "out_tok": done["out"]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="*", default=None,
                    help="run ids (default: every non-mock atom run)")
    ap.add_argument("--model", choices=sorted(ct.MODELS), default="sonnet")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None,
                    help="max NEW classifications per run (for subsamples)")
    ap.add_argument("--stride", type=int, default=1,
                    help="take every k-th record (for subsamples)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rids = args.runs or atom_runs()
    print(f"runs: {rids}", flush=True)
    totals = []
    for rid in rids:
        totals.append(run_pass(rid, args.model, args.workers, args.limit,
                               args.stride, args.dry_run))
    in_tok = sum(t["in_tok"] for t in totals)
    out_tok = sum(t["out_tok"] for t in totals)
    # rough public-price cost for the log line only
    price = {"sonnet": (3, 15), "haiku": (1, 5)}[args.model]
    cost = (in_tok * price[0] + out_tok * price[1]) / 1e6
    print("\n=== totals ===", flush=True)
    for t in totals:
        print(f"  {t['run']}: new={t['new']} cached={t['cached']} errors={t['errors']}")
    print(f"  tokens: {in_tok:,} in / {out_tok:,} out  (~${cost:.2f} {args.model})",
          flush=True)
    return 1 if any(t["errors"] for t in totals) else 0


if __name__ == "__main__":
    sys.exit(main())
