"""Segment the AV explanations of an existing claim-ablation run into the atomic
typed claim-set (decomp_demo/segment.py :: segment_v3), and cache the result.

This is the CPU/API pre-pass for the atomic-claim replication. It never touches a GPU:
it reads the AV explanations already stored in a prior run's records.jsonl and calls the
constrained LLM segmenter (Sonnet, temp 0) once per activation, caching to

    <athena>/segments/<key>.jsonl        # one JSON line per activation idx

The cache is resumable and idempotent: idx values already present are skipped, so CP1
(a small sample) and CP2 (all 400) share the same file and CP2 only fills the gaps.

Usage
  # CP1 sanity: segment a small diverse sample, verbose per-activation breakdown
  python segment_run.py qwen7b --reuse-run qwen7b-20260622-121049-fineweb \
      --only 0,20,50,100,137,200,300,380

  # CP2: segment everything (resumes; skips cached)
  python segment_run.py qwen7b --reuse-run qwen7b-20260622-121049-fineweb --workers 8

Each cache line:
  {idx, doc_id, read_token, z_inner, model, coverage:{base_frac,content_frac},
   n_atoms, n_content, n_connective, atoms:[{text,type,subtype,start,end,hedged,
   assertion,regex_confirmed}, ...]}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_REPL = Path(__file__).resolve().parent
_ATHENA = _REPL.parent
_DECOMP = _ATHENA / "decomp_demo"
for _p in (str(_REPL), str(_ATHENA), str(_DECOMP)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import segment as seg  # decomp_demo/segment.py  (segment_v3, CONTENT_TYPES)

CONTENT = set(seg.CONTENT_TYPES)  # {"theme","entity","detail"}
_write_lock = threading.Lock()


def runs_dir_default() -> Path:
    return Path(os.environ.get("CLAIM_ABLATION_RUNS", str(_ATHENA / "runs")))


def read_explanations(run_dir: Path) -> dict[int, dict]:
    """idx -> {z_inner, doc_id, read_token} for every 'ok' record in the prior run."""
    out = {}
    p = run_dir / "records.jsonl"
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("status") != "ok":
            continue
        zi = (r.get("z_inner") or "").strip()
        if not zi:
            continue
        out[int(r["idx"])] = dict(z_inner=zi, doc_id=r.get("doc_id", str(r["idx"])),
                                  read_token=r.get("read_token", ""))
    return out


def load_cache(path: Path) -> dict[int, dict]:
    out = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                r = json.loads(line)
                out[int(r["idx"])] = r
            except json.JSONDecodeError:
                pass
    return out


def segment_one(idx: int, meta: dict) -> dict:
    """One LLM segmentation call (two layers). Returns a cache-ready dict (or an error record)."""
    z = meta["z_inner"]
    try:
        res = seg.segment_audit(z)
        atoms = [dict(text=a["text"], type=a["type"], subtype=a.get("subtype", ""),
                      start=a["start"], end=a["end"], hedged=a.get("hedged", []) or [],
                      assertion=a.get("assertion", ""), role=a.get("role", ""),
                      regex_confirmed=a.get("regex_confirmed"))
                 for a in res["atoms"]]
        ents = [dict(text=e["text"], subtype=e["subtype"], start=e["start"], end=e["end"],
                     spans=e["spans"], n_occ=e["n_occ"], parent=e["parent"])
                for e in res.get("entities", [])]
        n_content = sum(1 for a in atoms if a["type"] in CONTENT)
        return dict(idx=idx, doc_id=meta["doc_id"], read_token=meta["read_token"],
                    z_inner=z, model=res.get("model", ""), coverage=res.get("coverage", {}),
                    n_atoms=len(atoms), n_content=n_content, n_connective=len(atoms) - n_content,
                    n_entity=len(ents), atoms=atoms, entities=ents, error=None)
    except Exception as e:  # tolerate API/parse failure; keep the batch alive
        return dict(idx=idx, doc_id=meta["doc_id"], read_token=meta["read_token"],
                    z_inner=z, model="", coverage={}, n_atoms=0, n_content=0,
                    n_connective=0, n_entity=0, atoms=[], entities=[],
                    error=f"{type(e).__name__}: {e}")


def append_line(path: Path, rec: dict):
    with _write_lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def print_breakdown(rec: dict):
    """Human-readable per-activation view for the CP1 sanity check (two layers)."""
    print("\n" + "=" * 92)
    cov = rec.get("coverage", {})
    covs = (f"coverage {cov.get('base_frac', 0) * 100:.0f}% - content "
            f"{cov.get('content_frac', 0) * 100:.0f}%") if cov else "(no coverage)"
    print(f"idx {rec['idx']}  read_token={rec['read_token']!r}  {covs}  | "
          f"{rec['n_content']} content + {rec['n_connective']} connective + "
          f"{rec.get('n_entity', 0)} entity")
    if rec.get("error"):
        print("  ERROR:", rec["error"]); return
    print("- AV explanation ".ljust(92, "-"))
    print(textwrap.indent(textwrap.fill(rec["z_inner"], 88), "  "))
    print("- LAYER 1: partition claims (theme / detail) with role ".ljust(92, "-"))
    for a in rec["atoms"]:
        if a["type"] not in CONTENT:
            continue
        hed = f"  [hedged: {', '.join(a['hedged'])}]" if a["hedged"] else ""
        rx = ""
        if a["type"] == "detail":
            rx = "  regex OK" if a["regex_confirmed"] else "  llm-only"
        txt = a["text"].replace("\n", " ")
        txt = (txt[:66] + "...") if len(txt) > 66 else txt
        print(f"  [{a['type']}:{a['subtype']} - {a.get('role', '')}]{hed}{rx}")
        print(f"      {txt!r}")
        if a["assertion"]:
            print(f"      -> {a['assertion']}")
    print("- LAYER 2: entity overlay (overlaps layer 1) ".ljust(92, "-"))
    ents = rec.get("entities", [])
    if not ents:
        print("  (no named entities)")
    for e in ents:
        occ = f"  x{e['n_occ']}" if e.get("n_occ", 1) > 1 else ""
        pin = ""
        p = e.get("parent")
        if p is not None and p < len(rec["atoms"]):
            pt = rec["atoms"][p]["text"].replace("\n", " ").strip()
            pt = (pt[:38] + "...") if len(pt) > 38 else pt
            pin = f"   (in {rec['atoms'][p]['type']}: {pt!r})"
        print(f"  <{e['subtype']}> {e['text']!r}{occ}{pin}")
    ncon = rec["n_connective"]
    if ncon:
        print(f"  ({ncon} connective/glue spans, kept for full coverage, not scored)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("key")
    ap.add_argument("--reuse-run", required=True, help="prior run id whose explanations to segment")
    ap.add_argument("--runs-dir", default=None)
    ap.add_argument("--out", default=None, help="cache path (default <athena>/segments/<key>.jsonl)")
    ap.add_argument("--only", default="", help="comma-separated idx list (subset)")
    ap.add_argument("--limit", type=int, default=0, help="cap number of activations (0=all)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--verbose", action="store_true", help="force per-activation breakdown")
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir) if args.runs_dir else runs_dir_default()
    run_dir = runs_dir / args.reuse_run
    if not (run_dir / "records.jsonl").exists():
        print(f"no records.jsonl at {run_dir}", file=sys.stderr); return 2

    seg_dir = _ATHENA / "segments"
    seg_dir.mkdir(exist_ok=True)
    out = Path(args.out) if args.out else seg_dir / f"{args.key}.jsonl"

    expl = read_explanations(run_dir)
    cache = load_cache(out)
    all_idx = sorted(expl)
    if args.only:
        want = [int(x) for x in args.only.split(",") if x.strip() != ""]
    else:
        want = all_idx[:args.limit] if args.limit else all_idx
    todo = [i for i in want if i not in cache or cache[i].get("error")]

    print(f"[segment] key={args.key} reuse-run={args.reuse_run}")
    print(f"[segment] {len(expl)} explanations available; {len(want)} selected; "
          f"{len(want) - len(todo)} already cached; {len(todo)} to segment -> {out}")
    if not todo:
        print("[segment] nothing to do (all selected idx cached).")
    verbose = args.verbose or len(want) <= 12
    t0 = time.time()
    done = 0
    results = {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(segment_one, i, expl[i]): i for i in todo}
        for fut in as_completed(futs):
            rec = fut.result()
            append_line(out, rec)
            results[rec["idx"]] = rec
            done += 1
            if rec.get("error"):
                print(f"  [{done}/{len(todo)}] idx {rec['idx']} ERROR {rec['error']}", flush=True)
            elif not verbose:
                if done % 25 == 0 or done == len(todo):
                    el = time.time() - t0
                    print(f"  [{done}/{len(todo)}] {el:.0f}s "
                          f"({el / max(done,1):.1f}s/act)", flush=True)

    # gather the full selected set (cached + freshly done) for reporting
    final = load_cache(out)
    sel = [final[i] for i in want if i in final]
    ok = [r for r in sel if not r.get("error")]
    if verbose:
        for r in sel:
            print_breakdown(r)

    # distribution summary
    if ok:
        import statistics as st
        nc = [r["n_content"] for r in ok]
        na = [r["n_atoms"] for r in ok]
        covb = [r["coverage"].get("base_frac", 0) for r in ok if r.get("coverage")]
        covc = [r["coverage"].get("content_frac", 0) for r in ok if r.get("coverage")]
        tcount = {}
        for r in ok:
            for a in r["atoms"]:
                if a["type"] in CONTENT:
                    tcount[a["type"]] = tcount.get(a["type"], 0) + 1
        print("\n" + "#" * 92)
        print(f"# SEGMENTATION SUMMARY  ({len(ok)} ok, {len(sel) - len(ok)} errors)")
        print(f"#   atoms/activation:   total median {st.median(na):.0f} "
              f"(min {min(na)}, max {max(na)})")
        print(f"#   content/activation: median {st.median(nc):.0f} "
              f"(min {min(nc)}, max {max(nc)})  total content atoms {sum(nc)}")
        if covb:
            print(f"#   coverage: mean {sum(covb)/len(covb)*100:.1f}%   "
                  f"content: mean {sum(covc)/len(covc)*100:.1f}%")
        tot = sum(tcount.values()) or 1
        print("#   partition-type mix: " + "  ".join(
            f"{t} {tcount.get(t,0)} ({tcount.get(t,0)/tot*100:.0f}%)"
            for t in ("theme", "detail")))
        ne = [r.get("n_entity", 0) for r in ok]
        esub, roles = {}, {}
        for r in ok:
            for e in r.get("entities", []):
                esub[e["subtype"]] = esub.get(e["subtype"], 0) + 1
            for a in r["atoms"]:
                if a["type"] in CONTENT:
                    roles[a.get("role", "")] = roles.get(a.get("role", ""), 0) + 1
        print(f"#   entities/activation: median {st.median(ne):.0f} (total {sum(ne)})")
        if esub:
            print("#   entity subtypes: " + "  ".join(
                f"{k} {v}" for k, v in sorted(esub.items(), key=lambda x: -x[1])))
        rtot = sum(roles.values()) or 1
        print("#   claim roles: " + "  ".join(
            f"{k} {roles.get(k,0)} ({roles.get(k,0)/rtot*100:.0f}%)"
            for k in ("describes", "quotes", "predicts")))
        print("#" * 92)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
