"""Inter-model label agreement on the taxonomy caches (sonnet vs haiku).

Joins the two models' cached classifications by (run, idx, claim position) and
reports raw agreement + Cohen's kappa per label axis, with confusion matrices.
Pure read-only offline analysis: no API calls, stdlib only.

  python3 repl/taxonomy_agreement.py                 # all atom runs, sonnet vs haiku
  python3 repl/taxonomy_agreement.py --runs qwen7b-cp6-fineweb-atoms
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import claim_taxonomy as ct
import taxonomy_pass as tp

BINARY = {"stated": "inferable-from-text", "inferable": "inferable-from-text",
          "beyond": "not-inferable", "contradicted": "not-inferable"}


def kappa(pairs: list[tuple[str, str]]) -> float | None:
    """Cohen's kappa, hand-rolled (no scipy/sklearn)."""
    n = len(pairs)
    if not n:
        return None
    po = sum(a == b for a, b in pairs) / n
    ma = Counter(a for a, _ in pairs)
    mb = Counter(b for _, b in pairs)
    pe = sum(ma[k] * mb.get(k, 0) for k in ma) / (n * n)
    if pe >= 1.0:
        return None                       # degenerate: a single shared label
    return (po - pe) / (1 - pe)


def confusion(pairs: list[tuple[str, str]], labels: tuple[str, ...]) -> str:
    lab = [l for l in labels if any(l in p for p in pairs)] or list(labels)
    cnt = Counter(pairs)
    w = max(6, max(len(l) for l in lab)) + 1
    head = " " * w + "".join(f"{l[:w-1]:>{w}}" for l in lab) + "   (rows=A, cols=B)"
    rows = [head]
    for a in lab:
        rows.append(f"{a:>{w}}" + "".join(f"{cnt.get((a, b), 0):>{w}}" for b in lab))
    return "\n".join(rows)


def collect(rid: str, model_a: str, model_b: str) -> dict[str, list[tuple[str, str]]]:
    """Position-joined label pairs for every claim classified by both models."""
    types = {}                            # idx -> [layer-1 type per claim position]
    for rec in tp.load_records(rid):
        if rec.get("snippet_results"):
            types[rec["idx"]] = [s.get("type") for s in rec["snippet_results"]]
    out: dict[str, list[tuple[str, str]]] = {
        "axis": [], "support": [], "support_binary": [],
        "taxo_theme": [], "taxo_detail": []}
    n_act = 0
    for idx, tlist in sorted(types.items()):
        pa = ct.cache_path(tp.TAXO, rid, idx, model_a)
        pb = ct.cache_path(tp.TAXO, rid, idx, model_b)
        if not (pa.exists() and pb.exists()):
            continue
        ca = {c["i"]: c for c in json.loads(pa.read_text())["claims"]}
        cb = {c["i"]: c for c in json.loads(pb.read_text())["claims"]}
        n_act += 1
        for i, typ in enumerate(tlist, start=1):
            a, b = ca.get(i), cb.get(i)
            if not a or not b or a.get("missing") or b.get("missing"):
                continue
            out["axis"].append((a["axis"], b["axis"]))
            out["support"].append((a["support"], b["support"]))
            out["support_binary"].append((BINARY[a["support"]], BINARY[b["support"]]))
            if typ == "theme":
                out["taxo_theme"].append((a["taxo"], b["taxo"]))
            elif typ == "detail":
                out["taxo_detail"].append((a["taxo"], b["taxo"]))
    out["_n_act"] = n_act  # type: ignore[assignment]
    return out


def report(name: str, pairs: list[tuple[str, str]], labels: tuple[str, ...],
           matrix: bool) -> None:
    if not pairs:
        print(f"  {name:<16} (no overlapping claims)")
        return
    agree = sum(a == b for a, b in pairs)
    k = kappa(pairs)
    ks = f"{k:+.3f}" if k is not None else "  n/a"
    print(f"  {name:<16} n={len(pairs):<5} agree={agree/len(pairs):6.1%}  kappa={ks}")
    if matrix:
        print("    " + confusion(pairs, labels).replace("\n", "\n    "))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="*", default=None,
                    help="run ids (default: every non-mock atom run)")
    ap.add_argument("--model-a", choices=sorted(ct.MODELS), default="sonnet")
    ap.add_argument("--model-b", choices=sorted(ct.MODELS), default="haiku")
    ap.add_argument("--matrices", action="store_true",
                    help="print confusion matrices (axis + support 4-way)")
    args = ap.parse_args()

    axes = [
        ("axis", ct.AXES, True),
        ("support 4-way", ct.SUPPORTS, True),
        ("support binary", ("inferable-from-text", "not-inferable"), False),
        ("taxo (theme)", ct.THEME_TAXO, False),
        ("taxo (detail)", ct.DETAIL_TAXO, False),
    ]
    key = {"axis": "axis", "support 4-way": "support",
           "support binary": "support_binary", "taxo (theme)": "taxo_theme",
           "taxo (detail)": "taxo_detail"}
    pooled: dict[str, list[tuple[str, str]]] = {k: [] for k in key.values()}
    tot_act = 0
    for rid in (args.runs or tp.atom_runs()):
        got = collect(rid, args.model_a, args.model_b)
        n_act = got.pop("_n_act")
        tot_act += n_act
        print(f"\n[{rid}] {n_act} activations classified by both "
              f"{args.model_a} and {args.model_b}")
        if not n_act:
            continue
        for name, labels, mat in axes:
            report(name, got[key[name]], labels, args.matrices and mat)
        for k, v in got.items():
            pooled[k].extend(v)

    print(f"\n=== pooled ({tot_act} activations) ===")
    for name, labels, mat in axes:
        report(name, pooled[key[name]], labels, args.matrices and mat)
    return 0


if __name__ == "__main__":
    sys.exit(main())
