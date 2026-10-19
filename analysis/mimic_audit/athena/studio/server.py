"""Claim-Ablation Studio — a read-only viewer for claim-ablation runs.

Mirrors NLA Studio's look and intent, but is deliberately stdlib-only (no FastAPI /
uvicorn) so it runs identically on a laptop and on the GPU box with zero extra deps.
It serves the single-page UI and a small JSON API over the run artifacts written by
run_claim_ablation.py. It never loads a model and never mutates a run; it just reads
the runs directory and lets the page poll for live updates.

Run:
  python studio/server.py                 # serves http://127.0.0.1:8011
  CLAIM_ABLATION_RUNS=/data/.../runs python studio/server.py --port 8011

Routes:
  GET /                          the studio page
  GET /api/runs                  list runs (lightweight run.json fields)
  GET /api/run/<id>              {run: run.json, summary: summary.json}
  GET /api/run/<id>/records      all records (records.jsonl parsed)
  GET /api/run/<id>/record/<i>   one record
  GET /api/run/<id>/vectors/<i>  per-activation vectors (downsampled; ?full=1 for raw)
  GET /api/taxonomy/runs         atom runs + per-model classification-cache counts
  GET /api/taxonomy/item/<id>[/<i>]  claims of one activation + cached classification
  POST /api/taxonomy/classify    classify ONE activation's claims (the single write
                                 path: an Anthropic call cached under taxonomy/;
                                 runs are still never mutated, and there is no
                                 batch endpoint by design)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import numpy as np

HERE = Path(__file__).resolve().parent
ATHENA = HERE.parent
WEB = HERE / "web"
SAFE_ID = re.compile(r"^[A-Za-z0-9._-]+$")

sys.path.insert(0, str(ATHENA / "repl"))
import claim_taxonomy as ct                     # stdlib-only at import (anthropic is lazy)
import cycle_studio as cyc                       # cycle-consistency data layer (read-only)


def runs_dir() -> Path:
    return Path(os.environ.get("CLAIM_ABLATION_RUNS", str(ATHENA / "runs")))


def taxonomy_dir() -> Path:
    return Path(os.environ.get("CLAIM_ABLATION_TAXONOMY", str(ATHENA / "taxonomy")))


def _load_json(p: Path, default=None):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return default


def _records(rundir: Path) -> list[dict]:
    out, p = [], rundir / "records.jsonl"
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass                                # tolerate a partially-written tail line
    return out


def list_runs() -> list[dict]:
    rd = runs_dir()
    if not rd.exists():
        return []
    runs = []
    for d in rd.iterdir():
        rj = _load_json(d / "run.json")
        if not rj:
            continue
        keep = ("run_id", "key", "model_label", "family", "status", "phase", "mock",
                "claim_mode", "n_target", "n_done", "n_ok", "n_skip", "n_cjk", "grounded_rate",
                "created_at", "updated_ts", "started_ts", "elapsed_s", "eta_s",
                "dataset", "smoke", "snippets_located", "snippets_grounded",
                "last_text", "error")
        runs.append({k: rj.get(k) for k in keep})
    runs.sort(key=lambda r: r.get("updated_ts") or 0, reverse=True)
    return runs


def _spark(arr, buckets=256):
    """Downsample |arr| to a fixed number of buckets (bucket-mean of abs) for a
    sparkline, plus summary stats. Keeps vector payloads small for the UI."""
    a = np.asarray(arr, dtype=np.float64)
    flat = np.abs(a.ravel())
    n = flat.size
    if n == 0:
        return dict(norm=0.0, min=0.0, max=0.0, mean=0.0, spark=[])
    if n > buckets:
        idx = (np.arange(buckets + 1) * n / buckets).astype(int)
        spark = [float(flat[idx[i]:max(idx[i] + 1, idx[i + 1])].mean())
                 for i in range(buckets)]
    else:
        spark = [float(x) for x in flat]
    return dict(norm=float(np.linalg.norm(a.ravel())), min=float(a.min()),
                max=float(a.max()), mean=float(a.mean()),
                spark=[round(x, 5) for x in spark])


def vectors_payload(rundir: Path, idx: int, full: bool) -> dict | None:
    p = rundir / "vectors" / f"{idx}.npz"
    if not p.exists():
        return None
    z = np.load(p)
    out = dict(idx=idx, d_model=int(z["h"].shape[0]),
               marker_pos=int(z["marker_pos"][0]) if "marker_pos" in z else None,
               cont_token_start=int(z["cont_token_start"][0]) if "cont_token_start" in z else None,
               ranges=z["ranges"].tolist() if "ranges" in z else [],
               null=z["null"].tolist() if "null" in z else [],
               tok_logp_h=[round(float(x), 4) for x in z["tok_logp_h"]] if "tok_logp_h" in z else [],
               tok_logp_ablate=[[round(float(x), 4) for x in row]
                                for row in z["tok_logp_ablate"]] if "tok_logp_ablate" in z else [],
               token_ids=z["token_ids"].tolist() if "token_ids" in z else [])
    out["h"] = _spark(z["h"])
    out["ar_full"] = _spark(z["ar_full"]) if "ar_full" in z else None
    out["deltas"] = [_spark(z["deltas"][i]) for i in range(z["deltas"].shape[0])] \
        if "deltas" in z else []
    if full:
        out["h_full"] = z["h"].tolist()
        out["deltas_full"] = z["deltas"].tolist() if "deltas" in z else []
        out["ar_full_full"] = z["ar_full"].tolist() if "ar_full" in z else []
        out["ar_minus_full"] = z["ar_minus"].tolist() if "ar_minus" in z else []
    return out


# ---------------------------------------------------------------- taxonomy lab
_REC_CACHE: dict[str, tuple[float, list]] = {}


def _records_cached(rundir: Path) -> list[dict]:
    """records.jsonl parsed once per mtime - the taxonomy views poll every few
    seconds during the mass pass and the runs themselves never change."""
    p = rundir / "records.jsonl"
    try:
        mt = p.stat().st_mtime
    except FileNotFoundError:
        return []
    hit = _REC_CACHE.get(str(p))
    if hit and hit[0] == mt:
        return hit[1]
    recs = _records(rundir)
    _REC_CACHE[str(p)] = (mt, recs)
    return recs


def _is_atom_run(r: dict) -> bool:
    return r.get("claim_mode") == "atoms" or str(r.get("run_id", "")).endswith("-atoms")


def taxonomy_runs_payload() -> list[dict]:
    out = []
    for r in list_runs():
        if not _is_atom_run(r) or r.get("mock"):
            continue
        d = taxonomy_dir() / str(r.get("run_id"))
        cached = {mk: (len(list(d.glob(f"*.{mk}.json"))) if d.is_dir() else 0)
                  for mk in ct.MODELS}
        out.append({"run_id": r.get("run_id"), "key": r.get("key"),
                    "model_label": r.get("model_label"), "status": r.get("status"),
                    "n_done": r.get("n_done"), "cached": cached})
    return out


def _cached_indices(run_id: str, model_key: str) -> list[int]:
    d = taxonomy_dir() / run_id
    if not d.is_dir():
        return []
    out = []
    for p in d.glob(f"*.{model_key}.json"):
        try:
            out.append(int(p.name.split(".")[0]))
        except ValueError:
            pass
    return sorted(out)


def taxonomy_item(rundir: Path, run_id: str, idx: int | None, model_key: str) -> dict:
    run = _load_json(rundir / "run.json", {})
    recs = _records_cached(rundir)
    indices = [r["idx"] for r in recs if r.get("snippet_results")]
    base = {"run_id": run_id, "key": run.get("key"), "model_label": run.get("model_label"),
            "model": model_key, "models": sorted(ct.MODELS),
            "indices": indices, "cached": _cached_indices(run_id, model_key)}
    if idx is None:
        return base
    rec = next((r for r in recs if r.get("idx") == idx), None)
    if rec is None:
        raise KeyError(f"no record {idx}")
    seg_row = ct.load_segments(run.get("key") or "").get(idx)
    ents = ct.entity_classes(rec, seg_row)
    claims = []
    for k, s in enumerate(rec.get("snippet_results") or []):
        claims.append({"i": k + 1, "text": s.get("text"), "type": s.get("type"),
                       "subtype": s.get("subtype"), "role": s.get("role"),
                       "hedged": s.get("hedged"), "assertion": s.get("assertion"),
                       "kw_bucket": ct.kw_bucket(s), "verdict": s.get("verdict"),
                       "located": s.get("located"), "entities": ents[k]})
    cls = _load_json(ct.cache_path(taxonomy_dir(), run_id, idx, model_key))
    base.update({"idx": idx, "doc_id": rec.get("doc_id"), "position": rec.get("position"),
                 "read_token": rec.get("read_token"),
                 "context_left": rec.get("context_left"),
                 "context_right": rec.get("context_right"),
                 "explanation": rec.get("z_inner"), "claims": claims,
                 "classification": cls})
    return base


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return max(0.0, c - h), min(1.0, c + h)


def _vblock(rows: list[dict]) -> dict:
    """Verdict roll-up for one label's claims. Rates over LOCATED claims, same
    denominator convention as summarize()'s by_type / by_bucket blocks."""
    loc = [r for r in rows if r.get("located")]
    g = sum(r["verdict"] == "grounded" for r in loc)
    e = sum(r["verdict"] == "entangled" for r in loc)
    c = sum(r["verdict"] == "confabulated" for r in loc)
    lo, hi = _wilson(g, len(loc))
    return {"n": len(rows), "located": len(loc), "grounded": g, "entangled": e,
            "confabulated": c,
            "grounded_rate": (g / len(loc)) if loc else None, "lo": lo, "hi": hi}


def taxonomy_agg(rundir: Path, run_id: str, model_key: str) -> dict:
    """Aggregate the cached per-activation classifications into verdict slices.
    Pure read: recomputed per poll so the tab fills in live during the mass pass."""
    run = _load_json(rundir / "run.json", {})
    recs = _records_cached(rundir)
    segs = ct.load_segments(run.get("key") or "")
    rows: list[dict] = []                     # one per labeled claim
    n_recs = n_class = n_claims = 0
    for rec in recs:
        snips = rec.get("snippet_results") or []
        if not snips:
            continue
        n_recs += 1
        n_claims += len(snips)
        cls = _load_json(ct.cache_path(taxonomy_dir(), run_id, rec["idx"], model_key))
        if not cls:
            continue
        n_class += 1
        ents = ct.entity_classes(rec, segs.get(rec["idx"]))
        by_i = {c.get("i"): c for c in (cls.get("claims") or [])}
        for k, s in enumerate(snips):
            L = by_i.get(k + 1)
            if not L or L.get("missing"):
                continue
            rows.append({"located": bool(s.get("located")),
                         "verdict": s.get("verdict"), "type": s.get("type"),
                         "kw": ct.kw_bucket(s), "taxo": L.get("taxo"),
                         "axis": L.get("axis"), "support": L.get("support"),
                         "entities": ents[k]})

    def by(key, labels, sub=None):
        pool = [r for r in rows if sub is None or sub(r)]
        return {lb: _vblock([r for r in pool if r.get(key) == lb]) for lb in labels}

    supports = list(ct.SUPPORTS)
    cross = {v: {s: 0 for s in supports}
             for v in ("grounded", "entangled", "confabulated")}
    for r in rows:
        if r["located"] and r["verdict"] in cross and r.get("support") in supports:
            cross[r["verdict"]][r["support"]] += 1
    axes = list(ct.AXES)
    agree = [r for r in rows if r.get("axis") and r.get("kw")]
    same = sum(r["axis"] == r["kw"] for r in agree)
    ent_labels = ("person", "place", "org", "title", "ticker", "other")
    by_entity = {lb: _vblock([r for r in rows if lb in (r.get("entities") or [])])
                 for lb in ent_labels}
    by_entity["none"] = _vblock([r for r in rows if not r.get("entities")])
    return {
        "run_id": run_id, "model": model_key, "prompt_version": ct.PROMPT_VERSION,
        "n_records": n_recs, "n_classified": n_class,
        "n_claims_total": n_claims, "n_claims_labeled": len(rows),
        "overall": _vblock(rows),
        "by_axis": by("axis", axes),
        "by_kw": by("kw", axes),
        "kw_axis_agreement": {"n": len(agree), "same": same,
                              "rate": (same / len(agree)) if agree else None},
        "by_support": by("support", supports),
        "by_support_binary": {
            "inferable-from-text": _vblock([r for r in rows
                                            if r.get("support") in ("stated", "inferable")]),
            "not-inferable": _vblock([r for r in rows
                                      if r.get("support") in ("beyond", "contradicted")])},
        "cross": cross,
        "by_taxo_theme": by("taxo", ct.THEME_TAXO, sub=lambda r: r["type"] == "theme"),
        "by_taxo_detail": by("taxo", ct.DETAIL_TAXO, sub=lambda r: r["type"] == "detail"),
        "by_entity": by_entity,
    }


class Handler(BaseHTTPRequestHandler):
    server_version = "ClaimAblationStudio/1.0"

    def log_message(self, *a):                  # quiet; uncomment to debug
        pass

    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False)
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype + ("; charset=utf-8"
                         if ctype.startswith(("text", "application/json")) else ""))
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _err(self, code, msg):
        self._send(code, {"error": msg})

    def _rundir(self, rid):
        if not SAFE_ID.match(rid or ""):
            return None
        d = runs_dir() / rid
        return d if (d / "run.json").exists() else None

    def do_GET(self):
        u = urlparse(self.path)
        path = u.path
        parts = [p for p in path.split("/") if p]
        try:
            if path == "/" or path == "/index.html":
                return self._serve_file(WEB / "index.html", "text/html")
            if path == "/api/health":
                return self._send(200, {"ok": True, "runs_dir": str(runs_dir())})
            if path == "/api/runs":
                return self._send(200, {"runs": list_runs(), "runs_dir": str(runs_dir())})
            if path == "/api/taxonomy/runs":
                return self._send(200, {"runs": taxonomy_runs_payload()})

            # ── cycle-consistency (read-only) ──────────────────────────────
            if path == "/cycle" or path == "/cycle.html":
                return self._serve_file(WEB / "cycle.html", "text/html")
            if path == "/api/cycle/runs":
                return self._send(200, {"runs": cyc.list_cycle_runs()})
            if len(parts) >= 4 and parts[0] == "api" and parts[1] == "cycle" \
                    and parts[2] in ("report", "claims"):
                key = parts[3]
                if not SAFE_ID.match(key):
                    return self._err(400, f"bad key {key!r}")
                data = (cyc.cycle_report(key) if parts[2] == "report"
                        else cyc.cycle_claims(key))
                return self._send(200, data) if data else self._err(404, f"no cycle run {key!r}")
            if len(parts) == 6 and parts[0] == "api" and parts[1] == "cycle" \
                    and parts[2] == "claim":
                key = parts[3]
                if not SAFE_ID.match(key):
                    return self._err(400, f"bad key {key!r}")
                try:
                    idx, ci = int(parts[4]), int(parts[5])
                except ValueError:
                    return self._err(400, "idx/ci must be integers")
                data = cyc.cycle_claim(key, idx, ci)
                return self._send(200, data) if data else self._err(404, "no such claim")

            # /api/taxonomy/{item|agg}/<id>[/<i>]?model=sonnet|haiku
            if len(parts) >= 4 and parts[0] == "api" and parts[1] == "taxonomy" \
                    and parts[2] in ("item", "agg"):
                rid = parts[3]
                rundir = self._rundir(rid)
                if not rundir:
                    return self._err(404, f"no run {rid!r}")
                model = parse_qs(u.query).get("model", ["sonnet"])[0]
                if model not in ct.MODELS:
                    return self._err(400, f"unknown model {model!r}")
                if parts[2] == "agg":
                    return self._send(200, taxonomy_agg(rundir, rid, model))
                idx = int(parts[4]) if len(parts) >= 5 else None
                try:
                    return self._send(200, taxonomy_item(rundir, rid, idx, model))
                except KeyError as e:
                    return self._err(404, str(e))

            # /api/run/<id>[/...]
            if len(parts) >= 3 and parts[0] == "api" and parts[1] == "run":
                rid = parts[2]
                rundir = self._rundir(rid)
                if not rundir:
                    return self._err(404, f"no run {rid!r}")
                if len(parts) == 3:
                    return self._send(200, {
                        "run": _load_json(rundir / "run.json", {}),
                        "summary": _load_json(rundir / "summary.json", {})})
                if parts[3] == "records":
                    return self._send(200, {"records": _records(rundir)})
                if parts[3] == "record" and len(parts) >= 5:
                    i = int(parts[4])
                    rec = next((r for r in _records(rundir) if r.get("idx") == i), None)
                    return self._send(200, rec) if rec else self._err(404, "no record")
                if parts[3] == "vectors" and len(parts) >= 5:
                    full = parse_qs(u.query).get("full", ["0"])[0] == "1"
                    v = vectors_payload(rundir, int(parts[4]), full)
                    return self._send(200, v) if v else self._err(404, "no vectors")
                return self._err(404, "unknown run route")

            # static assets under web/
            cand = (WEB / path.lstrip("/")).resolve()
            if cand.is_file() and str(cand).startswith(str(WEB)):
                ctype = ("text/css" if cand.suffix == ".css" else
                         "application/javascript" if cand.suffix == ".js" else
                         "text/html" if cand.suffix == ".html" else
                         "application/octet-stream")
                return self._serve_file(cand, ctype)
            return self._err(404, f"not found: {path}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            return self._err(500, str(e))

    def do_POST(self):
        u = urlparse(self.path)
        try:
            if u.path != "/api/taxonomy/classify":
                return self._err(404, f"not found: {u.path}")
            n = int(self.headers.get("Content-Length") or 0)
            if not (0 < n <= 4096):
                return self._err(400, "bad body")
            body = json.loads(self.rfile.read(n).decode("utf-8"))
            rid = str(body.get("run_id") or "")
            model = str(body.get("model") or "sonnet")
            idx = body.get("idx")
            rundir = self._rundir(rid)
            if not rundir:
                return self._err(404, f"no run {rid!r}")
            if model not in ct.MODELS:
                return self._err(400, f"unknown model {model!r}")
            if not isinstance(idx, int):
                return self._err(400, "idx must be an integer")
            rec = next((r for r in _records(rundir) if r.get("idx") == idx), None)
            if rec is None or not rec.get("snippet_results"):
                return self._err(404, f"no scored claims for idx {idx}")
            # ONE activation per call; the result is cached so a re-click is free.
            cp = ct.cache_path(taxonomy_dir(), rid, idx, model)
            if not cp.exists():
                res = ct.classify_activation(rec, model, run_id=rid)
                ct.write_cache(cp, res)
            return self._send(200, taxonomy_item(rundir, rid, idx, model))
        except Exception as e:
            import traceback
            traceback.print_exc()
            return self._err(500, str(e))

    def _serve_file(self, p: Path, ctype):
        if not p.is_file():
            return self._err(404, f"missing {p.name}")
        self._send(200, p.read_bytes(), ctype)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8011)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--runs-dir", default=None)
    args = ap.parse_args()
    if args.runs_dir:
        os.environ["CLAIM_ABLATION_RUNS"] = args.runs_dir
    rd = runs_dir()
    print(f"[studio] runs dir: {rd}  ({'exists' if rd.exists() else 'MISSING'})")
    print(f"[studio] serving http://{args.host}:{args.port}")
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[studio] bye")
    return 0


if __name__ == "__main__":
    sys.exit(main())
