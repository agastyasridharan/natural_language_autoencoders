"""build_static.py — freeze the Claim-Ablation Studio into a static bundle for GitHub Pages.

The live studio (`server.py`) serves a single-page UI plus a small JSON API read off a
`runs/` directory, and the page polls it for live updates. GitHub Pages serves static
files only, so this script:

  1. pre-renders every API response the page can request into static JSON files that
     mirror the live routes (using server.py's own functions, so the payloads are
     identical to what the live API returns), and
  2. copies the page with its `api()` fetch layer re-pointed at those static files and
     the live-polling intervals disabled.

Absolute filesystem paths (e.g. the cluster `gold_dir`) are redacted, since the output
is published to a public, indexable site.

  python studio/build_static.py [OUT_DIR]     # default OUT_DIR = the nla-viewers clone

Output tree:
  OUT_DIR/
    index.html
    data/runs.json                     # {runs:[...], runs_dir:"(published snapshot)"}
    data/run/<id>/meta.json            # {run:{...}, summary:{...}}
    data/run/<id>/records.json         # {records:[...]}
    data/run/<id>/vectors.json         # {"<idx>": payload, ...}   (only if vectors exist)
"""
from __future__ import annotations

import importlib.util
import json
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = Path("/Users/agastyasridharan/ruhr research/nla-viewers/nla-claim-ablation")

# Load server.py as a module so we reuse its exact payload builders.
_spec = importlib.util.spec_from_file_location("ca_server", HERE / "server.py")
server = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(server)  # type: ignore[union-attr]

# Redact absolute paths (cluster + laptop) so nothing internal reaches the public site.
_PATH_RE = re.compile(r"/(?:data|home|Users|mnt|scratch|srv)/[^\s\"'\\,)}\]]*")


def redact(obj):
    if isinstance(obj, str):
        return _PATH_RE.sub("(local path redacted)", obj)
    if isinstance(obj, list):
        return [redact(x) for x in obj]
    if isinstance(obj, dict):
        return {k: redact(v) for k, v in obj.items()}
    return obj


def write_json(path: Path, obj) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(redact(obj), ensure_ascii=False)
    path.write_text(data, encoding="utf-8")
    return len(data.encode("utf-8"))


def build_page(src: Path) -> str:
    """Copy the studio page, re-pointing api() at static files and killing live polling."""
    html = src.read_text(encoding="utf-8")

    old_api = ("async function api(path){ const r=await fetch(path); "
               "if(!r.ok) throw new Error((await r.json()).error||r.status); return r.json(); }")
    assert old_api in html, "api() shape changed in index.html; update build_static.py"
    new_api = (
        "const __CACHE={};\n"
        "async function __getJSON(f){\n"
        "  if(Object.prototype.hasOwnProperty.call(__CACHE,f)){ if(__CACHE[f]===null) throw new Error('404 '+f); return __CACHE[f]; }\n"
        "  let r; try{ r=await fetch(f); }catch(e){ __CACHE[f]=null; throw e; }\n"
        "  if(!r.ok){ __CACHE[f]=null; throw new Error(r.status+' '+f); }\n"
        "  const j=await r.json(); __CACHE[f]=j; return j;\n"
        "}\n"
        "// static snapshot: map the live JSON API onto pre-rendered files under data/\n"
        "async function api(path){\n"
        "  const p=path.split('?')[0];\n"
        "  if(p==='/api/runs') return __getJSON('data/runs.json');\n"
        "  const m=p.match(/^\\/api\\/run\\/([^/]+)(?:\\/(.*))?$/);\n"
        "  if(m){\n"
        "    const id=m[1], rest=m[2]||'';\n"
        "    if(rest==='') return __getJSON('data/run/'+id+'/meta.json');\n"
        "    if(rest==='records') return __getJSON('data/run/'+id+'/records.json');\n"
        "    let mm;\n"
        "    if(mm=rest.match(/^record\\/(\\d+)$/)){\n"
        "      const recs=(await __getJSON('data/run/'+id+'/records.json')).records||[];\n"
        "      const rec=recs.find(r=>String(r.idx)===mm[1]);\n"
        "      if(rec) return rec; throw new Error('no record');\n"
        "    }\n"
        "    if(mm=rest.match(/^vectors\\/(\\d+)$/)){\n"
        "      let vs=null; try{ vs=await __getJSON('data/run/'+id+'/vectors.json'); }catch(e){ vs=null; }\n"
        "      const v = vs ? vs[mm[1]] : null;\n"
        "      if(v) return v; throw new Error('no vectors');\n"
        "    }\n"
        "  }\n"
        "  throw new Error('not found: '+path);\n"
        "}"
    )
    html = html.replace(old_api, new_api)

    # Disable every live-update poller (data is frozen; nothing to poll). Match
    # any `setInterval(refresh, <ms>)` generically so new poll sites can be added
    # to the live page without silently surviving into the static bundle. The
    # play-animation timer (`setInterval(()=>...)`) is left alone: it steps
    # through already-loaded data and works offline.
    n_pollers = len(re.findall(r"setInterval\(refresh,\s*\d+\)", html))
    assert n_pollers >= 1, "no refresh poller found in index.html; check build_static"
    html = re.sub(r"setInterval\(refresh,\s*\d+\)",
                  "null /* static snapshot: no live polling */", html)
    assert "setInterval(refresh" not in html, "a refresh poller survived; check the strings"

    # Topnav link to the cycle tab is absolute (/cycle.html) for the live server;
    # on the subpath Pages bundle it must be relative to sit beside index.html.
    assert 'href="/cycle.html"' in html, "cycle-tab topnav link shape changed in index.html"
    html = html.replace('href="/cycle.html"', 'href="cycle.html"')
    return html


# static replacement for cycle.html's api() fetch layer: map the live cycle
# routes onto the pre-rendered files under data/cycle/ (mirrors build_page).
_CYCLE_OLD_API = ('async function api(p){ const r = await fetch(p); '
                  'if(!r.ok) throw new Error((await r.json().catch(()=>({}))).error||r.status); '
                  'return r.json(); }')
_CYCLE_NEW_API = (
    "const __CCACHE={};\n"
    "async function __getJSON(f){\n"
    "  if(Object.prototype.hasOwnProperty.call(__CCACHE,f)){ if(__CCACHE[f]===null) throw new Error('404 '+f); return __CCACHE[f]; }\n"
    "  let r; try{ r=await fetch(f); }catch(e){ __CCACHE[f]=null; throw e; }\n"
    "  if(!r.ok){ __CCACHE[f]=null; throw new Error(r.status+' '+f); }\n"
    "  const j=await r.json(); __CCACHE[f]=j; return j;\n"
    "}\n"
    "// static snapshot: map the live cycle API onto pre-rendered files under data/cycle/\n"
    "async function api(p){\n"
    "  const q=p.split('?')[0]; let m;\n"
    "  if(q==='/api/cycle/runs') return __getJSON('data/cycle/runs.json');\n"
    "  if(m=q.match(/^\\/api\\/cycle\\/claims\\/(.+)$/)) return __getJSON('data/cycle/claims/'+m[1]+'.json');\n"
    "  if(m=q.match(/^\\/api\\/cycle\\/claim\\/([^/]+)\\/([^/]+)\\/([^/]+)$/)){\n"
    "    const all=await __getJSON('data/cycle/claim/'+m[1]+'.json');\n"
    "    const v=all[m[2]+'_'+m[3]];\n"
    "    if(v) return v; throw new Error('no claim');\n"
    "  }\n"
    "  throw new Error('not found: '+p);\n"
    "}"
)


def build_cycle_page(src: Path) -> str:
    """Freeze cycle.html: re-point api() at static data/cycle/ files and make the
    'Studio' back-link relative. cycle.html has no live pollers to disable."""
    html = src.read_text(encoding="utf-8")
    assert _CYCLE_OLD_API in html, "api() shape changed in cycle.html; update build_static.py"
    html = html.replace(_CYCLE_OLD_API, _CYCLE_NEW_API)
    assert '<a href="/">← Studio</a>' in html, "back-link shape changed in cycle.html"
    html = html.replace('<a href="/">← Studio</a>', '<a href="index.html">← Studio</a>')
    return html


def build_cycle_data(out: Path) -> int:
    """Pre-render every cycle route the page can request. Returns bytes written.
    Bundles all per-claim payloads for a model into one file keyed '<idx>_<ci>',
    mirroring the vectors.json pattern, so the static api() shim is a dict lookup."""
    cyc = server.cyc                       # cycle-consistency data layer (server binds it)
    runs = cyc.list_cycle_runs()
    if not runs:
        print("[build] no cycle runs found; skipping cycle tab", file=sys.stderr)
        return 0
    cdata = out / "data" / "cycle"
    total = write_json(cdata / "runs.json", {"runs": runs})
    print(f"[build] cycle: {len(runs)} model(s) -> {out}")
    for r in runs:
        key = r["key"]
        claims = cyc.cycle_claims(key) or {"claims": []}
        n = write_json(cdata / "claims" / f"{key}.json", claims)
        bundle = {}
        for row in claims.get("claims", []):
            idx, ci = row.get("idx"), row.get("ci")
            full = cyc.cycle_claim(key, idx, ci)
            if full is not None:
                bundle[f"{idx}_{ci}"] = full
        n += write_json(cdata / "claim" / f"{key}.json", bundle)
        total += n
        print(f"  {key:12} claims={len(bundle):4}  {n/1e6:6.1f}MB")
    page = build_cycle_page(HERE / "web" / "cycle.html")
    (out / "cycle.html").write_text(page, encoding="utf-8")
    return total


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT
    runs = server.list_runs()
    if not runs:
        print(f"[build] no runs found in {server.runs_dir()} — nothing to publish", file=sys.stderr)
        return 1

    data = out / "data"
    if data.exists():
        shutil.rmtree(data)

    # /api/runs (runs_dir scrubbed to a static label)
    total = write_json(data / "runs.json", {"runs": runs, "runs_dir": "(published snapshot)"})

    print(f"[build] {len(runs)} runs -> {out}")
    for r in runs:
        rid = r["run_id"]
        rd = server.runs_dir() / rid
        n = write_json(data / "run" / rid / "meta.json", {
            "run": server._load_json(rd / "run.json", {}),
            "summary": server._load_json(rd / "summary.json", {}),
        })
        recs = server._records(rd)
        n += write_json(data / "run" / rid / "records.json", {"records": recs})

        vdir = rd / "vectors"
        nvec = 0
        if vdir.exists():
            payloads = {}
            for p in sorted(vdir.glob("*.npz"), key=lambda p: int(p.stem)):
                i = int(p.stem)
                v = server.vectors_payload(rd, i, full=False)
                if v is not None:
                    payloads[str(i)] = v
            if payloads:
                n += write_json(data / "run" / rid / "vectors.json", payloads)
                nvec = len(payloads)
        total += n
        print(f"  {rid:42} records={len(recs):4} vectors={nvec:4}  {n/1e6:6.1f}MB")

    page = build_page(HERE / "web" / "index.html")
    (out).mkdir(parents=True, exist_ok=True)
    (out / "index.html").write_text(page, encoding="utf-8")

    # Cycle-consistency tab: a second page + its own data/cycle/ tree, published
    # into the same bundle so the topnav link and the '← Studio' back-link resolve.
    total += build_cycle_data(out)

    print(f"[build] wrote index.html + cycle.html + data ({total/1e6:.1f}MB total) to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
