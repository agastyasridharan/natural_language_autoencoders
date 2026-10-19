"""Orchestrate the claim-ablation audit and write rich, incremental run artifacts.

Per audited activation it runs claim_ablation.audit_activation and stores EVERYTHING
for later reference:

  <runs_dir>/<run_id>/
    run.json        config + LIVE status (atomic-rewritten after every activation)
    records.jsonl   one JSON record per activation (text, snippets, per-snippet
                    dlogp / null samples / specificity row / ΔMSE / ||Δ|| / verdict)
    summary.json    corpus aggregates (grounded-rate, histograms, Δlogp-vs-ΔMSE corr)
    vectors/<idx>.npz   full vectors: h, Δ_i, AR(z), AR(z\\C_i), null matrix,
                        per-token logp under h and each ablation, token ids, spans

The GUI (studio/server.py) reads these and polls run.json / records.jsonl for live
updates, so the harness never talks to the GUI directly.

Modes:
  real   loads the AV (vector->text) + AR (text->vector) and runs the full test.
         python run_claim_ablation.py qwen7b --smoke           # 1 activation, verbose
         python run_claim_ablation.py qwen7b --n 400           # corpus run
  mock   no models, no torch - fabricates a schema-identical run so the GUI can be
         built/verified on a laptop. python run_claim_ablation.py qwen7b --mock --slow

Runs dir resolves to $CLAIM_ABLATION_RUNS, else <athena>/runs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_REPL = Path(__file__).resolve().parent
_ATHENA = _REPL.parent
for _p in (str(_REPL), str(_ATHENA), "/data/agastyas/nla_mimic/code",
           "/data/agastyas/nla_mimic/code/repl"):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from repl_common import MODELS  # noqa: E402

ROOT = Path("/data/agastyas/nla_mimic")


def runs_dir_default() -> Path:
    return Path(os.environ.get("CLAIM_ABLATION_RUNS", str(_ATHENA / "runs")))


# ── artifact helpers ─────────────────────────────────────────────────────────

def _atomic_write(path: Path, text: str):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def write_run_json(rundir: Path, state: dict):
    _atomic_write(rundir / "run.json", json.dumps(state, ensure_ascii=False, indent=2))


def append_record(rundir: Path, rec: dict):
    with (rundir / "records.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def read_records(rundir: Path) -> list[dict]:
    out = []
    p = rundir / "records.jsonl"
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass                            # tolerate a half-written trailing line
    return out


def _pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _wilson(k, n, z=1.96):
    """Wilson score 95% CI for a binomial rate k/n (hand-rolled; scipy not allowed)."""
    if n <= 0:
        return [None, None]
    p = k / n
    d = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, center - half), 4), round(min(1.0, center + half), 4)]


# ── atom-mode inputs: reuse the old run's explanations + the segmentation cache ──

def load_segments(path: Path) -> dict:
    """idx -> {z_inner, atoms, ...} from a segmentation cache jsonl (last line per idx wins,
    matching segment_run; error/degenerate lines dropped)."""
    by_idx = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            r = json.loads(line)
            by_idx[int(r["idx"])] = r
    return {i: r for i, r in by_idx.items() if not r.get("error")}


def resolve_reuse(arg: str, runs_dir: Path) -> Path:
    """--reuse-run may be a records.jsonl file, a run dir, or a run-id under runs_dir."""
    p = Path(arg)
    if p.is_file():
        return p
    if p.is_dir():
        return p / "records.jsonl"
    return runs_dir / arg / "records.jsonl"


def load_reuse(path: Path) -> dict:
    """idx -> the old run's record (we use explanation / z_inner / read_token / doc_id)."""
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            r = json.loads(line)
            out[int(r["idx"])] = r
    return out


def claim_bucket(s: dict) -> str:
    """Semantic category of a content claim - the axis that predicts groundedness far better
    than type or role (CP6: Cramer's V 0.16 vs 0.09). Keyword-bucketed over the segmenter's
    free-form subtypes; see docs/research/claim_ablation_atoms.md."""
    su = (s.get("subtype") or "").lower()
    if s.get("role") == "quotes":
        return "quote"
    if s.get("role") == "predicts" or any(k in su for k in (
            "expect", "predict", "next-token", "next token", "candidate", "continuation",
            "completion", "upcoming")):
        return "prediction"
    if any(k in su for k in ("position", "syntact", "structural", "structure", "token",
                             "clause", "parenthetic", "mid-", "boundary")):
        return "position"
    if any(k in su for k in ("genre", "topic", "tone", "style", "register", "rhetor",
                             "subject", "purpose", "narrative", "mood", "voice", "format",
                             "content", "implication", "theme")):
        return "gist"
    return "other"


def _by_type(snips: list[dict]) -> dict:
    """Per-type, per-subtype and per-semantic-bucket verdict contingency over content claims
    (atom mode). Denominator for grounded/confab rates is LOCATED claims of that slice. Also a
    hedged-vs-confident split. Pure-python; used live + final. Returns {} for block records."""
    typed = [s for s in snips if s.get("type")]
    if not typed:
        return {}

    def block(items):
        loc = [s for s in items if s.get("located")]
        g = [s for s in loc if s.get("verdict") == "grounded"]
        e = [s for s in loc if s.get("verdict") == "entangled"]
        c = [s for s in loc if s.get("verdict") == "confabulated"]
        dl = [s["dlogp"] for s in loc if s.get("dlogp") is not None]
        return dict(
            n=len(items), located=len(loc), grounded=len(g), entangled=len(e),
            confabulated=len(c), unlocated=len(items) - len(loc),
            grounded_rate=(len(g) / len(loc)) if loc else None,
            grounded_ci=_wilson(len(g), len(loc)),
            confab_rate=(len(c) / len(loc)) if loc else None,
            confab_ci=_wilson(len(c), len(loc)),
            dlogp_mean=float(np.mean(dl)) if dl else None,
            dlogp_median=float(np.median(dl)) if dl else None)

    types = sorted({s["type"] for s in typed})
    out = {"by_type": {t: block([s for s in typed if s["type"] == t]) for t in types},
           "by_subtype": {}, "hedged": {}}
    for t in types:
        subs = sorted({s.get("subtype", "") for s in typed if s["type"] == t})
        for su in subs:
            items = [s for s in typed if s["type"] == t and s.get("subtype", "") == su]
            if len(items) >= 3:                              # skip singletons in the table
                out["by_subtype"][f"{t}/{su}"] = block(items)
    out["hedged"] = {"hedged": block([s for s in typed if s.get("hedged")]),
                     "confident": block([s for s in typed if not s.get("hedged")])}
    out["by_bucket"] = {b: block([s for s in typed if claim_bucket(s) == b])
                        for b in ("gist", "quote", "position", "prediction")}
    return out


def summarize(records: list[dict]) -> dict:
    """Corpus aggregates over all 'ok' records. Pure (no torch); used live + final."""
    ok = [r for r in records if r.get("status") == "ok"]
    snips = [s for r in ok for s in r.get("snippet_results", [])]
    loc = [s for s in snips if s.get("located")]
    grounded = [s for s in loc if s.get("verdict") == "grounded"]
    entangled = [s for s in loc if s.get("verdict") == "entangled"]
    confab = [s for s in loc if s.get("verdict") == "confabulated"]
    dlogp = [s["dlogp"] for s in loc if s.get("dlogp") is not None]
    dmse = [s["dmse"] for s in loc if s.get("dmse") is not None]
    pair = [(s["dlogp"], s["dmse"]) for s in loc
            if s.get("dlogp") is not None and s.get("dmse") is not None]
    spec = [s["spec_offdiag"] for s in loc if s.get("spec_offdiag") is not None]
    g_dlogp = [s["dlogp"] for s in grounded if s.get("dlogp") is not None]
    c_dlogp = [s["dlogp"] for s in confab if s.get("dlogp") is not None]

    def hist(vals, lo, hi, bins=24):
        if not vals:
            return dict(edges=[], counts=[])
        edges = np.linspace(lo, hi, bins + 1)
        counts, _ = np.histogram(np.clip(vals, lo, hi), bins=edges)
        return dict(edges=[round(float(e), 3) for e in edges],
                    counts=[int(c) for c in counts])

    dl_lo = float(min(dlogp)) if dlogp else 0.0
    dl_hi = float(max(dlogp)) if dlogp else 1.0
    claim_mode = next((r.get("claim_mode") for r in ok if r.get("claim_mode")), "block")
    return dict(
        claim_mode=claim_mode, **_by_type(snips),
        n_activations=len(ok), n_snippets=len(snips), n_located=len(loc),
        n_grounded=len(grounded), n_entangled=len(entangled), n_confab=len(confab),
        grounded_rate=(len(grounded) / len(loc)) if loc else None,
        confab_rate=(len(confab) / len(loc)) if loc else None,
        entangled_rate=(len(entangled) / len(loc)) if loc else None,
        verdict_counts=dict(
            grounded=len(grounded), entangled=len(entangled),
            confabulated=len(confab),
            unlocated=sum(1 for s in snips if not s.get("located")),
        ),
        dlogp_mean=float(np.mean(dlogp)) if dlogp else None,
        dlogp_grounded_mean=float(np.mean(g_dlogp)) if g_dlogp else None,
        dlogp_confab_mean=float(np.mean(c_dlogp)) if c_dlogp else None,
        spec_offdiag_mean=float(np.mean(spec)) if spec else None,
        dlogp_dmse_pearson=_pearson([p[0] for p in pair], [p[1] for p in pair]),
        scatter=[[round(a, 4), round(b, 4)] for a, b in pair[:2000]],
        dlogp_hist=hist(dlogp, min(dl_lo, 0.0), max(dl_hi, 1.0)),
        per_activation_grounded=[
            dict(idx=r["idx"], grounded=r.get("grounded_count", 0),
                 located=r.get("located_count", 0), n=r.get("n_snippets", 0))
            for r in ok],
    )


def write_summary(rundir: Path):
    _atomic_write(rundir / "summary.json",
                  json.dumps(summarize(read_records(rundir)), ensure_ascii=False, indent=2))


# ── real run ─────────────────────────────────────────────────────────────────

def run_real(args, rundir: Path, state: dict):
    import torch
    from transformers import AutoTokenizer
    from huggingface_hub import snapshot_download
    import claim_ablation as ca
    from avgen_model import load_av, read_sidecar
    from nla_inference import NLACritic

    cfg = MODELS[args.key]
    gold_dir = Path(args.gold_dir) if args.gold_dir else ROOT / f"out_corpus_{args.key}"
    gold = np.load(gold_dir / "gold.npy")
    manifest = json.loads((gold_dir / "manifest.json").read_text(encoding="utf-8"))
    extract_meta = {}
    if (gold_dir / "extract_meta.json").exists():
        extract_meta = json.loads((gold_dir / "extract_meta.json").read_text())
    n_target = min(args.n, len(gold)) if args.n else len(gold)
    if args.smoke:
        n_target = 1

    atom_mode = args.claim_mode == "atoms"
    seg_by_idx = reuse_by_idx = None
    if atom_mode:
        if not args.reuse_run:
            raise SystemExit("--claim-mode atoms requires --reuse-run <run id/dir/records.jsonl>")
        seg_path = Path(args.segments) if args.segments else _ATHENA / "segments" / f"{args.key}.jsonl"
        reuse_runs_dir = Path(args.runs_dir) if args.runs_dir else runs_dir_default()
        reuse_path = resolve_reuse(args.reuse_run, reuse_runs_dir)
        seg_by_idx = load_segments(seg_path)
        reuse_by_idx = load_reuse(reuse_path)
        state.update(claim_mode="atoms", segments=str(seg_path), reuse_records=str(reuse_path),
                     spec_pct=args.spec_pct, spec_floor=args.spec_floor,
                     n_segmented=len(seg_by_idx), n_reuse=len(reuse_by_idx))
        print(f"[atoms] {len(seg_by_idx)} segmented claims-sets, {len(reuse_by_idx)} reuse "
              f"explanations from {reuse_path}", flush=True)

    state.update(gold_dir=str(gold_dir), dataset=extract_meta.get("dataset"),
                 n_target=n_target, n_available=len(gold), phase="loading models")
    write_run_json(rundir, state)

    print(f"[run:{args.key}] loading AV {cfg['av']}", flush=True)
    av_dir = snapshot_download(cfg["av"])
    tok = AutoTokenizer.from_pretrained(av_dir, trust_remote_code=True)
    sc = read_sidecar(av_dir)
    _, model = load_av(cfg)
    inj = ca.register_injector(model, sc, cfg["family"], tok)

    print(f"[run:{args.key}] loading AR {cfg['ar']}", flush=True)
    ar_dir = snapshot_download(cfg["ar"])
    critic = NLACritic(ar_dir, device_map="auto")
    state.update(injection_scale=sc["injection_scale"], mse_scale=critic.mse_scale,
                 phase="auditing")
    write_run_json(rundir, state)

    t0 = time.time()
    n_ok = n_skip = n_cjk = 0
    snips_tot = snips_loc = snips_g = 0
    for i in range(n_target):
        m = manifest[i] if i < len(manifest) else {}
        if atom_mode:
            seg = seg_by_idx.get(i)
            ru = reuse_by_idx.get(i)
            if seg is None or ru is None or not ru.get("explanation"):
                rec = dict(idx=i, doc_id=m.get("doc_id", str(i)), status="skip",
                           claim_mode="atoms", n_snippets=0, located_count=0,
                           grounded_count=0, snippet_results=[], z_inner="", cjk=False,
                           note=("no segmentation for idx" if seg is None
                                 else "no reuse explanation for idx"))
                vectors = None
            else:
                rec, vectors = ca.audit_activation_atoms(
                    model, tok, sc, cfg["family"], critic, inj, gold[i],
                    z_assistant=ru["explanation"], z_inner=seg.get("z_inner"),
                    atoms=seg["atoms"], R=args.R, alpha=args.alpha, null_pct=args.null_pct,
                    spec_margin=args.spec_margin, spec_pct=args.spec_pct,
                    spec_floor=args.spec_floor, seed=args.seed + i, idx=i,
                    doc_id=ru.get("doc_id", m.get("doc_id", str(i))),
                    position=m.get("position", -1), n_tokens=m.get("n_tokens", -1),
                    read_token=ru.get("read_token", m.get("read_token", "")))
        else:
            rec, vectors = ca.audit_activation(
                model, tok, sc, cfg["family"], critic, inj, gold[i],
                R=args.R, alpha=args.alpha, temperature=args.temperature,
                max_new=args.max_new, null_pct=args.null_pct,
                spec_margin=args.spec_margin, seed=args.seed + i,
                idx=i, doc_id=m.get("doc_id", str(i)), position=m.get("position", -1),
                n_tokens=m.get("n_tokens", -1), read_token=m.get("read_token", ""))
        rec["context_left"] = m.get("context_left", "")
        rec["context_right"] = m.get("context_right", "")
        rec["source_text"] = m.get("text", "")
        append_record(rundir, rec)
        if vectors is not None:
            np.savez_compressed(rundir / "vectors" / f"{i}.npz", **vectors)
        if rec["status"] == "ok":
            n_ok += 1
            snips_tot += rec["n_snippets"]
            snips_loc += rec["located_count"]
            snips_g += rec["grounded_count"]
        else:
            n_skip += 1
            n_cjk += int(rec.get("cjk", False))

        done = i + 1
        elapsed = time.time() - t0
        rate = elapsed / done
        state.update(
            phase="auditing", n_done=done, n_ok=n_ok, n_skip=n_skip, n_cjk=n_cjk,
            n_records=done, snippets_total=snips_tot, snippets_located=snips_loc,
            snippets_grounded=snips_g,
            grounded_rate=(snips_g / snips_loc) if snips_loc else None,
            updated_ts=time.time(), elapsed_s=round(elapsed, 1),
            eta_s=round(rate * (n_target - done), 1),
            last_text=(rec.get("z_inner", "") or rec.get("note", ""))[:240])
        write_run_json(rundir, state)
        write_summary(rundir)

        if args.smoke:
            _print_smoke(rec)
        elif done % 5 == 0 or done == n_target:
            gr = state["grounded_rate"]
            print(f"  [{done}/{n_target}] ok={n_ok} skip={n_skip} cjk={n_cjk} "
                  f"grounded_rate={gr if gr is None else round(gr,3)}", flush=True)
        if args.slow:
            time.sleep(args.slow)

    state.update(status="done", phase="done", updated_ts=time.time())
    write_run_json(rundir, state)
    write_summary(rundir)
    print(f"[run:{args.key}] done. grounded_rate={state['grounded_rate']} "
          f"({snips_g}/{snips_loc} located snippets). artifacts in {rundir}", flush=True)


def _print_smoke(rec):
    print("\n" + "=" * 72)
    print(f"SMOKE  idx={rec['idx']}  status={rec['status']}  {rec.get('note','')}")
    print(f"read_token={rec.get('read_token','')!r}  |h|={rec.get('h_norm'):.1f}")
    print(f"explanation:\n  {rec.get('z_inner','')[:400]}")
    for s in rec.get("snippet_results", []):
        print(f"  [{s['i']}] {s['verdict']:<12} dlogp={s['dlogp']} "
              f"null_thr={s['null_thr']} spec_off={s['spec_offdiag']} "
              f"dmse={s['dmse']} |Δ|={s['delta_norm']}")
        print(f"       {s['text'][:90]!r}")
    print("=" * 72 + "\n")


# ── mock run (no torch; for building/verifying the GUI on a laptop) ───────────

_MOCK_CLAIMS = [
    ("The text discusses a scientific or technical topic in a neutral register.", 0.15),
    ("This position sits inside a numbered list or enumeration.", 0.80),
    ("The activation encodes a transition to a named geographic location.", 0.70),
    ("The surrounding context is about historical events in the 20th century.", 0.55),
    ("The model is tracking a person's name introduced earlier in the passage.", 0.75),
    ("The tone is formal and academic.", 0.05),
    ("This token continues a quotation or reported speech.", 0.60),
    ("The passage is in English prose.", 0.02),
    ("The context concerns financial or economic figures.", 0.65),
    ("The activation reflects code or markup rather than natural language.", 0.72),
    ("There is a shift toward a more emotional or evaluative stance.", 0.30),
    ("The text references a legal or regulatory framework.", 0.58),
]
_MOCK_TOKENS = [" the", " of", " 1987", " London", " Dr.", " ,", " \"", " market",
                " def", " however", " court", " war"]


def _mock_record(idx, d, rng, R, null_pct, alpha, spec_margin):
    n = int(rng.integers(2, 4))
    picks = rng.choice(len(_MOCK_CLAIMS), size=n, replace=False)
    snippets = [_MOCK_CLAIMS[p][0] for p in picks]
    z_inner = "\n\n".join(snippets)
    explanation = f"<explanation>\n{z_inner}\n</explanation>"
    T = int(rng.integers(48, 72))               # pretend continuation length
    h_norm = float(rng.uniform(20, 200))

    span_lens = [int(rng.integers(6, 14)) for _ in range(n)]
    null = np.zeros((R, n), np.float32)
    for k in range(R):
        null[k] = rng.normal(0.0, 0.45, n)
    thr = [float(np.percentile(null[:, i], null_pct)) for i in range(n)]
    # pick a latent category per claim, then synthesize an effect matrix matching it
    cats, dlogp, col_max = [], np.zeros(n), np.zeros(n)
    for i in range(n):
        pg = _MOCK_CLAIMS[picks[i]][1]
        r = rng.random()
        cat = "grounded" if r < pg else ("entangled" if r < pg + 0.25 else "confabulated")
        cats.append(cat)
        if cat == "confabulated":               # within the null: not read from h
            dlogp[i] = float(rng.normal(0.1, 0.4)); col_max[i] = abs(rng.normal(0.1, 0.2))
        elif cat == "grounded":                 # big effect, own ablation dominates
            dlogp[i] = thr[i] + abs(rng.normal(3.0, 1.0)); col_max[i] = dlogp[i] / rng.uniform(2.2, 5)
        else:                                    # entangled: big effect but a sibling rivals it
            dlogp[i] = thr[i] + abs(rng.normal(3.0, 1.0)); col_max[i] = dlogp[i] / rng.uniform(1.05, 1.35)
    spec = np.zeros((n, n), np.float32)          # spec[i][j] = effect of ablate i on j
    for i in range(n):
        spec[i, i] = dlogp[i]
    for i in range(n):                           # plant col_max[i] as a sibling's effect on i
        others = [k for k in range(n) if k != i]
        if others:
            sib = int(rng.choice(others)); spec[sib, i] = col_max[i]
            for k in others:
                if k != sib:
                    spec[k, i] = float(rng.normal(0.0, 0.2))

    results = []
    counts = dict(grounded=0, entangled=0, confabulated=0, unlocated=0)
    for i in range(n):
        col_off = [spec[k, i] for k in range(n) if k != i]
        com = float(max(col_off)) if col_off else 0.0
        denom = max(com, 0.0)
        row_off = [spec[i, j] for j in range(n) if j != i]
        spec_off = float(np.mean(row_off)) if row_off else 0.0
        depends = bool(dlogp[i] > thr[i])
        specific = bool(dlogp[i] > spec_margin * denom)
        verdict = "confabulated" if not depends else ("grounded" if specific else "entangled")
        counts[verdict] += 1
        l0 = -float(span_lens[i]) * float(rng.uniform(1.5, 3.0))
        lo = 10 + sum(span_lens[:i]) + i
        results.append(dict(
            i=i, text=snippets[i], located=True,
            token_span=[lo, lo + span_lens[i]], n_span_tokens=span_lens[i],
            l0=round(l0, 3), l1=round(l0 - float(dlogp[i]), 3), dlogp=round(float(dlogp[i]), 3),
            dlogp_per_token=round(float(dlogp[i]) / span_lens[i], 3),
            null_mean=round(float(np.mean(null[:, i])), 3), null_thr=round(thr[i], 3),
            null_dlogps=[round(float(x), 3) for x in null[:, i]],
            spec_row=[round(float(spec[i, j]), 3) for j in range(n)],
            spec_offdiag=round(spec_off, 3), col_off_max=round(com, 3),
            spec_ratio=(round(float(dlogp[i] / denom), 3) if denom > 1e-6 else None),
            depends_on_h=depends, specific=specific,
            dmse=round(float(rng.normal(0.25 if verdict == "grounded" else 0.05, 0.1)), 3),
            delta_norm=round(float(rng.uniform(2, 30)), 3),
            proj_mag=round(float(rng.normal(0, h_norm * 0.3)), 3),
            proj_cos=round(float(rng.uniform(-0.4, 0.4)), 3),
            grounded=bool(verdict == "grounded"), verdict=verdict))

    rec = dict(
        idx=idx, doc_id=f"mock:{idx}", position=int(rng.integers(50, 400)),
        n_tokens=int(rng.integers(120, 800)), read_token=str(rng.choice(_MOCK_TOKENS)),
        h_norm=round(h_norm, 2), explanation=explanation, z_inner=z_inner,
        snippets=snippets, n_snippets=n, cjk=False, status="ok", note="",
        mse_full=round(float(rng.uniform(0.2, 0.9)), 3),
        cos_full=round(float(rng.uniform(0.55, 0.95)), 3),
        null_pct=null_pct, R=R, alpha=alpha, spec_margin=spec_margin,
        grounded_count=counts["grounded"], entangled_count=counts["entangled"],
        confab_count=counts["confabulated"], located_count=n, snippet_results=results,
        context_left="... mock left context ending at the read token" +
                     str(rng.choice(_MOCK_TOKENS)),
        context_right=" mock right context follows.",
        source_text="This is synthetic source text generated for GUI development. "
                    "It stands in for a real FineWeb document so every panel of the "
                    "studio has something to render before the GPU run exists.")
    vectors = dict(
        h=rng.normal(0, 1, d).astype(np.float32) * (h_norm / np.sqrt(d)),
        deltas=rng.normal(0, 1, (n, d)).astype(np.float32),
        ar_full=rng.normal(0, 1, d).astype(np.float32),
        ar_minus=rng.normal(0, 1, (n, d)).astype(np.float32),
        null=null, tok_logp_h=rng.normal(-2, 1, T).astype(np.float32),
        tok_logp_ablate=rng.normal(-2, 1, (n, T)).astype(np.float32),
        token_ids=rng.integers(0, 50000, T + 1).astype(np.int64),
        ranges=np.array([r["token_span"] for r in results], np.int64),
        marker_pos=np.array([7], np.int64), cont_token_start=np.array([9], np.int64))
    return rec, vectors


def run_mock(args, rundir: Path, state: dict):
    cfg = MODELS[args.key]
    d = cfg["d_model"]
    n_target = args.n or 12
    rng = np.random.default_rng(args.seed)
    state.update(n_target=n_target, n_available=n_target, mock=True,
                 dataset="(mock)", phase="auditing",
                 injection_scale=None, mse_scale=float(np.sqrt(d)))
    write_run_json(rundir, state)
    t0 = time.time()
    n_ok = snips_tot = snips_loc = snips_g = 0
    for i in range(n_target):
        rec, vectors = _mock_record(i, d, rng, args.R, args.null_pct, args.alpha,
                                    args.spec_margin)
        append_record(rundir, rec)
        np.savez_compressed(rundir / "vectors" / f"{i}.npz", **vectors)
        n_ok += 1
        snips_tot += rec["n_snippets"]
        snips_loc += rec["located_count"]
        snips_g += rec["grounded_count"]
        elapsed = time.time() - t0
        state.update(phase="auditing", n_done=i + 1, n_ok=n_ok, n_skip=0, n_cjk=0,
                     n_records=i + 1, snippets_total=snips_tot,
                     snippets_located=snips_loc, snippets_grounded=snips_g,
                     grounded_rate=(snips_g / snips_loc) if snips_loc else None,
                     updated_ts=time.time(), elapsed_s=round(elapsed, 1),
                     eta_s=round(elapsed / (i + 1) * (n_target - i - 1), 1),
                     last_text=rec["z_inner"][:240])
        write_run_json(rundir, state)
        write_summary(rundir)
        print(f"  [mock {i+1}/{n_target}] grounded {snips_g}/{snips_loc}", flush=True)
        if args.slow:
            time.sleep(args.slow)
    state.update(status="done", phase="done", updated_ts=time.time())
    write_run_json(rundir, state)
    write_summary(rundir)
    print(f"[mock:{args.key}] done -> {rundir}", flush=True)


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("key", choices=list(MODELS))
    ap.add_argument("--gold-dir", default=None, help="dir with gold.npy + manifest.json")
    ap.add_argument("--n", type=int, default=0, help="cap activations (0 = all)")
    ap.add_argument("--R", type=int, default=10, help="random-direction null samples")
    ap.add_argument("--alpha", type=float, default=1.0, help="ablation strength")
    ap.add_argument("--temperature", type=float, default=0.4)
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--null-pct", type=float, default=95.0)
    ap.add_argument("--spec-margin", type=float, default=1.5,
                    help="column-dominance factor: own ablation must exceed the "
                         "specificity denominator by this much to be 'grounded'")
    ap.add_argument("--claim-mode", choices=["block", "atoms"], default="block",
                    help="block = blank-line snippets (default); atoms = precomputed "
                         "two-layer content claims (needs --reuse-run + --segments)")
    ap.add_argument("--reuse-run", default=None,
                    help="atom mode: old run id / dir / records.jsonl to reuse the AV "
                         "explanations (precomputed_z); its gold_dir is reused too")
    ap.add_argument("--segments", default=None,
                    help="atom mode: segmentation cache jsonl (idx -> z_inner + atoms); "
                         "default <athena>/segments/<key>.jsonl")
    ap.add_argument("--spec-pct", type=float, default=90.0,
                    help="atom mode: specificity-denominator percentile of the collateral "
                         "column (CP4 rule; 90 = p90)")
    ap.add_argument("--spec-floor", type=float, default=1.0,
                    help="atom mode: min own-effect (nats) to be 'grounded' (CP4 floor)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="1 activation, verbose")
    ap.add_argument("--mock", action="store_true", help="no models; synthetic run")
    ap.add_argument("--slow", type=float, default=0.0,
                    help="seconds to sleep between activations (demo live updates)")
    ap.add_argument("--runs-dir", default=None)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--tag", default="", help="suffix appended to the run id")
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir) if args.runs_dir else runs_dir_default()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_id = args.run_id or (f"{args.key}-{'mock-' if args.mock else ''}{stamp}"
                             + (f"-{args.tag}" if args.tag else ""))
    if args.claim_mode == "atoms" and not args.mock and not run_id.endswith("-atoms"):
        run_id += "-atoms"
    rundir = runs_dir / run_id
    (rundir / "vectors").mkdir(parents=True, exist_ok=True)
    cfg = MODELS[args.key]

    state = dict(
        run_id=run_id, key=args.key, model_label=cfg["label"], family=cfg["family"],
        d_model=cfg["d_model"], layer=cfg["layer"], av=cfg["av"], ar=cfg["ar"],
        base=cfg["base"], code="claim_ablation", mock=bool(args.mock),
        smoke=bool(args.smoke), claim_mode=args.claim_mode, R=args.R, alpha=args.alpha,
        temperature=args.temperature, null_pct=args.null_pct,
        spec_margin=args.spec_margin, spec_pct=args.spec_pct, spec_floor=args.spec_floor,
        seed=args.seed,
        status="running", phase="starting", n_target=0, n_done=0, n_ok=0, n_skip=0,
        n_cjk=0, n_records=0, snippets_total=0, snippets_located=0,
        snippets_grounded=0, grounded_rate=None, created_at=stamp,
        started_ts=time.time(), updated_ts=time.time(), elapsed_s=0.0, eta_s=None,
        last_text="", error=None)
    write_run_json(rundir, state)
    print(f"[run] {run_id} -> {rundir}", flush=True)

    try:
        (run_mock if args.mock else run_real)(args, rundir, state)
    except Exception as e:
        import traceback
        state.update(status="failed", phase="error", error=str(e),
                     updated_ts=time.time())
        write_run_json(rundir, state)
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
