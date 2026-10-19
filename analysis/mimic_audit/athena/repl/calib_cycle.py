"""S2 calibration for the cycle-consistency test: measure everything the four
filters and the Step-7 thresholds need, on the calibration split, then propose
the thresholds. Two modes:

  measure  (GPU, Athena)   per calibration claim: AR directions + overlaps
                           (filter 2), per-token prior means under iota(h-bar)
                           (filter 3), FVE of every z* (filter 4b), supp(B|h),
                           supp(B*_m|h) and D*_m readings (Step 7 pre-checks,
                           kappa_0). Writes calib/<run>/claims/<idx>_<ci>.json
                           per claim (resumable) + status.json.
      python calib_cycle.py qwen7b --run qwen7b-cp6-fineweb-atoms

  analyze  (CPU, anywhere) reads the per-claim files and runs the SIGNED
                           package (S2 sign-off 2026-07-21): p90 f3/f4 bands,
                           spec-literal per-claim filter-2 widening with the
                           per-claim eps logged (band disabled as the final
                           fallthrough), per-family power (supp_b vs Q95 of the
                           claim's own surviving decoys' supp*), provisional
                           kappa_0 LODO -> calibration_signed.json + audit md.
      python calib_cycle.py qwen7b --run qwen7b-cp6-fineweb-atoms --analyze

  analyze-legacy           the pre-sign-off pooled analysis (kept so the frozen
                           d3g12 artifacts under calib/ stay re-derivable:
                           --decoys-dir .../decoys --out .../calib
                           --prompt-version d3) -> calibration_proposed.json.

Measurement discipline (spec + project invariants):
  - drives the EXISTING measurement path: build_tf_sequence -> LayerZeroInjector
    -> _inject_span_logp. No parallel reimplementation.
  - cycle-mode spans use first_char_ranges (disjoint, spec Step 4 "its own
    tokens only"); the published atom run's OVERLAP spans are recomputed only
    to cross-check l0 against the stored record (integrity link to the
    published run, warn-not-fail on fp drift).
  - Delta_B is recomputed via decoy_directions and asserted against the stored
    vectors/<idx>.npz deltas[ci] when present.
  - every claim/candidate that cannot be measured is recorded with a reason
    (unlocated span, degenerate direction, splice failure) - never dropped.
  - injection h-bar = raw mean of the VALID gold rows (fve.py's is_valid mask);
    FVE uses the separate unit-normalized mean + per-row D_i. Two h-bars, kept
    distinct on purpose (spec conflates them; see cycle_consistency docstring).
  - no LLM sees any number produced here (judge-free measurement).
"""
from __future__ import annotations

import argparse
import hashlib
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

import cycle_consistency as cc                                   # noqa: E402

ROOT = Path("/data/agastyas/nla_mimic")
EPS_OV_LADDER = (0.10, 0.15, 0.20, 0.30, 0.50)
_EPS = 1e-8


def _f(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def _atomic_write(path: Path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def is_valid_row(m: dict) -> bool:
    """fve.py's exact validity rule for gold/manifest rows."""
    ok = m.get("in_range", True)
    if "anchor_ok" in m:
        ok = ok and m["anchor_ok"]
    return ok


def load_run_records(runs_dir: Path, run_id: str) -> dict:
    recs = {}
    with open(runs_dir / run_id / "records.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            recs[int(r["idx"])] = r
    return recs


def decoy_cache_files(decoys_dir: Path, run_id: str, model_key: str):
    d = decoys_dir / run_id
    out = []
    for p in sorted(d.glob(f"*_*.{model_key}.json")):
        idx_s, ci_s = p.name.split(".")[0].split("_")
        out.append((int(idx_s), int(ci_s), p))
    return sorted(out)


def text_pass_candidates(cache: dict):
    """Indices (into cache['candidates']) that survived the text filters:
    hygiene-kept + absence-judged absent + naturalness pass (the pass bool and
    its threshold natural_pass_min were fixed at generation time by decoy_gen)."""
    return [i for i, c in enumerate(cache["candidates"])
            if c.get("kept") and c.get("absence") == "absent"
            and c.get("natural_pass")]


# ── measure mode (GPU) ───────────────────────────────────────────────────────

def measure(args):
    import torch                                                  # noqa: F401
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    import claim_ablation as ca
    import ar_cache as ac
    from avgen_model import load_av, read_sidecar
    from nla_inference import NLACritic
    from repl_common import MODELS

    cfg = MODELS[args.key]
    runs_dir = Path(args.runs_dir)
    gold_dir = Path(args.gold_dir) if args.gold_dir else ROOT / f"out_corpus_{args.key}"
    gold = np.load(gold_dir / "gold.npy")
    manifest = json.loads((gold_dir / "manifest.json").read_text(encoding="utf-8"))
    valid = [i for i, m in enumerate(manifest) if is_valid_row(m)]
    records = load_run_records(runs_dir, args.run)
    work = decoy_cache_files(Path(args.decoys_dir), args.run, args.model)
    if args.idx:
        work = [w for w in work if w[0] in set(args.idx)]
    if args.limit:
        work = work[:args.limit]
    if not work:
        raise SystemExit("no decoy caches found - run decoy_gen.py first")

    out_dir = Path(args.out) / args.run
    claims_dir = out_dir / "claims"
    claims_dir.mkdir(parents=True, exist_ok=True)

    # both h-bars, from VALID rows only (published FVE convention)
    hbar_inj = cc.corpus_hbar(gold[valid])
    ubar, _ = cc.unit_hbar_di(gold[valid])

    print(f"[calib:{args.key}] {len(work)} claims, {len(valid)}/{len(gold)} valid "
          f"gold rows; loading AV {cfg['av']}", flush=True)
    av_dir = snapshot_download(cfg["av"])
    tok = AutoTokenizer.from_pretrained(av_dir, trust_remote_code=True)
    sc = read_sidecar(av_dir)
    _, model = load_av(cfg)
    inj = ca.register_injector(model, sc, cfg["family"], tok)
    # phase-split: --ar-cache means the critic already ran (ar_precompute.py) and
    # its directions are on disk, so we skip loading the AR entirely (the 70B AV
    # + AR do not co-reside on 2x H100 NVL). Without it, the live critic path.
    ar_dir_cache = Path(args.ar_cache) if getattr(args, "ar_cache", None) else None
    if ar_dir_cache is None:
        ar_dir = snapshot_download(cfg["ar"])
        critic = NLACritic(ar_dir, device_map="auto")
        mse_scale = critic.mse_scale
    else:
        critic = None
        mse_scale = None
        print(f"[calib:{args.key}] AR phase-split: reading directions from "
              f"{ar_dir_cache}", flush=True)
    S = sc["injection_scale"]

    status = dict(key=args.key, run=args.run, alpha=args.alpha, n_work=len(work),
                  n_valid_gold=len(valid), injection_scale=S,
                  mse_scale=mse_scale, ar_cache=str(ar_dir_cache) if ar_dir_cache else None,
                  done=0, skipped=0, phase="measuring",
                  started=time.strftime("%Y-%m-%d %H:%M:%S"))
    _atomic_write(out_dir / "status.json", status)

    # per-activation shared state (2 forwards on z serve all its claims)
    act_cache = {}

    def activation_state(idx: int):
        if idx in act_cache:
            return act_cache[idx]
        rec = records[idx]
        h = np.asarray(gold[idx], dtype=np.float32)
        zi = rec["z_inner"]
        input_ids, p, offsets, full_text, cont = ca.build_tf_sequence(
            tok, sc, cfg["family"], rec["explanation"])
        assert p == inj.p, f"marker drift {p} vs {inj.p}"
        claims = rec["snippet_results"]
        z0 = cc.locate_z_inner(full_text, zi, cont)
        spans_abs = [None if z0 < 0 else
                     (z0 + c["char_span"][0], z0 + c["char_span"][1]) for c in claims]
        r_ov = ca.atom_spans(offsets, full_text, zi,
                             [{"char_span": c["char_span"]} for c in claims], cont)
        r_fc = cc.first_char_ranges(offsets, spans_abs)
        sums_h_ov, tlp_h = ca._inject_span_logp(model, inj, input_ids, h, S, r_ov)
        sums_h_fc = ca._span_sums(tlp_h, r_fc)
        sums_lb_fc, _ = ca._inject_span_logp(model, inj, input_ids, hbar_inj, S, r_fc)
        # integrity link to the published run: l0 under the OLD overlap spans
        l0_diff = [abs(a - b) for a, b, c in
                   zip(sums_h_ov, (c.get("l0") for c in claims), claims)
                   if a == a and isinstance(b, (int, float))]
        st = dict(rec=rec, h=h, zi=zi, claims=claims, spans_abs=spans_abs,
                  r_fc=r_fc, sums_h_fc=sums_h_fc, sums_lb_fc=sums_lb_fc,
                  l0_check_max=max(l0_diff) if l0_diff else None,
                  uh=h / max(float(np.linalg.norm(h)), _EPS))
        if st["l0_check_max"] is not None and st["l0_check_max"] > 0.05:
            print(f"  [warn] idx {idx}: l0 drift vs stored record "
                  f"max {st['l0_check_max']:.4f} nats", flush=True)
        act_cache.clear()                       # only ever need the current one
        act_cache[idx] = st
        return st

    npz_dir = runs_dir / args.run / "vectors"
    t0 = time.time()
    for wi, (idx, ci, cache_path) in enumerate(work):
        out_path = claims_dir / f"{idx}_{ci}.json"
        if out_path.exists() and not args.force:
            status["skipped"] += 1
            continue
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        st = activation_state(idx)
        rec, h, zi = st["rec"], st["h"], st["zi"]
        claim = st["claims"][ci]
        span = tuple(claim["char_span"])
        assert cache["claim"]["text"] == claim["text"], (idx, ci, "claim text drift")
        sha = hashlib.sha256(zi.encode()).hexdigest()[:16]
        assert cache["z_inner_sha"] == sha, (idx, ci, "stale decoy cache (z_inner)")
        assert cache.get("prompt_version") == args.prompt_version, \
            (idx, ci, f"stale decoy cache (prompt {cache.get('prompt_version')} "
                      f"!= {args.prompt_version})")

        keep_idx = text_pass_candidates(cache)
        if cache.get("n_text_pass") != len(keep_idx):
            print(f"  [warn] idx {idx}.{ci}: n_text_pass drift "
                  f"{cache.get('n_text_pass')} vs {len(keep_idx)}", flush=True)
        cand_texts = [cache["candidates"][i]["text"] for i in keep_idx]
        row = dict(idx=idx, claim_i=ci, doc_id=rec.get("doc_id"),
                   claim_text=claim["text"], claim_type=claim["type"],
                   claim_subtype=claim.get("subtype"), char_span=list(span),
                   taxo=cache.get("taxo"), n_candidates=len(cache["candidates"]),
                   n_text_pass=len(keep_idx), keep_idx=keep_idx,
                   alpha=args.alpha, candidates=[])

        # z-side (shared forwards, already cached per activation)
        fcr = st["r_fc"][ci]
        span_len = (fcr[1] - fcr[0]) if fcr else 0
        supp_b = (st["sums_h_fc"][ci] - st["sums_lb_fc"][ci]) if fcr else float("nan")
        row.update(located_fc=fcr is not None,
                   token_span_fc=list(fcr) if fcr else None,
                   supp_b=_f(supp_b),
                   mean_lbar_b=_f(st["sums_lb_fc"][ci] / span_len) if span_len else None,
                   l0_check_max=_f(st["l0_check_max"]),
                   m_bb_stored=_f(claim.get("dlogp")),
                   verdict_stored=claim.get("verdict"))

        if not keep_idx:
            row.update(status="no text-passing candidates")
            _atomic_write(out_path, row)
            status["done"] += 1
            continue

        # AR side: same-subtrahend directions + overlaps + cos for every z*
        if ar_dir_cache is not None:
            dd = ac.load_ar_directions(ar_dir_cache, args.run, idx, ci,
                                       cand_texts, zi, args.prompt_version)
        else:
            dd = cc.decoy_directions(critic, zi, span, cand_texts, h)
        d_idx = float(((st["uh"] - ubar) ** 2).sum())
        fve_z = cc.fve_from_cos(dd["cos_full"], d_idx)
        npz_check = None
        npz_path = npz_dir / f"{idx}.npz"
        if npz_path.exists():
            stored = np.load(npz_path)["deltas"][ci]
            npz_check = float(np.abs(dd["delta_b"] - stored).max())
            assert npz_check < 1e-3, (idx, ci, f"delta_b vs npz drift {npz_check}")
        row.update(ov_b=_f(dd["ov_b"]), cos_full=_f(dd["cos_full"]),
                   fve_z=_f(fve_z), d_idx=_f(d_idx),
                   delta_b_norm=_f(np.linalg.norm(dd["delta_b"])),
                   npz_delta_check=_f(npz_check))

        # per-candidate: 3 TF forwards on z*_m (h, h-bar, h' = project_out)
        zi0 = rec["explanation"].find(zi)
        assert zi0 >= 0, (idx, "z_inner not found in explanation")
        for k, cand_i in enumerate(keep_idx):
            cand = cache["candidates"][cand_i]
            c_row = dict(cand_i=cand_i, text=cand["text"],
                         natural=cand.get("natural_score"),
                         ov=_f(dd["ov"][k]), cos_star=_f(dd["cos_star"][k]),
                         fve_star=_f(cc.fve_from_cos(dd["cos_star"][k], d_idx)))
            z_star = dd["z_stars"][k]
            ss, se = dd["star_spans"][k]
            z_star_assist = (rec["explanation"][:zi0] + z_star
                             + rec["explanation"][zi0 + len(zi):])
            try:
                ids_s, p_s, off_s, full_s, cont_s = ca.build_tf_sequence(
                    tok, sc, cfg["family"], z_star_assist)
                assert p_s == inj.p, f"marker drift in z* {p_s} vs {inj.p}"
                zs0 = cc.locate_z_inner(full_s, z_star, cont_s)
                rng_s = (cc.first_char_ranges(off_s, [(zs0 + ss, zs0 + se)])[0]
                         if zs0 >= 0 else None)
            except AssertionError as e:
                c_row.update(status=f"tf-sequence failed: {e}", located=False)
                row["candidates"].append(c_row)
                continue
            if rng_s is None:
                c_row.update(status="decoy span unlocated in z*", located=False)
                row["candidates"].append(c_row)
                continue
            sl = rng_s[1] - rng_s[0]
            (s_h,), _ = ca._inject_span_logp(model, inj, ids_s, h, S, [rng_s])
            (s_lb,), _ = ca._inject_span_logp(model, inj, ids_s, hbar_inj, S, [rng_s])
            delta_s = dd["deltas"][k]
            dn = float(np.linalg.norm(delta_s))
            if dn < _EPS:
                c_row.update(status="degenerate decoy direction", located=True,
                             supp_star=_f(s_h - s_lb), mean_lbar_star=_f(s_lb / sl),
                             token_span=list(rng_s), dstar=None)
                row["candidates"].append(c_row)
                continue
            hp = ca.project_out(h, delta_s / dn, args.alpha)
            (s_hp,), _ = ca._inject_span_logp(model, inj, ids_s, hp, S, [rng_s])
            c_row.update(status="ok", located=True, token_span=list(rng_s),
                         n_span_tokens=sl,
                         supp_star=_f(s_h - s_lb),
                         mean_lbar_star=_f(s_lb / sl),
                         dstar=_f(s_h - s_hp),
                         proj_mag=_f(float(h @ (delta_s / dn))))
            row["candidates"].append(c_row)

        row["status"] = "ok"
        _atomic_write(out_path, row)
        status["done"] += 1
        if (wi + 1) % 5 == 0 or wi + 1 == len(work):
            status.update(elapsed_s=round(time.time() - t0, 1))
            _atomic_write(out_dir / "status.json", status)
            print(f"  [{wi + 1}/{len(work)}] idx {idx} claim {ci}: "
                  f"{len(keep_idx)} text-pass, {time.time() - t0:.0f}s", flush=True)

    status.update(phase="done", elapsed_s=round(time.time() - t0, 1))
    _atomic_write(out_dir / "status.json", status)
    print(f"[calib:{args.key}] done: {status['done']} measured, "
          f"{status['skipped']} already present", flush=True)


# ── analyze mode (CPU) ───────────────────────────────────────────────────────

def _pct(a, q):
    a = np.asarray([x for x in a if x is not None and np.isfinite(x)], np.float64)
    return float(np.percentile(a, q)) if a.size else float("nan")


def _load_rows(args):
    out_dir = Path(args.out) / args.run
    rows = [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted((out_dir / "claims").glob("*.json"))]
    if not rows:
        raise SystemExit(f"no measured claims under {out_dir}/claims")
    return out_dir, rows


def analyze_legacy(args):
    """Pre-sign-off pooled analysis (writes calibration_proposed.json). Kept
    verbatim so the frozen d3g12 artifacts stay re-derivable; the signed
    package lives in analyze_signed below."""
    out_dir, rows = _load_rows(args)
    k = args.k

    # flat candidate pool (text-passing, measured ok)
    cands = [(r, c) for r in rows for c in r.get("candidates", [])
             if c.get("status") in ("ok", "degenerate decoy direction")]
    ok = [(r, c) for r, c in cands if c.get("status") == "ok"]

    # marginal mismatch distributions (set eps_pr / fve_tol from these)
    pr_gap = [abs(c["mean_lbar_star"] - r["mean_lbar_b"]) for r, c in ok
              if c.get("mean_lbar_star") is not None and r.get("mean_lbar_b") is not None]
    fve_gap = [abs(c["fve_star"] - r["fve_z"]) for r, c in ok
               if c.get("fve_star") is not None and r.get("fve_z") is not None]
    eps_pr = args.eps_pr if args.eps_pr is not None else _pct(pr_gap, 80)
    fve_tol = args.fve_tol if args.fve_tol is not None else _pct(fve_gap, 80)
    assert np.isfinite(eps_pr) and np.isfinite(fve_tol), (
        "no measured candidates to set eps_pr/fve_tol from - a nan band would "
        "fail OPEN in survives(); measure first or pass --eps-pr/--fve-tol")

    def survives(r, c, eps_ov):
        if c.get("status") != "ok":
            return False
        if None in (c.get("ov"), r.get("ov_b")) or abs(r["ov_b"]) < 1e-12:
            return False
        if abs(c["ov"] - r["ov_b"]) > eps_ov * abs(r["ov_b"]):
            return False
        if abs(c["mean_lbar_star"] - r["mean_lbar_b"]) > eps_pr:
            return False
        return abs(c["fve_star"] - r["fve_z"]) <= fve_tol

    ladder_tbl = []
    for eps in EPS_OV_LADDER:
        surv = [sum(survives(r, c, eps) for c in r.get("candidates", [])) for r in rows]
        ladder_tbl.append(dict(eps_ov=eps,
                               frac_claims_ge_k=round(float(np.mean([s >= k for s in surv])), 3),
                               median_survivors=int(np.median(surv)),
                               total=int(np.sum(surv))))
    # HARNESS RULE (not spec): pick the first rung where >= 80% of claims reach
    # k survivors; the spec (filter 2) says only "widen the band until at least
    # k decoys survive per claim", with no cap and no coverage criterion. If no
    # rung qualifies we fall back to the ladder cap and SAY SO - a fallback
    # eps_ov is a failed calibration, not a calibrated value.
    base_eps = next((t["eps_ov"] for t in ladder_tbl if t["frac_claims_ge_k"] >= 0.8),
                    None)
    eps_criterion_met = base_eps is not None
    if base_eps is None:
        base_eps = EPS_OV_LADDER[-1]

    # tau_pow: pooled Q95 of decoy supp, floored at 0. HARNESS OPERATIONALIZATION
    # of the spec's "clearly positive" (Step 7 prescribes no procedure). Two
    # measured caveats (S2 qwen7b): this pool is NOT near zero (median +26.5
    # nats; theme-matched decoys inherit topical support), and span-sum supp is
    # length-confounded (corr with n_span_tokens ~0.74) - so the pooled sum-Q95
    # is very aggressive. Alternatives reported alongside for sign-off.
    supp_star = [c["supp_star"] for _, c in ok if c.get("supp_star") is not None]
    tau_pow = max(0.0, _pct(supp_star, 95))
    tau_val = _pct([abs(s) for s in supp_star], 99)

    # alternative power constructions (all mechanical, all disclosed):
    #   per_token: pooled Q95 of supp*/n_span_tokens, compared against
    #              supp_b/n_span_tokens_b (removes the length confound);
    #   per_family: supp_b > Q95 of the claim's OWN decoys' supp* (matched
    #               null - controls document/topic propensity, needs >= k
    #               measured decoys).
    st_tok = [c["supp_star"] / c["n_span_tokens"] for _, c in ok
              if c.get("supp_star") is not None and c.get("n_span_tokens")]
    tau_pow_tok = max(0.0, _pct(st_tok, 95))
    n_pow_tok = 0
    fam_pow, fam_tot = 0, 0
    for r in rows:
        sb = r.get("supp_b")
        if sb is None:
            continue
        ntok = r.get("token_span_fc")
        nb = (ntok[1] - ntok[0]) if ntok else None
        if nb:
            n_pow_tok += (sb / nb) > tau_pow_tok
        own = [c["supp_star"] for c in r.get("candidates", [])
               if c.get("status") == "ok" and c.get("supp_star") is not None]
        if len(own) >= k:
            fam_tot += 1
            fam_pow += sb > _pct(own, 95)

    # kappa_0 by leave-one-decoy-out over families with > k survivors at base_eps
    fam = []
    for r in rows:
        d = [c["dstar"] for c in r.get("candidates", [])
             if survives(r, c, base_eps) and c.get("dstar") is not None]
        if len(d) > k:
            fam.append(np.asarray(d, np.float64))
    grid, kappa_0 = [], None
    for kap in np.arange(0.0, 5.01, 0.1):
        n_pass = n_tot = 0
        for d in fam:
            for j in range(d.size):
                rest = np.delete(d, j)
                n_pass += d[j] > np.percentile(rest, 95) + kap
                n_tot += 1
        rate = n_pass / n_tot if n_tot else float("nan")
        grid.append(dict(kappa=round(float(kap), 2), lodo_pass_rate=round(rate, 4)))
        if kappa_0 is None and n_tot and rate <= 0.05:
            kappa_0 = float(kap)

    # headline coverage numbers at the proposed thresholds - these, not the
    # thresholds themselves, are what sign-off commits to
    n_resolved = sum(sum(survives(r, c, base_eps) for c in r.get("candidates", [])) >= k
                     for r in rows)
    n_powered = sum(1 for r in rows
                    if r.get("supp_b") is not None and r["supp_b"] > tau_pow)
    n_pow_and_res = sum(1 for r in rows
                        if r.get("supp_b") is not None and r["supp_b"] > tau_pow
                        and sum(survives(r, c, base_eps)
                                for c in r.get("candidates", [])) >= k)

    calib = dict(run=args.run, k=k, n_claims=len(rows),
                 n_candidates=len(cands), n_measured_ok=len(ok),
                 eps_ov=base_eps, eps_ov_ladder=ladder_tbl,
                 eps_ov_criterion=dict(
                     rule="first ladder rung with frac_claims_ge_k >= 0.8 "
                          "(harness rule - spec filter 2 prescribes per-claim "
                          "widening with no cap and no coverage criterion)",
                     met=eps_criterion_met,
                     note=None if eps_criterion_met else
                          "NO RUNG QUALIFIED - eps_ov is the ladder-cap "
                          "fallback, i.e. a failed calibration, not a "
                          "calibrated value"),
                 coverage_at_proposed=dict(
                     n_resolved_ge_k=n_resolved,
                     n_unresolved=len(rows) - n_resolved,
                     n_powered=n_powered,
                     n_powered_and_resolved=n_pow_and_res),
                 power_alternatives=dict(
                     note="spec Step 7 says only 'clearly positive'; all three "
                          "constructions below are harness operationalizations",
                     pooled_sum_q95=dict(tau=round(tau_pow, 4), n_pass=n_powered),
                     pooled_per_token_q95=dict(tau=round(tau_pow_tok, 4),
                                               n_pass=int(n_pow_tok)),
                     per_family_q95=dict(n_pass=int(fam_pow), n_eligible=int(fam_tot),
                                         needs="own family >= k measured decoys"),
                     supp_star_null=dict(
                         median=round(_pct(supp_star, 50), 4),
                         frac_positive=round(float(np.mean([s > 0 for s in supp_star])), 3),
                         caveat="NOT near zero (spec validity premise); "
                                "theme-matched decoys inherit topical support "
                                "from h; span-sum length-confounded")),
                 eps_pr=round(eps_pr, 4), eps_pr_pcts={q: round(_pct(pr_gap, q), 4)
                                                       for q in (50, 80, 90, 95)},
                 fve_tol=round(fve_tol, 4), fve_tol_pcts={q: round(_pct(fve_gap, q), 4)
                                                          for q in (50, 80, 90, 95)},
                 tau_pow=round(tau_pow, 4),
                 supp_star_pcts={q: round(_pct(supp_star, q), 4)
                                 for q in (5, 50, 95, 99)},
                 tau_val=round(tau_val, 4),
                 kappa_0=kappa_0, kappa_grid=grid,
                 n_lodo_families=len(fam),
                 note="proposed by calib_cycle.py --analyze; user sign-off required")
    _atomic_write(out_dir / "calibration_proposed.json", calib)

    epsc = ("met" if eps_criterion_met else
            "NOT MET at any rung - eps_ov below is the ladder-cap FALLBACK "
            "(failed calibration, not a calibrated value)")
    lines = ["# S2 decoy audit table", "",
             f"run `{args.run}`  claims {len(rows)}  candidates {len(cands)} "
             f"(measured ok {len(ok)})", "",
             f"proposed: eps_ov={base_eps} eps_pr={eps_pr:.3f} fve_tol={fve_tol:.3f} "
             f"tau_pow={tau_pow:.3f} tau_val={tau_val:.3f} kappa_0={kappa_0}", "",
             f"eps_ov 80%-coverage rule (harness rule, not spec): {epsc}", "",
             f"coverage at proposed thresholds: resolved(>=k) {n_resolved}/{len(rows)}, "
             f"unresolved {len(rows) - n_resolved}/{len(rows)}, "
             f"powered {n_powered}/{len(rows)}, "
             f"powered AND resolved {n_pow_and_res}/{len(rows)}", "",
             "| idx.ci | type | claim | gen | text | f2-4 pass | supp_b | note |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        n_surv = sum(survives(r, c, base_eps) for c in r.get("candidates", []))
        note = "" if n_surv >= k else f"UNRESOLVED (<{k} decoys)"
        if not r.get("located_fc", True):
            note = "claim span unlocated (fc)"
        ctext = " ".join(r["claim_text"][:60].split())     # markdown-safe cell
        lines.append(f"| {r['idx']}.{r['claim_i']} | {r['claim_type']} "
                     f"| {ctext} | {r['n_candidates']} "
                     f"| {r['n_text_pass']} | {n_surv} "
                     f"| {r['supp_b'] if r['supp_b'] is not None else 'nan'} | {note} |")
    (out_dir / "DECOY_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({k2: v for k2, v in calib.items()
                      if k2 not in ("kappa_grid", "eps_ov_ladder")}, indent=1))
    print(f"\nwrote {out_dir}/calibration_proposed.json and DECOY_AUDIT.md")


def analyze_signed(args):
    """SIGNED package (S2 hard-stop sign-off 2026-07-21), all mechanical:
      - f3/f4 bands at the P90 of the pooled mismatch marginals (was p80);
      - filter 2: spec-literal PER-CLAIM widening over EPS_OV_LADDER_SIGNED,
        run on each claim's f3/f4-passing pool, with the per-claim eps logged;
        after the finite ladder the band is DISABLED (spec puts no cap), so a
        claim is unresolved only when its pool itself has < k usable decoys;
      - power: per-family - supp_b > Q95 of the claim's own SURVIVING decoys'
        supp* (matched null; the pooled tau_pow is reported for comparison);
      - kappa_0: PROVISIONAL LODO re-fit on the resulting survivor families.
    Writes calibration_signed.json + DECOY_AUDIT.md (per-claim eps + power)."""
    out_dir, rows = _load_rows(args)
    k = args.k
    P_BAND = 90

    cands = [(r, c) for r in rows for c in r.get("candidates", [])
             if c.get("status") in ("ok", "degenerate decoy direction")]
    ok = [(r, c) for r, c in cands if c.get("status") == "ok"]

    pr_gap = [abs(c["mean_lbar_star"] - r["mean_lbar_b"]) for r, c in ok
              if c.get("mean_lbar_star") is not None and r.get("mean_lbar_b") is not None]
    fve_gap = [abs(c["fve_star"] - r["fve_z"]) for r, c in ok
               if c.get("fve_star") is not None and r.get("fve_z") is not None]
    eps_pr = args.eps_pr if args.eps_pr is not None else _pct(pr_gap, P_BAND)
    fve_tol = args.fve_tol if args.fve_tol is not None else _pct(fve_gap, P_BAND)
    assert np.isfinite(eps_pr) and np.isfinite(fve_tol), (
        "no measured candidates to set eps_pr/fve_tol from - a nan band would "
        "fail OPEN; measure first or pass --eps-pr/--fve-tol")

    def f34_pool(r):
        """ok candidates passing filters 3+4 at the p90 bands - the pool the
        per-claim filter-2 widening runs over. Missing z-side quantities
        (mean_lbar_b / fve_z) empty the pool: fail closed, never open."""
        out = []
        for c in r.get("candidates", []):
            if c.get("status") != "ok":
                continue
            if None in (c.get("mean_lbar_star"), r.get("mean_lbar_b"),
                        c.get("fve_star"), r.get("fve_z")):
                continue
            if abs(c["mean_lbar_star"] - r["mean_lbar_b"]) > eps_pr:
                continue
            if abs(c["fve_star"] - r["fve_z"]) <= fve_tol:
                out.append(c)
        return out

    nan = float("nan")
    per_claim, fam_dstar = [], []
    for r in rows:
        pool = f34_pool(r)
        ovb = r["ov_b"] if r.get("ov_b") is not None else nan
        ovs = [c["ov"] if c.get("ov") is not None else nan for c in pool]
        w = cc.widen_eps_until_k(ovb, ovs, k, cc.EPS_OV_LADDER_SIGNED,
                                 unbounded_final=True)
        surv = [c for c, m in zip(pool, w["mask"]) if m]
        power = cc.precheck_power_family(
            r["supp_b"] if r.get("supp_b") is not None else nan,
            [c["supp_star"] if c.get("supp_star") is not None else nan
             for c in surv])
        resolved = len(surv) >= k
        d = [c["dstar"] for c in surv if c.get("dstar") is not None]
        if len(d) > k:                       # LODO needs delete-one headroom
            fam_dstar.append(np.asarray(d, np.float64))
        per_claim.append(dict(
            idx=r["idx"], claim_i=r["claim_i"], claim_type=r["claim_type"],
            claim_text=r["claim_text"], located_fc=r.get("located_fc", True),
            n_candidates=r.get("n_candidates"), n_text_pass=r.get("n_text_pass"),
            n_pool_f34=len(pool), eps_used=w["eps_used"],
            band_disabled=w["band_disabled"], capped=w["capped"],
            n_surv=len(surv), resolved=resolved,
            supp_b=r.get("supp_b"), power=power))

    # provisional kappa_0: same LODO construction as the legacy analysis, on
    # the signed-package survivor families (re-fit again before the S3 gates)
    grid, kappa_0 = [], None
    for kap in np.arange(0.0, 5.01, 0.1):
        n_pass = n_tot = 0
        for d in fam_dstar:
            for j in range(d.size):
                rest = np.delete(d, j)
                n_pass += d[j] > np.percentile(rest, 95) + kap
                n_tot += 1
        # None (not nan) when there are no LODO families - json.dumps would
        # otherwise write a bare NaN token (invalid strict JSON)
        rate = (n_pass / n_tot) if n_tot else None
        grid.append(dict(kappa=round(float(kap), 2),
                         lodo_pass_rate=round(rate, 4) if rate is not None else None))
        if kappa_0 is None and rate is not None and rate <= 0.05:
            kappa_0 = float(kap)

    # tau_val unchanged by the package: pooled Q99 of |supp*| (validity 2)
    supp_star = [c["supp_star"] for _, c in ok if c.get("supp_star") is not None]
    tau_val = _pct([abs(s) for s in supp_star], 99)
    tau_pow_legacy = max(0.0, _pct(supp_star, 95))

    n_resolved = sum(p["resolved"] for p in per_claim)
    n_powered = sum(p["power"]["passed"] for p in per_claim)
    n_pow_res = sum(p["resolved"] and p["power"]["passed"] for p in per_claim)
    n_band_off = sum(p["band_disabled"] for p in per_claim)
    n_capped = sum(p["capped"] for p in per_claim)
    eps_hist = {}
    for p in per_claim:
        key = ("band_disabled" if p["band_disabled"] else
               "no_pool" if p["eps_used"] is None else str(p["eps_used"]))
        eps_hist[key] = eps_hist.get(key, 0) + 1

    calib = dict(
        run=args.run, k=k, n_claims=len(rows),
        n_candidates=len(cands), n_measured_ok=len(ok),
        package=dict(
            signed="S2 hard-stop sign-off 2026-07-21",
            p_band=P_BAND,
            filter2="per-claim widening over EPS_OV_LADDER_SIGNED, band "
                    "disabled as final fallthrough (eps_used=None)",
            power="per-family: supp_b > Q95 of own SURVIVING decoys' supp*",
            kappa="PROVISIONAL - LODO on the signed families, re-fit at S3",
            decoy_pool="d4 prompt, g=24"),
        eps_pr=round(eps_pr, 4),
        eps_pr_pcts={q: round(_pct(pr_gap, q), 4) for q in (50, 80, 90, 95)},
        fve_tol=round(fve_tol, 4),
        fve_tol_pcts={q: round(_pct(fve_gap, q), 4) for q in (50, 80, 90, 95)},
        eps_ov_ladder=list(cc.EPS_OV_LADDER_SIGNED),
        coverage=dict(n_resolved_ge_k=n_resolved,
                      n_unresolved=len(rows) - n_resolved,
                      n_powered_family=n_powered,
                      n_powered_and_resolved=n_pow_res,
                      n_band_disabled=n_band_off, n_capped=n_capped),
        eps_histogram=eps_hist,
        tau_val=round(tau_val, 4),
        legacy_comparison=dict(
            tau_pow_pooled_q95=round(tau_pow_legacy, 4),
            n_powered_pooled=sum(1 for r in rows if r.get("supp_b") is not None
                                 and r["supp_b"] > tau_pow_legacy),
            supp_star_median=round(_pct(supp_star, 50), 4),
            supp_star_frac_positive=round(
                float(np.mean([s > 0 for s in supp_star])), 3),
            note="pooled construction for comparison with the d3g12 "
                 "calibration_proposed.json; NOT the signed power rule"),
        kappa_0=kappa_0, kappa_grid=grid, n_lodo_families=len(fam_dstar),
        per_claim=per_claim,
        note="signed package (S2 sign-off 2026-07-21); kappa_0 provisional")
    _atomic_write(out_dir / "calibration_signed.json", calib)

    lines = ["# S2b decoy audit table (signed package)", "",
             f"run `{args.run}`  claims {len(rows)}  candidates {len(cands)} "
             f"(measured ok {len(ok)})", "",
             f"signed: p_band={P_BAND} eps_pr={eps_pr:.4f} fve_tol={fve_tol:.4f} "
             f"tau_val={tau_val:.3f} kappa_0={kappa_0} (provisional) "
             f"power=per-family Q95", "",
             f"coverage: resolved(>={k}) {n_resolved}/{len(rows)}, "
             f"unresolved {len(rows) - n_resolved}/{len(rows)}, "
             f"powered(family) {n_powered}/{len(rows)}, "
             f"powered AND resolved {n_pow_res}/{len(rows)}, "
             f"band-disabled {n_band_off}, capped {n_capped}", "",
             "| idx.ci | type | claim | gen | text | pool | eps_ov | surv "
             "| power | supp_b | note |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for p in per_claim:
        eps_cell = "band-off" if p["band_disabled"] else (
            f"{p['eps_used']:g}" if p["eps_used"] is not None else "-")
        pw = p["power"]
        pw_cell = ("n/a" if pw["margin"] is None else
                   f"{'pass' if pw['passed'] else 'fail'} "
                   f"{pw['margin']:+.1f}")
        note = "" if p["resolved"] else f"UNRESOLVED (<{k} decoys)"
        if not p["located_fc"]:
            note = "claim span unlocated (fc)"
        ctext = " ".join(p["claim_text"][:60].split())
        sb = f"{p['supp_b']:.2f}" if p["supp_b"] is not None else "nan"
        lines.append(f"| {p['idx']}.{p['claim_i']} | {p['claim_type']} "
                     f"| {ctext} | {p['n_candidates']} | {p['n_text_pass']} "
                     f"| {p['n_pool_f34']} | {eps_cell} | {p['n_surv']} "
                     f"| {pw_cell} | {sb} | {note} |")
    (out_dir / "DECOY_AUDIT.md").write_text("\n".join(lines) + "\n",
                                            encoding="utf-8")
    print(json.dumps({k2: v for k2, v in calib.items()
                      if k2 not in ("kappa_grid", "per_claim")}, indent=1))
    print(f"\nwrote {out_dir}/calibration_signed.json and DECOY_AUDIT.md")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("key", help="model key (repl_common.MODELS)")
    ap.add_argument("--run", required=True, help="atom run id (records + vectors)")
    ap.add_argument("--runs-dir", default=os.environ.get(
        "CLAIM_ABLATION_RUNS", str(_ATHENA / "runs")))
    # defaults point at the d4/g24 versioned dirs; the frozen d3g12 artifacts
    # (decoys/, calib/) are reached only by explicit flags, never by accident
    ap.add_argument("--decoys-dir", default=str(_ATHENA / "decoys_d4g24"))
    ap.add_argument("--out", default=str(_ATHENA / "calib_d4g24"))
    ap.add_argument("--gold-dir", default=None)
    ap.add_argument("--model", default="sonnet", help="decoy cache model key")
    ap.add_argument("--prompt-version", default="d4",
                    help="decoy_gen.PROMPT_VERSION the caches must carry - "
                         "stale-cache guard (pinned here because decoy_gen "
                         "needs the anthropic SDK, absent on the GPU box)")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--ar-cache", default=None,
                    help="phase-split: read AR directions from this dir "
                         "(ar_precompute.py output) instead of loading the "
                         "critic; for models whose AV+AR don't co-reside (70B)")
    ap.add_argument("--idx", type=int, action="append")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--analyze", action="store_true",
                    help="signed-package analysis -> calibration_signed.json")
    ap.add_argument("--analyze-legacy", action="store_true",
                    help="pre-sign-off pooled analysis (d3g12 re-derivation; "
                         "pass --out .../calib) -> calibration_proposed.json")
    ap.add_argument("--k", type=int, default=5, help="target decoy family size")
    ap.add_argument("--eps-pr", type=float, default=None)
    ap.add_argument("--fve-tol", type=float, default=None)
    args = ap.parse_args()
    if args.analyze_legacy:
        analyze_legacy(args)
    elif args.analyze:
        analyze_signed(args)
    else:
        measure(args)


if __name__ == "__main__":
    main()
