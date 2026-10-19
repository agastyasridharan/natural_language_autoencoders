"""Cycle-consistency test driver (spec Steps 6-7): the three readings + the
effect matrix -> five-way gated verdict per claim. Applies the thresholds fixed
by calib_cycle.py --analyze (calibration_signed.json); computes NOTHING an LLM
sees (judge-free measurement - the LLM's only role was decoy generation and the
four decoy-level judgements, already frozen in the decoy caches).

Per activation h (one atom-run record + its decoy caches):
  1. build the teacher-forced sequence, locate each claim span twice - the atom
     run's OVERLAP spans (integrity link: the effect-matrix diagonal must
     reproduce the stored dlogp) and the cycle-mode disjoint FIRST-CHAR spans
     (spec Step 4 "its own tokens only"; these feed every reading and gate).
  2. baseline forwards: score every span under h and under h-bar (-> supp(B|h)).
  3. EFFECT MATRIX M[i][j]: ablate each real claim i's direction Delta_i (from the
     published run's vectors/<idx>.npz, cross-checked), score every span j. The
     diagonal M[j][j] is M_BB; the row max over i!=B feeds Gate 2 (ownership).
  4. for each tested claim B (one with a decoy cache): measure every text-passing
     decoy (ov, supp*, mean_lbar*, fve*), pick the survivors with the SIGNED
     package (f3/f4 at the calibrated bands -> per-claim eps_ov widening, band
     disabled as the fallthrough), then on the survivors only measure D*_m
     (diagonal) and X_m (cross: ablate the decoy, score B). power (per-family) +
     validity pre-checks, then classify_claim -> verdict.

Reuses the EXISTING measurement path (claim_ablation._inject_span_logp /
project_out, cycle_consistency.decoy_directions / first_char_ranges /
classify_claim). No parallel reimplementation.

  measure+classify (GPU, Athena):
    python cycle_run.py qwen7b --run qwen7b-cp6-fineweb-atoms \
      --runs-dir .../claim_ablation_runs --decoys-dir .../decoys_d4g24 \
      --calib .../calib_d4g24/qwen7b-cp6-fineweb-atoms/calibration_signed.json \
      --out .../cycle_runs/qwen7b-d4g24 --gold-dir .../out_corpus_qwen7b \
      [--limit 1]        # S3 smoke: one activation
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
from calib_cycle import (is_valid_row, load_run_records,          # noqa: E402
                         decoy_cache_files, text_pass_candidates,
                         _atomic_write, _f)

ROOT = Path("/data/agastyas/nla_mimic")
_EPS = 1e-8


def _load_calib(path: Path) -> dict:
    c = json.loads(path.read_text(encoding="utf-8"))
    need = ("eps_pr", "fve_tol", "tau_val", "kappa_0", "k")
    miss = [n for n in need if c.get(n) is None]
    # kappa_0 may legitimately be None if calibration found no LODO families;
    # the driver then cannot apply Gate 1 and must be told explicitly.
    if [n for n in need if n != "kappa_0" and c.get(n) is None]:
        raise SystemExit(f"calibration missing {miss} in {path}")
    return c


def measure(args):
    import torch                                                  # noqa: F401
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    import claim_ablation as ca
    import ar_cache as ac
    from avgen_model import load_av, read_sidecar
    from nla_inference import NLACritic
    from repl_common import MODELS

    calib = _load_calib(Path(args.calib))
    k = calib["k"]
    eps_pr, fve_tol = calib["eps_pr"], calib["fve_tol"]
    tau_val, kappa_0 = calib["tau_val"], calib["kappa_0"]
    if kappa_0 is None:
        if args.kappa_0 is None:
            raise SystemExit("calibration kappa_0 is null; pass --kappa-0 to "
                             "apply Gate 1, or the run cannot classify")
        kappa_0 = args.kappa_0
    ladder = tuple(calib.get("eps_ov_ladder") or cc.EPS_OV_LADDER_SIGNED)

    cfg = MODELS[args.key]
    runs_dir = Path(args.runs_dir)
    gold_dir = Path(args.gold_dir) if args.gold_dir else ROOT / f"out_corpus_{args.key}"
    gold = np.load(gold_dir / "gold.npy")
    manifest = json.loads((gold_dir / "manifest.json").read_text(encoding="utf-8"))
    valid = [i for i, m in enumerate(manifest) if is_valid_row(m)]
    records = load_run_records(runs_dir, args.run)

    # activations to run = those with decoy caches (one cache per tested claim)
    caches = decoy_cache_files(Path(args.decoys_dir), args.run, args.model)
    by_act: dict[int, list] = {}
    for idx, ci, p in caches:
        by_act.setdefault(idx, []).append((ci, p))
    act_idxs = sorted(by_act)
    if args.idx:
        act_idxs = [i for i in act_idxs if i in set(args.idx)]
    if args.limit:
        act_idxs = act_idxs[:args.limit]
    if not act_idxs:
        raise SystemExit("no activations with decoy caches to run")

    out_dir = Path(args.out) / args.run
    claims_dir = out_dir / "claims"
    claims_dir.mkdir(parents=True, exist_ok=True)

    hbar_inj = cc.corpus_hbar(gold[valid])
    ubar, _ = cc.unit_hbar_di(gold[valid])

    print(f"[cycle:{args.key}] {len(act_idxs)} activations "
          f"({sum(len(by_act[i]) for i in act_idxs)} tested claims); "
          f"calib k={k} eps_pr={eps_pr:.3f} fve_tol={fve_tol:.4f} "
          f"tau_val={tau_val:.2f} kappa_0={kappa_0}", flush=True)
    av_dir = snapshot_download(cfg["av"])
    tok = AutoTokenizer.from_pretrained(av_dir, trust_remote_code=True)
    sc = read_sidecar(av_dir)
    _, model = load_av(cfg)
    inj = ca.register_injector(model, sc, cfg["family"], tok)
    # phase-split (--ar-cache): consume ar_precompute.py directions, skip the
    # critic entirely (70B AV + AR exceed 2x H100 NVL). Effect-matrix deltas come
    # from the stored atom npz either way, so only the live decoy_directions call
    # needs the critic.
    ar_dir_cache = Path(args.ar_cache) if getattr(args, "ar_cache", None) else None
    if ar_dir_cache is None:
        ar_dir = snapshot_download(cfg["ar"])
        critic = NLACritic(ar_dir, device_map="auto")
        mse_scale = critic.mse_scale
    else:
        critic = None
        mse_scale = None
        print(f"[cycle:{args.key}] AR phase-split: reading directions from "
              f"{ar_dir_cache}", flush=True)
    S = sc["injection_scale"]
    npz_dir = runs_dir / args.run / "vectors"

    status = dict(key=args.key, run=args.run, n_activations=len(act_idxs),
                  calib=str(args.calib), kappa_0=kappa_0, alpha=args.alpha,
                  injection_scale=S, mse_scale=mse_scale,
                  ar_cache=str(ar_dir_cache) if ar_dir_cache else None,
                  done=0, phase="measuring",
                  started=time.strftime("%Y-%m-%d %H:%M:%S"))
    _atomic_write(out_dir / "status.json", status)

    t0 = time.time()
    verdict_tally: dict[str, int] = {}
    for ai, idx in enumerate(act_idxs):
        rec = records[idx]
        h = np.asarray(gold[idx], dtype=np.float32)
        zi = rec["z_inner"]
        claims = rec["snippet_results"]
        input_ids, p, offsets, full_text, cont = ca.build_tf_sequence(
            tok, sc, cfg["family"], rec["explanation"])
        assert p == inj.p, f"marker drift {p} vs {inj.p}"
        z0 = cc.locate_z_inner(full_text, zi, cont)
        spans_abs = [None if z0 < 0 else
                     (z0 + c["char_span"][0], z0 + c["char_span"][1]) for c in claims]
        r_ov = ca.atom_spans(offsets, full_text, zi,
                             [{"char_span": c["char_span"]} for c in claims], cont)
        r_fc = cc.first_char_ranges(offsets, spans_abs)

        # baselines: ONE h-forward serves both span conventions; one h-bar forward
        _, tlp_h = ca._inject_span_logp(model, inj, input_ids, h, S, r_fc)
        sums_h_ov = ca._span_sums(tlp_h, r_ov)
        sums_h_fc = ca._span_sums(tlp_h, r_fc)
        sums_lb_fc, _ = ca._inject_span_logp(model, inj, input_ids, hbar_inj, S, r_fc)

        # effect matrix: ablate each real claim's stored direction, score all spans
        n = len(claims)
        stored = np.load(npz_dir / f"{idx}.npz")["deltas"] if (
            npz_dir / f"{idx}.npz").exists() else None
        M_fc = np.full((n, n), np.nan, np.float64)     # M[i][j]: ablate i, score j
        mbb_ov = [None] * n                            # diagonal under overlap spans
        for i in range(n):
            if stored is None:
                break
            di = np.asarray(stored[i], np.float32)
            nrm = float(np.linalg.norm(di))
            if nrm < _EPS:
                continue
            hp = ca.project_out(h, di / nrm, args.alpha)
            s_fc, tlp_hp = ca._inject_span_logp(model, inj, input_ids, hp, S, r_fc)
            for j in range(n):
                if r_fc[j] is not None and np.isfinite(sums_h_fc[j]):
                    M_fc[i][j] = sums_h_fc[j] - s_fc[j]
            s_ov = ca._span_sums(tlp_hp, [r_ov[i]])[0]     # diagonal only
            if r_ov[i] is not None and np.isfinite(sums_h_ov[i]):
                mbb_ov[i] = float(sums_h_ov[i] - s_ov)

        # integrity: diagonal M_BB (overlap) vs the published atom dlogp
        l0 = [abs(mbb_ov[i] - claims[i]["dlogp"]) for i in range(n)
              if mbb_ov[i] is not None and isinstance(claims[i].get("dlogp"), (int, float))]
        l0_max = max(l0) if l0 else None
        if l0_max is not None and l0_max > 0.05:
            print(f"  [warn] idx {idx}: M_BB(overlap) vs stored dlogp drift "
                  f"max {l0_max:.4f} nats", flush=True)

        uh = h / max(float(np.linalg.norm(h)), _EPS)
        d_idx = float(((uh - ubar) ** 2).sum())
        zi0 = rec["explanation"].find(zi)
        assert zi0 >= 0, (idx, "z_inner not found in explanation")

        for ci, cache_path in by_act[idx]:
            out_path = claims_dir / f"{idx}_{ci}.json"
            if out_path.exists() and not args.force:
                continue
            claim = claims[ci]
            span = tuple(claim["char_span"])
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
            assert cache["claim"]["text"] == claim["text"], (idx, ci, "claim drift")
            assert cache.get("prompt_version") == args.prompt_version, \
                (idx, ci, f"stale decoy cache {cache.get('prompt_version')}")
            keep_idx = text_pass_candidates(cache)
            cand_texts = [cache["candidates"][i]["text"] for i in keep_idx]

            fcr = r_fc[ci]
            supp_b = (sums_h_fc[ci] - sums_lb_fc[ci]) if fcr else float("nan")
            row = dict(idx=idx, claim_i=ci, doc_id=rec.get("doc_id"),
                       claim_text=claim["text"], claim_type=claim["type"],
                       char_span=list(span), taxo=cache.get("taxo"),
                       n_text_pass=len(keep_idx), located_fc=fcr is not None,
                       supp_b=_f(supp_b), m_bb_stored=_f(claim.get("dlogp")),
                       m_bb_overlap=_f(mbb_ov[ci]), l0_check_max=_f(l0_max),
                       verdict_stored=claim.get("verdict"),
                       kappa_0=kappa_0, alpha=args.alpha)

            m_bb = float(M_fc[ci][ci]) if np.isfinite(M_fc[ci][ci]) else float("nan")
            m_ib_col = [M_fc[i][ci] for i in range(n) if i != ci]
            m_ib_fin = [x for x in m_ib_col if np.isfinite(x)]
            m_ib_max = max(m_ib_fin) if m_ib_fin else None
            row.update(m_bb=_f(m_bb), m_ib_max=_f(m_ib_max),
                       n_neighbors=len(m_ib_fin))

            if not keep_idx or fcr is None:
                row.update(status="no decoys / span unlocated", candidates=[],
                           verdict="unresolved",
                           reason="no text-passing decoys" if not keep_idx
                                  else "claim span unlocated (fc)")
                _atomic_write(out_path, row)
                verdict_tally["unresolved"] = verdict_tally.get("unresolved", 0) + 1
                status["done"] += 1
                continue

            # AR directions + overlaps for every text-passing decoy (selection)
            if ar_dir_cache is not None:
                dd = ac.load_ar_directions(ar_dir_cache, args.run, idx, ci,
                                           cand_texts, zi, args.prompt_version)
            else:
                dd = cc.decoy_directions(critic, zi, span, cand_texts, h)
            ov_b = dd["ov_b"]
            npz_c = None
            if stored is not None:
                npz_c = float(np.abs(dd["delta_b"] - stored[ci]).max())
            row.update(ov_b=_f(ov_b), cos_full=_f(dd["cos_full"]),
                       npz_delta_check=_f(npz_c))

            # per-candidate z*-side forwards: supp*, mean_lbar*, fve* (for f2/f3/f4)
            cand = []
            for m, cand_i in enumerate(keep_idx):
                c = dict(cand_i=cand_i, text=cand_texts[m], ov=_f(dd["ov"][m]),
                         fve_star=_f(cc.fve_from_cos(dd["cos_star"][m], d_idx)))
                z_star = dd["z_stars"][m]
                ss, se = dd["star_spans"][m]
                z_star_assist = (rec["explanation"][:zi0] + z_star
                                 + rec["explanation"][zi0 + len(zi):])
                try:
                    ids_s, p_s, off_s, full_s, cont_s = ca.build_tf_sequence(
                        tok, sc, cfg["family"], z_star_assist)
                    assert p_s == inj.p
                    zs0 = cc.locate_z_inner(full_s, z_star, cont_s)
                    rng_s = (cc.first_char_ranges(off_s, [(zs0 + ss, zs0 + se)])[0]
                             if zs0 >= 0 else None)
                except AssertionError:
                    rng_s = None
                if rng_s is None:
                    c.update(status="decoy span unlocated", located=False,
                             delta=dd["deltas"][m])
                    cand.append(c)
                    continue
                sl = rng_s[1] - rng_s[0]
                (s_h,), _ = ca._inject_span_logp(model, inj, ids_s, h, S, [rng_s])
                (s_lb,), _ = ca._inject_span_logp(model, inj, ids_s, hbar_inj, S, [rng_s])
                c.update(status="ok", located=True, token_span=list(rng_s),
                         n_span_tokens=sl, supp_star=_f(s_h - s_lb),
                         mean_lbar_star=_f(s_lb / sl) if sl else None,
                         s_h_star=float(s_h),      # score(B*_m | h): reused for D*
                         ids_s=ids_s, rng_s=rng_s, delta=dd["deltas"][m])
                cand.append(c)

            # SIGNED-package survivor selection (mirrors calib analyze_signed).
            # fve_z is the CLAIM's own FVE reference; hoist + guard it (a
            # degenerate AR(z) reconstruction makes cos_full non-finite ->
            # fve_from_cos nan -> _f None; comparing against None must fail
            # CLOSED like analyze_signed.f34_pool, not raise TypeError and crash
            # the whole run).
            mean_lbar_b = _f(sums_lb_fc[ci] / (fcr[1] - fcr[0])) if fcr else None
            fve_z = _f(cc.fve_from_cos(dd["cos_full"], d_idx))
            pool_mask = []
            for c in cand:
                ok = (c.get("status") == "ok"
                      and c.get("mean_lbar_star") is not None
                      and mean_lbar_b is not None and c.get("fve_star") is not None
                      and fve_z is not None
                      and abs(c["mean_lbar_star"] - mean_lbar_b) <= eps_pr
                      and abs(c["fve_star"] - fve_z) <= fve_tol)
                pool_mask.append(bool(ok))
            pool = [c for c, keep in zip(cand, pool_mask) if keep]
            ovs = [c["ov"] if c.get("ov") is not None else float("nan") for c in pool]
            w = cc.widen_eps_until_k(ov_b if ov_b is not None else float("nan"),
                                     ovs, k, ladder, unbounded_final=True)
            surv = [c for c, keep in zip(pool, w["mask"]) if keep]

            # survivors only: D*_m (diagonal) + X_m (cross: ablate decoy, score B).
            # both share the ONE decoy-ablated vector hp (same direction, alpha).
            for c in surv:
                delta_s = c["delta"]
                dn = float(np.linalg.norm(delta_s))
                if dn < _EPS:
                    c.update(dstar=None, x_m=None, degenerate=True)
                    continue
                hp = ca.project_out(h, delta_s / dn, args.alpha)
                # D*_m: ablate decoy dir, score the DECOY span B*_m in z*_m
                (s_hp,), _ = ca._inject_span_logp(model, inj, c["ids_s"], hp, S,
                                                  [tuple(c["rng_s"])])
                c["dstar"] = _f(c["s_h_star"] - s_hp)
                # X_m: SAME ablation, score the REAL claim B in the original z
                (s_bx,), _ = ca._inject_span_logp(model, inj, input_ids, hp, S, [fcr])
                c["x_m"] = _f(sums_h_fc[ci] - s_bx)

            for c in cand:                       # drop non-JSON scratch fields
                for kk in ("ids_s", "rng_s", "delta", "s_h_star"):
                    c.pop(kk, None)

            # validity aligns supp*/D* per candidate (its assert requires equal
            # lengths); pair them on candidates carrying BOTH so a degenerate
            # survivor - structurally excluded (nan ov drops it upstream) but
            # cheap to guard - can never desync the two arrays.
            scored = [c for c in surv if c.get("supp_star") is not None
                      and c.get("dstar") is not None]
            supp_stars = [c["supp_star"] for c in scored]
            dstars = [c["dstar"] for c in scored]
            x_ms = [c["x_m"] for c in scored if c.get("x_m") is not None]
            power = cc.precheck_power_family(supp_b, supp_stars)
            validity = cc.precheck_validity(supp_stars, tau_val, dstar=dstars,
                                            min_k=k)
            verdict = cc.classify_claim(power, validity, m_bb, dstars, x_ms,
                                        m_ib_max, kappa_0)
            row.update(status="ok",
                       eps_ov_used=w["eps_used"], band_disabled=w["band_disabled"],
                       n_pool_f34=len(pool), n_surv=len(surv),
                       power=power, validity=validity, **{
                           kk: verdict[kk] for kk in
                           ("verdict", "gate", "reason", "grounding_estimate",
                            "q95_dstar", "med_dstar", "gate1_bar", "med_x")
                           if kk in verdict},
                       candidates=cand)
            _atomic_write(out_path, row)
            verdict_tally[verdict["verdict"]] = verdict_tally.get(
                verdict["verdict"], 0) + 1
            status["done"] += 1

        status.update(elapsed_s=round(time.time() - t0, 1),
                      verdicts=verdict_tally)
        _atomic_write(out_dir / "status.json", status)
        print(f"  [{ai + 1}/{len(act_idxs)}] idx {idx}: "
              f"{len(by_act[idx])} claims, {time.time() - t0:.0f}s  "
              f"tally={verdict_tally}", flush=True)

    status.update(phase="done", elapsed_s=round(time.time() - t0, 1),
                  verdicts=verdict_tally)
    _atomic_write(out_dir / "status.json", status)
    print(f"[cycle:{args.key}] done: {status['done']} claims  "
          f"verdicts={verdict_tally}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("key")
    ap.add_argument("--run", required=True)
    ap.add_argument("--runs-dir", default=str(ROOT / "claim_ablation_runs"))
    ap.add_argument("--decoys-dir", default=str(ROOT / "decoys_d4g24"))
    ap.add_argument("--calib", required=True, help="calibration_signed.json")
    ap.add_argument("--out", default=str(ROOT / "cycle_runs" / "qwen7b-d4g24"))
    ap.add_argument("--gold-dir", default=None)
    ap.add_argument("--model", default="sonnet", help="decoy cache model key")
    ap.add_argument("--prompt-version", default="d4")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--ar-cache", default=None,
                    help="phase-split: read AR directions from this dir "
                         "(ar_precompute.py output) instead of loading the critic")
    ap.add_argument("--kappa-0", type=float, default=None,
                    help="override / supply kappa_0 if the calibration's is null")
    ap.add_argument("--idx", type=int, action="append")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    measure(args)


if __name__ == "__main__":
    main()
