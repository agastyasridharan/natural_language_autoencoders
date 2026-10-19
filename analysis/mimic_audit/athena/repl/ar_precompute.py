"""Phase AR of the phase-split cycle pipeline: critic-only precompute.

For models whose AV + AR cannot co-reside on the available GPUs (the 70B NLA),
this pass loads ONLY the critic, computes cc.decoy_directions() for every
text-passing claim, and writes the result to the AR cache (ar_cache.py). The AV
phase (calib_cycle.measure / cycle_run with --ar-cache) then reads the cache and
never loads the critic.

Selection, validation and candidate ordering are IDENTICAL to the live path in
calib_cycle.measure, so the cached directions align 1:1 with what the AV phase
re-derives (asserted on load). Claims with no text-passing candidate are skipped
here exactly as the AV phase skips them before the AR call.

Usage (Athena, GPUs 2,3, critic only):
  CUDA_VISIBLE_DEVICES=2,3 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    python ar_precompute.py --key llama70b --run llama70b-l4-fineweb-atoms \
      --gold-dir out_corpus_llama70b_fineweb --decoys-dir decoys_d4g24 \
      --out ar_cache_d4g24 --prompt-version d4
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

import cycle_consistency as cc
import ar_cache as ac
from calib_cycle import (ROOT, is_valid_row, load_run_records,
                         decoy_cache_files, text_pass_candidates, _atomic_write)


def main():
    from huggingface_hub import snapshot_download
    from nla_inference import NLACritic
    from repl_common import MODELS

    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True, choices=sorted(MODELS))
    ap.add_argument("--run", required=True)
    ap.add_argument("--runs-dir", default=str(ROOT / "runs"))
    ap.add_argument("--gold-dir", default=None)
    ap.add_argument("--decoys-dir", default=str(ROOT / "decoys_d4g24"))
    ap.add_argument("--out", default=str(ROOT / "ar_cache_d4g24"))
    ap.add_argument("--model", default="sonnet", help="decoy model key (cache suffix)")
    ap.add_argument("--prompt-version", default="d4")
    ap.add_argument("--idx", type=int, nargs="*")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = MODELS[args.key]
    runs_dir = Path(args.runs_dir)
    gold_dir = Path(args.gold_dir) if args.gold_dir else ROOT / f"out_corpus_{args.key}"
    gold = np.load(gold_dir / "gold.npy")
    records = load_run_records(runs_dir, args.run)
    work = decoy_cache_files(Path(args.decoys_dir), args.run, args.model)
    if args.idx:
        work = [w for w in work if w[0] in set(args.idx)]
    if args.limit:
        work = work[:args.limit]
    if not work:
        raise SystemExit("no decoy caches found - run decoy_gen.py first")

    out_dir = Path(args.out)
    (out_dir / args.run).mkdir(parents=True, exist_ok=True)

    print(f"[ar_precompute:{args.key}] {len(work)} claims; loading AR {cfg['ar']}",
          flush=True)
    ar_dir = snapshot_download(cfg["ar"])
    critic = NLACritic(ar_dir, device_map="auto")
    S_mse = critic.mse_scale

    status = dict(key=args.key, run=args.run, n_work=len(work), phase="precompute-ar",
                  mse_scale=S_mse, done=0, skipped=0, empty=0,
                  started=time.strftime("%Y-%m-%d %H:%M:%S"))
    _atomic_write(out_dir / args.run / "status.json", status)

    t0 = time.time()
    for wi, (idx, ci, cache_path) in enumerate(work):
        if ac.has_ar_directions(out_dir, args.run, idx, ci) and not args.force:
            status["skipped"] += 1
            continue
        rec = records[idx]
        zi = rec["z_inner"]
        claim = rec["snippet_results"][ci]
        span = tuple(claim["char_span"])
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        # identical integrity gates to calib_cycle.measure
        assert cache["claim"]["text"] == claim["text"], (idx, ci, "claim text drift")
        sha = hashlib.sha256(zi.encode()).hexdigest()[:16]
        assert cache["z_inner_sha"] == sha, (idx, ci, "stale decoy cache (z_inner)")
        assert cache.get("prompt_version") == args.prompt_version, \
            (idx, ci, f"stale decoy cache (prompt {cache.get('prompt_version')})")

        keep_idx = text_pass_candidates(cache)
        if not keep_idx:                       # AV phase skips these before the AR call
            status["empty"] += 1
            continue
        cand_texts = [cache["candidates"][i]["text"] for i in keep_idx]
        h = np.asarray(gold[idx], dtype=np.float32)

        dd = cc.decoy_directions(critic, zi, span, cand_texts, h)
        ac.save_ar_directions(out_dir, args.run, idx, ci, dd, cand_texts, zi,
                              args.prompt_version, S_mse)
        status["done"] += 1
        if (wi + 1) % 5 == 0 or wi + 1 == len(work):
            status.update(elapsed_s=round(time.time() - t0, 1))
            _atomic_write(out_dir / args.run / "status.json", status)
            print(f"  [{wi + 1}/{len(work)}] idx {idx} claim {ci}: "
                  f"{len(keep_idx)} decoys cached, {time.time() - t0:.0f}s", flush=True)

    status.update(phase="done", elapsed_s=round(time.time() - t0, 1))
    _atomic_write(out_dir / args.run / "status.json", status)
    print(f"[ar_precompute:{args.key}] done: {status['done']} cached, "
          f"{status['skipped']} skipped, {status['empty']} empty, "
          f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
