"""Exp 7 — resample the AV verbalizer on FROZEN activations (GPU). Injection path reused VERBATIM.

Unlike Exp 1-6 (edit one fixed explanation, re-run only the AR), Exp 7 re-runs AV GENERATION many times
on the SAME frozen gold activation and asks how stable the verbalization is across resamples & temperature.
Everything that defines the random variable is held fixed -- gold[idx] (never re-extracted), read
position/token, injection_scale, marker position, actor prompt, top_p, max_new_tokens -- and ONLY the
sampler RNG + temperature vary. The injection hook, prompt builder, layer finder and vector normalization
are imported unchanged from fve_ufw.py so the read is byte-identical to the headline runs.

Three arms:
  main    inject gold[idx]                 -- the real read; full temperature grid
  floor   inject NOTHING (hook pass-through, marker token present, activation absent)
          -- the no-injection scaffold floor (H4): how stable is the AV with zero info from h?
  shuffle inject a DIFFERENT subsample doc's gold (deterministic derangement, different doc_id)
          -- if the "core" persists it is an AV prior, not activation-specific.

Grid: T=0 is drawn ONCE (greedy, do_sample=False) -- a deterministic decoder resampled N times would
manufacture a fake core. T>0 draws N stochastic samples. floor/shuffle run at T=0.4 only (N).

Resume-safe: checkpoints per arm; already-generated (idx,T,k) keys are skipped. Pilot outputs use
separate ``*_pilotK.json`` names so a completed pilot cannot make the later full run look complete.

Usage:
  python fve_resample.py <key> --arm main    [--pilot 10] [--batch 48]
  python fve_resample.py <key> --arm floor   [--pilot 10]
  python fve_resample.py <key> --arm shuffle [--pilot 10]
  key in {qwen7b, gemma12b, gemma27b}
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data/agastyas/nla_fve_ufw")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "code"))
from repl_common import MODELS                                  # noqa: E402
from nla_inference import normalize_activation                 # noqa: E402
from huggingface_hub import snapshot_download                  # noqa: E402
# injection path reused VERBATIM from the headline harness:
from fve_ufw import (BatchInjector, build_actor_prompt, find_layers, read_sidecar,  # noqa: E402
                     load_av, outdir, CJK, EXPLANATION_RE, TOP_P, MAX_NEW)

RES = ROOT / "results"
N_RESAMPLE = 24
GRID_MAIN = [0.0, 0.4, 0.7, 1.0, 1.5]
GRID_CTRL = [0.4]                     # floor / shuffle at the harness-default temperature only
CKPT_EVERY = 8


def _seed_int(*parts) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def donor_map(docs: list[dict]) -> dict:
    """idx -> a DIFFERENT subsample doc's idx (different doc_id), deterministic derangement."""
    n = len(docs)
    assert n >= 2, f"shuffle control needs >=2 docs (got {n}); use --pilot >= 2"
    off = max(1, n // 2 + (_seed_int("exp7-shuffle") % 97))
    dm = {}
    for i, d in enumerate(docs):
        j = (i + off) % n
        for _ in range(n):
            if docs[j]["doc_id"] != d["doc_id"]:
                break
            j = (j + 1) % n
        dm[d["idx"]] = docs[j]["idx"]
    assert all(dm[d["idx"]] != d["idx"] for d in docs), "donor_map self-mapped (not a derangement)"
    return dm


def _norm_vec(gold, idx, inj_scale):
    return normalize_activation(torch.as_tensor(gold[idx], dtype=torch.float32).view(1, -1), inj_scale)[0]


@torch.inference_mode()
def run_arm(key: str, arm: str, batch: int, pilot: int | None) -> None:
    cfg = MODELS[key]
    od = outdir(key)
    tag = f"resample_{arm}" + (f"_pilot{pilot}" if pilot is not None else "")
    if (od / f"{tag}.json").exists():
        print(f"[{tag}:{key}] resume: {tag}.json present, skip", flush=True)
        return
    assert batch >= N_RESAMPLE, f"--batch {batch} < N_RESAMPLE {N_RESAMPLE} (co-batching regime)"
    gold = np.load(od / "gold.npy")
    sub = json.loads((RES / f"exp7_subsample_{key}.json").read_text())
    docs = sub["docs"][:pilot] if pilot else sub["docs"]
    by_idx = {d["idx"]: d for d in docs}
    grid = GRID_MAIN if arm == "main" else GRID_CTRL
    dm = donor_map(docs) if arm == "shuffle" else {}

    av_dir = snapshot_download(cfg["av"])
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(av_dir, trust_remote_code=True)
    sc = read_sidecar(av_dir)
    model = load_av(cfg, av_dir)
    ids, p = build_actor_prompt(tok, sc, cfg["family"])
    prompt_len = len(ids)
    dev = model.get_input_embeddings().weight.device
    inj = BatchInjector(p, prompt_len)
    find_layers(model)[0].register_forward_pre_hook(inj, with_kwargs=True)
    eos = tok.eos_token_id
    print(f"[{tag}:{key}] AV on {dev}; inj_scale={sc['injection_scale']} prompt_len={prompt_len} "
          f"marker@{p} docs={len(docs)} grid={grid} N(T>0)={N_RESAMPLE}", flush=True)

    partial = od / f"{tag}_partial.json"
    samples = json.loads(partial.read_text())["samples"] if partial.exists() else {}

    # Batch over the FULL (non-pruned) per-temperature grid so the batch ORDINAL is resume-stable:
    # the sampler seed keys on (arm, T, batch_ordinal), which is identical across a fresh vs resumed
    # run. Fully-done batches are skipped; a partially-done batch is regenerated with the same seed +
    # identical composition -> byte-identical outputs (overwrites stored rows with equal values).
    n_regen = 0
    for T in grid:
        greedy = (arm == "main" and T == 0.0)
        nk = 1 if greedy else N_RESAMPLE
        full_rows = [(d["idx"], T, k, f"{d['idx']}|{T}|{k}") for d in docs for k in range(nk)]
        for bi, b0 in enumerate(range(0, len(full_rows), batch)):
            chunk = full_rows[b0:b0 + batch]
            if all(kstr in samples for _, _, _, kstr in chunk):
                continue                                       # fully done -> resume-skip
            if arm == "floor":
                inj.v = None                                   # pass-through: marker present, h absent
            else:
                src = (lambda i: i) if arm == "main" else (lambda i: dm[i])
                inj.v = torch.stack([_norm_vec(gold, src(i), sc["injection_scale"])
                                     for i, _, _, _ in chunk]).to(dev)
                assert inj.v.shape[0] == len(chunk)
            ids_b = torch.tensor([ids] * len(chunk), dtype=torch.long, device=dev)
            torch.manual_seed(_seed_int(arm, T, bi))           # resume-stable: ordinal over FULL grid
            gkw = dict(max_new_tokens=MAX_NEW, pad_token_id=eos)
            if greedy:
                out = model.generate(input_ids=ids_b, do_sample=False, **gkw)
            else:
                out = model.generate(input_ids=ids_b, do_sample=True, temperature=T, top_p=TOP_P, **gkw)
            for r, (i, _T, k, kstr) in enumerate(chunk):
                gen = tok.decode(out[r][prompt_len:], skip_special_tokens=True)
                mt = EXPLANATION_RE.search(gen)
                text = mt.group(1).strip() if mt else gen.strip()
                cjk, noexp, empty = bool(CJK.search(text)), mt is None, (text == "")
                d = by_idx[i]
                samples[kstr] = dict(idx=i, T=_T, k=k, text=text, cjk=cjk, noexp=noexp, empty=empty,
                                     stratum=d["stratum"], divergence=d["divergence"])
            n_regen += len(chunk)
            if bi % CKPT_EVERY == 0 or b0 + len(chunk) >= len(full_rows):
                partial.write_text(json.dumps({"samples": samples}, ensure_ascii=False), encoding="utf-8")
                print(f"  [{tag} T={T} {b0+len(chunk)}/{len(full_rows)}] regen={n_regen} "
                      f"have={len(samples)} last={samples[chunk[-1][3]]['text'][:60]!r}", flush=True)

    # counters + per-(temperature) effective-N recomputed from the FINAL set (resume-safe)
    from collections import Counter
    per_T, n_cjk, n_noexp, n_empty = {}, 0, 0, 0
    for s in samples.values():
        c = per_T.setdefault(str(s["T"]), Counter())
        bad = s["cjk"] or s["noexp"] or s.get("empty", s["text"] == "")
        c["n"] += 1; c["cjk"] += s["cjk"]; c["noexp"] += s["noexp"]
        c["empty"] += s.get("empty", s["text"] == ""); c["effective"] += (0 if bad else 1)
        n_cjk += s["cjk"]; n_noexp += s["noexp"]; n_empty += s.get("empty", s["text"] == "")

    out_obj = dict(key=key, arm=arm, grid=grid, n_resample=N_RESAMPLE, n_docs=len(docs),
                   top_p=TOP_P, max_new=MAX_NEW, n_cjk=n_cjk, n_noexp=n_noexp, n_empty=n_empty,
                   per_T={t: dict(c) for t, c in per_T.items()},
                   provenance=dict(inj_scale=sc["injection_scale"], marker_pos=p, prompt_len=prompt_len,
                                   batch=batch, subsample_seed=sub.get("seed"), bounds=sub.get("bounds"),
                                   seed_scheme="torch.manual_seed(sha256(arm|T|batch_ordinal_over_full_grid))"),
                   samples=samples)
    (od / f"{tag}.json").write_text(json.dumps(out_obj, ensure_ascii=False), encoding="utf-8")
    if partial.exists():
        partial.unlink()
    eff_by_T = {t: c["effective"] for t, c in per_T.items()}
    print(f"[{tag}:{key}] wrote {tag}.json ({len(samples)} samples); CJK {n_cjk} noexp {n_noexp} "
          f"empty {n_empty}; effective-N per T={eff_by_T}", flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("key", choices=list(MODELS))
    ap.add_argument("--arm", choices=["main", "floor", "shuffle"], default="main")
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--pilot", type=int, default=None, help="limit to first K subsample docs")
    args = ap.parse_args()
    run_arm(args.key, args.arm, args.batch, args.pilot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
