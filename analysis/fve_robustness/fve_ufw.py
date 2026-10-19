"""Mean FVE for one released open-weight NLA on held-out Ultra-FineWeb docs.

Single-file, phase-checkpointed (resume-safe). Four phases, each writes an
artifact and is skipped if that artifact already exists:

  A  extract   base model, one forward per doc, gold = hidden_states[K+1][pos]
                (pos = 1 random non-special token >= 50, per-doc keyed RNG,
                 mirroring nla.datagen.stage0_extract._sample_positions)
  B  avgen     AV verbalizer, BATCHED generation (identical actor prompt, the
                gold vector injected at the marker via a layer-0 pre-hook at the
                sidecar injection_scale — exactly as training injects). temp 0.4.
  C  score     AR reconstructor (NLACritic): cos(AR(read), gold) per doc.
  D  fve        FVE = 1 - sum_i 2(1-cos_i) / sum_i ||ghat_i - mean(ghat)||^2
                on unit-normalized golds (the paper metric; matches athena/repl/fve.py).

Usage:
  python fve_ufw.py <key> [--n 2000] [--batch 64] [--seed 42]
  key in {qwen7b, gemma12b, gemma27b, llama70b}

GPU: gpurun pins one GPU for qwen/gemma12b/gemma27b (phase-split frees each model
before loading the next). For llama70b run detached with CUDA_VISIBLE_DEVICES=2,3
(device_map='auto' shards base/AV/AR across both).
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import AutoTokenizer

sys.path.insert(0, "/data/agastyas/nla_fve_ufw/code")
from repl_common import MODELS, FAMILIES          # noqa: E402
from nla_inference import NLACritic, normalize_activation  # noqa: E402
from huggingface_hub import snapshot_download      # noqa: E402

ROOT = Path("/data/agastyas/nla_fve_ufw")
DOCS = ROOT / "data" / "ufw_docs.json"
MAX_LEN = 2048
MIN_POSITION = 50            # matches stage0_extract._MIN_POSITION
TEMPERATURE = 0.4            # matches the athena harness / NP corpus default
TOP_P = 0.95
MAX_NEW = 200
CKPT_EVERY = 8               # batches between partial-checkpoint writes

EXPLANATION_RE = re.compile(r"<explanation>\s*(.*?)\s*</explanation>", re.DOTALL)
CJK = re.compile(r"[　-鿿豈-﫿＀-￯]")


def outdir(key: str) -> Path:
    d = ROOT / f"out_{key}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _free(model) -> None:
    del model
    gc.collect()
    torch.cuda.empty_cache()


def load_base(cfg):
    from transformers import AutoModelForCausalLM
    kw = dict(dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True,
              trust_remote_code=True)
    if cfg["gemma"]:
        from transformers import AutoModelForImageTextToText
        try:
            return AutoModelForImageTextToText.from_pretrained(cfg["base"], **kw).eval()
        except Exception as e:
            print(f"[load_base] ImageTextToText failed ({e}); trying CausalLM")
    return AutoModelForCausalLM.from_pretrained(cfg["base"], **kw).eval()


def load_av(cfg, av_dir):
    from transformers import AutoModelForCausalLM
    kw = dict(dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True,
              trust_remote_code=True)
    if cfg["gemma"]:
        from transformers import AutoModelForImageTextToText
        try:
            return AutoModelForImageTextToText.from_pretrained(av_dir, **kw).eval()
        except Exception as e:
            print(f"[load_av] ImageTextToText failed ({e}); trying CausalLM")
    return AutoModelForCausalLM.from_pretrained(av_dir, **kw).eval()


def read_sidecar(av_dir: str) -> dict:
    meta = yaml.safe_load((Path(av_dir) / "nla_meta.yaml").read_text())
    t = meta["tokens"]
    return dict(
        injection_char=t["injection_char"], inj_id=t["injection_token_id"],
        left=t["injection_left_neighbor_id"], right=t["injection_right_neighbor_id"],
        actor_template=meta["prompt_templates"].get("av") or meta["prompt_templates"]["actor"],
        injection_scale=float(meta["extraction"]["injection_scale"]),
    )


def build_actor_prompt(tok, sc, family):
    """Fixed verbalizer prompt with family chat markers; marker is a standalone
    token flanked by the sidecar neighbors (matches training + avgen_model.py)."""
    F = FAMILIES[family]
    content = sc["actor_template"].format(injection_char=sc["injection_char"])
    prompt = F["single"]("user", content) + F["turn_end"] + F["single"](F["assistant"], "")
    ids = tok(prompt, add_special_tokens=True)["input_ids"]
    p = [i for i, x in enumerate(ids) if x == sc["inj_id"]]
    assert len(p) == 1, f"marker appears {len(p)}x (expected 1)"
    p = p[0]
    assert ids[p - 1] == sc["left"] and ids[p + 1] == sc["right"], (
        f"neighbor mismatch: got ({ids[p-1]},{ids[p+1]}) vs sidecar ({sc['left']},{sc['right']})")
    return ids, p


def find_layers(model):
    for path in ("model.layers", "model.language_model.layers",
                 "model.model.language_model.layers", "language_model.model.layers",
                 "language_model.layers", "model.text_model.layers", "transformer.h"):
        obj, ok = model, True
        for attr in path.split("."):
            if not hasattr(obj, attr):
                ok = False
                break
            obj = getattr(obj, attr)
        if ok and hasattr(obj, "__len__") and len(obj) > 0:
            print(f"[avgen] decoder layers at {path} ({len(obj)})")
            return obj
    raise RuntimeError("could not locate decoder layers")


class BatchInjector:
    """forward_pre_hook: on the prefill pass, overwrite hidden_states[:, p] with a
    per-row activation vector. Decode steps (seq len != prompt_len) pass through."""

    def __init__(self, p: int, prompt_len: int):
        self.p, self.prompt_len, self.v = p, prompt_len, None

    def __call__(self, module, args, kwargs):
        hs = kwargs.get("hidden_states")
        positional = hs is None
        if positional:
            hs = args[0]
        if self.v is None or hs.shape[1] != self.prompt_len:
            return None
        assert hs.shape[0] == self.v.shape[0], f"batch {hs.shape[0]} != vecs {self.v.shape[0]}"
        hs = hs.clone()
        hs[:, self.p] = self.v.to(hs.dtype).to(hs.device)
        if positional:
            return ((hs,) + tuple(args[1:]), kwargs)
        kwargs["hidden_states"] = hs
        return (args, kwargs)


# ── Phase A: extract gold ────────────────────────────────────────────────────
def sample_position(ids, special_ids, doc_id, seed):
    rng = random.Random(hashlib.sha256(f"{seed}|{doc_id}".encode()).digest())
    cands = [i for i, t in enumerate(ids) if i >= MIN_POSITION and t not in special_ids]
    return rng.sample(cands, 1)[0] if cands else None


@torch.inference_mode()
def phase_extract(key, cfg, docs, seed):
    od = outdir(key)
    if (od / "gold.npy").exists() and (od / "manifest.json").exists():
        print(f"[extract:{key}] resume: gold.npy + manifest.json present, skip")
        return
    tok = AutoTokenizer.from_pretrained(cfg["base"], trust_remote_code=True)
    special_ids = set(tok.all_special_ids)
    K, D = cfg["layer"], cfg["d_model"]
    base = load_base(cfg)
    emb_dev = base.get_input_embeddings().weight.device
    print(f"[extract:{key}] base loaded on {emb_dev}; K={K} d={D} n={len(docs)}", flush=True)

    gold = np.zeros((len(docs), D), dtype=np.float32)
    manifest, n_ok = [], 0
    for i, doc in enumerate(docs):
        enc = tok(doc["text"], add_special_tokens=True, truncation=True, max_length=MAX_LEN)
        ids = enc["input_ids"]
        pos = sample_position(ids, special_ids, doc["id"], seed)
        in_range = pos is not None
        n_ok += in_range
        if in_range:
            ids_t = torch.tensor([ids], dtype=torch.long, device=emb_dev)
            out = base(input_ids=ids_t, output_hidden_states=True, use_cache=False)
            gold[i] = out.hidden_states[K + 1][0, pos].float().cpu().numpy()
        manifest.append(dict(idx=i, doc_id=doc["id"], position=int(pos) if in_range else -1,
                             n_tokens=len(ids), in_range=bool(in_range)))
        if (i + 1) % 64 == 0 or i == len(docs) - 1:
            print(f"  [{i+1}/{len(docs)}] in_range {n_ok}", flush=True)

    np.save(od / "gold.npy", gold)
    (od / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    print(f"[extract:{key}] wrote gold.npy + manifest.json ({n_ok}/{len(docs)} in range)", flush=True)
    _free(base)


# ── Phase B: batched AV reads ────────────────────────────────────────────────
@torch.inference_mode()
def phase_avgen(key, cfg, batch):
    od = outdir(key)
    if (od / "av_reads.json").exists():
        print(f"[avgen:{key}] resume: av_reads.json present, skip")
        return
    manifest = json.loads((od / "manifest.json").read_text())
    gold = np.load(od / "gold.npy")

    av_dir = snapshot_download(cfg["av"])
    tok = AutoTokenizer.from_pretrained(av_dir, trust_remote_code=True)
    sc = read_sidecar(av_dir)
    model = load_av(cfg, av_dir)
    ids, p = build_actor_prompt(tok, sc, cfg["family"])
    prompt_len = len(ids)
    dev = model.get_input_embeddings().weight.device
    print(f"[avgen:{key}] AV loaded on {dev}; inj_char={sc['injection_char']!r} "
          f"inj_scale={sc['injection_scale']} prompt_len={prompt_len} marker@{p}", flush=True)

    inj = BatchInjector(p, prompt_len)
    find_layers(model)[0].register_forward_pre_hook(inj, with_kwargs=True)
    eos = tok.eos_token_id

    partial = od / "av_reads_partial.json"
    reads = json.loads(partial.read_text())["reads"] if partial.exists() else {}
    for m in manifest:                                  # out-of-range -> empty read
        if not m["in_range"]:
            reads[str(m["idx"])] = ""
    todo = [m["idx"] for m in manifest if m["in_range"] and str(m["idx"]) not in reads]
    print(f"[avgen:{key}] {len(todo)} to generate ({len(reads)} already done)", flush=True)

    torch.manual_seed(0)
    n_cjk = n_noexp = 0
    for b0 in range(0, len(todo), batch):
        chunk = todo[b0:b0 + batch]
        vecs = torch.stack([
            normalize_activation(torch.as_tensor(gold[i], dtype=torch.float32).view(1, -1),
                                 sc["injection_scale"])[0] for i in chunk])
        inj.v = vecs.to(dev)
        ids_b = torch.tensor([ids] * len(chunk), dtype=torch.long, device=dev)
        out = model.generate(input_ids=ids_b, max_new_tokens=MAX_NEW, do_sample=True,
                             temperature=TEMPERATURE, top_p=TOP_P, pad_token_id=eos)
        for row, i in enumerate(chunk):
            gen = tok.decode(out[row][prompt_len:], skip_special_tokens=True)
            mt = EXPLANATION_RE.search(gen)
            text = mt.group(1).strip() if mt else gen.strip()
            n_noexp += mt is None
            n_cjk += bool(CJK.search(text))
            reads[str(i)] = text
        done = b0 + len(chunk)
        if (b0 // batch) % CKPT_EVERY == 0 or done >= len(todo):
            partial.write_text(json.dumps({"reads": reads}), encoding="utf-8")
            last = reads[str(chunk[-1])][:70]
            print(f"  [{done}/{len(todo)}] cjk={n_cjk} noexp={n_noexp} last={last!r}", flush=True)

    (od / "av_reads.json").write_text(
        json.dumps({"reads": reads, "n_done": len(todo), "n_cjk": n_cjk,
                    "n_noexp": n_noexp, "temperature": TEMPERATURE}, ensure_ascii=False),
        encoding="utf-8")
    if partial.exists():
        partial.unlink()
    print(f"[avgen:{key}] wrote av_reads.json; CJK {n_cjk} (injection-fail smell), "
          f"no-<explanation> {n_noexp}", flush=True)
    if n_cjk > max(1, len(todo)) * 0.1:
        print(f"[avgen:{key}] WARNING: high CJK rate — injection may be misconfigured", flush=True)
    _free(model)


# ── Phase C: AR score (cos) ──────────────────────────────────────────────────
@torch.inference_mode()
def phase_score(key, cfg):
    od = outdir(key)
    if (od / "cos.json").exists():
        print(f"[score:{key}] resume: cos.json present, skip")
        return
    manifest = json.loads((od / "manifest.json").read_text())
    gold = np.load(od / "gold.npy")
    reads = json.loads((od / "av_reads.json").read_text())["reads"]

    ar_dir = snapshot_download(cfg["ar"])
    critic = NLACritic(ar_dir, dtype=torch.bfloat16, device_map="auto")
    print(f"[score:{key}] AR loaded; mse_scale={critic.mse_scale:.3f}", flush=True)

    valid = [m["idx"] for m in manifest
             if m["in_range"] and reads.get(str(m["idx"]), "").strip()]
    partial = od / "cos_partial.json"
    cos = json.loads(partial.read_text()) if partial.exists() else {}
    todo = [i for i in valid if str(i) not in cos]
    print(f"[score:{key}] {len(valid)} valid; {len(todo)} to score", flush=True)

    for n, i in enumerate(todo):
        _, c = critic.score(reads[str(i)], gold[i])
        cos[str(i)] = round(float(c), 5)
        if (n + 1) % 64 == 0 or n == len(todo) - 1:
            partial.write_text(json.dumps(cos), encoding="utf-8")
            print(f"  [{n+1}/{len(todo)}] cos={c:+.3f}", flush=True)

    (od / "cos.json").write_text(json.dumps(cos), encoding="utf-8")
    if partial.exists():
        partial.unlink()
    print(f"[score:{key}] wrote cos.json ({len(cos)} rows)", flush=True)
    _free(critic.backbone)


# ── Phase D: FVE ─────────────────────────────────────────────────────────────
def phase_fve(key):
    od = outdir(key)
    manifest = json.loads((od / "manifest.json").read_text())
    gold = np.load(od / "gold.npy").astype(np.float64)
    cos = json.loads((od / "cos.json").read_text())
    av = json.loads((od / "av_reads.json").read_text())

    idxs = [int(k) for k in cos]                        # scored rows only
    G = gold[idxs]
    Ghat = G / np.linalg.norm(G, axis=1, keepdims=True)
    hbar = Ghat.mean(0)
    Di = ((Ghat - hbar) ** 2).sum(1)                    # per-row predict-mean variance
    Li = np.array([2.0 * (1.0 - cos[str(i)]) for i in idxs])   # 2(1-cos) = ||ghat-phat||^2
    fve = float(1.0 - Li.sum() / Di.sum())
    mean_cos = float(np.mean([cos[str(i)] for i in idxs]))

    out = dict(
        key=key, model=MODELS[key]["label"], layer=MODELS[key]["layer"],
        n_docs=len(manifest), n_scored=len(idxs),
        FVE=round(fve, 4), mean_cos=round(mean_cos, 4),
        D_variance=round(float(Di.mean()), 5),
        norm_share_in_mean=round(float(np.linalg.norm(hbar) ** 2), 4),
        n_cjk=av.get("n_cjk"), n_noexp=av.get("n_noexp"),
        temperature=TEMPERATURE, seed_note="pos-sample seed via --seed; AV temp 0.4 seed 0",
    )
    (od / f"fve_{key}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n[fve:{key}] === FVE={fve:.4f}  mean_cos={mean_cos:.4f}  "
          f"n={len(idxs)}/{len(manifest)}  D={Di.mean():.4f}  "
          f"CJK={av.get('n_cjk')}  noexp={av.get('n_noexp')} ===", flush=True)
    print(f"[fve:{key}] wrote {od/f'fve_{key}.json'}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("key", choices=list(MODELS))
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = MODELS[args.key]
    docs = json.loads(DOCS.read_text(encoding="utf-8"))[:args.n]
    print(f"[fve_ufw] key={args.key} n={len(docs)} batch={args.batch} seed={args.seed} "
          f"base={cfg['base']} av={cfg['av']}", flush=True)

    phase_extract(args.key, cfg, docs, args.seed)
    phase_avgen(args.key, cfg, args.batch)
    phase_score(args.key, cfg)
    phase_fve(args.key)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
