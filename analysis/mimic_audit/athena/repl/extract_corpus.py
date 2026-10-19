"""Fresh activation set for the claim-ablation audit.

Streams a pretraining-like corpus (FineWeb for Qwen, UltraFineWeb for Gemma by
default), runs the model's BASE with output_hidden_states, and takes
hidden_states[K+1] (output of block K, the NLA's layer; the +1 is the documented
off-by-one, hidden_states[0] is the embedding) at one sampled position >= 50 per
document. Saves gold.npy [N,d] + manifest.json to out_corpus_<key>/, matching the
athena extract_model.py convention so run_claim_ablation.py can consume it.

Why fresh text (not the mimic corpus): the audit should run on activations the NLA
was not tuned to explain, so a grounded-vs-confabulated split is meaningful.

Invariants honored:
  - _MIN_POSITION = 50    (need left context; earlier positions decode to noise)
  - raw vectors, norm="none" (no normalization at extraction)
  - per-doc keyed RNG: position depends only on (seed, doc_id), reproducible

Usage:
  python extract_corpus.py qwen7b   --n 400
  python extract_corpus.py gemma12b --n 400 --dataset openbmb/UltraFineWeb
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/data/agastyas/nla_mimic/code")
from repl_common import MODELS, MAX_LEN  # noqa: E402

ROOT = Path("/data/agastyas/nla_mimic")
MIN_POSITION = 50
CTX_LEFT_CHARS = 800          # how much left context to store per read (decoded)
TEXT_CAP_CHARS = 6000         # cap stored doc text so manifest.json stays sane

# Default corpus per model family. Each entry: (hf dataset id, config/name, text column, split).
# UltraFineWeb exposes splits 'en'/'zh' (no 'train'); FineWeb uses 'train'.
DATASETS = {
    "qwen7b":   ("HuggingFaceFW/fineweb", "sample-10BT", "text", "train"),
    "gemma12b": ("openbmb/UltraFineWeb", "default", "content", "en"),
    "gemma27b": ("openbmb/UltraFineWeb", "default", "content", "en"),
    "llama70b": ("HuggingFaceFW/fineweb", "sample-10BT", "text", "train"),
}


def load_base(cfg):
    """Mirror extract_model.load_base: gemma-3-it is multimodal, load the
    image-text model and use its text tower; everyone else is a CausalLM."""
    from transformers import AutoModelForCausalLM
    kw = dict(dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True,
              trust_remote_code=True)
    if cfg["gemma"]:
        from transformers import AutoModelForImageTextToText
        try:
            return AutoModelForImageTextToText.from_pretrained(cfg["base"], **kw).eval()
        except Exception as e:
            print(f"[extract_corpus] ImageTextToText failed ({e}); trying CausalLM")
    return AutoModelForCausalLM.from_pretrained(cfg["base"], **kw).eval()


class _StopFwd(Exception):
    """Raised inside a forward hook to early-exit once block K is captured."""


def _decoder_layers(model):
    """Locate the text decoder's ModuleList of blocks across architectures
    (qwen: model.layers; gemma ITT: model.language_model.layers). Picks the
    longest '*.layers' ModuleList -- the text trunk (vision towers are shorter)."""
    best = None
    for name, mod in model.named_modules():
        if (name.split(".")[-1] == "layers" and isinstance(mod, torch.nn.ModuleList)
                and len(mod) > 1 and (best is None or len(mod) > len(best[1]))):
            best = (name, mod)
    if best is None:
        raise RuntimeError("no decoder layer ModuleList found")
    return best[1], best[0]


def doc_position(seed, doc_id, lo, hi):
    """Per-doc keyed position: depends only on (seed, doc_id), independent of stream
    order or process count. Stable across runs (hashlib, not Python's salted hash)."""
    key = int(hashlib.sha256(f"{seed}:{doc_id}".encode()).hexdigest()[:8], 16)
    return int(np.random.default_rng(key).integers(lo, hi + 1))


def _iter_docs(dataset, name, text_col, want, split="train", revision=None):
    """Yield (doc_id, text) from a streaming HF dataset. Falls back across a couple
    of common text column names so a wrong --text-col is not fatal, and across the
    available splits if the requested one is absent (e.g. UltraFineWeb = en/zh)."""
    from datasets import load_dataset
    try:
        ds = load_dataset(dataset, name=name, split=split, streaming=True, revision=revision)
    except ValueError as e:
        if "Bad split" not in str(e):
            raise
        import ast, re
        m = re.search(r"Available splits:\s*(\[.*\])", str(e))
        avail = ast.literal_eval(m.group(1)) if m else []
        pick = "en" if "en" in avail else (avail[0] if avail else split)
        print(f"[extract_corpus] split '{split}' absent ({avail}); using '{pick}'", flush=True)
        ds = load_dataset(dataset, name=name, split=pick, streaming=True, revision=revision)
    cols = [text_col, "text", "content", "raw_content"]
    seen = 0
    for i, row in enumerate(ds):
        txt = next((row[c] for c in cols if c in row and isinstance(row[c], str)
                    and row[c].strip()), None)
        if not txt:
            continue
        yield f"{dataset.split('/')[-1]}:{i}", txt
        seen += 1
        if seen >= want:
            return


def _iter_docs_local(manifest_path, want):
    """Yield (doc_id, text) from an existing corpus manifest.json. Offline doc
    source: reuse the exact documents a prior extraction stored, so no dataset
    streaming (and no HF-hub network) is needed. Keeps doc_id stable, so the
    per-doc keyed position RNG stays reproducible across models."""
    source = Path(manifest_path)
    if source.suffix == ".jsonl":
        with source.open(encoding="utf-8") as stream:
            recs = [json.loads(line) for line in stream if line.strip()]
    else:
        recs = json.loads(source.read_text(encoding="utf-8"))
    for r in recs[:want]:
        txt = r.get("text")
        if isinstance(txt, str) and txt.strip():
            yield r["doc_id"], txt


@torch.inference_mode()
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("key", choices=list(MODELS))
    ap.add_argument("--n", type=int, default=400, help="activations to extract")
    ap.add_argument("--dataset", default=None, help="override HF dataset id")
    ap.add_argument("--config", default=None, help="override dataset config/name")
    ap.add_argument("--split", default=None, help="override dataset split")
    ap.add_argument("--revision", default=None, help="immutable HF dataset revision")
    ap.add_argument("--text-col", default=None, help="override text column")
    ap.add_argument("--docs-from", default=None,
                    help="read (doc_id, text) from an existing corpus manifest.json instead of "
                         "streaming a HF dataset (fully offline; reuses the exact same documents)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-pos", type=int, default=MIN_POSITION)
    ap.add_argument("--max-len", type=int, default=MAX_LEN)
    ap.add_argument("--out", default=None, help="output dir (default out_corpus_<key>)")
    ap.add_argument("--scan-mult", type=int, default=4,
                    help="docs to scan per wanted activation (some get skipped)")
    args = ap.parse_args()

    cfg = MODELS[args.key]
    d_ds, d_cfg, d_col, d_split = DATASETS.get(
        args.key, ("HuggingFaceFW/fineweb", "sample-10BT", "text", "train"))
    dataset = args.dataset or d_ds
    name = args.config or d_cfg
    text_col = args.text_col or d_col
    split = args.split or d_split
    outdir = Path(args.out) if args.out else ROOT / f"out_corpus_{args.key}"
    outdir.mkdir(parents=True, exist_ok=True)
    K, D, fam = cfg["layer"], cfg["d_model"], cfg["family"]
    print(f"[extract_corpus:{args.key}] base={cfg['base']} K={K} d={D} "
          f"dataset={dataset}/{name}:{split} col={text_col} -> {outdir}", flush=True)

    tok = AutoTokenizer.from_pretrained(cfg["base"], trust_remote_code=True)
    base = load_base(cfg)
    emb_dev = base.get_input_embeddings().weight.device
    print(f"[extract_corpus:{args.key}] loaded; embed device {emb_dev}", flush=True)

    # Capture block K's output via a forward hook and early-exit: this gives
    # hidden_states[K+1] without running the upper layers or the LM head (whose
    # full-vocab logits OOM on gemma's 262k vocab and are never used here).
    layers, lpath = _decoder_layers(base)
    assert len(layers) > K, f"found {len(layers)} layers at {lpath}, need K={K}"
    _cap = {}

    def _grab(mod, inp, out):
        _cap["h"] = out[0] if isinstance(out, tuple) else out
        raise _StopFwd()

    layers[K].register_forward_hook(_grab)
    print(f"[extract_corpus:{args.key}] capture hook on {lpath}[{K}] of "
          f"{len(layers)} layers (skips LM head)", flush=True)

    if args.docs_from:
        doc_iter = _iter_docs_local(args.docs_from, args.n * args.scan_mult)
        print(f"[extract_corpus:{args.key}] offline doc source: {args.docs_from}", flush=True)
    else:
        doc_iter = _iter_docs(
            dataset,
            name,
            text_col,
            args.n * args.scan_mult,
            split,
            revision=args.revision,
        )

    gold = np.zeros((args.n, D), dtype=np.float32)
    manifest, kept, scanned = [], 0, 0
    for doc_id, text in doc_iter:
        scanned += 1
        if kept >= args.n:
            break
        enc = tok(text, add_special_tokens=True, return_offsets_mapping=True,
                  truncation=True, max_length=args.max_len)
        ids, offsets = enc["input_ids"], enc["offset_mapping"]
        if len(ids) < args.min_pos + 3:
            continue
        pos = doc_position(args.seed, doc_id, args.min_pos, len(ids) - 2)
        cs, ce = offsets[pos]
        if ce <= cs:                           # landed on a special token; nudge
            pos = min(pos + 1, len(ids) - 2)
            cs, ce = offsets[pos]
        ids_t = torch.tensor([ids], dtype=torch.long, device=emb_dev)
        _cap.clear()
        try:
            base(input_ids=ids_t, use_cache=False)
        except _StopFwd:
            pass
        gold[kept] = _cap["h"][0, pos].float().cpu().numpy()
        if kept % 50 == 0:
            torch.cuda.empty_cache()

        read_token = tok.decode([ids[pos]], skip_special_tokens=False)
        ctx_left = text[max(0, ce - CTX_LEFT_CHARS):ce]
        ctx_right = text[ce:ce + 120]
        manifest.append(dict(
            idx=kept, doc_id=doc_id, position=int(pos), n_tokens=len(ids),
            in_range=True, read_token=read_token,
            read_char=int(ce), context_left=ctx_left, context_right=ctx_right,
            text=text[:TEXT_CAP_CHARS], text_truncated=len(text) > TEXT_CAP_CHARS,
            h_norm=float(np.linalg.norm(gold[kept])),
        ))
        kept += 1
        if kept % 16 == 0 or kept == args.n:
            print(f"  [{kept}/{args.n}] scanned {scanned} last_token={read_token!r} "
                  f"|h|={manifest[-1]['h_norm']:.1f}", flush=True)

    gold = gold[:kept]
    np.save(outdir / "gold.npy", gold)
    (outdir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    (outdir / "extract_meta.json").write_text(json.dumps(dict(
        key=args.key, dataset=dataset, revision=args.revision, config=name, split=split, text_col=text_col,
        docs_from=args.docs_from,
        n=kept, scanned=scanned, seed=args.seed, min_pos=args.min_pos,
        max_len=args.max_len, layer=K, d_model=D, base=cfg["base"],
    ), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[extract_corpus:{args.key}] kept {kept}/{args.n} (scanned {scanned}); "
          f"wrote {outdir/'gold.npy'} + manifest.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
