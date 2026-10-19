"""Phase 1 (any model): extract the gold activation at each re-anchored position.

Loads the model's base, re-renders every document with that model's chat markers,
re-anchors each read position to the same content location (anchors.json), runs one
forward, and takes hidden_states[K+1] raw (no normalization). Saves gold.npy +
manifest.json to out_<key>/. Single H100 is enough for qwen/gemma.

Usage: python extract_model.py <key>     # key in {qwen7b, gemma12b, gemma27b}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/data/agastyas/nla_mimic/code")
from repl_common import MODELS, render, position_for_anchor, MAX_LEN  # noqa: E402

INPUT = "/data/agastyas/nla_mimic/data/run_input.json"
ANCHORS = "/data/agastyas/nla_mimic/data/anchors.json"
ROOT = Path("/data/agastyas/nla_mimic")


def load_base(cfg):
    from transformers import AutoModelForCausalLM
    kw = dict(dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True,
              trust_remote_code=True)
    if cfg["gemma"]:
        # gemma-3-it is multimodal; load the image-text model and use its text tower
        from transformers import AutoModelForImageTextToText
        try:
            return AutoModelForImageTextToText.from_pretrained(cfg["base"], **kw).eval()
        except Exception as e:
            print(f"[extract] AutoModelForImageTextToText failed ({e}); trying CausalLM")
    return AutoModelForCausalLM.from_pretrained(cfg["base"], **kw).eval()


@torch.inference_mode()
def main() -> int:
    key = sys.argv[1]
    cfg = MODELS[key]
    outdir = ROOT / f"out_{key}"
    outdir.mkdir(parents=True, exist_ok=True)
    items = json.loads(Path(INPUT).read_text(encoding="utf-8"))
    anchors = {a["idx"]: a for a in json.loads(Path(ANCHORS).read_text(encoding="utf-8"))}
    K, D, fam = cfg["layer"], cfg["d_model"], cfg["family"]
    print(f"[extract:{key}] {len(items)} items; base={cfg['base']} K={K} d={D}", flush=True)

    tok = AutoTokenizer.from_pretrained(cfg["base"], trust_remote_code=True)
    base = load_base(cfg)
    emb_dev = base.get_input_embeddings().weight.device
    print(f"[extract:{key}] loaded; embed device {emb_dev}", flush=True)

    gold = np.zeros((len(items), D), dtype=np.float32)
    manifest, n_ok, n_inrange = [], 0, 0
    for i, it in enumerate(items):
        a = anchors[it["idx"]]
        text, spans = render(it["payload"], fam)
        enc = tok(text, add_special_tokens=True, return_offsets_mapping=True,
                  truncation=True, max_length=MAX_LEN)
        ids, offsets = enc["input_ids"], enc["offset_mapping"]
        if a["turn_index"] is None:
            pos, tgt, aok = len(ids) - 1, -1, False     # fallback: last token
        else:
            pos, _, tgt, aok = position_for_anchor(offsets, spans, a["turn_index"], a["content_offset"])
        in_range = pos is not None and pos < len(ids)
        n_inrange += in_range
        n_ok += bool(aok and in_range)
        decoded = tok.decode([ids[pos]], skip_special_tokens=False) if in_range else ""
        if in_range:
            ids_t = torch.tensor([ids], dtype=torch.long, device=emb_dev)
            out = base(input_ids=ids_t, output_hidden_states=True, use_cache=False)
            gold[i] = out.hidden_states[K + 1][0, pos].float().cpu().numpy()
        manifest.append(dict(idx=it["idx"], theme=it["theme"], is_chat=it["is_chat"],
                             position=int(pos) if pos is not None else -1, n_tokens=len(ids),
                             in_range=bool(in_range), anchor_ok=bool(aok),
                             decoded_token=decoded, llama_token=it["token"],
                             target_char=int(tgt), cosine_stored=it["cosine_stored"],
                             mimic_text=it["mimic_text"]))
        if (i + 1) % 16 == 0 or i == len(items) - 1:
            print(f"  [{i+1}/{len(items)}] anchored_ok {n_ok}/{i+1}", flush=True)

    np.save(outdir / "gold.npy", gold)
    (outdir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    print(f"\n[extract:{key}] ANCHOR ok {n_ok}/{len(items)}, in_range {n_inrange}/{len(items)}", flush=True)
    print(f"[extract:{key}] wrote {outdir/'gold.npy'} + manifest.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
