"""Phase 1 (llama70b): extract the gold activation for every run_input position.

Loads the BASE model meta-llama/Llama-3.3-70B-Instruct across the visible GPUs
(device_map="auto"; set CUDA_VISIBLE_DEVICES to >=2 H100s before running — the
70B base does not fit one). For each item it re-tokenizes the payload EXACTLY as
the corpus did — raw text for raw docs, np_core.format_chat (llama3 markers) for
chat docs, then add_special_tokens=True (server-side BOS) — runs one forward, and
takes hidden_states[K+1] (K=53) at the recorded token: raw, no normalization.

ALIGNMENT GATE: it decodes the token actually sitting at `position` and compares
to the stored `token`. A high match rate proves the positions line up with the
corpus; mismatches are flagged (and dropped downstream).

Writes gold.npy [N, 8192] + manifest.json (idx-aligned) to OUTDIR. No AV, no AR.
Run via direct SSH (not gpurun) with the HF proxy/cache env exported.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

INPUT = "/data/agastyas/nla_mimic/data/run_input.json"
OUTDIR = Path("/data/agastyas/nla_mimic/out")
BASE = "meta-llama/Llama-3.3-70B-Instruct"
LAYER_K = 53                 # activation = hidden_states[54]
D_MODEL = 8192
MAX_LEN = 2048

# np_core.format_chat for the llama3 family, verbatim: each turn
# "<|start_header_id|>{role}<|end_header_id|>\n\n{content}", joined by "<|eot_id|>",
# final turn left open; BOS added by the tokenizer (add_special_tokens=True).
TURN_END = "<|eot_id|>"


def format_chat_llama(messages) -> str:
    out, n = [], len(messages)
    for i, m in enumerate(messages):
        chunk = f"<|start_header_id|>{m['role']}<|end_header_id|>\n\n{m['content']}"
        if i != n - 1:
            chunk += TURN_END
        out.append(chunk)
    return "".join(out)


@torch.inference_mode()
def main() -> int:
    items = json.loads(Path(INPUT).read_text(encoding="utf-8"))
    OUTDIR.mkdir(parents=True, exist_ok=True)
    print(f"[extract] {len(items)} items; torch {torch.__version__} "
          f"gpus={torch.cuda.device_count()}", flush=True)

    tok = AutoTokenizer.from_pretrained(BASE)
    base = AutoModelForCausalLM.from_pretrained(
        BASE, dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True).eval()
    emb_dev = base.get_input_embeddings().weight.device
    print(f"[extract] base loaded; embed device {emb_dev}", flush=True)

    gold = np.zeros((len(items), D_MODEL), dtype=np.float32)
    manifest, n_match, n_inrange = [], 0, 0
    for i, it in enumerate(items):
        payload = it["payload"]
        text = payload if isinstance(payload, str) else format_chat_llama(payload)
        enc = tok(text, return_tensors="pt", add_special_tokens=True,
                  truncation=True, max_length=MAX_LEN)
        ids = enc["input_ids"][0].tolist()
        pos = it["position"]
        in_range = pos < len(ids)
        n_inrange += in_range
        decoded = tok.decode([ids[pos]], skip_special_tokens=False) if in_range else ""
        match = in_range and decoded == it["token"]
        n_match += match
        if in_range:
            out = base(input_ids=enc["input_ids"].to(emb_dev),
                       attention_mask=enc["attention_mask"].to(emb_dev),
                       output_hidden_states=True, use_cache=False)
            gold[i] = out.hidden_states[LAYER_K + 1][0, pos].float().cpu().numpy()
        manifest.append({"idx": it["idx"], "theme": it["theme"], "position": pos,
                         "n_tokens": len(ids), "in_range": bool(in_range),
                         "stored_token": it["token"], "decoded_token": decoded,
                         "token_match": bool(match),
                         "cosine_stored": it["cosine_stored"], "mse_stored": it["mse_stored"]})
        if (i + 1) % 16 == 0 or i == len(items) - 1:
            print(f"  [{i+1}/{len(items)}] match so far {n_match}/{i+1}", flush=True)

    np.save(OUTDIR / "gold.npy", gold)
    (OUTDIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    print(f"\n[extract] ALIGNMENT: token_match {n_match}/{len(items)} "
          f"({100*n_match/len(items):.0f}%), in_range {n_inrange}/{len(items)}", flush=True)
    print(f"[extract] wrote {OUTDIR/'gold.npy'} and manifest.json", flush=True)
    if n_match < len(items):
        bad = [m for m in manifest if not m["token_match"]][:8]
        print(f"[extract] sample mismatches: "
              + "; ".join(f"idx{m['idx']} pos{m['position']} "
                          f"stored={m['stored_token']!r} got={m['decoded_token']!r}" for m in bad), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
