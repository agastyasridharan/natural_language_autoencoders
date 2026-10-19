"""Phase 1b (any model): generate the REAL AV read for each gold activation.

This is the piece the llama run skipped (it reused Neuronpedia reads). We load the
model's AV (vector->text) and inject each gold vector the way training did: a
forward PRE-hook on decoder layer 0 overwrites the marked token's hidden state
(post embedding-scaling) with the gold vector normalized to the sidecar's
injection_scale. That sidesteps the Gemma embed-scale ambiguity entirely — layer 0
sees exactly injection_scale-normalized v, as in training. CJK output = injection
failed (the loudest smoke test).

Usage: python avgen_model.py <key>
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/data/agastyas/nla_mimic/code")
from repl_common import MODELS, FAMILIES  # noqa: E402
from nla_inference import normalize_activation  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402

ROOT = Path("/data/agastyas/nla_mimic")
EXPLANATION_RE = re.compile(r"<explanation>\s*(.*?)\s*</explanation>", re.DOTALL)
CJK = re.compile(r"[　-鿿豈-﫿＀-￯]")
TEMPERATURE = 0.4          # matches the Neuronpedia corpus collection default
MAX_NEW = 200


def load_av(cfg, av_dir=None):
    kw = dict(dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True,
              trust_remote_code=True)
    av_dir = av_dir or snapshot_download(cfg["av"])
    if cfg["gemma"]:
        from transformers import AutoModelForImageTextToText
        try:
            return av_dir, AutoModelForImageTextToText.from_pretrained(av_dir, **kw).eval()
        except Exception as e:
            print(f"[avgen] ImageTextToText failed ({e}); trying CausalLM")
    from transformers import AutoModelForCausalLM
    return av_dir, AutoModelForCausalLM.from_pretrained(av_dir, **kw).eval()


def read_sidecar(av_dir):
    """Read the AV nla_meta.yaml directly. We avoid nla_inference.load_nla_config
    because its neighbor check calls apply_chat_template, which returns a
    BatchEncoding (not a flat id list) in this transformers version and breaks."""
    meta = yaml.safe_load((Path(av_dir) / "nla_meta.yaml").read_text())
    t = meta["tokens"]
    return dict(injection_char=t["injection_char"], inj_id=t["injection_token_id"],
                left=t["injection_left_neighbor_id"], right=t["injection_right_neighbor_id"],
                actor_template=meta["prompt_templates"].get("av") or meta["prompt_templates"]["actor"],
                injection_scale=float(meta["extraction"]["injection_scale"]))


def build_actor_prompt(tok, sc, family):
    """Render the actor prompt with family markers (matches training: marker stays
    a standalone token with the sidecar neighbors). Returns (ids_tensor, marker_pos)."""
    F = FAMILIES[family]
    content = sc["actor_template"].format(injection_char=sc["injection_char"])
    prompt = F["single"]("user", content) + F["turn_end"] + F["single"](F["assistant"], "")
    ids = tok(prompt, add_special_tokens=True)["input_ids"]   # BOS for gemma/llama, noop for qwen
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
        obj = model
        ok = True
        for attr in path.split("."):
            if not hasattr(obj, attr):
                ok = False
                break
            obj = getattr(obj, attr)
        if ok and hasattr(obj, "__len__") and len(obj) > 0:
            print(f"[avgen] decoder layers at {path} ({len(obj)})")
            return obj
    raise RuntimeError("could not locate decoder layers")


class Injector:
    """forward_pre_hook: overwrite hidden_states[0, p] on the prefill pass."""
    def __init__(self, p, prompt_len):
        self.p, self.prompt_len, self.v = p, prompt_len, None

    def __call__(self, module, args, kwargs):
        hs = kwargs.get("hidden_states")
        positional = hs is None
        if positional:
            hs = args[0]
        if self.v is None or hs.shape[1] != self.prompt_len:
            return None
        hs = hs.clone()
        hs[0, self.p] = self.v.to(hs.dtype).to(hs.device)
        if positional:
            return ((hs,) + tuple(args[1:]), kwargs)
        kwargs["hidden_states"] = hs
        return (args, kwargs)


@torch.inference_mode()
def main() -> int:
    key = sys.argv[1]
    cfg = MODELS[key]
    outdir = ROOT / f"out_{key}"
    manifest = json.loads((outdir / "manifest.json").read_text())
    gold = np.load(outdir / "gold.npy")
    print(f"[avgen:{key}] {len(manifest)} rows; av={cfg['av']}", flush=True)

    av_dir = snapshot_download(cfg["av"])
    tok = AutoTokenizer.from_pretrained(av_dir, trust_remote_code=True)
    sc = read_sidecar(av_dir)
    _, model = load_av(cfg)
    print(f"[avgen:{key}] AV loaded; inj_char={sc['injection_char']!r} "
          f"inj_scale={sc['injection_scale']}", flush=True)

    ids, p = build_actor_prompt(tok, sc, cfg["family"])
    prompt_len = len(ids)
    dev = model.get_input_embeddings().weight.device
    ids_t = torch.tensor([ids], dtype=torch.long, device=dev)
    print(f"[avgen:{key}] actor prompt {prompt_len} tok; marker@{p} "
          f"neighbors ok ({sc['left']},{sc['right']})", flush=True)

    inj = Injector(p, prompt_len)
    find_layers(model)[0].register_forward_pre_hook(inj, with_kwargs=True)
    eos = tok.eos_token_id

    reads, n_cjk, n_done, n_noexp = {}, 0, 0, 0
    torch.manual_seed(0)
    for i, m in enumerate(manifest):
        if not m["in_range"]:
            reads[m["idx"]] = ""
            continue
        v = torch.as_tensor(gold[i], dtype=torch.float32)
        inj.v = normalize_activation(v.view(1, -1), sc["injection_scale"])[0]
        out = model.generate(input_ids=ids_t, max_new_tokens=MAX_NEW,
                             do_sample=True, temperature=TEMPERATURE, top_p=0.95,
                             pad_token_id=eos)
        gen = tok.decode(out[0][prompt_len:], skip_special_tokens=True)
        mt = EXPLANATION_RE.search(gen)
        text = mt.group(1).strip() if mt else gen.strip()
        n_noexp += mt is None
        if CJK.search(text):
            n_cjk += 1
        reads[m["idx"]] = text
        n_done += 1
        if (i + 1) % 16 == 0 or i == len(manifest) - 1:
            print(f"  [{i+1}/{len(manifest)}] cjk={n_cjk} no_explan={n_noexp} "
                  f"last={text[:80]!r}", flush=True)

    (outdir / "av_reads.json").write_text(
        json.dumps({"reads": reads, "n_done": n_done, "n_cjk": n_cjk,
                    "n_noexp": n_noexp, "temperature": TEMPERATURE}, ensure_ascii=False),
        encoding="utf-8")
    print(f"\n[avgen:{key}] generated {n_done}; CJK {n_cjk} (injection-fail smell), "
          f"no <explanation> {n_noexp}", flush=True)
    print(f"[avgen:{key}] wrote {outdir/'av_reads.json'}", flush=True)
    if n_cjk > n_done * 0.1:
        print(f"[avgen:{key}] WARNING: high CJK rate — injection may be misconfigured", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
