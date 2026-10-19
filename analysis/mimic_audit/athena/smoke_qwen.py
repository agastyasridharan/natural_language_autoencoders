"""7B GPU smoke test for the mimic-vs-real activation-reconstruction harness.

Runs on Athena via `gpurun smoke_qwen.py` (1 GPU). Validates the MECHANICS of the
whole experiment end-to-end on the small, ungated qwen7b NLA:
  base model  --extract hidden_states[K+1] at a token-->  gold vector
  AR critic   --score(text, gold)-->  (mse, cos)   [mse == 2(1-cos)]

For each of the staged items it scores three texts against the same gold:
  cos_real    = score(real AV read,  gold)
  cos_mimic   = score(v2_3 mimic,    gold)
  cos_control = score(another item's real read, gold)   # chance floor

This is a WIRING test, not the result: qwen's tokenizer != the Llama corpus, so
the position/token do not align and the numbers are not corpus-matched. Success =
models load, extraction yields [T, d_model=3584] raw vectors, score() runs and
obeys mse==2(1-cos), and on-topic text tends to beat the mismatched control.

Staged on /data first (gpurun uploads only this .py):
  /data/agastyas/nla_mimic/code/nla_inference.py   (NLACritic)
  /data/agastyas/nla_mimic/data/smoke_input.json
gpurun sets CUDA_VISIBLE_DEVICES, the HF proxy, and HF_HOME — do NOT set them here.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/data/agastyas/nla_mimic/code")  # nla_inference.NLACritic
from nla_inference import NLACritic  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402

INPUT = "/data/agastyas/nla_mimic/data/smoke_input.json"
BASE = "Qwen/Qwen2.5-7B-Instruct"
AR_REPO = "kitft/nla-qwen2.5-7b-L20-ar"
LAYER_K = 20                 # activation = hidden_states[K+1]
D_MODEL = 3584
DEV = "cuda"
DT = torch.bfloat16


@torch.inference_mode()
def extract_gold(base, tok, payload, position: int) -> tuple[np.ndarray, int, str]:
    # payload is either raw text (str) or a chat transcript (list of {role,content}).
    # SMOKE-only tokenization: chat -> apply_chat_template, raw -> add_special_tokens.
    # (The real llama70b run must match the corpus tokenizer exactly; verified there
    # via the stored-token alignment gate, not here.)
    if isinstance(payload, str):
        enc = tok(payload, return_tensors="pt", add_special_tokens=True,
                  truncation=True, max_length=2048)
    else:
        # render chat -> string (template already carries its own special tokens),
        # then tokenize with add_special_tokens=False so none are doubled.
        rendered = tok.apply_chat_template(payload, tokenize=False,
                                           add_generation_prompt=False)
        enc = tok(rendered, return_tensors="pt", add_special_tokens=False,
                  truncation=True, max_length=2048)
    enc = {k: v.to(DEV) for k, v in enc.items()}
    out = base(**enc, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states[LAYER_K + 1][0]          # [T, d], raw (no norm)
    pos = min(position, hs.shape[0] - 1)            # qwen seq may be shorter -> clamp
    ids = enc["input_ids"][0].tolist()
    qtok = tok.convert_ids_to_tokens([ids[pos]])[0]
    return hs[pos].float().cpu().numpy().astype(np.float32), pos, qtok


def main() -> int:
    items = json.loads(Path(INPUT).read_text(encoding="utf-8"))
    print(f"[smoke] {len(items)} items; torch {torch.__version__} "
          f"cuda={torch.cuda.is_available()} dev={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}",
          flush=True)

    print(f"[smoke] loading base {BASE} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(BASE)
    base = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=DT, low_cpu_mem_usage=True).to(DEV).eval()

    print(f"[smoke] downloading + loading AR {AR_REPO} ...", flush=True)
    ar_dir = snapshot_download(AR_REPO)
    critic = NLACritic(ar_dir, device=DEV, dtype=DT)
    print(f"[smoke] AR mse_scale={critic.mse_scale:.3f}", flush=True)

    rows, ok = [], True
    for i, it in enumerate(items):
        gold, pos, qtok = extract_gold(base, tok, it["payload"], it["position"])
        # control = a DIFFERENT item's real read against THIS gold (floor)
        other = items[(i + 1) % len(items)]["real_text"]
        m_real, c_real = critic.score(it["real_text"], gold)
        m_mim, c_mim = critic.score(it["mimic_text"], gold)
        m_ctl, c_ctl = critic.score(other, gold)

        # mechanics asserts
        if gold.shape[0] != D_MODEL:
            print(f"  !! d_model {gold.shape[0]} != {D_MODEL}", flush=True); ok = False
        for nm, (mm, cc) in [("real", (m_real, c_real)), ("mimic", (m_mim, c_mim))]:
            if not (-1.01 <= cc <= 1.01):
                print(f"  !! {nm} cos out of range: {cc}", flush=True); ok = False
            if abs(mm - 2 * (1 - cc)) > 1e-3:
                print(f"  !! {nm} mse {mm:.4f} != 2(1-cos) {2*(1-cc):.4f}", flush=True); ok = False

        rows.append({"idx": it["idx"], "theme": it["theme"], "position": it["position"],
                     "qwen_pos": pos, "qwen_token": qtok, "llama_token": it["llama_token"],
                     "cosine_stored_llama": it["cosine_stored"],
                     "cos_real": round(c_real, 4), "cos_mimic": round(c_mim, 4),
                     "cos_control": round(c_ctl, 4),
                     "delta_real_minus_mimic": round(c_real - c_mim, 4)})
        print(f"  [{i+1}/{len(items)}] {it['theme']:15s} pos{it['position']:>3} "
              f"cos_real={c_real:+.3f} cos_mimic={c_mim:+.3f} cos_ctl={c_ctl:+.3f} "
              f"d={gold.shape[0]}", flush=True)

    real_gt_ctl = sum(1 for r in rows if r["cos_real"] > r["cos_control"])
    out = Path("smoke_results.json")
    out.write_text(json.dumps({"base": BASE, "ar": AR_REPO, "rows": rows,
                               "mechanics_ok": ok,
                               "real_beats_control": f"{real_gt_ctl}/{len(rows)}"},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n[smoke] mechanics_ok={ok}  real>control in {real_gt_ctl}/{len(rows)} items", flush=True)
    print(f"[smoke] wrote {out.resolve()}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
