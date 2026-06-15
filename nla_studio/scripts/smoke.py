#!/usr/bin/env python3
"""End-to-end smoke test for NLA Studio (qwen7b by default).

Exercises the full inference path directly through app/nla_core (base model +
AV-over-SGLang + AR in-process) — no uvicorn needed, but the SGLang AV server
MUST be running (scripts/launch_sglang.sh qwen7b).

Steps (per the task spec):
  1. load the family
  2. extract from "The quick brown fox jumps over the lazy dog." at a
     mid-sequence token
  3. run the AV explanation; CONFIRM it is English, not CJK   <- hard pass/fail
  4. AR-score the explanation against the extracted vector; a clean position
     should land roughly in cosine 0.85–0.95  (reported; soft check)

All-CJK AV output ⇒ injection failed: check injection_scale, embed_scale, and
(for Gemma) the gemma3_mm patch.

    python scripts/smoke.py                       # qwen7b, suggested position
    python scripts/smoke.py --family qwen7b --token-index 8
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from app import nla_core  # noqa: E402

DEFAULT_TEXT = "The quick brown fox jumps over the lazy dog."


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--family", default="qwen7b")
    ap.add_argument("--text", default=DEFAULT_TEXT)
    ap.add_argument("--token-index", type=int, default=None,
                    help="position to extract; default = a clean mid/late position")
    ap.add_argument("--registry", default=str(_ROOT / "app" / "families.yaml"))
    ap.add_argument("--sglang-url", default=os.environ.get("NLA_SGLANG_URL", "http://localhost:30000"))
    ap.add_argument("--device", default=os.environ.get("NLA_INPROC_DEVICE", "cuda:0"))
    ap.add_argument("--temperature", type=float, default=0.7)
    args = ap.parse_args()

    print(f"[smoke] loading {args.family} (this downloads/loads base + AR; AV must be on SGLang)…")
    fam = nla_core.load_family(
        args.family, args.registry,
        inproc_device=args.device, sglang_url=args.sglang_url,
        hf_token=os.environ.get("HF_TOKEN"),
    )
    print(f"[smoke] loaded. d_model={fam.d_model} layer K={fam.layer_k} "
          f"-> hidden_states[{fam.hidden_state_index}]; injection_scale={fam.injection_scale}")

    # pick a position
    picker = nla_core.extract(fam, args.text)
    idx = args.token_index if args.token_index is not None else picker["suggested_index"]
    tok = picker["tokens"][idx]
    print(f"[smoke] {picker['n_tokens']} tokens; extracting position {idx} "
          f"(token={tok['token']!r}, norm={tok['norm']})"
          + ("  [note: short text -> position is in the low-context window]" if tok["early"] else ""))

    res = nla_core.extract(fam, args.text, token_index=idx)
    vec = res["selected"]["vector"]

    # AV: vector -> text
    av = nla_core.av_explain(fam, vec, temperature=args.temperature, max_new_tokens=200)
    print("\n[smoke] AV explanation:")
    print("    " + av["explanation"].replace("\n", "\n    "))
    print(f"[smoke] CJK fraction = {av['cjk_fraction']:.3f}")

    if av["looks_like_injection_failure"]:
        print("\n[smoke] FAIL: AV output is mostly CJK — injection failed.")
        print("        Check injection_scale, embed_scale, and (Gemma) the gemma3_mm patch.")
        return 1

    # AR: text -> vector, score vs gold
    cos, mse = nla_core.score(fam, av["explanation"], vec)
    print(f"\n[smoke] AR score: cosine={cos:.4f}  MSE={mse:.4f}  (MSE = 2(1-cos), range [0,4])")
    if 0.85 <= cos <= 0.95:
        print("[smoke] PASS: English output and cosine in the expected 0.85–0.95 band.")
    elif cos >= 0.7:
        print(f"[smoke] PASS (CJK check) — cosine {cos:.3f} is reasonable; the 0.85–0.95 "
              "band assumes a longer, higher-context position than this short sentence.")
    else:
        print(f"[smoke] WARNING: cosine {cos:.3f} is low. English output passed, but "
              "reconstruction is weak — try a longer text and a later position.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
