#!/usr/bin/env python3
"""Fast tokenizer preflight — catch transformers/tokenizers version drift in seconds.

Downloads ONLY the AV tokenizer + nla_meta.yaml (no weights, no GPU) and runs the
exact vendored `load_nla_config` assertions. This is the same check that runs at
the start of /load_family, so if this passes, the real load will get past the
tokenizer gate. Run it BEFORE the multi-minute SGLang launch.

The failure it guards against:
  "injection token appears 0× in canonical prompt"
means the installed transformers/tokenizers tokenizes the injection char (e.g. ㈎)
into multiple tokens inside the prompt, so the injection anchor is gone. Fix by
pinning to the tested stack:  pip install "sglang[all]==0.5.6" "transformers==4.57.1"

    python scripts/preflight_tokenizer.py            # qwen7b
    python scripts/preflight_tokenizer.py --family gemma12b   # needs HF_TOKEN
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

# Tokenizer/sidecar/config only — never the multi-GB safetensors.
_ALLOW = ["*.json", "*.txt", "*.model", "*.yaml", "tokenizer*"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--family", default="qwen7b")
    ap.add_argument("--registry", default=str(_ROOT / "app" / "families.yaml"))
    args = ap.parse_args()

    import transformers
    import tokenizers
    print(f"transformers {transformers.__version__} | tokenizers {tokenizers.__version__}")

    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    from app.nla_core import load_registry
    from vendor.nla_inference import load_nla_config

    reg = load_registry(args.registry)
    if args.family not in reg:
        print(f"unknown family {args.family!r}; have {sorted(reg)}")
        return 2
    av_repo = reg[args.family]["av_repo"]
    print(f"[preflight] {args.family}: fetching tokenizer + sidecar from {av_repo} …")

    local = snapshot_download(av_repo, allow_patterns=_ALLOW, token=os.environ.get("HF_TOKEN"))
    tok = AutoTokenizer.from_pretrained(local, trust_remote_code=True)

    try:
        cfg = load_nla_config(local, tok)          # the exact /load_family gate
    except AssertionError as exc:
        print("\n[preflight] FAIL — tokenizer gate would reject this load:\n")
        print("   " + str(exc).replace("\n", "\n   "))
        print("\n[preflight] Almost always version drift. Pin the tested stack and restart:")
        print('   pip install "sglang[all]==0.5.6" "transformers==4.57.1"')
        return 1

    print(f"[preflight] OK — injection char {cfg.injection_char!r} (id "
          f"{cfg.injection_token_id}) tokenizes to a single in-context token; "
          f"neighbors verified. d_model={cfg.d_model}, injection_scale={cfg.injection_scale}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
