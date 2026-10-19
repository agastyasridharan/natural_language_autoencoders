"""Pre-fetch Gemma-27B + Llama-70B (base / AV / AR) into the HF cache so their
FVE runs are compute-only. CPU + proxy only; run detached, logs per repo."""
from huggingface_hub import snapshot_download

REPOS = [
    "google/gemma-3-27b-it",
    "kitft/nla-gemma3-27b-L41-av",
    "kitft/nla-gemma3-27b-L41-ar",
    "meta-llama/Llama-3.3-70B-Instruct",
    "kitft/Llama-3.3-70B-NLA-L53-av",
    "kitft/Llama-3.3-70B-NLA-L53-ar",
]

for r in REPOS:
    print(f"=== fetching {r}", flush=True)
    try:
        p = snapshot_download(r)
        print(f"done {r} -> {p}", flush=True)
    except Exception as e:
        print(f"FAIL {r}: {e}", flush=True)
print("ALL DONE", flush=True)
