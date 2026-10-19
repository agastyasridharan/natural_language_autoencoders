"""Bundle the 128 held-out pairs for the Axis-1 content audit.

Each item = {idx, theme, kind, cosine, prefix (full), real_av, mimic} where the
mimic is the locked-prompt (v1.1) output read back from corpus/calib_out_v1_1/.
No API calls.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SAMPLE = HERE / "corpus" / "calib_sample_v1_1.json"
MIMICDIR = HERE / "corpus" / "calib_out_v1_1"
OUT = HERE / "corpus" / "axis1_sample.json"


def main():
    tasks = json.loads(SAMPLE.read_text(encoding="utf-8"))["tasks"]
    items = []
    missing = 0
    for t in tasks:
        mp = MIMICDIR / f"mimic_{t['idx']}.txt"
        if not mp.exists():
            missing += 1
            continue
        items.append({
            "idx": t["idx"], "theme": t["theme"], "kind": t["kind"],
            "cosine": t["cosine"], "position": t["position"],
            "prefix": t["prefix"], "real_av": t["real_av"].strip(),
            "mimic": mp.read_text(encoding="utf-8").strip(),
        })
    OUT.write_text(json.dumps({"items": items}, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT.name}: {len(items)} pairs ({missing} missing mimic), "
          f"{OUT.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
