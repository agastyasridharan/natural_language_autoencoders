"""Build the held-out calibration sample for the single-prompt run.

Samples 4 eval reads per genre spanning that genre's cosine range (128 tasks),
each carrying the FULL prefix (fairness fix) and the real AV read. Also picks a
handful of genuine reference reads from TRAIN so the cross-model discriminator
knows what real AV output looks like before it judges. Bundles everything plus
the prompt template into corpus/calib_sample.json, which is passed to the
Workflow as `args`.

No API calls; pure data prep.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVAL = HERE / "corpus" / "eval.jsonl"
TRAIN = HERE / "corpus" / "train.jsonl"

# Optional tag arg selects the prompt + output names. Default reproduces v1
# (unsuffixed) names; pass e.g. "v1_1" to use prompt_v1_1.txt -> calib_sample_v1_1.json.
TAG = sys.argv[1] if len(sys.argv) > 1 else None
PROMPT = HERE / (f"prompt_{TAG}.txt" if TAG else "prompt_v1.txt")
OUT = HERE / "corpus" / (f"calib_sample_{TAG}.json" if TAG else "calib_sample.json")

PER_GENRE = 4

# Genuine reference reads for the discriminator: one per genre, varied formats.
# Drawn from TRAIN, avoiding the genres already shown inside the mimic prompt so
# the discriminator's reference set is not identical to the mimic's few-shot.
REF_GENRES = ["history", "science", "lyrics", "debate", "agentic_tool", "legal"]


def spaced(n, k):
    if n <= k:
        return list(range(n))
    return [round(i * (n - 1) / (k - 1)) for i in range(k)]


def main():
    ev = [json.loads(l) for l in EVAL.open(encoding="utf-8")]
    tr = [json.loads(l) for l in TRAIN.open(encoding="utf-8")]
    tr_by_genre = defaultdict(list)
    for r in tr:
        tr_by_genre[r["theme"]].append(r)

    by_genre = defaultdict(list)
    for r in ev:
        by_genre[r["theme"]].append(r)

    tasks = []
    idx = 0
    for g in sorted(by_genre):
        rows = sorted(by_genre[g], key=lambda r: r["cosine"])
        for i in spaced(len(rows), PER_GENRE):
            r = rows[i]
            tasks.append({
                "idx": idx,
                "doc_id": r["doc_id"], "theme": r["theme"], "kind": r["kind"],
                "position": r["position"], "n_tokens": r["n_tokens"],
                "cosine": r["cosine"], "mse": r["mse"],
                "prefix": r["prefix"],
                "real_av": r["av_description"],
            })
            idx += 1

    # reference reads: highest-cosine (cleanest) genuine read per chosen genre
    refs = []
    for g in REF_GENRES:
        cand = tr_by_genre.get(g)
        if not cand:
            continue
        best = max(cand, key=lambda r: r["cosine"])
        refs.append({"theme": best["theme"], "read": best["av_description"].strip()})

    template = PROMPT.read_text(encoding="utf-8")
    payload = {
        "prompt_template": template,
        "reference_reads": refs,
        "tasks": tasks,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT.name}: {len(tasks)} tasks, {len(refs)} reference reads, "
          f"{OUT.stat().st_size:,} bytes")
    cos = [t["cosine"] for t in tasks]
    print(f"task cosine span: {min(cos):.3f} .. {max(cos):.3f}")


if __name__ == "__main__":
    main()
