"""Emit corpus/refine_data.js (window.REFINE = {...}) for the viewer's Refinement tab.

Reads the refine_v2 workflow output (pass its .output path as argv[1]) and packages
the rubric, the prompt iterations + changelogs, the train/eval discriminator
accuracy, per-genre generalization, and NLA-vs-mimic example pairs (including the
per-claim content contrast).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "corpus" / "refine_data.js"


def main():
    out_path = Path(sys.argv[1])
    raw = json.loads(out_path.read_text())
    r = raw["result"] if isinstance(raw, dict) and "result" in raw else raw
    r = json.loads(r) if isinstance(r, str) else r

    iterations = [{
        "round": s["round"], "disc_acc": s["discriminator_acc"],
        "changelog": s["changelog"], "prompt_before": s["prompt_before"],
    } for s in r["snapshots"]]

    epg = {}
    for g, v in r.get("eval_per_genre", {}).items():
        epg[g] = {"n": v["n"], "correct": v["correct"],
                  "acc": round(v["correct"] / v["n"], 3) if v["n"] else None}

    round_examples = []
    for s in r["snapshots"]:
        for ex in s.get("examples", [])[:3]:
            round_examples.append({"round": s["round"], **ex})

    refine = {
        "meta": {
            "goal": ("Measure how much of an NLA explanation is recoverable from the text prior "
                     "alone (a style-matched text-only 'mimic') versus genuinely carried by the "
                     "injected activation. The mimic prompt is tuned until a style discriminator "
                     "cannot tell mimic from real AV; then per-claim content is compared on held-out eval."),
            "rounds": len(iterations),
            "n_eval": r.get("n_eval"),
            "train_acc_trajectory": r.get("train_acc_trajectory"),
            "eval_overall_acc": r.get("eval_overall_acc"),
            "v0_chars": len(r["v0"]), "final_chars": len(r["final_prompt"]),
        },
        "rubric": r["rubric"],
        "v0": r["v0"],
        "iterations": iterations,
        "final_prompt": r["final_prompt"],
        "eval_per_genre": epg,
        "contrast_examples": r.get("content_contrast", []),
        "round_examples": round_examples,
        "eval_examples": r.get("eval_examples", []),
    }
    js = "window.REFINE = " + json.dumps(refine, ensure_ascii=False) + ";\n"
    OUT.write_text(js, encoding="utf-8")
    print(f"wrote refine_data.js ({len(js):,} bytes) — rounds={len(iterations)} "
          f"contrast={len(refine['contrast_examples'])} eval_examples={len(refine['eval_examples'])} "
          f"train_traj={refine['meta']['train_acc_trajectory']} eval_acc={refine['meta']['eval_overall_acc']}")


if __name__ == "__main__":
    main()
