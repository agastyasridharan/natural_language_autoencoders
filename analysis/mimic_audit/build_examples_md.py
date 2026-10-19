"""Write the full (untruncated) v2.1 mimic vs real-AV pairs to a markdown file.

Usage: python build_examples_md.py v2_1   (default v2_1)
Reads corpus/calib_result_<tag>.json (+ calib_sample/eval for the read token) and
writes V2_1_EXAMPLES.md: all 128 held-out pairs, grouped by genre, sorted by
cosine, each with the conversation-context tail (read token marked), the real AV
read, and the mimic read, in full.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
C = HERE / "corpus"


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "v2_1"
    res = json.loads((C / f"calib_result_{tag}.json").read_text(encoding="utf-8"))
    rows = res["rows"]
    idx2doc = {t["idx"]: t["doc_id"] for t in json.loads((C / f"calib_sample_{tag}.json").read_text())["tasks"]}
    tok = {}
    for line in (C / "eval.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        tok[(r["doc_id"], r["position"])] = r["token"]

    out = HERE / f"{tag.upper()}_EXAMPLES.md"
    L = []
    L.append(f"# Mimic ({tag}) vs real AV — full held-out pairs")
    L.append("")
    L.append(f"All {len(rows)} held-out eval pairs. Each is the **same token** read two ways: "
             "the **real AV** (decoded from the activation by the hosted Llama-3.3-70B NLA) and "
             f"the **mimic {tag}** (text-only, from the locked prompt). "
             "Discriminator = Sonnet 4.6 judging style only (no prefix): *caught* = it identified "
             "the real one; *fooled* = it picked the mimic as genuine. "
             f"Overall discriminator on this version: **{res['overall_acc']*100:.1f}%** "
             "(50% = indistinguishable).")
    L.append("")
    L.append("In the context line, the token the AV actually read is wrapped in «guillemets».")
    L.append("")

    by_genre = {}
    for r in rows:
        by_genre.setdefault(r["theme"], []).append(r)

    for genre in sorted(by_genre):
        L.append(f"## {genre}")
        L.append("")
        for r in sorted(by_genre[genre], key=lambda x: x["cosine"]):
            doc = idx2doc.get(r["idx"])
            t = tok.get((doc, r["position"]), "")
            pre = r.get("prefix", "")
            base = pre[-320:].rstrip()
            ts = t.strip()
            if ts and any(ch.isalnum() for ch in ts) and base.endswith(ts):
                i = base.rfind(ts)
                shown = base[:i] + "«" + ts + "»"
            elif t:
                shown = base + f"  «read token: {t!r}»"
            else:
                shown = base
            disc = "caught the mimic" if r["correct"] == 1 else ("FOOLED — picked mimic as real" if r["correct"] == 0 else "n/a")
            L.append(f"### {genre} · cosine {r['cosine']:.2f} · discriminator: {disc}")
            L.append("")
            L.append("**Context (end of the text both read):**")
            L.append("")
            L.append("```")
            L.append("…" + shown)
            L.append("```")
            L.append("")
            L.append("**Real AV (with activation):**")
            L.append("")
            L.append("```")
            L.append(r["real_av"].strip())
            L.append("```")
            L.append("")
            L.append(f"**Mimic {tag} (text only):**")
            L.append("")
            L.append("```")
            L.append(r["mimic"].strip())
            L.append("```")
            if r["correct"] == 1 and r.get("tell"):
                L.append("")
                L.append(f"*Discriminator's tell:* {r['tell']}")
            L.append("")
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {out.name} ({out.stat().st_size:,} bytes, {len(rows)} pairs, {len(by_genre)} genres)")


if __name__ == "__main__":
    main()
