"""Broaden the Neuronpedia gemma27b cross-check: verbalize a SPREAD of positions
per doc (not just the read token), collect the cosine distribution, and convert it
to FVE using the cluster-measured gemma27b variance D=0.0744. Confirms (a) NP gemma
cosine is broadly saturated ~0.99 (matches ours), and (b) NP-implied FVE is medium
~0.69 (the honest number). Prints full errors for any failures.
"""
from __future__ import annotations
import json, re, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "nla_np"))
from np_core.client import NeuronpediaNLA, UpstreamUnavailable, RateLimited

D_GEMMA27B = 0.0744   # variance (1-||hbar||^2) from fve.py on the cluster


def parse_prefix(prefix: str) -> list[dict]:
    parts = re.split(r'(?:^|\n)(user|assistant|system): \n\n', prefix)
    msgs, i = [], 1
    while i < len(parts):
        msgs.append({"role": parts[i], "content": parts[i + 1] if i + 1 < len(parts) else ""})
        i += 2
    for m in msgs[:-1]:
        if m["content"].endswith("\n"):
            m["content"] = m["content"][:-1]
    return msgs


def pick_spread(n_tokens: int, k: int = 12) -> list[int]:
    lo = max(50, int(0.45 * n_tokens)); hi = n_tokens - 1
    if lo >= hi:
        lo = max(1, hi - k)
    step = max(1, (hi - lo) // (k - 1))
    return sorted({min(hi, lo + i * step) for i in range(k)} | {hi})


def main():
    tasks = {t["idx"]: t for t in json.loads(
        (HERE / "corpus" / "calib_sample_v2_3.json").read_text())["tasks"]}
    # diverse set, raw-heavy (raw genres = the "near-zero payload" ones); not too long
    by_len = sorted(tasks.values(), key=lambda t: len(t["prefix"]))
    pick = [by_len[int(i * (len(by_len) - 1) / 23)]["idx"] for i in range(24)]
    pick = sorted(set(pick))

    gem = NeuronpediaNLA("gemma", temperature=0.4, timeout=180.0)
    all_cos, errors, per_doc = [], [], []
    for idx in pick:
        t = tasks[idx]
        # chat docs -> parse to messages (correct gemma template); raw docs ->
        # send the plain text exactly as the corpus read it.
        msgs = parse_prefix(t["prefix"]) if t.get("kind") == "chat" else None
        transcript = msgs if msgs else t["prefix"]
        got = None
        for attempt in range(4):
            try:
                toks = gem.tokenize(transcript)
                pos = pick_spread(len(toks))
                res = gem.verbalize(transcript, positions=pos)
                cs = [v.cosine_similarity for v in res.results
                      if v.cosine_similarity == v.cosine_similarity]
                got = cs
                break
            except (UpstreamUnavailable, RateLimited) as e:
                time.sleep(3 + 5 * attempt); errors.append((idx, "5xx/429", str(e)[:80]))
            except Exception as e:  # noqa
                errors.append((idx, type(e).__name__, str(e)[:120])); break
        if got:
            all_cos += got
            per_doc.append((idx, t["theme"], len(got),
                            round(sum(got) / len(got), 3), round(min(got), 3), round(max(got), 3)))
            print(f"  idx {idx:>3} {t['theme'][:14]:<14} n={len(got):>2} "
                  f"cos mean={sum(got)/len(got):.3f} min={min(got):.3f} max={max(got):.3f}")
        else:
            print(f"  idx {idx:>3} {t['theme'][:14]:<14} FAILED")

    print("\n=== NP gemma27b cosine distribution (all positions) ===")
    if all_cos:
        n = len(all_cos); mean = sum(all_cos) / n
        sc = sorted(all_cos)
        p = lambda q: sc[min(n - 1, int(q * n))]
        print(f"n={n}  mean={mean:.3f}  min={sc[0]:.3f}  p10={p(.1):.3f}  "
              f"median={p(.5):.3f}  p90={p(.9):.3f}  max={sc[-1]:.3f}")
        fve = 1 - 2 * (1 - mean) / D_GEMMA27B
        print(f"NP-implied FVE (using cluster D={D_GEMMA27B}) = {fve:.3f}  "
              f"[our cluster FVE_real=0.689]")
        frac_high = sum(1 for c in all_cos if c > 0.95) / n
        print(f"fraction of NP gemma cosines > 0.95 = {frac_high:.0%}  (saturation)")
    if errors:
        print(f"\n{len(errors)} errors:")
        for idx, kind, msg in errors[:8]:
            print(f"  idx {idx}: {kind}: {msg}")
    json.dump(dict(cos=all_cos, per_doc=per_doc, errors=errors),
              open(HERE / "np_xcheck2_result.json", "w"), indent=2)


if __name__ == "__main__":
    main()
