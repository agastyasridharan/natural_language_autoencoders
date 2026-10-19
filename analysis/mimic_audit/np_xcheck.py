"""Neuronpedia cross-check for the gemma27b activation audit.

The audit reports gemma27b cos_real ~0.99 (mse ~0.02). Neuronpedia's web UI shows
MEDIUM mse for gemma. Since NP's stored mse == 2(1-cos) EXACTLY (verified on the
llama corpus), a medium NP mse implies a MODERATE NP cosine — contradicting our
0.99. This script queries the hosted gemma27b (and llama as a client sanity check)
on the SAME documents / read tokens and prints NP cosine+mse next to ours.

For each selected idx: parse the stored transcript prefix back into chat messages,
send to NP, verbalize the final token (the read token), record cosine/mse.
"""
from __future__ import annotations
import json, re, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "nla_np"))
from np_core.client import NeuronpediaNLA, UpstreamUnavailable, RateLimited

CORPUS = HERE / "corpus"


def parse_prefix(prefix: str) -> list[dict]:
    """Invert reconstruct_prefix: 'user: \\n\\n{c}\\nassistant: \\n\\n{c}...' -> messages.

    Completed turns end with a trailing '\\n' (the llama <|eot_id|> replacement);
    the final (open) turn ends at the read token. Split on role headers.
    """
    parts = re.split(r'(?:^|\n)(user|assistant|system): \n\n', prefix)
    msgs = []
    i = 1
    while i < len(parts):
        role = parts[i]
        content = parts[i + 1] if i + 1 < len(parts) else ""
        msgs.append({"role": role, "content": content})
        i += 2
    # strip the single trailing '\n' that the eot replacement added to every
    # COMPLETED turn (format_chat re-adds the turn terminator); leave the final
    # (open) turn's content untouched so its last token is the read token.
    for m in msgs[:-1]:
        if m["content"].endswith("\n"):
            m["content"] = m["content"][:-1]
    return msgs


def load_my_gemma_cos_real() -> dict[int, float]:
    """cos_real per idx from the built viewer data file."""
    js = (CORPUS / "act_audit_gemma27b.js").read_text(encoding="utf-8")
    start = js.index(".push(") + len(".push(")
    obj = json.loads(js[start: js.rindex(")")])
    return {r["idx"]: r["cos_real"] for r in obj["rows"]}, \
           {r["idx"]: r.get("cos_control") for r in obj["rows"]}


def verbalize_last(nla: NeuronpediaNLA, messages: list[dict], retries: int = 4):
    """Tokenize, verbalize the last 1-2 positions; return (cosine, mse, token, descr)."""
    last_err = None
    for attempt in range(retries):
        try:
            toks = nla.tokenize(messages)
            if not toks:
                return None
            pos = [toks[-1].position]
            if len(toks) >= 2:
                pos.append(toks[-2].position)
            res = nla.verbalize(messages, positions=pos)
            bp = res.by_position()
            v = bp.get(toks[-1].position) or next(iter(bp.values()))
            return dict(cosine=v.cosine_similarity, mse=v.mse,
                        token=v.token or toks[-1].token, descr=v.description[:90])
        except (UpstreamUnavailable, RateLimited) as e:
            last_err = e
            time.sleep(3 + 4 * attempt)
        except Exception as e:  # noqa
            return dict(error=str(e)[:140])
    return dict(error=f"unavailable after {retries} tries: {str(last_err)[:100]}")


def main():
    tasks = {t["idx"]: t for t in json.loads(
        (CORPUS / "calib_sample_v2_3.json").read_text())["tasks"]}
    my_real, my_ctl = load_my_gemma_cos_real()

    # select idx spanning the stored-cosine (difficulty) range
    ordered = sorted(tasks.values(), key=lambda t: t.get("cosine", 0))
    n = len(ordered)
    pick = [ordered[int(i * (n - 1) / 11)]["idx"] for i in range(12)]
    pick = sorted(set(pick))
    print(f"selected {len(pick)} idx spanning stored-cosine range: {pick}\n")

    gem = NeuronpediaNLA("gemma", temperature=0.4)
    lla = NeuronpediaNLA("llama", temperature=0.4)

    print(f"{'idx':>4} {'theme':<14} {'stored':>7} | {'NP_llama':>8} (client-sanity) | "
          f"{'NP_gemma_cos':>12} {'NP_gemma_mse':>12} | {'MY_gemma_cos':>12} {'MY_ctl':>7}")
    print("-" * 120)
    rows = []
    for idx in pick:
        t = tasks[idx]
        msgs = parse_prefix(t["prefix"])
        npl = verbalize_last(lla, msgs)
        npg = verbalize_last(gem, msgs)
        row = dict(idx=idx, theme=t["theme"], stored=t.get("cosine"),
                   np_llama=npl, np_gemma=npg,
                   my_gemma_cos=my_real.get(idx), my_gemma_ctl=my_ctl.get(idx))
        rows.append(row)
        npl_c = f"{npl.get('cosine'):.3f}" if npl and 'cosine' in npl else (npl or {}).get('error', 'n/a')[:18]
        npg_c = f"{npg.get('cosine'):.3f}" if npg and 'cosine' in npg else (npg or {}).get('error', 'n/a')[:18]
        npg_m = f"{npg.get('mse'):.3f}" if npg and 'mse' in npg else ""
        print(f"{idx:>4} {t['theme'][:14]:<14} {t.get('cosine',0):>7.3f} | {npl_c:>8} {'':<15} | "
              f"{npg_c:>12} {npg_m:>12} | {my_real.get(idx,0):>12.3f} {my_ctl.get(idx,0) or 0:>7.3f}")

    # summary over positions where gemma responded
    gc = [r["np_gemma"]["cosine"] for r in rows if r["np_gemma"] and "cosine" in r["np_gemma"]]
    gm = [r["np_gemma"]["mse"] for r in rows if r["np_gemma"] and "mse" in r["np_gemma"]]
    lc = [r["np_llama"]["cosine"] for r in rows if r["np_llama"] and "cosine" in r["np_llama"]]
    mc = [r["my_gemma_cos"] for r in rows if r["my_gemma_cos"] is not None]
    print("\n=== SUMMARY ===")
    if gc:
        print(f"NP gemma cosine: n={len(gc)} mean={sum(gc)/len(gc):.3f} "
              f"min={min(gc):.3f} max={max(gc):.3f}")
        print(f"NP gemma mse:    n={len(gm)} mean={sum(gm)/len(gm):.3f} "
              f"min={min(gm):.3f} max={max(gm):.3f}")
    else:
        print("NP gemma: NO successful responses (server likely 502). Rely on cluster check.")
    if lc:
        print(f"NP llama cosine: n={len(lc)} mean={sum(lc)/len(lc):.3f} (client sanity vs stored)")
    print(f"MY gemma cos_real (paired): mean={sum(mc)/len(mc):.3f}")
    (HERE / "np_xcheck_result.json").write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {HERE/'np_xcheck_result.json'}")


if __name__ == "__main__":
    main()
