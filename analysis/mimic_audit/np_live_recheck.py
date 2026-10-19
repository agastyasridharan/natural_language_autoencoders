"""Independent LIVE re-check of the activation audit against hosted Neuronpedia
NLAs (llama L53, gemma27b L41) — run fresh, not from stored JSON.

Confirms, on the live hosted models, the three load-bearing claims of the audit's
metric pivot:
  (1) gemma cosine is broadly SATURATED (~0.99) while llama is MODERATE (~0.7-0.9)
      on the same read locations  -> saturation is real hosted-model geometry.
  (2) mse == 2*(1-cos) EXACTLY (the identity the FVE recompute relies on).
  (3) gemma l2_norm >> llama l2_norm (gold magnitude / anisotropy), and both are
      in the ballpark of our cluster gold norms (llama ~90, gemma27b ~6e4).
Also prints NP cosine next to OUR stored cos_real per idx (paired reproducibility).

Rotates across the NEURONPEDIA_API_KEY* keys in .env on 429/5xx.
"""
from __future__ import annotations
import json, re, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "nla_np"))
from np_core.client import NeuronpediaNLA, UpstreamUnavailable, RateLimited  # noqa: E402

CORPUS = HERE / "corpus"


def load_keys() -> list[str]:
    env = HERE.parent.parent / ".env"
    keys = []
    for line in env.read_text().splitlines():
        line = line.strip()
        if line.startswith("NEURONPEDIA_API_KEY") and "=" in line:
            keys.append(line.split("=", 1)[1].strip().strip("'\""))
    return keys


KEYS = load_keys()


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


def our_cos(key: str) -> dict[int, float]:
    p = {"gemma": "athena/out_gemma27b/results.json", "llama": "athena/out/results.json"}[key]
    rows = json.loads((HERE / p).read_text())["rows"]
    return {r["idx"]: r["cos_real"] for r in rows}


def verbalize_read_token(alias: str, msgs: list[dict], ki: int):
    """Try across keys; return (dict|None, ki). Verbalize the LAST token (read token)."""
    for _ in range(len(KEYS)):
        nla = NeuronpediaNLA(alias, temperature=0.4, api_key=KEYS[ki % len(KEYS)], timeout=180.0)
        try:
            toks = nla.tokenize(msgs)
            if not toks:
                return {"error": "empty tokenize"}, ki
            pos = toks[-1].position
            res = nla.verbalize(msgs, positions=[pos])
            v = res.by_position().get(pos) or next(iter(res.by_position().values()))
            return dict(cos=v.cosine_similarity, mse=v.mse, l2=v.l2_norm,
                        token=v.token, descr=v.description[:70]), ki
        except (RateLimited, UpstreamUnavailable):
            ki += 1
            time.sleep(2)
        except Exception as e:  # noqa
            return {"error": str(e)[:120]}, ki
        finally:
            nla.close()
    return {"error": "all keys exhausted/5xx"}, ki


def main():
    tasks = {t["idx"]: t for t in json.loads((CORPUS / "calib_sample_v2_3.json").read_text())["tasks"]}
    chat = [t for t in tasks.values() if t.get("kind") == "chat"]
    chat.sort(key=lambda t: t.get("cosine", 0))
    n = len(chat)
    pick = sorted({chat[int(i * (n - 1) / 9)]["idx"] for i in range(10)})
    print(f"chat docs available: {n}; picked {len(pick)} spanning stored cosine: {pick}\n")

    og, ol = our_cos("gemma"), our_cos("llama")
    ki = 0
    print(f"{'idx':>4} {'theme':<14} {'stored':>6} | {'NPgem_cos':>9} {'NPgem_mse':>9} {'NPgem_l2':>9} {'ourGem':>7} | "
          f"{'NPlla_cos':>9} {'NPlla_l2':>9} {'ourLla':>7}")
    print("-" * 110)
    G, L = [], []
    for idx in pick:
        msgs = parse_prefix(tasks[idx]["prefix"])
        g, ki = verbalize_read_token("gemma", msgs, ki)
        l, ki = verbalize_read_token("llama", msgs, ki)
        if "cos" in g:
            G.append(g)
        if "cos" in l:
            L.append(l)
        gc = f"{g.get('cos'):.3f}" if "cos" in g else g.get("error", "?")[:9]
        gm = f"{g.get('mse'):.3f}" if "cos" in g else ""
        gl = f"{g.get('l2'):.0f}" if "cos" in g else ""
        lc = f"{l.get('cos'):.3f}" if "cos" in l else l.get("error", "?")[:9]
        ll = f"{l.get('l2'):.0f}" if "cos" in l else ""
        print(f"{idx:>4} {tasks[idx]['theme'][:14]:<14} {tasks[idx].get('cosine',0):>6.3f} | "
              f"{gc:>9} {gm:>9} {gl:>9} {og.get(idx,0):>7.3f} | {lc:>9} {ll:>9} {ol.get(idx,0):>7.3f}")

    def mse_resid(rows):
        return max(abs(r["mse"] - 2 * (1 - r["cos"])) for r in rows) if rows else float("nan")

    print("\n=== SUMMARY (live) ===")
    if G:
        gc = [r["cos"] for r in G]; gl = [r["l2"] for r in G]
        print(f"NP gemma27b: n={len(G)} cos mean={sum(gc)/len(gc):.3f} min={min(gc):.3f} max={max(gc):.3f} "
              f"| l2 mean={sum(gl)/len(gl):.0f} | max|mse-2(1-cos)|={mse_resid(G):.2e}")
    if L:
        lc = [r["cos"] for r in L]; ll = [r["l2"] for r in L]
        print(f"NP llama70b: n={len(L)} cos mean={sum(lc)/len(lc):.3f} min={min(lc):.3f} max={max(lc):.3f} "
              f"| l2 mean={sum(ll)/len(ll):.0f} | max|mse-2(1-cos)|={mse_resid(L):.2e}")
    print("\nExpected (cluster gold): llama norm ~90, gemma27b norm ~5.6e4-6.2e4; "
          "gemma cos saturated ~0.99, llama moderate.")
    (HERE / "np_live_recheck_result.json").write_text(json.dumps(dict(gemma=G, llama=L), indent=2))
    print(f"wrote {HERE/'np_live_recheck_result.json'}")


if __name__ == "__main__":
    main()
