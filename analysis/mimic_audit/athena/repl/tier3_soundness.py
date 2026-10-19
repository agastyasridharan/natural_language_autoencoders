import json, os, glob, argparse
ap = argparse.ArgumentParser()
ap.add_argument("--tier3", required=True)
ap.add_argument("--base", required=True)
ap.add_argument("--label", default="")
args = ap.parse_args()

def load(d):
    out = {}
    for p in glob.glob(d + "/claims/*.json"):
        if os.path.basename(p) == "status.json": continue
        out[os.path.basename(p)] = json.load(open(p))
    return out
t3, bs = load(args.tier3), load(args.base)
GE = ("grounded", "entangled")

def q95(r): return r.get("q95_dstar")
def sign_stats(rows):
    vals = [q95(r) for r in rows if q95(r) is not None]
    neg = sum(1 for v in vals if v < 0)
    return len(vals), neg

# baseline: base grounded/entangled null sign
base_ge = [bs[n] for n in bs if bs[n].get("verdict") in GE]
nb, negb = sign_stats(base_ge)
print("[%s] BASE grounded+entangled: N=%d  Q95(D*)<0: %d (%.0f%%)"
      % (args.label, nb, negb, 100*negb/max(1,nb)))

# tier3 recovered grounded/entangled
rec = [(n, bs[n], t3[n]) for n in t3 if n in bs
       and bs[n].get("verdict") == "unresolved" and t3[n].get("verdict") in GE]
nr, negr = sign_stats([t for _,_,t in rec])
print("[%s] RECOVERED grounded+entangled: N=%d  Q95(D*)<0: %d (%.0f%%)"
      % (args.label, nr, negr, 100*negr/max(1,nr)))
# base power breakdown of recoveries
bp_true = sum(1 for _,b,_ in rec if (b.get("power") or {}).get("passed") is True)
bp_false = sum(1 for _,b,_ in rec if (b.get("power") or {}).get("passed") is False)
print("[%s]   of recovered G/E: base power=True %d, base power=False %d"
      % (args.label, bp_true, bp_false))
# flag the ones that are BOTH negative-null AND/OR base power=False
for n, b, t in sorted(rec):
    q = q95(t); bp = (b.get("power") or {}).get("passed")
    flags = []
    if q is not None and q < 0: flags.append("NEG-NULL")
    if bp is False: flags.append("BASE-POWER-FALSE")
    tag = ("  <-- " + "+".join(flags)) if flags else ""
    print("[%s]     %-12s %s  M_BB=%.3f Q95=%.3f%s"
          % (args.label, n, t.get("verdict"), t.get("m_bb") or 0.0, q if q is not None else 0.0, tag))
