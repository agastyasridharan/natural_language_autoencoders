import json, os, glob, argparse
ap = argparse.ArgumentParser()
ap.add_argument("--tier3", required=True); ap.add_argument("--base", required=True)
ap.add_argument("--label", default="")
args = ap.parse_args()
def load(d):
    o = {}
    for p in glob.glob(d + "/claims/*.json"):
        if os.path.basename(p) == "status.json": continue
        o[os.path.basename(p)] = json.load(open(p))
    return o
t3, bs = load(args.tier3), load(args.base)
RES = ("grounded","entangled","seeded","confabulated")
rec = [(n, bs[n], t3[n]) for n in t3 if n in bs
       and bs[n].get("verdict")=="unresolved" and t3[n].get("verdict") in RES]
viol = [(n,b) for n,b,_ in rec if (b.get("validity") or {}).get("passed") is not False]
print("[%s] recovered=%d  charter-compliant (base validity=False)=%d  VIOLATIONS=%d"
      % (args.label, len(rec), len(rec)-len(viol), len(viol)))
for n,b in viol:
    print("   VIOLATION %s base validity.passed=%s" % (n,(b.get("validity") or {}).get("passed")))
# 153_2 detail if present
for n,b,t in rec:
    if n=="153_2.json":
        bv=b.get("validity") or {}; bp=b.get("power") or {}
        print("[%s] 153_2 BASE: n_surv=%s validity.n_clean=%s validity.passed=%s power.passed=%s power.reason=%s"
              % (args.label, b.get("n_surv"), bv.get("n_clean"), bv.get("passed"),
                 bp.get("passed"), (bp.get("reason") or "")[:60]))
