import json, os, glob, sys, argparse
ap = argparse.ArgumentParser()
ap.add_argument("--tier3", required=True)
ap.add_argument("--base", required=True)
ap.add_argument("--transitions", nargs="*", default=["unresolved->grounded"],
                help="which base->tier3 transitions to spot-check")
args = ap.parse_args()

def load(d):
    out = {}
    for p in glob.glob(d + "/claims/*.json"):
        if os.path.basename(p) == "status.json": continue
        out[os.path.basename(p)] = json.load(open(p))
    return out

t3, bs = load(args.tier3), load(args.base)
want = set(tuple(t.split("->")) for t in args.transitions)
targets = [n for n in t3 if n in bs
           and (bs[n].get("verdict"), t3[n].get("verdict")) in want]
print("spot-check transitions %s: %d claims" % (args.transitions, len(targets)))
for n in sorted(targets):
    b, t = bs[n], t3[n]
    bv = b.get("validity") or {}; tv = t.get("validity") or {}
    bp = b.get("power") or {}
    print("\n=== %s  (%s -> %s) ===" % (n, b.get("verdict"), t.get("verdict")))
    print("  BASE: power=%s validity=%s  n_clean=%s dropped=%s"
          % (bp.get("passed"), bv.get("passed"), bv.get("n_clean"), bv.get("dropped_idx")))
    print("  TIER3 validity: n_clean=%s dropped=%s passed=%s"
          % (tv.get("n_clean"), tv.get("dropped_idx"), tv.get("passed")))
    mbb = t.get("m_bb"); q = t.get("q95_dstar"); kap = t.get("kappa_0")
    ns = t.get("n_surv"); mib = t.get("max_m_ib")
    print("  TIER3: M_BB=%s Q95(D*)=%s kappa_0=%s n_surv=%s max_m_iB=%s"
          % (mbb, q, kap, ns, mib))
    if None not in (mbb, q, kap):
        bar = q + kap
        print("    GATE1: M_BB=%.3f > Q95+kappa=%.3f  -> %s  (margin x%.2f)"
              % (mbb, bar, mbb > bar, mbb / bar if bar else float("inf")))
    if None not in (mbb, mib) and mib not in (0, None):
        print("    GATE2: M_BB=%.3f > 1.5*max_m_iB=%.3f  -> %s"
              % (mbb, 1.5 * mib, mbb > 1.5 * mib))
