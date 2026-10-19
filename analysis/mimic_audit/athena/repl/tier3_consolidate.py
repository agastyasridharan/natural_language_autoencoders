import json, os, glob
B = "/data/agastyas/nla_mimic"
MODELS = [
    ("qwen7b",  "qwen7b-d4g24-full/qwen7b-cp6-fineweb-atoms",  "Qwen2.5-7B L20"),
    ("gemma12b","gemma12b-d4g24/gemma12b-cp7-fineweb-atoms",   "Gemma-3-12B L32"),
    ("gemma27b","gemma27b-d4g24/gemma27b-g4-fineweb-atoms",    "Gemma-3-27B L41"),
    ("llama70b","llama70b-d4g24/llama70b-l4-fineweb-atoms",    "Llama-3.3-70B L53"),
]
VERD = ("grounded","entangled","seeded","confabulated","unresolved")
GE = ("grounded","entangled"); RES = GE + ("seeded","confabulated")

def load(d):
    o = {}
    for p in glob.glob(d + "/claims/*.json"):
        if os.path.basename(p) == "status.json": continue
        o[os.path.basename(p)] = json.load(open(p))
    return o
def dist(rows):
    d = {v:0 for v in VERD}
    for r in rows: d[r.get("verdict")] = d.get(r.get("verdict"),0)+1
    return d

out = {"tier": 3, "band": "P95 (P90->P95 f3/f4 widening)", "models": {}}
for key, sub, disp in MODELS:
    t3 = load(f"{B}/cycle_runs_tier3/{sub}")
    bs = load(f"{B}/cycle_runs/{sub}")
    rep = json.load(open(f"{B}/cycle_runs_tier3/{sub}/cycle_report.json"))
    shared = sorted(set(t3)&set(bs))
    db, dt = dist([bs[n] for n in shared]), dist([t3[n] for n in shared])
    N = len(shared)
    mbb_drift = max((abs(bs[n]["m_bb"]-t3[n]["m_bb"]) for n in shared
                     if bs[n].get("m_bb") is not None and t3[n].get("m_bb") is not None), default=0.0)
    trans = {}
    for n in shared:
        o1,o2 = bs[n]["verdict"], t3[n]["verdict"]
        if o1!=o2: trans[f"{o1}->{o2}"] = trans.get(f"{o1}->{o2}",0)+1
    rec = [(n,bs[n],t3[n]) for n in shared if bs[n]["verdict"]=="unresolved" and t3[n]["verdict"] in RES]
    lost = [n for n in shared if bs[n]["verdict"] in RES and t3[n]["verdict"]=="unresolved"]
    recd = dist([t for _,_,t in rec])
    # soundness: G/E recoveries base power + null sign
    ge_rec = [(n,b,t) for n,b,t in rec if t["verdict"] in GE]
    ge_bp_true = sum(1 for _,b,_ in ge_rec if (b.get("power") or {}).get("passed") is True)
    ge_nodata  = sum(1 for _,b,_ in ge_rec if (b.get("n_surv") or 0)==0)
    base_ge = [bs[n] for n in bs if bs[n]["verdict"] in GE]
    def negfrac(rows):
        v=[r.get("q95_dstar") for r in rows if r.get("q95_dstar") is not None]
        return (sum(1 for x in v if x<0), len(v))
    oos = rep.get("kappa_out_of_sample") or {}
    out["models"][key] = dict(
        display=disp, N=N,
        dist_base={k:db[k] for k in VERD}, dist_tier3={k:dt[k] for k in VERD},
        unresolved_pct_base=round(100*db["unresolved"]/N,1),
        unresolved_pct_tier3=round(100*dt["unresolved"]/N,1),
        recovered=len(rec), lost=len(lost),
        recovered_dist={k:recd[k] for k in RES},
        transitions=trans,
        mbb_drift=mbb_drift,
        fp_full=rep.get("decoy_fp_rate_at_calibration_kappa"),
        fp_oos=oos.get("decoy_fp_rate_at_calibration_kappa"),
        fp_transfers=oos.get("transfers"),
        kappa_0=rep.get("kappa_0_calibration"),
        kappa_refit=rep.get("kappa_0_refit"),
        kappa_oos_refit=oos.get("kappa_0_refit"),
        ge_recoveries=len(ge_rec),
        ge_rec_base_power_true=ge_bp_true, ge_rec_nodata=ge_nodata,
        base_ge_negnull=negfrac(base_ge), rec_ge_negnull=negfrac([t for _,_,t in ge_rec]),
    )
json.dump(out, open(f"{B}/cycle_runs_tier3/tier3_summary.json","w"), indent=2)
print(json.dumps(out, indent=2))
