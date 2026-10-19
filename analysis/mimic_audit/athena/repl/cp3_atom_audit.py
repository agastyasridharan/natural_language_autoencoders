#!/usr/bin/env python3
"""CP3 - single-activation atom audit (idx 20, qwen7b). Self-contained for gpurun.

Reuses the exact activation gold.npy[20] and the exact AV explanation from run
qwen7b-20260622-121049-fineweb, plus its precomputed two-layer decomposition. Runs the
SAME claim-ablation measurement two ways on the same h and same text:
  * BLOCK mode  (old): split on blank lines -> ~3 coarse snippets.
  * ATOM  mode  (new): score the Layer-1 theme/detail content claims individually,
                       connectives excluded.
and prints a side-by-side so we can see what atomic resolution reveals that the block
verdict hid. Writes the full record to /data/agastyas/nla_mimic/claim_ablation_atoms/.
"""
import json, sys
from pathlib import Path
import numpy as np

for p in ("/data/agastyas/nla_mimic/code", "/data/agastyas/nla_mimic/code/repl"):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download
import claim_ablation as ca
from avgen_model import load_av, read_sidecar
from nla_inference import NLACritic
from repl_common import MODELS

DATA = json.loads(r'''{"idx": 20, "doc_id": "fineweb:20", "read_token": " possess", "position": 102, "n_tokens": 149, "explanation": "<explanation>\nNews report format with newspaper headline structure detailing a Georgia shooting incident, establishing factual narrative about a celebrity's firearms conviction.\n\nThe sentence \"The 53-year-old's license to carry firearms and his permit to possess\" is mid-clause describing a second conviction context, listing restrictions on a convicted felon's gun privileges, implying a specific license or permit related to his registered weapons.\n\nFinal token \"possess\" ends mid-phrase (\"permission to possess\"), part of a parenthetical clause listing qualifications (\"his license to possess firearms to qualify to possess\"), strongly expecting \"firearms\" or \"controlled weapons\" or \"a restricted license had expired\" or \"dangerous weapons was suspended.\"\n</explanation>", "z_inner": "News report format with newspaper headline structure detailing a Georgia shooting incident, establishing factual narrative about a celebrity's firearms conviction.\n\nThe sentence \"The 53-year-old's license to carry firearms and his permit to possess\" is mid-clause describing a second conviction context, listing restrictions on a convicted felon's gun privileges, implying a specific license or permit related to his registered weapons.\n\nFinal token \"possess\" ends mid-phrase (\"permission to possess\"), part of a parenthetical clause listing qualifications (\"his license to possess firearms to qualify to possess\"), strongly expecting \"firearms\" or \"controlled weapons\" or \"a restricted license had expired\" or \"dangerous weapons was suspended.\"", "atoms": [{"text": "News report format with newspaper headline structure", "type": "theme", "subtype": "genre and structural format", "start": 0, "end": 52, "hedged": [], "assertion": "The source is written in a news report format with a newspaper headline structure.", "role": "describes", "regex_confirmed": null}, {"text": " detailing a Georgia shooting incident,", "type": "theme", "subtype": "topic and subject matter", "start": 52, "end": 91, "hedged": [], "assertion": "The source details a shooting incident in Georgia.", "role": "describes", "regex_confirmed": null}, {"text": " establishing factual narrative about a celebrity's firearms conviction.", "type": "theme", "subtype": "narrative purpose and subject", "start": 91, "end": 163, "hedged": [], "assertion": "The source establishes a factual narrative about a celebrity's firearms conviction.", "role": "describes", "regex_confirmed": null}, {"text": "\n\nThe sentence ", "type": "connective", "subtype": "referential introducer", "start": 163, "end": 178, "hedged": [], "assertion": "", "role": "glue", "regex_confirmed": null}, {"text": "\"The 53-year-old's license to carry firearms and his permit to possess\"", "type": "detail", "subtype": "quoted string", "start": 178, "end": 249, "hedged": [], "assertion": "The source contains the quoted string \"The 53-year-old's license to carry firearms and his permit to possess\".", "role": "quotes", "regex_confirmed": true}, {"text": " is mid-clause", "type": "theme", "subtype": "syntactic position", "start": 249, "end": 263, "hedged": [], "assertion": "The source reads as a mid-clause construction.", "role": "describes", "regex_confirmed": null}, {"text": " describing a second conviction context,", "type": "theme", "subtype": "contextual framing", "start": 263, "end": 303, "hedged": [], "assertion": "The source describes a second conviction context.", "role": "describes", "regex_confirmed": null}, {"text": " listing restrictions on a convicted felon's gun privileges,", "type": "theme", "subtype": "content and subject matter", "start": 303, "end": 363, "hedged": [], "assertion": "The source lists restrictions on a convicted felon's gun privileges.", "role": "describes", "regex_confirmed": null}, {"text": " implying a specific license or permit related to his registered weapons.", "type": "theme", "subtype": "implication", "start": 363, "end": 436, "hedged": ["implying"], "assertion": "The source implies a specific license or permit related to the subject's registered weapons.", "role": "describes", "regex_confirmed": null}, {"text": "\n\nFinal token ", "type": "connective", "subtype": "referential introducer", "start": 436, "end": 450, "hedged": [], "assertion": "", "role": "glue", "regex_confirmed": null}, {"text": "\"possess\"", "type": "detail", "subtype": "quoted token", "start": 450, "end": 459, "hedged": [], "assertion": "The source contains the final token \"possess\".", "role": "quotes", "regex_confirmed": true}, {"text": " ends mid-phrase ", "type": "theme", "subtype": "syntactic position", "start": 459, "end": 476, "hedged": [], "assertion": "The source ends mid-phrase.", "role": "describes", "regex_confirmed": null}, {"text": "(\"permission to possess\")", "type": "detail", "subtype": "quoted phrase context", "start": 476, "end": 501, "hedged": [], "assertion": "The source contains the phrase \"permission to possess\" as the mid-phrase context.", "role": "quotes", "regex_confirmed": true}, {"text": ", part of a parenthetical clause listing qualifications ", "type": "theme", "subtype": "syntactic role and structure", "start": 501, "end": 557, "hedged": [], "assertion": "The source is part of a parenthetical clause listing qualifications.", "role": "describes", "regex_confirmed": null}, {"text": "(\"his license to possess firearms to qualify to possess\")", "type": "detail", "subtype": "quoted parenthetical string", "start": 557, "end": 614, "hedged": [], "assertion": "The source contains the quoted parenthetical string \"his license to possess firearms to qualify to possess\".", "role": "quotes", "regex_confirmed": true}, {"text": ", strongly expecting ", "type": "connective", "subtype": "expectation introducer", "start": 614, "end": 635, "hedged": [], "assertion": "", "role": "glue", "regex_confirmed": null}, {"text": "\"firearms\"", "type": "detail", "subtype": "expected next token", "start": 635, "end": 645, "hedged": ["strongly"], "assertion": "The source is strongly expecting the next token \"firearms\".", "role": "predicts", "regex_confirmed": true}, {"text": " or ", "type": "connective", "subtype": "disjunction", "start": 645, "end": 649, "hedged": [], "assertion": "", "role": "glue", "regex_confirmed": null}, {"text": "\"controlled weapons\"", "type": "detail", "subtype": "expected next token", "start": 649, "end": 669, "hedged": ["strongly"], "assertion": "The source is strongly expecting the next token \"controlled weapons\".", "role": "predicts", "regex_confirmed": true}, {"text": " or ", "type": "connective", "subtype": "disjunction", "start": 669, "end": 673, "hedged": [], "assertion": "", "role": "glue", "regex_confirmed": null}, {"text": "\"a restricted license had expired\"", "type": "detail", "subtype": "expected next token", "start": 673, "end": 707, "hedged": ["strongly"], "assertion": "The source is strongly expecting the next token \"a restricted license had expired\".", "role": "predicts", "regex_confirmed": true}, {"text": " or ", "type": "connective", "subtype": "disjunction", "start": 707, "end": 711, "hedged": [], "assertion": "", "role": "glue", "regex_confirmed": null}, {"text": "\"dangerous weapons was suspended.\"", "type": "detail", "subtype": "expected next token", "start": 711, "end": 745, "hedged": ["strongly"], "assertion": "The source is strongly expecting the next token \"dangerous weapons was suspended\".", "role": "predicts", "regex_confirmed": true}], "old_block": [{"text": "News report format with newspaper headline structure detailing a Georgia shooting incident, establishing factual narrative about a celebrity's firearms conviction.", "verdict": "confabulated", "dlogp": -0.2753477096557617, "null_thr": 0.2759617269039154}, {"text": "The sentence \"The 53-year-old's license to carry firearms and his permit to possess\" is mid-clause describing a second conviction context, listing restrictions on a convicted felon's gun privileges, implying a specific license or permit related to his registered weapons.", "verdict": "entangled", "dlogp": 4.851047515869141, "null_thr": -0.09134693443775177}, {"text": "Final token \"possess\" ends mid-phrase (\"permission to possess\"), part of a parenthetical clause listing qualifications (\"his license to possess firearms to qualify to possess\"), strongly expecting \"firearms\" or \"controlled weapons\" or \"a restricted license had expired\" or \"dangerous weapons was suspended.\"", "verdict": "grounded", "dlogp": 30.083744049072266, "null_thr": 0.14430028200149536}]}''')

KEY = "qwen7b"
IDX = DATA["idx"]
GOLD_DIR = Path("/data/agastyas/nla_mimic/out_corpus_qwen7b")
OUT_DIR = Path("/data/agastyas/nla_mimic/claim_ablation_atoms")
OUT_DIR.mkdir(parents=True, exist_ok=True)
R, ALPHA, NULL_PCT, SPEC_MARGIN, SEED = 10, 1.0, 95.0, 1.5, IDX  # seed=idx: matches old run's seed+i

def hr(s=""):
    print("\n" + "=" * 78 + ("\n" + s if s else ""), flush=True)

def main():
    cfg = MODELS[KEY]
    gold = np.load(GOLD_DIR / "gold.npy")
    manifest = json.loads((GOLD_DIR / "manifest.json").read_text())
    h = gold[IDX]
    m = manifest[IDX] if IDX < len(manifest) else {}

    hr(f"CP3  idx={IDX}  model={cfg['label']}  layer={cfg['layer']}  d_model={cfg['d_model']}")
    print(f"read_token={DATA['read_token']!r}  doc_id={DATA['doc_id']}  |h|={float(np.linalg.norm(h)):.2f}", flush=True)

    print(f"loading AV {cfg['av']} ...", flush=True)
    av_dir = snapshot_download(cfg["av"])
    tok = AutoTokenizer.from_pretrained(av_dir, trust_remote_code=True)
    sc = read_sidecar(av_dir)
    _, model = load_av(cfg)
    inj = ca.register_injector(model, sc, cfg["family"], tok)   # prints marker-neighbor assert
    print(f"loading AR {cfg['ar']} ...", flush=True)
    ar_dir = snapshot_download(cfg["ar"])
    critic = NLACritic(ar_dir, device_map="auto")
    print(f"injection_scale={sc['injection_scale']}  mse_scale={critic.mse_scale}", flush=True)

    kw = dict(idx=IDX, doc_id=DATA["doc_id"], position=DATA["position"],
              n_tokens=DATA["n_tokens"], read_token=DATA["read_token"],
              R=R, alpha=ALPHA, null_pct=NULL_PCT, spec_margin=SPEC_MARGIN, seed=SEED)

    # ---- BLOCK mode (reproduce the old coarse method on the reused text) ----
    hr("BLOCK MODE (old, blank-line snippets)")
    brec, _ = ca.audit_activation(model, tok, sc, cfg["family"], critic, inj, h,
                                  precomputed_z=DATA["explanation"], **kw)
    print(f"n_cjk_this={int(brec.get('cjk', False))}  status={brec['status']}", flush=True)
    for s in brec["snippet_results"]:
        print(f"  [{s['i']}] {s['verdict']:<12} dlogp={s['dlogp']:.2f} "
              f"null_thr={s['null_thr']:.2f} col_off_max={s['col_off_max']:.2f}  "
              f":: {s['text'][:64]!r}", flush=True)

    # ---- ATOM mode (new, typed content claims) ----
    hr("ATOM MODE (new, theme/detail content claims; connectives excluded)")
    arec, avec = ca.audit_activation_atoms(
        model, tok, sc, cfg["family"], critic, inj, h,
        z_assistant=DATA["explanation"], z_inner=DATA["z_inner"], atoms=DATA["atoms"], **kw)
    print(f"n_content={arec['n_content']}  n_connective={arec['n_connective']}  "
          f"located={arec['located_count']}  status={arec['status']}", flush=True)
    print(f"cos_full={arec['cos_full']:.3f}  mse_full={arec['mse_full']:.3f}", flush=True)
    print("\n  i  type   subtype                     role      hg  dlogp   nullthr  colmax  verdict", flush=True)
    for s in arec["snippet_results"]:
        hg = "H" if s["hedged"] else " "
        cm = s["col_off_max"]
        print(f"  {s['i']:<2} {s['type']:<6} {s['subtype'][:26]:<26} {s['role']:<9} {hg}  "
              f"{(s['dlogp'] if s['dlogp'] is not None else float('nan')):6.2f}  "
              f"{(s['null_thr'] if s['null_thr'] is not None else float('nan')):6.2f}  "
              f"{(cm if cm is not None else float('nan')):6.2f}  {s['verdict']}  :: {s['text'][:40]!r}", flush=True)
    print("\n  connectives (unscored):", flush=True)
    for s in arec["connective_results"]:
        print(f"    - {s['text'][:50]!r}", flush=True)

    hr("BY TYPE (located content claims)")
    for t, d in sorted(arec["by_type"].items()):
        print(f"  {t:<6} n={d['n']:<3} located={d['located']:<3} "
              f"grounded={d['grounded']} entangled={d['entangled']} confab={d['confabulated']} "
              f"unloc={d['unlocated']}", flush=True)

    # ---- specificity matrix (compact) + strongest off-diagonal entanglement ----
    hr("SPECIFICITY MATRIX  spec[i][j] = drop in claim j's logp when claim i is ablated")
    n = arec["n_content"]
    M = np.array([[ (s["spec_row"][j] if s["spec_row"][j] is not None else float('nan'))
                    for j in range(n)] for s in arec["snippet_results"]], float)
    with np.printoptions(precision=1, suppress=True, linewidth=200):
        print(M, flush=True)
    off = M.copy(); np.fill_diagonal(off, -1e9)
    fi, fj = np.unravel_index(np.nanargmax(off), off.shape)
    print(f"\n  strongest cross-claim leakage: ablating claim {fi} drops claim {fj} by "
          f"{off[fi, fj]:.2f} nats", flush=True)
    print(f"    {fi}: {arec['snippet_results'][fi]['text'][:60]!r}", flush=True)
    print(f"    {fj}: {arec['snippet_results'][fj]['text'][:60]!r}", flush=True)

    # ---- block -> atom fragmentation: how each old block's verdict splits at atomic scale ----
    hr("BLOCK -> ATOM  (how each coarse block fragments into typed claims)")
    z = DATA["z_inner"]
    snips = ca.split_snippets(z)
    bspan, cur = [], 0
    for sn in snips:
        st = z.find(sn, cur); bspan.append((st, st + len(sn))); cur = st + len(sn)
    for bi, (bs, be) in enumerate(bspan):
        bl = brec["snippet_results"][bi]
        print(f"\n  BLOCK {bi}  [{bl['verdict']}]  dlogp={bl['dlogp']:.1f}  :: {snips[bi][:56]!r}", flush=True)
        for s in arec["snippet_results"]:
            cs, ce = s["char_span"]
            mid = (cs + ce) / 2
            if bs <= mid < be:
                print(f"      atom {s['i']:<2} {s['type']:<6}/{s['subtype'][:20]:<20} "
                      f"{s['verdict']:<12} dlogp={(s['dlogp'] or float('nan')):6.2f}  :: {s['text'][:38]!r}", flush=True)

    # ---- injection sanity ----
    hr("INJECTION SANITY")
    tok_h = avec["tok_logp_h"]; tok_ab = avec["tok_logp_ablate"]
    print(f"  n_cjk (this activation) = {int(arec.get('cjk', False))}  (0 = injection healthy)", flush=True)
    print(f"  marker_pos = {int(avec['marker_pos'][0])}  == injector.p = {inj.p}  -> {int(avec['marker_pos'][0])==inj.p}", flush=True)
    print(f"  teacher-forced tokens = {tok_h.shape[0]}   ablation rows = {tok_ab.shape[0]} (== n_content)", flush=True)
    print(f"  block-mode reproduced old verdicts:", flush=True)
    for old, new in zip(DATA["old_block"], brec["snippet_results"]):
        print(f"    old={old['verdict']:<12}(dlogp {old['dlogp']:.1f})  now={new['verdict']:<12}(dlogp {new['dlogp']:.1f})  match={old['verdict']==new['verdict']}", flush=True)

    out = OUT_DIR / f"cp3_idx{IDX}.json"
    out.write_text(json.dumps(dict(block=brec, atom=arec, spec_matrix=M.tolist(),
                                   block_spans=bspan), ensure_ascii=False, indent=2))
    np.savez_compressed(OUT_DIR / f"cp3_idx{IDX}_vectors.npz", **avec)
    hr(f"WROTE {out}")

if __name__ == "__main__":
    main()
