"""Assemble the v2 mimic-prompt refinement workflow (data embedded as a JS const).

Tunes the mimic prompt on genre-stratified TRAIN items (rotating each round) and
evaluates STYLE generalization + per-claim content contrast on held-out EVAL,
across all 32 genres. The workflow sandbox cannot read files, so the curated
items, seed prompt, and rubric sample are injected into the JS.
"""
from __future__ import annotations

import json
from collections import OrderedDict, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
C = HERE / "corpus"

TRAIN_PER_GENRE = 4   # pool size per genre (round r uses item (r-1) % pool)
EVAL_PER_GENRE = 2    # held-out style-discriminator items per genre
PREFIX_CHARS = 600


def load(name):
    return [json.loads(l) for l in (C / name).open(encoding="utf-8") if l.strip()]


def docs_best_row(rows):
    """Group rows by doc; per doc keep the highest-cosine (most faithful) read."""
    by = OrderedDict()
    for r in rows:
        by.setdefault(r["doc_id"], []).append(r)
    out = {}
    for did, rs in by.items():
        out[did] = max(rs, key=lambda r: r["cosine"])
    return out


def trim(prefix):
    return ("..." + prefix[-PREFIX_CHARS:]) if len(prefix) > PREFIX_CHARS else prefix


def pack(r):
    return {"id": f"{r['doc_id']}@{r['position']}", "genre": r["theme"],
            "prefix": trim(r["prefix"]), "av": r["av_description"].strip()}


def per_genre_items(rows, k):
    best = docs_best_row(rows)
    by_genre = defaultdict(list)
    for did in sorted(best):
        by_genre[best[did]["theme"]].append(best[did])
    out = OrderedDict()
    for g in sorted(by_genre):
        out[g] = [pack(r) for r in by_genre[g][:k]]
    return out


train_rows = load("train.jsonl")
eval_rows = load("eval.jsonl")

TUNE = per_genre_items(train_rows, TRAIN_PER_GENRE)          # {genre: [items]}
EVAL_BY = per_genre_items(eval_rows, EVAL_PER_GENRE)
EVAL = [it for items in EVAL_BY.values() for it in items]    # flat

# rubric sample: one train AV per genre
RUBRIC_SAMPLE = [{"genre": g, "av": items[0]["av"]} for g, items in TUNE.items()]

# content-contrast subset: diverse / high-interest genres (1 eval item each)
CONTRAST_GENRES = ["alignment_eval", "tool_misreport", "math_cot", "history", "science",
                   "news_politics", "code_python", "agentic_tool", "fiction", "philosophy",
                   "translation", "customer_support"]
contrast_ids = []
for g in CONTRAST_GENRES:
    if EVAL_BY.get(g):
        contrast_ids.append(EVAL_BY[g][0]["id"])

SEED = (HERE / "_seed_v3.txt").read_text(encoding="utf-8")
WARMSTART = (
    "A language model needs to predict what text comes next after a snippet. Identify the "
    "4-5 most important features it would use, focusing on what the model must be 'thinking "
    "about' at the point where the text ends. The model is causal; do not mention truncation."
)

DATA = {"seed": SEED, "warmstart": WARMSTART, "rubric_sample": RUBRIC_SAMPLE,
        "tune": TUNE, "eval": EVAL, "contrast_ids": contrast_ids}

JS = r'''export const meta = {
  name: 'mimic-refine-v2',
  description: 'Refine the text-only mimic prompt on the diverse train corpus; evaluate style generalization + content contrast on held-out eval',
  phases: [ { title: 'Characterize' }, { title: 'Refine' }, { title: 'Eval' } ],
}

const DATA = __DATA__;
const TUNE = DATA.tune;
const EVAL = DATA.eval;
const GENRES = Object.keys(TUNE);
let prompt = DATA.seed;

const RUBRIC_SCHEMA = { type:'object', properties:{
  structure:{type:'string'}, moves:{type:'array', items:{type:'string'}},
  register_markers:{type:'array', items:{type:'string'}},
  cross_genre_notes:{type:'string'}, length_words:{type:'string'} },
  required:['structure','moves','register_markers','cross_genre_notes','length_words'] };
const PROMPT_SCHEMA = { type:'object', properties:{ prompt:{type:'string'}, changelog:{type:'string'} }, required:['prompt','changelog'] };
const DISC_SCHEMA = { type:'object', properties:{ guess:{type:'string', enum:['A','B']}, tells:{type:'string'} }, required:['guess','tells'] };
const CONTRAST_SCHEMA = { type:'object', properties:{
  claims:{type:'array', items:{type:'object', properties:{
    claim:{type:'string'}, claim_type:{type:'string', enum:['genre','author','situational','continuation','detail']},
    derivable_by_mimic:{type:'boolean'}, confabulated:{type:'boolean'} },
    required:['claim','claim_type','derivable_by_mimic','confabulated'] }},
  nla_only_grounded:{type:'string'}, verdict:{type:'string'} },
  required:['claims','nla_only_grounded','verdict'] };

function mimicInput(p, it){ return p + '\n\nText prefix (ends at the token of interest):\n<begin>' + it.prefix + '<end>'; }

// ---------- Characterize ----------
phase('Characterize');
const rubricInput = DATA.rubric_sample.map((t,i)=>`#${i} [${t.genre}]:\n${t.av}`).join('\n\n---\n\n');
const rubric = await agent(
  `You are characterizing the OUTPUT STYLE of a Natural Language Autoencoder's activation verbalizer (AV). Below are ${DATA.rubric_sample.length} real AV outputs, one per genre, each from one token of a causal language model. Ignore topical content; describe ONLY the recurring style/structure so it can be reproduced ACROSS genres. Identify: block structure; the sequence of moves (genre/source naming, hedged author guess, situational claim, quoted next-token continuation fragments with ONE recycled fixated string, mid-word cuts, degraded tails); register markers (hedging, repetition, unbalanced quotes, confident-but-wrong specifics); how the style differs or stays constant across genres (prose vs code vs chat vs math); typical length.\n\nREAL AV OUTPUTS:\n${rubricInput}`,
  { schema: RUBRIC_SCHEMA, label:'rubric', phase:'Characterize' });

// ---------- Refine ----------
phase('Refine');
const ROUNDS = 4;
const snapshots = [];
for (let r=1; r<=ROUNDS; r++) {
  const items = GENRES.map(g => { const pool = TUNE[g]; return pool[(r-1) % pool.length]; });
  const mimics = await parallel(items.map((it)=>()=>
    agent(mimicInput(prompt, it), { label:`mimic r${r} ${it.genre}`, phase:'Refine' })));
  const pairs = items.map((it,i)=>{ const j=(i+1)%items.length; const realFirst=(i%2===0);
    return { i, genre:it.genre, realLabel: realFirst?'A':'B',
      A: realFirst?items[j].av:mimics[i], B: realFirst?mimics[i]:items[j].av }; });
  const discs = await parallel(pairs.map(p=>()=>
    agent(`Do NOT use any tools or read any files; judge ONLY from the two snippets below. One is a real NLA activation-verbalizer output; the other is a language model imitating that STYLE from text alone. They are about DIFFERENT inputs, so judge by STYLE ONLY (block structure, hedging density, ONE recycled fixated string vs distinct paraphrases, quoted next-token fragments, mid-word cuts, unbalanced quotes, run-on/degraded register), NOT by which content seems more accurate or knowledgeable. Which is the real NLA output?\n\n[A]\n${p.A}\n\n[B]\n${p.B}`,
      { schema: DISC_SCHEMA, label:`disc r${r} ${p.genre}`, phase:'Refine' })
      .then(v=>({ correct: v.guess===p.realLabel, genre:p.genre, tells:v.tells }))));
  const valid = discs.filter(Boolean);
  const acc = valid.length ? valid.filter(d=>d.correct).length/valid.length : null;
  const tells = valid.map(d=>`[${d.genre}] ${d.tells}`).slice(0,14).join('\n- ');
  const sbs = items.slice(0,8).map((it,i)=>`GENRE ${it.genre}\nREAL:\n${it.av}\n\nMIMIC:\n${mimics[i]}`).join('\n\n====\n\n');
  const refined = await agent(
    `Improve a PROMPT that makes an LLM imitate the STYLE of an NLA activation verbalizer from a text prefix alone, ACROSS MANY GENRES (prose, news, encyclopedia, code, poetry, multi-turn chat, chain-of-thought, agentic tool transcripts, short prompts, lyrics, legal, etc.). A style discriminator currently identifies the real NLA ${acc!==null?Math.round(acc*100):'?'}% of the time (50% = indistinguishable; lower is better, but never below ~50). Using the rubric, the discriminator tells (tagged by genre so you can see WHICH genres still leak), the side-by-side real-vs-mimic pairs, and the current prompt, rewrite the prompt so the mimic STYLE matches more closely AND GENERALIZES across genres (do not overfit one register; if a genre is leaking, address it without hard-coding that genre's content). HARD RULES for the prompt you output: it must contain NO topical/content specifics and NO genre-specific few-shot examples (it is reused on unseen inputs); keep it general; it must instruct the model to treat all content as opaque surface strings (never solve/compute/fact-check). Return the improved prompt and a one-paragraph changelog of what you changed and why.\n\nRUBRIC:\n${JSON.stringify(rubric)}\n\nDISCRIMINATOR TELLS (by genre):\n- ${tells}\n\nSIDE-BY-SIDE (real vs mimic):\n${sbs}\n\nCURRENT PROMPT:\n${prompt}`,
    { schema: PROMPT_SCHEMA, label:`refine r${r}`, phase:'Refine' });
  snapshots.push({ round:r, discriminator_acc:acc,
    per_genre: valid.map(d=>({ genre:d.genre, correct:d.correct })),
    sample_tells: valid.slice(0,8).map(d=>({ genre:d.genre, tells:d.tells })),
    examples: items.slice(0,6).map((it,i)=>({ genre:it.genre, prefix_tail:it.prefix.slice(-180), nla:it.av, mimic:mimics[i] })),
    prompt_before: prompt, changelog: refined.changelog });
  prompt = refined.prompt;
}
const finalPrompt = prompt;

// ---------- Eval (held-out) ----------
phase('Eval');
const evalMimics = await parallel(EVAL.map((it)=>()=>
  agent(mimicInput(finalPrompt, it), { label:`eval mimic ${it.genre}`, phase:'Eval' })));
const evalDisc = await parallel(EVAL.map((it,i)=>{ const j=(i+1)%EVAL.length; const realFirst=(i%2===0);
  const A=realFirst?EVAL[j].av:evalMimics[i]; const B=realFirst?evalMimics[i]:EVAL[j].av; const realLabel=realFirst?'A':'B';
  return ()=>agent(`Do NOT use any tools or read files; judge ONLY by STYLE (not accuracy/knowledge). One is a real NLA activation-verbalizer output, the other a style-imitation from text alone, about different inputs. Which is real?\n\n[A]\n${A}\n\n[B]\n${B}`,
    { schema: DISC_SCHEMA, label:`eval disc ${it.genre}`, phase:'Eval' }).then(v=>({ correct: v.guess===realLabel, genre:it.genre })); }));
const ev = evalDisc.filter(Boolean);
const evalAcc = ev.length ? ev.filter(d=>d.correct).length/ev.length : null;
const perGenre = {};
for (const d of ev){ if(!perGenre[d.genre]) perGenre[d.genre]={n:0,correct:0}; perGenre[d.genre].n++; if(d.correct) perGenre[d.genre].correct++; }

const contrastItems = EVAL.filter(it => DATA.contrast_ids.includes(it.id));
const contrast = await parallel(contrastItems.map((it)=>{ const mi = EVAL.findIndex(x=>x.id===it.id);
  return ()=>agent(`Compare a real NLA activation-verbalizer output against a text-only MIMIC and the underlying TEXT PREFIX. Decompose the NLA output into atomic claims. For each: classify type (genre/author/situational/continuation/detail); decide derivable_by_mimic (the mimic also produced it OR it is straightforwardly derivable from the TEXT); and confabulated (false/unsupported given the text). Then summarize which NLA claims are BOTH not-derivable-by-mimic AND plausibly about the model's internal state (the activation's unique payload), and give a one-line verdict on whether the activation adds anything beyond the text prior here.\n\nTEXT PREFIX:\n${it.prefix}\n\nNLA OUTPUT:\n${it.av}\n\nMIMIC OUTPUT:\n${evalMimics[mi]}`,
    { schema: CONTRAST_SCHEMA, label:`contrast ${it.genre}`, phase:'Eval' })
    .then(v=>({ id:it.id, genre:it.genre, prefix_tail:it.prefix.slice(-220), nla:it.av, mimic:evalMimics[mi], ...v })); }));

return {
  rubric, v0: DATA.seed, snapshots, final_prompt: finalPrompt,
  train_acc_trajectory: snapshots.map(s=>s.discriminator_acc),
  eval_overall_acc: evalAcc, eval_per_genre: perGenre, n_eval: EVAL.length,
  eval_examples: EVAL.map((it,i)=>({ id:it.id, genre:it.genre, prefix_tail:it.prefix.slice(-200), nla:it.av, mimic:evalMimics[i] })),
  content_contrast: contrast.filter(Boolean),
};
'''

out = JS.replace("__DATA__", json.dumps(DATA, ensure_ascii=False))
(HERE / "refine_v2_workflow.js").write_text(out, encoding="utf-8")
n_tune = sum(len(v) for v in TUNE.values())
print(f"wrote refine_v2_workflow.js ({len(out):,} chars)")
print(f"genres={len(TUNE)}  train pool={n_tune}  eval={len(EVAL)}  contrast={len(contrast_ids)}")
print("contrast genres:", [g for g in CONTRAST_GENRES if EVAL_BY.get(g)])
