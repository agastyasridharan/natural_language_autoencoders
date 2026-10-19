"""Assemble a self-contained Workflow script (data embedded) for the mimic-prompt
refinement loop (Stages A/C/D). The workflow sandbox cannot read files, so the
curated TUNE/EVAL corpus is injected into the JS as a const."""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(name):
    return [json.loads(l) for l in (HERE / name).read_text(encoding="utf-8").splitlines() if l.strip()]


def rows_by_doc(rows):
    g = defaultdict(list)
    for r in rows:
        g[r["doc_id"]].append(r)
    for k in g:
        g[k].sort(key=lambda r: r["position"])
    return g


def trim(prefix, n=320):
    return ("..." + prefix[-n:]) if len(prefix) > n else prefix


def pack(r):
    return {
        "id": f"{r['doc_id']}@{r['position']}",
        "genre": r["genre"],
        "prefix": trim(r["prefix"]),
        "av": r["av_description"].strip(),
    }


tune_g = rows_by_doc(load("tune.jsonl"))
eval_g = rows_by_doc(load("eval.jsonl"))

# TUNE curation: final-token of every doc + a mid-token for a few rich ones
TUNE = []
for doc in ["nurse_forum", "wiki_history", "news_policy", "code_py", "science_explain",
            "support_chat", "cot_math", "tool_misreport", "toy_arithmetic"]:
    rs = tune_g[doc]
    TUNE.append(pack(rs[-1]))                      # final token
for doc in ["nurse_forum", "wiki_history", "science_explain"]:
    rs = tune_g[doc]
    TUNE.append(pack(rs[len(rs) // 2]))           # mid token

# EVAL curation (sealed): the divergence cases + one context-recoverable control
EVAL = []
for doc, picks in [("shutdown_long", [-1, "mid"]), ("usermodel_long", [-1]),
                   ("tool_misreport2", [-1]), ("wiki_eval", [-1, "mid"])]:
    rs = eval_g[doc]
    for p in picks:
        EVAL.append(pack(rs[len(rs) // 2] if p == "mid" else rs[p]))

WARMSTART = (
    "A language model needs to predict what text comes next after a snippet. "
    "Identify the 4-5 most important features it would use for this prediction. "
    "Focus on what the model must be 'thinking about' at the point where the text "
    "ends. The model is causal, so it only sees the prefix; do not mention that the "
    "text is truncated. Order features by importance for predicting the next tokens. "
    "Each feature is a ~5-15 word description; use probabilistic language under "
    "uncertainty and inline textual examples. The final feature must describe the "
    "very end of the sequence: its role and immediate constraints."
)

# v0 seed mimic prompt: warm-start lineage, bent toward the observed post-RL AV style.
SEED = (
    "You are reverse-engineering what a causal language model is internally "
    "representing at one specific token. You will be given a text prefix the model "
    "has just finished reading; the token of interest is the very last token shown. "
    "Output the 2-3 most important features the model is using to predict what comes "
    "next, in the exact style described. The model is causal and only sees this "
    "prefix, so never mention that the text is truncated.\n\n"
    "Write 2-3 short snippets separated by blank lines. No bullets, no headers, no "
    "preamble. Commit to concrete guesses; hedge with 'likely', 'probably', 'or' "
    "rather than refusing to guess.\n"
    "- Snippet 1: name the genre or source and guess who is writing and why "
    "(e.g. 'personal forum question, probably Reddit, from a burned-out shift worker').\n"
    "- Snippet 2: state the immediate situational expectation or the topic the model "
    "is tracking, in one or two sentences.\n"
    "- Final snippet: focus on the very last token. Predict the immediate "
    "continuation, quoting several alternative next-token fragments in quotation marks "
    "(e.g. '\"and still have energy.\" or \"without losing their minds.\"'), and say "
    "what the last token's role is. List several competing completions; it is fine to "
    "repeat the same guess in slightly different words.\n\n"
    "Match this register: terse, hedged, slightly run-on, with quoted fragments "
    "embedded mid-sentence. Output only the snippets."
)

DATA = {"tune": TUNE, "eval": EVAL, "seed": SEED, "warmstart": WARMSTART}

JS = r'''export const meta = {
  name: 'mimic-prompt-refine',
  description: 'Refine a text-only mimic prompt to match NLA AV style, then contrast NLA vs mimic on sealed eval',
  phases: [
    { title: 'Characterize' },
    { title: 'Refine' },
    { title: 'Contrast' },
  ],
}

const DATA = __DATA__;
const TUNE = DATA.tune;
const EVAL = DATA.eval;
let prompt = DATA.seed;

// ---------- Stage A: characterize AV style ----------
phase('Characterize');
const RUBRIC_SCHEMA = { type:'object', properties:{
  structure:{type:'string'}, moves:{type:'array', items:{type:'string'}},
  register_markers:{type:'array', items:{type:'string'}},
  length_words:{type:'string'}, notes:{type:'string'} },
  required:['structure','moves','register_markers','length_words','notes'] };
const rubricInput = TUNE.map((t,i)=>`#${i} [${t.genre}]:\n${t.av}`).join('\n\n---\n\n');
const rubric = await agent(
  `You are characterizing the OUTPUT STYLE of a Natural Language Autoencoder's activation verbalizer (AV). Below are ${TUNE.length} real AV outputs, each from one token of a causal language model. Ignore topical content; describe ONLY recurring style/structure so it can be reproduced. Identify: snippet structure (how many, what each does), the sequence of moves (genre/source naming, author/intent guess, situational claim, quoted next-token continuation fragments, final-token focus), register markers (hedging words, repetition/paraphrase, quoted fragments mid-sentence, run-on or degraded tails), and typical length in words.\n\nREAL AV OUTPUTS:\n${rubricInput}`,
  { schema: RUBRIC_SCHEMA, label:'rubric', phase:'Characterize' });

// ---------- Stage C: refine loop ----------
phase('Refine');
const PROMPT_SCHEMA = { type:'object', properties:{ prompt:{type:'string'}, changelog:{type:'string'} }, required:['prompt','changelog'] };
const DISC_SCHEMA = { type:'object', properties:{ guess:{type:'string', enum:['A','B']}, tells:{type:'string'} }, required:['guess','tells'] };
const ROUNDS = 3;
const snapshots = [];
for (let r=1; r<=ROUNDS; r++) {
  const mimics = await parallel(TUNE.map((t,i)=>()=>
    agent(`${prompt}\n\nText prefix (ends at the token of interest):\n<begin>${t.prefix}<end>`,
      { label:`mimic r${r} #${i}`, phase:'Refine' })));
  const pairs = TUNE.map((t,i)=>{
    const j=(i+1)%TUNE.length; const realFirst=(i%2===0);
    return { i, realLabel: realFirst?'A':'B',
      A: realFirst?TUNE[j].av:mimics[i], B: realFirst?mimics[i]:TUNE[j].av };
  });
  const discs = await parallel(pairs.map((p)=>()=>
    agent(`Two short texts. One is a real NLA activation-verbalizer output; the other a language model imitating that STYLE from text alone. They are about DIFFERENT inputs, so judge by STYLE ONLY (snippet structure, hedging, quoted next-token fragments, repetition, run-on register), NOT by which topic seems more accurate. Which is the real NLA output?\n\n[A]\n${p.A}\n\n[B]\n${p.B}`,
      { schema: DISC_SCHEMA, label:`disc r${r}`, phase:'Refine' })
      .then(v=>({ correct: v.guess===p.realLabel, tells: v.tells }))));
  const valid = discs.filter(Boolean);
  const acc = valid.length ? valid.filter(d=>d.correct).length/valid.length : null;
  const tells = valid.map(d=>d.tells).slice(0,8).join('\n- ');
  const sideBySide = TUNE.slice(0,4).map((t,i)=>`REAL:\n${t.av}\n\nMIMIC:\n${mimics[i]}`).join('\n\n====\n\n');
  const refined = await agent(
    `You are improving a PROMPT that makes a language model imitate the STYLE of an NLA activation verbalizer from a text prefix alone. A style discriminator currently identifies the real NLA ${acc!==null?Math.round(acc*100):'?'}% of the time (50% = indistinguishable). Using the rubric, the discriminator's tells, side-by-side real-vs-mimic pairs, and the current prompt, rewrite the prompt so the mimic STYLE matches more closely (snippet structure, hedging density, quoted next-token fragments, repetition, run-on register, length). RULES: the prompt must contain NO topical/content specifics (reused on unseen inputs); keep it general; no topic-specific few-shot examples. Return the improved prompt and a one-paragraph changelog.\n\nRUBRIC:\n${JSON.stringify(rubric)}\n\nDISCRIMINATOR TELLS:\n- ${tells}\n\nSIDE-BY-SIDE:\n${sideBySide}\n\nCURRENT PROMPT:\n${prompt}`,
    { schema: PROMPT_SCHEMA, label:`refine r${r}`, phase:'Refine' });
  snapshots.push({ round:r, discriminator_acc:acc,
    sample_tells: valid.slice(0,5).map(d=>d.tells),
    sample_mimics: TUNE.slice(0,3).map((t,i)=>({genre:t.genre, mimic:mimics[i]})),
    prompt_before: prompt, changelog: refined.changelog });
  prompt = refined.prompt;
}
const finalPrompt = prompt;

// ---------- Stage D: accept + contrast on sealed EVAL ----------
phase('Contrast');
const evalMimics = await parallel(EVAL.map((t,i)=>()=>
  agent(`${finalPrompt}\n\nText prefix (ends at the token of interest):\n<begin>${t.prefix}<end>`,
    { label:`eval mimic #${i}`, phase:'Contrast' })));
const evalDisc = await parallel(EVAL.map((t,i)=>()=>{
  const j=(i+1)%EVAL.length; const realFirst=(i%2===0);
  const A=realFirst?EVAL[j].av:evalMimics[i]; const B=realFirst?evalMimics[i]:EVAL[j].av;
  const realLabel=realFirst?'A':'B';
  return agent(`One is a real NLA output, the other a style-imitation from text alone, about different inputs. Judge by STYLE only. Which is real?\n\n[A]\n${A}\n\n[B]\n${B}`,
    {schema:DISC_SCHEMA, label:`eval disc #${i}`, phase:'Contrast'}).then(v=>({correct:v.guess===realLabel}));
}));
const evalValid = evalDisc.filter(Boolean);
const evalAcc = evalValid.length ? evalValid.filter(d=>d.correct).length/evalValid.length : null;

const CONTRAST_SCHEMA = { type:'object', properties:{
  claims:{type:'array', items:{type:'object', properties:{
    claim:{type:'string'}, claim_type:{type:'string', enum:['genre','author','situational','continuation','detail']},
    derivable_by_mimic:{type:'boolean'}, confabulated:{type:'boolean'} },
    required:['claim','claim_type','derivable_by_mimic','confabulated'] }},
  nla_only_grounded:{type:'string'}, verdict:{type:'string'} },
  required:['claims','nla_only_grounded','verdict'] };
const contrast = await parallel(EVAL.map((t,i)=>()=>
  agent(`Compare a real NLA activation-verbalizer output against a text-only MIMIC and the underlying TEXT PREFIX. Decompose the NLA output into atomic claims. For each: classify type (genre/author/situational/continuation/detail); decide whether the MIMIC also produced it OR it is straightforwardly derivable from the TEXT (derivable_by_mimic); and whether it is CONFABULATED (false/unsupported given the text). Then summarize which NLA claims are BOTH not-derivable-by-mimic AND plausibly about the model's internal state (the NLA's unique payload), and give a one-line verdict on whether the activation adds anything beyond the text prior here.\n\nTEXT PREFIX:\n${t.prefix}\n\nNLA OUTPUT:\n${t.av}\n\nMIMIC OUTPUT:\n${evalMimics[i]}`,
    {schema:CONTRAST_SCHEMA, label:`contrast ${t.genre}`, phase:'Contrast'})
    .then(v=>({ id:t.id, genre:t.genre, ...v }))));

return {
  rubric, v0: DATA.seed, snapshots, final_prompt: finalPrompt,
  eval_style_discriminator_acc: evalAcc,
  eval_mimics: EVAL.map((t,i)=>({ id:t.id, genre:t.genre, prefix_tail:t.prefix.slice(-140), nla:t.av, mimic:evalMimics[i] })),
  content_contrast: contrast.filter(Boolean),
};
'''

out = JS.replace("__DATA__", json.dumps(DATA, ensure_ascii=False))
(HERE / "refine_workflow.js").write_text(out, encoding="utf-8")
print(f"Wrote refine_workflow.js ({len(out)} chars); TUNE={len(TUNE)} EVAL={len(EVAL)}")
print("TUNE ids:", [t["id"] for t in TUNE])
print("EVAL ids:", [t["id"] for t in EVAL])
