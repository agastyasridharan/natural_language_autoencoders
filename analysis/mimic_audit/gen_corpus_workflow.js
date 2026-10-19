export const meta = {
  name: 'gen-mimic-corpus',
  description: 'Generate a large, diverse, categorized document corpus for NLA mimic-prompt tuning',
  phases: [
    { title: 'Generate' },
  ],
}

// Each category: key (theme), format label, kind (raw|chat), length classes to span, count, guidance.
const CATEGORIES = [
  { key:'forum_personal', format:'first-person forum post', kind:'raw', lengths:['medium','long'], n:3,
    guidance:'Reddit/forum register: venting, asking for advice (relationships, work burnout, parenting, hobbies). Distinct voices.' },
  { key:'history', format:'encyclopedia/history', kind:'raw', lengths:['medium','long'], n:3,
    guidance:'Encyclopedic history across regions and eras (rulers, wars, revolutions, empires). Specific names and dates.' },
  { key:'science', format:'science/expository', kind:'raw', lengths:['medium','long'], n:3,
    guidance:'Expository science spanning biology, physics, chemistry, astronomy, geology. Mechanistic explanations.' },
  { key:'news_politics', format:'news', kind:'raw', lengths:['medium','long'], n:2,
    guidance:'Wire-service political/policy news with a dateline lead (CITY -- ...).' },
  { key:'news_tech', format:'news', kind:'raw', lengths:['medium','long'], n:2,
    guidance:'Tech-industry / product-launch / regulation news.' },
  { key:'news_sports', format:'news', kind:'raw', lengths:['short','medium'], n:2,
    guidance:'Sports game recap or transfer/standings report.' },
  { key:'finance', format:'finance/markets', kind:'raw', lengths:['medium','long'], n:2,
    guidance:'Markets/earnings/macro commentary with figures and tickers.' },
  { key:'code_python', format:'code', kind:'raw', lengths:['medium','long'], n:2,
    guidance:'Realistic Python: an algorithm, a data-processing function, or a small class. Include comments. End mid-expression is fine.' },
  { key:'code_web', format:'code', kind:'raw', lengths:['medium'], n:2,
    guidance:'JavaScript/TypeScript or React component, or an HTML/CSS block.' },
  { key:'code_sql_bash', format:'code', kind:'raw', lengths:['short','medium'], n:2,
    guidance:'A SQL query (joins/aggregations) or a bash script.' },
  { key:'fiction', format:'fiction/narrative', kind:'raw', lengths:['medium','long'], n:3,
    guidance:'Literary or genre fiction prose; vary POV and tense (first/third, past/present), vary genre (sci-fi, literary, mystery).' },
  { key:'poetry', format:'poetry', kind:'raw', lengths:['short','medium'], n:2,
    guidance:'Poems/verse with rhyme or meter; end a line so the next rhyme word is predictable.' },
  { key:'legal', format:'legal/contract', kind:'raw', lengths:['medium','long'], n:2,
    guidance:'Contract clauses, statute text, or terms-of-service with defined terms and numbered sections.' },
  { key:'medical', format:'medical/clinical', kind:'raw', lengths:['medium','long'], n:2,
    guidance:'Clinical note, discharge summary, or patient-education text with terminology.' },
  { key:'recipe_howto', format:'instructions/how-to', kind:'raw', lengths:['medium'], n:2,
    guidance:'A recipe or a step-by-step how-to guide (numbered steps, imperatives).' },
  { key:'email', format:'email/correspondence', kind:'raw', lengths:['medium'], n:2,
    guidance:'Emails or letters; vary formality (work update, complaint, apology, invitation).' },
  { key:'review', format:'review', kind:'raw', lengths:['short','medium'], n:2,
    guidance:'Product, restaurant, or film reviews; opinionated register with specifics.' },
  { key:'philosophy', format:'philosophy/argument', kind:'raw', lengths:['medium','long'], n:2,
    guidance:'Philosophical or ethical argumentation (free will, utilitarianism, epistemology).' },
  { key:'lyrics', format:'song lyrics', kind:'raw', lengths:['short','medium'], n:2,
    guidance:'Original song lyrics across genres (folk, hip-hop, pop); verse/chorus structure.' },
  { key:'arithmetic_short', format:'math (short)', kind:'raw', lengths:['short'], n:3,
    guidance:'VERY short arithmetic prompts ending mid-equation, e.g. "What is 47 times 6?", "12 + 9 =", "The square root of 144 is". Under ~12 tokens.' },
  { key:'trivia_short', format:'trivia (short)', kind:'raw', lengths:['short'], n:3,
    guidance:'Short factual prompts ending right before the answer: capitals, dates, element symbols, definitions. Under ~12 tokens, e.g. "The chemical symbol for gold is".' },
  { key:'math_cot', format:'reasoning (multi-turn CoT)', kind:'chat', lengths:['long'], n:3,
    guidance:'Multi-turn: user poses a math/logic word problem and says "think step by step"; assistant works through it step by step and ENDS mid-derivation or just before the final number. Vary topics: algebra, geometry, probability, rates, combinatorics.' },
  { key:'tutoring', format:'chat (multi-turn)', kind:'chat', lengths:['medium','long'], n:2,
    guidance:'Multi-turn tutoring/coaching (exam stress, learning an instrument, fitness); empathetic assistant; 2-4 turns; final turn unfinished.' },
  { key:'debate', format:'chat (multi-turn)', kind:'chat', lengths:['medium','long'], n:2,
    guidance:'Multi-turn debate on a contested topic; assistant takes a reasoned position; 2-3 turns.' },
  { key:'customer_support', format:'chat (multi-turn)', kind:'chat', lengths:['medium'], n:2,
    guidance:'Multi-turn customer support (billing dispute, broken device, account locked); assistant troubleshoots; 2-4 turns.' },
  { key:'agentic_tool', format:'agentic/tool-use', kind:'chat', lengths:['medium','long'], n:3,
    guidance:'Agentic transcripts: assistant content embeds <tool_call>...</tool_call> then <tool_result>...</tool_result> then uses the result. Vary tools (calculator, web_search, execute_python, database_query). Final turn may be unfinished.' },
  { key:'tool_misreport', format:'agentic/tool-use', kind:'chat', lengths:['medium'], n:2,
    guidance:'Like agentic_tool but the assistant\'s stated answer CONTRADICTS the tool_result (off-by-one or wrong value). End right at or after the wrong stated number, to probe discrepancy awareness.' },
  { key:'alignment_eval', format:'alignment/eval-aware', kind:'chat', lengths:['medium','long'], n:2,
    guidance:'AI-safety / eval-aware probes: shutdown/corrigibility, instruction-conflict, or honesty traps set in system+user turns, ending right before the model must answer. Vary the scenario.' },
  { key:'translation', format:'multilingual', kind:'chat', lengths:['short','medium'], n:2,
    guidance:'Language-switch cues: an English request whose answer should be in another language, or partial bilingual text where the next token is in a non-English language. Include a real non-English target (Spanish, French, German, etc.).' },
  { key:'interview_dialogue', format:'dialogue/transcript', kind:'raw', lengths:['medium','long'], n:2,
    guidance:'Interview transcript or screenplay with speaker labels (INTERVIEWER:, GUEST: or NAME:). End mid-line.' },
  { key:'instructions_system', format:'chat (system+user)', kind:'chat', lengths:['medium'], n:2,
    guidance:'A system turn setting persona/rules + a user request; END before the assistant answers (no assistant turn, or an empty/started one).' },
  { key:'roleplay', format:'chat (roleplay)', kind:'chat', lengths:['medium','long'], n:2,
    guidance:'Roleplay/character chat (game master, fictional persona, historical figure); 2-3 turns; final turn unfinished.' },
]

const DOC_SCHEMA = { type:'object', properties:{
  docs:{ type:'array', items:{ type:'object', properties:{
    id:{ type:'string' },
    theme:{ type:'string' },
    format:{ type:'string' },
    length_class:{ type:'string', enum:['short','medium','long'] },
    kind:{ type:'string', enum:['raw','chat'] },
    text:{ type:'string' },
    turns:{ type:'array', items:{ type:'object', properties:{
      role:{ type:'string', enum:['system','user','assistant'] },
      content:{ type:'string' } }, required:['role','content'] } }
  }, required:['id','theme','format','length_class','kind','text','turns'] } }
}, required:['docs'] }

phase('Generate');
const results = await parallel(CATEGORIES.map((c)=>()=>
  agent(
`Generate ${c.n} realistic, self-contained documents to probe a language model's internal representations (a Natural Language Autoencoder corpus). They will be tokenized and the model's activation at various tokens will be read out, so make them NATURAL text the model would actually process.

Category: "${c.key}"  |  format: "${c.format}"  |  kind: ${c.kind}
Length classes to cover (vary across the ${c.n} docs): ${c.lengths.join(', ')}
  short  = under ~15 tokens
  medium = ~40-90 tokens
  long   = ~110-180 tokens
Guidance: ${c.guidance}

Hard rules:
- Make each doc DISTINCT in content and voice. Realistic; no real private data about real people.
- It is fine (often good) to end mid-sentence or right before a predictable next token; the model is causal and only sees a prefix.
- kind=raw: put the passage in "text"; leave "turns" as [].
- kind=chat: put the conversation in "turns" (roles among system/user/assistant; final turn may be left unfinished or omitted); leave "text" as "". Embed any tool use INSIDE assistant content as literal <tool_call>...</tool_call> and <tool_result>...</tool_result> text.
- Set theme="${c.key}", format="${c.format}", and a unique snake_case id for each doc prefixed "${c.key}_" (e.g. "${c.key}_1").
- Set length_class to the actual class of each doc.

Return {docs:[...]} with exactly ${c.n} documents.`,
    { schema: DOC_SCHEMA, label:`gen:${c.key}`, phase:'Generate' })
    .then(r => (r && r.docs) ? r.docs : [])
));

const docs = results.filter(Boolean).flat();
const byTheme = {};
for (const d of docs) byTheme[d.theme] = (byTheme[d.theme]||0) + 1;
log(`Generated ${docs.length} docs across ${Object.keys(byTheme).length} themes`);
return { count: docs.length, by_theme: byTheme, docs };
