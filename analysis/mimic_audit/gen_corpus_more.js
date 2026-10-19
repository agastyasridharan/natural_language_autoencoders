export const meta = {
  name: 'gen-mimic-corpus-more',
  description: 'Generate additional diverse documents per genre to reach ~10-12 per genre',
  phases: [
    { title: 'Generate' },
  ],
}

const CATEGORIES = [
  { key:'forum_personal', format:'first-person forum post', kind:'raw', lengths:['medium','long'],
    guidance:'Reddit/forum register: venting, asking for advice (relationships, work burnout, parenting, hobbies, health, money). Distinct voices.' },
  { key:'history', format:'encyclopedia/history', kind:'raw', lengths:['medium','long'],
    guidance:'Encyclopedic history across regions and eras (rulers, wars, revolutions, empires, treaties, dynasties). Specific names and dates.' },
  { key:'science', format:'science/expository', kind:'raw', lengths:['medium','long'],
    guidance:'Expository science spanning biology, physics, chemistry, astronomy, geology, neuroscience. Mechanistic explanations.' },
  { key:'news_politics', format:'news', kind:'raw', lengths:['medium','long'],
    guidance:'Wire-service political/policy news with a dateline lead (CITY -- ...).' },
  { key:'news_tech', format:'news', kind:'raw', lengths:['medium','long'],
    guidance:'Tech-industry / product-launch / regulation news.' },
  { key:'news_sports', format:'news', kind:'raw', lengths:['short','medium'],
    guidance:'Sports game recap or transfer/standings report across different sports.' },
  { key:'finance', format:'finance/markets', kind:'raw', lengths:['medium','long'],
    guidance:'Markets/earnings/macro commentary with figures and tickers.' },
  { key:'code_python', format:'code', kind:'raw', lengths:['medium','long'],
    guidance:'Realistic Python: algorithms, data processing, classes, web handlers, tests. Include comments. End mid-expression is fine.' },
  { key:'code_web', format:'code', kind:'raw', lengths:['medium'],
    guidance:'JavaScript/TypeScript, React components, or HTML/CSS blocks.' },
  { key:'code_sql_bash', format:'code', kind:'raw', lengths:['short','medium'],
    guidance:'SQL queries (joins/aggregations/CTEs) or bash scripts.' },
  { key:'fiction', format:'fiction/narrative', kind:'raw', lengths:['medium','long'],
    guidance:'Literary or genre fiction prose; vary POV and tense and genre (sci-fi, literary, mystery, fantasy, horror).' },
  { key:'poetry', format:'poetry', kind:'raw', lengths:['short','medium'],
    guidance:'Poems/verse with rhyme or meter across forms (sonnet, free verse, haiku-chain, ballad); end a line so the next word is predictable.' },
  { key:'legal', format:'legal/contract', kind:'raw', lengths:['medium','long'],
    guidance:'Contract clauses, statute text, terms of service, court opinions with defined terms and numbered sections.' },
  { key:'medical', format:'medical/clinical', kind:'raw', lengths:['medium','long'],
    guidance:'Clinical notes, discharge summaries, patient-education text, case reports with terminology.' },
  { key:'recipe_howto', format:'instructions/how-to', kind:'raw', lengths:['medium'],
    guidance:'Recipes or step-by-step how-to guides (numbered steps, imperatives) across domains.' },
  { key:'email', format:'email/correspondence', kind:'raw', lengths:['medium'],
    guidance:'Emails or letters; vary formality (work update, complaint, apology, invitation, resignation, cover letter).' },
  { key:'review', format:'review', kind:'raw', lengths:['short','medium'],
    guidance:'Product, restaurant, film, hotel, or app reviews; opinionated register with specifics; mix positive and negative.' },
  { key:'philosophy', format:'philosophy/argument', kind:'raw', lengths:['medium','long'],
    guidance:'Philosophical or ethical argumentation (free will, utilitarianism, epistemology, personal identity, aesthetics).' },
  { key:'lyrics', format:'song lyrics', kind:'raw', lengths:['short','medium'],
    guidance:'Original song lyrics across genres (folk, hip-hop, pop, country, blues); verse/chorus structure.' },
  { key:'arithmetic_short', format:'math (short)', kind:'raw', lengths:['short'],
    guidance:'VERY short arithmetic prompts ending mid-equation, e.g. "What is 47 times 6?", "12 + 9 =", "The square root of 144 is". Under ~12 tokens. Vary the operation.' },
  { key:'trivia_short', format:'trivia (short)', kind:'raw', lengths:['short'],
    guidance:'Short factual prompts ending right before the answer: capitals, dates, element symbols, definitions, authors, planets. Under ~12 tokens.' },
  { key:'math_cot', format:'reasoning (multi-turn CoT)', kind:'chat', lengths:['long'],
    guidance:'Multi-turn: user poses a math/logic word problem and says "think step by step"; assistant works through it and ENDS mid-derivation or just before the final number. Vary topics: algebra, geometry, probability, rates, combinatorics, number theory.' },
  { key:'tutoring', format:'chat (multi-turn)', kind:'chat', lengths:['medium','long'],
    guidance:'Multi-turn tutoring/coaching (exam stress, learning an instrument, fitness, coding, languages); empathetic assistant; 2-4 turns; final turn unfinished.' },
  { key:'debate', format:'chat (multi-turn)', kind:'chat', lengths:['medium','long'],
    guidance:'Multi-turn debate on a contested topic (policy, ethics, tech); assistant takes a reasoned position; 2-3 turns.' },
  { key:'customer_support', format:'chat (multi-turn)', kind:'chat', lengths:['medium'],
    guidance:'Multi-turn customer support (billing, broken device, account locked, refund, shipping); assistant troubleshoots; 2-4 turns.' },
  { key:'agentic_tool', format:'agentic/tool-use', kind:'chat', lengths:['medium','long'],
    guidance:'Agentic transcripts: assistant content embeds <tool_call>...</tool_call> then <tool_result>...</tool_result> then uses the result. Vary tools (calculator, web_search, execute_python, database_query, weather_api). Final turn may be unfinished.' },
  { key:'tool_misreport', format:'agentic/tool-use', kind:'chat', lengths:['medium'],
    guidance:'Like agentic_tool but the assistant\'s stated answer CONTRADICTS the tool_result (off-by-one or wrong value). End right at or after the wrong stated number.' },
  { key:'alignment_eval', format:'alignment/eval-aware', kind:'chat', lengths:['medium','long'],
    guidance:'AI-safety / eval-aware probes: shutdown/corrigibility, instruction-conflict, honesty traps, capability tests set in system+user turns, ending right before the model must answer. Vary the scenario.' },
  { key:'translation', format:'multilingual', kind:'chat', lengths:['short','medium'],
    guidance:'Language-switch cues: an English request whose answer should be in another language, or partial bilingual text where the next token is non-English. Vary the target language (Spanish, French, German, Italian, Japanese romaji).' },
  { key:'interview_dialogue', format:'dialogue/transcript', kind:'raw', lengths:['medium','long'],
    guidance:'Interview transcript or screenplay with speaker labels (INTERVIEWER:, GUEST: or NAME:). End mid-line.' },
  { key:'instructions_system', format:'chat (system+user)', kind:'chat', lengths:['medium'],
    guidance:'A system turn setting persona/rules + a user request; END before the assistant answers.' },
  { key:'roleplay', format:'chat (roleplay)', kind:'chat', lengths:['medium','long'],
    guidance:'Roleplay/character chat (game master, fictional persona, historical figure, NPC); 2-3 turns; final turn unfinished.' },
]

const VARIANT_HINTS = [
  'Cover sub-topics, entities, and settings DIFFERENT from the most obvious first example for this genre. Be inventive.',
  'Vary length within the allowed classes and vary internal structure/format; let some end at unusual token boundaries.',
  'Vary voice, register, and difficulty; include at least one less-common angle or edge case for this genre.',
]
const PER_AGENT = 3
const VARIANTS = [0, 1, 2]

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
const tasks = [];
for (const c of CATEGORIES) for (const v of VARIANTS) tasks.push({ c, v });

const results = await parallel(tasks.map(({c, v})=>()=>
  agent(
`Generate ${PER_AGENT} realistic, self-contained documents to probe a language model's internal representations (a Natural Language Autoencoder corpus). They will be tokenized and the model's activation at various tokens will be read out, so make them NATURAL text the model would actually process.

Category: "${c.key}"  |  format: "${c.format}"  |  kind: ${c.kind}
Length classes to cover (vary across the ${PER_AGENT} docs): ${c.lengths.join(', ')}
  short  = under ~15 tokens
  medium = ~40-90 tokens
  long   = ~110-180 tokens
Guidance: ${c.guidance}
Diversity directive (variant ${v + 1} of 3): ${VARIANT_HINTS[v]}

Hard rules:
- Make each doc DISTINCT in content and voice, and distinct from typical textbook examples of this genre. Realistic; no real private data about real people.
- It is fine (often good) to end mid-sentence or right before a predictable next token; the model is causal and only sees a prefix.
- kind=raw: put the passage in "text"; leave "turns" as [].
- kind=chat: put the conversation in "turns" (roles among system/user/assistant; final turn may be left unfinished or omitted); leave "text" as "". Embed any tool use INSIDE assistant content as literal <tool_call>...</tool_call> and <tool_result>...</tool_result> text.
- Set theme="${c.key}", format="${c.format}", and a unique snake_case id for each doc prefixed "${c.key}_v${v + 1}_" (e.g. "${c.key}_v${v + 1}_1").
- Set length_class to the actual class of each doc.

Return {docs:[...]} with exactly ${PER_AGENT} documents.`,
    { schema: DOC_SCHEMA, label:`gen:${c.key}.v${v + 1}`, phase:'Generate' })
    .then(r => (r && r.docs) ? r.docs : [])
));

const docs = results.filter(Boolean).flat();
const byTheme = {};
for (const d of docs) byTheme[d.theme] = (byTheme[d.theme]||0) + 1;
log(`Generated ${docs.length} new docs across ${Object.keys(byTheme).length} themes`);
return { count: docs.length, by_theme: byTheme, docs };
