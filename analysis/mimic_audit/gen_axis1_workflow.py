"""Generate axis1_workflow.js with the 128 held-out pairs embedded.

Phase 1 (Decompose, Opus 4.8): split each REAL AV note into atomic claims and
label each {category, supported_by_text, present_in_mimic}.
Phase 2 (Verify, Sonnet 4.6, cross-model): adversarially re-check the payload
candidates (claims with present_in_mimic=no AND supported_by_text=no).

Agents persist their own JSON to corpus/axis1_out/ (concurrency-safe). Python
aggregates from those files.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SAMPLE = HERE / "corpus" / "axis1_sample.json"
OUTDIR = HERE / "corpus" / "axis1_out"
OUT = HERE / "axis1_workflow.js"

payload = json.loads(SAMPLE.read_text(encoding="utf-8"))
payload["base"] = str(OUTDIR)
DATA = json.dumps(payload, ensure_ascii=True)

JS = r"""export const meta = {
  name: 'mimic-audit-axis1',
  description: 'Axis 1: decompose real AV reads into claims, label derivable-by-mimic + supported-by-text, cross-model verify payload candidates',
  phases: [
    { title: 'Decompose', detail: 'Opus 4.8 splits each real AV note into atomic labeled claims' },
    { title: 'Verify', detail: 'Sonnet 4.6 adversarially re-checks payload candidates' },
  ],
}

const DATA = __DATA__;
const ITEMS = DATA.items;
const BASE = DATA.base;

const DECOMP_SCHEMA = { type: 'object', additionalProperties: false, required: ['claims'],
  properties: { claims: { type: 'array', items: {
    type: 'object', additionalProperties: false,
    required: ['text', 'category', 'supported_by_text', 'present_in_mimic'],
    properties: {
      text: { type: 'string' },
      category: { type: 'string', enum: ['classification', 'attribution', 'local_structure', 'continuation', 'other'] },
      supported_by_text: { type: 'string', enum: ['yes', 'partial', 'no'] },
      present_in_mimic: { type: 'string', enum: ['yes', 'partial', 'no'] },
    } } } } };

const VERIFY_SCHEMA = { type: 'object', additionalProperties: false, required: ['verdicts'],
  properties: { verdicts: { type: 'array', items: {
    type: 'object', additionalProperties: false,
    required: ['claim', 'really_not_in_mimic', 'really_not_in_text', 'kind', 'reason'],
    properties: {
      claim: { type: 'string' },
      really_not_in_mimic: { type: 'boolean' },
      really_not_in_text: { type: 'boolean' },
      kind: { type: 'string', enum: ['confab_specific', 'substantive', 'other'] },
      reason: { type: 'string' },
    } } } } };

function decompPrompt(it) {
  return `You are auditing an "activation verbalizer" (AV). For one position in a text you are given the TEXT PREFIX (everything up to and including the token of interest), the REAL AV NOTE the AV produced, and a MIMIC NOTE a text-only model produced from the same prefix while imitating the AV.

Break the REAL AV NOTE into atomic claims (each one single checkable assertion; split compound statements) and label each.

For each claim assign:
- category: classification (genre/medium/format), attribution (a specific named author/publication/platform/library/brand/person/place), local_structure (what the current sentence or step is doing), continuation (a guess about what comes next), or other.
- supported_by_text: does the TEXT PREFIX support it? "yes" (stated or strongly entailed), "partial" (loosely suggested), "no" (not in the text; a guess or fabrication).
- present_in_mimic: does the MIMIC NOTE assert the same claim in ANY wording (same meaning counts, exact words not required)? "yes" (clearly), "partial" (related but weaker/different), "no" (absent). Be strict: only "yes" if the mimic genuinely asserts the same content.

TEXT PREFIX:
${it.prefix}

REAL AV NOTE:
${it.real_av}

MIMIC NOTE:
${it.mimic}

=== OUTPUT INSTRUCTIONS ===
1. Use the Write tool to save your answer as valid JSON to this exact path: ${BASE}/claims_${it.idx}.json
   The JSON must be {"claims": [ ... ]} exactly matching the labels above.
2. Return the same via structured output.
Do not read any other files.`;
}

function verifyPrompt(it, candidates) {
  const list = candidates.map((c, i) => `${i + 1}. ${c}`).join('\n');
  return `You are adversarially auditing claims an "activation verbalizer" (AV) made that a text-only mimic did NOT make and that appear NOT supported by the text. Each is a candidate for genuine activation-carried content OR pure confabulation. Default to skepticism.

Given the TEXT PREFIX, the MIMIC NOTE, and CANDIDATE CLAIMS, re-check each candidate and decide:
- really_not_in_mimic: true only if the MIMIC NOTE truly does not assert it in any wording (false if you find it, even paraphrased).
- really_not_in_text: true only if the TEXT PREFIX truly does not support it (false if it actually is supported/entailed).
- kind: "confab_specific" (a specific fabricated detail: invented name/brand/number/place not in the text), "substantive" (a meaningful claim about the content's meaning, situation, or continuation, not a mere fabricated label), or "other".
- reason: one short line.

TEXT PREFIX:
${it.prefix}

MIMIC NOTE:
${it.mimic}

CANDIDATE CLAIMS:
${list}

=== OUTPUT INSTRUCTIONS ===
1. Use the Write tool to save your answer as valid JSON to this exact path: ${BASE}/verify_${it.idx}.json
   The JSON must be {"verdicts": [ ... ]} with one verdict per candidate, matching the fields above.
2. Return the same via structured output.
Do not read any files.`;
}

phase('Decompose')
let decomposed = 0, verified = 0, totalClaims = 0, totalCandidates = 0;
let cntYes = 0, cntPartial = 0, cntNo = 0;

const out = await pipeline(
  ITEMS,
  (it) => agent(decompPrompt(it), {
    label: `decomp:${it.theme}#${it.idx}`, phase: 'Decompose',
    agentType: 'general-purpose', schema: DECOMP_SCHEMA,
  }),
  (d, it) => {
    if (!d || !d.claims) return null;
    decomposed++;
    totalClaims += d.claims.length;
    for (const c of d.claims) {
      if (c.present_in_mimic === 'yes') cntYes++;
      else if (c.present_in_mimic === 'partial') cntPartial++;
      else cntNo++;
    }
    const candidates = d.claims
      .filter((c) => c.present_in_mimic === 'no' && c.supported_by_text === 'no')
      .map((c) => c.text);
    totalCandidates += candidates.length;
    if (candidates.length === 0) return { idx: it.idx, n_claims: d.claims.length, n_candidates: 0, verified: false };
    return agent(verifyPrompt(it, candidates), {
      label: `verify:${it.theme}#${it.idx}`, phase: 'Verify',
      model: 'sonnet', agentType: 'general-purpose', schema: VERIFY_SCHEMA,
    }).then((v) => {
      if (v && v.verdicts) verified++;
      return { idx: it.idx, n_claims: d.claims.length, n_candidates: candidates.length, verified: !!(v && v.verdicts) };
    });
  }
);

const clean = out.filter(Boolean);
const derivable = (cntYes + 0.5 * cntPartial) / (totalClaims || 1);
log(`decomposed=${decomposed} verified=${verified} claims=${totalClaims} candidates=${totalCandidates} derivable=${derivable.toFixed(3)}`);
return {
  n_items: ITEMS.length, decomposed, verified,
  total_claims: totalClaims, candidates: totalCandidates,
  present_yes: cntYes, present_partial: cntPartial, present_no: cntNo,
  derivable_fraction: Number(derivable.toFixed(4)),
};
"""

OUT.write_text(JS.replace("__DATA__", DATA), encoding="utf-8")
print(f"wrote {OUT.name} ({OUT.stat().st_size:,} bytes); embedded {len(payload['items'])} pairs")
print(f"agents will write to {OUTDIR}")
