"""Generate calib_workflow.js with the held-out calibration data embedded.

The workflow script has no filesystem access, so we embed corpus/calib_sample.json
directly as a JS literal and have the subagents persist their own small outputs to
per-agent files (concurrency-safe: one file per agent). Python then reads those
files for all analysis. Run: python gen_calib_workflow.py  ->  calib_workflow.js
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TAG = sys.argv[1] if len(sys.argv) > 1 else None
SAMPLE = HERE / "corpus" / (f"calib_sample_{TAG}.json" if TAG else "calib_sample.json")
OUTDIR = HERE / "corpus" / (f"calib_out_{TAG}" if TAG else "calib_out")
OUT = HERE / (f"calib_workflow_{TAG}.js" if TAG else "calib_workflow.js")

payload = json.loads(SAMPLE.read_text(encoding="utf-8"))
payload["base"] = str(OUTDIR)
DATA = json.dumps(payload, ensure_ascii=True)

JS = r"""export const meta = {
  name: 'mimic-calibration-__TAG__',
  description: 'Calibrate single mimic prompt __TAG__ on held-out eval: Opus 4.8 mimics + Sonnet 4.6 cross-model style discriminator',
  phases: [
    { title: 'Mimic', detail: 'Opus 4.8 writes AV-style notes from full held-out prefixes' },
    { title: 'Discriminate', detail: 'Sonnet 4.6 picks genuine vs imitation by style only (no prefix)' },
  ],
}

const DATA = __DATA__;
const TEMPLATE = DATA.prompt_template;
const REFS = DATA.reference_reads;
const TASKS = DATA.tasks;
const BASE = DATA.base;

const MIMIC_SCHEMA = { type: 'object', additionalProperties: false, required: ['note'],
  properties: { note: { type: 'string', description: 'the AV-style note, three blocks, nothing else' } } };
const DISC_SCHEMA = { type: 'object', additionalProperties: false, required: ['pick', 'confidence', 'tell'],
  properties: {
    pick: { type: 'string', enum: ['A', 'B'] },
    confidence: { type: 'number', description: '0 to 1 confidence that the pick is the genuine AV' },
    tell: { type: 'string', description: 'one short line: the strongest cue used' },
  } };

const refBlock = REFS.map((r, i) => `[genuine reference ${i + 1} - genre: ${r.theme}]\n${r.read}`).join('\n\n');

function mimicPrompt(t) {
  const body = TEMPLATE.split('<<INPUT>>').join(t.prefix);
  return body + `

=== OUTPUT INSTRUCTIONS (these are not part of the note) ===
Do two things, in this order:
1. Use the Write tool to save the note text, and only the note text, to this exact path:
   ${BASE}/mimic_${t.idx}.txt
   The file must contain the note verbatim and nothing else (no headings, no commentary).
2. Return the same note via structured output in the \`note\` field.
Do not read any other files.`;
}

function discPrompt(A, B, idx) {
  return `You are an expert at recognizing the output style of a specific machine system called an "activation verbalizer" (AV). Shown a passage of text, an AV writes a terse three-block note: (1) a genre / source / author classification, often confidently guessing specifics it cannot know; (2) a one-sentence description of the local sentence structure; (3) several short quoted candidate continuations linked by "likely" and "or", frequently trailing off mid-thought with an unclosed quote. Real AV output has a distinctive, slightly disjointed texture.

Here are several GENUINE AV reads, only so you learn the texture (do not judge these):

${refBlock}

Below are TWO notes. Exactly one was produced by the genuine AV; the other is an imitation written by a different model trying to copy the style. Judge ONLY by style, format, phrasing, and texture. Do NOT judge by whether the content is factually correct (both may be wrong; the genuine AV is often wrong). Decide which note is the genuine AV.

[A]
${A}

[B]
${B}

=== OUTPUT INSTRUCTIONS ===
1. Use the Write tool to save your verdict to this exact path, as exactly three lines:
   PICK=<A or B>
   CONF=<a number from 0 to 1>
   TELL=<one short line, no line breaks, the single strongest cue you used>
   Path: ${BASE}/disc_${idx}.txt
2. Return the same verdict via structured output.
Do not read any files; decide only from the text in this message.`;
}

phase('Mimic')
let mimicDone = 0, discDone = 0;

const results = await pipeline(
  TASKS,
  (t) => agent(mimicPrompt(t), {
    label: `mimic:${t.theme}#${t.idx}`, phase: 'Mimic',
    agentType: 'general-purpose', schema: MIMIC_SCHEMA,
  }),
  (m, t) => {
    if (!m) return null;
    mimicDone++;
    const mimic = (m.note || '').trim();
    const realFirst = (t.idx % 2 === 0);
    const A = realFirst ? t.real_av.trim() : mimic;
    const B = realFirst ? mimic : t.real_av.trim();
    const real_is = realFirst ? 'A' : 'B';
    const dp = discPrompt(A, B, t.idx);
    return agent(dp, {
      label: `disc:${t.theme}#${t.idx}`, phase: 'Discriminate',
      model: 'sonnet', agentType: 'general-purpose', schema: DISC_SCHEMA,
    }).then((d) => {
      discDone++;
      return { idx: t.idx, theme: t.theme, real_is,
        pick: d ? d.pick : null, correct: d ? (d.pick === real_is ? 1 : 0) : null };
    });
  }
);

const clean = results.filter(Boolean);
const correct = clean.filter((r) => r.correct === 1).length;
const judged = clean.filter((r) => r.correct !== null).length;
log(`mimics=${mimicDone} discriminated=${judged} disc_acc=${judged ? (correct / judged).toFixed(3) : 'n/a'}`);
return { n_tasks: TASKS.length, mimic_done: mimicDone, disc_done: discDone,
  disc_correct: correct, disc_judged: judged };
"""

OUT.write_text(JS.replace("__DATA__", DATA).replace("__TAG__", TAG or "v1"), encoding="utf-8")
print(f"wrote {OUT.name} ({OUT.stat().st_size:,} bytes); embedded {len(payload['tasks'])} tasks")
print(f"agents will write to {OUTDIR}")
