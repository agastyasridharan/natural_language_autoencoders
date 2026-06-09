# What we're trying to do — research north star

**In one sentence:** make Natural Language Autoencoders (NLAs) *trustworthy* —
explanations of a model's internal state that you can actually believe — by
understanding their mechanism well enough to fix it.

---

## The setup, intuitively

A large language model "thinks" in vectors. At every layer and every token
position there's a high-dimensional **activation** encoding what the model is
representing right then. We can't read those vectors directly.

An **NLA** is a way to translate one such vector into a paragraph of plain
English — with no labels and no human supervision. It uses two copies of the
model:

- the **verbalizer (AV)** takes an activation and *writes* a description of it;
- the **reconstructor (AR)** reads that description and tries to *rebuild* the
  original vector.

Train them so the rebuilt vector matches the original (minimize reconstruction
error) and — remarkably — the descriptions come out reading like genuine
interpretations of what the model was doing. Nothing in the training rewards
"being interpretable"; it falls out of the reconstruction game. It's an
autoencoder whose latent code is a paragraph of English.

## The catch — why this research exists

NLAs are **fluent but not always faithful.** They reliably nail the *theme* of
an activation ("this is an AI talking about itself") but **confabulate the
specifics** — inventing entities, names, and details that aren't in the vector,
stated with full confidence. The reconstruction score stays high the whole
time, because it mostly measures coarse/thematic *direction* and is nearly blind
to specifics. So the metric says "great!" while the explanation quietly makes
things up.

Two things we don't yet understand well enough to fix this:

1. **How does the AV actually read the activation?** The vector comes from a
   deep layer, but it gets injected at the *input* as a fake token, kept alive
   by a large hand-tuned scaling number until some later layer "reads" it. We
   don't know where, or how, that read happens.
2. **Why does it fail the way it does?** Are confabulation, vagueness, the
   out-of-distribution cliff, and the rest a grab-bag of separate bugs — or one
   mechanism wearing many masks?

## Our goal

> **Understand the NLA's reading mechanism and its failure mechanism well enough
> to (a) put a trustworthy, label-free confidence number on every claim an
> explanation makes, and (b) change the method — its injection and its training
> objective — so explanations are faithful about *specifics*, not just themes.**

## Two angles, one goal

**A — How it reads: the *reading pathway*.** Reverse-engineer the circuit by
which the AV decodes the injected activation: at what depth the raw vector stops
being "a token" and becomes "read content," and which layers/heads do the
reading. Then use that understanding to replace the magic scaling number with
something principled — a *learned* injection map, or injecting the vector at its
*native* layer. *(This is the current thread; the white-box harness is built.)*

**B — How/why it fails: the *failure-mode program*.** Catalog every failure and
test whether they collapse to a single mechanism. The leading answer is
**evidence-gated prior substitution**: wherever the activation carries little
evidence about a claim, the model fills the gap with its own prior (conditioned
on the theme it *did* recover), and the reconstruction metric rewards it because
the metric only constrains coarse directions. The payoffs: a **ground-truth-free
"evidence" score** — built from how much each claim moves the reconstruction,
how often it recurs, and how in-distribution the activation is — that flags
untrustworthy claims at inference; and an **objective-level fix** (e.g. a
whitened reconstruction loss) that forces the model to care about specifics.

**Why they're the same goal.** Both are about the gap between *fidelity*
(reconstructs well) and *faithfulness* (actually true). Angle A finds the
**mechanism** — where the activation is read and where it's ignored. Angle B
measures the **consequence** — where it confabulates — and turns it into a
number. A's causal question ("where is this claim actually read, and how much
does it move the reconstruction?") is literally the mechanistic substrate of B's
"evidence field." And they run on one shared tool: the `nla/whitebox` white-box
harness.

## What success looks like

- A white-box harness that reproduces the released NLAs exactly.
  *(Phase 0 — built; pending the on-hardware validation gate.)*
- A map of the reading pathway: which layers/positions causally carry the
  injected activation into the explanation.
- A single label-free **trust score** that predicts, claim by claim, when an
  explanation is faithful — validated against held-out ground truth.
- At least one concrete **improvement** — principled injection and/or an
  evidence-sensitive loss — that measurably raises faithfulness about specifics
  without hurting fidelity.

## Where things live

| | |
|---|---|
| **this file** | the north star — *why* and *what*, intuitively |
| `~/.claude/plans/sparkling-tumbling-bird.md` | full failure-mode program plan (Part A + B, phase by phase) |
| `research/part_a_failure_taxonomy.md` | failure taxonomy + the evidence-gated-prior-substitution argument, grounded in the released decodes |
| `research/neuronpedia_nla_api.md` | hosted-inference API contract (the 2 hosted models) |
| `nla/whitebox/` | the shared white-box harness (`README.md`, `COLAB.md`); reading-pathway experiments live here; also the local inference backend for the failure-mode harness |
| `nla/failure_analysis/` | *(planned)* the inference-only failure-analysis instruments |
| `papers/nla paper.pdf`, `papers/building better AOs.pdf` | background |

---

*Two threads, one aim: close the gap between explanations that **sound** right
and explanations that **are** right — and make the NLA tell you which is which.*
