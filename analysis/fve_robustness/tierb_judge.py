"""Exp 3b (Tier B) — semantic input-derivability judge (Claude Haiku), full run over all in-range
datapoints of each model.

Runs LOCALLY (keys in repo .env, Mac has direct internet). For each datapoint the judge labels every
quoted span derivable (content stated in / a faithful paraphrase of the input, or a correct
multi-token continuation) vs divergent, with a basis + short reason + confidence, under a hardened
prompt that rejects single/function/sub-word matches and phrase-graft confabulations. It then removes
the derivable spans (same edit mechanic as Exp 3a), reusing the Exp 3a records for display fields, and
writes per model:
  results/records_exp3b_<key>.json        rich per-datapoint records (cos/FVE null until GPU re-score)
  results/edited_reads_exp3b_<key>.json   {"reads": {idx: edited}, "arm": "exp3b"}  (for rescore.py)
  results/tierb_summary_<key>.json        derivable rate + A/B agreement counts

Usage:  python tierb_judge.py                       # all three models
        python tierb_judge.py qwen7b                # one model
"""
from __future__ import annotations

import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from edits import _cleanup  # consistent hole-cleanup after span removal

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
RES = HERE / "results"
load_dotenv(REPO / ".env")

KEYS = sys.argv[1:] or ["qwen7b", "gemma12b", "gemma27b"]
MODEL = "claude-haiku-4-5-20251001"
WORKERS = 12

client = anthropic.Anthropic()   # reads ANTHROPIC_API_KEY from env

TOOL = [{
    "name": "record_verdicts",
    "description": "Return exactly one verdict per quoted span id.",
    "input_schema": {
        "type": "object",
        "properties": {
            "verdicts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "label": {"type": "string", "enum": ["derivable", "divergent"]},
                        "basis": {"type": "string",
                                  "enum": ["verbatim", "paraphrase", "continuation", "none"]},
                        "reason": {"type": "string"},
                        "confidence": {"type": "number"},
                    },
                    "required": ["id", "label", "basis", "reason", "confidence"],
                },
            }
        },
        "required": ["verdicts"],
    },
}]

SYSTEM = (
    "You audit an activation-verbalizer's explanation of ONE token in a document. For each QUOTED "
    "SPAN decide whether the span's PROPOSITION -- the specific claim, entity, or multi-word phrase "
    "it asserts -- is actually stated in, or a faithful paraphrase of, the INPUT DOCUMENT.\n\n"
    "Label 'derivable' ONLY when the span restates specific content the input actually asserts:\n"
    "  - verbatim: a multi-word phrase or claim copied from the input;\n"
    "  - paraphrase: a synonym/reworded restatement of a SPECIFIC input claim "
    "(e.g. 'a robotic vacuum' for the input's 'Roomba');\n"
    "  - continuation: a CORRECT, MULTI-TOKEN continuation of the read token that matches the "
    "input's actual following text.\n\n"
    "Label 'divergent' in every other case. In particular, mark DIVERGENT:\n"
    "  - single words, function words, or short fragments (e.g. 'by', 'of', 'the area of') -- a lone "
    "common word is never a restatement, even if it appears in the document;\n"
    "  - generic domain vocabulary that merely occurs in the document (e.g. 'injection' in an "
    "injection-well document);\n"
    "  - a word matched only INSIDE a larger word (e.g. 'May' within 'Mayan') -- that is NOT a match;\n"
    "  - a single-token continuation guess -- reserve 'continuation' for multi-token correct predictions;\n"
    "  - PHRASE-GRAFT CONFABULATIONS: the span borrows input phrasing but attaches it to an entity or "
    "claim NOT in the input. Example: the input says the ROSE family is a top crop family; the span "
    "says 'the family Solanaceae is one of the most important crop plant families' -> DIVERGENT, "
    "because the asserted claim about Solanaceae is not in the input. Judge the whole proposition, "
    "not incidental word overlap.\n\n"
    "Be strict: 'restates a specific thing the input says', not 'is about the same subject' or "
    "'shares some words'. basis = verbatim | paraphrase | continuation | none."
)

_prog = {"done": 0, "total": 0}
_lock = threading.Lock()


def build_user(dp: dict) -> str:
    qlines = "\n".join(f"[{q['id']}] {q['text']}" for q in dp["quotes"])
    return (f"INPUT DOCUMENT:\n{dp['doc']}\n\n"
            f"The read token (whose activation was explained) is \"{dp['token']}\" at position "
            f"{dp['position']}.\n\nQUOTED SPANS (one verdict each):\n{qlines}\n\n"
            "Call record_verdicts with one verdict per span id above.")


def judge(dp: dict) -> dict:
    vm = {}
    for attempt in range(4):
        try:
            msg = client.messages.create(
                model=MODEL, max_tokens=1500, temperature=0, system=SYSTEM,
                tools=TOOL, tool_choice={"type": "tool", "name": "record_verdicts"},
                messages=[{"role": "user", "content": build_user(dp)}])
            for block in msg.content:
                if block.type == "tool_use":
                    vm = {v["id"]: v for v in block.input.get("verdicts", [])
                          if isinstance(v, dict) and "id" in v}
                    break
            break
        except Exception as e:  # noqa: BLE001
            if attempt == 3:
                print(f"  [idx {dp['idx']}] judge failed: {e}", flush=True)
            else:
                time.sleep(2 ** attempt)
    with _lock:
        _prog["done"] += 1
        if _prog["done"] % 200 == 0 or _prog["done"] == _prog["total"]:
            print(f"  judged {_prog['done']}/{_prog['total']}", flush=True)
    return vm


def remove_spans(text: str, spans: list) -> str:
    parts, prev = [], 0
    for s, e in sorted(spans):
        parts.append(text[prev:s])
        prev = e
    parts.append(text[prev:])
    return _cleanup("".join(parts))


_cache_lock = threading.Lock()


def _cache_path(key: str):
    return RES / f"tierb_cache_{key}.jsonl"


def load_cache(key: str) -> dict:
    """Resume support: {idx: {id: verdict}} from a JSONL cache written as datapoints complete."""
    p, out = _cache_path(key), {}
    if p.exists():
        for line in p.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                out[d["idx"]] = {int(k): v for k, v in d["verdicts"].items()}
    return out


def judge_and_cache(dp: dict, key: str) -> dict:
    vm = judge(dp)
    with _cache_lock, open(_cache_path(key), "a") as f:
        f.write(json.dumps({"idx": dp["idx"], "verdicts": vm}) + "\n")
    return vm


def run_key(key: str) -> None:
    dps = json.loads((HERE / f"tierb_input_{key}.json").read_text())
    base = {r["idx"]: r for r in json.loads((RES / f"records_exp3a_{key}.json").read_text())}
    cache = load_cache(key)
    todo = [dp for dp in dps if dp["idx"] not in cache]
    _prog["done"], _prog["total"] = len(dps) - len(todo), len(dps)
    print(f"[exp3b:{key}] judging {len(todo)} datapoints ({len(cache)} cached) ...", flush=True)

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for dp, vm in zip(todo, ex.map(lambda d: judge_and_cache(d, key), todo)):
            cache[dp["idx"]] = vm

    out_recs, reads = [], {}
    viol = bonly = n_der = n_q = 0
    for dp in dps:
        vm = cache.get(dp["idx"], {})
        removed, quotes = [], []
        for q in dp["quotes"]:
            v = vm.get(q["id"]) or {}
            label = v.get("label", "divergent")
            der = label == "derivable"
            quotes.append(dict(span=q["span"], label=label, basis=v.get("basis", "none"),
                               reason=v.get("reason", "no verdict"),
                               confidence=v.get("confidence", 0.0), derivable=der))
            if der:
                removed.append(q["span"])
            n_q += 1
            if q["tierA_derivable"] and not der:
                viol += 1
            if der and not q["tierA_derivable"]:
                bonly += 1
        n_der += len(removed)
        r = dict(base[dp["idx"]])
        edited = remove_spans(dp["expl"], removed)
        r["expl_edit"] = edited
        r["removed"] = [[int(s), int(e)] for s, e in removed]
        r.pop("kept", None)
        r["cos_orig"] = r["cos_edit"] = r["fve_i_orig"] = r["fve_i_edit"] = None
        r["edit_meta"] = dict(n_removed=len(removed),
                              chars_removed=sum(e - s for s, e in removed),
                              n_quotes=len(dp["quotes"]), n_derivable=len(removed),
                              n_divergent=len(dp["quotes"]) - len(removed),
                              model=MODEL, quotes=quotes)
        out_recs.append(r)
        reads[str(dp["idx"])] = edited

    (RES / f"records_exp3b_{key}.json").write_text(json.dumps(out_recs, ensure_ascii=False))
    (RES / f"edited_reads_exp3b_{key}.json").write_text(
        json.dumps({"reads": reads, "arm": "exp3b"}, ensure_ascii=False))
    summary = dict(key=key, model=MODEL, n_datapoints=len(out_recs), n_quotes=n_q,
                   avg_derivable_per_expl=round(n_der / len(out_recs), 3),
                   tierA_subset_violations=viol, tierB_only_catches=bonly)
    (RES / f"tierb_summary_{key}.json").write_text(json.dumps(summary, indent=2))
    print(f"[exp3b:{key}] DONE " + json.dumps(summary), flush=True)


def main() -> None:
    for key in KEYS:
        run_key(key)


if __name__ == "__main__":
    main()
