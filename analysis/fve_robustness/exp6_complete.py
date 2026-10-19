"""Complete Exp. 6 arms P and C_cont, then materialize GPU-ready records.

P (artifact ``exp6p``) replaces every LCR quote with a length-targeted paraphrase of the A1 true
context slice. GPT-5.6 Luna generates each paraphrase and performs a second, conservative semantic
audit; only audited replacements are cached. Exact word matching proved incompatible with the many
truncated fragments, so accepted lengths use a logged 40–160% envelope and semantic failures fail
closed to the unmodified true slice.

C_cont (artifact ``exp6t``) freezes LCR/ECHO spans and replaces PC slots with consecutive, non-
overlapping word tiles from the true post-token continuation.  GPT-5.6 Luna labels whether any AV PC
candidate was semantically correct, permitting the preregistered wrong-continuation subset analysis.

The OpenAI ledger counts the prior Exp. 7 spend and enforces a combined internal stop at $90, leaving
$10 below the user's $100 ceiling.  Only ``OPENAI_API_KEY_2`` is read.

Usage:
  python3 exp6_complete.py [qwen7b gemma12b gemma27b] [--workers 12] [--batch-size 24]
  python3 exp6_complete.py --no-api                 # materialize from a complete audited cache
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
import math
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path

import httpx

from edits import _norm_match
from exp6_edits import continuation_case, paraphrase_lcr, substitute_true_continuation

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
RES = HERE / "results"
@lru_cache(maxsize=1)
def _source_documents():
    """Load the external corpus only when an experiment needs document text."""
    path = HERE / "data" / "ufw_docs.json"
    if not path.is_file():
        raise FileNotFoundError(f"Required experiment corpus is not bundled: {path}")
    return {d["id"]: d["text"] for d in json.loads(path.read_text())}
KEYS = ("qwen7b", "gemma12b", "gemma27b")

MODEL = "gpt-5.6-luna"
PROVIDER = "openai"
POLICY_VERSION = "exp6p-ccont-v1"
INPUT_USD_PER_MTOK = 0.20
OUTPUT_USD_PER_MTOK = 1.20
COMBINED_INTERNAL_CAP_USD = 90.0
USER_CEILING_USD = 100.0
CONSERVATIVE_INTERRUPTED_PROCESS_RESERVE_USD = 10.0
PRIOR_LEDGER = RES / "exp7_openai_budget.json"
BUDGET_PATH = RES / "exp6_openai_budget.json"
PARAPHRASE_CACHE = RES / "exp6p_openai_cache.jsonl"
CONT_CACHE = RES / "exp6t_openai_cache.jsonl"
PRICING_SOURCE = "https://developers.openai.com/api/docs/models/gpt-5.6-luna"
QUOTE_RE = re.compile(r'["“”„‟«»‹›`]')

PARAPHRASE_SYSTEM = (
    "Paraphrase each SOURCE conservatively within its supplied inclusive MIN_WORDS..MAX_WORDS band, "
    "aiming near TARGET_WORDS. Preserve every entity, relation, number, polarity, modality, uncertainty "
    "marker, and incomplete-fragment ending. Do not correct or complete a fragment. Never add a subject, "
    "object, referent, explanation, placeholder, dummy token, or filler to reach a length. Change wording "
    "or syntax substantially for sources of four or more words; names, identifiers, numbers, and short "
    "irreducible fragments may remain unchanged. Use no quotation marks, guillemets, or backticks. Return "
    "each p0..pN as one plain string."
)

AUDIT_SYSTEM = (
    "Conservatively audit whether each REPLACEMENT states exactly the same content as SOURCE. Entity "
    "identity, number, relation, polarity, modality, uncertainty, and whether a fragment is incomplete "
    "must be preserved. Changed word order and true synonyms pass. Added facts, invented completions, "
    "broader/narrower claims, or changed referents fail. Return a preservation verdict and short reason "
    "for every a0..aN field exactly once."
)

CONT_SYSTEM = (
    "Judge predicted continuations against the TRUE immediate document continuation. Set wrong=false "
    "if at least one candidate conveys the same immediate continuation by meaning; accept genuine "
    "paraphrases, aliases, and specific/general wording only when the true text supports the identity "
    "(for example Roomba and robotic vacuum). Set wrong=true when no candidate matches. Topic overlap "
    "alone is not a match. best_candidate is the zero-based matching candidate index, or -1 when none. "
    "Give a short reason and return every j0..jN field exactly once."
)


def _env_value(name: str) -> str:
    if os.environ.get(name):
        return os.environ[name]
    path = REPO / ".env"
    if path.exists():
        for raw in path.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:].lstrip()
            key, sep, value = line.partition("=")
            if sep and key.strip() == name:
                value = value.strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
                    value = value[1:-1]
                return value
    raise RuntimeError(f"{name} is not set in the environment or {path}")


def _prior_cost() -> float:
    if not PRIOR_LEDGER.exists():
        return 0.0
    return float(json.loads(PRIOR_LEDGER.read_text()).get("estimated_cost_usd", 0.0))


class BudgetLedger:
    """Thread-safe combined-task budget with conservative pre-request reservations."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.reserved_usd = 0.0
        if BUDGET_PATH.exists():
            self.data = json.loads(BUDGET_PATH.read_text())
            if self.data.get("model") != MODEL or self.data.get("policy_version") != POLICY_VERSION:
                raise RuntimeError("existing Exp. 6 budget ledger belongs to a different policy")
        else:
            self.data = dict(
                provider=PROVIDER, model=MODEL, policy_version=POLICY_VERSION,
                combined_internal_cap_usd=COMBINED_INTERNAL_CAP_USD,
                user_ceiling_usd=USER_CEILING_USD,
                input_usd_per_mtok=INPUT_USD_PER_MTOK,
                output_usd_per_mtok=OUTPUT_USD_PER_MTOK,
                pricing_source=PRICING_SOURCE, prior_ledger=str(PRIOR_LEDGER),
                input_tokens=0, output_tokens=0, estimated_cost_usd=0.0, responses=0,
            )
        # Early interrupted thread-pool runs briefly overlapped and could overwrite the file-based
        # counters. Keep a deliberately loose upper-bound reserve in every subsequent budget decision.
        self.data["conservative_interrupted_process_reserve_usd"] = max(
            float(self.data.get("conservative_interrupted_process_reserve_usd", 0.0)),
            CONSERVATIVE_INTERRUPTED_PROCESS_RESERVE_USD,
        )

    def _write(self) -> None:
        self.data["prior_cost_usd"] = round(_prior_cost(), 8)
        self.data["combined_estimated_cost_usd"] = round(
            self.data["prior_cost_usd"] + self.data["estimated_cost_usd"]
            + self.data["conservative_interrupted_process_reserve_usd"], 8)
        self.data["updated_at"] = datetime.now(timezone.utc).isoformat()
        tmp = BUDGET_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.data, indent=2))
        tmp.replace(BUDGET_PATH)

    def reserve(self, instructions: str, prompt: str, max_output_tokens: int) -> float:
        # English UTF-8 bytes / 2 overestimates normal input tokenization; reserve all possible output.
        input_upper = math.ceil(len((instructions + prompt).encode("utf-8")) / 2)
        amount = (input_upper * INPUT_USD_PER_MTOK
                  + max_output_tokens * OUTPUT_USD_PER_MTOK) / 1_000_000
        with self.lock:
            projected = (_prior_cost() + self.data["estimated_cost_usd"]
                         + self.data["conservative_interrupted_process_reserve_usd"]
                         + self.reserved_usd + amount)
            if projected > COMBINED_INTERNAL_CAP_USD:
                raise RuntimeError(
                    f"Exp. 6 OpenAI budget stop: combined projected ${projected:.4f} > "
                    f"${COMBINED_INTERNAL_CAP_USD:.2f}")
            self.reserved_usd += amount
        return amount

    def settle(self, reservation: float, usage: dict | None) -> None:
        usage = usage or {}
        inp = int(usage.get("input_tokens", 0))
        out = int(usage.get("output_tokens", 0))
        actual = (inp * INPUT_USD_PER_MTOK + out * OUTPUT_USD_PER_MTOK) / 1_000_000
        with self.lock:
            self.reserved_usd -= reservation
            self.data["input_tokens"] += inp
            self.data["output_tokens"] += out
            self.data["estimated_cost_usd"] = round(
                self.data["estimated_cost_usd"] + actual, 8)
            self.data["responses"] += 1
            if (_prior_cost() + self.data["estimated_cost_usd"]
                    + self.data["conservative_interrupted_process_reserve_usd"]
                    > COMBINED_INTERNAL_CAP_USD):
                raise RuntimeError("combined OpenAI usage exceeded the internal budget cap")
            self._write()

    def release(self, reservation: float) -> None:
        with self.lock:
            self.reserved_usd -= reservation

    def snapshot(self) -> dict:
        with self.lock:
            out = dict(self.data)
            out["prior_cost_usd"] = round(_prior_cost(), 8)
            out["combined_estimated_cost_usd"] = round(
                out["prior_cost_usd"] + out["estimated_cost_usd"]
                + out["conservative_interrupted_process_reserve_usd"], 8)
            return out


def _output_text(body: dict) -> str:
    texts = []
    for item in body.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                texts.append(content.get("text", ""))
            elif content.get("type") == "refusal":
                raise RuntimeError(f"OpenAI refusal: {content.get('refusal', '')}")
    if not texts:
        raise ValueError(f"response contained no output_text (status={body.get('status')})")
    return "".join(texts)


def _schema(name: str, properties: dict) -> dict:
    return {
        "type": "json_schema", "name": name, "strict": True,
        "schema": {"type": "object", "properties": properties,
                   "required": list(properties), "additionalProperties": False},
    }


def paraphrase_format(items) -> dict:
    n = items if isinstance(items, int) else len(items)
    return _schema("exp6_paraphrases", {f"p{i}": {"type": "string"} for i in range(n)})


def audit_format(n: int) -> dict:
    item = {
        "type": "object",
        "properties": {
            "preserved": {"type": "boolean"},
            "reason": {"type": "string"},
        },
        "required": ["preserved", "reason"],
        "additionalProperties": False,
    }
    return _schema("exp6_paraphrase_audit", {f"a{i}": item for i in range(n)})


def continuation_format(n: int) -> dict:
    item = {
        "type": "object",
        "properties": {
            "wrong": {"type": "boolean"},
            "best_candidate": {"type": "integer", "minimum": -1},
            "reason": {"type": "string"},
        },
        "required": ["wrong", "best_candidate", "reason"],
        "additionalProperties": False,
    }
    return _schema("exp6_continuation_judgments", {f"j{i}": item for i in range(n)})


def surface_change(source: str, replacement: str) -> float:
    return 1.0 - SequenceMatcher(None, _norm_match(source), _norm_match(replacement)).ratio()


def word_band(n_words: int) -> tuple[int, int]:
    if n_words <= 3:
        return 1, n_words + 2
    return max(1, math.floor(0.4 * n_words)), max(2, math.ceil(1.6 * n_words))


def validate_paraphrase(source: str, replacement: str, n_words: int,
                        exact: bool = True, allow_unchanged: bool = False,
                        allow_source_quotes: bool = False) -> None:
    if not replacement.strip():
        raise ValueError("empty paraphrase")
    if QUOTE_RE.search(replacement) and not (
            allow_source_quotes and replacement == source):
        raise ValueError("paraphrase contains a quotation character")
    actual = len(replacement.split())
    lo, hi = (n_words, n_words) if exact else word_band(n_words)
    if not lo <= actual <= hi:
        raise ValueError(f"paraphrase has {actual} words; expected {lo}..{hi}")
    if (not allow_unchanged and n_words >= 4
            and _norm_match(source) == _norm_match(replacement)):
        raise ValueError("multiword paraphrase is unchanged")


def paraphrase_key(source: str, n_words: int) -> str:
    raw = json.dumps({"v": POLICY_VERSION, "source": source, "n": n_words},
                     ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()


def continuation_key(case: dict) -> str:
    raw = json.dumps({"v": POLICY_VERSION, "true": case["true_continuation"],
                      "candidates": case["candidates"]}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()


def load_jsonl(path: Path, key_field: str) -> dict:
    out = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("policy_version") == POLICY_VERSION:
                out[row[key_field]] = row
    return out


class API:
    def __init__(self, ledger: BudgetLedger) -> None:
        self.ledger = ledger
        self.client = httpx.Client(
            base_url="https://api.openai.com/v1",
            headers={"Authorization": f"Bearer {_env_value('OPENAI_API_KEY_2')}",
                     "Content-Type": "application/json"},
            timeout=httpx.Timeout(120.0),
        )

    def request(self, instructions: str, prompt: str, fmt: dict,
                max_output_tokens: int) -> dict:
        last_error = None
        for attempt in range(4):
            reservation = 0.0
            try:
                reservation = self.ledger.reserve(instructions, prompt, max_output_tokens)
                response = self.client.post("/responses", json={
                    "model": MODEL, "instructions": instructions, "input": prompt,
                    "reasoning": {"effort": "none"},
                    "text": {"format": fmt, "verbosity": "low"},
                    "max_output_tokens": max_output_tokens, "store": False,
                })
                if response.status_code >= 400:
                    self.ledger.release(reservation)
                    reservation = 0.0
                    response.raise_for_status()
                body = response.json()
                settled = reservation
                reservation = 0.0
                self.ledger.settle(settled, body.get("usage"))
                return json.loads(_output_text(body))
            except Exception as exc:  # noqa: BLE001
                if reservation:
                    self.ledger.release(reservation)
                last_error = exc
                if "budget stop" in str(exc).lower() or "budget cap" in str(exc).lower():
                    raise
                if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code in (400, 401, 403):
                    break
                if attempt < 3:
                    time.sleep(2 ** attempt)
        raise RuntimeError(f"OpenAI request failed after four attempts: {last_error}")


def _paraphrase_batch(api: API, items: list[dict]) -> list[dict]:
    accepted: dict[int, dict] = {}
    pending = list(enumerate(items))
    errors: dict[int, str] = {}
    for _attempt in range(3):
        current = [item for _idx, item in pending]
        prompt = "ITEMS:\n" + json.dumps([
            {"id": i, "source": item["source"], "target_words": item["n_words"],
             "min_words": word_band(item["n_words"])[0],
             "max_words": word_band(item["n_words"])[1],
             **({"audit_feedback": item["feedback"]} if item.get("feedback") else {}),
             **({"mechanical_feedback": errors[orig_idx]} if orig_idx in errors else {})}
            for i, (orig_idx, item) in enumerate(pending)
        ], ensure_ascii=False)
        max_out = min(8192, max(256, 32 + 8 * sum(item["n_words"] for item in current)))
        parsed = api.request(PARAPHRASE_SYSTEM, prompt, paraphrase_format(current), max_out)
        retry = []
        for i, (orig_idx, item) in enumerate(pending):
            try:
                replacement = str(parsed[f"p{i}"]).strip()
                validate_paraphrase(item["source"], replacement, item["n_words"], exact=False,
                                    allow_unchanged=True)
                accepted[orig_idx] = dict(item, replacement=replacement)
            except Exception as exc:  # noqa: BLE001
                errors[orig_idx] = str(exc)
                retry.append((orig_idx, item))
        pending = retry
        if not pending:
            return [accepted[i] for i in range(len(items))]
    # Mechanical fidelity also fails closed: do not spend recursively on pathological fragments.
    for orig_idx, item in pending:
        accepted[orig_idx] = dict(item, replacement=item["source"], mechanical_fallback=True)
    return [accepted[i] for i in range(len(items))]


def _audit_batch(api: API, generated: list[dict]) -> list[dict]:
    active = [dict(item) for item in generated]
    reasons = [""] * len(active)
    rounds = [0] * len(active)
    pending = list(range(len(active)))
    for round_no in range(1, 4):
        prompt = "PAIRS:\n" + json.dumps([
            {"id": i, "source": active[idx]["source"],
             "replacement": active[idx]["replacement"],
             "word_count": active[idx]["n_words"]}
            for i, idx in enumerate(pending)
        ], ensure_ascii=False)
        max_out = min(8192, max(384, 80 * len(pending)))
        parsed = api.request(AUDIT_SYSTEM, prompt, audit_format(len(pending)), max_out)
        failed = []
        for i, idx in enumerate(pending):
            verdict = parsed[f"a{i}"]
            rounds[idx] = round_no
            reasons[idx] = str(verdict["reason"])
            if not bool(verdict["preserved"]):
                failed.append(idx)
        if not failed:
            return [dict(item, audit_reason=reasons[i], audit_rounds=rounds[i])
                    for i, item in enumerate(active)]
        if round_no == 3:
            # Semantic fidelity dominates surface change. A source-identical fallback is a transparent
            # no-op, whereas retaining a failed paraphrase would contaminate the content-vs-form arm.
            for idx in failed:
                active[idx]["replacement"] = active[idx]["source"]
                active[idx]["audit_fallback"] = True
                reasons[idx] = "fail-closed to source after three rejected paraphrases: " + reasons[idx]
            return [dict(item, audit_reason=reasons[i], audit_rounds=rounds[i])
                    for i, item in enumerate(active)]
        regen_items = [dict(active[idx], feedback=reasons[idx]) for idx in failed]
        regenerated = _paraphrase_batch(api, regen_items)
        for idx, fresh in zip(failed, regenerated):
            active[idx]["replacement"] = fresh["replacement"]
        pending = failed
    raise AssertionError("unreachable")


def _continuation_batch(api: API, items: list[dict]) -> list[dict]:
    base_prompt = "CASES:\n" + json.dumps([
        {"id": i, "true_continuation": item["case"]["true_continuation"],
         "candidates": item["case"]["candidates"]}
        for i, item in enumerate(items)
    ], ensure_ascii=False)
    max_out = min(8192, max(384, 80 * len(items)))
    last_error = None
    for _attempt in range(3):
        prompt = base_prompt + (("\n\nPrevious output failed validation: " + str(last_error))
                                if last_error is not None else "")
        try:
            parsed = api.request(CONT_SYSTEM, prompt, continuation_format(len(items)), max_out)
            out = []
            for i, item in enumerate(items):
                verdict = parsed[f"j{i}"]
                wrong, best = bool(verdict["wrong"]), int(verdict["best_candidate"])
                n = len(item["case"]["candidates"])
                if wrong and best != -1:
                    raise ValueError("wrong=true requires best_candidate=-1")
                if not wrong and not 0 <= best < n:
                    raise ValueError(f"best_candidate {best} outside 0..{n - 1}")
                out.append(dict(item, wrong=wrong, best_candidate=best,
                                reason=str(verdict["reason"])))
            return out
        except Exception as exc:  # noqa: BLE001
            if "budget stop" in str(exc).lower() or "budget cap" in str(exc).lower():
                raise
            last_error = exc
    if len(items) == 1:
        raise RuntimeError(f"singleton continuation judgment failed: {last_error}")
    mid = len(items) // 2
    return (_continuation_batch(api, items[:mid])
            + _continuation_batch(api, items[mid:]))


def _run_batches(items: list[dict], batch_size: int, workers: int, fn,
                 cache_path: Path, cache_key_field: str, label: str) -> dict:
    batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
    lock = threading.Lock()
    done = 0
    out = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fn, batch): batch for batch in batches}
        for future in as_completed(futures):
            rows = future.result()
            lines = []
            for row in rows:
                cached = dict(row, policy_version=POLICY_VERSION, model=MODEL)
                cached.pop("case", None)
                out[cached[cache_key_field]] = cached
                lines.append(json.dumps(cached, ensure_ascii=False))
            with lock:
                with cache_path.open("a", encoding="utf-8") as handle:
                    handle.write("\n".join(lines) + "\n")
                done += len(rows)
                if done % 240 < len(rows) or done == len(items):
                    print(f"[{label}] {done}/{len(items)}", flush=True)
    return out


def collect(keys: tuple[str, ...]) -> tuple[dict, dict]:
    para, cont = {}, {}
    for key in keys:
        base = json.loads((RES / f"records_exp6a_{key}.json").read_text())
        for rec in base:
            for quote in rec.get("edit_meta", {}).get("quotes", []):
                source = quote.get("injected", "")
                n_words = int(quote.get("n_words", len(source.split())))
                if source:
                    pk = paraphrase_key(source, n_words)
                    para.setdefault(pk, dict(cache_key=pk, source=source, n_words=n_words))
            if (rec.get("in_range") and rec.get("expl_orig") and rec.get("char_end") is not None
                    and _source_documents().get(rec["doc_id"])):
                case = continuation_case(rec["expl_orig"], rec.get("token") or "",
                                         _source_documents()[rec["doc_id"]], rec["char_end"],
                                         rec.get("char_start"))
                if case["true_continuation"] and case["candidates"]:
                    ck = continuation_key(case)
                    cont.setdefault(ck, dict(case_key=ck, case=case))
    return para, cont


def materialize(keys: tuple[str, ...], para_cache: dict, cont_cache: dict,
                budget_snapshot: dict) -> None:
    replacement_map = {row["source"]: row["replacement"] for row in para_cache.values()}
    for key in keys:
        base = json.loads((RES / f"records_exp6a_{key}.json").read_text())
        p_records, p_reads, t_records, t_reads = [], {}, [], {}
        p_quotes = p_unchanged = p_fallback = p_mechanical_fallback = 0
        t_pc = t_wrong = t_right = t_unlabelled = t_insufficient = 0
        surface_changes, word_deltas = [], []
        for source_rec in base:
            doc = _source_documents().get(source_rec["doc_id"], "")
            expl = source_rec.get("expl_orig", "")
            ce, cs = source_rec.get("char_end"), source_rec.get("char_start")
            valid = bool(source_rec.get("in_range") and expl and ce is not None and doc)

            prec = dict(source_rec)
            for field in ("cos_orig", "cos_edit", "D_i", "fve_i_orig", "fve_i_edit"):
                prec[field] = None
            if valid:
                ped = paraphrase_lcr(expl, source_rec.get("token") or "", doc, ce,
                                     replacement_map, cs)
                for quote in ped["edit_meta"]["quotes"]:
                    cached = para_cache[paraphrase_key(quote["source_true"], quote["n_words"])]
                    quote["audit_reason"] = cached["audit_reason"]
                    quote["audit_rounds"] = cached["audit_rounds"]
                    quote["audit_fallback"] = bool(cached.get("audit_fallback"))
                    quote["mechanical_fallback"] = bool(cached.get("mechanical_fallback"))
                    p_fallback += int(quote["audit_fallback"])
                    p_mechanical_fallback += int(quote["mechanical_fallback"])
                    change = surface_change(quote["source_true"], quote["injected"])
                    quote["surface_change"] = round(change, 4)
                    quote["delta_words"] = len(quote["injected"].split()) - quote["n_words"]
                    quote["length_band"] = list(word_band(quote["n_words"]))
                    word_deltas.append(quote["delta_words"])
                    surface_changes.append(change)
                    p_unchanged += int(_norm_match(quote["source_true"])
                                       == _norm_match(quote["injected"]))
                ped["edit_meta"].update(model=MODEL, policy_version=POLICY_VERSION,
                                        n_semantic_audited=len(ped["edit_meta"]["quotes"]),
                                        length_policy="target_words_40_to_160pct_logged_v1")
                prec.update(expl_edit=ped["expl_edit"], removed=ped["removed"],
                            added=ped["added"], edit_meta=ped["edit_meta"])
                p_reads[str(prec["idx"])] = ped["expl_edit"]
                p_quotes += ped["edit_meta"]["n_lcr"]
            else:
                prec.update(expl_edit=expl, edit_meta=dict(
                    arm="exp6p", n_removed=0, chars_removed=0, n_lcr=0, n_pc=0, quotes=[],
                    model=MODEL, policy_version=POLICY_VERSION, n_semantic_audited=0))
                p_reads[str(prec["idx"])] = expl
            p_records.append(prec)

            trec = dict(source_rec)
            for field in ("cos_orig", "cos_edit", "D_i", "fve_i_orig", "fve_i_edit"):
                trec[field] = None
            if valid:
                case = continuation_case(expl, source_rec.get("token") or "", doc, ce, cs)
                cached = cont_cache.get(continuation_key(case)) if case["candidates"] else None
                if cached:
                    verdict = dict(cached, true_continuation=case["true_continuation"])
                    t_wrong += int(cached["wrong"])
                    t_right += int(not cached["wrong"])
                else:
                    verdict = dict(wrong=None, best_candidate=None, reason="no PC candidates",
                                   true_continuation=case["true_continuation"])
                    t_unlabelled += 1
                ted = substitute_true_continuation(
                    expl, source_rec.get("token") or "", doc, ce, verdict, cs)
                ted["edit_meta"].update(model=MODEL, policy_version=POLICY_VERSION,
                                        lexical_scores=case["lexical_scores"])
                trec.update(expl_edit=ted["expl_edit"], removed=ted["removed"],
                            added=ted["added"], edit_meta=ted["edit_meta"])
                t_reads[str(trec["idx"])] = ted["expl_edit"]
                t_pc += ted["edit_meta"]["n_pc"]
                t_insufficient += ted["edit_meta"]["n_insufficient_tiles"]
            else:
                trec.update(expl_edit=expl, edit_meta=dict(
                    arm="exp6t", n_removed=0, chars_removed=0, n_lcr=0, n_pc=0, quotes=[],
                    continuation_wrong=None, n_insufficient_tiles=0,
                    model=MODEL, policy_version=POLICY_VERSION))
                t_reads[str(trec["idx"])] = expl
                t_unlabelled += 1
            t_records.append(trec)

        for arm, records, reads in (("exp6p", p_records, p_reads), ("exp6t", t_records, t_reads)):
            (RES / f"records_{arm}_{key}.json").write_text(
                json.dumps(records, ensure_ascii=False))
            (RES / f"edited_reads_{arm}_{key}.json").write_text(
                json.dumps({"reads": reads, "arm": arm}, ensure_ascii=False))
        (RES / f"exp6p_summary_{key}.json").write_text(json.dumps(dict(
            key=key, arm="exp6p", model=MODEL, policy_version=POLICY_VERSION,
            n_records=len(p_records), n_lcr_paraphrases=p_quotes,
            n_semantic_audited=p_quotes, n_unchanged_paraphrases=p_unchanged,
            n_fail_closed_to_source=p_fallback,
            n_mechanical_fail_closed_to_source=p_mechanical_fallback,
            mean_surface_change=(round(sum(surface_changes) / len(surface_changes), 4)
                                 if surface_changes else None),
            mean_delta_words=(round(sum(word_deltas) / len(word_deltas), 4)
                              if word_deltas else None),
            mean_abs_delta_words=(round(sum(abs(value) for value in word_deltas) / len(word_deltas), 4)
                                  if word_deltas else None),
            max_abs_delta_words=max((abs(value) for value in word_deltas), default=None),
            length_policy="target_words_40_to_160pct_logged_v1",
            budget=budget_snapshot,
        ), indent=2))
        (RES / f"exp6t_summary_{key}.json").write_text(json.dumps(dict(
            key=key, arm="exp6t", model=MODEL, policy_version=POLICY_VERSION,
            n_records=len(t_records), n_pc_replaced=t_pc, n_wrong=t_wrong,
            n_at_least_one_correct=t_right, n_unlabelled=t_unlabelled,
            n_insufficient_tiles=t_insufficient, budget=budget_snapshot,
        ), indent=2))
        print(f"[materialize:{key}] P={p_quotes} audited LCRs; "
              f"C_cont={t_pc} PC slots, wrong/right/unlabelled={t_wrong}/{t_right}/{t_unlabelled}",
              flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("keys", nargs="*")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--no-api", action="store_true")
    args = ap.parse_args()
    invalid = sorted(set(args.keys) - set(KEYS))
    if invalid:
        ap.error(f"unknown model key(s): {', '.join(invalid)}")
    keys = tuple(args.keys or KEYS)
    RES.mkdir(exist_ok=True)

    para_items, cont_items = collect(keys)
    para_cache = load_jsonl(PARAPHRASE_CACHE, "cache_key")
    cont_cache = load_jsonl(CONT_CACHE, "case_key")
    # Reject mechanically stale/corrupt paraphrase cache entries.
    para_cache = {k: row for k, row in para_cache.items()
                  if k in para_items and _cached_paraphrase_valid(para_items[k], row)}
    cont_cache = {k: row for k, row in cont_cache.items() if k in cont_items}
    missing_para = [item for k, item in para_items.items() if k not in para_cache]
    missing_cont = [item for k, item in cont_items.items() if k not in cont_cache]
    print(f"[exp6] unique paraphrases {len(para_items)} ({len(missing_para)} missing); "
          f"continuation cases {len(cont_items)} ({len(missing_cont)} missing)", flush=True)

    ledger = BudgetLedger()
    if args.no_api and (missing_para or missing_cont):
        raise RuntimeError("--no-api requested but the audited cache is incomplete")
    if missing_para or missing_cont:
        api = API(ledger)
        if missing_para:
            generated = _run_batches(
                missing_para, args.batch_size, args.workers,
                lambda batch: _audit_batch(api, _paraphrase_batch(api, batch)),
                PARAPHRASE_CACHE, "cache_key", "exp6p")
            para_cache.update(generated)
        if missing_cont:
            judged = _run_batches(
                missing_cont, args.batch_size, args.workers,
                lambda batch: _continuation_batch(api, batch),
                CONT_CACHE, "case_key", "exp6t")
            cont_cache.update(judged)

    if set(para_items) - set(para_cache) or set(cont_items) - set(cont_cache):
        raise RuntimeError("cache completeness invariant failed")
    materialize(keys, para_cache, cont_cache, ledger.snapshot())
    print("[budget] " + json.dumps(ledger.snapshot(), sort_keys=True), flush=True)
    return 0


def _cached_paraphrase_valid(item: dict, row: dict) -> bool:
    try:
        if row.get("source") != item["source"] or int(row.get("n_words", -1)) != item["n_words"]:
            return False
        validate_paraphrase(
            item["source"], row.get("replacement", ""), item["n_words"], exact=False,
            allow_unchanged=True,
            allow_source_quotes=(row.get("replacement") == item["source"]),
        )
        return bool(row.get("audit_reason")) and int(row.get("audit_rounds", 0)) >= 1
    except (TypeError, ValueError):
        return False


if __name__ == "__main__":
    raise SystemExit(main())
