"""Experiment 4b -- strictly paraphrase every quoted AV span (API, local).

For each explanation, Claude Haiku rewrites each detected double-quoted span as fluent unquoted prose.
Only the quote spans are replaced: every byte outside them is preserved. Successful calls are cached in
``results/exp4b_cache_<key>.jsonl``. Invalid tool output, residual quotation marks, or effectively
verbatim replacements are retried and then fail closed.

Outputs per model:
  results/records_exp4b_<key>.json
  results/edited_reads_exp4b_<key>.json
  results/exp4b_summary_<key>.json

Usage:
  python3 exp4_rewrite.py [qwen7b gemma12b gemma27b] [--workers 12] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
from pathlib import Path

import anthropic

from edits import _coalesce, _curly_spans, _norm_match, _straight_spans

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
RES = HERE / "results"
DOCS_PATH = HERE / "data" / "ufw_docs.json"
MODEL = "claude-haiku-4-5-20251001"
POLICY_VERSION = 5
KEYS = ("qwen7b", "gemma12b", "gemma27b")
QUOTE_MARKS = '"“”„‟«»‹›`'
QUOTE_RE = re.compile(f"[{re.escape(QUOTE_MARKS)}]")
CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
MIN_SURFACE_CHANGE = 0.08

SYSTEM = (
    "You edit an activation-verbalizer explanation without changing its claims. You receive the full "
    "explanation and numbered QUOTED SPANS. Return one replacement per span. Each replacement will "
    "replace the entire span, including its quotation marks, while all surrounding text stays byte-for-"
    "byte unchanged.\n\n"
    "Requirements:\n"
    "- Preserve the span's exact proposition, prediction, entity, number, polarity, and uncertainty.\n"
    "- Do not correct, delete, soften, strengthen, or add claims, even if a claim seems false.\n"
    "- Express the same meaning as fluent prose that fits naturally between the supplied BEFORE and "
    "AFTER context.\n"
    "- Destroy quotation form and verbatim surface wording: use no quotation marks, guillemets, or "
    "backticks, and substantially rephrase rather than merely removing delimiters.\n"
    "- Never return a span unchanged or repeat a complete multiword proposition. Substantially rework "
    "the syntax and wording. A one-word token must be embedded in a descriptive metalinguistic phrase "
    "rather than returned alone; repeat that token inside the description only when identity requires it.\n"
    "- Proper names, product names, acronyms, technical identifiers, and exact numerals are irreducible: "
    "retain those anchors when needed for the same claim, but embed them in genuinely reworked prose.\n"
    "- Preserve fine distinctions with equivalent wording: his owner's husband can become the male "
    "spouse of his owner; a robotic vacuum can become an automated vacuum cleaner; the family's "
    "cleaning staff can become personnel employed to clean for the family. Do not broaden a vacuum "
    "into a generic device or drop possession/gender.\n"
    "- Preserve the open slot in an incomplete clause rather than inventing its completion. For example, "
    "a truncated phrase must remain explicitly incomplete rather than receiving an invented object.\n"
    "- Return every id exactly once and no extra ids."
)

TOOL = [{
    "name": "record_replacements",
    "description": "Return one meaning-preserving, unquoted prose replacement for every span id.",
    "input_schema": {
        "type": "object",
        "properties": {
            "replacements": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "text": {"type": "string"},
                    },
                    "required": ["id", "text"],
                },
            }
        },
        "required": ["replacements"],
    },
}]

FIDELITY_SYSTEM = (
    "You audit whether a rewrite of a quoted span preserves its proposition. Judge only semantic "
    "fidelity, not truth, style, fluency, or grammatical form. Mark preserved=true when the replacement "
    "retains the concrete entities, predicates, relations, numbers, polarity, modality, uncertainty, and "
    "the role of an incomplete phrase. True synonyms, reordered syntax, and nominal/verbal alternations "
    "pass when denotation and claim-level relations are unchanged. For example, his owner's husband -> "
    "the male spouse of his owner, a robotic vacuum -> an automated vacuum cleaner, and in excess of "
    "thirty-five years -> more than 35 years all pass. Do not demand the same article or preposition when "
    "the relation remains equivalent. Broadening a vacuum to a generic device, dropping gender or "
    "possession, changing polarity/modality, or inventing a completion fails. Proper names, product names, "
    "acronyms, exact numbers, and name fragments may remain as irreducible anchors inside reworked prose. "
    "For ordinary one-word or incomplete fragments, a semantic or metalinguistic gloss passes if it "
    "preserves the same role or denotation; it may name the token inside that longer gloss. Do not penalize "
    "changed wording or quotation removal. For every "
    "failed pair, provide a corrected fluent replacement that preserves all source content, uses no "
    "quotation marks, and does not copy a complete multiword proposition verbatim. Return every id once."
)

FIDELITY_TOOL = [{
    "name": "record_fidelity",
    "description": "Record whether every source/replacement pair has exactly preserved meaning.",
    "input_schema": {
        "type": "object",
        "properties": {
            "verdicts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "preserved": {"type": "boolean"},
                        "reason": {"type": "string"},
                        "suggested_replacement": {"type": "string"},
                    },
                    "required": ["id", "preserved", "reason", "suggested_replacement"],
                },
            }
        },
        "required": ["verdicts"],
    },
}]

_cache_lock = threading.Lock()
_progress_lock = threading.Lock()
_progress = {"done": 0, "total": 0}


def load_env(path: Path) -> None:
    """Load simple KEY=VALUE entries without adding python-dotenv as a project dependency."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name, value = name.strip(), value.strip()
        if value[:1] == value[-1:] and value[:1] in ("'", '"'):
            value = value[1:-1]
        if name:
            os.environ.setdefault(name, value)


def quote_spans(text: str) -> list[tuple[int, int]]:
    return _coalesce(_straight_spans(text) + _curly_spans(text))


def quote_rows(text: str) -> list[dict]:
    rows = []
    for start, end in quote_spans(text):
        inner = text[start:end].strip(QUOTE_MARKS).strip()
        if not inner:
            continue
        rows.append({
            "id": len(rows),
            "span": [start, end],
            "text": inner,
            "before": text[max(0, start - 90):start],
            "after": text[end:min(len(text), end + 90)],
        })
    return rows


def build_user(expl: str, rows: list[dict]) -> str:
    blocks = []
    for row in rows:
        blocks.append(
            f"[{row['id']}]\nBEFORE: {row['before']}\nSPAN: {row['text']}\nAFTER: {row['after']}"
        )
    return (
        f"FULL EXPLANATION:\n{expl}\n\nQUOTED SPANS:\n" + "\n\n".join(blocks) +
        "\n\nCall record_replacements with one replacement per id."
    )


def surface_change(source: str, replacement: str) -> float:
    a, b = _norm_match(source), _norm_match(replacement)
    return 1.0 - SequenceMatcher(None, a, b).ratio()


def max_verbatim_run(source: str, replacement: str) -> int:
    """Longest contiguous normalized source-word run retained by the replacement."""
    src = _norm_match(source).split()
    rep = _norm_match(replacement).split()
    best = 0
    for i in range(len(src)):
        for j in range(len(rep)):
            k = 0
            while i + k < len(src) and j + k < len(rep) and src[i + k] == rep[j + k]:
                k += 1
            best = max(best, k)
    return best


def is_literal_wrapper(replacement: str) -> bool:
    return replacement.startswith(("the literal token ", "the literal fragment ",
                                   "the literal foreign-language label ",
                                   "the literal proper-name sequence "))


def canonicalize_irreducible_spans(rows: list[dict], replacements: dict[int, str]) -> None:
    """Fail-safe for spans whose identity cannot be preserved through lexical paraphrase.

    Ordinary propositions remain model-paraphrased. A short fragment or non-Latin label returned empty
    or with its complete surface intact is embedded in explicit unquoted metalinguistic prose and later
    reported separately, rather than silently dropping or changing its identity.
    """
    for row in rows:
        source = row["text"].strip()
        rep = replacements.get(row["id"], "").strip()
        words = _norm_match(source).split()
        full_retained = bool(words and max_verbatim_run(source, rep) == len(words))
        insufficient = not rep or surface_change(source, rep) < MIN_SURFACE_CHANGE
        non_latin = any(char.isalpha() and ord(char) > 127 for char in source)
        proper_name_like = sum(
            1 for word in source.split()
            if any(char.isupper() for char in word) or word.rstrip(".,").endswith(("Inc", "LLC"))
        ) >= 2
        if ((len(words) <= 3 and (full_retained or insufficient))
                or ((non_latin or proper_name_like) and full_retained)):
            if non_latin:
                replacements[row["id"]] = f"the literal foreign-language label {source}"
            elif proper_name_like and len(words) > 3:
                replacements[row["id"]] = f"the literal proper-name sequence {source}"
            else:
                noun = "token" if len(words) <= 1 else "fragment"
                replacements[row["id"]] = f"the literal {noun} {source}"


def is_irreducible_anchor(source: str) -> bool:
    """Short proper or technical identifiers may anchor otherwise reworked prose."""
    words = source.strip().split()
    if not 1 <= len(words) <= 3:
        return False
    return any(
        any(char.isdigit() for char in word)
        or any(char.isupper() for char in word)
        or (len(word) > 1 and not word.isalpha())
        for word in words
    )


def violates_surface_policy(source: str, replacement: str) -> bool:
    if is_literal_wrapper(replacement):
        return False
    src_words = _norm_match(source).split()
    run = max_verbatim_run(source, replacement)
    if len(src_words) == 1:
        return False
    return bool(len(src_words) >= 2 and run == len(src_words)
                and not is_irreducible_anchor(source))


def validate_replacements(rows: list[dict], replacements: dict[int, str]) -> None:
    expected = {row["id"] for row in rows}
    if set(replacements) != expected:
        raise ValueError(f"replacement ids {sorted(replacements)} != expected {sorted(expected)}")
    violations = []
    for row in rows:
        rep = replacements[row["id"]].strip()
        if not rep:
            violations.append(f"replacement {row['id']} is empty")
            continue
        if QUOTE_RE.search(rep):
            violations.append(f"replacement {row['id']} contains a quote mark")
            continue
        wrapper = is_literal_wrapper(rep)
        if not wrapper and surface_change(row["text"], rep) < MIN_SURFACE_CHANGE:
            violations.append(f"replacement {row['id']} is insufficiently rephrased")
        src_words = _norm_match(row["text"]).split()
        run = max_verbatim_run(row["text"], rep)
        if (not wrapper and len(src_words) >= 2 and run == len(src_words)
              and not is_irreducible_anchor(row["text"])):
            violations.append(f"replacement {row['id']} retains the full quoted phrase")
    if violations:
        raise ValueError("; ".join(violations))


def apply_replacements(expl: str, rows: list[dict], replacements: dict[int, str]) -> tuple[str, list[list[int]]]:
    validate_replacements(rows, replacements)
    out, added, prev, length = [], [], 0, 0
    for row in rows:
        start, end = row["span"]
        prefix = QUOTE_RE.sub("", expl[prev:start])
        out.append(prefix)
        length += len(prefix)
        rep = replacements[row["id"]].strip()
        added.append([length, length + len(rep)])
        out.append(rep)
        length += len(rep)
        prev = end
    out.append(QUOTE_RE.sub("", expl[prev:]))
    edited = "".join(out)
    if QUOTE_RE.search(edited):
        raise ValueError("residual double-quotation mark remains after all replacements")
    return edited, added


def parse_replacement_payload(payload) -> list[dict]:
    if not isinstance(payload, str):
        return payload
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        items = []
        pattern = re.compile(
            r'\{\s*"id"\s*:\s*(\d+)\s*,\s*"text"\s*:\s*(.*?)\s*\}(?=\s*,|\s*\])',
            re.S,
        )
        for match in pattern.finditer(payload):
            value = match.group(2).strip()
            if value.startswith('"') and value.endswith('"'):
                try:
                    value = json.loads(value)
                except json.JSONDecodeError:
                    value = value[1:-1]
            else:
                value = value.rstrip('"').strip()
            items.append({"id": int(match.group(1)), "text": value})
        if not items:
            raise
        return items


def request_replacements(client: anthropic.Anthropic, expl: str, rows: list[dict],
                         semantic_feedback: str = "", split_fallback: bool = True) -> dict[int, str]:
    last_error = None
    for attempt in range(4):
        try:
            retry_note = ""
            if last_error is not None:
                rejected_ids = {int(value) for value in re.findall(r"replacement (\d+)", str(last_error))}
                targeted_rows = [row for row in rows if row["id"] in rejected_ids]
                targeted = "".join(
                    f" For rejected span {row['id']}, do not reproduce the source string "
                    f"{row['text']!r}."
                    for row in targeted_rows
                )
                if targeted_rows:
                    targeted += (" If one is an atomic token or name, embed it in a longer descriptive "
                                 "metalinguistic phrase; do not return the bare token alone.")
                retry_note = (
                    "\n\nA previous attempt was rejected by the mechanical validator: "
                    f"{last_error}. Correct that exact defect; in particular, do not return short "
                    f"tokens or longer spans unchanged.{targeted}"
                )
            msg = client.messages.create(
                model=MODEL,
                max_tokens=2400,
                temperature=0,
                system=SYSTEM,
                tools=TOOL,
                tool_choice={"type": "tool", "name": "record_replacements"},
                messages=[{"role": "user", "content": build_user(expl, rows) + semantic_feedback + retry_note}],
            )
            payload = None
            for block in msg.content:
                if block.type == "tool_use":
                    payload = block.input.get("replacements", [])
                    break
            if payload is None:
                raise ValueError("tool result missing")
            payload = parse_replacement_payload(payload)
            replacements = {
                int(item["id"]): str(item["text"]).strip()
                for item in payload
                if isinstance(item, dict) and "id" in item and "text" in item
            }
            canonicalize_irreducible_spans(rows, replacements)
            validate_replacements(rows, replacements)
            apply_replacements(expl, rows, replacements)
            return replacements
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < 3:
                time.sleep(2 ** attempt)
    if split_fallback and len(rows) > 1:
        merged = {}
        for row in rows:
            singleton = dict(row, id=0)
            one = request_replacements(
                client, expl, [singleton], semantic_feedback, split_fallback=False)
            merged[row["id"]] = one[0]
        validate_replacements(rows, merged)
        apply_replacements(expl, rows, merged)
        return merged
    raise RuntimeError(f"rewrite failed after 4 attempts: {last_error}")


def audit_replacements(client: anthropic.Anthropic, rows: list[dict],
                       replacements: dict[int, str]) -> dict[int, dict]:
    pairs = "\n\n".join(
        f"[{row['id']}]\nBEFORE: {row['before']}\nSOURCE: {row['text']}\n"
        f"REPLACEMENT: {replacements[row['id']]}\nAFTER: {row['after']}"
        for row in rows
    )
    last_error = None
    for attempt in range(4):
        try:
            msg = client.messages.create(
                model=MODEL,
                max_tokens=1800,
                temperature=0,
                system=FIDELITY_SYSTEM,
                tools=FIDELITY_TOOL,
                tool_choice={"type": "tool", "name": "record_fidelity"},
                messages=[{"role": "user", "content": pairs + "\n\nCall record_fidelity."}],
            )
            payload = None
            for block in msg.content:
                if block.type == "tool_use":
                    payload = block.input.get("verdicts", [])
                    break
            if isinstance(payload, str):
                payload = json.loads(payload)
            verdicts = {
                int(item["id"]): {
                    "preserved": bool(item["preserved"]),
                    "reason": str(item["reason"]),
                    "suggested_replacement": str(item.get("suggested_replacement", "")).strip(),
                }
                for item in (payload or [])
                if isinstance(item, dict) and "id" in item and "preserved" in item and "reason" in item
            }
            expected = {row["id"] for row in rows}
            if set(verdicts) != expected:
                raise ValueError(f"fidelity ids {sorted(verdicts)} != expected {sorted(expected)}")
            return verdicts
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < 3:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"semantic audit failed after 4 attempts: {last_error}")


def cache_path(key: str) -> Path:
    return RES / f"exp4b_cache_{key}.jsonl"


def load_cache(key: str) -> dict[int, dict]:
    out = {}
    path = cache_path(key)
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            out[int(row["idx"])] = row
    return out


def migrate_policy4_candidates(key: str, source_records: list[dict]) -> int:
    """Reuse mechanically valid pilot candidates from the superseded all-spans audit policy.

    Policy 4 rejected an entire datapoint when any span failed the deliberately adversarial semantic
    judge. Policy 5 retains those verdicts as diagnostics but follows the handoff's one-rewrite-call
    production design. This migration avoids paying to generate identical candidates again.
    """
    source = {rec["idx"]: rec for rec in source_records}
    existing_v5 = set()
    candidates: dict[int, dict] = {}
    path = cache_path(key)
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            cached = json.loads(line)
            idx = int(cached["idx"])
            if cached.get("policy_version") == POLICY_VERSION:
                existing_v5.add(idx)
            elif cached.get("policy_version") == 4 and idx in source:
                candidates[idx] = cached
    failures = RES / f"exp4b_failures_{key}.jsonl"
    if failures.exists():
        for line in failures.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            failed = json.loads(line)
            idx = int(failed["idx"])
            if failed.get("policy_version") != 4 or idx not in source or not failed.get("attempts"):
                continue
            attempt = failed["attempts"][-1]
            candidates.setdefault(idx, {
                "idx": idx,
                "model": MODEL,
                "quotes": quote_rows(source[idx]["expl_orig"]),
                "replacements": attempt["replacements"],
                "fidelity": attempt["fidelity"],
                "semantic_rounds": attempt.get("round", 4),
            })
    migrated = []
    for idx, candidate in candidates.items():
        if idx in existing_v5:
            continue
        try:
            rows = quote_rows(source[idx]["expl_orig"])
            replacements = {int(k): v for k, v in candidate["replacements"].items()}
            apply_replacements(source[idx]["expl_orig"], rows, replacements)
            fidelity = candidate.get("fidelity") or {}
            migrated.append({
                "idx": idx,
                "model": MODEL,
                "policy_version": POLICY_VERSION,
                "quotes": rows,
                "replacements": {str(k): v for k, v in replacements.items()},
                "fidelity": {str(k): v for k, v in fidelity.items()},
                "semantic_rounds": int(candidate.get("semantic_rounds", 1)),
                "semantic_audited": True,
                "migrated_from_policy": 4,
            })
        except (KeyError, TypeError, ValueError):
            continue
    if migrated:
        with path.open("a", encoding="utf-8") as stream:
            for cached in migrated:
                stream.write(json.dumps(cached, ensure_ascii=False) + "\n")
    return len(migrated)


def cached_rewrite_is_valid(rec: dict, cached: dict) -> bool:
    try:
        if cached.get("policy_version") != POLICY_VERSION:
            return False
        rows = cached["quotes"]
        replacements = {int(k): v for k, v in cached["replacements"].items()}
        apply_replacements(rec["expl_orig"], rows, replacements)
        return True
    except (KeyError, TypeError, ValueError):
        return False


def rewrite_and_cache(client: anthropic.Anthropic, key: str, rec: dict,
                      semantic_audit: bool = False) -> dict:
    expl = rec["expl_orig"]
    rows = quote_rows(expl)
    feedback = ""
    replacements = None
    fidelity = None
    attempt_log = []
    for semantic_round in range(1, 5):
        try:
            replacements = request_replacements(client, expl, rows, feedback)
        except Exception as exc:
            raise RuntimeError(f"idx {rec['idx']}: {exc}") from exc
        if not semantic_audit:
            fidelity = {
                row["id"]: {
                    "preserved": None,
                    "reason": "not independently audited",
                    "suggested_replacement": "",
                }
                for row in rows
            }
            break
        fidelity = audit_replacements(client, rows, replacements)
        failed = {idx: verdict for idx, verdict in fidelity.items() if not verdict["preserved"]}
        attempt_log.append({
            "round": semantic_round,
            "replacements": {str(k): v for k, v in replacements.items()},
            "fidelity": {str(k): v for k, v in fidelity.items()},
        })
        if not failed:
            break
        repaired = dict(replacements)
        for idx, verdict in failed.items():
            if verdict.get("suggested_replacement"):
                repaired[idx] = verdict["suggested_replacement"]
        try:
            validate_replacements(rows, repaired)
            apply_replacements(expl, rows, repaired)
            repaired_fidelity = audit_replacements(client, rows, repaired)
            if all(verdict["preserved"] for verdict in repaired_fidelity.values()):
                replacements, fidelity = repaired, repaired_fidelity
                break
        except ValueError:
            pass
        details = " ".join(
            f"Span {idx} failed semantic preservation: {verdict['reason']}"
            for idx, verdict in sorted(failed.items())
        )
        feedback = (
            "\n\nAn independent semantic audit rejected the previous rewrite. Regenerate all spans, "
            "with special attention to these failures: " + details
        )
    else:
        failure = {"idx": rec["idx"], "policy_version": POLICY_VERSION, "attempts": attempt_log}
        with _cache_lock, (RES / f"exp4b_failures_{key}.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(failure, ensure_ascii=False) + "\n")
        raise RuntimeError(f"idx {rec['idx']}: semantic preservation failed after 4 rewrite rounds")
    assert replacements is not None and fidelity is not None
    cached = {
        "idx": rec["idx"],
        "model": MODEL,
        "policy_version": POLICY_VERSION,
        "quotes": rows,
        "replacements": {str(k): v for k, v in replacements.items()},
        "fidelity": {str(k): v for k, v in fidelity.items()},
        "semantic_rounds": semantic_round,
        "semantic_audited": semantic_audit,
    }
    with _cache_lock, cache_path(key).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(cached, ensure_ascii=False) + "\n")
    with _progress_lock:
        _progress["done"] += 1
        if _progress["done"] % 50 == 0 or _progress["done"] == _progress["total"]:
            print(f"  rewrote {_progress['done']}/{_progress['total']}", flush=True)
    return cached


def materialize(key: str, source_records: list[dict], cache: dict[int, dict]) -> dict:
    records, reads = [], {}
    n_quotes = source_cjk = new_cjk = residual = verbatim = literal_wrappers = 0
    semantic_rounds = []
    audited_quotes = preserved_quotes = 0
    changes = []
    for source in source_records:
        idx = source["idx"]
        cached = cache[idx]
        rows = cached["quotes"]
        replacements = {int(k): v for k, v in cached["replacements"].items()}
        fidelity = {int(k): v for k, v in cached["fidelity"].items()}
        if cached.get("semantic_audited"):
            semantic_rounds.append(int(cached.get("semantic_rounds", 1)))
        edited, added = apply_replacements(source["expl_orig"], rows, replacements)
        per_quote = []
        for row in rows:
            rep = replacements[row["id"]]
            change = surface_change(row["text"], rep)
            changes.append(change)
            verdict = fidelity.get(row["id"], {"preserved": None, "reason": "not independently audited"})
            if verdict.get("preserved") is not None:
                audited_quotes += 1
                preserved_quotes += int(bool(verdict["preserved"]))
            if violates_surface_policy(row["text"], rep):
                verbatim += 1
            literal_wrapper = is_literal_wrapper(rep)
            literal_wrappers += int(literal_wrapper)
            per_quote.append({
                "id": row["id"],
                "span": row["span"],
                "original": row["text"],
                "replacement": rep,
                "surface_change": round(change, 4),
                "atomic": len(_norm_match(row["text"]).split()) == 1,
                "max_verbatim_run": max_verbatim_run(row["text"], rep),
                "literal_wrapper": literal_wrapper,
                "fidelity": verdict,
            })
        residual += int(bool(QUOTE_RE.search(edited)))
        had_cjk = bool(CJK_RE.search(source["expl_orig"]))
        has_cjk = bool(CJK_RE.search(edited))
        source_cjk += int(had_cjk)
        new_cjk += int(has_cjk and not had_cjk)
        n_quotes += len(rows)
        rec = dict(source)
        rec["expl_edit"] = edited
        rec["removed"] = [row["span"] for row in rows]
        rec["added"] = added
        rec["cos_orig"] = rec["cos_edit"] = rec["fve_i_orig"] = rec["fve_i_edit"] = None
        rec["edit_meta"] = {
            "arm": "exp4b",
            "model": MODEL,
            "n_rewritten": len(rows),
            "chars_replaced": sum(e - s for s, e in (row["span"] for row in rows)),
            "quotes": per_quote,
        }
        records.append(rec)
        reads[str(idx)] = edited

    (RES / f"records_exp4b_{key}.json").write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    (RES / f"edited_reads_exp4b_{key}.json").write_text(
        json.dumps({"reads": reads, "arm": "exp4b"}, ensure_ascii=False), encoding="utf-8")
    summary = {
        "key": key,
        "model": MODEL,
        "policy_version": POLICY_VERSION,
        "arm": "exp4b",
        "rewrite_policy": "semantic_paraphrase_with_short_literal_wrappers_v1",
        "n_datapoints": len(records),
        "n_quotes_rewritten": n_quotes,
        "n_lexical_rewrites": n_quotes - literal_wrappers,
        "n_literal_wrappers": literal_wrappers,
        "literal_wrapper_rate": round(literal_wrappers / n_quotes, 4) if n_quotes else None,
        "residual_quote_datapoints": residual,
        "verbatim_phrase_violations": verbatim,
        "source_cjk_datapoints": source_cjk,
        "new_cjk_datapoints": new_cjk,
        "mean_surface_change": round(sum(changes) / len(changes), 4) if changes else None,
        "min_surface_change": round(min(changes), 4) if changes else None,
        "mean_semantic_rounds": round(sum(semantic_rounds) / len(semantic_rounds), 3)
        if semantic_rounds else None,
        "n_semantic_audited_quotes": audited_quotes,
        "n_semantic_preserved_quotes": preserved_quotes,
        "semantic_audit_pass_rate": round(preserved_quotes / audited_quotes, 4)
        if audited_quotes else None,
    }
    (RES / f"exp4b_summary_{key}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_key(key: str, workers: int, limit: int | None, semantic_audit: bool) -> None:
    source = json.loads((RES / f"records_exp3a_{key}.json").read_text(encoding="utf-8"))
    source = [r for r in source if r.get("in_range") and (r.get("expl_orig") or "").strip()]
    if limit is not None:
        source = source[:limit]
    docs = {d["id"]: d["text"] for d in json.loads(DOCS_PATH.read_text(encoding="utf-8"))}
    missing_docs = sorted({r["doc_id"] for r in source} - set(docs))
    if missing_docs:
        raise KeyError(f"{len(missing_docs)} source doc_ids are absent from {DOCS_PATH}")

    n_migrated = migrate_policy4_candidates(key, source)
    if n_migrated:
        print(f"[exp4b:{key}] migrated {n_migrated} mechanically valid policy-4 pilot candidates",
              flush=True)
    loaded_cache = load_cache(key)
    by_idx = {r["idx"]: r for r in source}
    cache = {idx: cached for idx, cached in loaded_cache.items()
             if idx in by_idx and cached_rewrite_is_valid(by_idx[idx], cached)}
    n_invalid_cache = sum(idx in by_idx for idx in loaded_cache) - len(cache)
    if n_invalid_cache:
        print(f"[exp4b:{key}] ignoring {n_invalid_cache} cached rewrites that fail current validation",
              flush=True)
    todo = [r for r in source if r["idx"] not in cache]
    _progress["done"], _progress["total"] = len(source) - len(todo), len(source)
    print(f"[exp4b:{key}] {len(todo)} API rewrites ({len(source) - len(todo)} cached)", flush=True)
    client = anthropic.Anthropic()
    failures = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(rewrite_and_cache, client, key, rec, semantic_audit): rec
            for rec in todo
        }
        for future in as_completed(futures):
            rec = futures[future]
            try:
                cache[rec["idx"]] = future.result()
            except Exception as exc:  # noqa: BLE001
                failures.append((rec["idx"], str(exc)))
                print(f"  [exp4b:{key}] FAILED idx {rec['idx']}: {exc}", flush=True)
    if failures:
        sample = "; ".join(f"{idx}: {msg}" for idx, msg in failures[:5])
        raise RuntimeError(f"{len(failures)} rewrites failed semantic/mechanical gates; {sample}")
    summary = materialize(key, source, cache)
    print(f"[exp4b:{key}] DONE {json.dumps(summary)}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("keys", nargs="*")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--semantic-audit", action="store_true",
                        help="independently audit and repair every span (diagnostic; much stricter/costlier)")
    args = parser.parse_args()
    load_env(REPO / ".env")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is not set")
    keys = args.keys or KEYS
    unknown = sorted(set(keys) - set(KEYS))
    if unknown:
        parser.error(f"unknown model key(s): {', '.join(unknown)}")
    for key in keys:
        run_key(key, args.workers, args.limit, args.semantic_audit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
