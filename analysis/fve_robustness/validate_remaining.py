"""Final integrity gates for completed Exp. 5b/5c and Exp. 6P/6T artifacts."""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from edits import _coalesce, _curly_spans, _straight_spans, split_blocks

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
KEYS = ("qwen7b", "gemma12b", "gemma27b")
ARMS = ("exp5b", "exp5c", "exp6p", "exp6t")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def span_skeleton(text: str, spans: list[list[int]]) -> str:
    out, previous = [], 0
    for start, end in sorted(spans):
        out.append(text[previous:start])
        previous = end
    out.append(text[previous:])
    return "".join(out)


def final_token_echo(text: str) -> str | None:
    for start, end in split_blocks(text):
        block = text[start:end]
        marker = re.search(r"Final token", block, re.IGNORECASE)
        if marker is None:
            continue
        quotes = _coalesce(_straight_spans(block) + _curly_spans(block))
        span = next(((qstart, qend) for qstart, qend in quotes if qstart >= marker.end()), None)
        return block[span[0]:span[1]] if span else None
    return None


def main() -> int:
    budget = json.loads((RES / "exp6_openai_budget.json").read_text())
    if float(budget["combined_estimated_cost_usd"]) >= float(budget["user_ceiling_usd"]):
        raise AssertionError("combined conservative API cost is not below the user ceiling")

    report = {"budget_upper_bound_usd": budget["combined_estimated_cost_usd"], "models": {}}
    for key in KEYS:
        model = {}
        for arm in ARMS:
            summary = json.loads((RES / f"fve_{arm}_{key}.json").read_text())
            records = json.loads((RES / f"records_{arm}_{key}.json").read_text())
            reads_path = RES / f"edited_reads_{arm}_{key}.json"
            if len(records) != 2000 or summary["n_paired"] != 2000 or summary["n_dropped_empty"] != 0:
                raise AssertionError(f"{key}:{arm} cardinality/drop invariant failed")
            if any(record.get("cos_edit") is None or record.get("D_i") is None for record in records):
                raise AssertionError(f"{key}:{arm} contains unscored records")
            if reads_path.exists() and summary["edited_reads_sha256"] != sha(reads_path):
                raise AssertionError(f"{key}:{arm} edited-read SHA mismatch")

            if arm == "exp5c":
                for record in records:
                    meta = record["edit_meta"]
                    if meta.get("policy_version") != "echo_frozen_v2":
                        raise AssertionError(f"{key}:exp5c stale policy")
                    if final_token_echo(record["expl_orig"]) != final_token_echo(record["expl_edit"]):
                        raise AssertionError(f"{key}:exp5c moved ECHO at idx={record['idx']}")
                    if Counter(record["expl_orig"]) != Counter(record["expl_edit"]):
                        raise AssertionError(f"{key}:exp5c changed the character multiset")
            elif arm in {"exp6p", "exp6t"}:
                for record in records:
                    if span_skeleton(record["expl_orig"], record.get("removed", [])) != span_skeleton(
                            record["expl_edit"], record.get("added", [])):
                        raise AssertionError(f"{key}:{arm} changed bytes outside selected quote spans")
                    if arm == "exp6p":
                        for quote in record["edit_meta"].get("quotes", []):
                            if not quote.get("audit_reason"):
                                raise AssertionError(f"{key}:exp6p missing semantic audit")
                    elif record["edit_meta"].get("continuation_wrong") not in (True, False, None):
                        raise AssertionError(f"{key}:exp6t invalid semantic label")
            model[arm] = dict(FVE_arm=summary["FVE_arm"], dFVE=summary["dFVE"],
                              ci95=summary["dFVE_ci95"])
        report["models"][key] = model
    (RES / "remaining_validation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
