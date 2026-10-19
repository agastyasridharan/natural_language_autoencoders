"""Experiment 4a: remove quotation delimiters while preserving quoted content exactly.

This deterministic control changes only double-quotation characters inside spans located by the
shared robustness helpers. Every other byte, including the literal content and whitespace inside a
span, is preserved. It isolates quotation punctuation/form from lexical surface dependence.

Outputs per model:
  results/records_exp4a_<key>.json
  results/edited_reads_exp4a_<key>.json
  results/exp4a_summary_<key>.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from edits import _coalesce, _curly_spans, _straight_spans

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
KEYS = ("qwen7b", "gemma12b", "gemma27b")
QUOTE_MARKS = '"“”„‟«»‹›`'
QUOTE_RE = re.compile(f"[{re.escape(QUOTE_MARKS)}]")
CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


def quote_spans(text: str) -> list[tuple[int, int]]:
    return _coalesce(_straight_spans(text) + _curly_spans(text))


def remove_delimiters(text: str) -> tuple[str, list[list[int]], list[list[int]]]:
    """Return edited text, detected quoted spans, and removed delimiter coordinates.

    All quote-like delimiters within a detected span are removed. Unmatched opening double quotes
    are represented by the shared helper as a span to EOF, so their content is retained as required.
    """
    spans = quote_spans(text)
    removed = [[i, i + 1] for i, char in enumerate(text) if char in QUOTE_MARKS]
    edited = "".join(char for char in text if char not in QUOTE_MARKS)
    return edited, [[s, e] for s, e in spans], removed


def run_key(key: str) -> dict:
    source = json.loads((RES / f"records_exp3a_{key}.json").read_text(encoding="utf-8"))
    source = [r for r in source if r.get("in_range") and (r.get("expl_orig") or "").strip()]
    records, reads = [], {}
    n_spans = n_delimiters = residual = source_cjk = new_cjk = 0
    for src in source:
        edited, spans, removed = remove_delimiters(src["expl_orig"])
        residual += int(bool(QUOTE_RE.search(edited)))
        had_cjk = bool(CJK_RE.search(src["expl_orig"]))
        has_cjk = bool(CJK_RE.search(edited))
        source_cjk += int(had_cjk)
        new_cjk += int(has_cjk and not had_cjk)
        n_spans += len(spans)
        n_delimiters += len(removed)
        rec = dict(src)
        rec["expl_edit"] = edited
        rec["removed"] = removed
        rec["added"] = []
        rec["cos_orig"] = rec["cos_edit"] = rec["fve_i_orig"] = rec["fve_i_edit"] = None
        rec["edit_meta"] = {
            "arm": "exp4a",
            "n_dequoted_spans": len(spans),
            "n_delimiters_removed": len(removed),
            "quoted_spans": spans,
            "literal_content_preserved": True,
        }
        records.append(rec)
        reads[str(src["idx"])] = edited

    (RES / f"records_exp4a_{key}.json").write_text(
        json.dumps(records, ensure_ascii=False), encoding="utf-8")
    (RES / f"edited_reads_exp4a_{key}.json").write_text(
        json.dumps({"reads": reads, "arm": "exp4a"}, ensure_ascii=False), encoding="utf-8")
    summary = {
        "key": key,
        "arm": "exp4a",
        "policy_version": 1,
        "rewrite_policy": "delimiter_only_v1",
        "n_datapoints": len(records),
        "n_quoted_spans": n_spans,
        "n_delimiters_removed": n_delimiters,
        "residual_quote_datapoints": residual,
        "source_cjk_datapoints": source_cjk,
        "new_cjk_datapoints": new_cjk,
        "literal_content_preserved": True,
    }
    (RES / f"exp4a_summary_{key}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[exp4a:{key}] {json.dumps(summary)}", flush=True)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("keys", nargs="*")
    args = parser.parse_args()
    keys = args.keys or KEYS
    unknown = sorted(set(keys) - set(KEYS))
    if unknown:
        parser.error(f"unknown model key(s): {', '.join(unknown)}")
    for key in keys:
        run_key(key)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
