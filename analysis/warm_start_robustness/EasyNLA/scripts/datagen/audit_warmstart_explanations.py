"""Audit warm-start teacher outputs and emit durable JSON/Markdown reports."""

import argparse
import json
import re
from pathlib import Path

import pyarrow.parquet as pq


_WORD_RE = re.compile(r"\b[\w’'-]+\b")
_META_RE = re.compile(
    r"\b(?:true feature|false feature|actual feature|plausible feature|"
    r"instead of|the text actually|in reality|this is false|nonsense)\b",
    re.IGNORECASE,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parquet", required=True)
    p.add_argument("--json-output", required=True)
    p.add_argument("--markdown-output", required=True)
    p.add_argument("--expected-features", type=int, default=3)
    p.add_argument("--min-words", type=int, default=45)
    p.add_argument("--max-words", type=int, default=100)
    args = p.parse_args()

    table = pq.read_table(
        args.parquet,
        columns=["doc_id", "detokenized_text_truncated", "api_explanation"],
    )
    rows = []
    for i, row in enumerate(table.to_pylist()):
        explanation = row["api_explanation"].strip()
        features = [x.strip() for x in explanation.split("\n\n") if x.strip()]
        n_words = len(_WORD_RE.findall(explanation))
        flags = []
        if len(features) != args.expected_features:
            flags.append(f"feature_count={len(features)}")
        if not args.min_words <= n_words <= args.max_words:
            flags.append(f"word_count={n_words}")
        if _META_RE.search(explanation):
            flags.append("meta_language")
        rows.append({
            "row": i,
            "doc_id": row["doc_id"],
            "feature_count": len(features),
            "word_count": n_words,
            "format_pass": not flags,
            "flags": flags,
            "source_tail": row["detokenized_text_truncated"][-1000:],
            "explanation": explanation,
        })

    word_counts = sorted(r["word_count"] for r in rows)
    summary = {
        "parquet": str(Path(args.parquet).resolve()),
        "row_count": len(rows),
        "format_pass_count": sum(r["format_pass"] for r in rows),
        "format_pass_rate": sum(r["format_pass"] for r in rows) / len(rows),
        "feature_count_histogram": {
            str(n): sum(r["feature_count"] == n for r in rows)
            for n in sorted({r["feature_count"] for r in rows})
        },
        "word_count": {
            "min": word_counts[0],
            "median": word_counts[len(word_counts) // 2],
            "max": word_counts[-1],
        },
        "semantic_review_required": True,
        "note": "Format checks cannot establish that every claim is implausible.",
    }
    report = {"summary": summary, "rows": rows}
    json_path = Path(args.json_output)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    lines = [
        "# Warm-start teacher-output audit",
        "",
        f"Rows: {len(rows)}; format pass: {summary['format_pass_count']}/{len(rows)} "
        f"({summary['format_pass_rate']:.1%}).",
        "",
        "Every row below still requires semantic review: format compliance does not "
        "show that its claims are false.",
    ]
    for r in rows:
        lines += [
            "",
            f"## Row {r['row']} — {r['doc_id']}",
            "",
            f"Format: {'PASS' if r['format_pass'] else 'FAIL'}; "
            f"features={r['feature_count']}; words={r['word_count']}; "
            f"flags={r['flags'] or 'none'}",
            "",
            "Source tail:",
            "",
            "```text",
            r["source_tail"],
            "```",
            "",
            "Explanation:",
            "",
            "```text",
            r["explanation"],
            "```",
        ]
    md_path = Path(args.markdown_output)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
