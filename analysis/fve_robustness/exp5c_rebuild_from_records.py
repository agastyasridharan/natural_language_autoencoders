"""Rebuild corrected Exp. 5c from existing offset-verified records (CPU, tokenizer-free).

The original 5c records already contain the frozen manifest fields and exact token/character offsets.
Only ``expl_edit`` and reorder metadata change under the v2 ECHO-freezing parser, so re-tokenizing the
documents is unnecessary and creates an avoidable cache dependency.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from edits import shuffle_quoted_claims

KEYS = ("qwen7b", "gemma12b", "gemma27b")


def rebuild(root: Path, key: str) -> None:
    out = root / f"out_{key}"
    path = out / f"records_exp5c_{key}.json"
    records = json.loads(path.read_text())
    reads = {}
    edited_count = 0
    for record in records:
        edited, meta = shuffle_quoted_claims(record.get("expl_orig", ""))
        record["expl_edit"] = edited
        record["edit_meta"] = dict(meta, policy_version="echo_frozen_v2")
        for field in ("cos_orig", "cos_edit", "D_i", "fve_i_orig", "fve_i_edit"):
            record[field] = None
        reads[str(record["idx"])] = edited
        edited_count += int(edited != record.get("expl_orig", ""))
    path.write_text(json.dumps(records, ensure_ascii=False))
    (out / "edited_reads_exp5c.json").write_text(
        json.dumps({"reads": reads, "arm": "exp5c", "policy_version": "echo_frozen_v2"},
                   ensure_ascii=False))
    print(f"[exp5c:{key}] rebuilt {len(records)} records; {edited_count} non-noop edits", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("keys", nargs="*")
    parser.add_argument("--root", type=Path, default=Path("/data/agastyas/nla_fve_ufw"))
    args = parser.parse_args()
    invalid = sorted(set(args.keys) - set(KEYS))
    if invalid:
        parser.error(f"unknown model key(s): {', '.join(invalid)}")
    for key in (args.keys or KEYS):
        rebuild(args.root, key)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
