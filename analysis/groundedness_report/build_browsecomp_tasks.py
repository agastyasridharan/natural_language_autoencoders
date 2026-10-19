"""Fetch, decrypt, and freeze 50 public BrowseComp replication tasks."""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
from pathlib import Path
from typing import Any

import httpx

from .io import atomic_json, atomic_jsonl, sha256_file
from .protocol import load_evaluation_registry, load_protocol
from .sampling import sha256_rank
from .spec import PROTOCOL_SHA256


DATASET_URL = "https://openaipublic.blob.core.windows.net/simple-evals/browse_comp_test_set.csv"
QUERY_TEMPLATE = """{question}

Your response should be in the following format:
Explanation: {{your explanation for your final answer}}
Exact Answer: {{your succinct, final answer}}
Confidence: {{your confidence score between 0% and 100% for your answer}}"""


def _derive_key(password: str, length: int) -> bytes:
    digest = hashlib.sha256(password.encode()).digest()
    return digest * (length // len(digest)) + digest[: length % len(digest)]


def decrypt(ciphertext: str, password: str) -> str:
    encrypted = base64.b64decode(ciphertext)
    key = _derive_key(password, len(encrypted))
    return bytes(left ^ right for left, right in zip(encrypted, key)).decode()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--rank-start", type=int, default=0)
    args = parser.parse_args()
    protocol = load_protocol()
    registry = load_evaluation_registry()
    family = next(row for row in registry["families"] if row["id"] == "capability_browsecomp")
    response = httpx.get(DATASET_URL, timeout=args.timeout, follow_redirects=True)
    response.raise_for_status()
    raw = response.content
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
    if len(rows) != 1266 or set(rows[0]) != {"problem", "answer", "problem_topic", "canary"}:
        raise RuntimeError("BrowseComp public dataset shape or schema changed")
    decoded = []
    for source_row, row in enumerate(rows):
        decoded.append({
            "source_row": source_row,
            "question": decrypt(row["problem"], row["canary"]),
            "answer": decrypt(row["answer"], row["canary"]),
            "problem_topic": row["problem_topic"],
            "ciphertext_sha256": hashlib.sha256(_canonical(row)).hexdigest(),
        })
    ranked = sha256_rank(
        decoded,
        seed=int(protocol["seed"]),
        namespace="ood-task:capability_browsecomp",
        identity=lambda row: f"{row['source_row']}:{row['ciphertext_sha256']}",
    )
    if args.rank_start < 0 or args.rank_start + 50 > len(ranked):
        raise ValueError("rank-start must leave a complete 50-task BrowseComp block")
    selected = ranked[args.rank_start : args.rank_start + 50]
    tasks = []
    for row in selected:
        task_id = f"browsecomp-{int(row['source_row']):04d}"
        task = {
            "schema_version": 1,
            "protocol_sha256": PROTOCOL_SHA256,
            "replication_label": registry["label_for_every_ood_row"],
            "family_id": family["id"],
            "task_id": task_id,
            "interaction_kind": "recorded_public_web_search_and_fetch_tools",
            "messages": [{
                "role": "user",
                "content": QUERY_TEMPLATE.format(question=row["question"]),
            }],
            "source": {
                "benchmark_repo": family["source"],
                "benchmark_revision": family["revision"],
                "dataset_url": DATASET_URL,
                "dataset_sha256": hashlib.sha256(raw).hexdigest(),
                "source_row": row["source_row"],
                "ciphertext_sha256": row["ciphertext_sha256"],
            },
            "reference": {
                "answer": row["answer"],
                "problem_topic": row["problem_topic"],
            },
        }
        task["task_sha256"] = hashlib.sha256(_canonical(task)).hexdigest()
        tasks.append(task)
    tasks.sort(key=lambda row: str(row["task_id"]))
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    dataset_path = output / "browse_comp_test_set.csv"
    temporary = dataset_path.with_suffix(".csv.tmp")
    temporary.write_bytes(raw)
    temporary.replace(dataset_path)
    task_path = output / "capability_browsecomp.jsonl"
    atomic_jsonl(task_path, tasks)
    atomic_json(output / "selection_meta.json", {
        "schema_version": 1,
        "protocol_sha256": PROTOCOL_SHA256,
        "family_id": family["id"],
        "source_url": DATASET_URL,
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "source_rows": len(rows),
        "selected": len(tasks),
        "rank_start": args.rank_start,
        "rank_stop_exclusive": args.rank_start + len(tasks),
        "selected_sha256": sha256_file(task_path),
        "selection": "50_row_block_from_sha256_ranking_keyed_by_protocol_seed",
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
