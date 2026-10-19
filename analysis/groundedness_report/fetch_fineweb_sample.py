"""Stage the frozen FineWeb document sample without downloading 30 GB of parquet."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any
import httpx

from .io import atomic_json, atomic_jsonl, read_json
from .protocol import load_protocol
from .spec import PROTOCOL_SHA256


HUB_API = "https://huggingface.co/api/datasets"
DATASETS_SERVER = "https://datasets-server.huggingface.co/rows"


def selected_blocks(seed: int, total_rows: int, width: int, count: int) -> list[int]:
    if width < 1 or count < 1 or total_rows // width < count:
        raise ValueError("invalid FineWeb block-sampling dimensions")
    available = total_rows // width
    output: list[int] = []
    seen: set[int] = set()
    counter = 0
    while len(output) < count:
        digest = hashlib.sha256(f"{seed}:fineweb-source:{counter}".encode()).digest()
        block = int.from_bytes(digest[:8], "big") % available
        if block not in seen:
            seen.add(block)
            output.append(block)
        counter += 1
    return output


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    protocol = load_protocol()
    fineweb = protocol["sampling"]["fineweb"]
    source = fineweb["source_sampling"]
    repository = fineweb["dataset"]
    revision = fineweb["revision"]
    output_dir = Path(args.out)
    responses_dir = output_dir / "responses"
    responses_dir.mkdir(parents=True, exist_ok=True)

    with httpx.Client(timeout=args.timeout, follow_redirects=True) as client:
        repo_response = client.get(f"{HUB_API}/{repository}")
        repo_response.raise_for_status()
        observed_sha = repo_response.json().get("sha")
        if observed_sha != revision:
            raise RuntimeError(
                f"FineWeb main is {observed_sha}, not frozen {revision}; "
                "use the revision parquet rather than the current datasets server"
            )

        blocks = selected_blocks(
            int(protocol["seed"]),
            int(source["total_rows_at_revision"]),
            int(source["rows_per_block"]),
            int(source["blocks"]),
        )
        rows: list[dict[str, Any]] = []
        response_records: list[dict[str, Any]] = []
        for order, block in enumerate(blocks):
            offset = block * int(source["rows_per_block"])
            params = {
                "dataset": repository,
                "config": fineweb["config"],
                "split": "train",
                "offset": offset,
                "length": int(source["rows_per_block"]),
            }
            cache_path = responses_dir / f"{order:03d}-{offset}.json"
            requested_url = str(httpx.Request("GET", DATASETS_SERVER, params=params).url)
            if cache_path.exists():
                payload = read_json(cache_path)
            else:
                last_error: Exception | None = None
                for attempt in range(8):
                    try:
                        response = client.get(DATASETS_SERVER, params=params)
                        response.raise_for_status()
                        payload = response.json()
                        break
                    except (httpx.HTTPError, ValueError) as error:
                        last_error = error
                        if attempt + 1 < 8:
                            time.sleep(min(20.0, 0.75 * (2**attempt)))
                else:
                    raise RuntimeError(
                        f"FineWeb block offset {offset} failed after 8 attempts: {last_error}"
                    )
            payload_rows = payload.get("rows") or []
            if len(payload_rows) != int(source["rows_per_block"]):
                raise RuntimeError(f"block {block} returned {len(payload_rows)} rows")
            response_sha = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
            atomic_json(cache_path, payload)
            response_records.append({
                "selection_order": order,
                "block_index": block,
                "offset": offset,
                "url": requested_url,
                "response_sha256": response_sha,
            })
            for within, wrapper in enumerate(payload_rows):
                row = wrapper.get("row") or {}
                text = row.get("text")
                if not isinstance(text, str) or not text.strip():
                    raise RuntimeError(f"empty text at source row {offset + within}")
                source_id = str(row.get("id") or "")
                rows.append({
                    "doc_id": f"fineweb:{offset + within}:{source_id}",
                    "text": text,
                    "source_row": offset + within,
                    "source_id": source_id,
                    "source_block": block,
                    "source_block_order": order,
                    "source_within_block": within,
                })
            print(f"[FineWeb {order + 1}/{len(blocks)}] offset={offset}", flush=True)

    if len(rows) != int(source["staged_documents"]):
        raise RuntimeError(f"staged {len(rows)} rows, expected {source['staged_documents']}")
    manifest_path = output_dir / "documents.jsonl"
    atomic_jsonl(manifest_path, rows)
    atomic_json(output_dir / "source_meta.json", {
        "schema_version": 1,
        "protocol_sha256": PROTOCOL_SHA256,
        "dataset": repository,
        "config": fineweb["config"],
        "revision": revision,
        "observed_hub_main_sha_at_fetch": observed_sha,
        "total_rows_at_revision": source["total_rows_at_revision"],
        "block_selection": source["block_selection"],
        "blocks": blocks,
        "rows_per_block": source["rows_per_block"],
        "documents": len(rows),
        "responses": response_records,
        "documents_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
