"""Stream the first N rows of a large Parquet shard into a pinned local corpus.

The companion manifest records the upstream revision and cryptographic hashes,
so a local filename does not erase corpus provenance.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--rows", type=int, required=True)
    p.add_argument("--source-repo", required=True)
    p.add_argument("--source-revision", required=True)
    p.add_argument("--source-path", required=True)
    p.add_argument("--source-sha256", required=True)
    args = p.parse_args()

    src = Path(args.input)
    dst = Path(args.output)
    assert src.is_file(), src
    assert args.rows > 0
    # The HF cache blob is content-addressed by LFS SHA-256. Fail before doing
    # work if the supplied provenance does not match that identity.
    if len(src.name) == 64 and all(c in "0123456789abcdef" for c in src.name):
        assert src.name == args.source_sha256, (
            f"cache blob name {src.name} != declared SHA-256 {args.source_sha256}"
        )

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    pf = pq.ParquetFile(src)
    remaining = args.rows
    written = 0
    writer: pq.ParquetWriter | None = None
    try:
        for batch in pf.iter_batches(batch_size=min(8192, args.rows)):
            table = pa.Table.from_batches([batch])
            if writer is None:
                writer = pq.ParquetWriter(tmp, table.schema, compression="zstd")
            take = min(remaining, table.num_rows)
            writer.write_table(table.slice(0, take))
            written += take
            remaining -= take
            if remaining == 0:
                break
    finally:
        if writer is not None:
            writer.close()
    assert written == args.rows, f"source contained only {written}/{args.rows} requested rows"
    os.replace(tmp, dst)

    manifest = {
        "source_repo": args.source_repo,
        "source_revision": args.source_revision,
        "source_path": args.source_path,
        "source_sha256": args.source_sha256,
        "source_size_bytes": src.stat().st_size,
        "selection": {"start": 0, "length": args.rows},
        "output_path": str(dst),
        "output_size_bytes": dst.stat().st_size,
        "output_sha256": sha256_file(dst),
        "row_count": pq.read_metadata(dst).num_rows,
        "schema": str(pq.read_schema(dst)),
    }
    manifest_path = dst.with_suffix(dst.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
