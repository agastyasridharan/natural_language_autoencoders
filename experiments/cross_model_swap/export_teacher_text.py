"""Export public EasyNLA labels and visible prefixes, WITHOUT Qwen3 vectors."""
import argparse
import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml


def role_for_doc(doc):
    bucket = int.from_bytes(hashlib.blake2b(f"42:{doc}".encode(), digest_size=8).digest(), "big") % 100
    return "av" if bucket < 25 else "ar" if bucket < 50 else "rl"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    files = [a.input_dir / f"{mode}_sft_{split}.parquet"
             for mode in ("av", "ar") for split in ("train", "val")]
    train_docs, val_docs = set(), set()
    for f in files:
        docs = set(pq.read_table(f, columns=["doc_id"])["doc_id"].to_pylist())
        (val_docs if "_val" in f.name else train_docs).update(docs)
    if train_docs & val_docs:
        raise ValueError("Published validation docs overlap training docs")
    schema = pa.schema([("doc_id", pa.string()), ("text", pa.string()),
                        ("explanation", pa.string()), ("role", pa.string())])
    a.output.parent.mkdir(parents=True, exist_ok=True)
    counts = dict.fromkeys(["av", "ar", "rl", "val"], 0)
    seen = set()
    repeated_prefixes = 0
    sources = []
    with pq.ParquetWriter(a.output, schema, compression="zstd") as writer:
        for f in files:
            meta = yaml.safe_load(Path(str(f)+".nla_meta.yaml").read_text())
            assert meta["extraction"]["base_model"] == "Qwen/Qwen3-8B"
            template = meta["prompt_templates"]["critic"]
            before, after = template.split("{explanation}")
            field = "response" if f.name.startswith("av") else "prompt"
            cols = ["doc_id", "detokenized_text_truncated", field]
            for batch in pq.ParquetFile(f).iter_batches(batch_size=2048, columns=cols):
                out = []
                for row in batch.to_pylist():
                    doc, text = row["doc_id"], row["detokenized_text_truncated"]
                    key = hashlib.sha256((doc+"\0"+text).encode()).digest()
                    if key in seen:
                        repeated_prefixes += 1
                    seen.add(key)
                    raw = row[field]
                    if field == "response":
                        assert raw.strip().startswith("<explanation>") and raw.strip().endswith("</explanation>")
                        explanation = raw.strip()[len("<explanation>"):-len("</explanation>")].strip()
                    else:
                        assert raw.startswith(before) and raw.endswith(after)
                        explanation = raw[len(before):len(raw)-len(after)].strip()
                    assert text and explanation
                    role = "val" if doc in val_docs else role_for_doc(doc)
                    counts[role] += 1
                    out.append(dict(doc_id=doc, text=text, explanation=explanation, role=role))
                writer.write_table(pa.Table.from_pylist(out, schema=schema))
            sources.append({"path": str(f), "rows": pq.ParquetFile(f).metadata.num_rows,
                            "sidecar": meta})
    receipt = {"counts": counts, "repeated_prefix_rows_retained": repeated_prefixes, "train_documents": len(train_docs), "val_documents": len(val_docs),
               "sources": sources, "new_source": "Qwen/Qwen2.5-7B-Instruct",
               "source_block_index": 20, "source_vectors_reused": False,
               "prefix_policy": "retokenize the entire exact visible prefix with Qwen2.5",
               "split_policy": "original validation retained; other docs blake2b(42:doc_id) mod100: AV25 AR25 RL50"}
    Path(str(a.output)+".json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps({k:v for k,v in receipt.items() if k != "sources"}, indent=2))


if __name__ == "__main__":
    main()
