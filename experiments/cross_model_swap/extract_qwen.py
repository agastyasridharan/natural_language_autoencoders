"""Shard exact visible prefixes across GPUs; extract raw Qwen2.5 block-20 outputs."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--rank", type=int, required=True)
    p.add_argument("--world-size", type=int, required=True)
    p.add_argument("--batch-size", type=int, default=32)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    torch.set_num_threads(2)
    cfg = AutoConfig.from_pretrained(a.checkpoint)
    assert cfg.model_type == "qwen2" and cfg.hidden_size == 3584 and cfg.num_hidden_layers == 28
    cfg.num_hidden_layers = 21  # Later causal blocks cannot affect block 20.
    if getattr(cfg, "layer_types", None) is not None:
        cfg.layer_types = cfg.layer_types[:21]
    tok = AutoTokenizer.from_pretrained(a.checkpoint, padding_side="right")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModel.from_pretrained(a.checkpoint, config=cfg, torch_dtype=torch.bfloat16,
                                      attn_implementation="sdpa", device_map="cuda:0")
    model.norm = torch.nn.Identity()  # Output of block 20, before final RMSNorm.
    model.eval()
    src = pq.ParquetFile(a.input)
    schema = src.schema_arrow.append(pa.field("activation_vector", pa.list_(pa.float32(), 3584))).append(pa.field("n_raw_tokens",pa.int32()))
    count, skipped_short = 0, 0
    start = time.monotonic()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.output.with_suffix(".incomplete")
    with pq.ParquetWriter(tmp, schema, compression="zstd") as writer:
        for batch in src.iter_batches(batch_size=2048):
            rows = [r for r in batch.to_pylist() if int.from_bytes(hashlib.blake2b(
                r["doc_id"].encode(), digest_size=8).digest(), "big") % a.world_size == a.rank]
            for i in range(0,len(rows),a.batch_size):
                subset=rows[i:i+a.batch_size]
                tokens=tok([r["text"] for r in subset], add_special_tokens=False, padding=True,
                           return_tensors="pt", truncation=False)
                lengths=tokens["attention_mask"].sum(-1)
                if int(lengths.max()) > 4096:
                    raise ValueError("Prefix exceeds 4096 Qwen tokens; refusing silent truncation")
                with torch.inference_mode():
                    h=model(**{k:v.cuda() for k,v in tokens.items()},use_cache=False).last_hidden_state
                    acts=h[torch.arange(len(subset),device=h.device),lengths.cuda()-1].float().cpu().numpy()
                assert np.isfinite(acts).all()
                result=[]
                for r,n,v in zip(subset,lengths.tolist(),acts):
                    if n-1 < 50:
                        skipped_short += 1
                        continue
                    result.append(dict(r,activation_vector=v.tolist(),n_raw_tokens=n))
                if result:
                    writer.write_table(pa.Table.from_pylist(result,schema=schema))
                    count+=len(result)
                if count and count//1000 != (count-len(result))//1000:
                    print(json.dumps({"rank":a.rank,"rows":count,"seconds":round(time.monotonic()-start,1)}),flush=True)
    tmp.rename(a.output)
    Path(str(a.output)+".json").write_text(json.dumps({"rank":a.rank,"rows":count,
        "skipped_below_position_50":skipped_short,"seconds":time.monotonic()-start,
        "source_checkpoint":a.checkpoint,"source_block_index":20,"norm":"none",
        "dtype":"bfloat16","stored_dtype":"float32","complete":True},indent=2))


if __name__ == "__main__":
    main()
