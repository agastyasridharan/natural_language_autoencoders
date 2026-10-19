"""Build Gemma-tokenized sidecars around newly extracted Qwen targets."""
import argparse
import copy
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from transformers import AutoTokenizer, AutoConfig
from nla.datagen.injection_tokens import build_token_meta
from nla.config import load_nla_config


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shards",type=Path,required=True)
    p.add_argument("--world-size",type=int,default=8)
    p.add_argument("--teacher-receipt",type=Path,required=True)
    p.add_argument("--target-checkpoint",required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    receipt=json.loads(a.teacher_receipt.read_text())
    templates=receipt["sources"][0]["sidecar"]["prompt_templates"]
    target=AutoConfig.from_pretrained(a.target_checkpoint)
    assert target.hidden_size==3584 and target.model_type=="gemma2"
    tok=AutoTokenizer.from_pretrained(a.target_checkpoint)
    tokens=asdict(build_token_meta(tok,templates["actor"],templates["critic"]))
    schema=pa.schema([("prompt",pa.string()),("response",pa.string()),
        ("activation_vector",pa.list_(pa.float32(),3584)),("doc_id",pa.string()),
        ("n_raw_tokens",pa.int32()),("activation_layer",pa.int32()),
        ("detokenized_text_truncated",pa.string())])
    actor_schema = schema.set(0, pa.field("prompt", pa.list_(pa.struct([pa.field("role", pa.string()), pa.field("content", pa.string())]))))
    actor=templates["actor"].format(injection_char="<INJECT>")
    a.output.mkdir(parents=True)
    names=["av","ar","rl","av_val","ar_val"]
    counts=dict.fromkeys(names,0); docsets={n:set() for n in names}
    schemas={n: schema if n.startswith("ar") else actor_schema for n in names}
    writers={n:pq.ParquetWriter(a.output/f"{n}.parquet",schemas[n],compression="zstd") for n in names}
    try:
        for rank in range(a.world_size):
            f=a.shards/f"rank{rank}.parquet"
            summary=json.loads(Path(str(f)+".json").read_text())
            assert summary["complete"] and summary["source_block_index"]==20
            for batch in pq.ParquetFile(f).iter_batches(batch_size=1024):
                groups={n:[] for n in names}
                for r in batch.to_pylist():
                    destinations=["av_val","ar_val"] if r["role"]=="val" else [r["role"]]
                    for n in destinations:
                        prompt=templates["critic"].format(explanation=r["explanation"]) if n.startswith("ar") else [{"role":"user","content":actor}]
                        groups[n].append(dict(prompt=prompt,response="<explanation>\n"+r["explanation"]+"\n</explanation>",
                            activation_vector=r["activation_vector"],doc_id=r["doc_id"],n_raw_tokens=r["n_raw_tokens"],
                            activation_layer=20,detokenized_text_truncated=r["text"]))
                        docsets[n].add(r["doc_id"])
                for n,rows in groups.items():
                    if rows:
                        writers[n].write_table(pa.Table.from_pylist(rows,schema=schemas[n])); counts[n]+=len(rows)
    finally:
        for writer in writers.values(): writer.close()
    for i,n in enumerate(["av","ar","rl","av_val"]):
        for m in ["av","ar","rl","av_val"][i+1:]:
            assert not docsets[n]&docsets[m],f"Document leakage {n}/{m}"
    for n in names:
        assert counts[n]>0,n
        meta={"kind":"nla_dataset","schema_version":2,"stage":"ar_sft" if n.startswith("ar") else "rl" if n=="rl" else "av_sft",
            "dataset_id":f"qwen25-L20-gemma2-swap-{n}","row_count":counts[n],
            "extraction":{"base_model":"Qwen/Qwen2.5-7B-Instruct","d_model":3584,"layer_index":20,"norm":"none","mse_scale":"sqrt_d_model"},
            "nla":{"base_model":"google/gemma-2-9b-it","checkpoint":a.target_checkpoint,"model_type":"gemma2","reconstructor_num_layers":32,
                   "injection":"EasyNLA additive norm-match at block-index 1","input_adapter":"none; AV LoRA learns the interface"},
            "tokens":tokens,"prompt_templates":templates,"teacher_provenance":str(a.teacher_receipt)}
        f=a.output/f"{n}.parquet"
        Path(str(f)+".nla_meta.yaml").write_text(yaml.safe_dump(meta,sort_keys=False))
        load_nla_config(str(f),tok)
    (a.output/"manifest.json").write_text(json.dumps({"counts":counts,"document_counts":{n:len(s) for n,s in docsets.items()},
        "source_model":"Qwen/Qwen2.5-7B-Instruct","target_model":"google/gemma-2-9b-it","no_train_val_document_overlap":True},indent=2))
    print((a.output/"manifest.json").read_text())


if __name__ == "__main__": main()
