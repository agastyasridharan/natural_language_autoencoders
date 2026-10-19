"""Build tiny Gemma fixtures for the real distributed SFT code paths."""
import json
from dataclasses import asdict
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml
from transformers import AutoTokenizer,Gemma2Config,Gemma2ForCausalLM
from nla.datagen.injection_tokens import build_token_meta

root=Path(__file__).resolve().parent
out=root/"tiny_smoke";out.mkdir(exist_ok=True)
tok=AutoTokenizer.from_pretrained(root/"checkpoints/gemma2-base")
cfg=Gemma2Config(vocab_size=len(tok),hidden_size=32,intermediate_size=64,num_hidden_layers=3,
    num_attention_heads=4,num_key_value_heads=2,head_dim=8,query_pre_attn_scalar=8,sliding_window=32)
torch.manual_seed(42)
Gemma2ForCausalLM(cfg).save_pretrained(out/"base");tok.save_pretrained(out/"base")
templates=json.loads((root/"teacher_text_v2.parquet.json").read_text())["sources"][0]["sidecar"]["prompt_templates"]
tokens=asdict(build_token_meta(tok,templates["actor"],templates["critic"]))
for mode in ["av","ar"]:
    rows=[]
    for i in range(16):
        explanation="The text discusses a small tree in a garden."
        prompt=templates["critic"].format(explanation=explanation) if mode=="ar" else [{"role":"user","content":templates["actor"].format(injection_char="<INJECT>")}]
        rows.append(dict(prompt=prompt,response="<explanation>\n"+explanation+"\n</explanation>",activation_vector=torch.randn(32).tolist()))
    f=out/(mode+".parquet");pq.write_table(pa.Table.from_pylist(rows),f)
    meta=dict(kind="nla_dataset",extraction=dict(d_model=32,layer_index=20,norm="none",mse_scale="sqrt_d_model"),
        nla=dict(model_type="gemma2",reconstructor_num_layers=3),tokens=tokens,prompt_templates=templates)
    Path(str(f)+".nla_meta.yaml").write_text(yaml.safe_dump(meta))
print("Tiny fixtures ready",flush=True)
