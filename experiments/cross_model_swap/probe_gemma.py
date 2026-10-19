"""Check actual Gemma HF/vLLM injected continuation probabilities before training."""
import argparse
import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from nla.datagen.injection_tokens import build_token_meta
from nla.utils import build_prompt_text, register_karvonen_hook
from nla.train_rl_vllm import rollout_batch_vllm, sync_actor_to_vllm


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--teacher-receipt",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    templates=json.loads(a.teacher_receipt.read_text())["sources"][0]["sidecar"]["prompt_templates"]
    tok=AutoTokenizer.from_pretrained(a.checkpoint)
    meta=build_token_meta(tok,templates["actor"],templates["critic"])
    prompt=build_prompt_text([{"role":"user","content":templates["actor"].format(injection_char="<INJECT>")}],meta.injection_char,tok)
    vectors=torch.randn(2,3584)
    actor=AutoModelForCausalLM.from_pretrained(a.checkpoint,torch_dtype=torch.bfloat16,
        attn_implementation="eager").cuda().eval()
    vref=[None]
    register_karvonen_hook(actor,vref,meta.injection_token_id,meta.injection_left_neighbor_id,
                           meta.injection_right_neighbor_id,layer_idx=1)
    from vllm import LLM
    llm=LLM(model=a.checkpoint,tokenizer=a.checkpoint,dtype="bfloat16",gpu_memory_utilization=.24,
        max_model_len=512,tensor_parallel_size=1,enforce_eager=True,enable_prefix_caching=False)
    responses=rollout_batch_vllm(llm,tok,[(prompt,v) for v in vectors],meta.injection_token_id,
        1,32,1.0,left_id=meta.injection_left_neighbor_id,right_id=meta.injection_right_neighbor_id)
    results=[]
    for r in responses:
        assert r["steer_verified"]
        vref[0]=vectors[r["prompt_idx"]:r["prompt_idx"]+1].cuda()
        ids=r["full_ids"].unsqueeze(0).cuda()
        with torch.no_grad():
            logits=actor(input_ids=ids,use_cache=False).logits.float()
            lp=logits[0,r["prompt_len"]-1:-1].log_softmax(-1).gather(1,ids[0,r["prompt_len"]:,None]).squeeze(1).cpu()
        delta=(lp-r["old_logp"]).abs()
        results.append({"mean_abs_logp_difference":delta.mean().item(),"max_abs_logp_difference":delta.max().item(),
                        "steer_verified":r["steer_verified"],"tokens":r["n_resp"]})
        assert delta.mean()<.15,results
    from peft import LoraConfig,get_peft_model
    actor=get_peft_model(actor,LoraConfig(r=8,lora_alpha=16,use_rslora=True,
        target_modules=["q_proj","k_proj","v_proj","o_proj"],task_type="CAUSAL_LM",lora_dropout=0.0))
    with torch.no_grad():
        for name,param in actor.named_parameters():
            if "lora_B" in name: param.normal_(std=0.001)
    actor.eval()
    sync_seconds=sync_actor_to_vllm(actor,llm,ipc=True)
    synced=rollout_batch_vllm(llm,tok,[(prompt,v) for v in vectors],meta.injection_token_id,
        1,32,1.0,left_id=meta.injection_left_neighbor_id,right_id=meta.injection_right_neighbor_id)
    sync_results=[]
    for r in synced:
        assert r["steer_verified"]
        vref[0]=vectors[r["prompt_idx"]:r["prompt_idx"]+1].cuda()
        ids=r["full_ids"].unsqueeze(0).cuda()
        with torch.no_grad():
            logits=actor(input_ids=ids,use_cache=False).logits.float()
            lp=logits[0,r["prompt_len"]-1:-1].log_softmax(-1).gather(1,ids[0,r["prompt_len"]:,None]).squeeze(1).cpu()
        delta=(lp-r["old_logp"]).abs()
        sync_results.append({"mean_abs_logp_difference":delta.mean().item(),"max_abs_logp_difference":delta.max().item()})
        assert delta.mean()<.15,sync_results
    a.output.write_text(json.dumps({"passed":True,"attention":"eager","results":results,
        "ipc_weight_sync_verified":True,"sync_seconds":sync_seconds,"synced_results":sync_results},indent=2))
    print(a.output.read_text(),flush=True)


if __name__=="__main__": main()
