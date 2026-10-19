"""Durable, fail-fast extraction -> SFT -> merge -> eight-GPU RL controller."""
import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
PY=str(ROOT/"venv/bin/python")
BASE=str(ROOT/"checkpoints/gemma2-base")
CODE=ROOT/"EasyNLA"
ENV=dict(os.environ,OMP_NUM_THREADS="2",TOKENIZERS_PARALLELISM="false",HF_HUB_OFFLINE="1",PYTHONUNBUFFERED="1")
# EasyNLA's trusted local apply_model RPC transports its own Python functions.
ENV["VLLM_ALLOW_INSECURE_SERIALIZATION"]="1"
ENV.pop("PYTORCH_CUDA_ALLOC_CONF",None)
ENV.pop("PYTORCH_ALLOC_CONF",None)


def state(phase,**extra):
    record=dict(phase=phase,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**extra)
    tmp=ROOT/"status.tmp";tmp.write_text(json.dumps(record,indent=2));tmp.replace(ROOT/"status.json")
    print(json.dumps(record),flush=True)


def launch(name,cmd,gpus=None):
    env=ENV.copy()
    if gpus is not None: env["CUDA_VISIBLE_DEVICES"]=gpus
    (ROOT/"commands").mkdir(exist_ok=True)
    (ROOT/"commands"/(name+".json")).write_text(json.dumps({"argv":cmd,"gpus":gpus},indent=2))
    stream=(ROOT/"logs"/(name+".log")).open("a")
    return subprocess.Popen(cmd,cwd=CODE,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)


def wait_jobs(jobs):
    try:
        while True:
            codes=[job.poll() for job in jobs]
            if any(c not in (None,0) for c in codes): raise RuntimeError(f"Child failed: {codes}; see logs")
            if all(c==0 for c in codes): return
            time.sleep(10)
    except BaseException:
        for job in jobs:
            if job.poll() is None: os.killpg(job.pid,signal.SIGTERM)
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rl-steps",type=int,default=2100)
    a=p.parse_args()
    assert not (ROOT/"PIPELINE_STARTED").exists(),"Controller already started; inspect before resuming"
    (ROOT/"PIPELINE_STARTED").write_text(datetime.datetime.now(datetime.timezone.utc).isoformat())
    state("extracting_and_preflight")
    start=time.monotonic()
    while not all((ROOT/"shards"/f"rank{i}.parquet.json").exists() for i in range(8)):
        if (ROOT/"PREFLIGHT_FAILED").exists(): raise RuntimeError("Gemma or distributed SFT preflight failed; see logs")
        if time.monotonic()-start>4*3600: raise TimeoutError("Extraction did not finish within four hours")
        if (ROOT/"phase.txt").read_text().strip()=="extraction_failed": raise RuntimeError("Extraction failed")
        time.sleep(15)
    while not (ROOT/"gemma_probe.json").exists():
        if time.monotonic()-start>4*3600: raise TimeoutError("Gemma preflight did not pass")
        time.sleep(15)
    probe=json.loads((ROOT/"gemma_probe.json").read_text())
    assert probe["passed"] and probe["ipc_weight_sync_verified"]
    state("building_data")
    if not (ROOT/"data/manifest.json").exists():
        wait_jobs([launch("build_data",[PY,str(ROOT/"build_data.py"),"--shards",str(ROOT/"shards"),
            "--teacher-receipt",str(ROOT/"teacher_text_v2.parquet.json"),"--target-checkpoint",BASE,"--output",str(ROOT/"data")])])
    state("sft",layout="AV GPUs 0-3; AR GPUs 4-7",effective_batch=128)
    common=["--base-ckpt",BASE,"--attn-implementation","eager","--max-len","1024",
            "--heldout-rows","128","--heldout-every","250","--save-every","500","--seed","42","--no-wandb"]
    jobs=[]
    for mode,gpus,micro,accum in [("av","0,1,2,3",8,4),("ar","4,5,6,7",16,2)]:
        args=[PY,"-m","torch.distributed.run","--standalone","--nproc-per-node=4","-m","nla.train_sft",
            "--mode",mode,"--parquet",str(ROOT/"data"/(mode+".parquet")),"--save-dir",str(ROOT/("sft_"+mode)),
            "--heldout-parquet",str(ROOT/"data"/(mode+"_val.parquet")),"--batch-size",str(micro),
            "--gradient-accumulation-steps",str(accum)]+common
        if mode=="av": args += ["--use-lora","--lr","1e-4","--gradient-checkpointing"]
        else: args += ["--ar-num-layers","32","--lr","2e-5","--full-ft-dtype","fp32"]
        jobs.append(launch("sft_"+mode,args,gpus))
    wait_jobs(jobs)
    av=sorted((ROOT/"sft_av").glob("iter_*"))[-1]
    ar=sorted((ROOT/"sft_ar").glob("iter_*"))[-1]
    (ROOT/"warmstart_checkpoints.json").write_text(json.dumps({"av":str(av),"ar":str(ar)},indent=2))
    state("merging_verbalizer")
    merged=ROOT/"av_merged"
    wait_jobs([launch("merge",[PY,"scripts/merge_lora_to_hf.py","--mode","av","--base-ckpt",BASE,
        "--av-dir",str(av),"--ar-dir",str(ar),"--av-out",str(merged),"--ar-out",str(ROOT/"unused_ar_merge")],"0")])
    state("rl",updates=a.rl_steps,batch_prompts=256,group_size=8,planned_explanations=a.rl_steps*256*8)
    args=[PY,"-m","torch.distributed.run","--standalone","--nproc-per-node=8","-m","nla.train_rl_vllm",
        "--base-ckpt",BASE,"--av-ckpt",str(merged),"--av-adapter",str(av),"--ar-ckpt",str(ar),
        "--rl-parquet",str(ROOT/"data/rl.parquet"),"--eval-parquet",str(ROOT/"data/av_val.parquet"),
        "--sidecar",str(ROOT/"data/rl.parquet"),"--save-dir",str(ROOT/"rl"),
        "--num-steps",str(a.rl_steps),"--batch-prompts","256","--group-size","8","--max-new-tokens","256",
        "--attn-implementation","eager","--lr","1e-4","--critic-lr","8e-5","--train-critic",
        "--gradient-checkpointing","--logp-micro-batch","4","--critic-micro-batch","8",
        "--vllm-gpu-mem","0.30","--vllm-tp","1","--vllm-max-len","1024","--ipc-weight-sync",
        "--vllm-max-num-seqs","128","--vllm-max-batched-tokens","4096",
        "--evals","base_fve","--eval-every","50","--eval-n-prompts","128","--save-every","100",
        "--assert-param-sync","--seed","42","--no-wandb"]
    wait_jobs([launch("rl",args,"0,1,2,3,4,5,6,7")])
    assert (ROOT/"rl"/f"iter_{a.rl_steps:06d}"/"adapter_config.json").exists(),"Final actor checkpoint missing"
    assert int((ROOT/"rl/critic_latest/saved_at_step.txt").read_text())==a.rl_steps,"Final critic checkpoint missing"
    state("complete",updates=a.rl_steps,planned_explanations=a.rl_steps*2048)
    (ROOT/"COMPLETE").touch()


if __name__=="__main__":
    try: main()
    except BaseException as exc:
        state("failed",error=str(exc))
        raise
