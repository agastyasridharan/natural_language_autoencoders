"""Record the deployed experiment code and dependency versions without credentials."""
import hashlib
import importlib.metadata
import json
from pathlib import Path

root=Path(__file__).resolve().parent
files=list(root.glob("*.py"))+list(root.glob("*.sh"))
files+=list((root/"EasyNLA/nla").rglob("*.py"))
files+=list((root/"EasyNLA/utils").rglob("*.py"))
manifest={
    "easy_nla_commit":"4d728477960c18cdfa36dc04ec738d7f55af9f0b",
    "source":{"model":"Qwen/Qwen2.5-7B-Instruct","revision":"a09a35458c702b33eeacc393d103063234e8bc28","block_index":20},
    "nla":{"model":"google/gemma-2-9b-it","revision":"11c9b309abf73637e4b6f9a3fa1e92e615547819","reconstructor_blocks":32},
    "recipe":{"av":"rsLoRA rank128 alpha16","ar":"full fine-tuning","sft_effective_batch":128,
        "rl_updates":2100,"rl_batch_prompts":256,"rl_group_size":8,"max_new_tokens":256,
        "injection":"additive norm-matched block1 output","attention":"eager; Gemma soft-capping preserved"},
    "dependencies":{name:importlib.metadata.version(name) for name in ["torch","transformers","vllm","vllm-lens","peft","bitsandbytes","pyarrow"]},
    "code_sha256":{str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(files)},
}
with (root/"teacher_text_v2.parquet").open("rb") as f:
    manifest["teacher_text_sha256"]=hashlib.file_digest(f,"sha256").hexdigest()
(root/"run_manifest.json").write_text(json.dumps(manifest,indent=2))
print("Run provenance saved",flush=True)
