"""Download pinned model shards through short-lived HF URLs; verify SHA-256."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import sys
import urllib.request

REVISION="11c9b309abf73637e4b6f9a3fa1e92e615547819"


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--export-urls",action="store_true")
    p.add_argument("--urls",type=Path)
    p.add_argument("--output",type=Path)
    a=p.parse_args()
    os.environ["https_proxy"]="http://www-cache.rz.ruhr-uni-bochum.de:80"
    if a.export_urls:
        from huggingface_hub import hf_hub_url,get_hf_file_metadata,get_token
        data={}
        for i in range(1,5):
            name=f"model-{i:05d}-of-00004.safetensors"
            m=get_hf_file_metadata(hf_hub_url("google/gemma-2-9b-it",name,revision=REVISION),token=get_token())
            assert m.commit_hash==REVISION
            data[name]={"url":m.location,"size":m.size,"sha256":m.etag}
        sys.stdout.write(json.dumps(data))
        return
    data=json.loads(a.urls.read_text());a.output.mkdir(parents=True,exist_ok=True)
    def download(item):
        name,info=item
        out=a.output/(name+".incomplete")
        digest=hashlib.sha256()
        with urllib.request.urlopen(info["url"],timeout=90) as response,out.open("wb") as f:
            while chunk:=response.read(8*1024*1024): f.write(chunk);digest.update(chunk)
        assert out.stat().st_size==info["size"]
        assert digest.hexdigest()==info["sha256"]
        out.rename(a.output/name)
        print("Verified",name,flush=True)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(download,data.items()))
    except BaseException as exc:
        # Signed URLs are credentials: never include request objects or URL-bearing exceptions in logs.
        print("Download failed:",type(exc).__name__,flush=True)
        raise SystemExit(1)
    manifest={name:{k:v for k,v in info.items() if k!="url"} for name,info in data.items()}
    (a.output/"weights_verified.json").write_text(json.dumps({"revision":REVISION,"files":manifest},indent=2))
    a.urls.unlink()


if __name__=="__main__": main()
