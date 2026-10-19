"""Run a small synchronous, crash-resumable OpenAI teacher-quality smoke."""

import argparse
import concurrent.futures
import hashlib
import json
import os
import time
from pathlib import Path

import httpx
import pyarrow as pa
import pyarrow.parquet as pq

from nla.datagen.stage2_api_explain import _extract_and_clean


def _atomic_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(tmp, path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--prompt-template", required=True)
    p.add_argument("--output-json", required=True)
    p.add_argument("--output-parquet", required=True)
    p.add_argument("--model", default="gpt-5-mini")
    p.add_argument("--api-key-env", default="OPENAI_API_KEY_2")
    p.add_argument("--max-rows", type=int, default=20)
    p.add_argument("--max-completion-tokens", type=int, default=300)
    p.add_argument("--reasoning-effort", default="minimal")
    p.add_argument("--verbosity", default="low")
    p.add_argument("--concurrency", type=int, default=5)
    p.add_argument("--input-price", type=float, default=0.25)
    p.add_argument("--cached-input-price", type=float, default=0.025)
    p.add_argument("--output-price", type=float, default=2.0)
    args = p.parse_args()

    assert args.max_rows > 0 and args.concurrency > 0
    key = os.environ.get(args.api_key_env)
    assert key, f"set {args.api_key_env}"
    template = Path(args.prompt_template).read_text()
    assert "{text}" in template

    table = pq.read_table(args.input)
    doc_ids = table.column("doc_id").to_pylist()
    chosen = []
    seen = set()
    for i, doc_id in enumerate(doc_ids):
        if doc_id in seen:
            continue
        seen.add(doc_id)
        chosen.append(i)
        if len(chosen) == args.max_rows:
            break
    assert len(chosen) == args.max_rows, (
        f"input has only {len(chosen)} unique documents, need {args.max_rows}"
    )
    selected = table.take(pa.array(chosen, type=pa.int64()))
    texts = selected.column("detokenized_text_truncated").to_pylist()
    prompts = [template.format(text=t) for t in texts]

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    prompt_sha = hashlib.sha256(template.encode()).hexdigest()
    artifact = {
        "config": {
            "input": str(Path(args.input).resolve()),
            "prompt_template": str(Path(args.prompt_template).resolve()),
            "prompt_sha256": prompt_sha,
            "model": args.model,
            "max_completion_tokens": args.max_completion_tokens,
            "reasoning_effort": args.reasoning_effort,
            "verbosity": args.verbosity,
            "prices_per_million_usd": {
                "input": args.input_price,
                "cached_input": args.cached_input_price,
                "output": args.output_price,
            },
        },
        "records": [],
    }
    if output_json.exists():
        existing = json.loads(output_json.read_text())
        assert existing["config"] == artifact["config"], (
            f"existing artifact config differs: {output_json}"
        )
        artifact = existing
    completed = {int(r["sample_index"]) for r in artifact["records"]}

    client = httpx.Client(
        base_url="https://api.openai.com/v1",
        headers={"Authorization": f"Bearer {key}"},
        timeout=180.0,
    )

    def request(i: int) -> dict:
        body = {
            "model": args.model,
            "messages": [{"role": "user", "content": prompts[i]}],
            "max_completion_tokens": args.max_completion_tokens,
            "reasoning_effort": args.reasoning_effort,
            "verbosity": args.verbosity,
        }
        last_error = None
        for attempt in range(5):
            try:
                response = client.post("/chat/completions", json=body)
                if response.status_code < 500 and response.status_code != 429:
                    return {
                        "sample_index": i,
                        "source_row": chosen[i],
                        "doc_id": doc_ids[chosen[i]],
                        "request": body,
                        "status_code": response.status_code,
                        "response": response.json(),
                    }
                last_error = f"HTTP {response.status_code}: {response.text[:500]}"
            except Exception as exc:  # preserve the final diagnostic in the artifact
                last_error = repr(exc)
            time.sleep(min(30, 2 ** attempt))
        return {
            "sample_index": i,
            "source_row": chosen[i],
            "doc_id": doc_ids[chosen[i]],
            "request": body,
            "status_code": None,
            "error": last_error,
        }

    pending = [i for i in range(len(chosen)) if i not in completed]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(request, i): i for i in pending}
        for future in concurrent.futures.as_completed(futures):
            record = future.result()
            artifact["records"].append(record)
            artifact["records"].sort(key=lambda r: int(r["sample_index"]))
            _atomic_json(output_json, artifact)
            print(
                f"saved {len(artifact['records'])}/{len(chosen)}: "
                f"sample={record['sample_index']} status={record['status_code']}",
                flush=True,
            )
    client.close()

    cleaned = []
    prompt_tokens = cached_tokens = completion_tokens = 0
    for record in artifact["records"]:
        body = record.get("response") or {}
        usage = body.get("usage") or {}
        prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
        completion_tokens += int(usage.get("completion_tokens", 0) or 0)
        cached_tokens += int(
            (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
        )
        choices = body.get("choices") or []
        raw = ((choices[0].get("message") or {}).get("content") if choices else None)
        cleaned.append(
            _extract_and_clean(raw, r"<analysis>\s*(.*?)\s*</analysis>")
            if isinstance(raw, str) else None
        )
    assert all(x is not None for x in cleaned), "some responses failed tag extraction"
    output_table = selected.append_column(
        "api_explanation", pa.array(cleaned, type=pa.string())
    )
    output_parquet = Path(args.output_parquet)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(output_table, output_parquet)

    uncached = max(0, prompt_tokens - cached_tokens)
    cost = (
        uncached * args.input_price
        + cached_tokens * args.cached_input_price
        + completion_tokens * args.output_price
    ) / 1_000_000
    artifact["usage"] = {
        "requests": len(artifact["records"]),
        "input_tokens": prompt_tokens,
        "cached_input_tokens": cached_tokens,
        "output_tokens": completion_tokens,
        "cost_usd": cost,
    }
    _atomic_json(output_json, artifact)
    print(json.dumps(artifact["usage"], indent=2))
    print(f"wrote {output_parquet}")


if __name__ == "__main__":
    main()
