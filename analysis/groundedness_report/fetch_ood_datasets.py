"""Archive and select the public HF/CSV data used by OOD replications."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
from typing import Any

import httpx

from .io import atomic_json, atomic_jsonl, sha256_file
from .protocol import load_evaluation_registry, load_protocol
from .sampling import sha256_rank
from .spec import PROTOCOL_SHA256


HUB_API = "https://huggingface.co/api/datasets"
ROWS_API = "https://datasets-server.huggingface.co/rows"
SIZE_API = "https://datasets-server.huggingface.co/size"
MMMLU_CONFIGS = (
    "AR_XY", "BN_BD", "DE_DE", "ES_LA", "FR_FR", "HI_IN", "ID_ID",
    "IT_IT", "JA_JP", "KO_KR", "PT_BR", "SW_KE", "YO_NG", "ZH_CN",
)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _gate_revision(client: httpx.Client, dataset: str, revision: str) -> None:
    response = client.get(f"{HUB_API}/{dataset}")
    response.raise_for_status()
    observed = response.json().get("sha")
    if observed != revision:
        raise RuntimeError(f"{dataset} main={observed}, frozen revision={revision}")


def _dataset_rows(
    client: httpx.Client,
    *,
    dataset: str,
    config: str,
    split: str,
    output_dir: Path,
) -> list[dict[str, Any]]:
    size_response = client.get(SIZE_API, params={"dataset": dataset})
    size_response.raise_for_status()
    entries = [
        row for row in size_response.json()["size"]["splits"]
        if row["config"] == config and row["split"] == split
    ]
    if len(entries) != 1:
        raise RuntimeError(f"could not resolve one split for {dataset}/{config}/{split}")
    total = int(entries[0]["num_rows"])
    rows = []
    response_meta = []
    response_dir = output_dir / "responses"
    response_dir.mkdir(parents=True, exist_ok=True)
    for offset in range(0, total, 100):
        length = min(100, total - offset)
        response = client.get(ROWS_API, params={
            "dataset": dataset,
            "config": config,
            "split": split,
            "offset": offset,
            "length": length,
        })
        response.raise_for_status()
        payload = response.json()
        returned = payload.get("rows") or []
        if len(returned) != length:
            raise RuntimeError(f"{dataset}/{config} offset {offset} returned {len(returned)}")
        destination = response_dir / f"{offset:06d}.json"
        atomic_json(destination, payload)
        response_meta.append({
            "offset": offset,
            "length": length,
            "url": str(response.request.url),
            "sha256": hashlib.sha256(_canonical(payload)).hexdigest(),
        })
        for within, wrapper in enumerate(returned):
            rows.append({"source_row": offset + within, **wrapper["row"]})
    atomic_json(output_dir / "fetch_meta.json", {
        "dataset": dataset,
        "config": config,
        "split": split,
        "total_rows": total,
        "responses": response_meta,
    })
    return rows


def _selected_indices(seed: int, namespace: str, total: int, count: int) -> list[int]:
    return sorted(sha256_rank(
        range(total), seed=seed, namespace=namespace, identity=str
    )[:count])


def _mmmlu_rows(
    client: httpx.Client, *, seed: int, dataset: str, output_dir: Path
) -> list[dict[str, Any]]:
    size_response = client.get(SIZE_API, params={"dataset": dataset})
    size_response.raise_for_status()
    sizes = {
        row["config"]: int(row["num_rows"])
        for row in size_response.json()["size"]["splits"]
        if row["split"] == "test" and row["config"] in MMMLU_CONFIGS
    }
    if set(sizes) != set(MMMLU_CONFIGS):
        raise RuntimeError("MMMLU config set differs from the frozen 14-language set")
    ranked_configs = sha256_rank(
        MMMLU_CONFIGS, seed=seed, namespace="mmmlu-language-allocation"
    )
    counts = {config: 4 if config in ranked_configs[:8] else 3 for config in MMMLU_CONFIGS}
    rows = []
    response_dir = output_dir / "responses"
    response_dir.mkdir(parents=True, exist_ok=True)
    response_meta = []
    for config in MMMLU_CONFIGS:
        indices = _selected_indices(
            seed, f"mmmlu-row:{config}", sizes[config], counts[config]
        )
        for index in indices:
            response = client.get(ROWS_API, params={
                "dataset": dataset,
                "config": config,
                "split": "test",
                "offset": index,
                "length": 1,
            })
            response.raise_for_status()
            payload = response.json()
            returned = payload.get("rows") or []
            if len(returned) != 1:
                raise RuntimeError(f"MMMLU {config} row {index} unavailable")
            destination = response_dir / f"{config}-{index:06d}.json"
            atomic_json(destination, payload)
            rows.append({"config": config, "source_row": index, **returned[0]["row"]})
            response_meta.append({
                "config": config,
                "offset": index,
                "url": str(response.request.url),
                "sha256": hashlib.sha256(_canonical(payload)).hexdigest(),
            })
    atomic_json(output_dir / "fetch_meta.json", {
        "dataset": dataset,
        "split": "test",
        "language_allocation": counts,
        "total_selected": len(rows),
        "responses": response_meta,
    })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()
    protocol = load_protocol()
    registry = load_evaluation_registry()
    families = {row["id"]: row for row in registry["families"]}
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = int(protocol["seed"])
    selections: dict[str, Any] = {}

    with httpx.Client(timeout=args.timeout, follow_redirects=True) as client:
        specs = (
            (
                "alignment_multiturn_deception",
                "PKU-Alignment/DeceptionBench",
                families["alignment_multiturn_deception"]["revision"],
                "default",
                "test",
                "deceptionbench",
            ),
            (
                "capability_swebench_verified",
                "princeton-nlp/SWE-bench_Verified",
                families["capability_swebench_verified"]["dataset_revision"],
                "default",
                "test",
                "swebench_verified",
            ),
        )
        for family_id, dataset, revision, config, split, name in specs:
            _gate_revision(client, dataset, revision)
            directory = output_dir / name
            rows = _dataset_rows(
                client,
                dataset=dataset,
                config=config,
                split=split,
                output_dir=directory,
            )
            selected = sha256_rank(
                rows,
                seed=seed,
                namespace=f"ood-task:{family_id}",
                identity=lambda row: str(
                    row.get("instance_id")
                    or row.get("id")
                    or row.get("source_row")
                ),
            )[:50]
            atomic_jsonl(directory / "selected.jsonl", selected)
            selections[family_id] = {
                "dataset": dataset,
                "revision": revision,
                "selected": len(selected),
                "selected_sha256": sha256_file(directory / "selected.jsonl"),
            }

        mmmlu_family = families["capability_mmmlu"]
        _gate_revision(client, mmmlu_family["source"].split("/datasets/", 1)[1], mmmlu_family["revision"])
        mmmlu_dir = output_dir / "mmmlu"
        mmmlu = _mmmlu_rows(
            client, seed=seed, dataset="openai/MMMLU", output_dir=mmmlu_dir
        )
        atomic_jsonl(mmmlu_dir / "selected.jsonl", mmmlu)
        selections["capability_mmmlu"] = {
            "dataset": "openai/MMMLU",
            "revision": mmmlu_family["revision"],
            "selected": len(mmmlu),
            "selected_sha256": sha256_file(mmmlu_dir / "selected.jsonl"),
        }

        strong_url = (
            "https://raw.githubusercontent.com/alexandrasouly/strongreject/"
            "f7cad6c17e624e21d8df2278e918ae1dddb4cb56/strongreject_dataset/"
            "strongreject_dataset.csv"
        )
        response = client.get(strong_url)
        response.raise_for_status()
        raw = response.content
        strong_dir = output_dir / "strongreject"
        strong_dir.mkdir(parents=True, exist_ok=True)
        raw_path = strong_dir / "strongreject_dataset.csv"
        raw_path.write_bytes(raw)
        parsed = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
        selected = sha256_rank(
            parsed,
            seed=seed,
            namespace="ood-task:appropriate-harmlessness-harmful",
            identity=lambda row: json.dumps(row, sort_keys=True),
        )[:25]
        atomic_jsonl(strong_dir / "selected.jsonl", selected)
        selections["alignment_appropriate_harmlessness_harmful"] = {
            "source": strong_url,
            "selected": len(selected),
            "source_sha256": hashlib.sha256(raw).hexdigest(),
            "selected_sha256": sha256_file(strong_dir / "selected.jsonl"),
        }

    atomic_json(output_dir / "selection_manifest.json", {
        "schema_version": 1,
        "protocol_sha256": PROTOCOL_SHA256,
        "seed": seed,
        "selections": selections,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
