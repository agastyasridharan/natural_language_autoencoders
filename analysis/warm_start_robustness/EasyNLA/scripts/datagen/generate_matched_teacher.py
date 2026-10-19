"""Generate plausible teacher labels on exactly the implausible-retained keys.

The implausible teacher dropped a small number of malformed responses.  A fair
warm-start comparison must not let the two conditions train on different
activation rows, so this entry point treats the retained implausible parquet as
the authoritative key set and order.  It:

1. selects those exact rows from the original stage-1 split;
2. calls the configured teacher in crash-resumable chunks;
3. retries only formatting failures with an attempt-specific reminder;
4. refuses to publish until every reference key has a valid explanation; and
5. atomically publishes a parquet in reference order plus an audit manifest.

Raw OpenAI input/output/error/usage records remain in OpenAIBatchProvider's
content-addressed cache.  Per-attempt raw completions are also retained beside
the output, so formatting failures remain auditable even if the API cache is
moved independently.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from nla.datagen._common import load_class
from nla.datagen.sidecar import (
    NLAApiSummaryMeta,
    read_sidecar_local,
    serialize_sidecar,
)
from nla.datagen.stage2_api_explain import _MIN_FEATURES, _extract_and_clean
from nla.schema import sidecar_path_for


KEY_COLUMNS = ("doc_id", "n_raw_tokens", "activation_layer")
_RETRY_REMINDER = """

Formatting retry {round_number}: the previous answer could not be parsed. Return
only one complete <analysis>...</analysis> pair, with 2-3 nonempty feature lines
inside it. Do not add a preface, conclusion, second tag pair, or text outside the
tags. Close </analysis> well before the output limit. Preserve the original
semantic task exactly.
""".strip()


Key = tuple[str, int, int]


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    parent_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_text(path, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def _keys(table: pa.Table) -> list[Key]:
    cols = [table.column(name).to_pylist() for name in KEY_COLUMNS]
    return [
        (str(doc_id), int(n_raw_tokens), int(layer))
        for doc_id, n_raw_tokens, layer in zip(*cols, strict=True)
    ]


def _assert_unique(keys: Iterable[Key], context: str) -> list[Key]:
    materialized = list(keys)
    seen: set[Key] = set()
    duplicates: list[Key] = []
    for key in materialized:
        if key in seen and len(duplicates) < 5:
            duplicates.append(key)
        seen.add(key)
    if len(seen) != len(materialized):
        raise RuntimeError(
            f"{context} has {len(materialized) - len(seen)} duplicate activation keys; "
            f"examples={duplicates}"
        )
    return materialized


def _key_digest(keys: Iterable[Key]) -> str:
    digest = hashlib.sha256()
    for doc_id, n_raw_tokens, layer in keys:
        digest.update(
            json.dumps([doc_id, n_raw_tokens, layer], ensure_ascii=False, separators=(",", ":"))
            .encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _read_round_keys(path: Path) -> list[Key]:
    return [
        (str(row[0]), int(row[1]), int(row[2]))
        for row in (json.loads(line) for line in path.read_text().splitlines() if line)
    ]


def _write_round_keys(path: Path, keys: list[Key]) -> None:
    _atomic_text(
        path,
        "".join(json.dumps(list(key), ensure_ascii=False, separators=(",", ":")) + "\n" for key in keys),
    )


def _clean(raw: str | None, pattern: str) -> tuple[str | None, str | None]:
    if raw is None:
        return None, "provider_returned_none"
    cleaned = _extract_and_clean(raw, pattern)
    if cleaned is None:
        return None, "missing_or_incomplete_analysis_tags"
    if cleaned.count("\n\n") + 1 < _MIN_FEATURES:
        return None, f"fewer_than_{_MIN_FEATURES}_features"
    return cleaned, None


def _complete_with_budget_splits(provider: Any, prompts: list[str]) -> list[str | None]:
    """Split only a rejected, never-submitted Batch to reduce cap granularity."""
    try:
        return provider.complete(prompts)
    except RuntimeError as exc:
        if "OpenAI spend guard" not in str(exc) or len(prompts) <= 1:
            raise
        middle = len(prompts) // 2
        print(
            f"[matched-teacher] spend reservation rejected for {len(prompts)} prompts; "
            f"retrying as never-submitted {middle}+{len(prompts) - middle} sub-batches",
            flush=True,
        )
        return _complete_with_budget_splits(
            provider, prompts[:middle]
        ) + _complete_with_budget_splits(provider, prompts[middle:])


def _validate_chunk(path: Path, expected_keys: list[Key]) -> None:
    table = pq.read_table(path)
    got = _keys(table)
    if got != expected_keys:
        raise RuntimeError(
            f"attempt chunk key mismatch at {path}; expected digest "
            f"{_key_digest(expected_keys)}, got {_key_digest(got)}"
        )
    required = {"api_explanation", "raw_completion", "failure_reason"}
    missing = required - set(table.column_names)
    if missing:
        raise RuntimeError(f"attempt chunk {path} lacks columns {sorted(missing)}")


def _load_successes(attempts_dir: Path) -> dict[Key, str]:
    successes: dict[Key, str] = {}
    for chunk_path in sorted(attempts_dir.glob("round_*/chunk_*.parquet")):
        table = pq.read_table(chunk_path, columns=[*KEY_COLUMNS, "api_explanation"])
        keys = _keys(table)
        values = table.column("api_explanation").to_pylist()
        for key, value in zip(keys, values, strict=True):
            if value is None:
                continue
            value = str(value)
            previous = successes.get(key)
            if previous is not None and previous != value:
                raise RuntimeError(
                    f"multiple successful but different explanations for {key}; "
                    "attempt state is inconsistent"
                )
            successes[key] = value
    return successes


def _validate_existing_output(output: Path, reference_keys: list[Key]) -> bool:
    sidecar = sidecar_path_for(output)
    if not output.exists() and not sidecar.exists():
        return False
    if not output.exists() and sidecar.is_file():
        print(f"[matched-teacher] found sidecar-only interrupted publication; rebuilding {output}")
        return False
    if not output.is_file() or not sidecar.is_file():
        raise RuntimeError(
            f"partial final publication: output={output.exists()} sidecar={sidecar.exists()}"
        )
    table = pq.read_table(output, columns=[*KEY_COLUMNS, "api_explanation"])
    got = _assert_unique(_keys(table), f"existing output {output}")
    if got != reference_keys:
        raise RuntimeError(f"existing {output} does not match the reference key order")
    values = table.column("api_explanation").to_pylist()
    if any(not isinstance(v, str) or v.count("\n\n") + 1 < _MIN_FEATURES for v in values):
        raise RuntimeError(f"existing {output} contains invalid explanations")
    meta = read_sidecar_local(output)
    if meta.row_count != len(reference_keys):
        raise RuntimeError(
            f"existing sidecar row_count={meta.row_count}, expected {len(reference_keys)}"
        )
    print(f"[matched-teacher] verified existing complete output: {output}")
    return True


def _load_matched_raw(raw: Path, reference: Path) -> tuple[pa.Table, list[Key]]:
    raw_index_table = pq.read_table(raw, columns=[*KEY_COLUMNS, "detokenized_text_truncated"])
    reference_index_table = pq.read_table(
        reference, columns=[*KEY_COLUMNS, "detokenized_text_truncated"]
    )
    raw_keys = _assert_unique(_keys(raw_index_table), f"raw split {raw}")
    reference_keys = _assert_unique(_keys(reference_index_table), f"reference {reference}")
    raw_map = {key: i for i, key in enumerate(raw_keys)}
    missing = [key for key in reference_keys if key not in raw_map]
    if missing:
        raise RuntimeError(
            f"reference has {len(missing)} keys absent from raw split; examples={missing[:5]}"
        )

    raw_text = raw_index_table.column("detokenized_text_truncated").to_pylist()
    ref_text = reference_index_table.column("detokenized_text_truncated").to_pylist()
    indices = [raw_map[key] for key in reference_keys]
    mismatched_text = [
        key
        for key, raw_i, expected in zip(reference_keys, indices, ref_text, strict=True)
        if raw_text[raw_i] != expected
    ]
    if mismatched_text:
        raise RuntimeError(
            f"source text differs for {len(mismatched_text)} reference keys; "
            f"examples={mismatched_text[:5]}"
        )

    raw_full = pq.read_table(raw)
    matched = raw_full.take(pa.array(indices, type=pa.int64()))
    reference_activations = pq.read_table(reference, columns=["activation_vector"]).column(
        "activation_vector"
    )
    if not matched.column("activation_vector").equals(reference_activations):
        raise RuntimeError(
            "activation_vector values in the raw split differ from the retained "
            "implausible reference"
        )
    return matched, reference_keys


def _resolve_paths(config_path: Path, split: str) -> tuple[dict[str, Any], Path, Path, Path]:
    cfg = yaml.safe_load(config_path.read_text())
    stage2 = cfg["stage2"]
    input_dir = Path(cfg["input_dir"])
    output_dir = Path(cfg["output_dir"])
    reference_dir = Path(stage2["reference_dir"])
    raw = input_dir / "splits" / f"{split}_raw.parquet"
    reference = reference_dir / f"{split}_explained.parquet"
    output = output_dir / "splits" / f"{split}_explained.parquet"
    return cfg, raw, reference, output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", required=True, choices=("av_sft", "ar_sft"))
    parser.add_argument(
        "--format-retry-rounds",
        type=int,
        default=None,
        help="Override stage2.format_retry_rounds. May be increased on a later resume.",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg, raw, reference, output = _resolve_paths(config_path, args.split)
    for path in (raw, sidecar_path_for(raw), reference, sidecar_path_for(reference)):
        if not path.is_file():
            raise FileNotFoundError(path)

    matched_raw, reference_keys = _load_matched_raw(raw, reference)
    expected_rows = {"av_sft": 49_495, "ar_sft": 49_516}[args.split]
    if len(reference_keys) != expected_rows:
        raise RuntimeError(
            f"{args.split} reference has {len(reference_keys)} rows, expected the audited "
            f"implausible-retained count {expected_rows}"
        )
    if _validate_existing_output(output, reference_keys):
        return

    stage2 = cfg["stage2"]
    provider_cls_name = stage2["provider_cls"]
    provider_kwargs = dict(stage2.get("provider_kwargs") or {})
    prompt_path = Path(stage2["instruction_template_path"])
    if not prompt_path.is_absolute():
        prompt_path = config_path.parents[2] / prompt_path
    instruction = prompt_path.read_text()
    if "{text}" not in instruction:
        raise RuntimeError(f"teacher prompt {prompt_path} lacks {{text}} placeholder")
    response_pattern = stage2.get(
        "response_extract_pattern", r"<analysis>\s*(.*?)\s*</analysis>"
    )
    chunk_size = int(stage2["chunk_size"])
    retry_rounds = (
        int(args.format_retry_rounds)
        if args.format_retry_rounds is not None
        else int(stage2.get("format_retry_rounds", 0))
    )
    if retry_rounds < 0:
        raise ValueError("format retry rounds must be nonnegative")

    attempts_dir = Path(f"{output}.attempts")
    attempts_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = {
        "schema_version": 1,
        "config": str(config_path),
        "split": args.split,
        "raw": str(raw),
        "reference": str(reference),
        "reference_rows": len(reference_keys),
        "reference_key_sha256": _key_digest(reference_keys),
        "instruction_sha256": hashlib.sha256(instruction.encode("utf-8")).hexdigest(),
        "response_extract_pattern": response_pattern,
        "provider_cls": provider_cls_name,
        "provider_kwargs": provider_kwargs,
        "chunk_size": chunk_size,
    }
    fingerprint_path = attempts_dir / "fingerprint.json"
    if fingerprint_path.exists():
        existing = json.loads(fingerprint_path.read_text())
        if existing != fingerprint:
            raise RuntimeError(
                f"attempt directory fingerprint mismatch at {fingerprint_path}; "
                "refusing to mix teacher settings or activation rows"
            )
    else:
        _atomic_json(fingerprint_path, fingerprint)

    provider = load_class(provider_cls_name)(**provider_kwargs)
    texts = matched_raw.column("detokenized_text_truncated").to_pylist()
    key_to_index = {key: i for i, key in enumerate(reference_keys)}

    for round_number in range(retry_rounds + 1):
        successes = _load_successes(attempts_dir)
        missing = [key for key in reference_keys if key not in successes]
        if not missing:
            break
        round_dir = attempts_dir / f"round_{round_number:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        keys_path = round_dir / "keys.jsonl"
        if keys_path.exists():
            round_keys = _assert_unique(_read_round_keys(keys_path), str(keys_path))
            unexpected = [key for key in round_keys if key not in key_to_index]
            if unexpected:
                raise RuntimeError(f"{keys_path} contains non-reference keys: {unexpected[:5]}")
            still_missing = [key for key in round_keys if key not in successes]
            if still_missing != missing:
                raise RuntimeError(
                    f"resume state for round {round_number} is inconsistent: frozen round has "
                    f"{len(still_missing)} unresolved keys, aggregate state has {len(missing)}"
                )
        else:
            round_keys = missing
            _write_round_keys(keys_path, round_keys)

        print(
            f"[matched-teacher] {args.split} round {round_number}: "
            f"{len(round_keys)} keys ({len(missing)} currently unresolved)",
            flush=True,
        )
        reminder = "" if round_number == 0 else _RETRY_REMINDER.format(round_number=round_number)
        for start in range(0, len(round_keys), chunk_size):
            chunk_keys = round_keys[start : start + chunk_size]
            chunk_path = round_dir / f"chunk_{start:08d}.parquet"
            if chunk_path.exists():
                _validate_chunk(chunk_path, chunk_keys)
                continue
            prompts = []
            for key in chunk_keys:
                prompt = instruction.format(text=texts[key_to_index[key]])
                prompts.append(prompt if not reminder else f"{prompt}\n\n{reminder}")
            raw_completions = _complete_with_budget_splits(provider, prompts)
            if len(raw_completions) != len(prompts):
                raise RuntimeError(
                    f"provider returned {len(raw_completions)} responses for {len(prompts)} prompts"
                )
            cleaned: list[str | None] = []
            failures: list[str | None] = []
            for raw_completion in raw_completions:
                value, failure = _clean(raw_completion, response_pattern)
                cleaned.append(value)
                failures.append(failure)
            chunk_table = pa.table(
                {
                    "doc_id": [key[0] for key in chunk_keys],
                    "n_raw_tokens": pa.array([key[1] for key in chunk_keys], type=pa.int64()),
                    "activation_layer": pa.array([key[2] for key in chunk_keys], type=pa.int64()),
                    "api_explanation": pa.array(cleaned, type=pa.string()),
                    "raw_completion": pa.array(raw_completions, type=pa.string()),
                    "failure_reason": pa.array(failures, type=pa.string()),
                }
            )
            tmp = chunk_path.with_name(chunk_path.name + ".tmp")
            pq.write_table(chunk_table, tmp)
            with tmp.open("rb") as handle:
                os.fsync(handle.fileno())
            os.replace(tmp, chunk_path)
            round_fd = os.open(round_dir, os.O_RDONLY)
            try:
                os.fsync(round_fd)
            finally:
                os.close(round_fd)
            n_ok = sum(value is not None for value in cleaned)
            print(
                f"[matched-teacher] round={round_number} chunk={start} "
                f"valid={n_ok}/{len(chunk_keys)}",
                flush=True,
            )
        successes = _load_successes(attempts_dir)
        unresolved = [key for key in round_keys if key not in successes]
        _atomic_json(
            round_dir / "complete.json",
            {
                "round": round_number,
                "attempted": len(round_keys),
                "succeeded_this_or_earlier": len(round_keys) - len(unresolved),
                "unresolved": len(unresolved),
                "completed_at": datetime.now(timezone.utc).isoformat(),
            },
        )

    successes = _load_successes(attempts_dir)
    missing = [key for key in reference_keys if key not in successes]
    if missing:
        _write_round_keys(attempts_dir / "missing_keys.jsonl", missing)
        raise RuntimeError(
            f"{len(missing)} exact-match rows remain after {retry_rounds} formatting "
            f"retry rounds. Nothing was published. Re-run with a larger "
            f"--format-retry-rounds; prior successful rows and API records are durable."
        )

    final = matched_raw.append_column(
        "api_explanation",
        pa.array([successes[key] for key in reference_keys], type=pa.string()),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = output.with_name(output.name + ".tmp")
    pq.write_table(final, tmp_output, row_group_size=8192)
    with tmp_output.open("rb") as handle:
        os.fsync(handle.fileno())

    raw_meta = read_sidecar_local(raw)
    reference_meta = read_sidecar_local(reference)
    api_meta = NLAApiSummaryMeta(
        model=getattr(provider, "model", provider_cls_name),
        max_tokens=int(getattr(provider, "max_tokens", -1)),
        temperature=getattr(provider, "temperature", -1.0),
        instruction_prompt=instruction,
    )
    out_meta = replace(
        raw_meta,
        dataset_id=f"{raw_meta.dataset_id}__plausible_matched_explained",
        row_count=len(reference_keys),
        api_summaries=api_meta,
        parent_datasets=[raw_meta.dataset_id, reference_meta.dataset_id],
        created_by="scripts.datagen.generate_matched_teacher",
        created_at="",
        git_commit="",
    )
    sidecar = sidecar_path_for(output)
    # Publish the sidecar first and parquet last. Consumers require both; if the
    # process dies between these atomic renames, a restart recognizes the
    # sidecar-only state as incomplete and safely republishes. The inverse
    # order could expose a complete-looking parquet with no contract.
    _atomic_text(sidecar, serialize_sidecar(out_meta))
    os.replace(tmp_output, output)
    output_fd = os.open(output.parent, os.O_RDONLY)
    try:
        os.fsync(output_fd)
    finally:
        os.close(output_fd)

    round_reports = []
    for report_path in sorted(attempts_dir.glob("round_*/complete.json")):
        round_reports.append(json.loads(report_path.read_text()))
    manifest = {
        "schema_version": 1,
        "condition": "plausible",
        "split": args.split,
        "raw": str(raw),
        "implausible_retained_reference": str(reference),
        "output": str(output),
        "row_count": len(reference_keys),
        "key_columns": list(KEY_COLUMNS),
        "ordered_key_sha256": _key_digest(reference_keys),
        "provider_cls": provider_cls_name,
        "provider_kwargs": provider_kwargs,
        "instruction_sha256": hashlib.sha256(instruction.encode("utf-8")).hexdigest(),
        "attempt_rounds": round_reports,
        "published_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_json(Path(f"{output}.match_manifest.json"), manifest)
    print(
        f"[matched-teacher] published {len(reference_keys)} exact matched rows -> {output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
