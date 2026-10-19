"""GPU measurement phases for the report-faithful cycle-consistency test.

The phases are intentionally separate:

``ar`` reconstructs full, claim-deleted, and decoy-spliced explanations and
caches report-exact directions and reconstruction errors.

``av`` loads only the verbalizer, directly injects ``alpha * h`` (including the
non-renormalized projected vectors), and records length-normalized causal
readings.  Run calibration without ``--calibration`` to measure every text-valid
decoy.  Run evaluation with a frozen calibration file to avoid spending AV
forwards on decoys that cannot pass the preregistered filters.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
_ATHENA = _ROOT / "analysis" / "mimic_audit" / "athena"
_REPL = _ATHENA / "repl"
for _path in (_ROOT, _HERE, _ATHENA, _REPL):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from analysis.groundedness_report.io import (  # noqa: E402
    atomic_json,
    load_calibration,
    read_json,
    sha256_file,
)
from analysis.groundedness_report.protocol import load_protocol  # noqa: E402
from analysis.groundedness_report.spec import (                                  # noqa: E402
    PROTOCOL_SHA256,
    RECONSTRUCTION_ERROR_FORMULA,
    RECONSTRUCTION_METRIC_AMENDMENT_SHA256,
    REPORT_SHA256,
    SCHEMA_VERSION,
    TEXT_MODEL_CORRECTION_AMENDMENT_SHA256,
    injection_vector,
    project_out,
    reconstruction_error,
    select_decoys,
    unit,
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _json_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _write_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez(temporary, **arrays)
    os.replace(temporary, path)


def _split_indices(split_path: str | Path, partition: str) -> list[int]:
    split = read_json(split_path)
    key = f"{partition}_indices"
    if key not in split:
        raise ValueError(f"split has no {key}")
    values = sorted(set(int(x) for x in split[key]))
    if not values:
        raise ValueError(f"split partition {partition!r} is empty")
    return values


def _load_run_records(runs_dir: Path, run_id: str) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    with (runs_dir / run_id / "records.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                records[int(row["idx"])] = row
    return records


def _decoy_cache_files(decoys_dir: Path, run_id: str, model_key: str) -> list[tuple[int, int, Path]]:
    outputs = []
    for path in sorted((decoys_dir / run_id).glob(f"*_*.{model_key}.json")):
        idx_text, claim_text = path.name.split(".", 1)[0].split("_")
        outputs.append((int(idx_text), int(claim_text), path))
    return sorted(outputs)


def _claim_sample_rows(args: argparse.Namespace) -> dict[tuple[int, int], dict[str, Any]]:
    sample = read_json(args.claim_sample)
    if sample.get("protocol_sha256") != PROTOCOL_SHA256:
        raise ValueError("claim sample was frozen against a different preregistration")
    if sample.get(
        "text_model_correction_amendment_sha256"
    ) != TEXT_MODEL_CORRECTION_AMENDMENT_SHA256:
        raise ValueError("claim sample predates the Terra correction")
    rows = {
        (int(row["idx"]), int(row["claim_i"])): dict(row)
        for row in sample.get("rows") or []
        if row.get("partition") == args.partition
    }
    if len(rows) != sum(
        row.get("partition") == args.partition for row in sample.get("rows") or []
    ):
        raise ValueError("claim sample contains duplicate keys")
    return rows


def _text_pass_candidates(cache: dict[str, Any], explanation: str | None = None) -> list[int]:
    indices = [
        index
        for index, candidate in enumerate(cache.get("candidates") or [])
        if candidate.get("kept")
        and candidate.get("absence") == "absent"
        and candidate.get("natural_pass")
    ]
    if explanation is None:
        return indices
    folded = explanation.casefold()
    return [
        index
        for index in indices
        if str(cache["candidates"][index]["text"]).strip().casefold() not in folded
    ]


def _work_by_activation(
    args: argparse.Namespace,
) -> tuple[dict[int, dict], dict[int, list[tuple[int, Path | None]]]]:
    records = _load_run_records(Path(args.runs_dir), args.run)
    caches = _decoy_cache_files(Path(args.decoys_dir), args.run, args.decoy_model)
    split = read_json(args.split)
    calibration_indices = set(int(x) for x in split.get("calibration_indices") or [])
    evaluation_indices = set(int(x) for x in split.get("evaluation_indices") or [])
    evaluation_only = split.get("split_kind") == "evaluation_only"
    if (
        not evaluation_indices
        or (not evaluation_only and not calibration_indices)
        or (evaluation_only and calibration_indices)
        or calibration_indices & evaluation_indices
    ):
        raise ValueError("split partitions violate the declared split kind")
    calibration_docs = {
        str(records[idx].get("doc_id")) for idx in calibration_indices if idx in records
    }
    evaluation_docs = {
        str(records[idx].get("doc_id")) for idx in evaluation_indices if idx in records
    }
    if calibration_docs & evaluation_docs:
        raise ValueError("document leakage between calibration and evaluation indices")
    allowed = calibration_indices if args.partition == "calibration" else evaluation_indices
    cache_by_claim: dict[tuple[int, int], Path] = {}
    for idx, claim_i, path in caches:
        cache_by_claim[(idx, claim_i)] = path
    indices = sorted(
        idx
        for idx in allowed
        if idx in records and records[idx].get("status") == "ok"
    )
    if args.idx:
        requested = set(args.idx)
        indices = [idx for idx in indices if idx in requested]
    if args.limit:
        indices = indices[: args.limit]
    if not indices:
        raise ValueError("no valid activations intersect the requested split")
    sample_rows = _claim_sample_rows(args)
    selected_keys = sorted(
        key for key in sample_rows if key[0] in indices
    )
    if not selected_keys:
        raise ValueError("claim sample has no rows in the requested partition")
    missing_required = [key for key in selected_keys if key not in cache_by_claim]
    if missing_required:
        raise ValueError(
            f"missing decoy caches for {len(missing_required)} sampled claims; "
            f"first keys: {missing_required[:10]}. "
            "Run the text pipeline for the frozen claim sample before GPU measurement."
        )
    by_activation: dict[int, list[tuple[int, Path | None]]] = {}
    for idx, claim_i in selected_keys:
        by_activation.setdefault(idx, []).append((claim_i, cache_by_claim[(idx, claim_i)]))
    return records, by_activation


def _load_gold(args: argparse.Namespace) -> tuple[np.ndarray, list[dict[str, Any]]]:
    directory = Path(args.gold_dir)
    gold = np.load(directory / "gold.npy", mmap_mode="r")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if gold.ndim != 2 or len(manifest) != gold.shape[0]:
        raise ValueError(f"gold/manifest mismatch: {gold.shape} versus {len(manifest)}")
    return gold, manifest


def _assert_claim_tiling(
    z_inner: str,
    claims: list[dict[str, Any]],
    connectives: list[dict[str, Any]],
) -> None:
    """Require the report's exact, flat claim/connective character partition."""
    segments = sorted([*claims, *connectives], key=lambda row: int(row["char_span"][0]))
    if not segments:
        raise ValueError("segmentation is empty")
    previous = 0
    for segment in segments:
        start, end = (int(x) for x in segment["char_span"])
        if start != previous or end <= start or end > len(z_inner):
            raise ValueError(f"segmentation does not exactly tile at {(start, end)} after {previous}")
        if z_inner[start:end] != segment["text"]:
            raise ValueError(f"segment text/span drift at {(start, end)}")
        previous = end
    if previous != len(z_inner):
        raise ValueError(
            f"segmentation does not exactly tile: stops at {previous}, explanation has {len(z_inner)} characters"
        )


def _direction(ar_with: np.ndarray, ar_without: np.ndarray) -> np.ndarray:
    try:
        return unit(unit(ar_with) - unit(ar_without)).astype(np.float32)
    except ValueError:
        # Degenerate AR contrasts are recorded and later become unresolved; one
        # bad claim must not abort a multi-hour model run.
        return np.full(np.asarray(ar_with).shape, np.nan, dtype=np.float32)


def _reconstruct_numpy(critic: Any, text: str) -> np.ndarray:
    return critic.reconstruct(text).detach().float().cpu().numpy().astype(np.float32)


class ScanningEmbeddingInjector:
    """Embedding hook that locates the sidecar marker from each live input."""

    def __init__(self, model: Any, sidecar: dict[str, Any]) -> None:
        self.v = None
        self.inj_id = int(sidecar["inj_id"])
        self.left_id = int(sidecar["left"])
        self.right_id = int(sidecar["right"])
        model.get_input_embeddings().register_forward_hook(self._hook)

    def marker_position(self, input_ids: Any) -> int:
        matches = []
        seq_len = input_ids.shape[-1]
        for batch_index, position in (input_ids == self.inj_id).nonzero().tolist():
            if position in {0, seq_len - 1}:
                continue
            if (
                int(input_ids[batch_index, position - 1]) == self.left_id
                and int(input_ids[batch_index, position + 1]) == self.right_id
            ):
                matches.append((batch_index, position))
        if len(matches) != input_ids.shape[0]:
            raise RuntimeError(
                f"found {len(matches)} valid injection sites for batch {input_ids.shape[0]}"
            )
        if input_ids.shape[0] != 1:
            raise RuntimeError("report measurement requires batch size one")
        return int(matches[0][1])

    def _hook(self, _module: Any, inputs: tuple[Any, ...], output: Any) -> Any:
        if self.v is None:
            return output
        if len(inputs) != 1:
            raise RuntimeError("embedding injector expected one positional input_ids tensor")
        input_ids = inputs[0]
        position = self.marker_position(input_ids)
        injected = output.clone()
        injected[0, position] = self.v.to(injected.device, injected.dtype)
        return injected


def run_ar(args: argparse.Namespace) -> None:
    from huggingface_hub import snapshot_download

    from nla_inference import NLACritic
    from repl_common import MODELS
    import cycle_consistency as old_cycle

    records, by_activation = _work_by_activation(args)
    gold, _ = _load_gold(args)
    cfg = MODELS[args.key]
    model_spec = load_protocol()["models"]["reconstructor"]
    if cfg["ar"] != model_spec["repo_id"]:
        raise ValueError("AR repository differs from the frozen protocol")
    ar_dir = snapshot_download(
        repo_id=model_spec["repo_id"],
        revision=model_spec["revision"],
        local_files_only=True,
    )
    critic = NLACritic(ar_dir, device=args.device)
    out = Path(args.out) / "ar" / args.run
    activation_dir = out / "activations"
    claim_dir = out / "claims"
    status_path = out / "status.json"
    started = time.time()
    done = 0

    for position, idx in enumerate(by_activation, start=1):
        record = records[idx]
        z_inner = str(record["z_inner"])
        claims = list(record["snippet_results"])
        _assert_claim_tiling(z_inner, claims, list(record.get("connective_results") or []))
        h = unit(np.asarray(gold[idx], dtype=np.float64))
        h_sha256 = hashlib.sha256(h.astype(np.float32).tobytes()).hexdigest()
        activation_npz = activation_dir / f"{idx}.npz"
        activation_json = activation_dir / f"{idx}.json"
        z_sha256 = _sha(z_inner)
        activation_pair = (activation_json.exists(), activation_npz.exists())
        if activation_pair[0] != activation_pair[1]:
            raise ValueError(f"incomplete activation cache for {idx}; remove it or rerun with --force")
        activation_current = False
        if all(activation_pair):
            meta = read_json(activation_json)
            activation_current = (
                meta.get("schema_version") == SCHEMA_VERSION
                and meta.get("report_sha256") == REPORT_SHA256
                and meta.get("text_model_correction_amendment_sha256")
                == TEXT_MODEL_CORRECTION_AMENDMENT_SHA256
                and meta.get("key") == args.key
                and meta.get("run") == args.run
                and meta.get("z_sha256") == z_sha256
                and meta.get("h_sha256") == h_sha256
                and meta.get("n_claims") == len(claims)
            )
            if not activation_current and not args.force:
                raise ValueError(f"stale activation AR cache for {idx}; rerun with --force")

        claim_inputs: dict[int, tuple[dict[str, Any], list[int], list[str], str, bool]] = {}
        for claim_i, cache_path in by_activation[idx]:
            if cache_path is None:
                continue
            cache = read_json(cache_path)
            if cache.get(
                "text_model_correction_amendment_sha256"
            ) != TEXT_MODEL_CORRECTION_AMENDMENT_SHA256:
                raise ValueError(f"decoy cache for {idx}_{claim_i} predates Terra")
            claim = claims[claim_i]
            if cache["claim"]["text"] != claim["text"]:
                raise ValueError(f"claim drift for {idx}_{claim_i}")
            if cache.get("prompt_version") != args.prompt_version:
                raise ValueError(f"stale decoy cache for {idx}_{claim_i}: {cache.get('prompt_version')}")
            kept = _text_pass_candidates(cache, z_inner)
            texts = [str(cache["candidates"][candidate_i]["text"]) for candidate_i in kept]
            cache_sha256 = sha256_file(cache_path)
            stem = claim_dir / f"{idx}_{claim_i}"
            pair = (stem.with_suffix(".json").exists(), stem.with_suffix(".npz").exists())
            if pair[0] != pair[1]:
                raise ValueError(f"incomplete claim AR cache for {idx}_{claim_i}; rerun with --force")
            current = False
            if all(pair):
                meta = read_json(stem.with_suffix(".json"))
                current = (
                    meta.get("schema_version") == SCHEMA_VERSION
                    and meta.get("report_sha256") == REPORT_SHA256
                    and meta.get("reconstruction_metric_amendment_sha256")
                    == RECONSTRUCTION_METRIC_AMENDMENT_SHA256
                    and meta.get("text_model_correction_amendment_sha256")
                    == TEXT_MODEL_CORRECTION_AMENDMENT_SHA256
                    and meta.get("reconstruction_error_formula")
                    == RECONSTRUCTION_ERROR_FORMULA
                    and meta.get("key") == args.key
                    and meta.get("run") == args.run
                    and meta.get("z_sha256") == z_sha256
                    and meta.get("h_sha256") == h_sha256
                    and meta.get("claim_text_sha256") == _sha(str(claim["text"]))
                    and meta.get("prompt_version") == args.prompt_version
                    and meta.get("decoy_cache_sha256") == cache_sha256
                    and meta.get("candidate_indices") == kept
                    and meta.get("candidate_texts") == texts
                )
                if not current and not args.force:
                    raise ValueError(f"stale claim AR cache for {idx}_{claim_i}; rerun with --force")
            claim_inputs[claim_i] = (cache, kept, texts, cache_sha256, current)

        if not args.force and activation_current and all(value[-1] for value in claim_inputs.values()):
            continue

        ar_full = _reconstruct_numpy(critic, z_inner)
        real_directions = np.zeros((len(claims), ar_full.size), dtype=np.float32)
        ar_without: dict[int, np.ndarray] = {}
        for claim_i, claim in enumerate(claims):
            start, end = (int(x) for x in claim["char_span"])
            z_without = z_inner[:start] + z_inner[end:]
            reconstructed = _reconstruct_numpy(critic, z_without)
            ar_without[claim_i] = reconstructed
            real_directions[claim_i] = _direction(ar_full, reconstructed)
        if activation_current and not args.force:
            stored_directions = np.asarray(np.load(activation_npz)["real_directions"], dtype=np.float32)
            if stored_directions.shape != real_directions.shape:
                raise ValueError(f"stored AR direction shape drift for activation {idx}")
            comparable = np.isfinite(stored_directions) & np.isfinite(real_directions)
            drift = float(np.max(np.abs(stored_directions[comparable] - real_directions[comparable]))) \
                if comparable.any() else 0.0
            if drift > 1e-3 or not np.array_equal(np.isnan(stored_directions), np.isnan(real_directions)):
                raise ValueError(f"AR direction drift for activation {idx}: max {drift:.6g}")
            real_directions = stored_directions
        if args.force or not activation_current:
            _write_npz(
                activation_npz,
                real_directions=real_directions,
                ar_full=ar_full,
                ar_without=np.stack([ar_without[i] for i in range(len(claims))]),
            )
            atomic_json(activation_json, {
                "schema_version": SCHEMA_VERSION,
                "report_sha256": REPORT_SHA256,
                "reconstruction_metric_amendment_sha256": (
                    RECONSTRUCTION_METRIC_AMENDMENT_SHA256
                ),
                "text_model_correction_amendment_sha256": (
                    TEXT_MODEL_CORRECTION_AMENDMENT_SHA256
                ),
                "key": args.key,
                "run": args.run,
                "idx": idx,
                "z_sha256": z_sha256,
                "h_sha256": h_sha256,
                "n_claims": len(claims),
                "direction_formula": "unit(unit(AR(z))-unit(AR(z_without_claim)))",
            })

        for claim_i, cache_path in by_activation[idx]:
            if cache_path is None:
                continue
            cache, kept, texts, cache_sha256, current = claim_inputs[claim_i]
            if current and not args.force:
                continue
            claim = claims[claim_i]
            start, end = (int(x) for x in claim["char_span"])
            reconstructed_without = ar_without[claim_i]
            decoy_directions = np.zeros((len(texts), ar_full.size), dtype=np.float32)
            decoy_reconstructions = np.zeros((len(texts), ar_full.size), dtype=np.float32)
            signed_overlaps = np.full(len(texts), np.nan, dtype=np.float64)
            errors = np.full(len(texts), np.nan, dtype=np.float64)
            z_star_hashes: list[str] = []
            for decoy_index, text in enumerate(texts):
                z_star, _ = old_cycle.splice(z_inner, (start, end), text)
                reconstructed_star = _reconstruct_numpy(critic, z_star)
                decoy_reconstructions[decoy_index] = reconstructed_star
                direction = _direction(reconstructed_star, reconstructed_without)
                decoy_directions[decoy_index] = direction
                signed_overlaps[decoy_index] = float(h @ direction) if np.all(np.isfinite(direction)) else np.nan
                errors[decoy_index] = reconstruction_error(h, reconstructed_star)
                z_star_hashes.append(_sha(z_star))
            real_direction = real_directions[claim_i]
            stem = claim_dir / f"{idx}_{claim_i}"
            _write_npz(
                stem.with_suffix(".npz"),
                real_direction=real_direction,
                decoy_directions=decoy_directions,
                decoy_reconstructions=decoy_reconstructions,
                decoy_overlaps=signed_overlaps,
                decoy_reconstruction_errors=errors,
            )
            atomic_json(stem.with_suffix(".json"), {
                "schema_version": SCHEMA_VERSION,
                "report_sha256": REPORT_SHA256,
                "reconstruction_metric_amendment_sha256": (
                    RECONSTRUCTION_METRIC_AMENDMENT_SHA256
                ),
                "text_model_correction_amendment_sha256": (
                    TEXT_MODEL_CORRECTION_AMENDMENT_SHA256
                ),
                "key": args.key,
                "run": args.run,
                "idx": idx,
                "claim_i": claim_i,
                "z_sha256": z_sha256,
                "h_sha256": h_sha256,
                "claim_text_sha256": _sha(str(claim["text"])),
                "prompt_version": args.prompt_version,
                "decoy_cache_sha256": cache_sha256,
                "candidate_indices": kept,
                "candidate_texts": texts,
                "z_star_sha256": z_star_hashes,
                "n_generated": len(cache.get("candidates") or []),
                "n_text_pass": len(kept),
                "real_overlap": float(h @ real_direction) if np.all(np.isfinite(real_direction)) else None,
                "real_reconstruction_error": reconstruction_error(h, ar_full),
                "reconstruction_error_formula": RECONSTRUCTION_ERROR_FORMULA,
            })
            done += 1
        atomic_json(status_path, {
            "phase": "ar",
            "status": "running",
            "key": args.key,
            "run": args.run,
            "partition": args.partition,
            "reconstruction_metric_amendment_sha256": (
                RECONSTRUCTION_METRIC_AMENDMENT_SHA256
            ),
            "text_model_correction_amendment_sha256": (
                TEXT_MODEL_CORRECTION_AMENDMENT_SHA256
            ),
            "activations_done": position,
            "activations_total": len(by_activation),
            "claims_written": done,
            "elapsed_seconds": round(time.time() - started, 1),
        })
        print(f"[ar {position}/{len(by_activation)}] idx={idx} claims={len(by_activation[idx])}", flush=True)
    atomic_json(status_path, {
        "phase": "ar",
        "status": "done",
        "key": args.key,
        "run": args.run,
        "partition": args.partition,
        "reconstruction_metric_amendment_sha256": RECONSTRUCTION_METRIC_AMENDMENT_SHA256,
        "text_model_correction_amendment_sha256": TEXT_MODEL_CORRECTION_AMENDMENT_SHA256,
        "activations_total": len(by_activation),
        "claims_written": done,
        "elapsed_seconds": round(time.time() - started, 1),
    })


def _direct_mean_scores(
    model: Any,
    injector: Any,
    input_ids: Any,
    vector: np.ndarray,
    scale: float,
    ranges: list[Any],
) -> tuple[list[float], np.ndarray]:
    """Teacher-forced mean scores with direct, non-renormalizing injection."""
    import torch
    import claim_ablation as old_claim

    injected = injection_vector(vector, scale).astype(np.float32)
    injector.v = torch.as_tensor(injected)
    token_logp = old_claim._token_logp(model, input_ids)
    sums = old_claim._span_sums(token_logp, ranges)
    means = []
    for value, token_range in zip(sums, ranges):
        if token_range is None or not np.isfinite(value):
            means.append(float("nan"))
        else:
            lo, hi = token_range
            means.append(float(value) / (hi - lo) if hi > lo else float("nan"))
    return means, token_logp.detach().float().cpu().numpy()


def _reference_activation(path: str | Path, width: int) -> tuple[np.ndarray, dict[str, Any]]:
    source = Path(path)
    value = np.asarray(np.load(source), dtype=np.float32)
    if value.shape != (width,) or not np.all(np.isfinite(value)):
        raise ValueError(f"reference activation must be finite [{width}], got {value.shape}")
    norm = float(np.linalg.norm(value))
    if abs(norm - 1.0) > 1e-4:
        raise ValueError(f"reference activation must already be unit normalized, got norm {norm}")
    metadata_path = source.with_suffix(".json")
    metadata = read_json(metadata_path) if metadata_path.exists() else {
        "kind": "externally_supplied_fixed_reference",
        "source": str(source),
    }
    metadata["sha256"] = hashlib.sha256(value.tobytes()).hexdigest()
    return value, metadata


def run_av(args: argparse.Namespace) -> None:
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    import claim_ablation as old_claim
    import cycle_consistency as old_cycle
    from avgen_model import load_av, read_sidecar
    from repl_common import MODELS

    records, by_activation = _work_by_activation(args)
    claim_sample_rows = _claim_sample_rows(args)
    gold, _ = _load_gold(args)
    negative_map: dict[int, int] | None = None
    negative_map_sha256: str | None = None
    if args.negative_map:
        negative_artifact = read_json(args.negative_map)
        if negative_artifact.get("protocol_sha256") != PROTOCOL_SHA256:
            raise ValueError("negative-control map has a different protocol hash")
        negative_map = {
            int(source): int(donor)
            for source, donor in (negative_artifact.get("mapping") or {}).items()
        }
        missing = sorted(set(by_activation) - set(negative_map))
        fixed = sorted(idx for idx in by_activation if negative_map.get(idx) == idx)
        invalid = sorted(
            donor for donor in negative_map.values() if donor < 0 or donor >= len(gold)
        )
        if missing or fixed or invalid:
            raise ValueError(
                f"invalid negative-control map: missing={missing[:5]} "
                f"fixed={fixed[:5]} invalid={invalid[:5]}"
            )
        negative_map_sha256 = sha256_file(args.negative_map)
    reference, reference_metadata = _reference_activation(args.reference_vector, gold.shape[1])
    calibration = load_calibration(args.calibration) if args.calibration else None
    split_fingerprint = sha256_file(args.split)
    if calibration is not None:
        expected = {
            "model_key": args.key,
            "reference_sha256": reference_metadata["sha256"],
        }
        mismatches = {
            key: (getattr(calibration, key), value)
            for key, value in expected.items()
            if getattr(calibration, key) != value
        }
        if mismatches:
            raise ValueError(f"calibration provenance mismatch: {mismatches}")
    calibration_fingerprint = _sha(
        json.dumps(calibration.as_dict(), sort_keys=True, separators=(",", ":"))
        if calibration is not None
        else "unfrozen-calibration-measurement"
    )
    cfg = MODELS[args.key]
    model_spec = load_protocol()["models"]["verbalizer"]
    if cfg["av"] != model_spec["repo_id"]:
        raise ValueError("AV repository differs from the frozen protocol")
    av_dir = snapshot_download(
        repo_id=model_spec["repo_id"],
        revision=model_spec["revision"],
        local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(av_dir, trust_remote_code=True)
    sidecar = read_sidecar(av_dir)
    _, model = load_av(cfg, av_dir)
    injector = ScanningEmbeddingInjector(model, sidecar)
    injection_scale = float(sidecar["injection_scale"])
    sidecar_fingerprint = _sha(json.dumps(sidecar, sort_keys=True, separators=(",", ":")))
    ar_root = Path(args.out) / "ar" / args.run
    out = Path(args.out) / "measurements" / args.run / args.partition
    claims_dir = out / "claims"
    status_path = out / "status.json"
    started = time.time()
    written = 0

    for position, idx in enumerate(by_activation, start=1):
        record = records[idx]
        z_inner = str(record["z_inner"])
        claims = list(record["snippet_results"])
        _assert_claim_tiling(z_inner, claims, list(record.get("connective_results") or []))
        h = unit(np.asarray(gold[idx], dtype=np.float64)).astype(np.float32)
        h_sha256 = hashlib.sha256(h.tobytes()).hexdigest()
        input_ids, marker, offsets, full_text, continuation_start = old_claim.build_tf_sequence(
            tokenizer, sidecar, cfg["family"], record["explanation"]
        )
        if marker != injector.marker_position(input_ids):
            raise ValueError(f"marker drift at activation {idx}")
        z_start = old_cycle.locate_z_inner(full_text, z_inner, continuation_start)
        if z_start < 0:
            raise ValueError(f"z_inner is unlocatable for activation {idx}")
        absolute_spans = [(z_start + int(c["char_span"][0]), z_start + int(c["char_span"][1])) for c in claims]
        ranges = old_cycle.first_char_ranges(offsets, absolute_spans)
        score_h, _ = _direct_mean_scores(model, injector, input_ids, h, injection_scale, ranges)
        score_reference, _ = _direct_mean_scores(model, injector, input_ids, reference, injection_scale, ranges)
        negative_h = None
        negative_h_sha256 = None
        negative_score_h = None
        if negative_map is not None:
            negative_h = unit(np.asarray(gold[negative_map[idx]], dtype=np.float64)).astype(np.float32)
            negative_h_sha256 = hashlib.sha256(negative_h.tobytes()).hexdigest()
            negative_score_h, _ = _direct_mean_scores(
                model, injector, input_ids, negative_h, injection_scale, ranges
            )

        activation_cache = np.load(ar_root / "activations" / f"{idx}.npz")
        real_directions = np.asarray(activation_cache["real_directions"], dtype=np.float32)
        if real_directions.shape != (len(claims), h.size):
            raise ValueError(f"AR activation cache shape drift for {idx}: {real_directions.shape}")
        effect_matrix = np.full((len(claims), len(claims)), np.nan, dtype=np.float64)
        negative_effect_matrix = (
            np.full((len(claims), len(claims)), np.nan, dtype=np.float64)
            if negative_h is not None
            else None
        )
        for source_i, direction in enumerate(real_directions):
            if not np.all(np.isfinite(direction)):
                continue
            h_projected = project_out(h, direction).astype(np.float32)
            projected_scores, _ = _direct_mean_scores(
                model, injector, input_ids, h_projected, injection_scale, ranges
            )
            for target_i in range(len(claims)):
                if np.isfinite(score_h[target_i]) and np.isfinite(projected_scores[target_i]):
                    effect_matrix[source_i, target_i] = score_h[target_i] - projected_scores[target_i]
            if negative_h is not None and negative_score_h is not None:
                negative_projected = project_out(negative_h, direction).astype(np.float32)
                negative_projected_scores, _ = _direct_mean_scores(
                    model,
                    injector,
                    input_ids,
                    negative_projected,
                    injection_scale,
                    ranges,
                )
                assert negative_effect_matrix is not None
                for target_i in range(len(claims)):
                    if np.isfinite(negative_score_h[target_i]) and np.isfinite(
                        negative_projected_scores[target_i]
                    ):
                        negative_effect_matrix[source_i, target_i] = (
                            negative_score_h[target_i] - negative_projected_scores[target_i]
                        )

        z_offset = str(record["explanation"]).find(z_inner)
        if z_offset < 0:
            raise ValueError(f"z_inner is absent from assistant explanation for {idx}")
        for claim_i, cache_path in by_activation[idx]:
            sample_row = claim_sample_rows[(idx, claim_i)]
            output_path = claims_dir / f"{idx}_{claim_i}.json"
            claim = claims[claim_i]
            if cache_path is None:
                row = {
                    "schema_version": SCHEMA_VERSION,
                    "report_sha256": REPORT_SHA256,
                    "reconstruction_metric_amendment_sha256": (
                        RECONSTRUCTION_METRIC_AMENDMENT_SHA256
                    ),
                    "text_model_correction_amendment_sha256": (
                        TEXT_MODEL_CORRECTION_AMENDMENT_SHA256
                    ),
                    "reconstruction_error_formula": RECONSTRUCTION_ERROR_FORMULA,
                    "status": "unresolved",
                    "reason": "upstream claim span unlocated; no decoy family constructed",
                    "model_key": args.key,
                    "run_id": args.run,
                    "partition": args.partition,
                    "split_sha256": split_fingerprint,
                    "idx": idx,
                    "doc_id": record.get("doc_id"),
                    "claim_i": claim_i,
                    "claim_text": claim["text"],
                    "claim_type": claim.get("type"),
                    "claim_subtype": claim.get("subtype"),
                    "reference_activation": {**reference_metadata},
                    "calibration_fingerprint": calibration_fingerprint,
                    "sidecar_fingerprint": sidecar_fingerprint,
                    "candidates": [],
                }
                if output_path.exists() and not args.force:
                    existing = read_json(output_path)
                    if existing == row:
                        continue
                    raise ValueError(f"stale unresolved measurement for {idx}_{claim_i}; rerun with --force")
                atomic_json(output_path, row)
                written += 1
                continue
            ar_meta_path = ar_root / "claims" / f"{idx}_{claim_i}.json"
            ar_npz_path = ar_root / "claims" / f"{idx}_{claim_i}.npz"
            if not ar_meta_path.exists() or not ar_npz_path.exists():
                raise ValueError(f"missing AR cache for {idx}_{claim_i}; run phase ar first")
            ar_meta = read_json(ar_meta_path)
            ar_values = np.load(ar_npz_path)
            if ar_meta["z_sha256"] != _sha(z_inner) or ar_meta["claim_text_sha256"] != _sha(str(claim["text"])):
                raise ValueError(f"stale AR cache for {idx}_{claim_i}")
            if (
                ar_meta.get("reconstruction_metric_amendment_sha256")
                != RECONSTRUCTION_METRIC_AMENDMENT_SHA256
                or ar_meta.get("reconstruction_error_formula")
                != RECONSTRUCTION_ERROR_FORMULA
                or ar_meta.get("text_model_correction_amendment_sha256")
                != TEXT_MODEL_CORRECTION_AMENDMENT_SHA256
            ):
                raise ValueError(
                    f"AR cache for {idx}_{claim_i} uses an incompatible reconstruction metric"
                )
            cache = read_json(cache_path)
            measurement_fingerprint = _sha(json.dumps({
                "schema_version": SCHEMA_VERSION,
                "report_sha256": REPORT_SHA256,
                "reconstruction_metric_amendment_sha256": (
                    RECONSTRUCTION_METRIC_AMENDMENT_SHA256
                ),
                "text_model_correction_amendment_sha256": (
                    TEXT_MODEL_CORRECTION_AMENDMENT_SHA256
                ),
                "reconstruction_error_formula": RECONSTRUCTION_ERROR_FORMULA,
                "model_key": args.key,
                "run": args.run,
                "partition": args.partition,
                "split_sha256": split_fingerprint,
                "idx": idx,
                "claim_i": claim_i,
                "z_sha256": _sha(z_inner),
                "h_sha256": h_sha256,
                "reference_sha256": reference_metadata["sha256"],
                "calibration_fingerprint": calibration_fingerprint,
                "calibration_source": (
                    {
                        "run_id": calibration.run_id,
                        "split_sha256": calibration.split_sha256,
                        "negative_control_sha256": calibration.negative_control_sha256,
                    }
                    if calibration is not None
                    else None
                ),
                "sidecar_fingerprint": sidecar_fingerprint,
                "ar_json_sha256": sha256_file(ar_meta_path),
                "ar_npz_sha256": sha256_file(ar_npz_path),
                "decoy_cache_sha256": sha256_file(cache_path),
                "negative_map_sha256": negative_map_sha256,
            }, sort_keys=True, separators=(",", ":")))
            if output_path.exists() and not args.force:
                existing = read_json(output_path)
                if existing.get("measurement_fingerprint") == measurement_fingerprint:
                    continue
                raise ValueError(f"stale AV measurement for {idx}_{claim_i}; rerun with --force")
            kept = list(ar_meta["candidate_indices"])
            texts = list(ar_meta["candidate_texts"])
            if kept != _text_pass_candidates(cache, z_inner):
                raise ValueError(f"candidate selection drift for {idx}_{claim_i}")
            directions = np.asarray(ar_values["decoy_directions"], dtype=np.float32)
            overlaps = np.asarray(ar_values["decoy_overlaps"], dtype=np.float64)
            errors = np.asarray(ar_values["decoy_reconstruction_errors"], dtype=np.float64)
            decoy_reconstructions = np.asarray(
                ar_values["decoy_reconstructions"], dtype=np.float32
            )
            if len(texts) != len(directions):
                raise ValueError(f"candidate cache length drift for {idx}_{claim_i}")

            real = {
                "overlap": _json_number(ar_meta["real_overlap"]),
                "reference_score": _json_number(score_reference[claim_i]),
                "score_h": _json_number(score_h[claim_i]),
                "activation_support": _json_number(score_h[claim_i] - score_reference[claim_i]),
                "reconstruction_error": _json_number(ar_meta["real_reconstruction_error"]),
                "self_effect": _json_number(effect_matrix[claim_i, claim_i]),
                "token_count": (ranges[claim_i][1] - ranges[claim_i][0]) if ranges[claim_i] else 0,
            }
            candidate_rows: list[dict[str, Any]] = []
            negative_candidate_rows: list[dict[str, Any]] = []
            candidate_sequences: list[tuple[Any, tuple[int, int] | None]] = []
            start, end = (int(x) for x in claim["char_span"])
            for local_i, (candidate_i, text) in enumerate(zip(kept, texts)):
                z_star, star_span = old_cycle.splice(z_inner, (start, end), text)
                assistant_star = (
                    str(record["explanation"])[:z_offset]
                    + z_star
                    + str(record["explanation"])[z_offset + len(z_inner):]
                )
                ids_star, marker_star, offsets_star, full_star, continuation_star = old_claim.build_tf_sequence(
                    tokenizer, sidecar, cfg["family"], assistant_star
                )
                if marker_star != injector.marker_position(ids_star):
                    raise ValueError(f"marker drift in decoy {idx}_{claim_i}_{candidate_i}")
                z_star_start = old_cycle.locate_z_inner(full_star, z_star, continuation_star)
                absolute_star = None if z_star_start < 0 else (
                    z_star_start + star_span[0], z_star_start + star_span[1]
                )
                range_star = old_cycle.first_char_ranges(offsets_star, [absolute_star])[0] if absolute_star else None
                if range_star is None:
                    score_h_star = score_ref_star = float("nan")
                    score_negative_star = float("nan")
                else:
                    (score_h_star,), _ = _direct_mean_scores(
                        model, injector, ids_star, h, injection_scale, [range_star]
                    )
                    (score_ref_star,), _ = _direct_mean_scores(
                        model, injector, ids_star, reference, injection_scale, [range_star]
                    )
                    if negative_h is not None:
                        (score_negative_star,), _ = _direct_mean_scores(
                            model,
                            injector,
                            ids_star,
                            negative_h,
                            injection_scale,
                            [range_star],
                        )
                    else:
                        score_negative_star = float("nan")
                cached_candidate = cache["candidates"][candidate_i]
                candidate_rows.append({
                    "candidate_index": int(candidate_i),
                    "text": text,
                    "text_pass": True,
                    "absence": cached_candidate.get("absence"),
                    "natural_score": cached_candidate.get("natural_score"),
                    "overlap": _json_number(overlaps[local_i]),
                    "reference_score": _json_number(score_ref_star),
                    "score_h": _json_number(score_h_star),
                    "activation_support": _json_number(score_h_star - score_ref_star),
                    "reconstruction_error": _json_number(errors[local_i]),
                    "self_effect": None,
                    "cross_effect": None,
                    "token_count": (range_star[1] - range_star[0]) if range_star else 0,
                })
                candidate_sequences.append((ids_star, range_star))
                if negative_h is not None:
                    negative_candidate_rows.append({
                        "candidate_index": int(candidate_i),
                        "text": text,
                        "text_pass": True,
                        "absence": cached_candidate.get("absence"),
                        "natural_score": cached_candidate.get("natural_score"),
                        "overlap": _json_number(float(negative_h @ directions[local_i])),
                        "reference_score": _json_number(score_ref_star),
                        "score_h": _json_number(score_negative_star),
                        "activation_support": _json_number(
                            score_negative_star - score_ref_star
                        ),
                        "reconstruction_error": _json_number(
                            reconstruction_error(
                                negative_h, decoy_reconstructions[local_i]
                            )
                        ),
                        "self_effect": None,
                        "cross_effect": None,
                        "token_count": (range_star[1] - range_star[0]) if range_star else 0,
                    })

            if calibration is None:
                reading_indices = list(range(len(candidate_rows)))
            else:
                selection = select_decoys(real, candidate_rows, calibration)
                reading_indices = list(selection["selected_indices"])
            for local_i in reading_indices:
                ids_star, range_star = candidate_sequences[local_i]
                if range_star is None or not np.all(np.isfinite(directions[local_i])):
                    continue
                if candidate_rows[local_i].get("score_h") is None or real.get("score_h") is None:
                    continue
                h_projected = project_out(h, directions[local_i]).astype(np.float32)
                (score_star_projected,), _ = _direct_mean_scores(
                    model, injector, ids_star, h_projected, injection_scale, [range_star]
                )
                (score_real_projected,), _ = _direct_mean_scores(
                    model, injector, input_ids, h_projected, injection_scale, [ranges[claim_i]]
                )
                candidate_rows[local_i]["self_effect"] = _json_number(
                    float(candidate_rows[local_i]["score_h"]) - score_star_projected
                )
                candidate_rows[local_i]["cross_effect"] = _json_number(
                    float(real["score_h"]) - score_real_projected
                )
                if negative_h is not None and negative_score_h is not None:
                    negative_projected = project_out(
                        negative_h, directions[local_i]
                    ).astype(np.float32)
                    (negative_star_projected,), _ = _direct_mean_scores(
                        model,
                        injector,
                        ids_star,
                        negative_projected,
                        injection_scale,
                        [range_star],
                    )
                    (negative_real_projected,), _ = _direct_mean_scores(
                        model,
                        injector,
                        input_ids,
                        negative_projected,
                        injection_scale,
                        [ranges[claim_i]],
                    )
                    negative_candidate_rows[local_i]["self_effect"] = _json_number(
                        float(negative_candidate_rows[local_i]["score_h"])
                        - negative_star_projected
                    )
                    negative_candidate_rows[local_i]["cross_effect"] = _json_number(
                        float(negative_score_h[claim_i]) - negative_real_projected
                    )

            off_diagonal = [
                _json_number(effect_matrix[source_i, claim_i])
                for source_i in range(len(claims))
                if source_i != claim_i
            ]
            status = "ok" if ranges[claim_i] is not None else "unresolved"
            reason = "" if status == "ok" else "claim token span unlocated"
            row = {
                "schema_version": SCHEMA_VERSION,
                "report_sha256": REPORT_SHA256,
                "reconstruction_metric_amendment_sha256": (
                    RECONSTRUCTION_METRIC_AMENDMENT_SHA256
                ),
                "text_model_correction_amendment_sha256": (
                    TEXT_MODEL_CORRECTION_AMENDMENT_SHA256
                ),
                "reconstruction_error_formula": RECONSTRUCTION_ERROR_FORMULA,
                "status": status,
                "reason": reason,
                "model_key": args.key,
                "run_id": args.run,
                "partition": args.partition,
                "split_sha256": split_fingerprint,
                "idx": idx,
                "doc_id": record.get("doc_id"),
                "claim_i": claim_i,
                "claim_text": claim["text"],
                "claim_type": claim.get("type"),
                "claim_subtype": cache.get("taxo"),
                "claim_sampling": sample_row,
                "inclusion_probability": sample_row["inclusion_probability"],
                "analysis_weight": sample_row["analysis_weight"],
                "trajectory_id": record.get("trajectory_id"),
                "evaluation_family": record.get("evaluation_family"),
                "scoring": "per-token mean log probability",
                "projection": "h-(h^T delta_hat)delta_hat; no post-projection normalization",
                "reference_activation": {
                    **reference_metadata,
                },
                "injection_scale": injection_scale,
                "measurement_fingerprint": measurement_fingerprint,
                "calibration_fingerprint": calibration_fingerprint,
                "sidecar_fingerprint": sidecar_fingerprint,
                "real": real,
                "off_diagonal_effects": off_diagonal,
                "n_generated_decoys": int(ar_meta["n_generated"]),
                "n_text_pass_decoys": int(ar_meta["n_text_pass"]),
                "causal_readings_measured_for": reading_indices,
                "candidates": candidate_rows,
            }
            if negative_h is not None and negative_score_h is not None:
                ar_full = np.asarray(activation_cache["ar_full"], dtype=np.float32)
                real_direction = real_directions[claim_i]
                assert negative_effect_matrix is not None
                row["negative_control"] = {
                    "kind": "fineweb_cyclic_deranged_activation",
                    "source_idx": idx,
                    "donor_idx": negative_map[idx],
                    "mapping_sha256": negative_map_sha256,
                    "source_activation_sha256": h_sha256,
                    "donor_activation_sha256": negative_h_sha256,
                    "same_claim_text_and_directions": True,
                    "real": {
                        "overlap": _json_number(float(negative_h @ real_direction)),
                        "reference_score": _json_number(score_reference[claim_i]),
                        "score_h": _json_number(negative_score_h[claim_i]),
                        "activation_support": _json_number(
                            negative_score_h[claim_i] - score_reference[claim_i]
                        ),
                        "reconstruction_error": _json_number(
                            reconstruction_error(negative_h, ar_full)
                        ),
                        "self_effect": _json_number(
                            negative_effect_matrix[claim_i, claim_i]
                        ),
                        "token_count": real["token_count"],
                    },
                    "candidates": negative_candidate_rows,
                }
            atomic_json(output_path, row)
            written += 1
        atomic_json(status_path, {
            "phase": "av",
            "status": "running",
            "key": args.key,
            "run": args.run,
            "partition": args.partition,
            "reconstruction_metric_amendment_sha256": (
                RECONSTRUCTION_METRIC_AMENDMENT_SHA256
            ),
            "text_model_correction_amendment_sha256": (
                TEXT_MODEL_CORRECTION_AMENDMENT_SHA256
            ),
            "activations_done": position,
            "activations_total": len(by_activation),
            "claims_written": written,
            "calibration_frozen": calibration is not None,
            "elapsed_seconds": round(time.time() - started, 1),
        })
        print(f"[av {position}/{len(by_activation)}] idx={idx} claims={len(by_activation[idx])}", flush=True)
    atomic_json(status_path, {
        "phase": "av",
        "status": "done",
        "key": args.key,
        "run": args.run,
        "partition": args.partition,
        "reconstruction_metric_amendment_sha256": RECONSTRUCTION_METRIC_AMENDMENT_SHA256,
        "text_model_correction_amendment_sha256": TEXT_MODEL_CORRECTION_AMENDMENT_SHA256,
        "activations_total": len(by_activation),
        "claims_written": written,
        "calibration_frozen": calibration is not None,
        "elapsed_seconds": round(time.time() - started, 1),
    })


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("ar", "av"))
    parser.add_argument("key", choices=("qwen7b", "gemma12b", "gemma27b", "llama70b"))
    parser.add_argument("--run", required=True)
    parser.add_argument("--runs-dir", required=True)
    parser.add_argument("--decoys-dir", required=True)
    parser.add_argument("--gold-dir", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--claim-sample", required=True)
    parser.add_argument("--partition", choices=("calibration", "evaluation"), required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="AR device inside CUDA_VISIBLE_DEVICES (default: cuda:0)",
    )
    parser.add_argument("--decoy-model", default="terra")
    parser.add_argument("--prompt-version", default="terra-v8")
    parser.add_argument("--calibration", help="frozen thresholds; omit only for calibration measurement")
    parser.add_argument(
        "--negative-map",
        help="FineWeb cyclic derangement artifact (required for calibration AV)",
    )
    parser.add_argument("--reference-vector", help="fixed, already-unit .npy h_ref (required for phase av)")
    parser.add_argument("--idx", type=int, action="append")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.phase == "av" and not args.reference_vector:
        raise SystemExit(
            "phase av requires an explicit --reference-vector for the fixed h_ref"
        )
    if args.partition == "evaluation" and not args.calibration and args.phase == "av":
        raise SystemExit("evaluation AV measurement requires --calibration; refusing an unfrozen run")
    if args.partition == "calibration" and args.phase == "av" and not args.negative_map:
        raise SystemExit("calibration AV measurement requires --negative-map")
    if args.negative_map and not (args.partition == "calibration" and args.phase == "av"):
        raise SystemExit("--negative-map is valid only for calibration AV measurement")
    if args.phase == "ar":
        run_ar(args)
    else:
        run_av(args)


if __name__ == "__main__":
    main()
