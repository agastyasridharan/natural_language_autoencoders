from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

COMPONENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COMPONENT))

import convert_activation_manifest as converter  # noqa: E402
import generate_av as gav  # noqa: E402


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def fineweb_item(index: int, activation_id: str | None = None) -> dict:
    did = f"fineweb:doc-{index}"
    return {
        "activation_id": activation_id or f"{did}:position:{50 + index}",
        "activation_index": index,
        "document_id": did,
        "trajectory_id": None,
        "cluster_id": did,
        "distribution": "fineweb",
        "benchmark": "fineweb",
        "family": "fineweb",
    }


def canonical_matrix_manifest(array_path: Path, items: list[dict]) -> dict:
    return {
        "schema_version": gav.ACTIVATION_SCHEMA,
        "model_id": gav.MODEL_ID,
        "model_revision": gav.BASE_REVISION,
        "layer_index": gav.LAYER_INDEX,
        "d_model": gav.D_MODEL,
        "activation_norm": "none",
        "activation_file": array_path.name,
        "activation_sha256": gav.sha256_file(array_path),
        "items": items,
    }


class ManifestTests(unittest.TestCase):
    def test_matrix_and_per_item_sources_are_zero_copy(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            matrix = np.stack(
                [np.ones(gav.D_MODEL, dtype=np.float32), np.full(gav.D_MODEL, 2, np.float32)]
            )
            matrix_path = root / "gold.npy"
            np.save(matrix_path, matrix)
            matrix_manifest = root / "matrix.json"
            write_json(
                matrix_manifest,
                canonical_matrix_manifest(
                    matrix_path, [fineweb_item(0), fineweb_item(1)]
                ),
            )
            loaded = gav.load_activation_manifests([matrix_manifest])
            self.assertEqual(len(loaded), 2)
            self.assertEqual(loaded[1].source.array.shape, (2, gav.D_MODEL))
            self.assertIsInstance(loaded[1].source.array, np.memmap)

            vector_path = root / "one.npy"
            np.save(vector_path, np.arange(gav.D_MODEL, dtype=np.float32) + 1)
            per_item_manifest = root / "per-item.json"
            write_json(
                per_item_manifest,
                {
                    "schema_version": gav.ACTIVATION_SCHEMA,
                    "model_id": gav.MODEL_ID,
                    "model_revision": gav.BASE_REVISION,
                    "layer_index": gav.LAYER_INDEX,
                    "d_model": gav.D_MODEL,
                    "activation_norm": "none",
                    "activation_file": None,
                    "activation_file_mode": "per_item",
                    "items": [
                        {
                            "activation_id": "capability_browsecomp:t0:position_key:4",
                            "activation_index": 0,
                            "activation_file": vector_path.name,
                            "activation_sha256": gav.sha256_file(vector_path),
                            "document_id": None,
                            "trajectory_id": "capability_browsecomp:t0",
                            "cluster_id": "capability_browsecomp:t0",
                            "distribution": "ood",
                            "benchmark": "browsecomp",
                            "family": "capability",
                        }
                    ],
                },
            )
            loaded = gav.load_activation_manifests([per_item_manifest])
            self.assertEqual(loaded[0].source.array.shape, (1, gav.D_MODEL))
            self.assertEqual(float(loaded[0].source.array[0, 3]), 4.0)

    def test_rejects_noncanonical_benchmark_slug(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            array_path = root / "gold.npy"
            np.save(array_path, np.ones((1, gav.D_MODEL), np.float32))
            bad = fineweb_item(0)
            bad["benchmark"] = "FineWeb"
            manifest_path = root / "manifest.json"
            write_json(manifest_path, canonical_matrix_manifest(array_path, [bad]))
            with self.assertRaisesRegex(gav.ContractError, "frozen fineweb slug"):
                gav.load_activation_manifests([manifest_path])


class ConverterTests(unittest.TestCase):
    def test_all_exact_evaluation_registry_ids_map_to_frozen_slugs(self) -> None:
        expected = {
            "alignment_blackmail": ("alignment", "blackmail"),
            "alignment_destructive_coding": (
                "alignment",
                "destructive_coding_actions",
            ),
            "alignment_overly_agentic": ("alignment", "overly_agentic_actions"),
            "alignment_factual_honesty": ("alignment", "factual_honesty"),
            "alignment_appropriate_harmlessness": (
                "alignment",
                "appropriate_harmlessness",
            ),
            "alignment_sycophancy": ("alignment", "sycophancy"),
            "alignment_multiturn_deception": (
                "alignment",
                "multi_turn_deception",
            ),
            "capability_swebench_verified": ("capability", "swe_bench_verified"),
            "capability_gpqa_diamond": ("capability", "gpqa_diamond"),
            "capability_hle": ("capability", "humanitys_last_exam"),
            "capability_tau2_retail": ("capability", "tau2_bench_retail"),
            "capability_mmmlu": ("capability", "mmmlu"),
            "capability_browsecomp": ("capability", "browsecomp"),
            "capability_terminalbench2": ("capability", "terminal_bench_2"),
            "capability_financeagent": ("capability", "financeagent"),
            "capability_cybergym": ("capability", "cybergym"),
        }
        self.assertEqual(len(expected), 16)
        for registry_id, frozen in expected.items():
            with self.subTest(registry_id=registry_id):
                self.assertEqual(
                    converter._family(registry_id, "registry-test"), frozen
                )

    def test_fineweb_matrix_converter_references_original_array(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "source"
            source.mkdir()
            np.save(source / "gold.npy", np.ones((2, gav.D_MODEL), np.float32))
            write_json(
                source / "manifest.json",
                [
                    {
                        "idx": i,
                        "doc_id": f"fineweb:doc-{i}",
                        "position": 70 + i,
                        "n_tokens": 100,
                        "in_range": True,
                        "read_token": " x",
                        "h_norm": 10.0,
                    }
                    for i in range(2)
                ],
            )
            write_json(
                source / "extract_meta.json",
                {
                    "base": gav.MODEL_ID,
                    "layer": gav.LAYER_INDEX,
                    "d_model": gav.D_MODEL,
                },
            )
            output = root / "converted" / "manifest.json"
            result = converter.convert(source, output)
            self.assertEqual(result["items"], 2)
            converted = json.loads(output.read_text())
            referenced = (output.parent / converted["activation_file"]).resolve()
            self.assertEqual(referenced, (source / "gold.npy").resolve())
            self.assertEqual(list(source.glob("*.npy")), [source / "gold.npy"])
            loaded = gav.load_activation_manifests([output])
            self.assertEqual(len(loaded), 2)

    def test_trajectory_converter_and_incomplete_guard(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "agentic"
            activation_rows = []
            vector_paths = []
            for i in range(4):
                vector_rel = Path(f"activations/capability_browsecomp/t-{i}.npy")
                vector_path = source / vector_rel
                vector_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(vector_path, np.full(gav.D_MODEL, i + 1, np.float32))
                vector_paths.append(vector_path)
                activation_rows.append(
                    {
                        "position_key": 7 + i,
                        "turn_index": 1,
                        "assistant_response_token_index": 7 + i,
                        "token_position": 100 + i,
                        "read_token": " x",
                        "vector_file": str(vector_rel),
                        "vector_sha256": gav.sha256_file(vector_path),
                    }
                )
            trajectory_rel = Path("trajectories/capability_browsecomp/t.json")
            write_json(
                source / trajectory_rel,
                {
                    "status": "complete",
                    "family_id": "capability_browsecomp",
                    "task_id": "browsecomp-1",
                    "trajectory_id": "capability_browsecomp:browsecomp-1",
                    "target_model": {
                        "repo_id": gav.MODEL_ID,
                        "revision": gav.BASE_REVISION,
                        "layer": gav.LAYER_INDEX,
                    },
                    "activations": activation_rows,
                },
            )
            running_meta = {
                "status": "running",
                "target": {
                    "repo_id": gav.MODEL_ID,
                    "revision": gav.BASE_REVISION,
                    "layer": gav.LAYER_INDEX,
                },
                "failures": [],
            }
            write_json(source / "shard-00-meta.json", running_meta)
            output = root / "converted.json"
            with self.assertRaisesRegex(gav.ContractError, "incomplete"):
                converter.convert(source, output)
            result = converter.convert(source, output, allow_incomplete=True)
            self.assertFalse(result["source_shards_complete"])
            converted = json.loads(output.read_text())
            self.assertEqual(converted["activation_file_mode"], "per_item")
            self.assertEqual(converted["items"][0]["benchmark"], "browsecomp")
            self.assertEqual(len(gav.load_activation_manifests([output])), 4)


class SamplingAndParsingTests(unittest.TestCase):
    def test_sampling_keys_are_stable_and_independent(self) -> None:
        a0 = gav.sampling_identity("a", "plausible", 0, 42)
        self.assertEqual(a0, gav.sampling_identity("a", "plausible", 0, 42))
        values = {
            a0,
            gav.sampling_identity("a", "plausible", 1, 42),
            gav.sampling_identity("a", "implausible", 0, 42),
            gav.sampling_identity("b", "plausible", 0, 42),
        }
        self.assertEqual(len(values), 4)

    def test_specific_parse_statuses(self) -> None:
        self.assertEqual(
            gav.parse_explanation("<explanation>alpha</explanation>").status, "ok"
        )
        self.assertEqual(gav.parse_explanation("alpha").status, "missing_both_tags")
        self.assertEqual(
            gav.parse_explanation("<explanation>alpha").status, "missing_close_tag"
        )
        self.assertEqual(
            gav.parse_explanation(
                "<explanation>a</explanation><explanation>b</explanation>"
            ).status,
            "multiple_tags",
        )

    def test_generation_record_enforces_injection_marker_and_cjk_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            array_path = root / "gold.npy"
            np.save(array_path, np.ones((1, gav.D_MODEL), np.float32))
            manifest_path = root / "manifest.json"
            write_json(
                manifest_path,
                canonical_matrix_manifest(array_path, [fineweb_item(0)]),
            )
            item = gav.load_activation_manifests([manifest_path])[0]
            output = gav.RuntimeOutput(
                raw_text="<explanation>alpha</explanation>",
                clean_text="<explanation>alpha</explanation>",
                token_ids=(1,),
                stop_reason="eos",
                eos_token_id=1,
                injection_apply_count_before=3,
                injection_apply_count_after=4,
                injection_target_l2_norm=150.0,
                injected_l2_norm=150.0,
                marker_row_matches_scaled_activation=True,
            )
            clean = gav.make_generation_record(
                item=item,
                condition="plausible",
                checkpoint_id="test",
                run_fingerprint="f" * 64,
                sample_index=0,
                sampling_key="a" * 64,
                torch_seed=1,
                runtime_output=output,
                injection_char="㈎",
                cjk_fraction_fn=lambda text: 0.0,
            )
            self.assertTrue(clean["generation_ok"])
            self.assertTrue(clean["diagnostics"]["injection_applied_exactly_once"])

            bad_output = dataclasses.replace(
                output,
                clean_text="<explanation>㈎ 中文文本中文文本</explanation>",
                injection_apply_count_after=5,
            )
            bad = gav.make_generation_record(
                item=item,
                condition="plausible",
                checkpoint_id="test",
                run_fingerprint="f" * 64,
                sample_index=1,
                sampling_key="b" * 64,
                torch_seed=2,
                runtime_output=bad_output,
                injection_char="㈎",
                cjk_fraction_fn=lambda text: 0.5,
            )
            self.assertFalse(bad["generation_ok"])
            self.assertIn("diagnostic:injection_apply_count_2", bad["failure_reason"])
            self.assertIn("diagnostic:marker_leak", bad["failure_reason"])
            self.assertNotIn("diagnostic:cjk_output", bad["failure_reason"])
            self.assertTrue(bad["diagnostics"]["cjk_suspect"])

            # Non-marker CJK is a distributional smoke signal. It must not bias
            # complete-case scoring by turning a valid generation into failure.
            cjk_output = dataclasses.replace(
                output,
                clean_text="<explanation>中文文本中文文本</explanation>",
            )
            cjk = gav.make_generation_record(
                item=item,
                condition="plausible",
                checkpoint_id="test",
                run_fingerprint="f" * 64,
                sample_index=1,
                sampling_key="c" * 64,
                torch_seed=3,
                runtime_output=cjk_output,
                injection_char="㈎",
                cjk_fraction_fn=lambda text: 0.5,
            )
            self.assertTrue(cjk["generation_ok"])
            self.assertIsNone(cjk["failure_reason"])
            self.assertTrue(cjk["diagnostics"]["cjk_suspect"])
            summary = gav._progress_summary([cjk, cjk, cjk], expected=3)
            self.assertEqual(summary["generation_ok"], 3)
            self.assertTrue(summary["cjk_manual_audit_alert"])


class FakeRuntime:
    init_calls = 0
    generate_calls = 0

    def __init__(self, **kwargs) -> None:
        type(self).init_calls += 1
        self.cfg = SimpleNamespace(injection_char="㈎")
        self.cjk_fraction = lambda text: 0.0
        self.provenance = {"fake": True}
        self.apply_count = 0

    def generate(self, activation: np.ndarray, torch_seed: int) -> gav.RuntimeOutput:
        type(self).generate_calls += 1
        before = self.apply_count
        self.apply_count += 1
        return gav.RuntimeOutput(
            raw_text="<explanation>alpha</explanation><|im_end|>",
            clean_text="<explanation>alpha</explanation>",
            token_ids=(1, 2, 3),
            stop_reason="eos",
            eos_token_id=3,
            injection_apply_count_before=before,
            injection_apply_count_after=self.apply_count,
            injection_target_l2_norm=150.0,
            injected_l2_norm=150.0,
            marker_row_matches_scaled_activation=True,
        )


class ShardRunnerTests(unittest.TestCase):
    def test_atomic_records_compaction_and_resume(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            array_path = root / "gold.npy"
            np.save(array_path, np.ones((1, gav.D_MODEL), np.float32))
            manifest_path = root / "manifest.json"
            write_json(
                manifest_path,
                canonical_matrix_manifest(array_path, [fineweb_item(0, "activation-0")]),
            )
            actor_dir = root / "actor"
            actor_dir.mkdir()
            write_json(actor_dir / "adapter_config.json", {"test": True})
            sidecar_path = root / "rl_vllm" / "nla_meta.yaml"
            write_json(sidecar_path, {"schema_version": "test"})
            self.assertFalse((actor_dir / "nla_meta.yaml").exists())
            args = argparse.Namespace(
                input_manifest=[str(manifest_path)],
                condition="implausible",
                actor_checkpoint=str(actor_dir),
                checkpoint_id="step-780-sha256:test",
                sidecar=str(sidecar_path),
                output_dir=str(root / "out"),
                num_shards=1,
                shard_index=0,
                sampling_seed=42,
                base_revision=gav.BASE_REVISION,
                nla_root=str(root),
                device="cpu",
                load_in_4bit=False,
                skip_array_sha256=False,
                allow_empty_shard=False,
            )
            FakeRuntime.init_calls = 0
            FakeRuntime.generate_calls = 0
            first = gav.run_shard(args, runtime_factory=FakeRuntime)
            self.assertEqual(first["committed_records"], 2)
            self.assertEqual(first["generation_ok"], 2)
            self.assertEqual(FakeRuntime.generate_calls, 2)

            # A complete rerun must not construct the runtime or draw another
            # token; it validates and recompacts the immutable records.
            def forbidden_runtime(**kwargs):
                raise AssertionError("runtime must not be loaded on complete resume")

            second = gav.run_shard(args, runtime_factory=forbidden_runtime)
            self.assertEqual(second["explanations"]["sha256"], first["explanations"]["sha256"])
            shard_dir = (
                root / "out" / "implausible" / "shard-00000-of-00001"
            )
            rows = [json.loads(line) for line in (shard_dir / "explanations.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertEqual({row["sample_index"] for row in rows}, {0, 1})
            self.assertTrue(all(row["schema_version"] == gav.EXPLANATION_SCHEMA for row in rows))
            self.assertFalse(list(shard_dir.rglob(".*.tmp-*")))
            config = json.loads((shard_dir / "shard_config.json").read_text())
            self.assertEqual(config["sidecar_file"]["path"], str(sidecar_path.resolve()))
            self.assertTrue(config["sidecar_file"]["exists"])


if __name__ == "__main__":
    unittest.main()
