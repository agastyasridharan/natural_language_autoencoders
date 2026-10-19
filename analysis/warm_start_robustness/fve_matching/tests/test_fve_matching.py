from __future__ import annotations

import dataclasses
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

COMPONENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COMPONENT))

import core  # noqa: E402
import evaluate_checkpoint as evaluator_cli  # noqa: E402
import select_checkpoint as selector_cli  # noqa: E402


def write(path: Path, payload: bytes = b"x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def make_transaction(root: Path, step: int = 20) -> Path:
    transaction = root / f"step_{step:06d}"
    write(transaction / "actor/adapter_config.json", b"{}")
    write(transaction / "actor/adapter_model.safetensors")
    write(transaction / "actor/reference/adapter_config.json", b"{}")
    write(transaction / "actor/reference/adapter_model.safetensors")
    write(transaction / "critic/config.json", b"{}")
    write(transaction / "critic/value_head.safetensors")
    write(transaction / "critic/model.safetensors")
    write(transaction / "critic/saved_at_step.txt", str(step).encode())
    write(transaction / "optim.pt")
    config = {
        "eval_n_prompts": 256,
        "eval_temperature": 0.0,
        "max_new_tokens": 150,
        "save_every": 20,
        "seed": 42,
        "train_critic": True,
        "rl_parquet": "/data/rl.parquet",
        "av_ckpt": "/data/merged-av",
        "base_ckpt": "Qwen/Qwen2.5-7B-Instruct",
        "av_adapter": "/data/av-sft",
        "vllm_tp": 1,
        "vllm_gpu_mem": 0.25,
        "vllm_max_len": 512,
        "max_rows": 0,
        "eval_skip_rows": 0,
        "val_rows": 10000,
        "num_steps": 780,
        "eval_every": 10,
    }
    write(transaction / "run_config.yaml", yaml.safe_dump(config).encode())
    write(transaction / "nla_meta.yaml", b"kind: nla_dataset\n")
    metrics = b"".join(
        (json.dumps({"step": index}) + "\n").encode() for index in range(step)
    )
    write(transaction / "metrics.jsonl", metrics)
    inventory = [
        {
            "path": str(path.relative_to(transaction)),
            "size": path.stat().st_size,
        }
        for path in sorted(transaction.rglob("*"))
        if path.is_file()
    ]
    marker = {
        "schema_version": 1,
        "step": step,
        "actor_source": f"iter_{step:06d}",
        "files_before_marker": len(inventory),
        "inventory": inventory,
    }
    write(
        transaction / "COMPLETE.json",
        (json.dumps(marker, sort_keys=True) + "\n").encode(),
    )
    return transaction


def fake_result(condition: str, step: int, fve: float) -> dict:
    protocol = {
        "schema_version": core.EVALUATION_SCHEMA,
        "eval_set_sha256": "e" * 64,
        "rl_parquet_sha256": "f" * 64,
        "n_prompts": 256,
        "temperature": 0.0,
    }
    protocol_hash = core.sha256_json(protocol)
    protocol_record = {
        **protocol,
        "sha256": protocol_hash,
        "evaluated_checkpoint": {"transaction_step": step},
    }
    result = {
        "schema_version": core.EVALUATION_SCHEMA,
        "status": "complete",
        "condition": condition,
        "evaluation_protocol_sha256": protocol_hash,
        "evaluation_protocol": protocol_record,
        "checkpoint": {
            "transaction_step": step,
            "critic_saved_at_step": step,
            "transaction_fingerprint": f"{step:064x}"[-64:],
            "base_checkpoint_tree_sha256": "b" * 64,
            "merged_av_checkpoint_tree_sha256": (
                "c" * 64 if condition == "plausible" else "d" * 64
            ),
            "relevant_file_inventory": [
                {"path": "actor/adapter_model.safetensors", "sha256": "a" * 64}
            ],
        },
        "checkpoint_state_mapping": {
            "zero_based_last_update_index": step - 1,
        },
        "heldout": {
            "n_prompts": 256,
            "eval_set_sha256": "e" * 64,
            "rl_parquet_sha256": "f" * 64,
        },
        "metrics": {
            "n_prompts": 256,
            "baseline_mse": 0.75,
            "fve_pct": fve,
        },
    }
    result["result_payload_sha256"] = core.sha256_json(result)
    return result


class MappingTests(unittest.TestCase):
    def test_machine_protocol_freezes_mapping_and_separate_analysis_sets(self) -> None:
        protocol = yaml.safe_load((COMPONENT / "protocol.yaml").read_text())
        self.assertEqual(
            protocol["checkpoint_state_mapping"][
                "logged_eval_step_N_maps_to_transaction_step"
            ],
            "N_plus_1",
        )
        self.assertTrue(
            protocol["claim_analysis_sets"]["must_generate_and_score_separately"]
        )
        self.assertEqual(
            [
                row["analysis_set_id"]
                for row in protocol["claim_analysis_sets"]["sets"]
            ],
            ["final_vs_final", "fve_matched"],
        )

    def test_eval_label_is_offset_from_checkpoint_iteration(self) -> None:
        mapping = core.checkpoint_state_mapping(20, eval_every=10)
        self.assertEqual(mapping["zero_based_last_update_index"], 19)
        self.assertEqual(mapping["same_state_log_eval_label"], 19)
        self.assertFalse(mapping["same_state_log_eval_was_scheduled"])
        self.assertEqual(mapping["logged_eval_step_N_maps_to_transaction_step"], "N+1")
        self.assertIn("transaction 21", mapping["warning"])

    def test_locked_run_config_preserves_offset(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            transaction = core.audit_transaction(make_transaction(Path(td)))
            mapping = core.validate_locked_run_config(
                transaction.run_config, transaction.step
            )
            self.assertEqual(mapping["actor_iteration"], "iter_000020")
            self.assertEqual(mapping["zero_based_last_update_index"], 19)

    def test_post_auto_resolved_split_values_are_recognized(self) -> None:
        config = {
            "val_rows": 10_000,
            "max_rows": 123_456,
            "eval_skip_rows": 113_456,
        }
        audit = core.validate_resolved_doc_hash_split(config, 123_456)
        self.assertEqual(audit["expected_auto_eval_skip_rows"], 113_456)
        config["eval_skip_rows"] = 113_455
        with self.assertRaisesRegex(core.ContractError, "expected auto value"):
            core.validate_resolved_doc_hash_split(config, 123_456)


class TransactionTests(unittest.TestCase):
    def test_actor_and_historical_critic_are_one_hashed_transaction(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            transaction = core.audit_transaction(make_transaction(Path(td)))
            self.assertEqual(transaction.step, 20)
            self.assertEqual(
                int((transaction.critic / "saved_at_step.txt").read_text()), 20
            )
            self.assertTrue(all(row["sha256"] for row in transaction.relevant_files))
            self.assertEqual(len(transaction.fingerprint), 64)

    def test_mixed_critic_step_fails(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = make_transaction(Path(td))
            # Same byte length preserves the COMPLETE size inventory and reaches
            # the semantic actor/critic-step check.
            (path / "critic/saved_at_step.txt").write_text("21")
            with self.assertRaisesRegex(core.ContractError, "critic is at step 21"):
                core.audit_transaction(path)


class HeldoutTests(unittest.TestCase):
    def test_first_256_val_rows_and_digest_are_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "rl.parquet"
            n_rows = 300
            prompts = [f"prompt-{index}" for index in range(n_rows)]
            activations = [
                [float(index + 1), float(index + 2)] for index in range(n_rows)
            ]
            table = pa.table(
                {
                    "prompt": prompts,
                    "activation_vector": pa.array(activations, type=pa.list_(pa.float32())),
                    "doc_id": [f"doc-{index}" for index in range(n_rows)],
                }
            )
            pq.write_table(table, path, row_group_size=37)
            rows, total, permille = core.load_fixed_heldout(
                path,
                val_rows=100,
                val_permille_fn=lambda val, total: 1000,
                is_val_doc_fn=lambda document, bucket: True,
            )
            self.assertEqual(total, 300)
            self.assertEqual(permille, 1000)
            self.assertEqual([row.parquet_row_index for row in rows], list(range(256)))
            first = core.heldout_descriptor(rows)
            second = core.heldout_descriptor(rows)
            self.assertEqual(first["eval_set_sha256"], second["eval_set_sha256"])
            altered = list(rows)
            changed_activation = rows[0].activation.copy()
            changed_activation[0] += 1
            altered[0] = dataclasses.replace(rows[0], activation=changed_activation)
            changed = core.heldout_descriptor(altered)
            self.assertNotEqual(first["eval_set_sha256"], changed["eval_set_sha256"])


class MetricTests(unittest.TestCase):
    def test_fve_uses_valid_rewards_and_failure_floor_separately(self) -> None:
        result = core.aggregate_fve([-0.25, None, -0.75], baseline_mse=1.0)
        self.assertEqual(result["n_valid_for_fve"], 2)
        self.assertAlmostEqual(result["reward_mean_valid"], -0.5)
        self.assertAlmostEqual(result["fve_pct"], 50.0)
        self.assertAlmostEqual(
            result["reward_mean_with_failure_floor"], (-0.25 - 2.0 - 0.75) / 3
        )

    def test_resume_payload_hash_detects_mutation(self) -> None:
        sealed = evaluator_cli._seal_payload({"rows": [1, 2]}, "payload_sha256")
        evaluator_cli._verify_payload(sealed, "payload_sha256", "test")
        sealed["rows"].append(3)
        with self.assertRaisesRegex(core.ContractError, "payload_sha256 mismatch"):
            evaluator_cli._verify_payload(sealed, "payload_sha256", "test")


class SelectorTests(unittest.TestCase):
    def _records(self) -> list[tuple[Path, dict]]:
        records = []
        for step in range(20, 781, 20):
            fve = 70.0 if step in (600, 620) else (80.0 if step == 780 else 90.0)
            records.append((Path(f"plausible-{step}.json"), fake_result("plausible", step, fve)))
        records.append(
            (Path("implausible-780.json"), fake_result("implausible", 780, 70.0))
        )
        return records

    def test_holds_lower_final_and_tie_breaks_to_later_checkpoint(self) -> None:
        selection = core.select_fve_matched(self._records())
        self.assertEqual(selection["held_lower_final"]["condition"], "implausible")
        self.assertEqual(selection["held_lower_final"]["transaction_step"], 780)
        self.assertEqual(selection["matched_higher"]["condition"], "plausible")
        self.assertEqual(selection["matched_higher"]["transaction_step"], 620)
        self.assertEqual(selection["selected_absolute_fve_difference_pct_points"], 0.0)
        sets = {
            row["analysis_set_id"]: row
            for row in selection["claim_analysis_sets"]["sets"]
        }
        self.assertEqual(set(sets), {"final_vs_final", "fve_matched"})
        self.assertEqual(
            sets["final_vs_final"]["members"]["plausible"]["transaction_step"],
            780,
        )
        self.assertEqual(
            sets["fve_matched"]["members"]["plausible"]["transaction_step"],
            620,
        )
        self.assertIn(
            "omit checkpoint_id",
            selection["claim_analysis_sets"]["activation_identity_warning"],
        )

    def test_missing_higher_condition_transaction_fails_closed(self) -> None:
        records = [pair for pair in self._records() if pair[0].name != "plausible-400.json"]
        with self.assertRaisesRegex(core.ContractError, "lacks exact historical-critic"):
            core.select_fve_matched(records)

    def test_different_heldout_set_fails_closed(self) -> None:
        records = self._records()
        result = records[0][1]
        result["heldout"]["eval_set_sha256"] = "0" * 64
        result.pop("result_payload_sha256")
        result["result_payload_sha256"] = core.sha256_json(result)
        with self.assertRaisesRegex(core.ContractError, "different/missing held-out"):
            core.select_fve_matched(records)

    def test_cli_selection_is_atomic_hashed_and_resumable(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            paths = []
            for path, result in self._records():
                destination = root / path.name
                destination.write_text(json.dumps(result))
                paths.append(str(destination))
            output = root / "selection.json"
            args = type("Args", (), {"result": paths, "output": str(output)})()
            first = selector_cli.run(args)
            second = selector_cli.run(args)
            self.assertEqual(first, second)
            payload = dict(first)
            declared = payload.pop("selection_payload_sha256")
            self.assertEqual(declared, core.sha256_json(payload))
            self.assertEqual(
                {row["path"] for row in first["selection_implementation"]},
                {"protocol.yaml", "core.py", "select_checkpoint.py"},
            )


if __name__ == "__main__":
    unittest.main()
