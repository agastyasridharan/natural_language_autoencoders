"""CPU-only unit checks for the matched plausible warm-start safeguards."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from argparse import Namespace
from pathlib import Path
from unittest.mock import Mock

import pyarrow as pa
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from nla.datagen.providers import OpenAIBatchProvider
from scripts.datagen.generate_matched_teacher import (
    _clean,
    _complete_with_budget_splits,
    _load_matched_raw,
)
from scripts.datagen.reorder_to_reference import _assert_exact_payload
from scripts.archive_immutable_rl_history import build_manifest
from scripts.archive_openai_batch_cache import archive_stop_ready, build_incomplete_manifest
from scripts.archive_sft_checkpoints import build_manifest as build_sft_manifest
from nla.utils.sft_resume import (
    assert_resume_config_matches,
    stable_run_config,
    wandb_init_kwargs,
)


def _base_table() -> pa.Table:
    return pa.table(
        {
            "n_raw_tokens": pa.array([51, 52, 53], type=pa.int64()),
            "detokenized_text_truncated": ["zero", "one", "two"],
            "activation_vector": pa.array(
                [[0.0, 0.5], [1.0, 1.5], [2.0, 2.5]],
                type=pa.list_(pa.float32(), 2),
            ),
            "activation_layer": pa.array([20, 20, 20], type=pa.int64()),
            "doc_id": ["d0", "d1", "d2"],
        }
    )


def test_exact_reference_selection() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        raw = root / "raw.parquet"
        ref = root / "reference.parquet"
        base = _base_table()
        pq.write_table(base, raw)
        reference = base.take(pa.array([2, 0], type=pa.int64())).append_column(
            "api_explanation", pa.array(["a\n\nb", "c\n\nd"])
        )
        pq.write_table(reference, ref)
        matched, keys = _load_matched_raw(raw, ref)
        assert keys == [("d2", 53, 20), ("d0", 51, 20)]
        assert matched.column("doc_id").to_pylist() == ["d2", "d0"]
        assert matched.column("activation_vector").equals(reference.column("activation_vector"))


def test_payload_comparison_rejects_changed_activation() -> None:
    reference = _base_table().take(pa.array([1, 0], type=pa.int64()))
    changed = reference.set_column(
        reference.schema.get_field_index("activation_vector"),
        "activation_vector",
        pa.array([[9.0, 9.0], [0.0, 0.5]], type=pa.list_(pa.float32(), 2)),
    )
    try:
        _assert_exact_payload(changed, reference, "unit test")
    except RuntimeError as exc:
        assert "activation_vector differs" in str(exc)
    else:
        raise AssertionError("changed activation was accepted")


def test_format_cleaning() -> None:
    pattern = r"<analysis>\s*(.*?)\s*</analysis>"
    valid, failure = _clean("<analysis>first\nsecond</analysis>", pattern)
    assert valid == "first\n\nsecond" and failure is None
    invalid, failure = _clean("<analysis>never closed", pattern)
    assert invalid is None and failure == "missing_or_incomplete_analysis_tags"

    class SplitProvider:
        def complete(self, prompts):
            if len(prompts) > 2:
                raise RuntimeError("OpenAI spend guard: reservation too coarse")
            return [f"done:{prompt}" for prompt in prompts]

    prompts = [str(index) for index in range(7)]
    assert _complete_with_budget_splits(SplitProvider(), prompts) == [
        f"done:{prompt}" for prompt in prompts
    ]


def test_project_budget_floor_cap_and_cumulative_ledger() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        prior = root / "prior"
        prior.mkdir()
        (prior / "usage.json").write_text(
            json.dumps(
                {
                    "requests_succeeded": 99_947,
                    "requests_failed": 0,
                    "input_tokens": 1,
                    "cached_input_tokens": 0,
                    "output_tokens": 1,
                    "cost_usd": 42.2396,
                }
            )
        )
        old = os.environ.get("OPENAI_API_KEY_2")
        os.environ["OPENAI_API_KEY_2"] = "unit-test-key"
        try:
            provider = OpenAIBatchProvider(
                cache_dir=str(root),
                spend_cap_usd=87.23,
                project_spend_cap_usd=100.0,
                known_external_spend_usd=0.07056656,
                reserved_downstream_spend_usd=12.0,
                minimum_prior_cache_spend_usd=42.2395,
            )
            lines = provider._request_lines(["ASCII", "café"])
            expected_upper = (
                (sum(len(line.encode("utf-8")) for line in lines) + 2 * 128)
                * provider.input_price
                + 2 * provider.max_tokens * provider.output_price
            ) / 1_000_000
            assert provider._pessimistic_cost(lines, 2) == expected_upper
            state_dir = root / "new"
            state_dir.mkdir()
            state = provider._reserve_spend(state_dir / "state.json", {}, 44.0)
            assert state["pessimistic_cost_usd"] == 44.0
            # A terminal Batch can contain billable partial successes. With no
            # reconciled usage.json, its full reservation must remain committed.
            state.update({"status": "expired", "request_counts": {"completed": 500}})
            provider._atomic_json(state_dir / "state.json", state)
            assert provider._committed_spend() == 44.0
            ledger = json.loads((root / "cumulative_usage.json").read_text())
            assert ledger["cache_actual_spend_usd"] == 42.2396
            assert ledger["cache_committed_worst_case_usd"] == 44.0
            assert ledger["reserved_downstream_spend_usd"] == 12.0

            other = root / "too_expensive"
            other.mkdir()
            try:
                provider._reserve_spend(other / "state.json", {}, 1.0)
            except RuntimeError as exc:
                assert "spend guard" in str(exc)
            else:
                raise AssertionError("spend guard accepted a reservation above the cap")

            try:
                OpenAIBatchProvider(cache_dir=str(root), terminal_batch_retries=1)
            except ValueError as exc:
                assert "partial successes" in str(exc)
            else:
                raise AssertionError("unsafe automatic terminal retry was accepted")

            missing_prior = root / "missing_prior"
            missing_prior.mkdir()
            missing_provider = OpenAIBatchProvider(
                cache_dir=str(missing_prior),
                spend_cap_usd=87.23,
                project_spend_cap_usd=100.0,
                known_external_spend_usd=0.07056656,
                reserved_downstream_spend_usd=12.0,
                minimum_prior_cache_spend_usd=42.2395,
            )
            missing_state = missing_prior / "new"
            missing_state.mkdir()
            try:
                missing_provider._reserve_spend(missing_state / "state.json", {}, 0.1)
            except RuntimeError as exc:
                assert "prior-spend floor" in str(exc)
            else:
                raise AssertionError("missing prior spend metadata was accepted")
        finally:
            if old is None:
                os.environ.pop("OPENAI_API_KEY_2", None)
            else:
                os.environ["OPENAI_API_KEY_2"] = old


def test_terminal_batch_evidence_and_secret_guard() -> None:
    with tempfile.TemporaryDirectory() as temp:
        run = Path(temp) / ("a" * 64)
        run.mkdir()
        (run / "input.jsonl").write_text('{"custom_id":"r0000000"}\n')
        (run / "state.json").write_text(
            json.dumps(
                {
                    "batch_id": "batch_failed",
                    "status": "expired",
                    "pessimistic_cost_usd": 1.25,
                    "request_counts": {"completed": 10, "failed": 2},
                }
            )
        )
        (run / "output.terminal.jsonl").write_text("partial billable output\n")
        (run / "errors.terminal.jsonl").write_text("terminal errors\n")
        manifest = build_incomplete_manifest(run, b"unit-secret-not-present")
        assert manifest["status"] == "expired"
        assert manifest["cost_reconciled"] is False
        assert manifest["pessimistic_cost_usd"] == 1.25
        assert {row["path"] for row in manifest["files"]} >= {
            "input.jsonl",
            "state.json",
            "output.terminal.jsonl",
            "errors.terminal.jsonl",
        }
        (run / "errors.terminal.jsonl").write_text("leaked unit-secret")
        try:
            build_incomplete_manifest(run, b"unit-secret")
        except RuntimeError as exc:
            assert "credential" in str(exc)
        else:
            raise AssertionError("credential-containing API artifact was accepted")
        assert not archive_stop_ready(
            stop_requested=True,
            pending_completed=[],
            unarchived_incomplete=[run.name],
            latest_ledger_sha="old",
            current_ledger_sha="new",
        )
        assert archive_stop_ready(
            stop_requested=True,
            pending_completed=[],
            unarchived_incomplete=[],
            latest_ledger_sha="same",
            current_ledger_sha="same",
        )


def test_sft_resume_config_and_archive_contract() -> None:
    fresh = Namespace(
        config="recipe.yaml",
        resume_from=None,
        mode="av",
        lr=1e-4,
        num_steps=774,
        wandb_project="project",
        wandb_name="name",
        wandb_group="group",
        wandb_tags="plausible,matched",
    )
    resumed = Namespace(**{**vars(fresh), "resume_from": "/run/iter_0000020"})
    assert stable_run_config(fresh) == stable_run_config(resumed)
    previous = {**stable_run_config(fresh), "resume_from": None}
    assert_resume_config_matches(previous, resumed)
    mock_init = Mock()
    mock_init(**wandb_init_kwargs(resumed, "run-id"))
    kwargs = mock_init.call_args.kwargs
    assert "resume_from" not in kwargs["config"]
    assert kwargs["resume"] == "must"
    drifted = Namespace(**{**vars(resumed), "lr": 2e-4})
    try:
        assert_resume_config_matches(previous, drifted)
    except RuntimeError as exc:
        assert "resume config drift for lr" in str(exc)
    else:
        raise AssertionError("substantive SFT resume recipe drift was accepted")

    with tempfile.TemporaryDirectory() as temp:
        checkpoint = Path(temp) / "iter_0000020"
        checkpoint.mkdir()
        for name, payload in {
            "adapter_model.safetensors": b"adapter",
            "trainer_state.pt": b"optimizer-and-cursor",
            "run_config.yaml": b"num_steps: 774\n",
            "nla_meta.yaml": b"kind: nla_dataset\n",
            "training.log": b"through step 20\n",
        }.items():
            (checkpoint / name).write_bytes(payload)
        (checkpoint / "provenance").mkdir()
        (checkpoint / "provenance" / "code_snapshot.tar.gz").write_bytes(b"code")
        (checkpoint / "metrics.jsonl").write_text(
            "".join(json.dumps({"step": step}) + "\n" for step in range(20))
        )
        (checkpoint / "COMPLETE.json").write_text(
            json.dumps({"mode": "av", "completed_steps": 20})
        )
        manifest = build_sft_manifest(checkpoint, "av")
        assert manifest["step"] == 20
        assert build_sft_manifest(checkpoint, "av") == manifest
        assert {row["path"] for row in manifest["files"]} >= {
            "trainer_state.pt",
            "training.log",
            "metrics.jsonl",
        }


def test_immutable_checkpoint_transaction() -> None:
    from nla.utils.checkpoint_history import publish_immutable_checkpoint_snapshot

    with tempfile.TemporaryDirectory() as temp:
        save = Path(temp) / "run"
        actor = save / "iter_000020"
        (actor / "reference").mkdir(parents=True)
        (actor / "adapter_model.safetensors").write_bytes(b"actor")
        (actor / "reference" / "adapter_model.safetensors").write_bytes(b"reference")
        critic = save / "critic_latest"
        critic.mkdir()
        (critic / "value_head.safetensors").write_bytes(b"critic")
        (critic / "saved_at_step.txt").write_text("20")
        (save / "optim_latest.pt").write_bytes(b"optimizer")
        (save / "metrics.jsonl").write_text(
            "".join(json.dumps({"step": step, "fve_pct": step}) + "\n" for step in range(20))
        )
        (save / "run_config.yaml").write_text("num_steps: 780\n")
        (save / "nla_meta.yaml").write_text("kind: nla_dataset\n")
        (save / "provenance").mkdir()
        (save / "provenance" / "code_snapshot.tar.gz").write_bytes(b"code")
        log = save / "live.log"
        log.write_text("through step 20\n")

        snapshot = publish_immutable_checkpoint_snapshot(
            save, actor, 20, run_log_path=str(log)
        )
        assert snapshot == save / "history" / "step_000020"
        assert (snapshot / "COMPLETE.json").is_file()
        assert os.stat(actor / "adapter_model.safetensors").st_ino == os.stat(
            snapshot / "actor" / "adapter_model.safetensors"
        ).st_ino
        assert os.stat(save / "optim_latest.pt").st_ino == os.stat(snapshot / "optim.pt").st_ino
        assert os.stat(save / "metrics.jsonl").st_ino != os.stat(snapshot / "metrics.jsonl").st_ino
        with (save / "metrics.jsonl").open("a") as handle:
            handle.write(json.dumps({"step": 20}) + "\n")
        snap_rows = (snapshot / "metrics.jsonl").read_text().splitlines()
        assert len(snap_rows) == 20
        manifest = build_manifest(snapshot, 20)
        assert manifest["step"] == 20 and manifest["file_count"] >= 9
        assert build_manifest(snapshot, 20) == manifest


def main() -> None:
    tests = [
        test_exact_reference_selection,
        test_payload_comparison_rejects_changed_activation,
        test_format_cleaning,
        test_project_budget_floor_cap_and_cumulative_ledger,
        test_terminal_batch_evidence_and_secret_guard,
        test_sft_resume_config_and_archive_contract,
        test_immutable_checkpoint_transaction,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")


if __name__ == "__main__":
    main()
