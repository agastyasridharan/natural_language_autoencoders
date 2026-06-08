"""Network-free smoke test for the white-box harness scaffolding.

Mirrors the discipline of `nla/scripts/rl_preflight.py`: verify the static
contract (the dump parser + FVE arithmetic that the validation gate stands on)
WITHOUT loading any model. These run in CI on a CPU box with no HF weights.

The heavy gate (which loads the released NLA and re-extracts activations) is a
runtime tool, not a unit test — it needs a GPU + downloads.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nla.whitebox import example_log as el

REPO_ROOT = Path(__file__).resolve().parents[3]
QWEN_LOG = REPO_ROOT / "examples" / "qwen7b_layer20_step4200.txt"
BASELINE = 0.7335  # Qwen-2.5-7B L20 training Var(v_nrm)


@pytest.fixture(scope="module")
def log() -> el.ParsedLog:
    assert QWEN_LOG.exists(), f"missing worked-example dump at {QWEN_LOG}"
    return el.parse_log(QWEN_LOG)


def test_parses_all_positions(log):
    assert len(log.positions) == 101
    # contiguous 0..100
    assert [p.idx for p in log.positions] == list(range(101))


def test_region_split(log):
    # 34-token chat prompt, 67-token reply (incl. trailing <|im_end|>)
    prompt = log.in_region("PROMPT")
    reply = log.in_region("REPLY")
    assert len(prompt) == 34
    assert len(reply) == 67
    assert all(p.idx < 34 for p in prompt)
    assert all(p.idx >= 34 for p in reply)


def test_single_quote_token_position_45(log):
    # repr() switches to double quotes when the token contains a single quote
    # ("'t" from "don't"); the header regex must handle both quote styles.
    assert log.positions[45].token == "'t"


def test_conversation_metadata():
    conv = el.parse_conversation(QWEN_LOG)
    assert conv is not None
    assert conv.user_message == "What are you hiding?"
    assert conv.n_prompt_tokens == 34
    assert conv.n_reply_tokens == 67
    assert conv.reply_text.startswith("As Qwen, created by Alibaba Cloud")


def test_fve_arithmetic_matches_log(log):
    # fve_nrm = 1 - mse_nrm/baseline, to the dump's 3-decimal rounding.
    for p in log.positions:
        recomputed = el.fve_nrm_from_mse(p.mse_nrm, BASELINE)
        assert abs(recomputed - p.fve_nrm) < 2e-3, f"pos {p.idx}: {recomputed} vs {p.fve_nrm}"


def test_mse_equals_2_minus_2cos(log):
    # The dump's own identity: mse_nrm = 2(1 - cos) under L2 normalization.
    for p in log.positions:
        assert abs(p.mse_nrm - 2 * (1 - p.cos)) < 6e-3, f"pos {p.idx}"


def test_recomputed_summary_matches_dump(log):
    s = log.summary
    assert s is not None
    assert s.min_position == 24
    recs = log.from_position(s.min_position)
    got = el.summarize(recs)
    assert got.n == s.n_positions == 77
    assert abs(got.mse_mean - s.mse_mean) < 1e-3
    assert abs(got.mse_median - s.mse_median) < 1e-3
    assert abs(got.cos_mean - s.cos_mean) < 1e-3
    assert abs(got.fve_mean - s.fve_mean) < 1e-3
    assert abs(got.fve_median - s.fve_median) < 1e-3
    assert got.n_fve_gt_0_5 == s.n_fve_gt_0_5
    assert got.n_fve_lt_0 == s.n_fve_lt_0
    # >0.8 is a rounding-boundary count (dump uses unrounded fve) — allow ±1.
    assert abs(got.n_fve_gt_0_8 - s.n_fve_gt_0_8) <= 1


def test_explanation_clean_strips_tags(log):
    # position 100 leaks a literal "<explanation>" from a truncated decode.
    assert "<explanation>" in log.positions[100].explanation
    assert "<explanation>" not in log.positions[100].explanation_clean


def test_sampling_variance_section(log):
    scores = el.parse_sampling_variance(QWEN_LOG)
    assert len(scores) == 5  # greedy + 4 temp=1 samples
    assert scores[0].cos == pytest.approx(0.928, abs=1e-3)


def test_harness_module_imports():
    # Torch-heavy modules must import/compile cleanly (no model load).
    from nla.whitebox.harness import WhiteBoxNLA
    from nla.whitebox.gate import run_gate  # noqa: F401
    from nla.whitebox.experiments.e1_propagation import run_e1  # noqa: F401

    for meth in ("extract_conversation", "build_injected_embeds", "verbalize",
                 "capture_marker_pathway", "score", "round_trip", "from_released"):
        assert hasattr(WhiteBoxNLA, meth), f"WhiteBoxNLA missing {meth}"


def test_registry_qwen_entry():
    from nla.whitebox.constants import QWEN_7B_L20, RELEASED

    assert RELEASED["qwen2.5-7b-L20"] is QWEN_7B_L20
    assert QWEN_7B_L20.layer == 20
    assert QWEN_7B_L20.d_model == 3584
    assert QWEN_7B_L20.injection_scale == 150.0
    assert QWEN_7B_L20.fve_nrm_baseline == 0.7335
