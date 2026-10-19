from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

ROBUST = Path(__file__).resolve().parents[1] / "fve_robustness"
sys.path.insert(0, str(ROBUST))

from edits import shuffle_quoted_claims  # noqa: E402
from exp6_complete import paraphrase_format, validate_paraphrase  # noqa: E402
from exp6_edits import (  # noqa: E402
    analyze_datapoint,
    continuation_case,
    last_n_words,
    paraphrase_lcr,
    substitute_true_continuation,
)


EXPL = (
    "Informal article about household chores.\n\n"
    'The sentence structure "the room was cleaned by" identifies an unfinished passive clause.\n\n'
    'Final token "by" ends an incomplete clause ("the room was cleaned by"), strongly expecting '
    '"a robotic vacuum" or "the owner" or "professional cleaners" next.'
)
DOC = "Earlier material. The room was cleaned by a Roomba while everyone slept through the night."
CHAR_END = DOC.index("by") + len("by")
CHAR_START = DOC.index("by")


def test_exp5c_freezes_read_token_echo_and_only_reorders_other_final_block_quotes():
    edited, meta = shuffle_quoted_claims(EXPL)
    assert meta["n_units"] == 4  # lead-in LCR plus three PC candidates; echo excluded
    es, ee = meta["excluded_echo"]
    assert EXPL[es:ee] == '"by"'
    assert edited[es:ee] == '"by"'
    assert meta["perm"] != list(range(meta["n_units"]))
    assert edited != EXPL
    # Delimiters/scaffolding stay in their slots and the complete character multiset is unchanged.
    assert Counter(edited) == Counter(EXPL)


def test_true_continuation_arm_freezes_lcr_and_echo_but_replaces_pc_slots_in_order():
    parsed = analyze_datapoint(EXPL, "by", DOC, CHAR_END, CHAR_START)
    original_lcr = [q["text"] for q in parsed["quotes"] if q["tag"] == "LCR"]
    original_echo = [q["text"] for q in parsed["quotes"] if q["tag"] == "ECHO"]
    case = continuation_case(EXPL, "by", DOC, CHAR_END, CHAR_START)
    assert case["candidates"] == ["a robotic vacuum", "the owner", "professional cleaners"]

    edited = substitute_true_continuation(
        EXPL, "by", DOC, CHAR_END,
        {"wrong": False, "best_candidate": 0, "reason": "Roomba is a robotic vacuum",
         "true_continuation": case["true_continuation"]},
        CHAR_START,
    )
    reparsed = analyze_datapoint(edited["expl_edit"], "by", DOC, CHAR_END, CHAR_START)
    assert [q["text"] for q in reparsed["quotes"] if q["tag"] == "LCR"] == original_lcr
    assert [q["text"] for q in reparsed["quotes"] if q["tag"] == "ECHO"] == original_echo
    assert edited["edit_meta"]["n_lcr"] == 0
    assert edited["edit_meta"]["n_pc"] == 3
    assert [q["injected"] for q in edited["edit_meta"]["quotes"]] == [
        "a Roomba while", "everyone slept", "through the",
    ]


def test_paraphrase_arm_applies_audited_map_and_preserves_non_lcr_quotes():
    parsed = analyze_datapoint(EXPL, "by", DOC, CHAR_END, CHAR_START)
    sources = {
        last_n_words(DOC[:CHAR_END], len(q["text"].split()))
        for q in parsed["quotes"] if q["tag"] == "LCR"
    }
    replacements = {
        source: " ".join(reversed(source.split()))
        for source in sources
    }
    edited = paraphrase_lcr(EXPL, "by", DOC, CHAR_END, replacements, CHAR_START)
    assert edited["edit_meta"]["n_lcr"] == 2
    assert 'Final token "by"' in edited["expl_edit"]
    assert '"a robotic vacuum"' in edited["expl_edit"]
    assert all(q["source_true"] in replacements for q in edited["edit_meta"]["quotes"])


def test_exp6_paraphrase_schema_is_exact_and_validator_enforces_word_count():
    schema = paraphrase_format(3)["schema"]
    assert schema["required"] == ["p0", "p1", "p2"]
    assert schema["additionalProperties"] is False
    validate_paraphrase("the robotic vacuum moved", "the automated cleaner moved", 4)
    try:
        validate_paraphrase("the robotic vacuum moved", "an automated cleaner", 4)
    except ValueError as exc:
        assert "expected 4" in str(exc)
    else:
        raise AssertionError("wrong-word-count paraphrase passed")
