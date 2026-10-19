from __future__ import annotations

import sys
from pathlib import Path

ROBUST = Path(__file__).resolve().parents[1] / "fve_robustness"
sys.path.insert(0, str(ROBUST))

from exp4_rewrite import (  # noqa: E402
    apply_replacements,
    canonicalize_irreducible_spans,
    is_irreducible_anchor,
    max_verbatim_run,
    parse_replacement_payload,
    quote_rows,
    surface_change,
    validate_replacements,
)


def test_apply_replacements_preserves_nonquoted_bytes_and_tracks_added_spans():
    source = 'Genre says "a formal report". Final token “Jos” expects "Joseon Dynasty".'
    rows = quote_rows(source)
    replacements = {
        0: "the register of a structured technical account",
        1: "an incomplete Korean proper noun beginning with J and ending in an s sound",
        2: "the full name of the Korean royal era",
    }
    edited, added = apply_replacements(source, rows, replacements)
    assert edited == (
        "Genre says the register of a structured technical account. Final token "
        "an incomplete Korean proper noun beginning with J and ending in an s sound expects "
        "the full name of the Korean royal era."
    )
    assert [edited[s:e] for s, e in added] == list(replacements.values())


def test_rejects_quote_marks_and_verbatim_long_phrase():
    source = 'Prediction: "the model will answer with forty two".'
    rows = quote_rows(source)
    try:
        validate_replacements(rows, {0: '"the answer is forty-two"'})
    except ValueError as exc:
        assert "quote mark" in str(exc)
    else:
        raise AssertionError("quoted replacement was accepted")
    try:
        validate_replacements(rows, {0: "the model will answer with forty two in this case"})
    except ValueError as exc:
        assert "full quoted phrase" in str(exc)
    else:
        raise AssertionError("verbatim phrase was accepted")


def test_surface_change_detects_real_rephrasing():
    assert surface_change("an automated cleaning device", "a robotic appliance that cleans floors") > 0.4


def test_atomic_token_must_be_embedded_in_descriptive_prose():
    rows = quote_rows('Final token "of" ends an incomplete phrase.')
    validate_replacements(rows, {0: "a preposition marking possession or association"})
    validate_replacements(rows, {0: "the preposition of"})
    try:
        validate_replacements(rows, {0: "of"})
    except ValueError as exc:
        assert "insufficiently rephrased" in str(exc)
    else:
        raise AssertionError("bare atomic token was accepted")


def test_short_incomplete_phrase_must_not_survive_verbatim():
    rows = quote_rows('The clause "the area of" remains incomplete.')
    validate_replacements(rows, {0: "an unfinished reference to a particular region"})
    try:
        validate_replacements(rows, {0: "the area of a still unspecified thing"})
    except ValueError as exc:
        assert "retains the full quoted phrase" in str(exc)
    else:
        raise AssertionError("verbatim short phrase was accepted")


def test_allows_short_run_but_rejects_full_multiword_span():
    rows = quote_rows('The text says "the investigation region remains within the area".')
    validate_replacements(rows, {0: "the investigation region stays inside a specified zone"})
    try:
        validate_replacements(rows, {0: "the investigation region remains within the area as stated"})
    except ValueError as exc:
        assert "full quoted phrase" in str(exc)
    else:
        raise AssertionError("full verbatim span was accepted")


def test_max_verbatim_run():
    assert max_verbatim_run("alpha beta gamma delta", "other beta gamma words") == 2


def test_irreducible_identifier_can_anchor_reworked_prose():
    assert is_irreducible_anchor("Adobe Generator")
    rows = quote_rows('The tool is "Adobe Generator".')
    validate_replacements(rows, {0: "the Adobe Generator product"})


def test_short_unchanged_fragment_gets_countable_literal_wrapper():
    rows = quote_rows('The citation begins "Source: CNN".')
    replacements = {0: "Source: CNN"}
    canonicalize_irreducible_spans(rows, replacements)
    assert replacements[0] == "the literal fragment Source: CNN"
    validate_replacements(rows, replacements)


def test_non_latin_label_gets_countable_literal_wrapper():
    rows = quote_rows('Product "Фильтры механической очистки воды B-0".')
    replacements = {0: "Фильтры механической очистки воды B-0"}
    canonicalize_irreducible_spans(rows, replacements)
    assert replacements[0].startswith("the literal foreign-language label ")
    validate_replacements(rows, replacements)


def test_parses_observed_stringified_malformed_tool_payload():
    payload = '''[
      {"id": 0, "text": Although studies promote the supplement"},
      {"id": 1, "text": "a properly quoted value"}
    ]'''
    assert parse_replacement_payload(payload) == [
        {"id": 0, "text": "Although studies promote the supplement"},
        {"id": 1, "text": "a properly quoted value"},
    ]
