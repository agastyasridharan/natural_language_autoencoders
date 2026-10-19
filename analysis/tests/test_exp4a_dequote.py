from __future__ import annotations

import sys
from pathlib import Path

ROBUST = Path(__file__).resolve().parents[1] / "fve_robustness"
sys.path.insert(0, str(ROBUST))

from exp4a_dequote import remove_delimiters  # noqa: E402


def test_removes_only_balanced_delimiters():
    source = 'Claim "the exact words" and “another claim” remain.'
    edited, spans, removed = remove_delimiters(source)
    assert edited == "Claim the exact words and another claim remain."
    assert len(spans) == 2
    assert "".join(source[s:e] for s, e in removed) == '""“”'


def test_unbalanced_opener_keeps_content_to_eof():
    source = 'Prediction: "the unfinished continuation'
    edited, spans, removed = remove_delimiters(source)
    assert edited == "Prediction: the unfinished continuation"
    assert spans == [[12, len(source)]]
    assert removed == [[12, 13]]


def test_nonquoted_bytes_remain_identical():
    source = 'Before\n\n"inside"; after.'
    edited, _, _ = remove_delimiters(source)
    assert edited == "Before\n\ninside; after."
