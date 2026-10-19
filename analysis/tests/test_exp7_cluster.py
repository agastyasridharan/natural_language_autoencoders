from __future__ import annotations

import sys
from pathlib import Path

ROBUST = Path(__file__).resolve().parents[1] / "fve_robustness"
sys.path.insert(0, str(ROBUST))

from exp7_cluster import surface_clusters  # noqa: E402


def test_surface_clusters_are_word_level_not_character_level():
    # These strings have character ROUGE-L 0.9, but assert unrelated one-token surfaces.
    assert surface_clusters(["abcdefghij", "abcdefghiX"]) == [[0], [1]]


def test_surface_clusters_accept_exact_threshold_and_duplicates():
    base = "one two three four five six seven eight nine ten"
    one_substitution = "one two three four five six seven eight nine changed"
    assert surface_clusters([base, one_substitution, base]) == [[0, 1, 2]]


def test_surface_clusters_split_when_overlap_upper_bound_is_below_threshold():
    assert surface_clusters([
        "one two three four five six seven eight nine ten",
        "one two three four five six seven eight changed altered",
    ]) == [[0], [1]]
