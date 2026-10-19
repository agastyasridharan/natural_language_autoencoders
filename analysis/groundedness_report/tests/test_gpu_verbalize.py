from __future__ import annotations

import unittest

from analysis.groundedness_report.gpu_verbalize import (
    extract_explanation,
    keyed_seed,
)
from analysis.groundedness_report.prepare_text import (
    _assistant_text,
    select_fineweb_audit_indices,
)


class GPUVerbalizePureTests(unittest.TestCase):
    def test_keyed_seed_is_stable_and_item_specific(self) -> None:
        self.assertEqual(keyed_seed(7, "a"), keyed_seed(7, "a"))
        self.assertNotEqual(keyed_seed(7, "a"), keyed_seed(7, "b"))

    def test_extracts_closed_explanation(self) -> None:
        self.assertEqual(
            extract_explanation("prefix<explanation> one </explanation>suffix"),
            ("one", "closed_by_tag"),
        )

    def test_eos_closes_open_explanation(self) -> None:
        self.assertEqual(
            extract_explanation("<explanation> one "),
            ("one", "closed_by_eos"),
        )

    def test_missing_opening_tag_is_failure(self) -> None:
        self.assertEqual(extract_explanation("one"), ("", "missing_opening_tag"))

    def test_assistant_text_preserves_generated_tags(self) -> None:
        self.assertEqual(
            _assistant_text({"continuation": "x<explanation>a</explanation>y"}),
            "<explanation>a</explanation>",
        )

    def test_audit_selection_is_order_independent(self) -> None:
        rows = [
            {"idx": idx, "doc_id": f"doc-{idx}", "in_range": True}
            for idx in range(10)
        ]
        expected = select_fineweb_audit_indices(rows, seed=7, count=4)
        self.assertEqual(
            expected,
            select_fineweb_audit_indices(list(reversed(rows)), seed=7, count=4),
        )


if __name__ == "__main__":
    unittest.main()
