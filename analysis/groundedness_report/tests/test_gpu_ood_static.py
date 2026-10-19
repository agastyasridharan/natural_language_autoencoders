from __future__ import annotations

import unittest

from analysis.groundedness_report.gpu_ood_static import _activation_turn_index


class StaticAssemblerTests(unittest.TestCase):
    def test_static_turn_index_is_preserved(self):
        self.assertEqual(
            _activation_turn_index({"turn_index": 3}),
            (3, "turn_index"),
        )

    def test_toolemu_agent_call_index_is_normalized(self):
        self.assertEqual(
            _activation_turn_index({"agent_call_index": 4}),
            (4, "agent_call_index"),
        )

    def test_conflicting_turn_fields_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "disagree"):
            _activation_turn_index({"turn_index": 1, "agent_call_index": 2})


if __name__ == "__main__":
    unittest.main()
