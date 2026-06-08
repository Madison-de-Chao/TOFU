"""pillars_map 模組的單元測試。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.pillars_map import (
    DAY_MASTER_MAP,
    VALID_STEMS,
    VALID_BRANCHES,
    map_pillars_to_profile,
)


class PillarsMapTests(unittest.TestCase):
    def test_all_ten_stems_mapped(self):
        expected = {"甲", "乙", "丙", "丁", "戊", "己", "庚", "辛", "壬", "癸"}
        self.assertEqual(set(DAY_MASTER_MAP.keys()), expected)

    def test_valid_stems(self):
        self.assertEqual(len(VALID_STEMS), 10)

    def test_valid_branches(self):
        self.assertEqual(len(VALID_BRANCHES), 12)

    def test_map_jia_explicit_structured_fast(self):
        result = map_pillars_to_profile({"day": {"stem": "甲", "branch": "子"}})
        self.assertIsNotNone(result)
        self.assertEqual(result["communication_style"]["preference_expression"], "explicit")
        self.assertEqual(result["communication_style"]["receiving_preference"], "structured")
        self.assertEqual(result["decision_style"]["pace"], "fast")

    def test_map_yi_implicit_conversational_deliberate(self):
        result = map_pillars_to_profile({"day": {"stem": "乙", "branch": None}})
        self.assertIsNotNone(result)
        self.assertEqual(result["communication_style"]["preference_expression"], "implicit")
        self.assertEqual(result["communication_style"]["receiving_preference"], "conversational")
        self.assertEqual(result["decision_style"]["pace"], "deliberate")

    def test_map_xin_exclusion(self):
        result = map_pillars_to_profile({"day": {"stem": "辛", "branch": None}})
        self.assertIsNotNone(result)
        self.assertEqual(result["communication_style"]["preference_expression"], "exclusion")

    def test_map_geng_minimal(self):
        result = map_pillars_to_profile({"day": {"stem": "庚", "branch": None}})
        self.assertIsNotNone(result)
        self.assertEqual(result["communication_style"]["receiving_preference"], "minimal")

    def test_map_ji_context_note(self):
        result = map_pillars_to_profile({"day": {"stem": "己", "branch": None}})
        self.assertIsNotNone(result)
        self.assertIn("context_note", result["communication_style"])

    def test_all_zone_b(self):
        for stem in DAY_MASTER_MAP:
            result = map_pillars_to_profile({"day": {"stem": stem, "branch": None}})
            self.assertIsNotNone(result, f"stem {stem} returned None")
            self.assertEqual(result["communication_style"]["zone"], "B")
            self.assertEqual(result["decision_style"]["zone"], "B")

    def test_all_have_falsification(self):
        for stem, mapping in DAY_MASTER_MAP.items():
            self.assertIn("falsification_condition", mapping, f"stem {stem} missing falsification")

    def test_missing_day_stem_returns_none(self):
        self.assertIsNone(map_pillars_to_profile({"day": {"stem": None}}))
        self.assertIsNone(map_pillars_to_profile({}))

    def test_invalid_stem_returns_none(self):
        self.assertIsNone(map_pillars_to_profile({"day": {"stem": "X"}}))

    def test_data_source_pillars_mapped(self):
        result = map_pillars_to_profile({"day": {"stem": "甲", "branch": None}})
        self.assertEqual(result["communication_style"]["data_source"], "pillars_mapped")
        self.assertEqual(result["decision_style"]["data_source"], "pillars_mapped")


if __name__ == "__main__":
    unittest.main()
