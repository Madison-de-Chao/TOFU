"""baseline 模組的單元測試——純程式碼邏輯，不碰 LLM。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.middleware.baseline import (
    compute_baseline,
    high_frequency_fields,
    is_mature,
    ALLOWED_GAP_CATEGORIES,
)


def _mk_start(
    goal: str,
    constraints: list[str],
    gap_categories: list[str],
    modified: bool = False,
    modifications: str = "",
) -> dict:
    return {
        "type": "start",
        "event_id": "evt",
        "start_data": {
            "user_input": goal,
            "tofu_understanding": "",
            "gap_questions": [],
            "user_confirmation": "modified" if modified else "confirmed",
            "user_modifications": modifications,
            "goal": goal,
            "motivation": "",
            "constraints": constraints,
            "gap_categories": gap_categories,
        },
    }


class BaselineTests(unittest.TestCase):
    def test_empty_baseline(self):
        baseline = compute_baseline([])
        self.assertEqual(baseline["total_starts"], 0)
        self.assertEqual(baseline["top_goal_keywords"], [])
        self.assertEqual(baseline["top_gap_categories"], [])
        self.assertFalse(is_mature(baseline))

    def test_counts_gap_categories(self):
        endpoints = [
            _mk_start("辦派對", ["預算一萬"], ["budget", "venue"]),
            _mk_start("辦婚禮", ["預算五萬"], ["budget", "time"]),
            _mk_start("辦研討會", [], ["budget", "audience"]),
        ]
        baseline = compute_baseline(endpoints)
        self.assertEqual(baseline["total_starts"], 3)
        self.assertIn("budget", baseline["top_gap_categories"])
        # budget 出現 3 次，應該是第一名
        self.assertEqual(baseline["top_gap_categories"][0], "budget")
        self.assertTrue(is_mature(baseline))

    def test_high_frequency_fields_threshold(self):
        endpoints = [
            _mk_start("a", [], ["budget", "time"]),
            _mk_start("b", [], ["budget"]),
            _mk_start("c", [], ["budget", "venue"]),
            _mk_start("d", [], ["time"]),
        ]
        baseline = compute_baseline(endpoints)
        # budget 3/4 = 0.75 → 高頻；time 2/4 = 0.5 → 高頻；venue 1/4 = 0.25 → 不夠
        high = high_frequency_fields(baseline, ratio_threshold=0.3)
        self.assertIn("budget", high)
        self.assertIn("time", high)
        self.assertNotIn("venue", high)

    def test_modification_stats(self):
        endpoints = [
            _mk_start(
                "辦派對", [], ["budget"], modified=True, modifications="其實是 30 人不是 20"
            ),
            _mk_start("辦婚禮", [], ["budget"]),
        ]
        baseline = compute_baseline(endpoints)
        self.assertAlmostEqual(baseline["modified_ratio"], 0.5)
        self.assertTrue(len(baseline["top_corrections"]) > 0)

    def test_allowed_categories_shape(self):
        # 確保字典清單沒被意外改動
        self.assertIn("budget", ALLOWED_GAP_CATEGORIES)
        self.assertIn("time", ALLOWED_GAP_CATEGORIES)
        self.assertIn("venue", ALLOWED_GAP_CATEGORIES)


if __name__ == "__main__":
    unittest.main()
