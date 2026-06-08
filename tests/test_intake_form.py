"""intake_form 模組測試。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.middleware.baseline import compute_baseline
from src.middleware.intake_form import (
    compose_natural_input,
    generate_form,
    render_form,
    should_generate,
)


def _mk_start(gap_categories: list[str]) -> dict:
    return {
        "type": "start",
        "event_id": "evt",
        "start_data": {
            "user_input": "demo",
            "tofu_understanding": "",
            "gap_questions": [],
            "user_confirmation": "confirmed",
            "user_modifications": "",
            "goal": "demo",
            "motivation": "",
            "constraints": [],
            "gap_categories": gap_categories,
        },
    }


class IntakeFormTests(unittest.TestCase):
    def test_should_generate_threshold(self):
        self.assertFalse(should_generate(4))
        self.assertTrue(should_generate(5))
        self.assertTrue(should_generate(10))

    def test_form_contains_high_frequency_fields(self):
        endpoints = [
            _mk_start(["budget", "time"]),
            _mk_start(["budget", "time"]),
            _mk_start(["budget", "venue"]),
            _mk_start(["budget"]),
            _mk_start(["time"]),
        ]
        baseline = compute_baseline(endpoints)
        fields = generate_form(baseline)
        keys = [f["key"] for f in fields]
        self.assertEqual(keys[0], "goal")  # goal 永遠第一
        self.assertIn("budget", keys)
        self.assertIn("time", keys)

    def test_form_evolves_with_more_data(self):
        before = compute_baseline([_mk_start(["budget"]) for _ in range(5)])
        after = compute_baseline(
            [_mk_start(["budget"]) for _ in range(5)]
            + [_mk_start(["deadline"]) for _ in range(5)]
        )
        before_keys = [f["key"] for f in generate_form(before)]
        after_keys = [f["key"] for f in generate_form(after)]
        self.assertNotEqual(before_keys, after_keys)
        self.assertIn("deadline", after_keys)

    def test_fallback_when_baseline_sparse(self):
        baseline = compute_baseline([])
        fields = generate_form(baseline)
        # 冷啟動時沒有高頻欄位，要 fallback 到通用欄位
        self.assertGreaterEqual(len(fields), 4)
        self.assertEqual(fields[0]["key"], "goal")

    def test_render_form_contains_labels(self):
        fields = [
            {"key": "goal", "label": "目標（你這次想完成什麼？）"},
            {"key": "budget", "label": "預算"},
        ]
        out = render_form(fields)
        self.assertIn("目標", out)
        self.assertIn("預算", out)
        self.assertIn("/skip", out)

    def test_compose_natural_input(self):
        fields = [
            {"key": "goal", "label": "目標"},
            {"key": "budget", "label": "預算"},
            {"key": "time", "label": "時間"},
        ]
        natural = compose_natural_input(
            {"goal": "辦尾牙", "budget": "三萬", "time": "1/15"},
            fields,
        )
        self.assertIn("辦尾牙", natural)
        self.assertIn("三萬", natural)
        self.assertIn("1/15", natural)


if __name__ == "__main__":
    unittest.main()
