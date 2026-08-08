# -*- coding: utf-8 -*-
"""v0.9 修復一驗收：補位方向三態語義（施工單 2026-08-07）。

核心公式：
    自帶率 = answered_hit / (gap_hit + answered_hit)
    從未出現（appeared == 0）→ unknown，不得判定
    樣本不足（appeared < 3）→ insufficient，不下判斷

兩個舊公式各錯一半：
    gap/total     → 十次都漏預算被判成「強項，不需要問」
    1 - gap/total → 從沒聊過的面向被判成「自帶率 100% 強項」
"""
import unittest

from src.middleware.baseline import (
    CAT_BLIND,
    CAT_CONTEXTUAL,
    CAT_INSUFFICIENT,
    CAT_STRENGTH,
    CAT_UNKNOWN,
    classify_categories,
    compute_baseline,
    compute_strategy_brief,
)


def _mk(gap=None, answered=None, i=0):
    return {
        "type": "start",
        "event_id": f"e{i}",
        "start_data": {
            "user_input": f"第{i}件事",
            "goal": f"目標{i}",
            "gap_categories": gap or [],
            "answered_categories": answered or [],
            "topic": "t",
        },
    }


class BaselineSemanticsTests(unittest.TestCase):
    def test_frequently_missed_not_judged_as_strength(self):
        """常缺不得判成強項：10 筆全 gap=budget、無 answered。"""
        rows = [_mk(gap=["budget"], i=i) for i in range(10)]
        baseline = compute_baseline(rows)
        states = classify_categories(baseline)
        self.assertEqual(states["budget"], CAT_BLIND)
        brief = compute_strategy_brief(baseline)
        self.assertNotIn("強項", brief)
        self.assertNotIn("不需要問", brief)
        self.assertIn("盲區", brief)

    def test_frequently_answered_is_strength(self):
        """常自帶應判成強項：10 筆全 answered=budget、無 gap。"""
        rows = [_mk(answered=["budget"], i=i) for i in range(10)]
        baseline = compute_baseline(rows)
        states = classify_categories(baseline)
        self.assertEqual(states["budget"], CAT_STRENGTH)
        brief = compute_strategy_brief(baseline)
        self.assertIn("強項", brief)
        self.assertIn("預算", brief)

    def test_never_appeared_is_unknown_and_absent_from_brief(self):
        """無資料不得判成盲區：10 筆均不含 venue。"""
        rows = [_mk(gap=["budget"], i=i) for i in range(10)]
        baseline = compute_baseline(rows)
        states = classify_categories(baseline)
        self.assertEqual(states["venue"], CAT_UNKNOWN)
        brief = compute_strategy_brief(baseline)
        self.assertNotIn("場地", brief)

    def test_mixed_case_rate_uses_appeared_denominator(self):
        """混合情況：7 筆 gap、3 筆 answered → 自帶率 3/10 = 30% → 情境相關。"""
        rows = [_mk(gap=["budget"], i=i) for i in range(7)]
        rows += [_mk(answered=["budget"], i=7 + i) for i in range(3)]
        baseline = compute_baseline(rows)
        states = classify_categories(baseline)
        # 30% 正好在情境相關下緣（>= 0.3）
        self.assertEqual(states["budget"], CAT_CONTEXTUAL)

    def test_insufficient_samples_not_judged(self):
        """樣本不足不下判斷：2 筆 gap → 不進強項／盲區清單。"""
        rows = [_mk(gap=["budget"], i=i) for i in range(2)]
        baseline = compute_baseline(rows)
        states = classify_categories(baseline)
        self.assertEqual(states["budget"], CAT_INSUFFICIENT)
        brief = compute_strategy_brief(baseline)
        self.assertNotIn("預算", brief)
        self.assertNotIn("強項", brief)
        self.assertNotIn("盲區", brief)

    def test_legacy_endpoints_without_answered_field(self):
        """舊端點沒有 answered_categories 欄位 → 視為 0，不崩潰。"""
        rows = []
        for i in range(5):
            r = _mk(gap=["budget"], i=i)
            del r["start_data"]["answered_categories"]
            rows.append(r)
        baseline = compute_baseline(rows)
        states = classify_categories(baseline)
        self.assertEqual(states["budget"], CAT_BLIND)


if __name__ == "__main__":
    unittest.main()
