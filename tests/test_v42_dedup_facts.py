"""v4.2：已確認事實狀態機 與 _deduplicate_gap_questions 單元測試。

涵蓋 Copilot review 的兩項修正（src 修正見 commit d8f9650）：
- 去重逐題拆開比對（避免整輪問題併成一個 token 集合造成短問題誤判）。
- goal/motivation 單值覆蓋、constraints 去重 append、使用者補充只在本輪
  無維度更新時才補（避免重複注入）。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.main import (  # noqa: E402
    _deduplicate_gap_questions,
    _make_confirmed_facts_state,
    _render_confirmed_facts,
    _update_confirmed_facts,
)


class ConfirmedFactsStateTests(unittest.TestCase):
    def test_short_confirmation_renders_empty(self):
        """短確認、無維度內容 → render 為空。"""
        state = _make_confirmed_facts_state()
        _update_confirmed_facts(
            state, {"goal": "", "motivation": "", "constraints": [], "user_input": ""}, "好"
        )
        self.assertEqual(_render_confirmed_facts(state), [])

    def test_renders_each_dimension(self):
        """有 goal/motivation/constraints → 各自渲染成事實短句。"""
        state = _make_confirmed_facts_state()
        _update_confirmed_facts(
            state,
            {
                "goal": "辦生日派對",
                "motivation": "慶生",
                "constraints": ["預算一萬", "30 人"],
                "user_input": "原始輸入",
            },
            "確認",
        )
        facts = _render_confirmed_facts(state)
        self.assertIn("目標：辦生日派對", facts)
        self.assertIn("動機：慶生", facts)
        self.assertIn("條件：預算一萬", facts)
        self.assertIn("條件：30 人", facts)

    def test_goal_kept_single_latest_across_rounds(self):
        """goal 跨輪 refine → 只保留最新值，不堆疊多筆「目標：…」。"""
        state = _make_confirmed_facts_state()
        _update_confirmed_facts(
            state, {"goal": "辦派對", "user_input": "x", "constraints": []}, "確認"
        )
        _update_confirmed_facts(
            state, {"goal": "辦生日派對慶祝", "user_input": "x", "constraints": []}, "確認"
        )
        facts = _render_confirmed_facts(state)
        goal_lines = [f for f in facts if f.startswith("目標：")]
        self.assertEqual(goal_lines, ["目標：辦生日派對慶祝"])

    def test_user_note_skipped_when_dimension_changed(self):
        """本輪有維度更新時，不再重複注入原始回覆。"""
        state = _make_confirmed_facts_state()
        _update_confirmed_facts(
            state,
            {"goal": "辦生日派對", "user_input": "原始", "constraints": []},
            "我想辦一個有儀式感的生日派對",
        )
        facts = _render_confirmed_facts(state)
        self.assertFalse(any(f.startswith("使用者補充") for f in facts))

    def test_user_note_added_when_no_dimension_changed(self):
        """本輪 goal/motivation/constraints 都沒變 → 才補使用者補充。"""
        state = _make_confirmed_facts_state()
        # 第一輪先確立 goal
        _update_confirmed_facts(
            state, {"goal": "辦派對", "user_input": "x", "constraints": []}, "確認"
        )
        # 第二輪 goal 不變、無新維度，但使用者長補充
        _update_confirmed_facts(
            state,
            {"goal": "辦派對", "user_input": "x", "constraints": []},
            "這個我想再多補充一些細節",
        )
        facts = _render_confirmed_facts(state)
        self.assertIn("使用者補充：這個我想再多補充一些細節", facts)

    def test_constraints_dedup_append(self):
        """constraints 跨輪去重 append。"""
        state = _make_confirmed_facts_state()
        _update_confirmed_facts(
            state, {"goal": "x", "user_input": "y", "constraints": ["預算一萬"]}, "確認"
        )
        _update_confirmed_facts(
            state,
            {"goal": "x", "user_input": "y", "constraints": ["預算一萬", "30 人"]},
            "確認",
        )
        facts = _render_confirmed_facts(state)
        constraint_lines = [f for f in facts if f.startswith("條件：")]
        self.assertEqual(constraint_lines, ["條件：預算一萬", "條件：30 人"])


class DeduplicateGapQuestionsTests(unittest.TestCase):
    def test_exact_duplicate_dropped(self):
        prior = ["第 1 輪提問：預算多少；使用者回：確認"]
        kept = _deduplicate_gap_questions(["預算多少"], prior)
        self.assertEqual(kept, [])

    def test_unrelated_question_kept(self):
        prior = ["第 1 輪提問：預算多少；使用者回：確認"]
        kept = _deduplicate_gap_questions(["改成線上"], prior)
        self.assertEqual(kept, [0])

    def test_no_prior_keeps_all(self):
        kept = _deduplicate_gap_questions(["預算多少", "場地在哪"], [])
        self.assertEqual(kept, [0, 1])

    def test_cross_question_overlap_not_false_dropped(self):
        """逐題拆開的關鍵修正：新問題分別與兩道舊問題各重疊一點，
        合併集合下會誤判重複（>0.6），逐題比對則應保留。

        v0.8 斷詞改用繁中大詞典後，「場地在哪」只剩單一 token（「在哪」
        被停用詞/長度規則濾掉），單 token 舊題會讓 min-len 重疊率直接打滿；
        改用兩題各含兩個實詞 token 的例句，維持原測試意圖。"""
        prior = ["第 1 輪提問：預算大概多少錢、場地要選哪個地區；使用者回：確認"]
        kept = _deduplicate_gap_questions(["預算場地"], prior)
        self.assertEqual(kept, [0])


if __name__ == "__main__":
    unittest.main()
