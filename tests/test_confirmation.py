"""confirmation 模組的否定詞偵測與偏差偵測測試。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.middleware.confirmation import (
    check_action_specificity,
    check_falsification,
    check_source_traceability,
    classify_confirmation,
    detect_deviation,
    is_modification,
)


class NegationDetectionTests(unittest.TestCase):
    def test_explicit_negation_is_modification(self):
        self.assertTrue(is_modification("不對，是 30 人"))
        self.assertTrue(is_modification("錯了，預算只有 5000"))
        self.assertTrue(is_modification("不是這樣，我想要室外"))
        self.assertTrue(is_modification("wrong, it should be next week"))
        self.assertTrue(is_modification("No, the venue is TBD"))

    def test_no_negation_is_confirmation(self):
        self.assertFalse(is_modification("預算大概 5000"))
        self.assertFalse(is_modification("對，場地要室內的"))
        self.assertFalse(is_modification("還不錯，就這樣吧"))  # 『不錯』不該被誤判
        self.assertFalse(is_modification("沒問題"))

    def test_classify_confirmation_returns_correct_pair(self):
        conf, mod = classify_confirmation("預算 5000")
        self.assertEqual(conf, "confirmed")
        self.assertEqual(mod, "")

        conf, mod = classify_confirmation("不對，應該是 30 人")
        self.assertEqual(conf, "modified")
        self.assertIn("30", mod)


class DeviationDetectionTests(unittest.TestCase):
    def test_matching_keywords_no_deviation(self):
        self.assertEqual(
            detect_deviation("辦一場生日派對", "建議先找適合的生日派對場地"),
            "",
        )

    def test_no_overlap_flags_deviation(self):
        result = detect_deviation("辦一場生日派對", "建議先準備好一套健身計畫")
        self.assertIn("偏離", result)

    def test_empty_goal_or_result_returns_empty(self):
        self.assertEqual(detect_deviation("", "something"), "")
        self.assertEqual(detect_deviation("goal", ""), "")


class CheckActionSpecificityTests(unittest.TestCase):
    def test_specific_with_deliverable_and_time_window(self):
        text = "請在 3 天內完成一份報告，包含來源引用。"
        result = check_action_specificity(text)
        self.assertTrue(result["is_specific"])
        self.assertTrue(result["has_time_window"])

    def test_specific_with_all_three_elements(self):
        text = "產出一份企劃書（3 頁），7 天內交稿，至少 5 個引用來源。"
        result = check_action_specificity(text)
        self.assertTrue(result["has_deliverable"])
        self.assertTrue(result["has_time_window"])
        self.assertTrue(result["has_verification"])
        self.assertTrue(result["is_specific"])

    def test_vague_blacklist_phrase_not_specific(self):
        text = "建議進一步研究後持續觀察市場變化。"
        result = check_action_specificity(text)
        self.assertFalse(result["is_specific"])
        self.assertGreater(len(result["vague_phrases"]), 0)

    def test_empty_text_not_specific(self):
        result = check_action_specificity("")
        self.assertFalse(result["is_specific"])

    def test_insufficient_elements_not_specific(self):
        # Only one element present (time window only)
        text = "請在 7 天內完成。"
        result = check_action_specificity(text)
        self.assertTrue(result["has_time_window"])
        self.assertFalse(result["has_deliverable"])
        self.assertFalse(result["has_verification"])
        self.assertFalse(result["is_specific"])


class CheckFalsificationTests(unittest.TestCase):
    def test_specific_falsification_condition(self):
        text = "若 60 天內未見改善，本建議即告失效。"
        result = check_falsification(text)
        self.assertTrue(result["has_falsification"])
        self.assertFalse(result["is_generic"])
        self.assertTrue(result["has_specific_content"])

    def test_generic_falsification_flagged(self):
        text = "若有新證據出現，本結論可能改變。"
        result = check_falsification(text)
        self.assertTrue(result["has_falsification"])
        self.assertTrue(result["is_generic"])
        self.assertIn("若有新證據", result["generic_phrases"])

    def test_no_falsification(self):
        text = "這是一個普通的建議，沒有條件句。"
        result = check_falsification(text)
        self.assertFalse(result["has_falsification"])

    def test_empty_text(self):
        result = check_falsification("")
        self.assertFalse(result["has_falsification"])
        self.assertFalse(result["is_generic"])


class CheckSourceTraceabilityTests(unittest.TestCase):
    def test_zone_a_year_without_named_source_is_not_traceable(self):
        text = "根據 2023 年的研究報告，結果顯示..."
        result = check_source_traceability(text, zone="A")
        self.assertTrue(result["needs_check"])
        self.assertFalse(result["has_specific_source"])
        self.assertTrue(result["should_downgrade"])

    def test_zone_a_with_url(self):
        text = "詳見 https://example.com/report"
        result = check_source_traceability(text, zone="A")
        self.assertTrue(result["has_specific_source"])
        self.assertFalse(result["should_downgrade"])

    def test_zone_a_vague_source_should_downgrade(self):
        text = "根據研究，這個方向是正確的。"
        result = check_source_traceability(text, zone="A")
        self.assertTrue(result["needs_check"])
        self.assertFalse(result["has_specific_source"])
        self.assertTrue(result["should_downgrade"])
        self.assertIn("根據研究", result["vague_sources"])

    def test_zone_b_skips_check(self):
        text = "根據研究，這個方向是正確的。"
        result = check_source_traceability(text, zone="B")
        self.assertFalse(result["needs_check"])
        self.assertFalse(result["should_downgrade"])

    def test_zone_c_skips_check(self):
        text = "根據研究，這個方向是正確的。"
        result = check_source_traceability(text, zone="C")
        self.assertFalse(result["needs_check"])
        self.assertFalse(result["should_downgrade"])

    def test_zone_a_empty_text_should_downgrade(self):
        result = check_source_traceability("", zone="A")
        self.assertTrue(result["needs_check"])
        self.assertTrue(result["should_downgrade"])


if __name__ == "__main__":
    unittest.main()
