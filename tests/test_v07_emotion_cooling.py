"""v4.0 P1：情緒三軸偵測 + 冷卻模式單元測試。

對應規格 20260418_逗福Tofu_三機制落地_開發規格_v4_0.md §P1 的 18 項
驗收測試清單（CASE_05 迴歸需真實對話資料，這裡以合成樣本涵蓋邏輯）。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.middleware.cooling_mode import (
    COOLING_MODE_PROMPT_ADDENDUM,
    append_emotion_sample,
    should_enter_cooling_mode,
)
from src.middleware.emotion_detector import (
    detect_emotion_rule_based,
    detect_mood,
    detect_preference_axis,
    detect_seven_emotion,
    normalize_emotion_state,
)


class RuleBasedDetectionTests(unittest.TestCase):
    def test_rule_based_detect_joy(self):
        es = detect_emotion_rule_based("太好了！我超喜歡這個方案 😄")
        self.assertEqual(es["seven_emotion"], "喜")
        self.assertEqual(es["preference_axis"], "+")

    def test_rule_based_detect_anger(self):
        es = detect_emotion_rule_based("幹，這個超煩!!!")
        self.assertEqual(es["seven_emotion"], "怒")
        self.assertEqual(es["mood_axis"], "差")

    def test_rule_based_detect_worry(self):
        es = detect_emotion_rule_based("我好擔心會來不及")
        self.assertEqual(es["seven_emotion"], "憂")

    def test_rule_based_detect_sadness(self):
        es = detect_emotion_rule_based("有點難過...")
        self.assertEqual(es["seven_emotion"], "悲")

    def test_rule_based_neutral_when_no_signal(self):
        es = detect_emotion_rule_based("請推薦台北週六的拉麵店")
        self.assertEqual(es["seven_emotion"], "中性")
        self.assertEqual(es["preference_axis"], "中性")
        self.assertEqual(es["mood_axis"], "中性")

    def test_rule_based_negative_priority_over_positive(self):
        # 句中同時有喜跟怒 → 取怒（優先序原則）
        es = detect_emotion_rule_based("超開心又超生氣")
        self.assertEqual(es["seven_emotion"], "怒")

    def test_detect_mood_by_punctuation(self):
        self.assertEqual(detect_mood("到底什麼意思？？？？"), "差")
        self.assertEqual(detect_mood("太棒了！！！！"), "差")  # 3+ !  判差（高情緒強度）

    def test_detect_mood_by_profanity(self):
        self.assertEqual(detect_mood("他媽的搞什麼"), "差")

    def test_detect_mood_positive_emoji(self):
        self.assertEqual(detect_mood("還不錯喔 :)"), "好")

    def test_detect_preference_positive(self):
        self.assertEqual(detect_preference_axis("我很喜歡這個"), "+")

    def test_detect_preference_negative(self):
        self.assertEqual(detect_preference_axis("我討厭那種風格"), "-")

    def test_detect_seven_emotion_on_empty(self):
        self.assertEqual(detect_seven_emotion(""), "中性")


class NormalizeEmotionStateTests(unittest.TestCase):
    def test_normalize_valid_llm_output(self):
        raw = {
            "seven_emotion": "怒",
            "preference_axis": "-",
            "mood_axis": "差",
            "confidence": 0.8,
        }
        es = normalize_emotion_state(raw)
        self.assertEqual(es["seven_emotion"], "怒")
        self.assertEqual(es["confidence"], 0.8)
        self.assertEqual(es["detection_source"], "llm_inferred")

    def test_normalize_invalid_values_replaced(self):
        raw = {"seven_emotion": "暴怒", "preference_axis": "♥", "mood_axis": "nope"}
        es = normalize_emotion_state(raw)
        self.assertEqual(es["seven_emotion"], "中性")
        self.assertEqual(es["preference_axis"], "中性")
        self.assertEqual(es["mood_axis"], "中性")

    def test_normalize_non_dict_returns_none(self):
        self.assertIsNone(normalize_emotion_state("not a dict"))
        self.assertIsNone(normalize_emotion_state(None))

    def test_normalize_clamps_confidence(self):
        es = normalize_emotion_state({"confidence": 2.5})
        self.assertEqual(es["confidence"], 1.0)
        es = normalize_emotion_state({"confidence": -1})
        self.assertEqual(es["confidence"], 0.0)


class CoolingModeTriggersTests(unittest.TestCase):
    def test_triggers_on_angry_bad_mood(self):
        emotion = {"seven_emotion": "怒", "preference_axis": "-", "mood_axis": "差"}
        self.assertTrue(should_enter_cooling_mode(emotion, {}))

    def test_triggers_on_sad_bad_mood(self):
        emotion = {"seven_emotion": "悲", "preference_axis": "中性", "mood_axis": "差"}
        self.assertTrue(should_enter_cooling_mode(emotion, {}))

    def test_not_triggered_on_neutral_input(self):
        emotion = {"seven_emotion": "中性", "preference_axis": "中性", "mood_axis": "中性"}
        self.assertFalse(should_enter_cooling_mode(emotion, {}))

    def test_triggers_on_3_consecutive_same_negative(self):
        emotion = {"seven_emotion": "喜", "preference_axis": "中性", "mood_axis": "好"}
        baseline = {
            "last_emotion_sample": [
                {"seven_emotion": "憂", "mood_axis": "差"},
                {"seven_emotion": "憂", "mood_axis": "差"},
                {"seven_emotion": "憂", "mood_axis": "差"},
            ],
        }
        self.assertTrue(should_enter_cooling_mode(emotion, baseline))

    def test_not_triggered_on_3_different_negatives(self):
        emotion = {"seven_emotion": "中性", "preference_axis": "中性", "mood_axis": "中性"}
        baseline = {
            "last_emotion_sample": [
                {"seven_emotion": "怒", "mood_axis": "差"},
                {"seven_emotion": "悲", "mood_axis": "差"},
                {"seven_emotion": "憂", "mood_axis": "差"},
            ],
        }
        self.assertFalse(should_enter_cooling_mode(emotion, baseline))

    def test_triggers_on_preference_flip_with_mood_drop(self):
        emotion = {"seven_emotion": "中性", "preference_axis": "-", "mood_axis": "差"}
        baseline = {
            "last_emotion_sample": [
                {"preference_axis": "+", "mood_axis": "好", "seven_emotion": "喜"},
            ],
        }
        self.assertTrue(should_enter_cooling_mode(emotion, baseline))

    def test_no_trigger_on_none_emotion_state(self):
        self.assertFalse(should_enter_cooling_mode(None, {}))

    def test_case_35_social_farewell_not_triggering(self):
        # 「謝謝，今天聊得很愉快」這類收尾語 → 中性/中性/中性 → 不觸發
        es = detect_emotion_rule_based("謝謝，今天聊得很愉快")
        self.assertEqual(es["seven_emotion"], "中性")
        self.assertFalse(should_enter_cooling_mode(es, {}))


class CoolingModePromptTests(unittest.TestCase):
    def test_prompt_excludes_advice_language(self):
        # prompt 明確列出「不做：給建議」
        self.assertIn("不做", COOLING_MODE_PROMPT_ADDENDUM)
        self.assertIn("給建議", COOLING_MODE_PROMPT_ADDENDUM)

    def test_prompt_includes_closing_question(self):
        self.assertIn("這樣的理解", COOLING_MODE_PROMPT_ADDENDUM)


class EmotionBaselineTests(unittest.TestCase):
    def test_append_emotion_sample_limits_to_max(self):
        baseline: dict = {}
        for i in range(15):
            append_emotion_sample(
                baseline,
                {"seven_emotion": "中性", "preference_axis": "中性", "mood_axis": "中性"},
                max_samples=10,
            )
        self.assertEqual(len(baseline["last_emotion_sample"]), 10)

    def test_append_emotion_sample_preserves_order(self):
        baseline: dict = {}
        for i, em in enumerate(["怒", "悲", "憂"]):
            append_emotion_sample(
                baseline,
                {"seven_emotion": em, "preference_axis": "中性", "mood_axis": "差"},
            )
        emotions = [s["seven_emotion"] for s in baseline["last_emotion_sample"]]
        self.assertEqual(emotions, ["怒", "悲", "憂"])

    def test_append_to_non_dict_baseline_returns_dict(self):
        baseline = append_emotion_sample(
            None,  # type: ignore[arg-type]
            {"seven_emotion": "喜", "preference_axis": "+", "mood_axis": "好"},
        )
        self.assertIn("last_emotion_sample", baseline)


if __name__ == "__main__":
    unittest.main()
