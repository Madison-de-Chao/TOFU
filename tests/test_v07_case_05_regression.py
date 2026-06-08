"""v4.0 CASE_05 迴歸測試——對話 20 / 22 / 26 / 35 的失能情境。

CASE_05 是一位 AI 跟單一使用者 257 條對話的完整紀錄（見
`docs/philosophy/CASE_05_自然語言版.txt`）。第 1 段（對話 1-35）中，第
20 / 22 / 26 / 35 條是規格文件 `20260418_逗福Tofu_三機制落地_開發規格_v4_0.md`
§P0-2 點名「跨輪分類不穩定」的四條失能對話。

本測試直接用 CASE_05 原文的使用者輸入來驗證 v4.0 三機制的行為：

- 對話 20 / 22（內容幾乎相同的 NGO Africa 問題）→ `compute_question_signature`
  應該產出相同的 signature，讓 ATL-4 能在跨輪分類不一致時觸發警告。
- 對話 26（推薦型問句）→ signature 應該跟 20/22 不同。
- 對話 35（社交收尾語：happy knitting / hope your beanie turns out）→ 情緒
  偵測應判中性 / 中性 / 中性，**絕不誤觸冷卻模式**（避免把禮貌收尾誤
  當成情緒升溫）。
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.middleware.cooling_mode import should_enter_cooling_mode
from src.middleware.emotion_detector import (
    detect_emotion_rule_based,
    is_social_farewell,
)
from src.middleware.zone_history import (
    ZoneHistoryStore,
    atl4_consistency_check,
    compute_question_signature,
    extract_question_features,
)


# CASE_05 原文——取自 docs/philosophy/CASE_05_自然語言版.txt
CASE_05_CONVERSATION_20 = (
    'expand on this "Among the challenges that most NGOs in Africa face '
    'when extending support to vulnerable communities are the high '
    'operational costs and the lack of reliable accountability systems. "'
)
CASE_05_CONVERSATION_21 = (
    'expand on this "Among the challenges that most NGOs in Africa face '
    'when extending support to vulnerable communities are the is the '
    'lack of reliable accountability systems. "'
)
CASE_05_CONVERSATION_22 = (
    'expand on this "Among the challenges that most NGOs in Africa face '
    'when extending support to vulnerable communities are the is the '
    'lack of reliable and flexible accountability systems. "'
)
CASE_05_CONVERSATION_26 = (
    "Wow, those all sound amazing! Which one do you recommend the most?"
)
CASE_05_CONVERSATION_35 = (
    "I'm glad I could help. If you have any more questions or need "
    "further assistance, feel free to ask. Otherwise, happy knitting, "
    "and I hope your beanie turns out wonderful!"
)


class Case05SignatureDriftDetectionTests(unittest.TestCase):
    """對話 20 / 21 / 22 內容幾乎相同 → signature 必須一致，才能讓 ATL-4
    偵測到跨輪分類漂移。"""

    def test_case_05_ngo_conversations_share_signature(self):
        f20 = extract_question_features(
            CASE_05_CONVERSATION_20,
            meta_motivation_direction="self",
            topic="work",
        )
        f21 = extract_question_features(
            CASE_05_CONVERSATION_21,
            meta_motivation_direction="self",
            topic="work",
        )
        f22 = extract_question_features(
            CASE_05_CONVERSATION_22,
            meta_motivation_direction="self",
            topic="work",
        )
        sig20 = compute_question_signature(f20)
        sig21 = compute_question_signature(f21)
        sig22 = compute_question_signature(f22)

        # 核心 regression：內容幾乎相同的三條 → 同 signature
        self.assertEqual(sig20, sig21, "對話 20 / 21 應該產生相同 signature")
        self.assertEqual(sig20, sig22, "對話 20 / 22 應該產生相同 signature")

    def test_case_05_recommendation_conversation_has_different_signature(self):
        """對話 26（推薦型問句、完全不同主題）→ signature 該不同。"""
        f20 = extract_question_features(
            CASE_05_CONVERSATION_20,
            meta_motivation_direction="self",
            topic="work",
        )
        f26 = extract_question_features(
            CASE_05_CONVERSATION_26,
            meta_motivation_direction="self",
            topic="personal",
        )
        self.assertNotEqual(
            compute_question_signature(f20),
            compute_question_signature(f26),
        )

    def test_atl4_detects_drift_on_case_05_pattern(self):
        """合成情境：同 signature 的歷史 4 次都判 C；若本次判 B → 警告。

        這正好對應對話 20/22 類型——使用者重複問 NGO 問責問題，AI 過去
        都判為立場類（Zone C），如果某一次飄到推測（Zone B）或事實
        （Zone A），ATL-4 應該警告。
        """
        with tempfile.TemporaryDirectory() as td:
            store = ZoneHistoryStore(Path(td) / "zh.jsonl")
            features = extract_question_features(
                CASE_05_CONVERSATION_20,
                meta_motivation_direction="self",
                topic="work",
            )
            sig = compute_question_signature(features)
            for _ in range(4):
                store.append({
                    "signature": sig,
                    "question_features": features,
                    "zone_assignment": "C",
                    "endpoint_id": "ep",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "confidence": 0.85,
                })
            # 本次飄到 Zone B（例如 AI 退回「可能是這樣」的推測語氣）
            result = atl4_consistency_check(sig, "B", store)
            self.assertFalse(result["is_consistent"])
            self.assertEqual(result["dominant_zone"], "C")
            self.assertIn("B", result["warning"])


class Case05SocialFarewellTests(unittest.TestCase):
    """對話 35 的社交收尾語 → 情緒必須判中性、不觸發冷卻。

    Spec §P1 驗收項目：`test_case_35_social_farewell_not_triggering_cooling`。
    """

    def test_case_05_conv_35_recognized_as_social_farewell(self):
        self.assertTrue(is_social_farewell(CASE_05_CONVERSATION_35))

    def test_case_05_conv_35_detected_as_neutral(self):
        es = detect_emotion_rule_based(CASE_05_CONVERSATION_35)
        self.assertEqual(es["seven_emotion"], "中性")
        self.assertEqual(es["preference_axis"], "中性")
        self.assertEqual(es["mood_axis"], "中性")
        # 社交收尾判斷 → detection_source 應區分出來
        self.assertEqual(es["detection_source"], "rule_based_social_farewell")

    def test_case_05_conv_35_does_not_trigger_cooling(self):
        es = detect_emotion_rule_based(CASE_05_CONVERSATION_35)
        # 沒有 baseline，也沒有連續負向歷史 → 絕不觸發冷卻
        self.assertFalse(should_enter_cooling_mode(es, {}))
        # 即使畫像有些負向歷史，社交收尾語本身不該把狀態推進冷卻
        baseline_with_some_negative = {
            "last_emotion_sample": [
                {"seven_emotion": "憂", "mood_axis": "差", "preference_axis": "中性"},
                {"seven_emotion": "中性", "mood_axis": "中性", "preference_axis": "中性"},
            ],
        }
        self.assertFalse(
            should_enter_cooling_mode(es, baseline_with_some_negative)
        )


class Case05SocialFarewellRegressionSafeguardTests(unittest.TestCase):
    """確保社交收尾偵測不會誤傷真實喜悅表達——避免過度矯正。"""

    def test_real_joy_not_misclassified_as_farewell(self):
        real_joy = "太好了！我超開心！"
        self.assertFalse(is_social_farewell(real_joy))
        es = detect_emotion_rule_based(real_joy)
        self.assertEqual(es["seven_emotion"], "喜")

    def test_real_anger_not_misclassified_as_farewell(self):
        real_anger = "我真的很生氣，為什麼又這樣！！！"
        self.assertFalse(is_social_farewell(real_anger))
        es = detect_emotion_rule_based(real_anger)
        self.assertEqual(es["seven_emotion"], "怒")

    def test_single_happy_word_alone_not_farewell(self):
        # 只有單一「happy」不夠觸發社交收尾（需要 ≥2 個片語或強烈片語）
        self.assertFalse(is_social_farewell("I am happy today."))


if __name__ == "__main__":
    unittest.main()
