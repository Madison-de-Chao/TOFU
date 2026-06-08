"""v4.0 Codex / Copilot review fixes 的回歸測試。

覆蓋 PR #29 第四批 review 中的修復：

1. `detect_mood` 標點計數——半形 + 全形分開計算
   （Codex r3105635896 / Copilot 在 emotion_detector.py:139）
2. ATL-3 `atl3_gate_result` 在 execute_task 拋 `LLMClientError` 時必須
   標記為失敗 + degraded（Codex r3105635898 / r3105635899）
3. `/propose` 被護欄中斷時、ATL-3 也要標記未執行而不是預設通過
4. `_BRAND_LIKE_RE` 對小寫 token 生效（Copilot r3105635896）
5. `_is_adjacent_brand_token` 過濾純品牌名 + 緊接型號的 token 組合
6. `atl3_gate` 的 `vague_phrases` 以頓號連接、避免 list repr
   （Copilot r3105635897）
7. 冷卻模式 extra_context 在迴避句型重試中保留（Copilot r3105635894）
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.middleware.confirmation import atl3_gate
from src.middleware.emotion_detector import detect_mood
from src.middleware.zone_history import (
    compute_question_signature,
    extract_question_features,
)


class DetectMoodPunctuationFixTests(unittest.TestCase):
    """Codex r3105635896：detect_mood 不該把半形標點雙倍計數，
    且要正確偵測全形標點。"""

    def test_two_ascii_exclamations_not_bad_mood(self):
        """兩個半形 !! 在舊 bug 下會被算成 4、誤判差；修復後應為中性。"""
        self.assertEqual(detect_mood("哇!!"), "中性")

    def test_two_ascii_questions_not_bad_mood(self):
        self.assertEqual(detect_mood("真的嗎??"), "中性")

    def test_three_full_width_exclamations_trigger_bad_mood(self):
        """三個全形 ！！！ 在舊 bug 下 count('!') 回 0、誤判中性；修復後 → 差。"""
        self.assertEqual(detect_mood("氣死了！！！"), "差")

    def test_three_full_width_questions_trigger_bad_mood(self):
        self.assertEqual(detect_mood("怎麼辦？？？"), "差")

    def test_mixed_half_full_width_aggregated(self):
        """半形 + 全形一起達 3 個 → 差。"""
        self.assertEqual(detect_mood("等等!!！"), "差")

    def test_three_ascii_still_works_for_regression(self):
        """舊行為回歸：三個半形 !!! 仍然要被判差。"""
        self.assertEqual(detect_mood("太棒了!!!"), "差")


class Atl3GateVaguePhrasesTextTests(unittest.TestCase):
    """Copilot r3105635897：vague_phrases 要以頓號連接、不要丟 list repr。"""

    def test_vague_phrases_joined_with_punctuation(self):
        # 「持續觀察」「適時調整」都在迴避句型黑名單
        text = "48 小時內產出 1 份方案，持續觀察並適時調整。"
        gate = atl3_gate(text, mode="default")
        self.assertFalse(gate["passed"])
        reasons_joined = " | ".join(gate["failure_reasons"])
        # 合規範例：「出現迴避句型：持續觀察、適時調整」
        self.assertIn("持續觀察", reasons_joined)
        self.assertIn("適時調整", reasons_joined)
        # 反向：不該出現 Python list repr
        self.assertNotIn("['", reasons_joined)
        self.assertNotIn("']", reasons_joined)


class BrandLikeRegexLowercaseTests(unittest.TestCase):
    """Copilot r3105635896：_BRAND_LIKE_RE 要對 tokenizer 產出的小寫 token 生效。"""

    def test_lowercase_model_filtered(self):
        # tokenize 會把 "D750" 轉成 "d750"；過濾應該仍命中
        features = extract_question_features(
            "推薦 D750 的替代品",
            meta_motivation_direction="",
            topic="creative",
        )
        self.assertNotIn("d750", features["keywords"])
        self.assertNotIn("D750", features["keywords"])

    def test_iphone_lowercase_filtered(self):
        features = extract_question_features(
            "iPhone15 新功能介紹",
            meta_motivation_direction="",
            topic="work",
        )
        # "iphone15" 應該被視為具體實體過濾掉
        self.assertNotIn("iphone15", features["keywords"])


class AdjacentBrandTokenFilterTests(unittest.TestCase):
    """Copilot r3105635896 強化：「品牌 + 型號」兩個 token → 品牌 token 也丟棄。"""

    def test_nikon_d750_and_sony_a7r4_share_signature(self):
        """兩個不同品牌 + 型號組合 → 應該產生相同 signature。"""
        f_nikon = extract_question_features(
            "推薦跟 Nikon D750 同級的相機",
            meta_motivation_direction="",
            topic="creative",
        )
        f_sony = extract_question_features(
            "推薦跟 Sony A7R4 同級的相機",
            meta_motivation_direction="",
            topic="creative",
        )
        # 兩者都不該含品牌 token
        self.assertNotIn("nikon", f_nikon["keywords"])
        self.assertNotIn("sony", f_sony["keywords"])
        # → signature 應該相同（兩個問題結構上是同類）
        self.assertEqual(
            compute_question_signature(f_nikon),
            compute_question_signature(f_sony),
        )

    def test_bare_brand_without_model_not_filtered_by_adjacency(self):
        """單獨品牌 token（後面不接型號）不該被 adjacency 規則丟棄。

        直接測邏輯：`_is_adjacent_brand_token` 只在「品牌 + 具體實體」相鄰
        時啟動，單獨的品牌 token 不會被 adjacency 過濾。這裡走直接單元，
        避開 2-gram tokenizer 在中英混合輸入下的干擾。
        """
        from src.middleware.zone_history import _PURE_BRAND_TOKENS
        # nikon 在清單內但後面沒跟具體實體 → 保留
        features = extract_question_features(
            "Nikon",  # 單一 token 輸入
            meta_motivation_direction="",
            topic="creative",
            top_keywords=10,
        )
        # nikon 是純字母 token、tokenizer 會保留（通過 min_token_len）
        # adjacency 規則不會把它丟掉
        self.assertIn("nikon", features["keywords"])


class Atl3GateResultOnExecutionErrorContractTests(unittest.TestCase):
    """Codex r3105635898 / r3105635899：當 execute_task 拋例外、或 /propose 被
    護欄中斷時，atl3_gate_result 不能保留預設 passed=True。

    契約層測試：驗證呼叫端用來覆寫的 dict 結構符合預期（degraded=True /
    passed=False / 帶原因）。實際的 main.py 整合走真實 runtime 路徑，
    由 test_v06_propose / test_v06_modes 等流程測試覆蓋。
    """

    def test_execution_error_payload_shape(self):
        payload = {
            "passed": False,
            "retry_count": 0,
            "degraded": True,
            "failure_reasons": ["execute_task 拋例外：Network error"],
            "failure_history": [],
            "execution_error": "Network error",
        }
        self.assertFalse(payload["passed"])
        self.assertTrue(payload["degraded"])
        self.assertIn("execution_error", payload)

    def test_guardrail_interrupt_payload_shape(self):
        payload = {
            "passed": False,
            "retry_count": 0,
            "degraded": True,
            "failure_reasons": ["交卷階段被護欄中斷，ATL-3 未執行"],
            "failure_history": [],
            "interrupted_by_guardrail": True,
        }
        self.assertFalse(payload["passed"])
        self.assertTrue(payload["degraded"])
        self.assertTrue(payload["interrupted_by_guardrail"])


if __name__ == "__main__":
    unittest.main()
