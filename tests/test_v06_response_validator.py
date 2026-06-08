"""v0.6 PR-G step 2：response_validator 單元測試。

MOMO 2026-04-16 bug report 根治修法：迴避句型偵測 + 重試 hint 產生器。

本檔只測 `src/middleware/response_validator.py`（純字串處理）；
validator 接入 run_one_interaction / run_propose_mode 的整合測試在
test_v06_propose.py 和 test_v06_modes.py。
"""

from __future__ import annotations

import unittest

from src.middleware.response_validator import (
    DEGRADED_WARNING_PREFIX,
    ENFORCED_MODES,
    EVASION_PATTERNS,
    build_retry_hint,
    detect_evasion_phrases,
    is_evasive_response,
    mark_degraded,
)


class DetectEvasionPhrasesTests(unittest.TestCase):
    def test_empty_text_returns_empty(self):
        self.assertEqual(detect_evasion_phrases(""), [])
        self.assertEqual(detect_evasion_phrases(None), [])  # type: ignore[arg-type]

    def test_clean_text_returns_empty(self):
        text = "建議 4/20 前訂台北→仙台機票，預算約 NT$9 萬。"
        self.assertEqual(detect_evasion_phrases(text), [])

    def test_detects_single_evasion_phrase(self):
        text = "你可以思考一下你的預算需求"
        hits = detect_evasion_phrases(text)
        self.assertIn("你可以思考一下", hits)

    def test_detects_multiple_distinct_phrases(self):
        text = (
            "你可以思考一下你想去哪裡。建議你先確認預算範圍，"
            "然後請你再補充一些偏好資訊。"
        )
        hits = detect_evasion_phrases(text)
        # 至少應抓到 3 種不同的句型
        self.assertIn("你可以思考一下", hits)
        self.assertIn("建議你先確認", hits)
        self.assertIn("請你再補", hits)

    def test_dedupes_repeated_phrase(self):
        text = "你可以思考一下 A。你可以思考一下 B。"
        hits = detect_evasion_phrases(text)
        self.assertEqual(hits.count("你可以思考一下"), 1)

    def test_case_insensitive_for_english_words(self):
        """Google / google / GOOGLE 都該命中『你可以 google』。"""
        text_lower = "你可以 google 看看這個地點"
        text_upper = "你可以 Google 看看這個地點"
        self.assertIn("你可以 google", detect_evasion_phrases(text_lower))
        self.assertIn("你可以 google", detect_evasion_phrases(text_upper))

    def test_detects_fake_proposal_pattern(self):
        """MOMO 原始觀察的「我建議你先從 X 選出…」偽裝反問。"""
        text = "我建議你先從東北亞深度文化體驗類型中選出至少三種"
        hits = detect_evasion_phrases(text)
        self.assertIn("我建議你先從", hits)


class IsEvasiveResponseTests(unittest.TestCase):
    def test_enforced_modes_flag_evasive_text(self):
        evasive = "你可以思考一下"
        for mode in ("default", "free", "propose_final"):
            self.assertTrue(
                is_evasive_response(evasive, mode),
                msg=f"mode={mode} 應視為 evasive",
            )

    def test_non_enforced_modes_never_flag(self):
        """/risk 和 propose 收資訊輪允許出現部分類似句型，不該被 validator 擋。"""
        evasive = "建議你先確認場地容量後再辦"  # 這是 /risk 正常輸出
        self.assertFalse(is_evasive_response(evasive, "risk"))
        self.assertFalse(is_evasive_response(evasive, "propose"))
        self.assertFalse(is_evasive_response(evasive, "check"))

    def test_unknown_mode_returns_false(self):
        self.assertFalse(is_evasive_response("你可以思考", "unknown_mode"))

    def test_clean_response_returns_false(self):
        clean = "建議 4/20 前訂台北→仙台機票，預算約 NT$9 萬。"
        self.assertFalse(is_evasive_response(clean, "default"))
        self.assertFalse(is_evasive_response(clean, "free"))
        self.assertFalse(is_evasive_response(clean, "propose_final"))

    def test_enforced_modes_constant_matches_doc(self):
        self.assertEqual(
            ENFORCED_MODES, frozenset({"default", "free", "propose_final"})
        )


class BuildRetryHintTests(unittest.TestCase):
    def test_empty_list_returns_empty_string(self):
        self.assertEqual(build_retry_hint([]), "")

    def test_hint_contains_matched_phrases(self):
        hint = build_retry_hint(["你可以思考一下", "建議你先確認"])
        self.assertIn("你可以思考一下", hint)
        self.assertIn("建議你先確認", hint)

    def test_hint_caps_at_three_phrases(self):
        """給 5 個命中句型，hint 只列前 3 個（避免給太多負面範例）。"""
        phrases = [
            "你可以思考一下",
            "建議你先確認",
            "請你再補",
            "你可以在搜尋引擎",
            "我建議你先從",
        ]
        hint = build_retry_hint(phrases)
        self.assertIn("你可以思考一下", hint)
        self.assertIn("建議你先確認", hint)
        self.assertIn("請你再補", hint)
        # 第 4、5 個不該在
        self.assertNotIn("你可以在搜尋引擎", hint)
        self.assertNotIn("我建議你先從", hint)

    def test_hint_mentions_help_me_do_directive(self):
        """Hint 應該重述「幫我做 vs 教我思考」的核心指示。"""
        hint = build_retry_hint(["你可以思考一下"])
        self.assertIn("幫我做", hint)

    def test_hint_allows_explicit_unknown_fallback(self):
        """Hint 要包含「確實不知道時直接說不確定」的合法出口，
        避免模型誤以為任何承認都違規。"""
        hint = build_retry_hint(["你可以思考一下"])
        self.assertIn("不確定", hint)


class MarkDegradedTests(unittest.TestCase):
    def test_prefix_prepended(self):
        result = mark_degraded("原始內容")
        self.assertTrue(result.startswith(DEGRADED_WARNING_PREFIX))
        self.assertIn("原始內容", result)

    def test_empty_text_still_gets_prefix(self):
        result = mark_degraded("")
        self.assertEqual(result, DEGRADED_WARNING_PREFIX)

    def test_warning_prefix_mentions_degradation(self):
        self.assertIn("警告", DEGRADED_WARNING_PREFIX)
        self.assertIn("未完整", DEGRADED_WARNING_PREFIX)


class EvasionPatternsSanityTests(unittest.TestCase):
    def test_no_empty_patterns(self):
        for pattern in EVASION_PATTERNS:
            self.assertTrue(pattern.strip(), f"空 pattern: {pattern!r}")

    def test_patterns_are_unique(self):
        self.assertEqual(
            len(EVASION_PATTERNS), len(set(EVASION_PATTERNS)),
            "EVASION_PATTERNS 有重複",
        )

    def test_patterns_do_not_match_clean_examples(self):
        """清潔輸出例子不該命中任何句型（避免誤判）。"""
        clean_examples = [
            "建議 4/20 前訂台北→仙台機票，預算約 NT$9 萬。",
            "推薦日本青森的十和田湖，有大山大海、少觀光客。",
            "Day 1-3 仙台；Day 4-6 青森；Day 7-10 首爾。",
            "假設你預算 NT$10 萬，建議選淡季。",
        ]
        for example in clean_examples:
            hits = detect_evasion_phrases(example)
            self.assertEqual(
                hits, [], f"誤判：{example!r} 命中 {hits}",
            )


if __name__ == "__main__":
    unittest.main()
