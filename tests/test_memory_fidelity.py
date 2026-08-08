# -*- coding: utf-8 -*-
"""v0.9 修復二驗收：記憶存原話，不存改寫版（施工單 2026-08-07）。

症狀：使用者說「我就直接取消，改天再說」，密碼表送給模型的卻是
改寫後的 goal「完成爬山活動」——未來每一輪撈到這條端點都讀到相反的事實。

修法：GOAL 欄位原話優先；逗福的詮釋保留在 DERIVED 欄位並標明來源，
CODEBOOK 說明兩者衝突時以 GOAL 為準。
"""
import unittest

from src.middleware.endpoint import (
    CODEBOOK,
    MEMORY_CODEBOOK,
    encode_retrieved_endpoints,
)


def _mk_row(user_input, goal, result="好的"):
    sd = {
        "user_input": user_input,
        "goal": goal,
        "topic": "t",
        "zone": "B",
    }
    if user_input is None:
        del sd["user_input"]
    return {
        "type": "start",
        "event_id": "e1",
        "start_data": sd,
        "end_data": {"result": result},
        "timestamp": "2026-08-07T00:00:00+00:00",
    }


class MemoryFidelityTests(unittest.TestCase):
    def test_codebook_goal_contains_original_words(self):
        """密碼表 GOAL 必須含原話（「取消」二字），不是改寫版。"""
        row = _mk_row(
            user_input="這週末原本要去爬山，結果預報說會下雨，我就直接取消，改天再說。",
            goal="完成爬山活動",
        )
        text = encode_retrieved_endpoints([row])
        goal_line = next(l for l in text.splitlines() if l.startswith("GOAL:"))
        self.assertIn("取消", goal_line)
        self.assertNotIn("完成爬山活動", goal_line)

    def test_derived_field_separated(self):
        """逗福的詮釋保留在 DERIVED，與原話分離。"""
        row = _mk_row(
            user_input="這週末原本要去爬山，我就直接取消，改天再說。",
            goal="完成爬山活動",
        )
        text = encode_retrieved_endpoints([row])
        self.assertIn("DERIVED: 完成爬山活動", text)

    def test_legacy_endpoint_without_user_input(self):
        """舊端點沒有 user_input 欄位 → 退回 goal，仍能正常編碼。"""
        row = _mk_row(user_input=None, goal="完成爬山活動")
        text = encode_retrieved_endpoints([row])
        self.assertIn("GOAL: 完成爬山活動", text)
        # goal 同時是原話 fallback 與 derived → 不得重複輸出
        self.assertNotIn("DERIVED:", text)

    def test_no_derived_when_identical(self):
        """原話與 goal 相同時不輸出 DERIVED（避免重複佔 token）。"""
        row = _mk_row(user_input="幫我訂餐廳", goal="幫我訂餐廳")
        text = encode_retrieved_endpoints([row])
        self.assertNotIn("DERIVED:", text)

    def test_long_input_negation_survives_truncation(self):
        """Copilot review #11：決定性轉折出現在 80 字之後也不得被切掉。

        GOAL 上限 80 → 160；此夾具為 90+ 字的單一長句，「取消」在句尾。"""
        long_input = (
            "這週末原本計畫跟公司同事還有兩位大學同學一起去陽明山走那條新開放的步道，"
            "登山裝備、交通共乘和午餐地點都已經提前安排好了，"
            "結果昨天晚上氣象預報說整個週末山區都會下大雨，"
            "跟大家討論考慮之後我就直接取消，改天再說。"
        )
        self.assertGreater(len(long_input), 80)
        row = _mk_row(user_input=long_input, goal="完成爬山活動")
        text = encode_retrieved_endpoints([row])
        goal_line = next(l for l in text.splitlines() if l.startswith("GOAL:"))
        self.assertIn("取消", goal_line)

    def test_codebook_legend_explains_priority(self):
        """CODEBOOK 說明區塊必須講清楚 DERIVED 的來源與優先序。"""
        for legend in (CODEBOOK, MEMORY_CODEBOOK):
            self.assertIn("DERIVED", legend)
            self.assertIn("NOT the user's words", legend)
            self.assertIn("wins", legend)


if __name__ == "__main__":
    unittest.main()
