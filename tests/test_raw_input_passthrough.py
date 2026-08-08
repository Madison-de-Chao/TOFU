# -*- coding: utf-8 -*-
"""v0.9 修復三驗收：執行階段看得到原話（施工單 2026-08-07）。

症狀：一輪互動的三次 API 呼叫各自獨立，execute_task 只收到改寫後的
goal/motivation/constraints，原話裡的決定性資訊（「我就直接取消」）
丟失，執行段與復述段自相矛盾。

驗收（程式碼層）：
1. execute_task 的 prompt 含原話，且優先序明寫
2. 未傳新參數時 prompt 與改動前完全一致（向後相容）
3. 舊簽名 mock client 不會因新參數 TypeError
（真實 API 的行為驗收——第 5/11 輪重跑——需在本機以 Haiku 執行）
"""
import unittest

from src.api.claude_client import LLMClient
from src.main import _call_execute_task


class _PromptCapturingClient(LLMClient):
    """攔截 _call，記錄 execute_task 實際組出的 prompt。"""

    def __init__(self):
        super().__init__(api_key="__TEST__")
        self.fallback_mode = False
        self._client = object()  # 不會被用到
        self.captured = {}

    def _call(self, system, user, max_tokens=1024):
        self.captured = {"system": system, "user": user}
        return "OK"


class _LegacyMockClient:
    """舊簽名 mock：execute_task 不認得 v0.9 的新參數。"""

    def execute_task(self, goal, motivation, constraints, user_profile=None, *,
                     strategy_brief=None, encoded_top_k=None,
                     output_mode="default", extra_context=None):
        return f"legacy:{goal}"


class RawInputPassthroughTests(unittest.TestCase):
    RAW = "這週末原本要去爬山，結果預報說會下雨，我就直接取消，改天再說。"

    def _run(self, **kw):
        c = _PromptCapturingClient()
        c.execute_task(
            goal="完成爬山活動", motivation="", constraints=[], **kw,
        )
        return c.captured["user"]

    def test_prompt_contains_raw_input(self):
        """原話有傳入 → prompt 含原文與優先序聲明。"""
        prompt = self._run(raw_user_input=self.RAW,
                           confirmed_understanding="使用者已取消爬山")
        self.assertIn("我就直接取消", prompt)
        self.assertIn("最高優先", prompt)
        self.assertIn("以此為準", prompt)
        self.assertIn("使用者已取消爬山", prompt)
        self.assertIn("可能有遺漏或改寫", prompt)
        # 原話必須排在結構化欄位之前
        self.assertLess(prompt.index("我就直接取消"), prompt.index("目標："))

    def test_backward_compatible_prompt_unchanged(self):
        """未傳新參數 → prompt 與改動前一致（無任何新增區塊）。"""
        prompt = self._run()
        self.assertNotIn("使用者原話", prompt)
        self.assertNotIn("最高優先", prompt)
        self.assertNotIn("可能有遺漏或改寫", prompt)
        self.assertTrue(prompt.startswith("目標："))

    def test_no_duplicate_understanding_when_identical(self):
        """理解與原話相同時不重複輸出。"""
        prompt = self._run(raw_user_input=self.RAW,
                           confirmed_understanding=self.RAW)
        self.assertEqual(prompt.count(self.RAW), 1)

    def test_legacy_mock_degrades_without_typeerror(self):
        """舊簽名 client 經 _call_execute_task 傳新參數 → TypeError 退化。"""
        out = _call_execute_task(
            _LegacyMockClient(),
            goal="G", motivation="", constraints=[], user_profile=None,
            strategy_brief=None, encoded_top_k=None,
            output_mode="default", extra_context=None,
            raw_user_input=self.RAW,
            confirmed_understanding="理解",
        )
        self.assertEqual(out, "legacy:G")

    def test_analyze_deviation_accepts_raw_input(self):
        """出口檢查收原話，prompt 含格式限制豁免指引。"""
        c = _PromptCapturingClient()
        c.analyze_deviation(
            goal="回答測驗", result="C", deviation="結果過短",
            raw_user_input="只回覆一個大寫字母，不要解釋。",
        )
        self.assertIn("只回覆一個大寫字母", c.captured["user"])
        self.assertIn("不算偏差", c.captured["system"])


if __name__ == "__main__":
    unittest.main()
