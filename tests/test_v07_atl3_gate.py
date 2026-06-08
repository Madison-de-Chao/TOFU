"""v4.0 P0-1：ATL-3 前驗證閘門的單元測試。

對應規格 20260418_逗福Tofu_三機制落地_開發規格_v4_0.md §P0-1 的 15 項
驗收測試清單。本檔實作純程式碼層能涵蓋的部分（閘門判定、重試提示
建構）；execute_task 的 I/O 重試邏輯走 main._execute_task_with_atl3_gate，
以 mock client 另行測試（見 test_v07_atl3_gate_integration.py）。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.api.claude_client import ATL3_RETRY_PROMPT, build_atl3_retry_hint
from src.middleware.confirmation import atl3_gate


# 一段完整具備產出物 + 時間窗 + 驗收條件的 result，用於正向測試
_GOOD_RESULT = (
    "先產出 1 份活動流程規劃書（包含至少 3 個流程節點與時間表），"
    "48 小時內交付；驗收標準：參與者能依文件在不額外提問下完成報到。"
)

# 缺產出物（沒有「份/篇/個/頁」等量詞）
_NO_DELIVERABLE = (
    "建議你 48 小時內跟場地方聯繫確認細節，且要求對方回覆時"
    "包含具體數據說明。"
)

# 缺時間窗
_NO_TIME_WINDOW = (
    "產出 1 份活動流程規劃書，包含至少 3 個節點；"
    "驗收標準：流程可執行。"
)

# propose 情境下缺驗收條件
_NO_VERIFICATION = (
    "48 小時內產出 1 份活動流程規劃書，列出 3 個流程節點與時間表。"
)

# 含迴避句型
_VAGUE_PHRASES = (
    "48 小時內產出 1 份包含來源的規劃書；建議你持續觀察市場變化並適時調整。"
)


class Atl3GatePassTests(unittest.TestCase):
    def test_atl3_gate_pass_with_full_specificity(self):
        gate = atl3_gate(_GOOD_RESULT, mode="propose")
        self.assertTrue(gate["passed"], f"should pass, got: {gate}")
        self.assertEqual(gate["failure_reasons"], [])

    def test_atl3_gate_pass_for_default_mode_without_verification(self):
        # default 模式不強制要求驗收條件
        text = "48 小時內產出 1 份活動流程規劃書，列出 3 個節點。"
        gate = atl3_gate(text, mode="default")
        self.assertTrue(gate["passed"], gate)


class Atl3GateFailTests(unittest.TestCase):
    def test_atl3_gate_fail_missing_deliverable(self):
        gate = atl3_gate(_NO_DELIVERABLE, mode="default")
        self.assertFalse(gate["passed"])
        self.assertTrue(
            any("產出物" in r for r in gate["failure_reasons"]),
            gate["failure_reasons"],
        )

    def test_atl3_gate_fail_missing_time_window(self):
        gate = atl3_gate(_NO_TIME_WINDOW, mode="default")
        self.assertFalse(gate["passed"])
        self.assertTrue(
            any("時間窗" in r for r in gate["failure_reasons"]),
            gate["failure_reasons"],
        )

    def test_atl3_gate_fail_missing_verification_in_propose(self):
        gate = atl3_gate(_NO_VERIFICATION, mode="propose")
        self.assertFalse(gate["passed"])
        self.assertTrue(
            any("驗收條件" in r for r in gate["failure_reasons"]),
            gate["failure_reasons"],
        )

    def test_atl3_gate_fail_vague_phrases(self):
        gate = atl3_gate(_VAGUE_PHRASES, mode="default")
        self.assertFalse(gate["passed"])
        self.assertTrue(
            any("迴避句型" in r for r in gate["failure_reasons"]),
            gate["failure_reasons"],
        )

    def test_atl3_gate_fail_on_empty_result(self):
        gate = atl3_gate("", mode="default")
        self.assertFalse(gate["passed"])

    def test_atl3_gate_specificity_score_below_2(self):
        # 只有時間窗，產出物與驗收條件皆缺 → score = 1 < 2
        text = "48 小時內再回頭看看這件事。"
        gate = atl3_gate(text, mode="default")
        self.assertFalse(gate["passed"])
        # 同時應該附帶缺產出物、具體性不足兩個失敗原因
        joined = " ".join(gate["failure_reasons"])
        self.assertIn("產出物", joined)


class Atl3RetryHintTests(unittest.TestCase):
    def test_retry_hint_contains_failure_reasons(self):
        reasons = ["缺產出物", "缺時間窗"]
        hint = build_atl3_retry_hint(reasons)
        for r in reasons:
            self.assertIn(r, hint)

    def test_retry_hint_contains_template_markers(self):
        hint = build_atl3_retry_hint(["缺產出物"])
        self.assertIn("產出物", hint)
        self.assertIn("時間窗", hint)
        self.assertIn("禁止", hint)

    def test_retry_hint_handles_empty_reasons(self):
        hint = build_atl3_retry_hint([])
        # 不應炸，且 template 文字仍在
        self.assertIn("產出物", hint)


class Atl3GateIntegrationWithExistingCheckTests(unittest.TestCase):
    def test_atl3_gate_returns_action_check_payload(self):
        gate = atl3_gate(_GOOD_RESULT, mode="default")
        self.assertIn("action_check", gate)
        self.assertIn("specificity_score", gate["action_check"])

    def test_atl3_retry_prompt_is_non_empty_constant(self):
        # 確保常數沒有被不小心清空
        self.assertIsInstance(ATL3_RETRY_PROMPT, str)
        self.assertGreater(len(ATL3_RETRY_PROMPT), 50)


if __name__ == "__main__":
    unittest.main()
