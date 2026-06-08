"""v4.0 P0-1 整合測試：_execute_task_with_atl3_gate 的重試 / 降級行為。

使用 mock client 模擬多次 execute_task 呼叫的行為，驗證：
- 通過一次就不再重試
- 失敗→重試直到通過
- 3 次都失敗→degraded=True
- 重試時 extra_context 前綴帶 ATL3_RETRY_PROMPT
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.main import _execute_task_with_atl3_gate


_PASSING = (
    "先 48 小時內產出 1 份活動規劃書（含 3 個流程節點與時間表）；"
    "驗收標準：能依文件完成報到。"
)

_FAILING_NO_DELIVERABLE = (
    "建議你持續觀察場地市場 48 小時，再決定下一步。"
)


class _MockClient:
    """Mock client：照序列回傳預設 result，並記錄每次呼叫的 extra_context。"""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self._calls: list[dict] = []
        self.fallback_mode = False

    @property
    def calls(self) -> list[dict]:
        return self._calls

    def execute_task(self, **kwargs):
        self._calls.append(dict(kwargs))
        if self._responses:
            return self._responses.pop(0)
        return self._calls[-1].get("extra_context", "") or ""


class ExecuteTaskWithAtl3GateTests(unittest.TestCase):
    def _run(self, responses: list[str], mode: str = "default", max_retries: int = 2):
        client = _MockClient(responses)
        result, gate_result = _execute_task_with_atl3_gate(
            client=client,  # type: ignore[arg-type]
            goal="goal",
            motivation="",
            constraints=[],
            user_profile=None,
            strategy_brief=None,
            encoded_top_k=None,
            output_mode=mode,
            extra_context=None,
            max_retries=max_retries,
        )
        return client, result, gate_result

    def test_execute_task_passes_first_try(self):
        client, result, gate = self._run([_PASSING])
        self.assertEqual(result, _PASSING)
        self.assertTrue(gate["passed"])
        self.assertEqual(gate["retry_count"], 0)
        self.assertFalse(gate["degraded"])
        self.assertEqual(len(client.calls), 1)

    def test_execute_task_retry_once_then_pass(self):
        client, result, gate = self._run([_FAILING_NO_DELIVERABLE, _PASSING])
        self.assertEqual(result, _PASSING)
        self.assertTrue(gate["passed"])
        self.assertEqual(gate["retry_count"], 1)
        self.assertFalse(gate["degraded"])
        # 第二次呼叫的 extra_context 應該帶 ATL3 重試 prompt
        second_extra = client.calls[1].get("extra_context") or ""
        self.assertIn("沒有通過具體性檢查", second_extra)

    def test_execute_task_retry_twice_then_pass(self):
        client, result, gate = self._run(
            [_FAILING_NO_DELIVERABLE, _FAILING_NO_DELIVERABLE, _PASSING]
        )
        self.assertTrue(gate["passed"])
        self.assertEqual(gate["retry_count"], 2)
        self.assertFalse(gate["degraded"])

    def test_execute_task_retry_exhausted_degrades(self):
        responses = [_FAILING_NO_DELIVERABLE] * 4
        client, result, gate = self._run(responses, max_retries=2)
        self.assertFalse(gate["passed"])
        self.assertTrue(gate["degraded"])
        self.assertEqual(gate["retry_count"], 3)  # 初始 + 2 次重試 = 3 次呼叫
        self.assertEqual(len(client.calls), 3)
        # failure_history 應該累積 3 筆
        self.assertEqual(len(gate["failure_history"]), 3)

    def test_execute_task_retry_prompt_contains_reasons(self):
        client, _, _ = self._run([_FAILING_NO_DELIVERABLE, _PASSING])
        retry_extra = client.calls[1]["extra_context"]
        # 失敗原因（例如「缺產出物」）會列在 retry hint 裡
        self.assertIn("缺產出物", retry_extra)


if __name__ == "__main__":
    unittest.main()
