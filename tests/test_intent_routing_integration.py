"""意圖分流端到端行為驗收（規格 06209671 §3.2）。

用記錄式 mock client 串 run_one_interaction，驗證：
1. 規劃句式（陳述/疑問/「想/規劃」命令）→ execute_task 收到 output_mode="consult"。
2. 明確執行命令（幫我訂…）→ execute_task 仍收到 output_mode="default"（工單路徑）。
3. consult 模式下，端點 start_data 帶 intent_mode="plan" / output_mode="consult"。
4. consult 模式不因「缺時間窗/產出物/驗收條件」而被當不合格（不打回工單）。

注意：本檔驗證的是「分流是否把對的句式導到對的 prompt 路徑」這個程式邏輯，
不驗證 LLM 文字品質（那需獨立腳本/多 AI 交叉驗證，見規格 §5）。
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.main import run_one_interaction  # noqa: E402
from src.middleware.endpoint import EndpointStore  # noqa: E402


class _RecordingClient:
    """記錄 execute_task 收到的 output_mode 的 mock client。"""

    fallback_mode = False

    def __init__(self) -> None:
        self.execute_calls: list[dict] = []

    def generate_restate(self, user_input, baseline_summary, allowed_categories, **kwargs):
        return {
            "restate_text": f"你想「{user_input}」。",
            "gap_questions": [],  # 空 → info_sufficient，單輪收斂
            "gap_categories": [],
            "inferred_goal": user_input,
            "inferred_motivation": "",
            "inferred_constraints": [],
        }

    def merge_confirmation(self, original_input, restate_text, user_confirmation_text):
        return {
            "goal": original_input,
            "motivation": "",
            "constraints": [],
            "answered_categories": [],
        }

    def execute_task(
        self,
        goal,
        motivation,
        constraints,
        user_profile=None,
        *,
        strategy_brief=None,
        encoded_top_k=None,
        output_mode="default",
        extra_context=None,
    ):
        self.execute_calls.append({"goal": goal, "output_mode": output_mode})
        if output_mode == "consult":
            # 規劃輸出：不含 deadline / 驗收條件，先列關注重點再給選項。
            return (
                "我抓到的關注重點（你可以修正）：你想輕鬆度假、不趕車。\n"
                "我把沿線可以走的方向擺給你看，你看順不順。"
            )
        # 執行/工單輸出：含時間窗與產出物（模擬現有工單格式）。
        return (
            "建議產出 1 份行程表，48 小時內完成訂位，至少包含 3 個確認點。"
        )


class IntentRoutingIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "endpoints.json")
        self.store = EndpointStore(self.path)
        self.client = _RecordingClient()

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, text):
        self.client.execute_calls.clear()
        ok = run_one_interaction(
            text, self.store, self.client,
            ask_satisfaction=False, auto_confirm=True,
        )
        self.assertTrue(ok)
        return self.client.execute_calls[-1]["output_mode"]

    def test_statement_routes_to_consult(self):
        # 陳述句（婚禮原案）→ consult
        self.assertEqual(
            self._run("輕鬆度假，坐長榮，米蘭進羅馬出"), "consult"
        )

    def test_question_routes_to_consult(self):
        self.assertEqual(
            self._run("義大利兩週怎麼安排比較好？"), "consult"
        )

    def test_plan_command_routes_to_consult(self):
        self.assertEqual(self._run("幫我規劃義大利行程"), "consult")

    def test_execute_command_routes_to_default_workorder(self):
        # §3.2 第 5 點：明確執行命令仍走工單路徑（output_mode="default"）
        self.assertEqual(
            self._run("幫我訂長榮米蘭進羅馬出的機票"), "default"
        )

    def test_consult_endpoint_records_intent_mode(self):
        self._run("我想去義大利玩，輕鬆度假")
        rows = self.store.all()
        starts = [r for r in rows if r.get("type") == "start"]
        self.assertTrue(starts)
        sd = starts[-1]["start_data"]
        self.assertEqual(sd.get("intent_mode"), "plan")
        self.assertEqual(sd.get("output_mode"), "consult")

    def test_execute_endpoint_records_intent_mode(self):
        self._run("幫我把這三晚住宿訂下來")
        rows = self.store.all()
        starts = [r for r in rows if r.get("type") == "start"]
        sd = starts[-1]["start_data"]
        self.assertEqual(sd.get("intent_mode"), "execute")
        self.assertEqual(sd.get("output_mode"), "default")

    def test_consult_output_not_bounced_for_missing_specificity(self):
        # §3.2：規劃輸出不含 deadline/驗收條件，但不應被當不合格而降級。
        # 驗證 end_data 的 atl3_gate_result 沒有 degraded（default 路徑本就 skip
        # gate），且回應原樣保留（沒有被降級前綴污染）。
        self._run("幫我列出可以考慮的城市")
        rows = self.store.all()
        ends = [r for r in rows if r.get("type") == "end"]
        self.assertTrue(ends)
        ed = ends[-1]["end_data"]
        self.assertFalse(ed.get("atl3_gate_result", {}).get("degraded", False))
        self.assertNotIn("警告", ed.get("result", ""))


if __name__ == "__main__":
    unittest.main()
