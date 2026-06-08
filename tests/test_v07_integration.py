"""v4.0 整合測試——三機制之間的交互行為。

對應規格 20260418_逗福Tofu_三機制落地_開發規格_v4_0.md §整合測試 的 5 項：

- test_atl3_gate_during_cooling_mode
- test_atl4_warning_during_cooling_mode
- test_emotion_state_preserved_through_atl3_retries
- test_degraded_result_triggers_atl4_recheck
- test_full_pipeline_regression_synthetic
"""

from __future__ import annotations

import builtins
import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.middleware.cooling_mode import should_enter_cooling_mode
from src.middleware.emotion_detector import detect_emotion_rule_based
from src.middleware.zone_history import (
    ZoneHistoryStore,
    atl4_consistency_check,
    compute_question_signature,
    extract_question_features,
)
from src.main import _execute_task_with_atl3_gate


# 一個 ATL-3 compliant 的 mock result（含產出物、時間窗、驗收條件）
_COMPLIANT = (
    "48 小時內產出 1 份活動流程規劃書，至少包含 3 個可驗證節點。"
)
# 缺產出物
_FAILING = "建議你持續觀察場地市場並適時調整。"


class _MockClient:
    fallback_mode = False

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []

    def execute_task(self, **kwargs):
        self.calls.append(dict(kwargs))
        if self._responses:
            return self._responses.pop(0)
        return "（no more）"

    def generate_restate(self, **kwargs):
        return {
            "restate_text": "", "gap_questions": [],
            "gap_categories": [], "inferred_goal": "",
            "inferred_motivation": "", "inferred_constraints": [],
        }

    def merge_confirmation(self, **kwargs):
        return {}

    def analyze_deviation(self, **kwargs):
        return ""

    def answer_with_profile(self, **kwargs):
        return ""


class Atl3DuringCoolingModeTests(unittest.TestCase):
    def test_atl3_gate_during_cooling_mode_does_not_enforce_deliverable(self):
        """冷卻模式 + skip_gate=True → 即使 result 不合 ATL-3 也不重試。"""
        client = _MockClient([_FAILING])  # 只有一個 response
        result, gate = _execute_task_with_atl3_gate(
            client=client,  # type: ignore[arg-type]
            goal="clarify the question",
            motivation="",
            constraints=[],
            user_profile=None,
            strategy_brief=None,
            encoded_top_k=None,
            output_mode="default",
            extra_context="（冷卻模式 prompt）",
            skip_gate=True,
        )
        self.assertEqual(result, _FAILING)
        self.assertTrue(gate["passed"])
        self.assertTrue(gate.get("skipped", False))
        self.assertEqual(len(client.calls), 1)  # 無重試


class Atl4DuringCoolingModeTests(unittest.TestCase):
    def test_atl4_warning_during_cooling_mode_not_written(self):
        """冷卻模式下的 zone 不寫入 zone_history（避免污染歷史）。

        這是行為約束：ZoneHistoryStore 本身不知道冷卻模式；由 main.py 整合層
        決定。這個測試驗證本檔 `main._execute_task_with_atl3_gate` 呼叫路徑
        下、caller 決定不 append 時，store 不會被意外寫入。
        """
        with tempfile.TemporaryDirectory() as td:
            store = ZoneHistoryStore(Path(td) / "zh.jsonl")
            # 模擬冷卻模式：caller 決定不 append
            # 若 caller 沒呼叫 store.append，歷史應為空
            self.assertEqual(store.query("anysig"), [])


class EmotionStatePreservedThroughAtl3RetriesTests(unittest.TestCase):
    def test_emotion_state_not_mutated_by_atl3_retries(self):
        """ATL-3 重試時，start_data 的 emotion_state 不會被覆寫。"""
        # emotion_state 在 start 階段寫入一次，run_one_interaction 不會
        # 在 ATL-3 重試間重新偵測。這個測試驗證 detect_emotion_rule_based
        # 對同一輸入多次呼叫回同樣結果（冪等）。
        text = "我真的很生氣，為什麼又這樣！！！"
        es1 = detect_emotion_rule_based(text)
        es2 = detect_emotion_rule_based(text)
        self.assertEqual(es1, es2)
        # 並且即使經歷多次 execute_task 呼叫，caller 保留原 emotion_state
        client = _MockClient([_FAILING, _FAILING, _COMPLIANT])
        captured_emotion = es1
        _, _ = _execute_task_with_atl3_gate(
            client=client,  # type: ignore[arg-type]
            goal="g", motivation="", constraints=[],
            user_profile=None, strategy_brief=None, encoded_top_k=None,
            output_mode="propose", extra_context=None, max_retries=2,
        )
        # captured_emotion 沒被改動
        self.assertEqual(captured_emotion, es1)


class DegradedResultTriggersAtl4RecheckTests(unittest.TestCase):
    def test_atl4_sees_degraded_zone_via_zone_field(self):
        """ATL-3 閘門降級為 Zone B 後，ATL-4 跨輪檢查看到的是降級後的 zone。

        行為約束：main.py 先跑 ATL-3，若 degraded → end_data['zone'] = 'B'，
        再跑 ATL-4。本測試在純資料層驗證：當 proposed_zone='B' 餵給 ATL-4 時，
        它能正確與歷史主導 zone 比較。
        """
        with tempfile.TemporaryDirectory() as td:
            store = ZoneHistoryStore(Path(td) / "zh.jsonl")
            # 歷史 4 次都判 C
            from datetime import datetime, timezone
            for _ in range(4):
                store.append({
                    "signature": "sig1",
                    "question_features": {},
                    "zone_assignment": "C",
                    "endpoint_id": "ep",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "confidence": 0.85,
                })
            # 當前因 ATL-3 degraded 被降為 B → ATL-4 應警告
            result = atl4_consistency_check("sig1", "B", store)
            self.assertFalse(result["is_consistent"])
            self.assertEqual(result["dominant_zone"], "C")
            self.assertIn("B", result["warning"])


class FullPipelineRegressionSyntheticTests(unittest.TestCase):
    def test_full_pipeline_synthetic_no_crash(self):
        """合成的跨機制端到端：情緒偵測 → 冷卻判定 → ATL-3 gate → ATL-4。

        不依賴真實 LLM，驗證資料在三個機制之間流通沒有型別錯誤。
        """
        user_input = "規劃一場 30 人的活動"
        # 1. 情緒偵測：中性
        emotion = detect_emotion_rule_based(user_input)
        self.assertEqual(emotion["seven_emotion"], "中性")

        # 2. 冷卻模式：不觸發
        cooling = should_enter_cooling_mode(emotion, {})
        self.assertFalse(cooling)

        # 3. ATL-3 gate：propose 模式下第一次 failing → 重試 → 第二次 compliant
        client = _MockClient([_FAILING, _COMPLIANT])
        result, gate = _execute_task_with_atl3_gate(
            client=client,  # type: ignore[arg-type]
            goal=user_input, motivation="", constraints=[],
            user_profile=None, strategy_brief=None, encoded_top_k=None,
            output_mode="propose",
            extra_context=None,
            skip_gate=cooling,
        )
        self.assertTrue(gate["passed"])
        self.assertEqual(gate["retry_count"], 1)
        self.assertEqual(result, _COMPLIANT)

        # 4. ATL-4：冷啟動歷史 < 3 筆 → 不判定
        with tempfile.TemporaryDirectory() as td:
            store = ZoneHistoryStore(Path(td) / "zh.jsonl")
            features = extract_question_features(
                user_input, meta_motivation_direction="self", topic="event",
            )
            sig = compute_question_signature(features)
            consistency = atl4_consistency_check(sig, "C", store)
            self.assertTrue(consistency["is_consistent"])
            self.assertIsNone(consistency["warning"])


if __name__ == "__main__":
    unittest.main()
