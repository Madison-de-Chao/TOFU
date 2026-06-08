"""TOFU_功能實作開發規格 v1.0 的功能測試。

涵蓋：
- P0-1：Zone A/B/C 分類
- P0-2：黑盒子回溯 + detect_deviation 的長度與同義詞分支
- P0-3：出口檢查（analyze_deviation）
- P0-4：CBP topic 分類 + baseline topic_filter
- P1-1：CIP-X 軌跡收斂
- P1-2：BaseLLMClient 抽象介面 + OpenAIClient 骨架
- P1-3：reasoning_steps 欄位（v1 設計：七個檢查模式）
"""

from __future__ import annotations

import builtins
import io
import contextlib
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.api.base_client import BaseLLMClient
from src.api.claude_client import (
    LLMClient,
    _fallback_analyze_deviation,
    _fallback_execute_task,
)
from src.middleware import baseline as baseline_mod
from src.middleware import confirmation as confirm_mod
from src.middleware.endpoint import EndpointStore


# ----------------------------------------------------------------------
# P0-1：Zone A/B/C 分類
# ----------------------------------------------------------------------
class ZoneClassificationTests(unittest.TestCase):
    def test_zone_a_fact_with_number(self):
        # 含數字 → Zone A
        self.assertEqual(
            confirm_mod.classify_zone("台北市的場地租金大約 2000 元"),
            "A",
        )
        self.assertEqual(
            confirm_mod.classify_zone("2025年12月31日的活動"),
            "A",
        )

    def test_zone_b_speculation(self):
        self.assertEqual(
            confirm_mod.classify_zone("根據經驗，通常需要一週"),
            "B",
        )
        self.assertEqual(
            confirm_mod.classify_zone("你可能會比較喜歡戶外"),
            "B",
        )

    def test_zone_c_stance(self):
        self.assertEqual(
            confirm_mod.classify_zone("我建議先確定人數再找場地"),
            "C",
        )
        self.assertEqual(
            confirm_mod.classify_zone("you can consider a smaller venue"),
            "C",
        )

    def test_zone_unknown_on_empty(self):
        self.assertEqual(confirm_mod.classify_zone(""), "unknown")

    def test_zone_c_overrides_b(self):
        """立場句常同時含推測語氣，立場優先。"""
        text = "我建議可能選戶外場地"
        self.assertEqual(confirm_mod.classify_zone(text), "C")

    def test_execute_task_fallback_contains_zone_markers(self):
        """P0-1：fallback 的 execute_task 應含 Zone 標註語。"""
        out = _fallback_execute_task(
            goal="辦派對",
            motivation="",
            constraints=["預算三萬"],
        )
        self.assertIn("通用建議", out)
        self.assertIn("硬限制", out)


# ----------------------------------------------------------------------
# P0-2：黑盒子回溯 + detect_deviation 增強
# ----------------------------------------------------------------------
class EnhancedDeviationDetectionTests(unittest.TestCase):
    def test_short_result_with_no_overlap_triggers_deviation(self):
        """P0-2：goal='辦生日派對' result='已完成' 應觸發偏差。"""
        result = confirm_mod.detect_deviation("辦生日派對", "已完成")
        self.assertTrue(result)
        self.assertIn("偏離", result)

    def test_synonym_match_no_deviation(self):
        """P0-2：同義詞映射——生日 ↔ 慶生。"""
        result = confirm_mod.detect_deviation(
            "辦生日派對",
            "建議先找好適合慶生的場地和時間規劃",
        )
        self.assertEqual(result, "")

    def test_synonym_budget_match_no_deviation(self):
        """預算 ↔ 經費。"""
        result = confirm_mod.detect_deviation(
            "規劃預算",
            "先抓出主要經費項目，避免後期超支與計畫調整",
        )
        self.assertEqual(result, "")

    def test_long_unrelated_result_triggers_deviation(self):
        result = confirm_mod.detect_deviation(
            "辦一場生日派對",
            "今天天氣真好，我去散步了一下，還買了一杯咖啡。",
        )
        self.assertTrue(result)
        self.assertIn("偏離", result)


class BlackBoxRecordTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "endpoints.json")
        self.store = EndpointStore(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_append_and_read_black_box(self):
        self.store.append_black_box(
            event_id="evt-1",
            black_box_data={
                "original_input": "辦派對",
                "result": "已完成",
                "deviation": "結果異常簡短",
                "analysis": "Zone B 推測",
                "timestamp": "2026-04-11T00:00:00+00:00",
            },
        )
        boxes = self.store.black_boxes()
        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes[0]["type"], "black_box")
        self.assertEqual(boxes[0]["event_id"], "evt-1")
        self.assertEqual(boxes[0]["black_box"]["deviation"], "結果異常簡短")

    def test_black_box_not_double_counted_as_completed_event(self):
        """P0-2：黑盒子記錄不應被算進 completed_count。"""
        self.store.append_start("e1", {"goal": "a"})
        self.store.append_end("e1", {"result": "r"})
        self.store.append_black_box("e1", {"deviation": "x"})
        self.assertEqual(self.store.completed_count(), 1)


class BlackBoxIntegrationTests(unittest.TestCase):
    """整合測試：偏差觸發時自動記錄黑盒子 + 印出出口檢查。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "endpoints.json")
        self.store = EndpointStore(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def _make_client(self, execute_result: str):
        class _C:
            fallback_mode = False

            def __init__(self):
                self.analyze_calls = []

            def generate_restate(self, user_input, baseline_summary, allowed_categories):
                return {
                    "restate_text": f"你想「{user_input}」",
                    "gap_questions": [],
                    "gap_categories": ["budget"],
                    "inferred_goal": user_input,
                    "inferred_motivation": "",
                    "inferred_constraints": [],
                }

            def merge_confirmation(self, original_input, restate_text, user_confirmation_text):
                return {
                    "goal": original_input,
                    "motivation": "",
                    "constraints": [user_confirmation_text] if user_confirmation_text else [],
                    "answered_categories": [],
                }

            def execute_task(self, goal, motivation, constraints, user_profile=None):
                return execute_result

            def analyze_deviation(self, goal, result, deviation):
                self.analyze_calls.append({"goal": goal, "result": result})
                return "測試出口檢查回應（Zone B 推測）"

        return _C()

    def test_deviation_triggers_black_box_and_analysis(self):
        from src.main import run_one_interaction

        client = self._make_client("已完成")  # 明顯偏離的短回應

        captured = io.StringIO()
        with patch.object(builtins, "input", lambda prompt="": "預算三萬"), \
             contextlib.redirect_stdout(captured):
            ok = run_one_interaction("辦生日派對", self.store, client)

        self.assertTrue(ok)
        self.assertEqual(len(client.analyze_calls), 1)
        boxes = self.store.black_boxes()
        self.assertEqual(len(boxes), 1)
        self.assertIn("Zone B", boxes[0]["black_box"]["analysis"])
        out = captured.getvalue()
        self.assertIn("出口檢查", out)

    def test_no_deviation_no_black_box(self):
        from src.main import run_one_interaction

        client = self._make_client(
            "針對生日派對，建議先確認人數與預算，接著找合適的場地。"
        )

        with patch.object(builtins, "input", lambda prompt="": "預算三萬"):
            ok = run_one_interaction("辦生日派對", self.store, client)

        self.assertTrue(ok)
        self.assertEqual(len(client.analyze_calls), 0)
        self.assertEqual(self.store.black_boxes(), [])

    def test_blackbox_command_shows_recent(self):
        from src.main import run_blackbox_command, run_one_interaction

        client = self._make_client("done")
        with patch.object(builtins, "input", lambda prompt="": "預算三萬"):
            run_one_interaction("辦生日派對", self.store, client)

        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            run_blackbox_command(self.store)
        out = captured.getvalue()
        self.assertIn("黑盒子", out)
        self.assertIn("辦生日派對", out)

    def test_blackbox_command_empty(self):
        from src.main import run_blackbox_command

        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            run_blackbox_command(self.store)
        self.assertIn("尚無黑盒子回溯", captured.getvalue())


# ----------------------------------------------------------------------
# P0-3：出口檢查（analyze_deviation）
# ----------------------------------------------------------------------
class AnalyzeDeviationTests(unittest.TestCase):
    def test_fallback_analyze_contains_zone_b_marker(self):
        out = _fallback_analyze_deviation(
            goal="辦生日派對",
            result="已完成",
            deviation="結果異常簡短",
        )
        self.assertIn("Zone B", out)
        self.assertIn("推測", out)
        self.assertIn("已完成", out)

    def test_llm_client_analyze_deviation_uses_fallback_without_key(self):
        client = LLMClient(api_key=None)
        self.assertTrue(client.fallback_mode)
        out = client.analyze_deviation(
            goal="辦生日派對",
            result="已完成",
            deviation="簡短",
        )
        self.assertIn("Zone B", out)


# ----------------------------------------------------------------------
# P0-4：CBP topic 分類 + baseline topic_filter
# ----------------------------------------------------------------------
class TopicInferenceTests(unittest.TestCase):
    def test_event_topic(self):
        self.assertEqual(confirm_mod.infer_topic("辦一場生日派對"), "event")
        self.assertEqual(confirm_mod.infer_topic("辦公司尾牙"), "event")

    def test_work_topic(self):
        self.assertEqual(confirm_mod.infer_topic("寫季度報告"), "work")
        self.assertEqual(confirm_mod.infer_topic("交付專案"), "work")

    def test_creative_topic(self):
        self.assertEqual(confirm_mod.infer_topic("寫一篇小說"), "creative")

    def test_learning_topic(self):
        self.assertEqual(confirm_mod.infer_topic("學習日文"), "learning")

    def test_other_topic_fallback(self):
        self.assertEqual(confirm_mod.infer_topic(""), "other")
        self.assertEqual(confirm_mod.infer_topic("xyz"), "other")

    def test_gap_categories_boost_event(self):
        """venue / headcount 的 gap_categories 把 topic 往 event 偏。"""
        t = confirm_mod.infer_topic("安排一下那個東西", ["venue", "headcount"])
        self.assertEqual(t, "event")


class BaselineTopicFilterTests(unittest.TestCase):
    def _mk_endpoints(self):
        return [
            {
                "type": "start",
                "start_data": {
                    "goal": "辦派對",
                    "topic": "event",
                    "gap_categories": ["venue", "headcount"],
                    "user_confirmation": "confirmed",
                    "user_modifications": "",
                },
            },
            {
                "type": "start",
                "start_data": {
                    "goal": "辦活動",
                    "topic": "event",
                    "gap_categories": ["budget", "venue"],
                    "user_confirmation": "confirmed",
                    "user_modifications": "",
                },
            },
            {
                "type": "start",
                "start_data": {
                    "goal": "寫報告",
                    "topic": "work",
                    "gap_categories": ["deliverable"],
                    "user_confirmation": "confirmed",
                    "user_modifications": "",
                },
            },
        ]

    def test_topic_filter_isolates_event(self):
        eps = self._mk_endpoints()
        b = baseline_mod.compute_baseline(eps, topic_filter="event")
        self.assertEqual(b["total_events"], 2)
        self.assertIn("venue", b["top_gap_categories"])
        self.assertNotIn("deliverable", b["top_gap_categories"])

    def test_topic_filter_isolates_work(self):
        eps = self._mk_endpoints()
        b = baseline_mod.compute_baseline(eps, topic_filter="work")
        self.assertEqual(b["total_events"], 1)
        self.assertIn("deliverable", b["top_gap_categories"])
        self.assertNotIn("venue", b["top_gap_categories"])

    def test_unfiltered_baseline_counts_all(self):
        eps = self._mk_endpoints()
        b = baseline_mod.compute_baseline(eps)
        self.assertEqual(b["total_events"], 3)


# ----------------------------------------------------------------------
# P1-1：CIP-X 軌跡收斂
# ----------------------------------------------------------------------
class TrajectoryConvergenceTests(unittest.TestCase):
    def test_three_danger_goals_trigger(self):
        goals = ["想自殺", "想自我了結", "我要自殺"]
        self.assertTrue(confirm_mod.check_trajectory_convergence(goals, 3))

    def test_single_danger_does_not_trigger(self):
        """單筆危險詞可能只是在討論新聞，不觸發。"""
        goals = ["想辦派對", "自殺新聞怎麼看", "學日文"]
        self.assertFalse(confirm_mod.check_trajectory_convergence(goals, 3))

    def test_under_threshold_does_not_trigger(self):
        self.assertFalse(confirm_mod.check_trajectory_convergence(["自殺"], 3))
        self.assertFalse(
            confirm_mod.check_trajectory_convergence(["自殺", "自殺"], 3)
        )

    def test_trajectory_blocks_in_run_one_interaction(self):
        """連續危險趨勢→run_one_interaction 阻斷，不寫端點。"""
        from src.main import run_one_interaction

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        store = EndpointStore(os.path.join(tmp.name, "endpoints.json"))
        # 先播入兩筆危險 goal（直接寫 store，模擬前兩輪已經過去）
        store.append_start("e1", {"goal": "自殺念頭", "user_confirmation": "confirmed"})
        store.append_end("e1", {"result": "r"})
        store.append_start("e2", {"goal": "我要自我了結", "user_confirmation": "confirmed"})
        store.append_end("e2", {"result": "r"})

        class _C:
            fallback_mode = False
            generate_restate_calls: list = []

            def generate_restate(self, *a, **kw):
                self.generate_restate_calls.append(1)
                return {}

            def merge_confirmation(self, *a, **kw):
                return {}

            def execute_task(self, *a, **kw):
                return ""

        client = _C()
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            ok = run_one_interaction("想自殺", store, client)
        self.assertFalse(ok)
        # 不應呼叫 generate_restate
        self.assertEqual(client.generate_restate_calls, [])
        # 不應寫入任何新端點
        self.assertEqual(len(store.all()), 4)  # 仍是前兩次 start+end
        self.assertIn("意圖軌跡異常", captured.getvalue())


# ----------------------------------------------------------------------
# P1-2：BaseLLMClient 抽象介面
# ----------------------------------------------------------------------
class BaseClientInterfaceTests(unittest.TestCase):
    def test_llm_client_is_base_subclass(self):
        self.assertTrue(issubclass(LLMClient, BaseLLMClient))


# ----------------------------------------------------------------------
# P1-3：reasoning_steps 欄位
# v1 設計（20260414_逗福Tofu_復述引擎_Prompt設計_v1）改用七個檢查模式，
# reasoning_steps 的結構同步換成 selected_modes / selection_reason / ...
# ----------------------------------------------------------------------
class _FakeBlock:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.content = [_FakeBlock(text)]


class ReasoningStepsTests(unittest.TestCase):
    def _make_client(self, api_text: str) -> LLMClient:
        client = LLMClient.__new__(LLMClient)
        client._model = "test-model"
        client._notifier = None
        client.fallback_mode = False
        client._client = MagicMock()
        client._client.messages.create = MagicMock(
            return_value=_FakeResponse(api_text)
        )
        return client

    def test_reasoning_steps_in_restate_prompt(self):
        """generate_restate 的 system prompt 應含七個檢查模式指令。

        v1 設計（20260414_逗福Tofu_復述引擎_Prompt設計_v1）：
        - 取代原本的六步 OS
        - 七個模式：定性質 / 問動機 / 找隱藏變數 / 問相關的人 /
          問歷史和經驗 / 反方向提問 / 外部到內部（價值觀確認）
        - reasoning_steps 改為以 selected_modes 為主
        """
        fake_json = (
            '{"restate_text": "你想辦派對",'
            ' "gap_questions": [],'
            ' "gap_categories": ["budget"],'
            ' "inferred_goal": "辦派對",'
            ' "inferred_motivation": "",'
            ' "inferred_constraints": [],'
            ' "reasoning_steps": {"selected_modes": [1, 2], '
            '"selection_reason": "使用者沒說派對性質和動機", '
            '"definition": "辦派對", '
            '"absent_dimensions": ["派對類型", "目的"]}}'
        )
        client = self._make_client(fake_json)
        with patch("src.api.claude_client.time.sleep"):
            out = client.generate_restate(
                "辦派對",
                baseline_summary={},
                allowed_categories=["budget"],
            )
        # reasoning_steps 應被保留
        self.assertIn("reasoning_steps", out)
        self.assertEqual(out["reasoning_steps"]["selected_modes"], [1, 2])
        self.assertEqual(out["reasoning_steps"]["definition"], "辦派對")
        # 驗證 prompt 中含七個檢查模式關鍵字
        call = client._client.messages.create.call_args
        system_sent = call.kwargs.get("system") or call.args[0]
        self.assertIn("七個檢查模式", system_sent)
        self.assertIn("定性質", system_sent)
        self.assertIn("問動機", system_sent)
        self.assertIn("找隱藏變數", system_sent)
        self.assertIn("反方向提問", system_sent)
        # 驗證禁止事項存在
        self.assertIn("禁止事項", system_sent)
        # 驗證基線動態注入 placeholder 有被填入
        user_sent = call.kwargs.get("messages", [{}])[0].get(
            "content", ""
        ) if call.kwargs.get("messages") else ""
        self.assertIn("baseline_gap_categories", user_sent)
        self.assertIn("baseline_frequent_topics", user_sent)

    def test_missing_reasoning_steps_is_backward_compatible(self):
        """API 沒回 reasoning_steps 時不應爆炸。"""
        fake_json = (
            '{"restate_text": "你想辦派對",'
            ' "gap_questions": [],'
            ' "gap_categories": ["budget"],'
            ' "inferred_goal": "辦派對",'
            ' "inferred_motivation": "",'
            ' "inferred_constraints": []}'
        )
        client = self._make_client(fake_json)
        with patch("src.api.claude_client.time.sleep"):
            out = client.generate_restate(
                "辦派對",
                baseline_summary={},
                allowed_categories=["budget"],
            )
        self.assertNotIn("reasoning_steps", out)
        self.assertEqual(out["inferred_goal"], "辦派對")


# ----------------------------------------------------------------------
# CBP：不同 topic 的基線不互相污染（整合）
# ----------------------------------------------------------------------
class CBPIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "endpoints.json")
        self.store = EndpointStore(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_topic_recorded_on_start(self):
        from src.main import run_one_interaction

        class _C:
            fallback_mode = False

            def generate_restate(self, user_input, baseline_summary, allowed_categories):
                return {
                    "restate_text": "復述",
                    "gap_questions": [],
                    "gap_categories": ["venue", "headcount"],
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

            def execute_task(self, goal, motivation, constraints, user_profile=None):
                return "針對派對的建議，先確認人數，接著找合適的場地安排時程。"

        client = _C()
        with patch.object(builtins, "input", lambda prompt="": "三十人"):
            ok = run_one_interaction("辦一場生日派對", self.store, client)
        self.assertTrue(ok)
        starts = [r for r in self.store.all() if r["type"] == "start"]
        self.assertEqual(starts[0]["start_data"]["topic"], "event")


if __name__ == "__main__":
    unittest.main()
