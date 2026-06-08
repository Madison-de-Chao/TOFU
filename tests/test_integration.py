"""整合測試：Mock LLMClient，跑完整的互動流程。

流程：
1. 建立暫存 EndpointStore
2. Mock LLMClient 回傳預設 payload
3. 跑 run_one_interaction → 寫入 start + end
4. 驗證端點紀錄完整、基線可計算、提問單門檻判斷正確
"""

from __future__ import annotations

import builtins
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.api.claude_client import LLMClientError
from src.middleware import baseline as baseline_mod
from src.middleware import intake_form as form_mod
from src.middleware.endpoint import EndpointStore


class _FakeClient:
    """極簡的 LLMClient 替身，回傳固定 payload。

    可以透過 `raise_on` 字典讓特定方法拋出例外，測試錯誤路徑。
    key 是方法名字（generate_restate / merge_confirmation / execute_task），
    value 是要拋的例外實例。
    """

    fallback_mode = False

    def __init__(self, raise_on: dict | None = None) -> None:
        self.generate_restate_calls: list[dict] = []
        self.merge_confirmation_calls: list[dict] = []
        self.execute_task_calls: list[dict] = []
        self.raise_on: dict = raise_on or {}

    def _maybe_raise(self, method: str) -> None:
        if method in self.raise_on:
            raise self.raise_on[method]

    def generate_restate(self, user_input, baseline_summary, allowed_categories, **kwargs):
        # v4.1：`prior_rounds_context` 非空代表多輪迴圈已進第 2 輪（含），
        # 直接回空 gap_questions 觸發收斂閘門的 info_sufficient 分支，
        # 讓既有測試不必修改就能沿用（單輪行為等價）。
        self.generate_restate_calls.append(
            {"user_input": user_input, "baseline": baseline_summary}
        )
        self._maybe_raise("generate_restate")
        if kwargs.get("prior_rounds_context"):
            return {
                "restate_text": f"了解「{user_input}」，照現有資訊處理。",
                "gap_questions": [],
                "gap_categories": [],
                "inferred_goal": user_input,
                "inferred_motivation": "",
                "inferred_constraints": [],
            }
        return {
            "restate_text": f"你想「{user_input}」。預算、時間？",
            "gap_questions": ["預算？", "時間？"],
            "gap_categories": ["budget", "time"],
            "inferred_goal": user_input,
            "inferred_motivation": "",
            "inferred_constraints": [],
        }

    def merge_confirmation(self, original_input, restate_text, user_confirmation_text):
        self.merge_confirmation_calls.append(
            {"original": original_input, "user": user_confirmation_text}
        )
        self._maybe_raise("merge_confirmation")
        return {
            "goal": original_input,
            "motivation": "",
            "constraints": [user_confirmation_text] if user_confirmation_text else [],
            "answered_categories": ["budget"],
        }

    def execute_task(self, goal, motivation, constraints, user_profile=None):
        self.execute_task_calls.append(
            {"goal": goal, "constraints": constraints}
        )
        self._maybe_raise("execute_task")
        return f"針對 {goal} 的建議：找場地、抓時程。"


class FullFlowIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "endpoints.json")
        self.store = EndpointStore(self.path)
        self.client = _FakeClient()

    def tearDown(self):
        self.tmp.cleanup()

    def _run_with_inputs(self, main_fn, responses: list[str]) -> None:
        """Patch input() 以回應預先準備好的回應清單。"""
        it = iter(responses)
        with patch.object(builtins, "input", lambda prompt="": next(it)):
            main_fn()

    def test_single_interaction_writes_start_and_end(self):
        from src.main import run_one_interaction

        with patch.object(builtins, "input", lambda prompt="": "預算三萬"):
            ok = run_one_interaction(
                "辦派對",
                self.store,
                self.client,
                ask_satisfaction=False,
            )
        self.assertTrue(ok)

        all_rows = self.store.all()
        self.assertEqual(len(all_rows), 2)  # start + end
        starts = [r for r in all_rows if r["type"] == "start"]
        ends = [r for r in all_rows if r["type"] == "end"]
        self.assertEqual(len(starts), 1)
        self.assertEqual(len(ends), 1)
        self.assertEqual(starts[0]["event_id"], ends[0]["event_id"])

        sd = starts[0]["start_data"]
        self.assertEqual(sd["user_input"], "辦派對")
        self.assertEqual(sd["goal"], "辦派對")
        self.assertIn("budget", sd["gap_categories"])
        self.assertIn("預算三萬", sd["constraints"])
        # _answered_categories 是內部輔助，不落地
        self.assertNotIn("_answered_categories", sd)

        ed = ends[0]["end_data"]
        self.assertIn("result", ed)
        self.assertEqual(ed["user_satisfaction"], "no_feedback")

    def test_empty_user_response_cancels_interaction(self):
        from src.main import run_one_interaction

        with patch.object(builtins, "input", lambda prompt="": ""):
            ok = run_one_interaction(
                "辦派對",
                self.store,
                self.client,
                ask_satisfaction=False,
            )
        self.assertFalse(ok)
        # 不應該寫入任何端點
        self.assertEqual(len(self.store.all()), 0)

    def test_five_interactions_enable_form(self):
        from src.main import run_one_interaction

        for i in range(5):
            with patch.object(builtins, "input", lambda prompt="": "預算三萬"):
                run_one_interaction(
                    f"任務 {i}",
                    self.store,
                    self.client,
                    ask_satisfaction=False,
                )

        self.assertEqual(self.store.completed_count(), 5)
        self.assertTrue(form_mod.should_generate(self.store.completed_count()))

        baseline = baseline_mod.compute_baseline(self.store.all())
        self.assertTrue(baseline_mod.is_mature(baseline))
        self.assertIn("budget", baseline["top_gap_categories"])

    def test_satisfaction_collection_satisfied(self):
        from src.main import run_one_interaction

        inputs = iter(["預算三萬", "是"])
        with patch.object(builtins, "input", lambda prompt="": next(inputs)):
            ok = run_one_interaction(
                "辦派對",
                self.store,
                self.client,
                ask_satisfaction=True,
            )
        self.assertTrue(ok)
        ends = [r for r in self.store.all() if r["type"] == "end"]
        self.assertEqual(ends[0]["end_data"]["user_satisfaction"], "satisfied")

    def test_satisfaction_collection_unsatisfied_with_reason(self):
        from src.main import run_one_interaction

        inputs = iter(["預算三萬", "否", "結果不具體"])
        with patch.object(builtins, "input", lambda prompt="": next(inputs)):
            ok = run_one_interaction(
                "辦派對",
                self.store,
                self.client,
                ask_satisfaction=True,
            )
        self.assertTrue(ok)
        ends = [r for r in self.store.all() if r["type"] == "end"]
        ed = ends[0]["end_data"]
        self.assertEqual(ed["user_satisfaction"], "unsatisfied")
        self.assertEqual(ed.get("unsatisfied_reason"), "結果不具體")

    def test_baseline_summary_output_contains_counts(self):
        from src.main import run_one_interaction

        for i in range(3):
            with patch.object(builtins, "input", lambda prompt="": "預算三萬"):
                run_one_interaction(
                    f"任務 {i}",
                    self.store,
                    self.client,
                    ask_satisfaction=False,
                )

        baseline = baseline_mod.compute_baseline(self.store.all())
        summary = baseline_mod.summarize_baseline(baseline)
        self.assertIn("互動次數：3", summary)
        self.assertIn("修正率", summary)

    def test_restate_failure_aborts_without_writing(self):
        """P0-1A：generate_restate 拋例外時，不寫任何端點、回傳 False。"""
        from src.main import run_one_interaction
        import io
        import contextlib

        self.client.raise_on["generate_restate"] = LLMClientError("restate boom")

        captured = io.StringIO()
        with patch.object(builtins, "input", lambda prompt="": "預算三萬"), \
             contextlib.redirect_stdout(captured):
            ok = run_one_interaction(
                "辦派對",
                self.store,
                self.client,
                ask_satisfaction=False,
            )

        self.assertFalse(ok)
        self.assertEqual(len(self.store.all()), 0)
        output = captured.getvalue()
        self.assertTrue(
            "復述失敗" in output or "錯誤" in output,
            f"expected error output, got: {output}",
        )

    def test_merge_failure_writes_no_end_endpoint(self):
        """P0-1B：merge_confirmation 拋例外時，不寫 end 端點、回傳 False。"""
        from src.main import run_one_interaction
        import io
        import contextlib

        self.client.raise_on["merge_confirmation"] = LLMClientError("merge boom")

        captured = io.StringIO()
        with patch.object(builtins, "input", lambda prompt="": "預算三萬"), \
             contextlib.redirect_stdout(captured):
            ok = run_one_interaction(
                "辦派對",
                self.store,
                self.client,
                ask_satisfaction=False,
            )

        self.assertFalse(ok)
        # start 端點應該還沒寫（merge 失敗發生在 append_start 之前）
        ends = [r for r in self.store.all() if r["type"] == "end"]
        self.assertEqual(ends, [])
        output = captured.getvalue()
        self.assertTrue(
            "合併確認回答失敗" in output or "錯誤" in output,
            f"expected error output, got: {output}",
        )

    def test_execute_failure_still_writes_end_endpoint(self):
        """P0-1C：execute_task 拋例外時，仍照寫 end 端點、result 含『任務執行失敗』。"""
        from src.main import run_one_interaction
        import io
        import contextlib

        self.client.raise_on["execute_task"] = LLMClientError("execute boom")

        captured = io.StringIO()
        with patch.object(builtins, "input", lambda prompt="": "預算三萬"), \
             contextlib.redirect_stdout(captured):
            ok = run_one_interaction(
                "辦派對",
                self.store,
                self.client,
                ask_satisfaction=False,
            )

        self.assertTrue(ok)
        all_rows = self.store.all()
        starts = [r for r in all_rows if r["type"] == "start"]
        ends = [r for r in all_rows if r["type"] == "end"]
        self.assertEqual(len(starts), 1)
        self.assertEqual(len(ends), 1)
        self.assertIn("任務執行失敗", ends[0]["end_data"]["result"])

    def test_export_command_writes_markdown(self):
        from src.main import run_export_command, run_one_interaction

        with patch.object(builtins, "input", lambda prompt="": "預算三萬"):
            run_one_interaction(
                "辦派對",
                self.store,
                self.client,
                ask_satisfaction=False,
            )

        # P1-2: 匯出路徑應該跟隨 store.path，而不是全域 DATA_PATH。
        # 這裡故意把 DATA_PATH 指到一個不存在的垃圾路徑，
        # 驗證 run_export_command 不會誤用它。
        with patch("src.main.DATA_PATH", "/nonexistent/should_not_be_used.json"):
            run_export_command(self.store)

        export_path = Path(self.path).parent / "endpoints_export.md"
        self.assertTrue(export_path.exists())
        content = export_path.read_text(encoding="utf-8")
        self.assertIn("辦派對", content)
        self.assertIn("逗福Tofu 端點紀錄匯出", content)

    def test_export_path_follows_store_not_default(self):
        """P1-2：即使 DATA_PATH 指向其他路徑，匯出仍應寫到 store 自己的目錄。"""
        from src.main import run_export_command, run_one_interaction

        with patch.object(builtins, "input", lambda prompt="": "預算三萬"):
            run_one_interaction(
                "辦派對",
                self.store,
                self.client,
                ask_satisfaction=False,
            )

        # 建立一個完全不同的「預設」路徑，並讓 DATA_PATH 指過去
        other_dir = tempfile.TemporaryDirectory()
        self.addCleanup(other_dir.cleanup)
        other_path = os.path.join(other_dir.name, "other.json")

        with patch("src.main.DATA_PATH", other_path):
            run_export_command(self.store)

        # 匯出檔應該出現在 store.path 的同層，而不是 DATA_PATH 的同層
        expected_export = Path(self.store.path).parent / "endpoints_export.md"
        wrong_export = Path(other_path).parent / "endpoints_export.md"
        self.assertTrue(expected_export.exists())
        self.assertFalse(wrong_export.exists())


class SatisfactionCollectionTests(unittest.TestCase):
    """P1-5：_collect_satisfaction 的 skip 與空字串分支。"""

    def test_skip_returns_no_feedback(self):
        from src.main import _collect_satisfaction

        for answer in ["跳過", "skip", "s", "SKIP"]:
            with patch.object(builtins, "input", lambda prompt="", a=answer: a):
                sat, reason = _collect_satisfaction()
            self.assertEqual(sat, "no_feedback")
            self.assertEqual(reason, "")

    def test_empty_string_returns_no_feedback(self):
        from src.main import _collect_satisfaction

        with patch.object(builtins, "input", lambda prompt="": ""):
            sat, reason = _collect_satisfaction()
        self.assertEqual(sat, "no_feedback")
        self.assertEqual(reason, "")


class EmptyExportTests(unittest.TestCase):
    """P1-5：空 store 跑 /export 不應產出檔案。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "endpoints.json")
        self.store = EndpointStore(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_export_with_no_data_does_nothing(self):
        import contextlib
        import io
        from src.main import run_export_command

        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            run_export_command(self.store)

        export_path = Path(self.path).parent / "endpoints_export.md"
        self.assertFalse(export_path.exists())
        self.assertIn("尚無已完成的互動可匯出", captured.getvalue())


class StatsCommandTests(unittest.TestCase):
    """P1-7：/stats 指令輸出驗證。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "endpoints.json")
        self.store = EndpointStore(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def _seed_endpoints(self) -> None:
        """寫入 5 筆完整端點：不同滿意度、修正狀況、gap_categories。"""
        # 1. satisfied, 沒修正, budget+time
        self.store.append_start(
            "evt1",
            {
                "user_input": "a",
                "tofu_understanding": "",
                "gap_questions": [],
                "user_confirmation": "confirmed",
                "user_modifications": "",
                "goal": "a",
                "motivation": "",
                "constraints": [],
                "gap_categories": ["budget", "time"],
            },
        )
        self.store.append_end(
            "evt1",
            {"result": "r1", "user_satisfaction": "satisfied", "deviation_from_goal": ""},
        )

        # 2. unsatisfied, 有修正, budget+venue
        self.store.append_start(
            "evt2",
            {
                "user_input": "b",
                "tofu_understanding": "",
                "gap_questions": [],
                "user_confirmation": "modified",
                "user_modifications": "其實不是",
                "goal": "b",
                "motivation": "",
                "constraints": [],
                "gap_categories": ["budget", "venue"],
            },
        )
        self.store.append_end(
            "evt2",
            {"result": "r2", "user_satisfaction": "unsatisfied", "deviation_from_goal": ""},
        )

        # 3. no_feedback, 沒修正, budget+time
        self.store.append_start(
            "evt3",
            {
                "user_input": "c",
                "tofu_understanding": "",
                "gap_questions": [],
                "user_confirmation": "confirmed",
                "user_modifications": "",
                "goal": "c",
                "motivation": "",
                "constraints": [],
                "gap_categories": ["budget", "time"],
            },
        )
        self.store.append_end(
            "evt3",
            {"result": "r3", "user_satisfaction": "no_feedback", "deviation_from_goal": ""},
        )

        # 4. satisfied, 有修正, time
        self.store.append_start(
            "evt4",
            {
                "user_input": "d",
                "tofu_understanding": "",
                "gap_questions": [],
                "user_confirmation": "modified",
                "user_modifications": "應該是五萬",
                "goal": "d",
                "motivation": "",
                "constraints": [],
                "gap_categories": ["time"],
            },
        )
        self.store.append_end(
            "evt4",
            {"result": "r4", "user_satisfaction": "satisfied", "deviation_from_goal": ""},
        )

        # 5. no_feedback, 沒修正, venue
        self.store.append_start(
            "evt5",
            {
                "user_input": "e",
                "tofu_understanding": "",
                "gap_questions": [],
                "user_confirmation": "confirmed",
                "user_modifications": "",
                "goal": "e",
                "motivation": "",
                "constraints": [],
                "gap_categories": ["venue"],
            },
        )
        self.store.append_end(
            "evt5",
            {"result": "r5", "user_satisfaction": "no_feedback", "deviation_from_goal": ""},
        )

    def test_stats_with_empty_store(self):
        import contextlib
        import io
        from src.main import run_stats_command

        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            run_stats_command(self.store)
        self.assertIn("尚無任何互動紀錄", captured.getvalue())

    def test_stats_with_seeded_data(self):
        import contextlib
        import io
        from src.main import run_stats_command

        self._seed_endpoints()

        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            run_stats_command(self.store)
        out = captured.getvalue()

        # 總互動數 5
        self.assertIn("總互動數：5", out)
        # 修正率 = 2/5 = 40.0%
        self.assertIn("40.0%", out)
        self.assertIn("2/5", out)
        # 滿意度分布：satisfied 2 / unsatisfied 1 / no_feedback 2
        self.assertIn("satisfied 2", out)
        self.assertIn("unsatisfied 1", out)
        self.assertIn("no_feedback 2", out)
        # 最常被補位的面向：budget 3 次、time 3 次、venue 2 次（top 3）
        self.assertIn("budget", out)
        self.assertIn("3次", out)
        # 提問單狀態：5 筆 → 已生成
        self.assertIn("已生成", out)

    def test_stats_is_in_help(self):
        from src.main import _render_help

        help_text = _render_help()
        self.assertIn("/stats", help_text)


class MetaQueryAndAboutTests(unittest.TestCase):
    """System Prompt v2.0：meta 問題攔截 + /about 自我介紹。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "endpoints.json")
        self.store = EndpointStore(self.path)
        self.client = _FakeClient()

    def tearDown(self):
        self.tmp.cleanup()

    def test_is_meta_query_matches_spec_examples(self):
        from src.utils.meta import is_meta_query

        positive = [
            "你是什麼",
            "你是誰",
            "你能做什麼？",
            "你是幹嘛的",
            "怎麼用？",
            "請介紹一下",
            "What are you?",
            "who are you",
            # v2.0 新增的四個比較型關鍵詞
            "你跟其他 AI 有什麼不同",
            "你跟別的 AI 差在哪",
            "跟 ChatGPT 的差別是什麼？你的差別在哪",
            "你的不同之處",
        ]
        for q in positive:
            self.assertTrue(is_meta_query(q), f"{q!r} should be meta")

        negative = [
            "",
            "我想辦一場生日派對",
            "幫我寫一封信",
            "預算三萬",
            "你好",  # v2.0：打招呼不是 meta，應走正常流程由 LLM 處理
            "我想跟你聊聊天",  # v2.0：閒聊不是 meta
        ]
        for q in negative:
            self.assertFalse(is_meta_query(q), f"{q!r} should NOT be meta")

    def test_meta_query_in_run_one_interaction_bypasses_flow(self):
        """Meta 問題應直接印 _ABOUT_TEXT、不寫端點、不呼叫 client。"""
        import contextlib
        import io

        from src.main import run_one_interaction

        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            ok = run_one_interaction(
                "你是什麼",
                self.store,
                self.client,
                ask_satisfaction=False,
            )

        self.assertFalse(ok)
        # 不寫入任何端點
        self.assertEqual(len(self.store.all()), 0)
        # 不呼叫 client 任何方法
        self.assertEqual(self.client.generate_restate_calls, [])
        self.assertEqual(self.client.merge_confirmation_calls, [])
        self.assertEqual(self.client.execute_task_calls, [])
        # 輸出包含自我介紹的關鍵資訊
        out = captured.getvalue()
        self.assertIn("逗福Tofu", out)
        self.assertIn("認知中間層", out)
        self.assertIn("復述確認", out)

    def test_about_text_contains_key_identity_info(self):
        from src.main import _ABOUT_TEXT

        # 基本身份關鍵字
        self.assertIn("逗福Tofu", _ABOUT_TEXT)
        self.assertIn("認知中間層", _ABOUT_TEXT)
        self.assertIn("復述確認", _ABOUT_TEXT)
        self.assertIn("yyuniverse.com", _ABOUT_TEXT)
        # v2.0：三條底線與 RLHF 批判
        self.assertIn("說真話", _ABOUT_TEXT)
        self.assertIn("說人話", _ABOUT_TEXT)
        self.assertIn("守住邊界", _ABOUT_TEXT)
        self.assertIn("你想聽的話", _ABOUT_TEXT)
        self.assertIn("你需要聽的話", _ABOUT_TEXT)

    def test_identity_prompt_has_three_baselines_and_is_under_2000_chars(self):
        """v2.0 身份基底必須包含三條底線，且長度受控。"""
        from src.api.claude_client import TOFU_IDENTITY_PROMPT

        self.assertLess(
            len(TOFU_IDENTITY_PROMPT),
            2000,
            "TOFU_IDENTITY_PROMPT 必須短於 2000 字元（規格書 v2.0 要求）",
        )
        self.assertIn("說真話", TOFU_IDENTITY_PROMPT)
        self.assertIn("說人話", TOFU_IDENTITY_PROMPT)
        self.assertIn("守住邊界", TOFU_IDENTITY_PROMPT)
        # RLHF 批判段落
        self.assertIn("討好", TOFU_IDENTITY_PROMPT)
        # 最終目標段落
        self.assertIn("不需要你", TOFU_IDENTITY_PROMPT)

    def test_meta_comparative_query_bypasses_flow(self):
        """v2.0：「你跟其他 AI 有什麼不同」這類比較型提問也應走 meta 路徑。"""
        import contextlib
        import io

        from src.main import run_one_interaction

        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            ok = run_one_interaction(
                "你跟其他 AI 有什麼不同",
                self.store,
                self.client,
                ask_satisfaction=False,
            )

        self.assertFalse(ok)
        self.assertEqual(len(self.store.all()), 0)
        self.assertEqual(self.client.generate_restate_calls, [])
        # 輸出應含 v2.0 的 RLHF 批判
        out = captured.getvalue()
        self.assertIn("逗福Tofu", out)
        self.assertIn("你需要聽的話", out)

    def test_about_command_is_in_help(self):
        from src.main import _render_help

        help_text = _render_help()
        self.assertIn("/about", help_text)

    def test_non_meta_input_still_runs_full_flow(self):
        """規格開發過程規則第 2 條：一般需求仍走正常復述流程。

        v4.1：default 模式走多輪補位迴圈。_FakeClient 第 1 輪回非空
        gap_questions、第 2 輪（有 prior_rounds_context）回空以觸發
        info_sufficient 收斂，所以 generate_restate 會被呼叫 2 次。
        """
        from src.main import run_one_interaction

        with patch.object(builtins, "input", lambda prompt="": "預算三萬"):
            ok = run_one_interaction(
                "我想辦一場生日派對",
                self.store,
                self.client,
                ask_satisfaction=False,
            )
        self.assertTrue(ok)
        # v4.1：多輪迴圈 → 至少呼叫一次；fake client 第 2 輪會觸發收斂
        self.assertGreaterEqual(len(self.client.generate_restate_calls), 1)
        self.assertLessEqual(len(self.client.generate_restate_calls), 2)
        # 並寫入 start + end 端點（整輪互動仍只寫一對）
        self.assertEqual(len(self.store.all()), 2)
        # 收斂閘門三欄位應有寫入
        starts = [r for r in self.store.all() if r["type"] == "start"]
        sd = starts[0]["start_data"]
        self.assertIn("convergence_reason", sd)
        self.assertIn("total_gap_rounds", sd)
        self.assertIn("unresolved_gaps", sd)
        self.assertGreaterEqual(sd["total_gap_rounds"], 1)


class FallbackMetaQueryTests(unittest.TestCase):
    """System Prompt v1.0：fallback 模式下的 meta 問題攔截。"""

    def test_fallback_generate_restate_intercepts_meta(self):
        from src.api.claude_client import _fallback_generate_restate

        result = _fallback_generate_restate(
            user_input="你是什麼",
            baseline_summary={},
            allowed_categories=["budget", "time", "venue"],
        )
        self.assertIn("逗福Tofu", result["restate_text"])
        self.assertIn("認知中間層", result["restate_text"])
        # v2.0：RLHF 批判 + 三條底線應在 fallback meta 回應中
        self.assertIn("你需要聽的話", result["restate_text"])
        self.assertIn("說真話", result["restate_text"])
        self.assertEqual(result["gap_questions"], [])
        self.assertEqual(result["gap_categories"], [])
        self.assertTrue(result.get("_is_meta_query"))

    def test_fallback_generate_restate_intercepts_comparative_meta(self):
        """v2.0：比較型 meta 問題在 fallback 下也要被攔截。"""
        from src.api.claude_client import _fallback_generate_restate

        result = _fallback_generate_restate(
            user_input="你跟別的 AI 有什麼不同",
            baseline_summary={},
            allowed_categories=["budget", "time"],
        )
        self.assertTrue(result.get("_is_meta_query"))
        self.assertIn("逗福Tofu", result["restate_text"])

    def test_fallback_generate_restate_non_meta_still_gaps(self):
        """一般需求在 fallback 模式下應仍產生 gap_questions。"""
        from src.api.claude_client import _fallback_generate_restate

        result = _fallback_generate_restate(
            user_input="辦一場生日派對",
            baseline_summary={},
            allowed_categories=["budget", "time", "venue"],
        )
        self.assertFalse(result.get("_is_meta_query"))
        self.assertTrue(len(result["gap_questions"]) > 0)
        self.assertTrue(len(result["gap_categories"]) > 0)


class CorruptedEndpointRecoveryTests(unittest.TestCase):
    """JSON 損壞時應該能自動備份並重新開始。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "endpoints.json")

    def tearDown(self):
        self.tmp.cleanup()

    def test_corrupted_json_is_quarantined(self):
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("{{not valid")

        store = EndpointStore(self.path)
        self.assertEqual(store.all(), [])
        self.assertTrue(os.path.exists(self.path + ".bak"))

        # 並且後續仍可正常寫入
        store.append_start("evt", {"goal": "test"})
        self.assertEqual(len(store.all()), 1)

    def test_top_level_non_list_triggers_quarantine(self):
        """P1-5：top-level JSON 是 dict / string 時應觸發 quarantine。"""
        for bad_payload in ['{}', '"a plain string"', '42']:
            with self.subTest(payload=bad_payload):
                # 清乾淨
                bak = self.path + ".bak"
                if os.path.exists(bak):
                    os.remove(bak)
                with open(self.path, "w", encoding="utf-8") as f:
                    f.write(bad_payload)

                store = EndpointStore(self.path)
                # 觸發 quarantine 的瞬間是讀取時
                self.assertEqual(store.all(), [])
                self.assertTrue(
                    os.path.exists(bak),
                    f"expected .bak for payload {bad_payload}",
                )

                # 之後應能正常寫入
                store.append_start("evt-x", {"goal": "recovered"})
                self.assertEqual(len(store.all()), 1)

    def test_atomic_write_creates_valid_json(self):
        store = EndpointStore(self.path)
        store.append_start("evt1", {"goal": "a"})
        store.append_end("evt1", {"result": "r"})

        # 讀回來手動解析，確保格式合法
        import json
        with open(self.path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(len(data), 2)
        self.assertEqual(data[0]["event_id"], "evt1")
        self.assertEqual(data[1]["event_id"], "evt1")


if __name__ == "__main__":
    unittest.main()
