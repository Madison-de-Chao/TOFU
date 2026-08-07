"""v4.1 收斂閘門（Convergence Gate）的單元 + 整合測試。

對應規格 docs/philosophy/20260424_逗福Tofu_收斂閘門_實作規格_v1.md §測試計畫。
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

from src.middleware import confirmation as confirm_mod
from src.middleware.endpoint import EndpointStore, make_start_data


# ----------------------------------------------------------------------
# 單元測試：should_continue_gap_filling
# ----------------------------------------------------------------------
class ShouldContinueGapFillingTests(unittest.TestCase):
    def test_make_convergence_state_defaults(self):
        state = confirm_mod.make_convergence_state()
        self.assertEqual(state["total_rounds"], 0)
        self.assertEqual(state["empty_round_count"], 0)
        self.assertFalse(state["converged"])
        self.assertIsNone(state["convergence_reason"])

    def test_consecutive_three_empty_rounds_triggers_empty_streak(self):
        """規格 §條件一：連續 3 輪空輪 → empty_streak，第 3 輪後停。"""
        state = confirm_mod.make_convergence_state()
        # 輪 1：空輪
        self.assertTrue(confirm_mod.should_continue_gap_filling(state, False))
        # 輪 2：空輪
        self.assertTrue(confirm_mod.should_continue_gap_filling(state, False))
        # 輪 3：空輪 → 收斂
        self.assertFalse(confirm_mod.should_continue_gap_filling(state, False))
        self.assertEqual(state["convergence_reason"], "empty_streak")
        self.assertEqual(state["total_rounds"], 3)
        self.assertEqual(state["empty_round_count"], 3)
        self.assertTrue(state["converged"])

    def test_five_rounds_with_info_triggers_max_rounds(self):
        """規格 §條件二：5 輪都有效資訊 → max_rounds。"""
        state = confirm_mod.make_convergence_state()
        for _ in range(4):
            self.assertTrue(confirm_mod.should_continue_gap_filling(state, True))
        # 第 5 輪：總數達硬上限 → 收斂
        self.assertFalse(confirm_mod.should_continue_gap_filling(state, True))
        self.assertEqual(state["convergence_reason"], "max_rounds")
        self.assertEqual(state["total_rounds"], 5)
        self.assertEqual(state["empty_round_count"], 0)

    def test_effective_round_resets_empty_streak(self):
        """規格 §測試計畫 row 3：空/空/有/空/空 → empty_streak 不觸發、max_rounds 觸發。"""
        state = confirm_mod.make_convergence_state()
        self.assertTrue(confirm_mod.should_continue_gap_filling(state, False))  # 輪1：空
        self.assertTrue(confirm_mod.should_continue_gap_filling(state, False))  # 輪2：空
        self.assertTrue(confirm_mod.should_continue_gap_filling(state, True))   # 輪3：有 → 歸零
        self.assertEqual(state["empty_round_count"], 0)
        self.assertTrue(confirm_mod.should_continue_gap_filling(state, False))  # 輪4：空
        # 輪 5：空，empty=2 < 3，但 total=5 → 先到 max_rounds
        self.assertFalse(confirm_mod.should_continue_gap_filling(state, False))
        self.assertEqual(state["convergence_reason"], "max_rounds")
        self.assertEqual(state["total_rounds"], 5)

    def test_empty_then_max_rounds_would_never_trigger_both(self):
        """先到先停：empty_streak 先到時不再檢查 max_rounds。"""
        state = confirm_mod.make_convergence_state()
        for _ in range(2):
            self.assertTrue(confirm_mod.should_continue_gap_filling(state, False))
        # 輪 3：空 → empty_streak 立即觸發、不看 total_rounds
        self.assertFalse(confirm_mod.should_continue_gap_filling(state, False))
        self.assertEqual(state["convergence_reason"], "empty_streak")
        self.assertLess(state["total_rounds"], confirm_mod.CONVERGENCE_MAX_ROUNDS)


# ----------------------------------------------------------------------
# 單元測試：has_effective_info
# ----------------------------------------------------------------------
class HasEffectiveInfoTests(unittest.TestCase):
    def test_empty_response_is_ineffective(self):
        self.assertFalse(confirm_mod.has_effective_info("", []))
        self.assertFalse(confirm_mod.has_effective_info("   ", []))

    def test_short_response_under_threshold_is_ineffective(self):
        """規格 §「有效資訊」的判定 方式一：純短回應（≤ 5 字）→ 無效。"""
        for short in ["好", "對", "沒有", "沒", "OK", "嗯", "好的", "是"]:
            with self.subTest(short=short):
                self.assertFalse(confirm_mod.has_effective_info(short, []))

    def test_long_novel_response_is_effective(self):
        """回應夠長，且含新名詞 → 有效。"""
        existing: list[dict] = []
        self.assertTrue(
            confirm_mod.has_effective_info(
                "我有五位同事會一起參加、預算三萬元、地點在台北信義區",
                existing,
            )
        )

    def test_response_with_only_known_nouns_is_ineffective(self):
        """回應的 token 全部已在既有端點中 → 無效（沒有新增事實）。"""
        # 先用含常見詞的端點當既有資料
        existing = [{
            "start_data": {
                "goal": "我要在台北信義區辦一場三萬元的生日派對",
                "user_input": "辦派對",
                "motivation": "",
                "tofu_understanding": "辦一場三萬元的生日派對",
                "constraints": ["台北", "信義區", "三萬"],
            }
        }]
        # 完全重複已知名詞的長回應。
        # v0.8 斷詞改用繁中大詞典後，「生日派對」是單一 token；
        # 回應若只寫「生日」會被視為新 token（規格「寧可判多不判少」
        # 容許的方向），此處用與端點相同的複合詞驗證「全部已知 → 無效」。
        self.assertFalse(
            confirm_mod.has_effective_info(
                "台北 信義區 三萬 派對 生日派對",
                existing,
            )
        )

    def test_response_with_new_nouns_beyond_endpoints_is_effective(self):
        """回應中含一個沒見過的名詞（專有名詞）→ 有效。"""
        existing = [{
            "start_data": {
                "goal": "辦生日派對",
                "user_input": "生日派對",
                "motivation": "",
                "tofu_understanding": "生日派對",
                "constraints": [],
            }
        }]
        # "麻婆豆腐" 不在既有資料中
        self.assertTrue(
            confirm_mod.has_effective_info(
                "還想準備麻婆豆腐當主餐給大家吃",
                existing,
            )
        )

    def test_existing_endpoints_none_treated_as_empty(self):
        self.assertTrue(
            confirm_mod.has_effective_info("一個很長的新陳述包含各種資訊", None)
        )


# ----------------------------------------------------------------------
# 整合測試：run_one_interaction default 分支的多輪收斂
# ----------------------------------------------------------------------
class _ScriptedClient:
    """逐輪腳本化的 LLMClient 替身。

    gap_questions 依 `rounds_script` 序列逐輪回傳；用完就回空（= info_sufficient）。
    merge_confirmation / execute_task 永遠成功。
    """

    fallback_mode = False

    def __init__(self, rounds_script: list[list[str]]):
        self.rounds_script = list(rounds_script)
        self.call_idx = 0
        self.generate_restate_calls: list[dict] = []
        self.merge_confirmation_calls: list[dict] = []
        self.execute_task_calls: list[dict] = []

    def generate_restate(self, user_input, baseline_summary, allowed_categories, **kwargs):
        self.generate_restate_calls.append({
            "user_input": user_input,
            "prior_rounds_context": kwargs.get("prior_rounds_context"),
        })
        if self.call_idx < len(self.rounds_script):
            gap_questions = list(self.rounds_script[self.call_idx])
        else:
            gap_questions = []
        self.call_idx += 1
        cats = ["budget", "time"][: len(gap_questions)]
        return {
            "restate_text": f"我聽到的是「{user_input}」。" + (
                "、".join(gap_questions) + "？" if gap_questions else ""
            ),
            "gap_questions": gap_questions,
            "gap_categories": cats,
            "inferred_goal": user_input,
            "inferred_motivation": "",
            "inferred_constraints": [],
        }

    def merge_confirmation(self, original_input, restate_text, user_confirmation_text):
        self.merge_confirmation_calls.append({
            "user": user_confirmation_text,
        })
        return {
            "goal": original_input,
            "motivation": "",
            "constraints": [user_confirmation_text] if user_confirmation_text else [],
            "answered_categories": [],
        }

    def execute_task(self, goal, motivation, constraints, user_profile=None, **kwargs):
        self.execute_task_calls.append({"goal": goal})
        return f"針對 {goal} 的建議。"

    def analyze_deviation(self, goal, result, deviation):  # pragma: no cover
        return "（分析）"


class ConvergenceGateIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "endpoints.json")
        self.store = EndpointStore(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, client, user_inputs):
        from src.main import run_one_interaction
        it = iter(user_inputs)
        with patch.object(builtins, "input", lambda prompt="": next(it)):
            return run_one_interaction(
                "幫我規劃下週去日本的行程",
                self.store,
                client,
                ask_satisfaction=False,
            )

    def _start_data(self):
        starts = [r for r in self.store.all() if r["type"] == "start"]
        self.assertEqual(len(starts), 1, "should write exactly one start endpoint")
        return starts[0]["start_data"]

    def test_three_empty_rounds_triggers_empty_streak(self):
        """規格 §測試計畫 row 1：使用者連續三輪回「沒有」「沒」「好」→ empty_streak。"""
        client = _ScriptedClient(rounds_script=[
            ["性質？"],
            ["動機？"],
            ["時間？"],
            ["預算？"],  # 不會走到
            ["人員？"],  # 不會走到
        ])
        ok = self._run(client, ["沒有", "沒", "好"])
        self.assertTrue(ok)
        sd = self._start_data()
        self.assertEqual(sd["convergence_reason"], "empty_streak")
        self.assertEqual(sd["total_gap_rounds"], 3)
        # 連續 3 輪空輪都沒答出東西 → 每個問過的 category 都應該變成 unresolved
        self.assertGreaterEqual(len(sd["unresolved_gaps"]), 1)

    def test_five_rounds_of_real_answers_triggers_max_rounds(self):
        """規格 §測試計畫 row 2：五輪都有新資訊 → max_rounds。"""
        client = _ScriptedClient(rounds_script=[
            ["性質？"],
            ["動機？"],
            ["時間？"],
            ["預算？"],
            ["人員？"],
            ["價值？"],  # 不會被使用（5 輪已達上限）
        ])
        ok = self._run(client, [
            "想體驗日本祭典氛圍帶孩子增廣見聞",
            "這次是為了慶祝結婚十週年的紀念旅程",
            "預計十月中旬連續七天旅行",
            "預算大約新台幣二十萬元左右",
            "主要夥伴是配偶和兩個就學中的孩子",
        ])
        self.assertTrue(ok)
        sd = self._start_data()
        self.assertEqual(sd["convergence_reason"], "max_rounds")
        self.assertEqual(sd["total_gap_rounds"], confirm_mod.CONVERGENCE_MAX_ROUNDS)

    def test_effective_round_resets_streak_then_max_rounds_wins(self):
        """規格 §測試計畫 row 3：空/空/有效/空/空 → 以 max_rounds 觸發。"""
        client = _ScriptedClient(rounds_script=[
            ["性質？"],
            ["動機？"],
            ["時間？"],
            ["預算？"],
            ["人員？"],
        ])
        ok = self._run(client, [
            "沒",                                     # 輪 1：空
            "不知道",                                 # 輪 2：空（4 字 ≤ 5）
            "預計在十月中旬連續旅行七天左右",           # 輪 3：有 → 歸零
            "沒",                                     # 輪 4：空
            "不想講",                                 # 輪 5：空
        ])
        self.assertTrue(ok)
        sd = self._start_data()
        self.assertEqual(sd["convergence_reason"], "max_rounds")
        self.assertEqual(sd["total_gap_rounds"], confirm_mod.CONVERGENCE_MAX_ROUNDS)

    def test_first_round_sufficient_skips_loop(self):
        """規格 §測試計畫 row 4：第一輪 LLM 回空 gap_questions → info_sufficient。"""
        client = _ScriptedClient(rounds_script=[[]])  # 第 1 輪直接空
        ok = self._run(client, [])  # 不應該需要任何輸入
        self.assertTrue(ok)
        sd = self._start_data()
        self.assertEqual(sd["convergence_reason"], "info_sufficient")
        self.assertEqual(sd["total_gap_rounds"], 1)
        self.assertEqual(sd["unresolved_gaps"], [])
        # Copilot #5：首輪即收斂時 user_confirmation 改為 auto_info_sufficient
        self.assertEqual(sd["user_confirmation"], "auto_info_sufficient")
        # 不會呼叫 merge_confirmation（沒有使用者回答可合併）
        self.assertEqual(len(client.merge_confirmation_calls), 0)

    def test_info_sufficient_on_round_two_records_correct_total(self):
        """Copilot review #4：第 2 輪才收斂 → total_gap_rounds 應為 2（非 1）。"""
        client = _ScriptedClient(rounds_script=[
            ["性質？"],  # 輪 1 有問題
            [],           # 輪 2 資訊已足
        ])
        ok = self._run(client, ["我想帶家人一起去日本賞楓"])
        self.assertTrue(ok)
        sd = self._start_data()
        self.assertEqual(sd["convergence_reason"], "info_sufficient")
        self.assertEqual(sd["total_gap_rounds"], 2)
        # 呼叫 restate 兩次
        self.assertEqual(len(client.generate_restate_calls), 2)

    def test_auto_confirm_short_circuits_multi_round(self):
        """Copilot review #6：auto_confirm=True 只跑單輪 restate，避免 max_rounds 浪費。"""
        from src.main import run_one_interaction
        # Fake client 每輪都會給 gap_questions（模擬 LLM 一直想追問）
        client = _ScriptedClient(rounds_script=[
            ["性質？"], ["動機？"], ["時間？"], ["預算？"], ["人員？"],
        ])
        ok = run_one_interaction(
            "幫我規劃下週去日本的行程",
            self.store,
            client,
            ask_satisfaction=False,
            auto_confirm=True,  # 關鍵：batch / CI 情境
        )
        self.assertTrue(ok)
        sd = self._start_data()
        # 只應該有 1 次 restate call（而非 5 次跑到 max_rounds）
        self.assertEqual(len(client.generate_restate_calls), 1)
        self.assertEqual(sd["total_gap_rounds"], 1)
        self.assertEqual(sd["convergence_reason"], "info_sufficient")


# ----------------------------------------------------------------------
# 效能 / 快取行為測試
# ----------------------------------------------------------------------
class ExtractEndpointTokensTests(unittest.TestCase):
    """Copilot review #1：預先抽出端點 token 集合以避免多輪重複斷詞。"""

    def test_empty_or_none_returns_empty_set(self):
        self.assertEqual(confirm_mod.extract_endpoint_tokens(None), set())
        self.assertEqual(confirm_mod.extract_endpoint_tokens([]), set())

    def test_collects_tokens_from_all_five_start_data_fields(self):
        """五個欄位（goal / user_input / motivation / tofu_understanding /
        constraints）都要被掃到。使用能被 2-gram fallback 與 jieba 都命中
        的簡單詞（避免測試因 tokenizer 差異產生 flakiness）。"""
        endpoints = [{
            "type": "start",
            "start_data": {
                "goal": "辦生日",
                "user_input": "派對",
                "motivation": "慶祝",
                "tofu_understanding": "台北",
                "constraints": ["三萬"],
            },
        }]
        tokens = confirm_mod.extract_endpoint_tokens(endpoints)
        # 所有五個欄位各放一個明顯的 2-char token，逐一驗證都被抽到
        self.assertIn("生日", tokens)  # goal
        self.assertIn("派對", tokens)  # user_input
        self.assertIn("慶祝", tokens)  # motivation
        self.assertIn("台北", tokens)  # tofu_understanding
        self.assertIn("三萬", tokens)  # constraints

    def test_has_effective_info_accepts_precomputed_tokens(self):
        """預先算好的 token set 與現場抽出的結果等價。"""
        endpoints = [{"start_data": {
            "goal": "辦生日派對", "user_input": "派對",
            "motivation": "", "tofu_understanding": "",
            "constraints": [],
        }}]
        cached = confirm_mod.extract_endpoint_tokens(endpoints)
        # 兩種 call 方式結果應一致
        self.assertEqual(
            confirm_mod.has_effective_info("想準備麻婆豆腐當主餐", endpoints),
            confirm_mod.has_effective_info("想準備麻婆豆腐當主餐", existing_tokens=cached),
        )
        # 重複已知名詞的回應：兩種 call 方式都判無效
        self.assertFalse(
            confirm_mod.has_effective_info("派對 派對 派對", existing_tokens=cached)
        )


# ----------------------------------------------------------------------
# 欄位輸出測試
# ----------------------------------------------------------------------
class MakeStartDataConvergenceFieldsTests(unittest.TestCase):
    def test_default_values_when_kwargs_absent(self):
        sd = make_start_data(
            user_input="x",
            tofu_understanding="x",
            gap_questions=[],
            user_confirmation="confirmed",
            user_modifications="",
            goal="x",
            motivation="",
            constraints=[],
        )
        self.assertIn("unresolved_gaps", sd)
        self.assertIn("convergence_reason", sd)
        self.assertIn("total_gap_rounds", sd)
        self.assertEqual(sd["unresolved_gaps"], [])
        self.assertIsNone(sd["convergence_reason"])
        self.assertEqual(sd["total_gap_rounds"], 0)

    def test_explicit_kwargs_are_persisted(self):
        sd = make_start_data(
            user_input="x",
            tofu_understanding="x",
            gap_questions=[],
            user_confirmation="confirmed",
            user_modifications="",
            goal="x",
            motivation="",
            constraints=[],
            unresolved_gaps=["budget", "headcount"],
            convergence_reason="empty_streak",
            total_gap_rounds=3,
        )
        self.assertEqual(sd["unresolved_gaps"], ["budget", "headcount"])
        self.assertEqual(sd["convergence_reason"], "empty_streak")
        self.assertEqual(sd["total_gap_rounds"], 3)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
